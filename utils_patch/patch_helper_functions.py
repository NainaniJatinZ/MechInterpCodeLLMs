import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import linregress
import numpy as np
import einops
from fancy_einsum import einsum
import tqdm.notebook as tqdm
import random
import os
import time
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader
from torchtyping import TensorType as TT
from typing import List, Union, Optional, Callable
from typing_extensions import Literal
from functools import partial
import copy
import itertools
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import sys
import gc

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets

# THIS IS A LOCAL (MODIFIED) VERSION OF TRANSFORMER_LENS - UNINSTALL PIP/CONDA VERSION BEFORE USE!
import transformer_lens
import transformer_lens.utils as utils
import transformer_lens.patching as patching
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)

import os
from config import HF_TOKEN, HF_PATH
# Set environment variables
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["TRANSFORMERS_CACHE"] = HF_PATH
os.environ["HF_DATASETS_CACHE"] = HF_PATH
os.environ["HF_HOME"] = HF_PATH

# When using multiple GPUs we use GPU 0 as the primary and switch to the next when it is 90% full
num_gpus = torch.cuda.device_count()
device_id = 0
if num_gpus > 0:
    device = "cuda:0"
else:
    device = "cpu"
# Check GPU memory usage
def check_gpu_memory(max_alloc=0.9):
    if not torch.cuda.is_available():
        return
    global device_id, device
    print("Primary device:", device)
    torch.cuda.empty_cache()
    max_alloc = 1 if max_alloc > 1 else max_alloc
    for gpu in range(num_gpus):
        memory_reserved = torch.cuda.memory_reserved(device=gpu)
        memory_allocated = torch.cuda.memory_allocated(device=gpu)
        total_memory = torch.cuda.get_device_properties(gpu).total_memory 
        print(f"GPU {gpu}: {total_memory / (1024**2):.2f} MB  Allocated: {memory_allocated / (1024**2):.2f} MB  Reserved: {memory_reserved / (1024**2):.2f} MB")
                
        # Check if the current GPU is getting too full, and if so we switch the primary device to the next GPU
        if memory_reserved > max_alloc * total_memory:
            if device_id < num_gpus - 1:
                device_id += 1
                device = f"cuda:{device_id}"
                print(f"Switching primary device to {device}")
            else:
                print("Cannot switch primary device, all GPUs are nearly full")


def timeit(func):
    """Decorator to measure the execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__!r} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper


# Logit difference metric
def get_logit_diff(logits, answer_token_indices=answer_token_indices, device="cpu"):
    if len(logits.shape) == 3:
        # Get final logits only
        logits = logits[:, -1, :]
    logits = logits.to(answer_token_indices.device)
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits).mean()

@timeit
def get_cache_fwd_and_bwd(model, tokens, metric, answer_indices):
    model.reset_hooks()
    cache = {}

    def forward_cache_hook(act, hook):
        act = act.to(device)
        torch.cuda.empty_cache()
        cache[hook.name] = act

    model.add_hook(filter_not_qkv_input, forward_cache_hook, "fwd")
    grad_cache = {}

    def backward_cache_hook(act, hook):
        act = act.to(device)
        torch.cuda.empty_cache()
        grad_cache[hook.name] = act

    model.add_hook(filter_not_qkv_input, backward_cache_hook, "bwd")
    
    result = model(tokens).to(device)
    torch.cuda.empty_cache()
    value = metric(result, answer_indices)
    value.backward()

    # Reset hooks and clear unused GPU memory
    value = value.item()
    model.reset_hooks()
    torch.cuda.empty_cache()
    
    cache = ActivationCache(cache, model).to(device)
    grad_cache = ActivationCache(grad_cache, model).to(device)
    
    return value, cache, grad_cache

@timeit
def create_attention_attr(
    clean_cache, clean_grad_cache, device
) -> TT["batch", "layer", "head_index", "dest", "src"]:
    attention_stack = torch.stack(
        [clean_cache["pattern", l] for l in range(model.cfg.n_layers)], dim=0
    ).to(device)
    attention_grad_stack = torch.stack(
        [clean_grad_cache["pattern", l] for l in range(model.cfg.n_layers)], dim=0
    ).to(device)
    attention_attr = attention_grad_stack * attention_stack
    attention_attr = einops.rearrange(
        attention_attr,
        "layer batch head_index dest src -> batch layer head_index dest src",
    )
    return attention_attr

@timeit
def attr_patch_residual(
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    corrupted_grad_cache: ActivationCache,
    device,
) -> TT["component", "pos"]:
    clean_residual, residual_labels = clean_cache.accumulated_resid(
        -1, incl_mid=True, return_labels=True
    )
    corrupted_residual = corrupted_cache.accumulated_resid(
        -1, incl_mid=True, return_labels=False
    )
    corrupted_grad_residual = corrupted_grad_cache.accumulated_resid(
        -1, incl_mid=True, return_labels=False
    )
    residual_attr = einops.reduce(
        corrupted_grad_residual * (clean_residual - corrupted_residual),
        "component batch pos d_model -> component pos",
        "sum",
    ).to(device)
    return residual_attr, residual_labels

@timeit
def attr_patch_layer_out(
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    corrupted_grad_cache: ActivationCache,
    device
) -> TT["component", "pos"]:
    clean_layer_out, labels = clean_cache.decompose_resid(-1, return_labels=True)
    corrupted_layer_out = corrupted_cache.decompose_resid(-1, return_labels=False)
    corrupted_grad_layer_out = corrupted_grad_cache.decompose_resid(
        -1, return_labels=False
    )
    layer_out_attr = einops.reduce(
        corrupted_grad_layer_out * (clean_layer_out - corrupted_layer_out),
        "component batch pos d_model -> component pos",
        "sum",
    ).to(device)
    return layer_out_attr, labels

@timeit
def attr_patch_head_out(
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    corrupted_grad_cache: ActivationCache,
    device
) -> TT["component", "pos"]:
    labels = HEAD_NAMES

    clean_head_out = clean_cache.stack_head_results(-1, return_labels=False).to(device)
    corrupted_head_out = corrupted_cache.stack_head_results(-1, return_labels=False).to(device)
    corrupted_grad_head_out = corrupted_grad_cache.stack_head_results(
        -1, return_labels=False
    ).to(device)
    head_out_attr = einops.reduce(
        corrupted_grad_head_out * (clean_head_out - corrupted_head_out),
        "component batch pos d_model -> component pos",
        "sum",
    ).to(device)
    return head_out_attr, labels

def stack_head_vector_from_cache(
    cache, activation_name: Literal["q", "k", "v", "z"], device
) -> TT["layer_and_head_index", "batch", "pos", "d_head"]:
    """Stacks the head vectors from the cache from a specific activation (key, query, value or mixed_value (z)) into a single tensor."""
    stacked_head_vectors = torch.stack(
        [cache[activation_name, l] for l in range(model.cfg.n_layers)], dim=0
    ).to(device)
    stacked_head_vectors = einops.rearrange(
        stacked_head_vectors,
        "layer batch pos head_index d_head -> (layer head_index) batch pos d_head",
    ).to(device)
    return stacked_head_vectors

@timeit
def attr_patch_head_vector(
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    corrupted_grad_cache: ActivationCache,
    activation_name: Literal["q", "k", "v", "z"],
    device
) -> TT["component", "pos"]:
    labels = HEAD_NAMES

    clean_head_vector = stack_head_vector_from_cache(clean_cache, activation_name, "cpu").to(device)
    corrupted_head_vector = stack_head_vector_from_cache(
        corrupted_cache, activation_name, "cpu"
    ).to(device)
    corrupted_grad_head_vector = stack_head_vector_from_cache(
        corrupted_grad_cache, activation_name, "cpu"
    ).to(device)
    head_vector_attr = einops.reduce(
        corrupted_grad_head_vector * (clean_head_vector - corrupted_head_vector),
        "component batch pos d_head -> component pos",
        "sum",
    )
    return head_vector_attr, labels

def stack_head_pattern_from_cache(
    cache,
    device
) -> TT["layer_and_head_index", "batch", "dest_pos", "src_pos"]:
    """Stacks the head patterns from the cache into a single tensor."""
    stacked_head_pattern = torch.stack(
        [cache["pattern", l] for l in range(model.cfg.n_layers)], dim=0
    ).to(device)
    stacked_head_pattern = einops.rearrange(
        stacked_head_pattern,
        "layer batch head_index dest_pos src_pos -> (layer head_index) batch dest_pos src_pos",
    ).to(device)
    return stacked_head_pattern

@timeit
def attr_patch_head_pattern(
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    corrupted_grad_cache: ActivationCache,
    device
) -> TT["component", "dest_pos", "src_pos"]:
    labels = HEAD_NAMES

    clean_head_pattern = stack_head_pattern_from_cache(clean_cache, "cpu").to(device)
    corrupted_head_pattern = stack_head_pattern_from_cache(corrupted_cache, "cpu").to(device)
    corrupted_grad_head_pattern = stack_head_pattern_from_cache(corrupted_grad_cache, "cpu").to(device)
    head_pattern_attr = einops.reduce(
        corrupted_grad_head_pattern * (clean_head_pattern - corrupted_head_pattern),
        "component batch dest_pos src_pos -> component dest_pos src_pos",
        "sum",
    ).to(device)
    return head_pattern_attr, labels

def get_head_vector_grad_input_from_grad_cache(
    grad_cache: ActivationCache, activation_name: Literal["q", "k", "v"], layer: int, device
) -> TT["batch", "pos", "head_index", "d_model"]:
    vector_grad = grad_cache[activation_name, layer].to(device)
    ln_scales = grad_cache["scale", layer, "ln1"].to(device)
    attn_layer_object = model.blocks[layer].attn
    if activation_name == "q":
        W = attn_layer_object.W_Q.to(device)
    elif activation_name == "k":
        W = attn_layer_object.W_K.to(device)
    elif activation_name == "v":
        W = attn_layer_object.W_V.to(device)
    else:
        raise ValueError("Invalid activation name")

    # Original notebook used (batch pos) for second input but that seems to be wrong - double check this computation
    return einsum(
        "batch pos head_index d_head, batch pos head_index, head_index d_model d_head -> batch pos head_index d_model",
        vector_grad,
        ln_scales.squeeze(-1),
        W,
    )

def get_stacked_head_vector_grad_input(
    grad_cache, activation_name: Literal["q", "k", "v"], device
) -> TT["layer", "batch", "pos", "head_index", "d_model"]:
    return torch.stack(
        [
            get_head_vector_grad_input_from_grad_cache(grad_cache, activation_name, l, "cpu")
            for l in range(model.cfg.n_layers)
        ],
        dim=0,
    ).to(device)

def get_full_vector_grad_input(
    grad_cache, device
) -> TT["qkv", "layer", "batch", "pos", "head_index", "d_model"]:
    return torch.stack([get_stacked_head_vector_grad_input(grad_cache, activation_name, "cpu").to(device) for activation_name in ["q", "k", "v"]], dim=0).to(device)

@timeit
def attr_patch_head_path(
    clean_cache: ActivationCache,
    corrupted_cache: ActivationCache,
    corrupted_grad_cache: ActivationCache,
    device
) -> TT["qkv", "dest_component", "src_component", "pos"]:
    """
    Computes the attribution patch along the path between each pair of heads.

    Sets this to zero for the path from any late head to any early head

    """
    start_labels = HEAD_NAMES
    end_labels = HEAD_NAMES_QKV
    full_vector_grad_input = get_full_vector_grad_input(corrupted_grad_cache, "cpu")
    clean_head_result_stack = clean_cache.stack_head_results(-1)
    corrupted_head_result_stack = corrupted_cache.stack_head_results(-1)
    diff_head_result = einops.rearrange(
        clean_head_result_stack - corrupted_head_result_stack,
        "(layer head_index) batch pos d_model -> layer batch pos head_index d_model",
        layer=model.cfg.n_layers,
        head_index=model.cfg.n_heads,
    )
    path_attr = einsum(
        "qkv layer_end batch pos head_end d_model, layer_start batch pos head_start d_model -> qkv layer_end head_end layer_start head_start pos",
        full_vector_grad_input,
        diff_head_result,
    )
    correct_layer_order_mask = (
        torch.arange(model.cfg.n_layers)[None, :, None, None, None, None]
        > torch.arange(model.cfg.n_layers)[None, None, None, :, None, None]
    ).to(path_attr.device)
    zero = torch.zeros(1, device=path_attr.device)
    path_attr = torch.where(correct_layer_order_mask, path_attr, zero)

    path_attr = einops.rearrange(
        path_attr,
        "qkv layer_end head_end layer_start head_start pos -> (layer_end head_end qkv) (layer_start head_start) pos",
    )
    return path_attr, end_labels, start_labels

def get_variable_sizes():
    """Get sizes of all variables in the global scope."""
    sizes = {name: sys.getsizeof(obj) for name, obj in globals().items()}
    return sizes

def print_largest_variables(n=5):
    """Print the largest n variables by size."""
    sizes = get_variable_sizes()
    largest = sorted(sizes.items(), key=lambda x: x[1], reverse=True)[:n]
    for name, size in largest:
        print(f'{name}: {size / 1024:.2f} KB')

def delete_largest_variables(n=5):
    """Delete the largest n variables by size."""
    sizes = get_variable_sizes()
    largest = sorted(sizes.items(), key=lambda x: x[1], reverse=True)[:n]
    for name, _ in largest:
        if name not in ['psutil', 'sys', 'gc', 'get_variable_sizes', 'print_largest_variables', 'delete_largest_variables']:
            print(f'Deleting {name}...')
            del globals()[name]
    # Perform garbage collection to free up memory
    gc.collect()




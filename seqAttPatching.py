import sys
sys.path.append('../transformer_lens')

import transformer_lens
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets

import argparse

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

import utils_patch.patch_helper_functions as helper

import os
from config import HF_TOKEN, HF_PATH

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["TRANSFORMERS_CACHE"] = HF_PATH
os.environ["HF_DATASETS_CACHE"] = HF_PATH
os.environ["HF_HOME"] = HF_PATH

num_gpus = torch.cuda.device_count()
device_id = 0
if num_gpus > 0:
    device = "cuda:0"
else:
    device = "cpu"

    
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

print("Number of GPUs:", num_gpus)
check_gpu_memory()

def timeit(func):
    """Decorator to measure the execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__!r} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

# Create transformer
model = HookedTransformer.from_pretrained("CodeLlama-7b-hf", n_devices=num_gpus)

# We need these so that individual attention heads and MLP inputs can be edited
model.set_use_attn_in(True)
model.set_use_hook_mlp_in(True)
model.set_use_attn_result(True) # Documentation says this easily burns through GPU memory

check_gpu_memory()


# Loading input

parser = argparse.ArgumentParser(description="Process model inputs.")
parser.add_argument("--logit_difference", type=float, required=True, help="The logit difference value to use in the script.")
parser.add_argument("--NumberOfPrompts", type=int, required=True, help="The number of prompts to process.")

# Parse arguments
args = parser.parse_args()

# Extract values from arguments
logit_difference_from_prev = args.logit_difference
NumberOfPrompts = args.NumberOfPrompts

## Info Retrieval Data Loading

with open('data/prompt_dataset_zero_shot.json', 'r') as f:
    data = json.load(f)

# Take the first few dictionaries (e.g., first 3)
subset = data[:NumberOfPrompts]

# Initialize lists
prompts = []
answers = []

# Extract prompts and outputs
outputs = []
for ind, item in enumerate(subset):
    # if ind in [0, 1, 3, 4, 5, 6, 7, 9]:
    prompts.append(item["prompt"])
    outputs.append(item["output"])

# Group pairs of outputs and reverse the tuples alternately
for i in range(0, len(outputs) - 1, 2):
    answers.append((outputs[i], outputs[i + 1]))
    answers.append((outputs[i + 1], outputs[i]))

# Display the results
print("Prompts:", prompts)
print("Answers:", answers)

clean_tokens = model.to_tokens(prompts)
# Swap each adjacent pair, with a hacky list comprehension
corrupted_tokens = clean_tokens[
    [(i+1 if i%2==0 else i-1) for i in range(len(clean_tokens)) ]
    ]
print("Clean string 0", model.to_string(clean_tokens[0]))
print("Corrupted string 0", model.to_string(corrupted_tokens[0]))
check_gpu_memory()

answer_token_indices = torch.tensor([[model.to_single_token(answers[i][j]) for j in range(2)] for i in range(len(answers))], device=device)
print("Answer token indices", answer_token_indices)
print("Number of tokens in clean: ", len(clean_tokens[0]))
print("Number of tokens in corrupted: ", len(corrupted_tokens[0]))
print("answer shape: ", answer_token_indices.shape)
print(f"Clean token shape: {len(clean_tokens)} * {len(clean_tokens[0])}")
print(f"Corrupted token shape: {len(corrupted_tokens)} * {len(corrupted_tokens[0])}")

# the above sizes are in the format of N * P, N -> number of prompts, P -> number of tokens in each prompt
# Our goal is to do all memory intensive stuff one prompt at a time

# Empty CUDA cache
torch.cuda.empty_cache()

# Optionally check memory to confirm
check_gpu_memory()

CLEAN_BASELINE = logit_difference_from_prev #clean_logit_diff
CORRUPTED_BASELINE = -logit_difference_from_prev  #corrupted_logit_diff

def ioi_metric(logits, answer_token_indices=answer_token_indices):
    logits = logits.to(device)
    torch.cuda.empty_cache()
    return (helper.get_logit_diff(logits, answer_token_indices) - CORRUPTED_BASELINE) / (
        CLEAN_BASELINE - CORRUPTED_BASELINE
    )

# first prompt processing 
Metric = Callable[[TT["batch_and_pos_dims", "d_model"]], float]
filter_not_qkv_input = lambda name: "_input" not in name

answer_token_indices_first = answer_token_indices[0:1]
clean_value, clean_cache_first, clean_grad_cache_first = helper.get_cache_fwd_and_bwd(model, clean_tokens[0:1], ioi_metric, answer_token_indices_first, device)
clean_cache_first = clean_cache_first.to('cpu')
clean_grad_cache_first = clean_grad_cache_first.to('cpu')
check_gpu_memory()

corrupted_value, corrupted_cache_first, corrupted_grad_cache_first = helper.get_cache_fwd_and_bwd(model, corrupted_tokens[0:1], ioi_metric, answer_token_indices_first, device)
corrupted_cache_first = corrupted_cache_first.to('cpu')
corrupted_grad_cache_first = corrupted_grad_cache_first.to('cpu')
check_gpu_memory()

for i in range(1, len(clean_tokens)):
    
    single_clean_tokens = clean_tokens[i:i+1]
    single_corrupted_tokens = corrupted_tokens[i:i+1]
    single_answer_token_indices = answer_token_indices[i:i+1]
    
    
    clean_value, clean_cache, clean_grad_cache = helper.get_cache_fwd_and_bwd(model, single_clean_tokens, ioi_metric, single_answer_token_indices, device)
    clean_cache = clean_cache.to('cpu')
    clean_grad_cache = clean_grad_cache.to('cpu')
    clean_cache_first = clean_cache_first.concatenate(clean_cache)
    clean_grad_cache_first = clean_grad_cache_first.concatenate(clean_grad_cache)

    check_gpu_memory()
    
    
    corrupted_value, corrupted_cache, corrupted_grad_cache = helper.get_cache_fwd_and_bwd(model, single_corrupted_tokens, ioi_metric, single_answer_token_indices, device)
    corrupted_cache = corrupted_cache.to('cpu')
    corrupted_grad_cache = corrupted_grad_cache.to('cpu')
    corrupted_cache_first = corrupted_cache_first.concatenate(corrupted_cache)
    corrupted_grad_cache_first = corrupted_grad_cache_first.concatenate(corrupted_grad_cache)

    check_gpu_memory()

torch.cuda.empty_cache()

# Test if shapes worked
print("cache shape: ,", corrupted_cache_first["hook_embed"].shape)
check_gpu_memory()

residual_attr, residual_labels = helper.attr_patch_residual(
    clean_cache, corrupted_cache, corrupted_grad_cache, "cpu"
)
check_gpu_memory()

# Plotting the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(residual_attr.detach().numpy(), yticklabels=residual_labels, annot=False, cmap='viridis', center=0)

plt.xlabel('Position')
plt.ylabel('Component')
plt.title('Residual Attribution Patching')
plt.savefig('plots/residual_attribution_patching.png')


layer_out_attr, layer_out_labels = helper.attr_patch_layer_out(
    clean_cache, corrupted_cache, corrupted_grad_cache, "cpu"
)
check_gpu_memory()

plt.figure(figsize=(10, 6))
sns.heatmap(layer_out_attr.detach().numpy(), yticklabels=layer_out_labels, annot=False, cmap='viridis', center=0)
plt.xlabel('Position')
plt.ylabel('Component')
plt.title('Layer Out Attribution Patching')
plt.savefig('plots/layer_attribution_patching.png')

HEAD_NAMES = [
    f"L{l}H{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)
]
HEAD_NAMES_SIGNED = [f"{name}{sign}" for name in HEAD_NAMES for sign in ["+", "-"]]
HEAD_NAMES_QKV = [
    f"{name}{act_name}" for name in HEAD_NAMES for act_name in ["Q", "K", "V"]
]
print(HEAD_NAMES[:5])
print(HEAD_NAMES_SIGNED[:5])
print(HEAD_NAMES_QKV[:5])

head_out_attr, head_out_labels = helper.attr_patch_head_out(
    clean_cache, corrupted_cache, corrupted_grad_cache, "cpu", HEAD_NAMES
)

sum_head_out_attr = einops.reduce(
    head_out_attr,
    "(layer head) pos -> layer head",
    "sum",
    layer=model.cfg.n_layers,
    head=model.cfg.n_heads,
)
check_gpu_memory()

plt.figure(figsize=(12, 8))
sns.heatmap(head_out_attr.detach().numpy(), yticklabels=head_out_labels, annot=False, cmap='viridis', center=0)
plt.yticks(fontsize = 4)
plt.xlabel('Position')
plt.ylabel('Component')
plt.title('Head Output Attribution Patching')
plt.savefig('plots/head_op_attribution_patching.png')


plt.figure(figsize=(10, 6))
sns.heatmap(sum_head_out_attr.detach().numpy(), annot=False, cmap='viridis', center=0)
plt.xlabel('Position')
plt.ylabel('Component')
plt.title('Head Output Attribution Patching Sum Over Position')
plt.savefig('plots/head_sum_attribution_patching.png')


head_pattern_attr, labels = helper.attr_patch_head_pattern(
    clean_cache, corrupted_cache, corrupted_grad_cache, "cpu", HEAD_NAMES, model.cfg.n_layers
)

head_pattern_attr = einops.rearrange(
        head_pattern_attr,
        "(layer head) dest src -> layer head dest src",
        layer=model.cfg.n_layers,
        head=model.cfg.n_heads,
    )
check_gpu_memory()


head_path_attr, end_labels, start_labels = helper.attr_patch_head_path(
    clean_cache, corrupted_cache, corrupted_grad_cache, "cpu", HEAD_NAMES, HEAD_NAMES_QKV, model.cfg.n_heads, model.cfg.n_layers, model
)


plt.figure(figsize=(32, 12))
sns.set_style("white")
sns.heatmap(head_path_attr.sum(-1).detach().numpy(), yticklabels=end_labels, xticklabels = start_labels, annot=False, cmap='viridis', center=0)
plt.yticks(fontsize = 5)
plt.xlabel('Path Start (Head Output)')
plt.ylabel('Path End (Head Input)')
plt.title('Head Path Attribution Patching')
plt.savefig('plots/head_path_attribution_patching.png')


head_out_values, head_out_indices = head_out_attr.sum(-1).abs().sort(descending=True)
top_head_indices = head_out_indices[:22].sort().values
top_end_indices = []
top_end_labels = []
top_start_indices = []
top_start_labels = []

plt.plot(head_out_values.detach().numpy())

for i in top_head_indices:
    i = i.item()
    top_start_indices.append(i)
    top_start_labels.append(start_labels[i])
    for j in range(3):
        top_end_indices.append(3 * i + j)
        top_end_labels.append(end_labels[3 * i + j])

top_head_path_attr = einops.rearrange(
    head_path_attr[top_end_indices, :][:, top_start_indices].sum(-1),
    "(head_end qkv) head_start -> qkv head_end head_start",
    qkv=3,
)

check_gpu_memory()

plt.figure(figsize=(28, 16))
sns.heatmap(head_path_attr[top_end_indices, :][:, top_start_indices].sum(-1).detach().numpy(), yticklabels = top_end_labels, xticklabels= top_start_labels, annot=False, cmap='viridis', center=0)
plt.xlabel('Path Start (Head Output)')
plt.ylabel('Path End (Head Input)')
plt.title('Head Path Attribution Patching')
plt.savefig('plots/head_path_2_attribution_patching.png')

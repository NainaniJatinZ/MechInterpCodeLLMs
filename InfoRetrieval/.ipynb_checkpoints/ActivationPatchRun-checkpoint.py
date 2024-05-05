import transformer_lens
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
os.environ["HF_TOKEN"] = "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"
hf_path = "/work/pi_jensen_umass_edu/svaidyanatha_umass_edu/huggingface"
os.environ["TRANSFORMERS_CACHE"] = hf_path
os.environ["HF_DATASETS_CACHE"] = hf_path
os.environ["HF_HOME"] = hf_path

# When using multiple GPUs we use GPU 0 as the primary and switch to the next when it is 90% full
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


# Some examples to work with - replace with code examples

prompts = ['When J and M went to the shops, J gave the bag to ', 'When J and M went to the shops, M gave the bag to ', 'When T and J went to the park, J gave the ball to ', 'When T and J went to the park, T gave the ball to ', 'When D and S went to the shops, S gave an apple to ', 'When D and S went to the shops, D gave an apple to ', 'After R and A went to the park, A gave a drink to ', 'After R and A went to the park, R gave a drink to ']
answers = [('M', 'J'), ('J', 'M'), ('T', 'J'), ('J', 'T'), ('D', 'S'), ('S', 'D'), ('R', 'A'), ('A', 'R')]

clean_tokens = model.to_tokens(prompts)
# Swap each adjacent pair, with a hacky list comprehension
corrupted_tokens = clean_tokens[
    [(i+1 if i%2==0 else i-1) for i in range(len(clean_tokens)) ]
    ]
print("Clean string 0", model.to_string(clean_tokens[0]))
print("Corrupted string 0", model.to_string(corrupted_tokens[0]))

answer_token_indices = torch.tensor([[model.to_single_token(answers[i][j]) for j in range(2)] for i in range(len(answers))], device=device)
print("Answer token indices", answer_token_indices)
check_gpu_memory()




# Logit difference metric
def get_logit_diff(logits, answer_token_indices=answer_token_indices, device="cpu"):
    if len(logits.shape) == 3:
        # Get final logits only
        logits = logits[:, -1, :]
    logits = logits.to(answer_token_indices.device)
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits).mean()

clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

clean_logit_diff = get_logit_diff(clean_logits, answer_token_indices).item()
print(f"Clean logit diff: {clean_logit_diff:.4f}")

corrupted_logit_diff = get_logit_diff(corrupted_logits, answer_token_indices).item()
print(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")
check_gpu_memory()



# IOI metric - does this mean indirect object identification? Yes

CLEAN_BASELINE = clean_logit_diff
CORRUPTED_BASELINE = corrupted_logit_diff

def ioi_metric(logits, answer_token_indices=answer_token_indices):
    logits = logits.to(device)
    torch.cuda.empty_cache()
    return (get_logit_diff(logits, answer_token_indices) - CORRUPTED_BASELINE) / (
        CLEAN_BASELINE - CORRUPTED_BASELINE
    )

print(f"Clean Baseline is 1: {ioi_metric(clean_logits).item():.4f}")
print(f"Corrupted Baseline is 0: {ioi_metric(corrupted_logits).item():.4f}")
check_gpu_memory()

Metric = Callable[[TT["batch_and_pos_dims", "d_model"]], float]

# Delete tensors
del clean_logits
del corrupted_logits
del clean_logit_diff 
del corrupted_logit_diff
clean_cache = clean_cache.to("cpu")
del corrupted_cache

# Empty CUDA cache
torch.cuda.empty_cache()

# Optionally check memory to confirm
check_gpu_memory()

### Validating experiment (ACTIVATION STUFF)

## Per block

str_tokens = model.to_str_tokens(clean_tokens[0])
context_length = len(str_tokens)
every_block_act_patch_result = patching.get_act_patch_block_every(
    model, corrupted_tokens, clean_cache, ioi_metric
).to("cpu")
check_gpu_memory()


## Per head

every_head_all_pos_act_patch_result = patching.get_act_patch_attn_head_all_pos_every(
    model, corrupted_tokens, clean_cache, ioi_metric
)

every_head_by_pos_act_patch_result = patching.get_act_patch_attn_head_by_pos_every(
    model, corrupted_tokens, clean_cache, ioi_metric
)
every_head_by_pos_act_patch_result = einops.rearrange(
    every_head_by_pos_act_patch_result,
    "act_type layer pos head -> act_type (layer head) pos",
)
# Description: This script is used to load the CodeLlama-7b model and patch it with the sequence attention patching method.
import sys
sys.path.append('../transformer_lens')
# Import necessary libraries
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
import gc
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
from utils_patch import *
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



def load_model(model_name: str, n_devices: int = 1) -> HookedTransformer:
    """Load a model from the Hugging Face model hub and wrap it in a HookedTransformer."""
    model = HookedTransformer.from_pretrained(model_name, n_devices=n_devices)
    # We need these so that individual attention heads and MLP inputs can be edited
    model.set_use_attn_in(True)
    model.set_use_hook_mlp_in(True)
    model.set_use_attn_result(True) # Documentation says this easily burns through GPU memory

    return model


# function to load data from a json file and input for taking first n dictionaries
def load_data(file_path: str, n: int = 28, model: HookedTransformer):n 
    """Model is needed to check the token length of the prompt."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    subset = data[:n]

    # Initialize lists
    prompts = []
    answers = []

    # Extract prompts and outputs
    outputs = []
    for ind, item in enumerate(subset):
        # if ind in [0, 1, 3, 4, 5, 6, 7, 9]:
        # print(()
        if model.to_tokens(item["prompt"]).shape[-1] == 34:
            prompts.append(item["prompt"])
            outputs.append(item["output"])

    # Group pairs of outputs and reverse the tuples alternately
    for i in range(0, len(outputs) - 1, 2):
        answers.append((outputs[i], outputs[i + 1]))
        answers.append((outputs[i + 1], outputs[i]))

    # Display the results
    print("Prompts:", prompts)
    print("Answers:", answers)
    print("Prompts:", len(prompts))
    print("Answers:", len(answers))

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
    return prompts, answers, clean_tokens, corrupted_tokens, answer_token_indices

# Function to delete variables and perform garbage collection
def delete_variable(var_name):
    if var_name in globals():
        del globals()[var_name]
    gc.collect()
    
def get_memory_usage():
    """Returns the current memory usage in MB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    print(mem_info.rss / 1024 ** 2)

def get_baseline_logits(model, clean_tokens, corrupted_tokens, answer_token_indices):
    """Get the baseline logits for clean and corrupted tokens."""
    with torch.no_grad():
        clean_logits = model(clean_tokens)[:, -1, :].to("cpu")
        print(clean_logits.shape)
        
    with torch.no_grad():
        corr_logits = model(corrupted_tokens)[:, -1, :].to("cpu")
        print(corr_logits.shape)

    clean_logit_diff = get_logit_diff(clean_logits, answer_token_indices).item()
    print(f"Clean logit diff: {clean_logit_diff:.4f}")

    corrupted_logit_diff = get_logit_diff(corr_logits, answer_token_indices).item()
    print(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")
    check_gpu_memory()

    CLEAN_BASELINE = clean_logit_diff
    CORRUPTED_BASELINE = corrupted_logit_diff

    def ioi_metric(logits, answer_token_indices=answer_token_indices):
        logits = logits.to(device)
        torch.cuda.empty_cache()
        return (get_logit_diff(logits, answer_token_indices) - CORRUPTED_BASELINE) / (
            CLEAN_BASELINE - CORRUPTED_BASELINE
        )

    delete_variable("clean_logits")
    delete_variable("corr_logits")
    get_memory_usage()

    return ioi_metric

# function to get the cache for a given input
def get_all_cache(model, tokens, ioi_metric, answer_token_indices):
    """Sequentially go through prompts to get the cache for each and concatenate them."""
    Metric = Callable[[TT["batch_and_pos_dims", "d_model"]], float]
    filter_not_qkv_input = lambda name: "_input" not in name
    # Process the first prompt
    answer_token_indices_first = answer_token_indices[0:1]
    clean_value, clean_cache_first, clean_grad_cache_first = get_cache_fwd_and_bwd(model, clean_tokens[0:1], ioi_metric, answer_token_indices_first)
    clean_cache_first = clean_cache_first.to('cpu')
    clean_grad_cache_first = clean_grad_cache_first.to('cpu')
    check_gpu_memory()
    delete_variable('clean_value')
    get_memory_usage()

    corrupted_value, corrupted_cache_first, corrupted_grad_cache_first = get_cache_fwd_and_bwd(model, corrupted_tokens[0:1], ioi_metric, answer_token_indices_first)
    corrupted_cache_first = corrupted_cache_first.to('cpu')
    corrupted_grad_cache_first = corrupted_grad_cache_first.to('cpu')
    check_gpu_memory()

    delete_variable('corrupted_value')

    for i in range(1, len(clean_tokens)):
        single_clean_tokens = clean_tokens[i:i+1]
        single_corrupted_tokens = corrupted_tokens[i:i+1]
        single_answer_token_indices = answer_token_indices[i:i+1]

        clean_value, clean_cache, clean_grad_cache = get_cache_fwd_and_bwd(model, single_clean_tokens, ioi_metric, single_answer_token_indices)
        clean_cache = clean_cache.to('cpu')
        clean_grad_cache = clean_grad_cache.to('cpu')
        clean_cache_first = clean_cache_first.concatenate(clean_cache)
        clean_grad_cache_first = clean_grad_cache_first.concatenate(clean_grad_cache)
        check_gpu_memory()

        delete_variable('clean_value')
        delete_variable('clean_cache')
        delete_variable('clean_grad_cache')
        get_memory_usage()

        corrupted_value, corrupted_cache, corrupted_grad_cache = get_cache_fwd_and_bwd(model, single_corrupted_tokens, ioi_metric, single_answer_token_indices)
        corrupted_cache = corrupted_cache.to('cpu')
        corrupted_grad_cache = corrupted_grad_cache.to('cpu')
        corrupted_cache_first = corrupted_cache_first.concatenate(corrupted_cache)
        corrupted_grad_cache_first = corrupted_grad_cache_first.concatenate(corrupted_grad_cache)
        check_gpu_memory()

        delete_variable('corrupted_value')
        delete_variable('corrupted_cache')
        delete_variable('corrupted_grad_cache')
        get_memory_usage()
        print("CURRENT INDEX: ", i)

    torch.cuda.empty_cache()

    # Test if shapes worked
    print("cache shape: ,", corrupted_cache_first["hook_embed"].shape)
    return clean_cache_first, corrupted_cache_first, corrupted_grad_cache_first, clean_grad_cache_first

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


def residual_attr_patching():
    return ""

def layer_op_attr_patching():
    return ""

def head_op_attr_patching(clean_cache_first, corrupted_cache_first, corrupted_grad_cache_first, device="cpu", save_path="plots/script_", plot=True):

    head_out_attr, head_out_labels = attr_patch_head_out(
        clean_cache_first, corrupted_cache_first, corrupted_grad_cache_first, "cpu"
    )

    sum_head_out_attr = einops.reduce(
        head_out_attr,
        "(layer head) pos -> layer head",
        "sum",
        layer=model.cfg.n_layers,
        head=model.cfg.n_heads,
    )
    check_gpu_memory()

    # Assuming head_out_values is a tensor
    head_out_values, head_out_indices = head_out_attr.sum(-1).abs().sort(descending=True)

    if plot:

        plt.figure(figsize=(10, 6))
        sns.heatmap(sum_head_out_attr.detach().numpy(), annot=False, cmap='viridis', center=0)
        plt.xlabel('Position')
        plt.ylabel('Component')
        plt.title('Head Output Attribution Patching Sum Over Position')
        plt.savefig(save_path+'head_op_sum_pos.png')

        # Convert head_out_values to numpy array for plotting
        head_out_values_np = head_out_values.detach().numpy()
        # Define the number of ticks you want
        num_ticks_x = 20
        num_ticks_y = 10

        # Generate tick positions and labels for x-axis and y-axis
        x_ticks = np.linspace(0, len(head_out_values_np) - 1, num_ticks_x, dtype=int)
        y_ticks = np.linspace(min(head_out_values_np), max(head_out_values_np), num_ticks_y)

        plt.figure(figsize=(12, 8))
        plt.xlabel('Neuron rank')
        plt.ylabel('Logit Difference ')

        plt.plot(head_out_values_np)

        # Set x and y ticks
        plt.xticks(x_ticks, x_ticks)
        plt.yticks(y_ticks, [f'{ytick:.2f}' for ytick in y_ticks])

        plt.grid(True)
        plt.savefig(save_path+'head_op_neuron_rank_vs_logit.png')

    return head_out_attr, head_out_labels, sum_head_out_attr


def head_pattern_attr_patching(clean_cache_first, corrupted_cache_first, corrupted_grad_cache_first, device="cpu"):
    head_pattern_attr, labels = attr_patch_head_pattern(
        clean_cache_first, corrupted_cache_first, corrupted_grad_cache_first, "cpu"
    )
    head_pattern_attr = einops.rearrange(
        head_pattern_attr,
        "(layer head) dest src -> layer head dest src",
        layer=model.cfg.n_layers,
        head=model.cfg.n_heads,
    )
    return head_pattern_attr, labels


def head_path_attr_patching(clean_cache_first, corrupted_cache_first, corrupted_grad_cache_first, device="cpu", save_path="plots/NL_head_path_attribution_patching_script.png", plot=True):
    head_path_attr, end_labels, start_labels = attr_patch_head_path(
        clean_cache_first, corrupted_cache_first, corrupted_grad_cache_first, "cpu"
    )
    if plot:
        plt.figure(figsize=(32, 12))
        sns.set_style("white")
        sns.heatmap(head_path_attr.sum(-1).detach().numpy(), yticklabels=end_labels, xticklabels = start_labels, annot=False, cmap='viridis', center=0)
        plt.yticks(fontsize = 5)
        plt.xlabel('Path Start (Head Output)')
        plt.ylabel('Path End (Head Input)')
        plt.title('Head Path Attribution Patching')
        plt.savefig(save_path)
    # plt.plot()
    return head_path_attr, end_labels, start_labels

def thresholded_path_patching(head_path_attr, head_out_attr, end_labels, start_labels, thres_neurons = 22,save_path="plots/NL_head_path_attribution_patching_script_thresholded.png", plot=True):
    head_out_values, head_out_indices = head_out_attr.sum(-1).abs().sort(descending=True)
    top_head_indices = head_out_indices[:thres_neurons].sort().values
    top_end_indices = []
    top_end_labels = []
    top_start_indices = []
    top_start_labels = []

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

    if plot:
        plt.figure(figsize=(28, 16))
        sns.heatmap(head_path_attr[top_end_indices, :][:, top_start_indices].sum(-1).detach().numpy(), yticklabels = top_end_labels, xticklabels= top_start_labels, annot=False, cmap='viridis', center=0)
        plt.xlabel('Path Start (Head Output)')
        plt.ylabel('Path End (Head Input)')
        plt.title('Head Path Attribution Patching')
        plt.savefig(save_path)

    num_elements = head_path_attr.numel()

    # Determine the size of each element in bytes
    # For example, float32 (default for torch.randn) is 4 bytes
    element_size = head_path_attr.element_size()

    # Calculate the total memory usage in bytes
    total_memory_bytes = num_elements * element_size

    # Convert bytes to megabytes for easier readability
    total_memory_megabytes = total_memory_bytes / (1024 ** 2)

    print(f"Expected memory usage: {total_memory_bytes} bytes ({total_memory_megabytes:.2f} MB)")
    torch.save(head_path_attr, 'patch_results/script_NL_head_path_attr.pt')
    
    return top_head_path_attr, top_end_indices, top_start_indices, top_end_labels, top_start_labels

def form_circuit(head_path_attr, user_threshold, end_labels, start_labels, top_end_labels, top_start_labels, top_end_indices, top_start_indices, save_path="circuits/codellama/script_infoRet_30prompts_data3.json", plot=True):
    # Calculate the sum over the last dimension to get the correlation values
    correlation_values = head_path_attr.sum(-1)

    # Calculate mean and standard deviation
    mean_value = correlation_values.mean().item()
    std_dev = correlation_values.std().item()

    # Define the threshold as mean + std_dev
    threshold = mean_value + std_dev

    # Calculate the absolute values of the correlation matrix
    abs_correlation_values = correlation_values.abs()
    print(f"Suggested Threshold: {threshold}")
    print(f"Provided Threshold: {user_threshold}")
    # Create a boolean mask where absolute values are greater than the threshold
    mask = abs_correlation_values > user_threshold

    # Get the indices where the condition is met
    indices = mask.nonzero(as_tuple=True)

    # Convert indices to a list of tuples
    path_indices = list(zip(indices[0].tolist(), indices[1].tolist()))

    print("Indices of paths with absolute correlation values greater than the threshold:")
    print(path_indices)

    # Create a new matrix for plotting with values above the threshold
    thresholded_matrix = correlation_values.clone()
    thresholded_matrix[~mask] = np.nan  # Set values not meeting the threshold to NaN for better heatmap visualization

    if plot:
        # Plotting the heatmap
        plt.figure(figsize=(28, 16))
        sns.heatmap(thresholded_matrix.detach().numpy(), yticklabels=top_end_labels, xticklabels=top_start_labels, annot=False, cmap='viridis', center=0)
        plt.xlabel('Path Start (Head Output)')
        plt.ylabel('Path End (Head Input)')
        plt.title('Head Path Attribution Patching 2 (Thresholded)')
        plt.savefig('plots/NL_head_path_2_attribution_patching_thresholded.png')
    # plt.show()

    circuit_dictionary = {"indices":[],
                      "labels":[]}

    for i, j in path_indices:
        circuit_dictionary["indices"].append((i, j))
        circuit_dictionary["labels"].append((end_labels[i], start_labels[j]))
        
    # Specify the file name
    # file_name = 'circuits/codellama/infoRet_30prompts_data3.json'

    # Save the dictionary as a JSON file
    with open(save_path, 'w') as json_file:
        json.dump(circuit_dictionary, json_file, indent=4)



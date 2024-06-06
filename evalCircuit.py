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



keys_comps = []
for i in range(len(list(circuit_dictionary['labels']))):
    ind = list(circuit_dictionary['indices'])[i][1]
    layer_ind = ind//32
    head_ind = ind%32
    # key = f'blocks.{layer_ind}.attn.hook_result'
    keys_comps.append((layer_ind, head_ind))
print(len(keys_comps))

for i in range(len(list(circuit_dictionary['labels']))):
    ind = list(circuit_dictionary['indices'])[i][0]
    # print(ind//(32*3), (ind%(32*3))//3)
    layer_ind = ind//32
    head_ind = ind%32
    # key = f'blocks.{layer_ind}.attn.hook_result'
    keys_comps.append((layer_ind, head_ind))
print(len(keys_comps))


list_length = len(keys_comps)

# Generate a new list of random tuples with the same length
random_keys_comps = [(random.randint(0, 31), random.randint(0, 31)) for _ in range(list_length)]

print(random_keys_comps)


# get mean activations 
mean_activations = {}
for l in range(model.cfg.n_layers):
    key = f'blocks.{l}.attn.hook_result'
    mean_activations[l] = torch.sum(clean_cache_first[key], dim=0).to("cpu")/28 #.shape
check_gpu_memory()

def ablate_setter(corrupted_activation, layer_ind, clean_activation, mean_activation=mean_activations,  keys_comps=keys_comps):
    for hed in range(32):
        if (layer_ind, hed) not in keys_comps:
            # print("here")
            mean_activation_broadcasted = mean_activation[layer_ind][:, hed, :].unsqueeze(0).expand(28, -1, -1)
            # Replace the values at the 3rd dimension of "hed" in new_cache[key]
            clean_activation[:, :, hed, :] = mean_activation_broadcasted
    return clean_activation

def ablating_hook(corrupted_activation, layer_ind, hook, clean_activation, mean_activation=mean_activations,  keys_comps=keys_comps):
    return ablate_setter(corrupted_activation, layer_ind, clean_activation, mean_activation=mean_activations,  keys_comps=keys_comps)

hooks_tuple = []

for l in range(model.cfg.n_layers):
    key = f'blocks.{l}.attn.hook_result'
    
    # Make a partial copy of the original cache for the current key
    new_cache_temp = clean_cache_first[key].clone()
    
    current_hook = partial(
    ablating_hook,
    layer_ind=l,
    clean_activation = new_cache_temp
    )   
    hooks_tuple.append((key, current_hook))

print(type(hooks_tuple))
check_gpu_memory()

patched_logits = model.to("cpu").run_with_hooks(
             clean_tokens, fwd_hooks=hooks_tuple, bwd_hooks=None)
# print(patched_logits)
check_gpu_memory()

model_answers = torch.argmax(patched_logits[:, -1, :], dim=-1)
for some_ind in range(len(model_answers)):
    print("Predicted: ", model.to_string(model_answers[some_ind]))
    print("Act answer: ", answers[some_ind][0])


# Specify the file name
file_name = 'circuits/codellama/infoRet_30prompts_data3_thres00022.json'

# Read the dictionary from the JSON file
with open(file_name, 'r') as json_file:
    struc_circuit = json.load(json_file)

# Print the loaded dictionary to verify its contents
# print(loaded_circuit_dictionary)

struc_keys_comps = []
for i in range(len(list(struc_circuit['labels']))):
    ind = list(struc_circuit['indices'])[i][1]
    layer_ind = ind//32
    head_ind = ind%32
    # key = f'blocks.{layer_ind}.attn.hook_result'
    struc_keys_comps.append((layer_ind, head_ind))
print(len(struc_keys_comps))

for i in range(len(list(struc_circuit['labels']))):
    ind = list(struc_circuit['indices'])[i][0]
    # print(ind//(32*3), (ind%(32*3))//3)
    layer_ind = ind//32
    head_ind = ind%32
    # key = f'blocks.{layer_ind}.attn.hook_result'
    struc_keys_comps.append((layer_ind, head_ind))
print(len(struc_keys_comps))


# Specify the file name
file_name = 'circuits/codellama/infoRet_30prompts_data3_NL_thres0017.json'

# Read the dictionary from the JSON file
with open(file_name, 'r') as json_file:
    nl_circuit = json.load(json_file)

# Print the loaded dictionary to verify its contents
# print(loaded_circuit_dictionary)

nl_keys_comps = []
for i in range(len(list(nl_circuit['labels']))):
    ind = list(nl_circuit['indices'])[i][1]
    layer_ind = ind//32
    head_ind = ind%32
    # key = f'blocks.{layer_ind}.attn.hook_result'
    nl_keys_comps.append((layer_ind, head_ind))
print(len(nl_keys_comps))

for i in range(len(list(nl_circuit['labels']))):
    ind = list(nl_circuit['indices'])[i][0]
    # print(ind//(32*3), (ind%(32*3))//3)
    layer_ind = ind//32
    head_ind = ind%32
    # key = f'blocks.{layer_ind}.attn.hook_result'
    nl_keys_comps.append((layer_ind, head_ind))
print(len(nl_keys_comps))



# Convert the lists to sets and find the intersection
set1 = set(struc_keys_comps)
set2 = set(nl_keys_comps)
intersection = set1.intersection(set2)

# Convert the intersection back to a list if needed
overlap_list = list(intersection)

# Print the result
print("Overlap:", overlap_list)

for ele in overlap_list:
    print(ele[0], ele[1])
    # print(ele[0]*32 + ele[1])
    if ele[0]*32 + ele[1] > 1024: 
        continue
    print(head_out_values_np[ele[0]*32 + ele[1]])



# SIGNIFICANCE OF EACH OVERLAPPED NEURON

highlight_indices = []
for ele in overlap_list:
    # print(ele[0], ele[1])
    # print(ele[0]*32 + ele[1])
    if ele[0]*32 + ele[1] > 1024: 
        continue
    highlight_indices.append(ele[0]*32 + ele[1])
    # print(head_out_values_np[ele[0]*32 + ele[1]])

head_path_token_summed = head_path_attr.sum(-1)

# 1. Sum along the second axis (dim=0)
sum_along_second_axis = head_path_token_summed.sum(dim=0)  # Resulting shape: [1024]

# 2. Sum along the first axis (dim=1)
sum_along_first_axis = head_path_token_summed.sum(dim=1)  # Resulting shape: [3072]

# Sum in successive groups of three to transform the shape to [1024]
sum_in_groups_of_three = sum_along_first_axis.view(-1, 3).sum(dim=1)  # Resulting shape: [1024]

print("Sum along second axis (shape [1024]):", sum_along_second_axis.shape)
print("Sum along first axis and then in groups of three (shape [1024]):", sum_in_groups_of_three.shape)

# Sort the arrays by absolute values in descending order and get the original indices
sorted_indices_second_axis = np.argsort(-np.abs(sum_along_second_axis_np))
sorted_sum_along_second_axis = np.abs(sum_along_second_axis_np[sorted_indices_second_axis])

sorted_indices_groups_of_three = np.argsort(-np.abs(sum_in_groups_of_three_np))
sorted_sum_in_groups_of_three = np.abs(sum_in_groups_of_three_np[sorted_indices_groups_of_three])

# Convert highlight indices to the sorted array indices
highlight_indices_second_axis = [np.where(sorted_indices_second_axis == idx)[0][0] for idx in highlight_indices]
highlight_indices_groups_of_three = [np.where(sorted_indices_groups_of_three == idx)[0][0] for idx in highlight_indices]

# Create plots
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Plot sum along the second axis
axes[0].plot(sorted_sum_along_second_axis, label='Sum along second axis')
axes[0].scatter(highlight_indices_second_axis, sorted_sum_along_second_axis[highlight_indices_second_axis], color='red', zorder=5)
axes[0].set_title('Head output score of overlapped heads (NL task)')
axes[0].set_xlabel('Neuron Rank')
axes[0].set_ylabel('Logit Difference (Absolute)')
axes[0].legend()

# Plot sum in groups of three
axes[1].plot(sorted_sum_in_groups_of_three, label='Sum in groups of three')
axes[1].scatter(highlight_indices_groups_of_three, sorted_sum_in_groups_of_three[highlight_indices_groups_of_three], color='red', zorder=5)
axes[1].set_title('Head input score of overlapped heads (NL task)')
axes[1].set_xlabel('Neuron Rank')
axes[1].set_ylabel('Logit Difference (Absolute)')
axes[1].legend()

# Display the plots
plt.tight_layout()
plt.show()


head_path_attr_struc = torch.load("patch_results/head_path_attr.pt")
head_path_attr_struc.shape



head_path_token_summed = head_path_attr_struc.sum(-1)

# 1. Sum along the second axis (dim=0)
sum_along_second_axis = head_path_token_summed.sum(dim=0)  # Resulting shape: [1024]

# 2. Sum along the first axis (dim=1)
sum_along_first_axis = head_path_token_summed.sum(dim=1)  # Resulting shape: [3072]

# Sum in successive groups of three to transform the shape to [1024]
sum_in_groups_of_three = sum_along_first_axis.view(-1, 3).sum(dim=1)  # Resulting shape: [1024]

print("Sum along second axis (shape [1024]):", sum_along_second_axis.shape)
print("Sum along first axis and then in groups of three (shape [1024]):", sum_in_groups_of_three.shape)

# Convert tensors to numpy arrays for easier plotting
sum_along_second_axis_np = sum_along_second_axis.detach().numpy()
sum_in_groups_of_three_np = sum_in_groups_of_three.detach().numpy()

# Sort the arrays by absolute values in descending order and get the original indices
sorted_indices_second_axis = np.argsort(-np.abs(sum_along_second_axis_np))
sorted_sum_along_second_axis = np.abs(sum_along_second_axis_np[sorted_indices_second_axis])

sorted_indices_groups_of_three = np.argsort(-np.abs(sum_in_groups_of_three_np))
sorted_sum_in_groups_of_three = np.abs(sum_in_groups_of_three_np[sorted_indices_groups_of_three])

# Convert highlight indices to the sorted array indices
highlight_indices_second_axis = [np.where(sorted_indices_second_axis == idx)[0][0] for idx in highlight_indices]
highlight_indices_groups_of_three = [np.where(sorted_indices_groups_of_three == idx)[0][0] for idx in highlight_indices]

# Create plots
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Plot sum along the second axis
axes[0].plot(sorted_sum_along_second_axis, label='Sum along second axis')
axes[0].scatter(highlight_indices_second_axis, sorted_sum_along_second_axis[highlight_indices_second_axis], color='red', zorder=5)
axes[0].set_title('Head output score of overlapped heads (Struc task)')
axes[0].set_xlabel('Neuron Rank')
axes[0].set_ylabel('Logit Difference (Absolute)')
axes[0].legend()

# Plot sum in groups of three
axes[1].plot(sorted_sum_in_groups_of_three, label='Sum in groups of three')
axes[1].scatter(highlight_indices_groups_of_three, sorted_sum_in_groups_of_three[highlight_indices_groups_of_three], color='red', zorder=5)
axes[1].set_title('Head input score of overlapped heads (Struc task)')
axes[1].set_xlabel('Neuron Rank')
axes[1].set_ylabel('Logit Difference (Absolute)')
axes[1].legend()

# Display the plots
plt.tight_layout()
plt.show()
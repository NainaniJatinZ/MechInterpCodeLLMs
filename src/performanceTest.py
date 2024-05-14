from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
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

model = HookedTransformer.from_pretrained("CodeLlama-7b-hf", n_devices=num_gpus)

with open('data/info_retrieval/instructed_trial2.json', 'r') as f:
    data2 = json.load(f)

# Take the first few dictionaries (e.g., first 3)
subset2 = data2  #[9:]

# Initialize lists
prompts2 = []

# Extract prompts and outputs
outputs2 = []
for ind, item in enumerate(subset2):
    # if ind in [0, 1, 3, 4, 5, 6, 7, 9]:
    prompts2.append(item["prompt"])
    outputs2.append(item["output"])


def evaluate_next_token_accuracy(prompts, answers, model):
    # total_examples = len(prompts)
    correct_predictions = 0
    for ind in range(len(prompts)):
        # print(prompts[ind])
        with torch.no_grad():
            clean_tokens = model.to_tokens(prompts[ind])
            logits = model(clean_tokens)[:, -1, :]
            model_answers = torch.argmax(logits, dim=-1)
            print("pred: ",model.to_string(model_answers))
            if model.to_string(model_answers) == answers[ind]:
                # print("pred: ",model.to_string(model_answers))
                print("ans: ",answers[ind])
                correct_predictions+=1
    return correct_predictions / len(prompts)

print("Accuracy is: ", evaluate_next_token_accuracy(prompts2, outputs2, model))



# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# model = AutoModelForCausalLM.from_pretrained("gpt2")

# # Evaluation function
# def evaluate_next_token_accuracy(prompts_dataset, tokenizer, model):
#     total_examples = len(prompts_dataset)
#     correct_predictions = 0
    
#     for input_text, next_token in prompts_dataset:
#         # Tokenize input text
#         input_ids = tokenizer.encode(input_text, return_tensors="pt")
        
#         # Generate predictions
#         with torch.no_grad():
#             logits = model(input_ids)[0][:, -1, :]  # Get logits for the next token
#             predicted_token_id = torch.argmax(logits, dim=-1).item()
#             predicted_token = tokenizer.decode(predicted_token_id)
        
#         # Check if prediction matches the ground truth
#         if predicted_token == next_token:
#             correct_predictions += 1
    
#     accuracy = correct_predictions / total_examples
#     return accuracy

# # Example prompts dataset
# prompts = [("The cat", "is"), ("I am", "going"), ("OpenAI is a", "research")]

# # Evaluate accuracy
# accuracy = evaluate_next_token_accuracy(prompts, tokenizer, model)
# print("Accuracy:", accuracy)
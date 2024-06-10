import sys
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
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets

import transformer_lens
import transformer_lens.utils as utils
import transformer_lens.patching as patching
from transformer_lens.hook_points import HookedRootModule, HookPoint
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

from config import HF_TOKEN, HF_PATH
import psutil
import gc


class TransformerLensPipeline:
    def __init__(self, model_name: str, data_file: str, num_prompts: int = 28):
        self.model_name = model_name
        self.data_file = data_file
        self.num_prompts = num_prompts
        self.device = self.get_device()
        self.model = self.load_model()
        self.prompts, self.answers, self.clean_tokens, self.corrupted_tokens, self.answer_token_indices = self.load_data()
        self.CLEAN_BASELINE, self.CORRUPTED_BASELINE = self.get_baseline_logits()

    def get_device(self):
        num_gpus = torch.cuda.device_count()
        device_id = 0
        if num_gpus > 0:
            device = "cuda:0"
        else:
            device = "cpu"
        return device

    def check_gpu_memory(self, max_alloc=0.9):
        if not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
        max_alloc = 1 if max_alloc > 1 else max_alloc
        num_gpus = torch.cuda.device_count()
        for gpu in range(num_gpus):
            memory_reserved = torch.cuda.memory_reserved(device=gpu)
            memory_allocated = torch.cuda.memory_allocated(device=gpu)
            total_memory = torch.cuda.get_device_properties(gpu).total_memory
            if memory_reserved > max_alloc * total_memory:
                if device_id < num_gpus - 1:
                    device_id += 1
                    self.device = f"cuda:{device_id}"
                else:
                    print("Cannot switch primary device, all GPUs are nearly full")

    def timeit(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"Function {func.__name__!r} executed in {end_time - start_time:.4f} seconds")
            return result
        return wrapper

    def load_model(self) -> HookedTransformer:
        model = HookedTransformer.from_pretrained(self.model_name, n_devices=torch.cuda.device_count())
        model.set_use_attn_in(True)
        model.set_use_hook_mlp_in(True)
        model.set_use_attn_result(True)
        return model

    def load_data(self):
        with open(self.data_file, 'r') as f:
            data = json.load(f)
        subset = data[:self.num_prompts]

        prompts = []
        answers = []
        outputs = []
        for ind, item in enumerate(subset):
            if self.model.to_tokens(item["prompt"]).shape[-1] == 34:
                prompts.append(item["prompt"])
                outputs.append(item["output"])

        for i in range(0, len(outputs) - 1, 2):
            answers.append((outputs[i], outputs[i + 1]))
            answers.append((outputs[i + 1], outputs[i]))

        clean_tokens = self.model.to_tokens(prompts)
        corrupted_tokens = clean_tokens[
            [(i + 1 if i % 2 == 0 else i - 1) for i in range(len(clean_tokens))]
        ]

        answer_token_indices = torch.tensor([[self.model.to_single_token(answers[i][j]) for j in range(2)] for i in range(len(answers))], device=self.device)
        return prompts, answers, clean_tokens, corrupted_tokens, answer_token_indices

    def delete_variable(self, var_name):
        if var_name in globals():
            del globals()[var_name]
        gc.collect()

    def get_memory_usage(self):
        process = psutil.Process()
        mem_info = process.memory_info()
        print(mem_info.rss / 1024 ** 2)

    def get_baseline_logits(self):
        with torch.no_grad():
            clean_logits = self.model(self.clean_tokens)[:, -1, :].to("cpu")

        with torch.no_grad():
            corr_logits = self.model(self.corrupted_tokens)[:, -1, :].to("cpu")

        clean_logit_diff = self.get_logit_diff(clean_logits).item()
        corrupted_logit_diff = self.get_logit_diff(corr_logits).item()
        check_gpu_memory()

        self.delete_variable("clean_logits")
        self.delete_variable("corr_logits")
        self.get_memory_usage()

        return clean_logit_diff, corrupted_logit_diff

    def get_logit_diff(self, logits):
        logits = logits.to(self.answer_token_indices.device)
        correct_logits = logits.gather(1, self.answer_token_indices[:, 0].unsqueeze(1))
        incorrect_logits = logits.gather(1, self.answer_token_indices[:, 1].unsqueeze(1))
        return (correct_logits - incorrect_logits).mean()

    @timeit
    def get_cache_fwd_and_bwd(self, tokens, metric, answer_indices):
        self.model.reset_hooks()
        cache = {}

        def forward_cache_hook(act, hook):
            act = act.to(self.device)
            torch.cuda.empty_cache()
            cache[hook.name] = act

        self.model.add_hook(lambda name: "_input" not in name, forward_cache_hook, "fwd")
        grad_cache = {}

        def backward_cache_hook(act, hook):
            act = act.to(self.device)
            torch.cuda.empty_cache()
            grad_cache[hook.name] = act

        self.model.add_hook(lambda name: "_input" not in name, backward_cache_hook, "bwd")

        result = self.model(tokens).to(self.device)
        torch.cuda.empty_cache()
        value = metric(result, answer_indices)
        value.backward()

        self.model.reset_hooks()
        torch.cuda.empty_cache()

        cache = ActivationCache(cache, self.model).to(self.device)
        grad_cache = ActivationCache(grad_cache, self.model).to(self.device)

        return value.item(), cache, grad_cache

    @timeit
    def get_all_cache(self, ioi_metric):
        answer_token_indices_first = self.answer_token_indices[0:1]
        clean_value, clean_cache_first, clean_grad_cache_first = self.get_cache_fwd_and_bwd(self.clean_tokens[0:1], ioi_metric, answer_token_indices_first)
        clean_cache_first = clean_cache_first.to('cpu')
        clean_grad_cache_first = clean_grad_cache_first.to('cpu')
        self.check_gpu_memory()
        self.delete_variable('clean_value')
        self.get_memory_usage()

        corrupted_value, corrupted_cache_first, corrupted_grad_cache_first = self.get_cache_fwd_and_bwd(self.corrupted_tokens[0:1], ioi_metric, answer_token_indices_first)
        corrupted_cache_first = corrupted_cache_first.to('cpu')
        corrupted_grad_cache_first = corrupted_grad_cache_first.to('cpu')
        self.check_gpu_memory()
        self.delete_variable('corrupted_value')

        for i in range(1, len(self.clean_tokens)):
            single_clean_tokens = self.clean_tokens[i:i + 1]
            single_corrupted_tokens = self.corrupted_tokens[i:i + 1]
            single_answer_token_indices = self.answer_token_indices[i:i + 1]

            clean_value, clean_cache, clean_grad_cache = self.get_cache_fwd_and_bwd(self.model, single_clean_tokens, ioi_metric, single_answer_token_indices)
            clean_cache = clean_cache.to('cpu')
            clean_grad_cache = clean_grad_cache.to('cpu')
            clean_cache_first = clean_cache_first.concatenate(clean_cache)
            clean_grad_cache_first = clean_grad_cache_first.concatenate(clean_grad_cache)
            self.check_gpu_memory()
            self.delete_variable('clean_value')
            self.delete_variable('clean_cache')
            self.delete_variable('clean_grad_cache')
            self.get_memory_usage()

            corrupted_value, corrupted_cache, corrupted_grad_cache = self.get_cache_fwd_and_bwd(self.model, single_corrupted_tokens, ioi_metric, single_answer_token_indices)
            corrupted_cache = corrupted_cache.to('cpu')
            corrupted_grad_cache = corrupted_grad_cache.to('cpu')
            corrupted_cache_first = corrupted_cache_first.concatenate(corrupted_cache)
            corrupted_grad_cache_first = corrupted_grad_cache_first.concatenate(corrupted_grad_cache)
            self.check_gpu_memory()
            self.delete_variable('corrupted_value')
            self.delete_variable('corrupted_cache')
            self.delete_variable('corrupted_grad_cache')
            self.get_memory_usage()
            print("CURRENT INDEX: ", i)

        torch.cuda.empty_cache()
        print("cache shape: ,", corrupted_cache_first["hook_embed"].shape)
        return clean_cache_first, corrupted_cache_first, corrupted_grad_cache_first, clean_grad_cache_first

    @timeit
    def attr_patch_head_out(self, clean_cache, corrupted_cache, corrupted_grad_cache, device="cpu"):
        head_out_attr, head_out_labels = self.attr_patch_head_out(clean_cache, corrupted_cache, corrupted_grad_cache, device)
        sum_head_out_attr = einops.reduce(
            head_out_attr,
            "(layer head) pos -> layer head",
            "sum",
            layer=self.model.cfg.n_layers,
            head=self.model.cfg.n_heads,
        )
        self.check_gpu_memory()
        head_out_values, head_out_indices = head_out_attr.sum(-1).abs().sort(descending=True)

        plt.figure(figsize=(10, 6))
        sns.heatmap(sum_head_out_attr.detach().numpy(), annot=False, cmap='viridis', center=0)
        plt.xlabel('Position')
        plt.ylabel('Component')
        plt.title('Head Output Attribution Patching Sum Over Position')
        plt.savefig('plots/NL_head_sum_attribution_patching.png')
        plt.show()

        head_out_values_np = head_out_values.detach().numpy()
        num_ticks_x = 20
        num_ticks_y = 10
        x_ticks = np.linspace(0, len(head_out_values_np) - 1, num_ticks_x, dtype=int)
        y_ticks = np.linspace(min(head_out_values_np), max(head_out_values_np), num_ticks_y)

        plt.figure(figsize=(12, 8))
        plt.xlabel('Neuron rank')
        plt.ylabel('Logit Difference ')
        plt.plot(head_out_values_np)
        plt.xticks(x_ticks, x_ticks)
        plt.yticks(y_ticks, [f'{ytick:.2f}' for ytick in y_ticks])
        plt.grid(True)
        plt.savefig('plots/NL_head_op_neuron_rank_vs_logit.png')

        return head_out_attr, head_out_labels, sum_head_out_attr

    def attr_patch_head_pattern(self, clean_cache, corrupted_cache, corrupted_grad_cache, device="cpu"):
        head_pattern_attr, labels = self.attr_patch_head_pattern(clean_cache, corrupted_cache, corrupted_grad_cache, device)
        head_pattern_attr = einops.rearrange(
            head_pattern_attr,
            "(layer head) dest src -> layer head dest src",
            layer=self.model.cfg.n_layers,
            head=self.model.cfg.n_heads,
        )
        return head_pattern_attr, labels

    @timeit
    def attr_patch_head_path(self, clean_cache, corrupted_cache, corrupted_grad_cache, device="cpu"):
        head_path_attr, end_labels, start_labels = self.attr_patch_head_path(clean_cache, corrupted_cache, corrupted_grad_cache, device)
        plt.figure(figsize=(32, 12))
        sns.set_style("white")
        sns.heatmap(head_path_attr.sum(-1).detach().numpy(), yticklabels=end_labels, xticklabels=start_labels, annot=False, cmap='viridis', center=0)
        plt.yticks(fontsize=5)
        plt.xlabel('Path Start (Head Output)')
        plt.ylabel('Path End (Head Input)')
        plt.title('Head Path Attribution Patching')
        plt.savefig('plots/NL_head_path_attribution_patching.png')
        plt.show()
        return head_path_attr, end_labels, start_labels

    def thresholded_path_patching(self, head_path_attr, head_out_attr, end_labels, start_labels, thres_neurons=22, save_path="plots/NL_head_path_attribution_patching_thresholded.png"):
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
        self.check_gpu_memory()
        plt.figure(figsize=(28, 16))
        sns.heatmap(head_path_attr[top_end_indices, :][:, top_start_indices].sum(-1).detach().numpy(), yticklabels=top_end_labels, xticklabels=top_start_labels, annot=False, cmap='viridis', center=0)
        plt.xlabel('Path Start (Head Output)')
        plt.ylabel('Path End (Head Input)')
        plt.title('Head Path Attribution Patching')
        plt.savefig(save_path)
        plt.show()
        num_elements = head_path_attr.numel()
        element_size = head_path_attr.element_size()
        total_memory_bytes = num_elements * element_size
        total_memory_megabytes = total_memory_bytes / (1024 ** 2)
        print(f"Expected memory usage: {total_memory_bytes} bytes ({total_memory_megabytes:.2f} MB)")
        torch.save(head_path_attr, 'patch_results/NL_head_path_attr.pt')
        return top_head_path_attr, top_end_indices, top_start_indices, top_end_labels, top_start_labels

    def form_circuit(self, head_path_attr, user_threshold, end_labels, start_labels, top_end_labels, top_start_labels, top_end_indices, top_start_indices, save_path="circuits/codellama/infoRet_30prompts_data3_NL_thres0017.json"):
        correlation_values = head_path_attr.sum(-1)
        mean_value = correlation_values.mean().item()
        std_dev = correlation_values.std().item()
        threshold = mean_value + std_dev
        abs_correlation_values = correlation_values.abs()
        mask = abs_correlation_values > user_threshold
        indices = mask.nonzero(as_tuple=True)
        path_indices = list(zip(indices[0].tolist(), indices[1].tolist()))
        print(f"Suggested Threshold: {threshold}")
        print(f"Provided Threshold: {user_threshold}")
        print("Indices of paths with absolute correlation values greater than the threshold:")
        print(path_indices)
        thresholded_matrix = correlation_values.clone()
        thresholded_matrix[~mask] = np.nan
        plt.figure(figsize=(28, 16))
        sns.heatmap(thresholded_matrix.detach().numpy(), yticklabels=top_end_labels, xticklabels=top_start_labels, annot=False, cmap='viridis', center=0)
        plt.xlabel('Path Start (Head Output)')
        plt.ylabel('Path End (Head Input)')
        plt.title('Head Path Attribution Patching 2 (Thresholded)')
        plt.savefig('plots/NL_head_path_2_attribution_patching_thresholded.png')
        plt.show()
        circuit_dictionary = {"indices": [], "labels": []}
        for i, j in path_indices:
            circuit_dictionary["indices"].append((i, j))
            circuit_dictionary["labels"].append((end_labels[i], start_labels[j]))
        with open(save_path, 'w') as json_file:
            json.dump(circuit_dictionary, json_file, indent=4)

    def run_pipeline(self, user_threshold):
        ioi_metric = self.get_baseline_logits()
        clean_cache_first, corrupted_cache_first, corrupted_grad_cache_first, clean_grad_cache_first = self.get_all_cache(ioi_metric)
        head_out_attr, head_out_labels, sum_head_out_attr = self.attr_patch_head_out(clean_cache_first, corrupted_cache_first, corrupted_grad_cache_first, device="cpu")
        head_pattern_attr, labels = self.attr_patch_head_pattern(clean_cache_first, corrupted_cache_first, corrupted_grad_cache_first, device="cpu")
        head_path_attr, end_labels, start_labels = self.attr_patch_head_path(clean_cache_first, corrupted_cache_first, corrupted_grad_cache_first, device="cpu")
        top_head_path_attr, top_end_indices, top_start_indices, top_end_labels, top_start_labels = self.thresholded_path_patching(head_path_attr, head_out_attr, end_labels, start_labels, thres_neurons=22, save_path="plots/NL_head_path_attribution_patching_script_thresholded.png")
        self.form_circuit(head_path_attr, user_threshold, end_labels, start_labels, top_end_labels, top_start_labels, top_end_indices, top_start_indices, save_path="circuits/codellama/script_infoRet_30prompts_data3.json")


if __name__ == "__main__":
    pipeline = TransformerLensPipeline(model_name="CodeLlama-7b-hf", data_file='data/info_retrieval/instructed_trial4_NL.json')
    pipeline.run_pipeline(user_threshold=0.017)

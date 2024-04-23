import sys
import math
import torch
import matplotlib.pyplot as plt
from datasets import Dataset
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer, LlamaForCausalLM
from baukit import TraceDict, nethook
from einops import rearrange, einsum
from tqdm import tqdm

sys.path.append("../")
from data.data_utils import *
from experiment_1.pp_utils import compute_prev_query_box_pos

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 5
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

tokenizer = LlamaTokenizer.from_pretrained(
    "hf-internal-testing/llama-tokenizer", padding_side="right"
)
tokenizer.pad_token_id = tokenizer.eos_token_id

data_file = "../data/dataset.jsonl"
batch_size = 1

raw_data = sample_box_data(
    tokenizer=tokenizer,
    num_samples=500,
    data_file=data_file,
)

dataset = Dataset.from_dict(
    {
        "input_ids": raw_data[0],
        "last_token_indices": raw_data[1],
        "labels": raw_data[2],
    }
).with_format("torch")

print(f"Length of dataset: {len(dataset)}")

dataloader = DataLoader(dataset, batch_size=batch_size)

path = "/home/local_nikhil/Projects/llama_weights/7B"

model = LlamaForCausalLM.from_pretrained(path).to(device)
tokenizer = LlamaTokenizer.from_pretrained(
    "hf-internal-testing/llama-tokenizer", padding_side="right"
)
tokenizer.pad_token_id = tokenizer.eos_token_id
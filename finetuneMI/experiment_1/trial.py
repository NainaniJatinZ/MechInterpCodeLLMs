import sys
import math
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from torch.cuda.amp import autocast
import torch
import torch
import matplotlib.pyplot as plt
from datasets import Dataset
from collections import defaultdict
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer, LlamaForCausalLM
import json
from transformers import LlamaConfig
from einops import rearrange, einsum
from tqdm import tqdm
import sys
import os
sys.path.insert(0, '/home/jnainani_umass_edu/codellm/MechInterpCodeLLMs/finetuneMI')
print("Current Working Directory:", os.getcwd())
from data.data_utils import *
from baukit.baukit import nethook
from baukit.baukit.nethook import TraceDict
from experiment_1.pp_utils import compute_prev_query_box_pos
print("here")
pretrained_model_dir = os.path.abspath("../../pretrained_models/llama/llama-2-7b")
print("LLAMA 2 Weights: ", pretrained_model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is: ", device)

seed = 5
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

data_file = "data/dataset.jsonl"
batch_size = 1

tokenizer = LlamaTokenizer.from_pretrained(
    "hf-internal-testing/llama-tokenizer", padding_side="right"
)
tokenizer.pad_token_id = tokenizer.eos_token_id

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
with open('/home/jnainani_umass_edu/codellm/pretrained_models/llama/llama-2-7b/params.json', 'r') as f:
    config_dict = json.load(f)
config = LlamaConfig(**config_dict)

model = LlamaForCausalLM(config)
model_weights = torch.load(f'{pretrained_model_dir}/consolidated.00.pth', map_location=device)
model.load_state_dict(model_weights)


model = torch.nn.DataParallel(model).to(device)  # Data parallelism
model.eval()

dummy_text = "Your dummy text here."
input_ids = tokenizer.encode(dummy_text, return_tensors="pt")

model.eval()

# If you are using a GPU, move the input ids to the GPU
if torch.cuda.is_available():
    input_ids = input_ids.to('cuda')
    model = model.to('cuda')

# Perform inference
model.eval()
with torch.no_grad():
    with autocast():  # Enable mixed precision
        outputs = model.generate(input_ids, pad_token_id=tokenizer.pad_token_id, max_length=50)

# Decode predictions
predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Predicted Text:", predicted_text)
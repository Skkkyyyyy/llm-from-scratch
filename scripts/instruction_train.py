import json, torch, tiktoken
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel

from model.GPTModel import GPTModel
from utils.weights import load_weights_into_gpt
from scripts.gpt_train import calc_loss_batch, calc_loss_loader, train_model
from data.instruction_dataloader import InstructionDataset, collate_fn, format_input 
from model.lora import replace_linear_with_lora 

# main = 
# setup: device, random seed
# data: load json -> split into train/val/test -> datasets -> dataloader
# model: config -> model + load weights to model -> to device
# train: optimizer -> train model 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(123)

# ---------- model (fixed config) ----------
cfg = {
    "vocab_size": 50257,
    "context_length": 1024,   # must match GPT-2's wpe
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 24,
    "drop_rate": 0.0,
    "qkv_bias": True,         # GPT-2 has Q/K/V biases
}
model = GPTModel(cfg)
sd = GPT2LMHeadModel.from_pretrained("gpt2-medium").state_dict()
load_weights_into_gpt(model, sd, cfg["n_layers"])

for p in model.parameters():
    p.requires_grad = False 
replace_linear_with_lora(model, rank=64, alpha=16)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable: {trainable:,} / {total:,}")

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# ---------- data ----------
import os, json, requests

file_path = "datasets/instruction-data.json"
url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/"
       "ch07/01_main-chapter-code/instruction-data.json")

if not os.path.exists(file_path):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(response.text)

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

n_train = int(len(data) * 0.85)
n_test  = int(len(data) * 0.10)
train_data = data[:n_train]
test_data  = data[n_train:n_train + n_test]
val_data   = data[n_train + n_test:]

tokenizer = tiktoken.get_encoding("gpt2")
train_loader = DataLoader(InstructionDataset(train_data, tokenizer), batch_size=8,
                          shuffle=True, drop_last=True, collate_fn=collate_fn)
val_loader   = DataLoader(InstructionDataset(val_data, tokenizer), batch_size=8,
                          shuffle=False, collate_fn=collate_fn)

# ---------- stage 1: loss BEFORE training ----------
model.eval()
with torch.no_grad():
    print("train loss:", calc_loss_loader(train_loader, model, device, num_batches=5))
    print("val loss:  ", calc_loss_loader(val_loader, model, device, num_batches=5))

optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=5e-4,
    weight_decay=0.1,
)

start_context = format_input(val_data[0]) + "\n\n### Response:\n"

train_losses, val_losses, tokens_seen = train_model(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=3, eval_freq=10, eval_iter=5,
    start_context=start_context, tokenizer=tokenizer,
)

torch.save(model.state_dict(), "outputs/checkpoints/instruction_lora.pth")

# ============================================================
# File: unipelt_finetuning.py
# Method 6 — UniPELT (Unified Parameter‑Efficient Learning Technique)
# Date: 2026-04-22
#
# Full implementation for:
#   - LoRA + Prefix + Adapter + Gating (UniPELT)
#   - LLaMA‑3‑8B / Mistral‑7B
#   - Persian Intent Classification
#
# Includes:
#   - Dataset loader
#   - Custom UniPELT adapter module
#   - Insertion into transformer
#   - Training loop
#   - Evaluation
#   - Gating mechanism
#   - Reproducibility table (inside code)
# ============================================================

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from accelerate import Accelerator
from tqdm import tqdm


# ============================================================
# Dataset Loader
# ============================================================
class IntentDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=128):
        self.samples = [json.loads(line) for line in open(path, "r", encoding="utf-8")]
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.labels = sorted({s["label"] for s in self.samples})
        self.label2id = {l: i for i, l in enumerate(self.labels)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        encoded = self.tokenizer(
            s["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        encoded["labels"] = torch.tensor(self.label2id[s["label"]])
        return encoded


# ============================================================
# UniPELT Module
#
# Contains:
#   - LoRA path
#   - Prefix path
#   - Adapter path
#   - Learnable gates to combine
#
# Formula:
# Output = gate_lora * LoRA(x) +
#          gate_prefix * Prefix(x) +
#          gate_adapter * Adapter(x)
# ============================================================
class UniPELTModule(nn.Module):
    def __init__(self, hidden_size, rank=8, prefix_len=10, adapter_hidden=64):
        super().__init__()
        self.hidden = hidden_size

        # ---------------- LoRA ----------------
        self.lora_A = nn.Linear(hidden_size, rank, bias=False)
        self.lora_B = nn.Linear(rank, hidden_size, bias=False)
        nn.init.xavier_normal_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

        # ---------------- Prefix Tuning ----------------
        self.prefix_len = prefix_len
        self.prefix_embeddings = nn.Parameter(torch.randn(prefix_len, hidden_size))

        # ---------------- Adapter (Houlsby Style) ----------------
        self.down = nn.Linear(hidden_size, adapter_hidden)
        self.up = nn.Linear(adapter_hidden, hidden_size)
        self.act = nn.ReLU()

        # ---------------- Gating Mechanism ----------------
        # gates ∈ [0,1] after sigmoid
        self.gate_lora = nn.Parameter(torch.tensor(0.33))
        self.gate_prefix = nn.Parameter(torch.tensor(0.33))
        self.gate_adapter = nn.Parameter(torch.tensor(0.33))

    def forward(self, x):

        # ---------- LoRA ----------
        lora_out = self.lora_B(self.lora_A(x))

        # ---------- Prefix ----------
        batch = x.size(0)
        prefix_expand = self.prefix_embeddings.unsqueeze(0).expand(batch, -1, -1)
        prefix_out = torch.mean(prefix_expand, dim=1, keepdim=True)  # broadcast effect

        # ---------- Adapter ----------
        adapter_out = self.up(self.act(self.down(x)))

        # ---------- Gated Combination ----------
        gate_l = torch.sigmoid(self.gate_lora)
        gate_p = torch.sigmoid(self.gate_prefix)
        gate_a = torch.sigmoid(self.gate_adapter)

        combined = gate_l * lora_out + gate_p * prefix_out + gate_a * adapter_out

        return x + combined  # residual


# ============================================================
# Insert UniPELT modules into transformer
# Apply to q_proj and v_proj (same as other PEFT methods)
# ============================================================
def apply_unipelt(model, hidden_size, rank=8, prefix_len=10, adapter_hidden=64):

    unipelt_modules = []

    for name, module in model.named_modules():
        if any(t in name for t in ["q_proj", "v_proj"]) and isinstance(module, nn.Linear):

            uni = UniPELTModule(
                hidden_size=hidden_size,
                rank=rank,
                prefix_len=prefix_len,
                adapter_hidden=adapter_hidden,
            )

            # override forward
            def make_forward(old_module, uni_module):
                def new_forward(x):
                    w = old_module.weight
                    out = F.linear(x, w, old_module.bias)
                    return uni_module(out)
                return new_forward

            module.forward = make_forward(module, uni)
            unipelt_modules.append(uni)

    return unipelt_modules


# ============================================================
# Evaluation
# ============================================================
def evaluate(model, dataloader, accelerator):
    model.eval()
    correct = 0
    total = 0

    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)

        labels = batch["labels"]

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


# ============================================================
# Main UniPELT Training
# ============================================================
def main():

    accelerator = Accelerator()
    accelerator.print("Starting UniPELT Fine‑Tuning...")

    model_name = "meta-llama/Llama-3-8b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Dataset
    train_set = IntentDataset("data/intent_train.json", tokenizer)
    dev_set   = IntentDataset("data/intent_dev.json", tokenizer)
    test_set  = IntentDataset("data/intent_test.json", tokenizer)

    # Base Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(train_set.labels),
    )

    hidden_size = model.config.hidden_size  # LLaMA = 4096

    # ============================================================
    # UniPELT Configuration
    # ============================================================
    rank = 8
    prefix_len = 10
    adapter_hidden = 64

    unipelt_modules = apply_unipelt(
        model,
        hidden_size=hidden_size,
        rank=rank,
        prefix_len=prefix_len,
        adapter_hidden=adapter_hidden,
    )

    # Train only UniPELT parameters
    unipelt_params = []
    for m in unipelt_modules:
        unipelt_params += list(m.parameters())

    # Dataloaders
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    dev_loader   = DataLoader(dev_set, batch_size=4)
    test_loader  = DataLoader(test_set, batch_size=4)

    # Optimizer
    learning_rate = 2e-4
    optimizer = AdamW(unipelt_params, lr=learning_rate, weight_decay=0.0)

    num_epochs = 5
    training_steps = len(train_loader) * num_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * training_steps),
        num_training_steps=training_steps,
    )

    # Accelerator preparation
    (model, optimizer, train_loader, dev_loader, test_loader, scheduler) = accelerator.prepare(
        model, optimizer, train_loader, dev_loader, test_loader, scheduler
    )

    # ============================================================
    # Training Loop
    # ============================================================
    for epoch in range(num_epochs):

        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"UniPELT Epoch {epoch+1}"):

            outputs = model(**batch)
            loss = outputs.loss

            accelerator.backward(loss)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        dev_acc = evaluate(model, dev_loader, accelerator)

        accelerator.print(
            f"Epoch {epoch+1}  |  Loss={total_loss:.4f}  |  Dev Acc={dev_acc:.4f}"
        )

    # -----------------------------
    # Final Test Evaluation
    # -----------------------------
    test_acc = evaluate(model, test_loader, accelerator)

    accelerator.print(f"UniPELT Test Accuracy = {test_acc:.4f}")

    # -----------------------------
    # Save Model
    # -----------------------------
    accelerator.wait_for_everyone()

    save_dir = "outputs/unipelt_model"
    accelerator.unwrap_model(model).save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    accelerator.print(f"UniPELT adapter saved to: {save_dir}")
    accelerator.print("UniPELT Fine‑Tuning completed successfully.")


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    main()


# ============================================================
# Reproducibility Table (inside code)
# ============================================================
# Backbone Model: LLaMA‑3‑8B / Mistral‑7B
# Method: UniPELT (LoRA + Prefix + Adapter + Gating)
#
# ┌───────────────────────────────┬──────────────────────────┐
# │ Parameter                     │ Value                    │
# ├───────────────────────────────┼──────────────────────────┤
# │ Rank (LoRA)                   │ 8                        │
# │ Prefix Length                 │ 10                       │
# │ Adapter Hidden Size           │ 64                       │
# │ Gates Initialized             │ 0.33 each                │
# │ Target Modules                │ q_proj, v_proj           │
# │ Batch Size                    │ 4                        │
# │ Epochs                        │ 5                        │
# │ Warmup Ratio                  │ 10%                      │
# │ Optimizer                     │ AdamW                    │
# │ Scheduler                     │ Linear                   │
# │ Weight Decay                  │ 0.0                      │
# │ Mixed Precision               │ bf16                     │
# └───────────────────────────────┴──────────────────────────┘

# ============================================================
# File: plap_lora_finetuning.py
# Method 7 — PLAP‑LoRA (Proposed Method)
# Date: 2026-04-22
#
# Full implementation for:
#   - PLAP = Probabilistic Layer‑wise Adaptive Patterning
#   - LoRA + Probabilistic Scaling + Pattern Masking
# Includes:
#   - PLAP‑LoRA module
#   - Dataset loader
#   - Integration into LLaMA/Mistral
#   - Full training/eval loops
#   - Reproducibility table
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
# PLAP‑LoRA Module
#
# Includes:
#   - Standard LoRA
#   - Activation Pattern Detector (variance‑based mask)
#   - Layer‑wise Probabilistic Scaling (trainable α_l)
# ============================================================
class PLAPLoRAModule(nn.Module):
    def __init__(self, hidden_size, rank=8, lambda_mask=0.7):
        super().__init__()

        self.hidden = hidden_size
        self.rank = rank
        self.lambda_mask = lambda_mask  # threshold for pattern mask

        # ---------------- LoRA ----------------
        self.lora_A = nn.Linear(hidden_size, rank, bias=False)
        self.lora_B = nn.Linear(rank, hidden_size, bias=False)
        nn.init.xavier_normal_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

        # ---------------- Probabilistic Layer Weight ----------------
        # α_l → scalar, later normalized across layers
        self.alpha_l = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, prob_scale=1.0):

        # ----- LoRA -----
        lora_out = self.lora_B(self.lora_A(x))

        # ----- Pattern Detection (variance mask) -----
        var = x.var(dim=-1, keepdim=True)                          # [B, T, 1]
        mask = (var > self.lambda_mask * var.mean()).float()       # binary mask
        lora_masked = lora_out * mask                              # masked LoRA

        # ----- Probabilistic scaling -----
        weighted = prob_scale * lora_masked

        return x + weighted  # residual


# ============================================================
# Insert PLAP‑LoRA into transformer
# Applies to q_proj, v_proj modules
# ============================================================
def apply_plap_lora(model, hidden_size, rank=8, lambda_mask=0.7):

    plap_modules = []

    for name, module in model.named_modules():
        if any(t in name for t in ["q_proj", "v_proj"]) and isinstance(module, nn.Linear):

            plap = PLAPLoRAModule(hidden_size, rank, lambda_mask)

            # override forward
            def make_forward(old_module, plap_module):
                def new_forward(x, plap_scale=1.0):
                    w = old_module.weight
                    out = F.linear(x, w, old_module.bias)
                    return plap_module(out, prob_scale=plap_scale)
                return new_forward

            module.forward = make_forward(module, plap)
            plap_modules.append(plap)

    return plap_modules


# ============================================================
# Compute normalized α_l for all layers (softmax)
# ============================================================
def compute_prob_scales(plap_modules):
    alphas = torch.stack([m.alpha_l for m in plap_modules])
    probs = torch.softmax(alphas, dim=0)
    return probs


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
# Main PLAP‑LoRA Training
# ============================================================
def main():

    accelerator = Accelerator()
    accelerator.print("Starting PLAP‑LoRA Fine‑Tuning...")

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
    # PLAP‑LoRA Configuration
    # ============================================================
    rank = 8
    lambda_mask = 0.7

    plap_modules = apply_plap_lora(
        model,
        hidden_size=hidden_size,
        rank=rank,
        lambda_mask=lambda_mask,
    )

    # Train only PLAP parameters
    trainable_params = []
    for m in plap_modules:
        trainable_params += list(m.parameters())

    # DataLoaders
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    dev_loader   = DataLoader(dev_set, batch_size=4)
    test_loader  = DataLoader(test_set, batch_size=4)

    # Optimizer
    learning_rate = 2e-4
    optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=0.0)

    num_epochs = 5
    training_steps = len(train_loader) * num_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * training_steps),
        num_training_steps=training_steps,
    )

    (
        model, optimizer, train_loader, dev_loader, test_loader, scheduler
    ) = accelerator.prepare(model, optimizer, train_loader, dev_loader, test_loader, scheduler)

    # ============================================================
    # Training Loop
    # ============================================================
    for epoch in range(num_epochs):

        model.train()
        total_loss = 0

        # compute normalized layer weights
        prob_scales = compute_prob_scales(plap_modules)  # softmax across layers
        prob_scales = prob_scales.to(accelerator.device)

        # assign each scale to module
        for i, m in enumerate(plap_modules):
            m.prob_runtime = prob_scales[i]

        for batch in tqdm(train_loader, desc=f"PLAP‑LoRA Epoch {epoch+1}"):

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
    accelerator.print(f"PLAP‑LoRA Test Accuracy = {test_acc:.4f}")

    # Save
    save_dir = "outputs/plap_lora_model"
    accelerator.unwrap_model(model).save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    accelerator.print(f"PLAP‑LoRA model saved to: {save_dir}")
    accelerator.print("PLAP‑LoRA Fine‑Tuning completed.")


if __name__ == "__main__":
    main()


# ============================================================
# Reproducibility Table (inside code)
# ============================================================
# Backbone: LLaMA‑3‑8B / Mistral‑7B
# Method: PLAP‑LoRA (Proposed)
#
# ┌───────────────────────────────┬──────────────────────────┐
# │ Parameter                     │ Value                    │
# ├───────────────────────────────┼──────────────────────────┤
# │ Rank                          │ 8                        │
# │ Pattern Mask λ                │ 0.7                      │
# │ Probabilistic α_l Init        │ 1.0 (softmax normalized) │
# │ Target Modules                │ q_proj, v_proj           │
# │ Batch Size                    │ 4                        │
# │ Epochs                        │ 5                        │
# │ Warmup                        │ 10%                      │
# │ Optimizer                     │ AdamW                    │
# │ Scheduler                     │ Linear                   │
# │ Weight Decay                  │ 0.0                      │
# │ Mixed Precision               │ bf16                     │

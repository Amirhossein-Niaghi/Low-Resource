# ============================================================
# File: oft_finetuning.py
# Method 5 — OFT (Orthogonal Fine‑Tuning)
# Date: 2026-04-22
# Full implementation for:
#   - Orthogonal Fine‑Tuning (OFT)
#   - LLaMA‑3‑8B / Mistral‑7B
#   - Persian Intent Classification
#
# Notes:
#   - OFT is implemented by learning a low‑rank matrix ΔW = B A
#     where B is constrained to be orthogonal using a penalty term.
#
# Includes:
#   - Dataset loader
#   - Custom OFT module
#   - Injection into transformer
#   - Training loop
#   - Evaluation
#   - Saving outputs
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
    get_linear_schedule_with_warmup
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
            return_tensors="pt"
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        encoded["labels"] = torch.tensor(self.label2id[s["label"]])
        return encoded


# ============================================================
# OFT Module (Low‑Rank ΔW = B A)
# ============================================================
class OFTModule(nn.Module):
    def __init__(self, weight_shape, rank=8, orth_weight=0.5):
        super().__init__()
        self.out_dim, self.in_dim = weight_shape
        self.rank = rank
        self.orth_weight = orth_weight

        # Learned matrices
        self.A = nn.Parameter(torch.zeros(rank, self.in_dim))
        self.B = nn.Parameter(torch.zeros(self.out_dim, rank))

        # Initialization
        nn.init.xavier_normal_(self.A)
        nn.init.orthogonal_(self.B)

    def forward(self, base_w):
        delta_w = self.B @ self.A
        return base_w + delta_w

    def orthogonal_loss(self):
        # ||B^T B - I||^2
        BTB = self.B.T @ self.B
        I = torch.eye(self.rank, device=self.B.device)
        return self.orth_weight * F.mse_loss(BTB, I)


# ============================================================
# Inject OFT modules into transformer layers
# ============================================================
def apply_oft_to_linear_layers(model, rank=8, orth_weight=0.5, target_modules=["q_proj", "v_proj"]):
    oft_modules = []

    for name, module in model.named_modules():
        if any(t in name for t in target_modules) and isinstance(module, nn.Linear):

            oft = OFTModule(
                weight_shape=module.weight.shape,
                rank=rank,
                orth_weight=orth_weight
            )

            # Replace forward pass
            def make_forward(m, oft_module):
                def new_forward(x):
                    w = oft_module(module.weight)
                    return F.linear(x, w, module.bias)
                return new_forward

            module.forward = make_forward(module, oft)
            oft_modules.append(oft)

    return oft_modules


# ============================================================
# Evaluation
# ============================================================
def evaluate(model, dataloader, accelerator, oft_modules):
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

    # Orthogonality penalty excluded for eval
    return correct / total


# ============================================================
# Main OFT Training
# ============================================================
def main():

    accelerator = Accelerator()
    accelerator.print("Starting OFT Fine‑Tuning...")

    # -----------------------------
    # Model & Tokenizer
    # -----------------------------
    model_name = "meta-llama/Llama-3-8b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # -----------------------------
    # Dataset
    # -----------------------------
    train_set = IntentDataset("data/intent_train.json", tokenizer)
    dev_set   = IntentDataset("data/intent_dev.json", tokenizer)
    test_set  = IntentDataset("data/intent_test.json", tokenizer)

    # -----------------------------
    # Base model
    # -----------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(train_set.labels)
    )

    # ============================================================
    # OFT CONFIGURATION
    # ============================================================
    rank = 8
    orth_weight = 0.5
    target_modules = ["q_proj", "v_proj"]

    oft_modules = apply_oft_to_linear_layers(
        model,
        rank=rank,
        orth_weight=orth_weight,
        target_modules=target_modules
    )

    # Only OFT parameters will be trained
    oft_params = []
    for oft in oft_modules:
        oft_params += list(oft.parameters())

    # -----------------------------
    # DataLoader
    # -----------------------------
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    dev_loader   = DataLoader(dev_set, batch_size=4)
    test_loader  = DataLoader(test_set, batch_size=4)

    # -----------------------------
    # Optimizer
    # -----------------------------
    learning_rate = 2e-4
    optimizer = AdamW(oft_params, lr=learning_rate, weight_decay=0.0)

    num_epochs = 5
    training_steps = len(train_loader) * num_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(training_steps * 0.1),
        num_training_steps=training_steps
    )

    # prepare with accelerator
    (model, optimizer, train_loader, dev_loader, test_loader, scheduler) = accelerator.prepare(
        model, optimizer, train_loader, dev_loader, test_loader, scheduler
    )

    # ============================================================
    # Training Loop
    # ============================================================
    for epoch in range(num_epochs):

        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"OFT Epoch {epoch+1}"):

            # Forward
            outputs = model(**batch)
            ce_loss = outputs.loss

            # Orthogonality penalty
            orth_loss = sum([m.orthogonal_loss() for m in oft_modules])

            loss = ce_loss + orth_loss
            total_loss += loss.item()

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        dev_acc = evaluate(model, dev_loader, accelerator, oft_modules)

        accelerator.print(
            f"Epoch {epoch+1} | Loss={total_loss:.4f} | Dev Acc={dev_acc:.4f}"
        )

    # -----------------------------
    # Final Test
    # -----------------------------
    test_acc = evaluate(model, test_loader, accelerator, oft_modules)
    accelerator.print(f"OFT Test Accuracy = {test_acc:.4f}")

    # -----------------------------
    # Save Model
    # -----------------------------
    accelerator.wait_for_everyone()
    save_dir = "outputs/oft_model"
    accelerator.unwrap_model(model).save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    accelerator.print(f"OFT weights saved to {save_dir}")
    accelerator.print("OFT Fine‑Tuning Finished.")


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    main()


# ============================================================
# Reproducibility Table (inside code)
# ============================================================
# Backbone: LLaMA‑3‑8B / Mistral‑7B
# Method: OFT (Orthogonal Fine‑Tuning)
#
# ┌───────────────────────────────┬──────────────────────────┐
# │ Parameter                     │ Value                    │
# ├───────────────────────────────┼──────────────────────────┤
# │ Rank r                        │ 8                        │
# │ Orthogonality Weight          │ 0.5                      │
# │ Target Modules                │ q_proj, v_proj           │
# │ Batch Size                    │ 4                        │
# │ Epochs                        │ 5                        │
# │ Warmup Ratio                  │ 10%                      │
# │ Optimizer                     │ AdamW                    │
# │ Scheduler                     │ Linear                   │
# │ Weight Decay                  │ 0.0                      │
# │ Mixed Precision               │ bf16                     │
# └───────────────────────────────┴──────────────────────────┘

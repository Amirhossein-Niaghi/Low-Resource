# ============================================================
# File: dora_finetuning.py
# Method 4 — DoRA (Decoupled Low‑Rank Adaptation)
# Date: 2026-04-22
#
# Full implementation for:
#   - DoRA PEFT fine‑tuning
#   - LLaMA‑3‑8B / Mistral‑7B
#   - Persian Intent Classification
#
# Includes:
#   - Dataset loader
#   - DoRA configuration
#   - Training loop
#   - Evaluation
#   - Adapter saving
#   - Reproducibility table (inside code)
# ============================================================

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model
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
        sample = self.samples[idx]

        encoded = self.tokenizer(
            sample["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        encoded["labels"] = torch.tensor(self.label2id[sample["label"]])

        return encoded


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
# Main DoRA Training
# ============================================================
def main():

    accelerator = Accelerator()
    accelerator.print("Starting DoRA Fine‑Tuning...")

    model_name = "meta-llama/Llama-3-8b"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False
    )

    # -----------------------------
    # Dataset
    # -----------------------------
    train_set = IntentDataset("data/intent_train.json", tokenizer)
    dev_set   = IntentDataset("data/intent_dev.json", tokenizer)
    test_set  = IntentDataset("data/intent_test.json", tokenizer)

    # -----------------------------
    # Load Model
    # -----------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(train_set.labels)
    )

    # ============================================================
    # DoRA CONFIGURATION
    # ============================================================
    # DoRA extends LoRA by decomposing weight magnitude and direction
    dora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        use_dora=True,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="SEQ_CLS"
    )

    model = get_peft_model(model, dora_config)

    accelerator.print(model.print_trainable_parameters())

    # -----------------------------
    # DataLoaders
    # -----------------------------
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    dev_loader   = DataLoader(dev_set, batch_size=4)
    test_loader  = DataLoader(test_set, batch_size=4)

    # -----------------------------
    # Optimizer
    # -----------------------------
    learning_rate = 2e-4

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0
    )

    num_epochs = 5

    training_steps = len(train_loader) * num_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(training_steps * 0.1),
        num_training_steps=training_steps
    )

    # -----------------------------
    # Accelerator preparation
    # -----------------------------
    (
        model,
        optimizer,
        train_loader,
        dev_loader,
        test_loader,
        scheduler
    ) = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        dev_loader,
        test_loader,
        scheduler
    )

    # ============================================================
    # Training Loop
    # ============================================================
    for epoch in range(num_epochs):

        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"DoRA Epoch {epoch+1}"):

            outputs = model(**batch)

            loss = outputs.loss
            total_loss += loss.item()

            accelerator.backward(loss)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        dev_acc = evaluate(model, dev_loader, accelerator)

        accelerator.print(
            f"Epoch {epoch+1} | Loss={total_loss:.4f} | Dev Accuracy={dev_acc:.4f}"
        )

    # -----------------------------
    # Final Test Evaluation
    # -----------------------------
    test_acc = evaluate(model, test_loader, accelerator)

    accelerator.print(f"DoRA Test Accuracy = {test_acc:.4f}")

    # -----------------------------
    # Save Adapter
    # -----------------------------
    accelerator.wait_for_everyone()

    save_dir = "outputs/dora_model"

    accelerator.unwrap_model(model).save_pretrained(save_dir)

    tokenizer.save_pretrained(save_dir)

    accelerator.print(f"DoRA adapter saved to {save_dir}")
    accelerator.print("DoRA training completed successfully.")


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    main()


# ============================================================
# Reproducibility Table (inside code)
# ============================================================
# Backbone Model: LLaMA‑3‑8B / Mistral‑7B
# Method: DoRA (Decoupled Low‑Rank Adaptation)
#
# ┌───────────────────────────────┬──────────────────────────┐
# │ Parameter                     │ Value                    │
# ├───────────────────────────────┼──────────────────────────┤
# │ Rank r                        │ 16                       │
# │ LoRA Alpha                    │ 32                       │
# │ LoRA Dropout                  │ 0.05                     │
# │ DoRA Enabled                  │ True                     │
# │ Target Modules                │ q_proj, v_proj           │
# │ Batch Size                    │ 4                        │
# │ Epochs                        │ 5                        │
# │ Warmup Ratio                  │ 10%                      │
# │ Optimizer                     │ AdamW                    │
# │ Scheduler                     │ Linear                   │
# │ Weight Decay                  │ 0.0                      │
# │ Mixed Precision               │ bf16                     │
# └───────────────────────────────┴──────────────────────────┘

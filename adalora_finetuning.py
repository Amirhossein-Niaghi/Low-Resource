# ============================================================
# File: adalora_finetuning.py
# Method 3 — AdaLoRA (Adaptive Low‑Rank Adaptation)
# Date: 2026-04-22
# Full implementation for:
#   - AdaLoRA PEFT training
#   - LLaMA‑3‑8B / Mistral‑7B
#   - Persian Intent Classification
#
# Includes:
#   - Dataset loader
#   - AdaLoRA config
#   - Training loop
#   - Evaluation
#   - Saving adapter
#   - Reproducibility table (inside code)
# ============================================================

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    AdamW,
)
from peft import AdaLoraConfig, get_peft_model
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
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        encoded["labels"] = torch.tensor(self.label2id[s["label"]])
        return encoded


# ============================================================
# Evaluation
# ============================================================
def evaluate(model, dataloader, accelerator):
    model.eval()
    correct, total = 0, 0
    for batch in dataloader:
        with torch.no_grad():
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


# ============================================================
# MAIN AdaLoRA Training
# ============================================================
def main():
    accelerator = Accelerator()
    accelerator.print("Starting AdaLoRA Fine‑Tuning...")

    # Choose model
    model_name = "meta-llama/Llama-3-8b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # Dataset
    train_set = IntentDataset("data/intent_train.json", tokenizer)
    dev_set   = IntentDataset("data/intent_dev.json", tokenizer)
    test_set  = IntentDataset("data/intent_test.json", tokenizer)

    # Load pretrained model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(train_set.labels)
    )

    # ============================================================
    # AdaLoRA CONFIGURATION
    # ============================================================
    adalora_config = AdaLoraConfig(
        init_r=12,             # initial rank
        target_r=8,            # final target rank after pruning
        lora_alpha=32,
        lora_dropout=0.05,
        beta1=0.85,
        beta2=0.85,
        tinit=200,             # iterations before rank adaptation starts
        tfinal=1000,           # when target_r is reached
        deltaT=50,             # adaptation frequency
        orth_reg_weight=0.5,
        target_modules=["q_proj", "v_proj"],
        task_type="SEQ_CLS"
    )

    # Apply AdaLoRA adapter
    model = get_peft_model(model, adalora_config)
    accelerator.print(model.print_trainable_parameters())

    # Dataloaders
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    dev_loader   = DataLoader(dev_set, batch_size=4)
    test_loader  = DataLoader(test_set, batch_size=4)

    # Optimizer & scheduler
    learning_rate = 2e-4
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)

    num_epochs = 5
    training_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(training_steps * 0.1),
        num_training_steps=training_steps
    )

    # Prepare for distributed training
    (
        model,
        optimizer,
        train_loader,
        dev_loader,
        test_loader,
        scheduler
    ) = accelerator.prepare(model, optimizer, train_loader, dev_loader, test_loader, scheduler)

    # ============================================================
    # Training Loop
    # ============================================================
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"AdaLoRA Epoch {epoch+1}"):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        dev_acc = evaluate(model, dev_loader, accelerator)
        accelerator.print(f"Epoch {epoch+1} | Loss={total_loss:.4f} | Dev Acc={dev_acc:.4f}")

    # Final test evaluation
    test_acc = evaluate(model, test_loader, accelerator)
    accelerator.print(f"AdaLoRA Test Accuracy = {test_acc:.4f}")

    # Save adapter
    accelerator.wait_for_everyone()
    save_dir = "outputs/adalora_model"
    accelerator.unwrap_model(model).save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    accelerator.print(f"AdaLoRA adapter saved in {save_dir}")
    accelerator.print("AdaLoRA Fine‑Tuning completed.")


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    main()


# ============================================================
# Reproducibility Table for AdaLoRA  (inside code)
# ============================================================
# Backbone Model: LLaMA‑3‑8B / Mistral‑7B
# Method: AdaLoRA
#
# ┌───────────────────────────────┬───────────────────────────┐
# │ Parameter                     │ Value                     │
# ├───────────────────────────────┼───────────────────────────┤
# │ init_r                        │ 12                        │
# │ target_r                      │ 8                         │
# │ lora_alpha                    │ 32                        │
# │ lora_dropout                  │ 0.05                      │
# │ tinit                         │ 200                       │
# │ tfinal                        │ 1000                      │
# │ deltaT                        │ 50                        │
# │ beta1/beta2                   │ 0.85 / 0.85               │
# │ Orthogonality Weight          │ 0.5                       │
# │ Target Modules                │ q_proj, v_proj            │
# │ Learning Rate                 │ 2e‑4                      │
# │ Batch Size                    │ 4                         │
# │ Epochs                        │ 5                         │
# │ Warmup Ratio                  │ 10%                       │
# │ Weight Decay                  │ 0.0                       │
# │ Optimizer                     │ AdamW                     │
# │ Scheduler                     │ Linear                    │
# │ Mixed Precision               │ bf16                      │
# │ GPUs                          │ 1×A100‑40GB               │
# └───────────────────────────────┴───────────────────────────┘

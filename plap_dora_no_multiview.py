#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
plap_dora_no_multiview.py

PLAP–DoRA without Multi-View Consistency (MVC Ablation).
Active PLAP components:
- Semantic Alignment (MSE + cosine)
- Drift Suppression (cosine, batch-wise)

Disabled:
- Multi-View Consistency regularization

Everything else (training loop, DoRA replacement, evaluation, scheduler, seed,
optimizer, logging) remains identical to the official Full PLAP–DoRA script.
"""

import os
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

from accelerate import Accelerator
from accelerate.utils import set_seed

import pandas as pd


# ============================================================
# REPRODUCIBILITY TABLE (PLAP–DoRA WITHOUT MVC)
# ============================================================
"""
Hyperparameter / Setting           Value / Description
---------------------------------------------------------------
Backbone model                     e.g., "meta-llama/Meta-Llama-3-8B-Instruct"
Task                               Persian intent classification
Dataset file                       Persian_Intent_Unified_utf8sig.csv
Text column                        "text"
Label column                       "label"
Max sequence length                128
Batch size (train)                 8
Batch size (eval)                  8
Learning rate                      2e-5
Optimizer                          
mW (betas=(0.9,0.98), eps=1e-8)
Weight decay                       0.01
Scheduler                          linear warmup
Warmup ratio                       0.06
Epochs                             5
Precision                          bf16 if supported
Seed                               42
DoRA parameterization              All linear layers (unless overridden)
PLAP active components             Semantic + Drift
PLAP inactive components           Multi-View Consistency (MVC)
Semantic loss                      MSE + cosine
Drift loss                         1 - cosine(h_new, h_pre)
Loss weights                       lambda_sem = 0.6
                                   lambda_drift = 0.2
MVC weight                         Not used
Evaluation metrics                 Accuracy, Macro-F1
"""


# ============================================================
# DATASET + COLLATE
# ============================================================

class IntentDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        label2id: Optional[Dict[str, int]] = None,
        id2label: Optional[Dict[int, str]] = None,
    ):
        df = pd.read_csv(csv_path)

        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must contain 'text' and 'label' columns.")

        self.texts = df["text"].astype(str).tolist()
        raw_labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Build mapping
        if isinstance(raw_labels[0], str):
            if label2id is None:
                uniq = sorted(list(set(raw_labels)))
                label2id = {lab: i for i, lab in enumerate(uniq)}
                id2label = {i: lab for lab, i in label2id.items()}
        else:
            if label2id is None:
                uniq = sorted(list(set(raw_labels)))
                label2id = {str(l): l for l in uniq}
                id2label = {l: str(l) for l in uniq}

        self.label2id = label2id
        self.id2label = id2label

        # numeric labels
        self.labels = []
        for x in raw_labels:
            if isinstance(x, str):
                self.labels.append(self.label2id[x])
            else:
                self.labels.append(int(x))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": self.labels[idx]}


def collate_fn(batch, tokenizer: AutoTokenizer, max_length: int, with_aug_views=True):
    texts = [b["text"] for b in batch]
    labels = [b["label"] for b in batch]

    enc_clean = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )

    if not with_aug_views:
        return {
            "input_ids": enc_clean["input_ids"],
            "attention_mask": enc_clean["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    # Paraphrase view (pseudo)
    para_texts = []
    for t in texts:
        toks = t.split()
        if len(toks) > 5:
            mid = toks[1:-1]
            random.shuffle(mid)
            para_texts.append(" ".join([toks[0]] + mid + [toks[-1]]))
        else:
            para_texts.append(t)

    enc_para = tokenizer(
        para_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )

    return {
        "input_ids_clean": enc_clean["input_ids"],
        "attention_mask_clean": enc_clean["attention_mask"],
        "input_ids_para": enc_para["input_ids"],
        "attention_mask_para": enc_para["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# ============================================================
# DoRA Modules
# ============================================================

class DoRALinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.u = nn.Parameter(torch.empty(out_features, in_features))
        self.a = nn.Parameter(torch.ones(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.u, a=math.sqrt(5))

    def forward(self, x):
        u_norm = self.u / (self.u.norm(dim=-1, keepdim=True) + 1e-8)
        W = self.a.unsqueeze(-1) * u_norm
        return nn.functional.linear(x, W, self.bias)


def replace_linears_with_dora(model: nn.Module, target_modules=None):
    for name, module in list(model.named_children()):
        replace_linears_with_dora(module, target_modules)
        if isinstance(module, nn.Linear):
            if target_modules is not None:
                if not any(t in name for t in target_modules):
                    continue
            dora_layer = DoRALinear(
                module.in_features, module.out_features, bias=(module.bias is not None)
            )
            with torch.no_grad():
                dora_layer.u.copy_(module.weight.data)
                if module.bias is not None:
                    dora_layer.bias.copy_(module.bias.data)
            setattr(model, name, dora_layer)


# ============================================================
# PLAP LOSS FUNCTIONS (MVC REMOVED)
# ============================================================

def cosine_distance(a, b, eps=1e-8):
    a_norm = a / (a.norm(dim=-1, keepdim=True) + eps)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + eps)
    return 1.0 - (a_norm * b_norm).sum(dim=-1).mean()


def semantic_alignment_loss(h_clean, h_para):
    mse = nn.functional.mse_loss(h_clean, h_para)
    cdist = cosine_distance(h_clean, h_para)
    return mse + cdist


def drift_suppression_loss(h_new, h_pre):
    return cosine_distance(h_new, h_pre)


# ============================================================
# Model
# ============================================================

class PLAPDoRA_NoMVC(nn.Module):
    """
    PLAP–DoRA model WITHOUT Multi-View Consistency.
    Only Semantic Alignment + Drift Suppression are active.
    """

    def __init__(
        self,
        base_model_name: str,
        num_labels: int,
        lambda_sem: float = 0.6,
        lambda_drift: float = 0.2,
        dora_target_modules=None,
    ):
        super().__init__()
        self.lambda_sem = lambda_sem
        self.lambda_drift = lambda_drift

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        cfg = self.base_model.config
        hidden = cfg.hidden_size
        self.num_labels = num_labels

        replace_linears_with_dora(self.base_model, dora_target_modules)

        self.classifier = nn.Linear(hidden, num_labels)

        # frozen pre-trained encoder for drift
        self.pretrained_encoder = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        self.pretrained_encoder.eval()
        for p in self.pretrained_encoder.parameters():
            p.requires_grad = False

    def _bos_repr(self, input_ids, attn_mask, model=None):
        if model is None:
            model = self.base_model
        out = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = out.hidden_states[-1]
        return hs[:, 0, :]

    def forward(self, batch, compute_plap_losses=True):
        h_clean = self._bos_repr(
            batch["input_ids_clean"],
            batch["attention_mask_clean"],
            model=self.base_model,
        )

        logits = self.classifier(h_clean)
        cls_loss = nn.CrossEntropyLoss()(logits, batch["labels"])

        total_loss = cls_loss
        log_vals = {
            "cls_loss": cls_loss.detach(),
            "semantic_loss": torch.tensor(0.0, device=logits.device),
            "drift_loss": torch.tensor(0.0, device=logits.device),
        }

        if compute_plap_losses:
            # SEMANTIC ALIGNMENT
            h_para = self._bos_repr(
                batch["input_ids_para"],
                batch["attention_mask_para"],
                model=self.base_model,
            )
            sem_loss = semantic_alignment_loss(h_clean, h_para)
            total_loss += self.lambda_sem * sem_loss
            log_vals["semantic_loss"] = sem_loss.detach()

            # DRIFT SUPPRESSION
            h_pre = self._bos_repr(
                batch["input_ids_clean"],
                batch["attention_mask_clean"],
                model=self.pretrained_encoder,
            ).detach()
            drift_loss = drift_suppression_loss(h_clean, h_pre)
            total_loss += self.lambda_drift * drift_loss
            log_vals["drift_loss"] = drift_loss.detach()

        log_vals["loss"] = total_loss
        log_vals["logits"] = logits
        return log_vals


# ============================================================
# Evaluation
# ============================================================

def compute_accuracy(preds, labels):
    return sum(p == l for p, l in zip(preds, labels)) / len(labels)


def compute_macro_f1(preds, labels, num_labels):
    eps = 1e-8
    f1s = []
    for c in range(num_labels):
        tp = sum(p == c and l == c for p, l in zip(preds, labels))
        fp = sum(p == c and l != c for p, l in zip(preds, labels))
        fn = sum(p != c and l == c for p, l in zip(preds, labels))
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        f1s.append(f1)
    return sum(f1s) / num_labels


def evaluate(model, loader, accelerator, num_labels):
    model.eval()
    preds, gold = [], []
    tot_loss = 0
    steps = 0

    with torch.no_grad():
        for batch in loader:
            for k in batch:
                batch[k] = batch[k].to(accelerator.device)

            out = model(batch, compute_plap_losses=False)
            loss = out["loss"]
            logits = out["logits"]

            preds.extend(logits.argmax(-1).cpu().tolist())
            gold.extend(batch["labels"].cpu().tolist())
            tot_loss += loss.item()
            steps += 1

    return {
        "eval_loss": tot_loss / steps,
        "eval_accuracy": compute_accuracy(preds, gold),
        "eval_macro_f1": compute_macro_f1(preds, gold, num_labels),
    }


# ============================================================
# Training
# ============================================================

def parse_args():
    p = argparse.ArgumentParser("PLAP–DoRA Without MVC")

    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--train_file", type=str, default="Persian_Intent_Unified_utf8sig.csv")
    p.add_argument("--eval_file", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./outputs/plap_dora_no_mvc")

    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--train_batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_train_epochs", type=int, default=5)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--lambda_sem", type=float, default=0.6)
    p.add_argument("--lambda_drift", type=float, default=0.2)

    p.add_argument("--dora_target_modules", type=str, nargs="*", default=None)
    p.add_argument("--split_ratio", type=float, default=0.9)

    return p.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision="bf16")
    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    full_dataset = IntentDataset(args.train_file, tokenizer, args.max_length)
    num_labels = len(full_dataset.label2id)

    # Train/eval split
    if args.eval_file:
        eval_dataset = IntentDataset(
            args.eval_file,
            tokenizer,
            args.max_length,
            label2id=full_dataset.label2id,
            id2label=full_dataset.id2label,
        )
        train_dataset = full_dataset
    else:
        idxs = list(range(len(full_dataset)))
        random.shuffle(idxs)
        split = int(args.split_ratio * len(idxs))
        train_idx = idxs[:split]
        eval_idx = idxs[split:]
        train_dataset = [full_dataset[i] for i in train_idx]
        eval_dataset = [full_dataset[i] for i in eval_idx]

        class Wrap(Dataset):
            def __init__(self, data): self.data = data
            def __len__(self): return len(self.data)
            def __getitem__(self, i): return self.data[i]

        train_dataset = Wrap(train_dataset)
        eval_dataset = Wrap(eval_dataset)

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length, with_aug_views=True),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length, with_aug_views=True),
    )

    # Model
    model = PLAPDoRA_NoMVC(
        base_model_name=args.model_name_or_path,
        num_labels=num_labels,
        lambda_sem=args.lambda_sem,
        lambda_drift=args.lambda_drift,
        dora_target_modules=args.dora_target_modules,
    )

    # Optimizer + Scheduler
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "ln_f.weight"]
    grouped = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(grouped, lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-8)

    total_steps = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.num_train_epochs
    warmup = int(args.warmup_ratio * total_steps)

    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)

    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )

    # Training
    best_f1 = 0
    global_step = 0

    for epoch in range(args.num_train_epochs):
        model.train()
        tot_loss = tot_sem = tot_drift = tot_cls = 0
        steps = 0

        for step, batch in enumerate(train_loader):
            for k in batch:
                batch[k] = batch[k].to(accelerator.device)

            with accelerator.accumulate(model):
                out = model(batch, compute_plap_losses=True)
                loss = out["loss"]

                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                steps += 1

                tot_loss += loss.item()
                tot_cls += out["cls_loss"].item()
                tot_sem += out["semantic_loss"].item()
                tot_drift += out["drift_loss"].item()

            if accelerator.is_main_process and step % 50 == 0:
                print(
                    f"[epoch {epoch+1} step {step}] "
                    f"loss={tot_loss/steps:.4f}, cls={tot_cls/steps:.4f}, "
                    f"sem={tot_sem/steps:.4f}, drift={tot_drift/steps:.4f}"
                )

        # Eval
        metrics = evaluate(model, eval_loader, accelerator, num_labels)
        if accelerator.is_main_process:
            print(
                f"--- Epoch {epoch+1} Eval ---\n"
                f"loss={metrics['eval_loss']:.4f}\n"
                f"acc={metrics['eval_accuracy']:.4f}\n"
                f"macro_f1={metrics['eval_macro_f1']:.4f}"
            )
            if metrics["eval_macro_f1"] > best_f1:
                best_f1 = metrics["eval_macro_f1"]
                unwrapped = accelerator.unwrap_model(model)
                save_path = os.path.join(args.output_dir, "best_model")
                os.makedirs(save_path, exist_ok=True)
                unwrapped.base_model.save_pretrained(save_path)
                torch.save(unwrapped.state_dict(), os.path.join(save_path, "plap_dora_no_mvc.pt"))
                tokenizer.save_pretrained(save_path)
                print(f"Saved best model to {save_path}")

    if accelerator.is_main_process:
        print("Training complete (No MVC).")


if __name__ == "__main__":
    main()

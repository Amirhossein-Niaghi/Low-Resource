#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
plap_dora_no_semantic_alignment.py

PLAP–DoRA without Semantic Alignment (Ablation).
Active PLAP components:
- Multi-View Consistency (MVC)
- Drift Suppression

Disabled:
- Semantic Alignment

Everything else (training loop, DoRA parameterization, optimizer, scheduler,
dataset, evaluation) remains identical to the full PLAP–DoRA implementation.
"""

import os
import math
import random
import argparse
from typing import Optional, Dict
import torch
import torch.nn as nn
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from accelerate import Accelerator
from accelerate.utils import set_seed

# ============================================================
# REPRODUCIBILITY TABLE — PLAP–DoRA WITHOUT SEMANTIC ALIGNMENT
# ============================================================
"""
Hyperparameter / Setting           Value / Description
---------------------------------------------------------------
Backbone                           LLaMA3-8B / Mistral-7B
Task                               Persian Intent Classification
Dataset                            Persian_Intent_Unified_utf8sig.csv
Text column                        "text"
Label column                       "label"
Sequence length                    128
Batch size                         8
Learning rate                      2e-5
Optimizer                          AdamW
Weight decay                       0.01
Epochs                             5
Scheduler                          linear warmup
Warmup ratio                       0.06
Precision                          bf16
Seed                               42
DoRA Target                        Linear layers
Active PLAP components             MVC + Drift
Inactive PLAP components           Semantic Alignment
Lambda_mvc                         0.4
Lambda_drift                       0.2
"""


# ============================================================
# DATASET + COLLATE
# ============================================================

class IntentDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer: AutoTokenizer,
                 max_length: int = 128,
                 label2id: Optional[Dict[str, int]] = None,
                 id2label: Optional[Dict[int, str]] = None):

        df = pd.read_csv(csv_path)
        self.texts = df["text"].astype(str).tolist()
        raw_labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

        if isinstance(raw_labels[0], str):
            if label2id is None:
                uniq = sorted(set(raw_labels))
                label2id = {lab: i for i, lab in enumerate(uniq)}
                id2label = {i: lab for lab, i in label2id.items()}
        else:
            if label2id is None:
                uniq = sorted(set(raw_labels))
                label2id = {str(x): x for x in uniq}
                id2label = {x: str(x) for x in uniq}

        self.labels = [label2id[x] if isinstance(x, str) else int(x) for x in raw_labels]
        self.label2id = label2id
        self.id2label = id2label

    def __len__(self): return len(self.texts)
    def __getitem__(self, idx): return {"text": self.texts[idx], "label": self.labels[idx]}


def collate_fn(batch, tokenizer, max_length, with_aug_views=True):
    texts = [b["text"] for b in batch]
    labels = [b["label"] for b in batch]

    enc_clean = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    if not with_aug_views:
        return {
            "input_ids": enc_clean["input_ids"],
            "attention_mask": enc_clean["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    # Simple paraphrase-like augmentation
    para_texts = []
    for t in texts:
        toks = t.split()
        if len(toks) > 5:
            mid = toks[1:-1]
            random.shuffle(mid)
            para_texts.append(" ".join([toks[0]] + mid + [toks[-1]]))
        else:
            para_texts.append(t)

    enc_para = tokenizer(para_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    return {
        "input_ids_clean": enc_clean["input_ids"],
        "attention_mask_clean": enc_clean["attention_mask"],
        "input_ids_para": enc_para["input_ids"],
        "attention_mask_para": enc_para["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# ============================================================
# DoRA MODULE
# ============================================================

class DoRALinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.u = nn.Parameter(torch.empty(out_features, in_features))
        self.a = nn.Parameter(torch.ones(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.u, a=math.sqrt(5))

    def forward(self, x):
        u_norm = self.u / (self.u.norm(dim=-1, keepdim=True) + 1e-8)
        W = self.a.unsqueeze(-1) * u_norm
        return nn.functional.linear(x, W, self.bias)


def replace_linears_with_dora(model, target_modules=None):
    for name, module in list(model.named_children()):
        replace_linears_with_dora(module, target_modules)
        if isinstance(module, nn.Linear):
            if target_modules and not any(t in name for t in target_modules):
                continue
            new_layer = DoRALinear(module.in_features, module.out_features, bias=module.bias is not None)
            with torch.no_grad():
                new_layer.u.copy_(module.weight.data)
                if module.bias is not None:
                    new_layer.bias.copy_(module.bias.data)
            setattr(model, name, new_layer)


# ============================================================
# PLAP LOSS FUNCTIONS (NO SEMANTIC)
# ============================================================

def cosine_distance(a, b, eps=1e-8):
    a_norm = a / (a.norm(dim=-1, keepdim=True) + eps)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + eps)
    return 1 - (a_norm * b_norm).sum(dim=-1).mean()


def mvc_loss(h_clean, h_para):
    return cosine_distance(h_clean, h_para)


def drift_loss(h_new, h_pre):
    return cosine_distance(h_new, h_pre)


# ============================================================
# MODEL
# ============================================================

class PLAPDoRA_NoSemantic(nn.Module):
    def __init__(self, base_model_name, num_labels,
                 lambda_mvc=0.4,
                 lambda_drift=0.2,
                 dora_target_modules=None):
        super().__init__()

        self.lambda_mvc = lambda_mvc
        self.lambda_drift = lambda_drift

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        hidden = self.base_model.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)

        replace_linears_with_dora(self.base_model, dora_target_modules)

        self.pretrained_encoder = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        self.pretrained_encoder.eval()
        for p in self.pretrained_encoder.parameters():
            p.requires_grad = False

    def _bos(self, ids, mask, model=None):
        if model is None:
            model = self.base_model
        out = model(ids, attention_mask=mask, output_hidden_states=True, return_dict=True)
        return out.hidden_states[-1][:, 0, :]

    def forward(self, batch, compute_plap=True):
        h_clean = self._bos(batch["input_ids_clean"], batch["attention_mask_clean"])
        logits = self.classifier(h_clean)
        cls_loss = nn.CrossEntropyLoss()(logits, batch["labels"])

        total = cls_loss
        logs = {"cls_loss": cls_loss.detach(),
                "mvc_loss": torch.tensor(0.0, device=logits.device),
                "drift_loss": torch.tensor(0.0, device=logits.device)}

        if compute_plap:

            # MVC
            h_para = self._bos(batch["input_ids_para"], batch["attention_mask_para"])
            L_mvc = mvc_loss(h_clean, h_para)
            total += self.lambda_mvc * L_mvc
            logs["mvc_loss"] = L_mvc.detach()

            # Drift
            h_pre = self._bos(batch["input_ids_clean"], batch["attention_mask_clean"],
                              model=self.pretrained_encoder).detach()
            L_drift = drift_loss(h_clean, h_pre)
            total += self.lambda_drift * L_drift
            logs["drift_loss"] = L_drift.detach()

        logs["loss"] = total
        logs["logits"] = logits
        return logs

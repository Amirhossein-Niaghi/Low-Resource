#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
full_plap_dora.py

Full PLAP–DoRA finetuning for Persian intent classification.

Implements:
- Standard supervised classification loss (cross-entropy)
- Multi-View Consistency regularization (cosine distance)
- Semantic Alignment constraints (MSE + cosine distance)
- Representation-Drift Suppression (cosine distance, batch-wise)

Backbone:
- Any causal LLM from HuggingFace (e.g., LLaMA, Mistral) used as encoder of the [BOS] token.

DoRA-style parameterization:
- Weight W is decomposed into direction u (unit vector) and magnitude a (scalar), W = a * normalize(u).

Reproducibility:
- See "REPRODUCIBILITY TABLE" below and the CLI arguments.
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


# ============================================================
# REPRODUCIBILITY TABLE (PLAP–DoRA, Full Version)
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
Optimizer                          AdamW (betas=(0.9, 0.98), eps=1e-8)
Weight decay                       0.01
Scheduler                          linear warmup
Warmup ratio                       0.06
Epochs                             5
Precision                          bf16 (if supported), otherwise fp16/fp32
Gradient clipping                  1.0
Seed                               42
Validation frequency               every epoch (and/or via eval steps)
Gradient accumulation steps        1 (can be increased)
DoRA parameterization              Decompose linear weights into magnitude & direction
PLAP enabled components            MVC, Semantic, Drift (all active)
MVC loss                           L_mvc = 1 - cosine(h_clean, h_noisy)
Semantic loss                      L_sem = MSE(h_clean, h_para) + (1 - cosine(h_clean, h_para))
Drift loss                         L_drift = 1 - cosine(h_new, h_pre) over full batch
Loss weights                       lambda_mvc = 0.4
                                   lambda_sem = 0.6
                                   lambda_drift = 0.2
Representation point               pooled BOS hidden state
Tokenizer padding side             "right"
Tokenizer truncation               "longest_first"
Label smoothing                    None (pure cross-entropy)
Evaluation metrics                 Accuracy, macro F1
Model save directory               ./outputs/full_plap_dora/
"""


# ============================================================
# Utility: Simple dataset for intent classification
# ============================================================

import pandas as pd


class IntentDataset(Dataset):
    """
    A simple dataset for Persian intent classification.

    Expects a CSV file with at least two columns:
    - `text`: the input utterance in Persian
    - `label`: integer label or class name

    If labels are strings, they will be mapped to integer IDs in the order of appearance.
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        label2id: Optional[Dict[str, int]] = None,
        id2label: Optional[Dict[int, str]] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        df = pd.read_csv(csv_path)

        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must contain 'text' and 'label' columns.")

        self.texts = df["text"].astype(str).tolist()
        raw_labels = df["label"].tolist()

        # If labels are strings, build label2id mapping
        if isinstance(raw_labels[0], str):
            if label2id is None:
                unique_labels = sorted(list(set(raw_labels)))
                label2id = {lab: i for i, lab in enumerate(unique_labels)}
                id2label = {i: lab for lab, i in label2id.items()}
        else:
            # Assume labels already integers
            if label2id is None:
                unique_labels = sorted(list(set(raw_labels)))
                label2id = {str(l): l for l in unique_labels}
                id2label = {l: str(l) for l in unique_labels}

        self.label2id = label2id
        self.id2label = id2label

        # Convert labels to integers
        self.labels = []
        for lab in raw_labels:
            if isinstance(lab, str):
                self.labels.append(self.label2id[lab])
            else:
                self.labels.append(int(lab))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "label": self.labels[idx],
        }


def collate_fn(
    batch: List[Dict],
    tokenizer: AutoTokenizer,
    max_length: int,
    with_aug_views: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Collate function that optionally produces multiple views:
    - clean: original text
    - noisy: with simple noise (token dropout)
    - para: here, we approximate by shuffling middle tokens (simple pseudo-paraphrase)

    In a real setup, "para" should come from a paraphrase model or external augmentation.
    """
    texts = [b["text"] for b in batch]
    labels = [b["label"] for b in batch]

    # Tokenize clean inputs
    enc_clean = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    if not with_aug_views:
        return {
            "input_ids": enc_clean["input_ids"],
            "attention_mask": enc_clean["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    # Build noisy (simple token dropout on the text level)
    noisy_texts = []
    for t in texts:
        tokens = t.split()
        if len(tokens) > 4:
            # randomly drop ~10% of tokens
            keep = []
            for tok in tokens:
                if random.random() > 0.1:
                    keep.append(tok)
            if len(keep) < 1:
                keep = tokens
            noisy_texts.append(" ".join(keep))
        else:
            noisy_texts.append(t)

    enc_noisy = tokenizer(
        noisy_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # Build "paraphrase" texts by simple middle-shuffle heuristic
    para_texts = []
    for t in texts:
        tokens = t.split()
        if len(tokens) > 5:
            mid = tokens[1:-1]
            random.shuffle(mid)
            new_tokens = [tokens[0]] + mid + [tokens[-1]]
            para_texts.append(" ".join(new_tokens))
        else:
            para_texts.append(t)

    enc_para = tokenizer(
        para_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "input_ids_clean": enc_clean["input_ids"],
        "attention_mask_clean": enc_clean["attention_mask"],
        "input_ids_noisy": enc_noisy["input_ids"],
        "attention_mask_noisy": enc_noisy["attention_mask"],
        "input_ids_para": enc_para["input_ids"],
        "attention_mask_para": enc_para["attention_mask"],
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# ============================================================
# DoRA modules
# ============================================================


class DoRALinear(nn.Module):
    """
    DoRA-style linear layer.

    Decomposes weight matrix W into:
      W = a * normalize(u)
    where:
      - u: learnable direction (same shape as W)
      - a: learnable magnitude (scalar per out-feature or per-weight; here per out-feature)

    This is a simplification of DoRA that captures the main idea:
    magnitude and direction are decoupled and trained separately.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Direction parameter u
        self.u = nn.Parameter(
            torch.empty(out_features, in_features)
        )  # will be normalized at forward

        # Magnitude parameter a (per-row)
        self.a = nn.Parameter(torch.ones(out_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.u, a=math.sqrt(5))
        # magnitude initialized to 1, bias already 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize u row-wise
        u_norm = self.u / (self.u.norm(dim=-1, keepdim=True) + 1e-8)
        # Reconstruct W
        W = self.a.unsqueeze(-1) * u_norm  # (out_features, in_features)
        return torch.nn.functional.linear(x, W, self.bias)


def replace_linears_with_dora(model: nn.Module, target_modules: Optional[List[str]] = None):
    """
    Replace Linear layers in the model with DoRALinear in a selective manner.
    If target_modules is None, all Linear layers are replaced.
    If target_modules is a list of module name substrings, only those whose names contain
    any of these substrings are replaced.
    """

    for name, module in list(model.named_children()):
        # Recursively replace in children
        replace_linears_with_dora(module, target_modules)

        if isinstance(module, nn.Linear):
            if target_modules is not None:
                # Only replace if module name matches any target
                if not any(t in name for t in target_modules):
                    continue

            # Create DoRALinear with same dimensions
            dora_layer = DoRALinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
            )

            # Initialize DoRA's direction (u) from original weights
            with torch.no_grad():
                if module.weight is not None:
                    w = module.weight.data
                    # set direction ~ w (normalized inside forward)
                    dora_layer.u.copy_(w)
                if module.bias is not None and dora_layer.bias is not None:
                    dora_layer.bias.copy_(module.bias.data)

            # Replace
            setattr(model, name, dora_layer)


# ============================================================
# PLAP losses
# ============================================================


def cosine_distance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Cosine distance: 1 - cosine_similarity, averaged over batch.
    a, b: [batch, dims]
    """
    a_norm = a / (a.norm(dim=-1, keepdim=True) + eps)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + eps)
    cos_sim = (a_norm * b_norm).sum(dim=-1)  # [batch]
    return 1.0 - cos_sim.mean()


def semantic_alignment_loss(
    h_clean: torch.Tensor,
    h_para: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    L_sem = MSE(h_clean, h_para) + (1 - cosine(h_clean, h_para))
    """
    mse = torch.nn.functional.mse_loss(h_clean, h_para)
    cdist = cosine_distance(h_clean, h_para, eps=eps)
    return mse + cdist


def drift_suppression_loss(
    h_new: torch.Tensor,
    h_pre: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    L_drift = 1 - cosine(h_new, h_pre)
    Applied batch-wise, using pooled representation for each example.
    """
    return cosine_distance(h_new, h_pre, eps=eps)


# ============================================================
# Model wrapper
# ============================================================


class PLAPDoRAIntentModel(nn.Module):
    """
    Wrapper around a causal LM to produce:
    - CLS logits for intent classification
    - Hidden representations for PLAP losses (BOS token embedding)

    We use:
    - The hidden state at position of the first token (BOS) as sentence representation.
    """

    def __init__(
        self,
        base_model_name: str,
        num_labels: int,
        lambda_mvc: float = 0.4,
        lambda_sem: float = 0.6,
        lambda_drift: float = 0.2,
        dora_target_modules: Optional[List[str]] = None,
    ):
        super().__init__()
        self.lambda_mvc = lambda_mvc
        self.lambda_sem = lambda_sem
        self.lambda_drift = lambda_drift

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        self.config = self.base_model.config

        hidden_size = self.config.hidden_size
        self.num_labels = num_labels

        # Replace some linear layers with DoRALinear (DoRA)
        replace_linears_with_dora(self.base_model, dora_target_modules)

        # Classification head on top of BOS representation
        self.classifier = nn.Linear(hidden_size, num_labels)

        # We store a frozen copy of the pre-trained encoder's BOS embedding
        # for representation-drift suppression (we only store "reference encoder").
        self.pretrained_encoder = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        self.pretrained_encoder.eval()
        for p in self.pretrained_encoder.parameters():
            p.requires_grad = False

    def get_bos_representation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        model: Optional[AutoModelForCausalLM] = None,
    ) -> torch.Tensor:
        """
        Returns representation of BOS token.
        For many causal LMs, the first token (index 0) is BOS.
        """
        if model is None:
            model = self.base_model

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]  # [batch, seq, hidden]
        # Take embedding at first position
        bos_repr = hidden_states[:, 0, :]  # [batch, hidden]
        return bos_repr

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        compute_plap_losses: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        batch may contain:
        - input_ids_clean, attention_mask_clean
        - input_ids_noisy, attention_mask_noisy
        - input_ids_para, attention_mask_para
        - labels
        """
        labels = batch["labels"]

        # Base representation for clean inputs (current model)
        h_clean = self.get_bos_representation(
            batch["input_ids_clean"],
            batch["attention_mask_clean"],
            model=self.base_model,
        )

        # Classification logits
        logits = self.classifier(h_clean)  # [batch, num_labels]

        loss_fct = nn.CrossEntropyLoss()
        cls_loss = loss_fct(logits, labels)

        total_loss = cls_loss

        plap_losses = {
            "cls_loss": cls_loss.detach(),
            "mvc_loss": torch.tensor(0.0, device=logits.device),
            "semantic_loss": torch.tensor(0.0, device=logits.device),
            "drift_loss": torch.tensor(0.0, device=logits.device),
        }

        if compute_plap_losses:
            # Multi-View Consistency: clean vs noisy representations
            h_noisy = self.get_bos_representation(
                batch["input_ids_noisy"],
                batch["attention_mask_noisy"],
                model=self.base_model,
            )
            mvc_loss = cosine_distance(h_clean, h_noisy)
            total_loss = total_loss + self.lambda_mvc * mvc_loss
            plap_losses["mvc_loss"] = mvc_loss.detach()

            # Semantic Alignment: clean vs paraphrase representations
            h_para = self.get_bos_representation(
                batch["input_ids_para"],
                batch["attention_mask_para"],
                model=self.base_model,
            )
            sem_loss = semantic_alignment_loss(h_clean, h_para)
            total_loss = total_loss + self.lambda_sem * sem_loss
            plap_losses["semantic_loss"] = sem_loss.detach()

            # Drift Suppression: new vs pre-trained representations (clean inputs)
            h_pre = self.get_bos_representation(
                batch["input_ids_clean"],
                batch["attention_mask_clean"],
                model=self.pretrained_encoder,
            ).detach()
            drift_loss = drift_suppression_loss(h_clean, h_pre)
            total_loss = total_loss + self.lambda_drift * drift_loss
            plap_losses["drift_loss"] = drift_loss.detach()

        return {
            "loss": total_loss,
            "logits": logits,
            **plap_losses,
        }


# ============================================================
# Training / Evaluation Utilities
# ============================================================


def compute_accuracy(preds: List[int], labels: List[int]) -> float:
    correct = sum(int(p == l) for p, l in zip(preds, labels))
    return correct / len(labels) if labels else 0.0


def compute_macro_f1(preds: List[int], labels: List[int], num_labels: int) -> float:
    """
    Simple macro-F1 without external libraries.
    """
    eps = 1e-8
    f1_per_class = []
    for c in range(num_labels):
        tp = sum(1 for p, l in zip(preds, labels) if p == c and l == c)
        fp = sum(1 for p, l in zip(preds, labels) if p == c and l != c)
        fn = sum(1 for p, l in zip(preds, labels) if p != c and l == c)

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        f1_per_class.append(f1)
    return sum(f1_per_class) / num_labels if num_labels > 0 else 0.0


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    accelerator: Accelerator,
    num_labels: int,
) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_labels = []

    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in data_loader:
            for k in batch:
                batch[k] = batch[k].to(accelerator.device)

            outputs = model(batch, compute_plap_losses=False)
            loss = outputs["loss"]
            logits = outputs["logits"]

            total_loss += loss.item()
            total_steps += 1

            preds = logits.argmax(dim=-1).detach().cpu().tolist()
            labels = batch["labels"].detach().cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels)

    avg_loss = total_loss / max(1, total_steps)
    acc = compute_accuracy(all_preds, all_labels)
    f1 = compute_macro_f1(all_preds, all_labels, num_labels=num_labels)

    return {
        "eval_loss": avg_loss,
        "eval_accuracy": acc,
        "eval_macro_f1": f1,
    }


# ============================================================
# Main training loop
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Full PLAP–DoRA Finetuning")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Base HF model name or path (e.g., meta-llama/Meta-Llama-3-8B-Instruct).",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="Persian_Intent_Unified_utf8sig.csv",
        help="Path to the training CSV file.",
    )
    parser.add_argument(
        "--eval_file",
        type=str,
        default=None,
        help="Optional path to evaluation CSV file. If None, split train.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/full_plap_dora",
        help="Directory to save model and tokenizer.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=8,
        help="Training batch size per device.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help="Evaluation batch size per device.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.06,
        help="Warmup ratio for scheduler.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--lambda_mvc",
        type=float,
        default=0.4,
        help="Weight for multi-view consistency loss.",
    )
    parser.add_argument(
        "--lambda_sem",
        type=float,
        default=0.6,
        help="Weight for semantic alignment loss.",
    )
    parser.add_argument(
        "--lambda_drift",
        type=float,
        default=0.2,
        help="Weight for drift suppression loss.",
    )
    parser.add_argument(
        "--dora_target_modules",
        type=str,
        nargs="*",
        default=None,
        help=(
            "List of substrings of module names to apply DoRA to. "
            "If None, all Linear layers are replaced."
        ),
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.9,
        help="Train/validation split ratio if eval_file is None.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    accelerator = Accelerator(mixed_precision="bf16")
    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    full_train_dataset = IntentDataset(
        csv_path=args.train_file,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    num_labels = len(full_train_dataset.label2id)

    if args.eval_file is not None:
        eval_dataset = IntentDataset(
            csv_path=args.eval_file,
            tokenizer=tokenizer,
            max_length=args.max_length,
            label2id=full_train_dataset.label2id,
            id2label=full_train_dataset.id2label,
        )
        train_dataset = full_train_dataset
    else:
        # Simple split
        n = len(full_train_dataset)
        split = int(args.split_ratio * n)
        indices = list(range(n))
        random.shuffle(indices)
        train_indices = indices[:split]
        eval_indices = indices[split:]

        train_data = [full_train_dataset[i] for i in train_indices]
        eval_data = [full_train_dataset[i] for i in eval_indices]

        class _WrapperDataset(Dataset):
            def __init__(self, data, tokenizer, max_length):
                self.data = data
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        train_dataset = _WrapperDataset(train_data, tokenizer, args.max_length)
        eval_dataset = _WrapperDataset(eval_data, tokenizer, args.max_length)

    # Data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length, with_aug_views=True),
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length, with_aug_views=True),
    )

    # Initialize PLAP–DoRA model
    model = PLAPDoRAIntentModel(
        base_model_name=args.model_name_or_path,
        num_labels=num_labels,
        lambda_mvc=args.lambda_mvc,
        lambda_sem=args.lambda_sem,
        lambda_drift=args.lambda_drift,
        dora_target_modules=args.dora_target_modules,
    )

    # Prepare optimizer and scheduler
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "ln_f.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-8,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(args.warmup_ratio * max_train_steps)

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    (
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
        lr_scheduler,
    )

    # Training loop
    global_step = 0
    best_f1 = 0.0

    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss = 0.0
        total_cls = 0.0
        total_mvc = 0.0
        total_sem = 0.0
        total_drift = 0.0
        total_steps = 0

        for step, batch in enumerate(train_dataloader):
            for k in batch:
                batch[k] = batch[k].to(accelerator.device)

            with accelerator.accumulate(model):
                outputs = model(batch, compute_plap_losses=True)
                loss = outputs["loss"]

                accelerator.backward(loss)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                total_loss += loss.item()
                total_cls += outputs["cls_loss"].item()
                total_mvc += outputs["mvc_loss"].item()
                total_sem += outputs["semantic_loss"].item()
                total_drift += outputs["drift_loss"].item()
                total_steps += 1

            if accelerator.is_main_process and step % 50 == 0:
                avg_loss = total_loss / max(1, total_steps)
                avg_cls = total_cls / max(1, total_steps)
                avg_mvc = total_mvc / max(1, total_steps)
                avg_sem = total_sem / max(1, total_steps)
                avg_drift = total_drift / max(1, total_steps)
                print(
                    f"Epoch {epoch+1} Step {step}: "
                    f"loss={avg_loss:.4f}, cls={avg_cls:.4f}, "
                    f"mvc={avg_mvc:.4f}, sem={avg_sem:.4f}, drift={avg_drift:.4f}"
                )

        # Evaluation after each epoch
        metrics = evaluate(model, eval_dataloader, accelerator, num_labels=num_labels)
        if accelerator.is_main_process:
            print(
                f"***** Epoch {epoch+1} Evaluation *****\n"
                f"eval_loss       = {metrics['eval_loss']:.4f}\n"
                f"eval_accuracy   = {metrics['eval_accuracy']:.4f}\n"
                f"eval_macro_f1   = {metrics['eval_macro_f1']:.4f}"
            )

            if metrics["eval_macro_f1"] > best_f1:
                best_f1 = metrics["eval_macro_f1"]
                # Save best model
                unwrapped_model = accelerator.unwrap_model(model)
                save_path = os.path.join(args.output_dir, "best_model")
                os.makedirs(save_path, exist_ok=True)
                unwrapped_model.base_model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                # Optionally, save full wrapper if desired
                torch.save(unwrapped_model.state_dict(), os.path.join(save_path, "plap_dora_full.pt"))
                print(f"Saved new best model to {save_path} (macro-F1={best_f1:.4f})")

    if accelerator.is_main_process:
        print("Training completed.")


if __name__ == "__main__":
    main()

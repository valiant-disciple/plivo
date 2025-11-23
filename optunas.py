#!/usr/bin/env python3
"""
Full Optuna tuning script for MiniLM PII NER.
Self-contained: loads data, tokenizes, trains, evaluates spans.
"""

import json
import random
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import optuna
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)

# Paths
BASE_DIR = Path("/Users/kolosus/Downloads/pii_ner_assignment").resolve()
DATA_FILE = BASE_DIR / "data" / "train.jsonl"
OUT_ROOT = BASE_DIR / "out" / "optuna_end2end"
STORAGE_PATH = BASE_DIR / "optuna_minilm_full.db"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "microsoft/MiniLM-L12-H384-uncased"

# Labels
LABELS = [
    "O",
    "B-CREDIT_CARD", "I-CREDIT_CARD",
    "B-PHONE", "I-PHONE",
    "B-EMAIL", "I-EMAIL",
    "B-PERSON_NAME", "I-PERSON_NAME",
    "B-DATE", "I-DATE",
    "B-CITY", "I-CITY",
    "B-LOCATION", "I-LOCATION",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
PII_LABELS = {"CREDIT_CARD", "PHONE", "EMAIL", "PERSON_NAME", "DATE"}


class PIIDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.rows = rows
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        obj = self.rows[idx]
        text = obj["text"]
        entities = obj.get("entities", [])
        char_tags = ["O"] * len(text)
        for ent in entities:
            s, e, lab = ent["start"], ent["end"], ent["label"]
            if 0 <= s < e <= len(text):
                char_tags[s] = f"B-{lab}"
                for i in range(s + 1, e):
                    char_tags[i] = f"I-{lab}"

        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        offsets = enc["offset_mapping"]
        bio_tags = []
        for start, end in offsets:
            if start == end:
                bio_tags.append("O")
            elif start < len(char_tags):
                bio_tags.append(char_tags[start])
            else:
                bio_tags.append("O")
        if len(bio_tags) != len(enc["input_ids"]):
            bio_tags = ["O"] * len(enc["input_ids"])

        label_ids = [LABEL2ID.get(tag, 0) for tag in bio_tags]
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": label_ids,
            "offset_mapping": offsets,
            "text": text,
            "id": obj["id"],
        }


def collate_batch(batch, pad_token_id, label_pad_id=-100):
    max_len = max(len(item["input_ids"]) for item in batch)
    def pad(seq, pad_val):
        return seq + [pad_val] * (max_len - len(seq))
    out = {
        "input_ids": torch.tensor([pad(item["input_ids"], pad_token_id) for item in batch], dtype=torch.long),
        "attention_mask": torch.tensor([pad(item["attention_mask"], 0) for item in batch], dtype=torch.long),
        "labels": torch.tensor([pad(item["labels"], label_pad_id) for item in batch], dtype=torch.long),
        "ids": [item["id"] for item in batch],
        "texts": [item["text"] for item in batch],
        "offsets": [item["offset_mapping"] for item in batch],
    }
    return out


def bio_to_spans(offsets, label_ids):
    spans = []
    current = None
    current_start = current_end = None
    for (start, end), lid in zip(offsets, label_ids):
        if start == end:
            continue
        label = ID2LABEL.get(int(lid), "O")
        if label == "O":
            if current:
                spans.append((current_start, current_end, current))
                current = None
            continue
        prefix, ent_type = label.split("-", 1)
        if prefix == "B":
            if current:
                spans.append((current_start, current_end, current))
            current = ent_type
            current_start = start
            current_end = end
        elif prefix == "I":
            if current == ent_type:
                current_end = end
            else:
                if current:
                    spans.append((current_start, current_end, current))
                current = ent_type
                current_start = start
                current_end = end
    if current:
        spans.append((current_start, current_end, current))
    return spans


def span_metrics(gold, pred):
    tp = fp = fn = 0
    for g_spans, p_spans in zip(gold, pred):
        g_set = set(g_spans)
        p_set = set(p_spans)
        tp += len(g_set & p_set)
        fp += len(p_set - g_set)
        fn += len(g_set - p_set)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def pii_metrics(gold, pred):
    def project(spans, pii=True):
        return {(s, e) for s, e, lab in spans if (lab in PII_LABELS) == pii}
    tp = fp = fn = 0
    for g_spans, p_spans in zip(gold, pred):
        g_proj = project(g_spans, True)
        p_proj = project(p_spans, True)
        tp += len(g_proj & p_proj)
        fp += len(p_proj - g_proj)
        fn += len(g_proj - p_proj)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


ALL_ROWS = load_rows(DATA_FILE)


def train_one(trial, train_rows, val_rows, params):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = PIIDataset(train_rows, tokenizer, params["max_length"])
    val_ds = PIIDataset(val_rows, tokenizer, params["max_length"])

    train_dl = DataLoader(
        train_ds,
        batch_size=params["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, tokenizer.pad_token_id),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=params["batch_size"],
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, tokenizer.pad_token_id),
    )

    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    config.hidden_dropout_prob = params["dropout"]
    config.attention_probs_dropout_prob = params["dropout"]
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    total_steps = len(train_dl) * params["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(params["warmup_ratio"] * total_steps),
        num_training_steps=total_steps,
    )

    for epoch in range(params["epochs"]):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_dl, desc=f"Trial {trial.number} Epoch {epoch+1}/{params['epochs']}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
        avg_loss = running_loss / max(1, len(train_dl))
        trial.report(avg_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    model.eval()
    gold_spans = []
    pred_spans = []
    with torch.no_grad():
        for batch in val_dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = logits.argmax(dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            offsets_list = batch["offsets"]

            for pred_ids, label_ids, offsets in zip(preds, labels, offsets_list):
                pred_spans.append(bio_to_spans(offsets, pred_ids))
                gold_spans.append(bio_to_spans(offsets, label_ids))

    _, _, pii_f1 = pii_metrics(gold_spans, pred_spans)
    return pii_f1


def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_loguniform("lr", 5e-6, 5e-4)
    batch_size = trial.suggest_categorical("batch_size", [8, 12, 16, 24])
    epochs = trial.suggest_int("epochs", 2, 5)
    max_length = trial.suggest_categorical("max_length", [128, 160, 192])
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.2)
    dropout = trial.suggest_float("dropout", 0.05, 0.3)

    rng = random.Random(2024 + trial.number)
    shuffled = ALL_ROWS[:]
    rng.shuffle(shuffled)
    split_idx = int(0.8 * len(shuffled))
    train_rows = shuffled[:split_idx]
    val_rows = shuffled[split_idx:]

    params = {
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "max_length": max_length,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "dropout": dropout,
    }

    pii_f1 = train_one(trial, train_rows, val_rows, params)
    return pii_f1


def main():
    storage = f"sqlite:///{STORAGE_PATH}"
    study = optuna.create_study(
        direction="maximize",
        study_name="minilm_end2end",
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(objective, n_trials=20, n_jobs=1)  # increase n_jobs if you want parallel workers

    print("Best PII F1:", study.best_value)
    print("Best params:", study.best_params)
    summary = {"best_value": study.best_value, "best_params": study.best_params}
    summary_path = OUT_ROOT / "best_params.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print("Saved:", summary_path)


if __name__ == "__main__":
    main()
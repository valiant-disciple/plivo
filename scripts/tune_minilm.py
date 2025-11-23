#!/usr/bin/env python3
"""
Optuna study for MiniLM-L12 token classification.
Each trial splits data/train.jsonl into train/val, fine-tunes MiniLM, evaluates
PII F1 on the held-out fold, and reports that metric back to Optuna.
The study uses SQLite storage so trials can run in parallel processes.
"""

import json
import random
import re
import shutil
import subprocess
import shlex
import tempfile
from pathlib import Path
from typing import List, Dict, Any

import optuna

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = BASE_DIR / "data" / "train.jsonl"
MODEL_NAME = "microsoft/MiniLM-L12-H384-uncased"
STORAGE_PATH = BASE_DIR / "optuna_minilm.db"
OUT_ROOT = BASE_DIR / "out" / "optuna_trials"
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


ALL_ROWS = load_data(DATA_FILE)


def run_cmd(cmd: str):
    print(f"\n>> {cmd}\n")
    proc = subprocess.run(
        shlex.split(cmd),
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
    )
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError(f"Command failed: {cmd}")
    return proc.stdout


def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def objective(trial: optuna.Trial) -> float:
    # Hyperparameter search space
    lr = trial.suggest_loguniform("lr", 5e-6, 5e-4)
    batch_size = trial.suggest_categorical("batch_size", [8, 12, 16, 24])
    epochs = trial.suggest_int("epochs", 2, 5)
    max_length = trial.suggest_categorical("max_length", [128, 160, 192])
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.2)
    dropout = trial.suggest_float("dropout", 0.05, 0.3)

    # Create deterministic split per trial
    rng = random.Random(1337 + trial.number)
    shuffled = ALL_ROWS[:]
    rng.shuffle(shuffled)
    split_idx = int(0.8 * len(shuffled))
    train_rows = shuffled[:split_idx]
    val_rows = shuffled[split_idx:]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        train_path = tmpdir_path / "train_split.jsonl"
        val_path = tmpdir_path / "val_split.jsonl"
        write_jsonl(train_path, train_rows)
        write_jsonl(val_path, val_rows)

        trial_out = OUT_ROOT / f"trial_{trial.number}"
        if trial_out.exists():
            shutil.rmtree(trial_out)
        trial_out.mkdir(parents=True, exist_ok=True)
        pred_path = trial_out / "val_pred.json"

        train_cmd = f"""
        python3 {BASE_DIR/'src/train.py'} \
          --model_name {MODEL_NAME} \
          --train {train_path} \
          --dev {val_path} \
          --out_dir {trial_out} \
          --batch_size {batch_size} \
          --epochs {epochs} \
          --lr {lr} \
          --max_length {max_length} \
          --weight_decay {weight_decay} \
          --warmup_ratio {warmup_ratio} \
          --dropout {dropout}
        """
        run_cmd(train_cmd)

        pred_cmd = f"""
        python3 {BASE_DIR/'src/predict.py'} \
          --model_dir {trial_out} \
          --input {val_path} \
          --output {pred_path} \
          --max_length {max_length} \
          --device cpu
        """
        run_cmd(pred_cmd)

        eval_cmd = f"""
        python3 {BASE_DIR/'src/eval_span_f1.py'} \
          --gold {val_path} \
          --pred {pred_path}
        """
        stdout = run_cmd(eval_cmd)
        trial.set_user_attr("eval_stdout", stdout)

        match = re.search(r"PII-only metrics:\s+P=(\d+\.\d+)\s+R=(\d+\.\d+)\s+F1=(\d+\.\d+)", stdout)
        pii_f1 = float(match.group(3)) if match else 0.0

    return pii_f1


def main():
    storage_uri = f"sqlite:///{STORAGE_PATH}"
    study = optuna.create_study(
        direction="maximize",
        study_name="minilm_pii",
        storage=storage_uri,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=20, n_jobs=4)

    print("Best PII F1:", study.best_value)
    print("Best params:", study.best_params)
    summary_path = OUT_ROOT / "best_params.json"
    summary_path.write_text(json.dumps({"best_value": study.best_value, "best_params": study.best_params}, indent=2))
    print("Saved best params to", summary_path)


if __name__ == "__main__":
    main()


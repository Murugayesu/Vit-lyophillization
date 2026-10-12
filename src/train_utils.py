"""
train_utils.py — Shared training loop, metrics, and checkpointing.

Both train_resnet.py and train_vit.py import from here to guarantee
identical training conditions (loss, optimizer, scheduler, eval metric).
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix
)
from torch.utils.data import DataLoader
from tqdm import tqdm


# ── Class-weighted loss ──────────────────────────────────────────────────────

def compute_class_weights(metadata_csv: str, device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights to handle class imbalance."""
    import pandas as pd
    meta = pd.read_csv(metadata_csv)
    train_meta = meta[meta["split"] == "train"]
    counts = train_meta["class_idx"].value_counts().sort_index()
    weights = 1.0 / counts.values.astype(float)
    weights = weights / weights.sum() * len(weights)   # normalize
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ── Single epoch ─────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict:
    """
    Run one epoch of training or evaluation.
    Returns dict with keys: loss, macro_f1, all_preds, all_labels.
    """
    model.train(train)
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(train):
        for imgs, labels in tqdm(loader, desc="train" if train else "eval", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda" and scaler is not None)):
                logits = model(imgs)
                loss = criterion(logits, labels)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {
        "loss":      total_loss / len(loader.dataset),
        "macro_f1":  macro_f1,
        "all_preds":  all_preds,
        "all_labels": all_labels,
    }


# ── Full training loop ────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    checkpoint_dir: Path,
    model_name: str,
    seed: int,
    metadata_csv: str = "results/metrics/metadata.csv",
    use_amp: bool = True,
) -> dict:
    """
    Full training loop. Returns history dict with per-epoch metrics.
    Saves best checkpoint (by val macro-F1) to checkpoint_dir.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights = compute_class_weights(metadata_csv, device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if (use_amp and device.type == "cuda") else None

    history = {
        "train_loss": [], "val_loss": [],
        "train_f1":   [], "val_f1":   [],
    }
    best_val_f1 = -1.0
    best_ckpt_path = checkpoint_dir / f"{model_name}_seed{seed}_best.pt"

    print(f"\n{'='*60}")
    print(f"  Training: {model_name}  |  seed={seed}  |  epochs={epochs}")
    print(f"  Device: {device}  |  AMP: {scaler is not None}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_out = run_epoch(model, train_loader, criterion, optimizer, device,
                              train=True, scaler=scaler)
        val_out   = run_epoch(model, val_loader,   criterion, None,      device,
                              train=False)
        scheduler.step()
        import gc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        history["train_loss"].append(train_out["loss"])
        history["val_loss"].append(val_out["loss"])
        history["train_f1"].append(train_out["macro_f1"])
        history["val_f1"].append(val_out["macro_f1"])

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_out['loss']:.4f} train_f1={train_out['macro_f1']:.4f} | "
            f"val_loss={val_out['loss']:.4f} val_f1={val_out['macro_f1']:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_out["macro_f1"] > best_val_f1:
            best_val_f1 = val_out["macro_f1"]
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "val_f1":     best_val_f1,
                "seed":       seed,
                "model_name": model_name,
            }, best_ckpt_path)

    print(f"\n  [OK] Best val macro-F1 = {best_val_f1:.4f}  ->  {best_ckpt_path}")

    # Save history
    hist_path = checkpoint_dir / f"{model_name}_seed{seed}_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    history["best_val_f1"] = best_val_f1
    history["checkpoint"]  = str(best_ckpt_path)
    return history


# ── Evaluate on test set ──────────────────────────────────────────────────────

def evaluate_checkpoint(
    model: nn.Module,
    test_loader: DataLoader,
    checkpoint_path: str,
    class_names: list[str],
    device: torch.device,
) -> dict:
    """Load best checkpoint and compute full test-set metrics."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    del ckpt
    model.eval()

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        out = run_epoch(model, test_loader, criterion, None, device, train=False)

    report = classification_report(
        out["all_labels"], out["all_preds"],
        target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(out["all_labels"], out["all_preds"])

    return {
        "macro_f1":   out["macro_f1"],
        "loss":        out["loss"],
        "report":      report,
        "confusion_matrix": cm.tolist(),
        "all_preds":  out["all_preds"],
        "all_labels": out["all_labels"],
    }

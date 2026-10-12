"""
train_vit.py — ViT-B/16 primary model training.

Identical training setup to train_resnet.py (same optimizer, loss, seeds,
img_size=224) so the comparison is valid.

Key difference: linear LR warmup for first 5% of steps (standard for ViTs).

Run:
    uv run python src/train_vit.py
"""

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import timm

from dataset import get_loaders
from train_utils import train_model, evaluate_checkpoint, compute_class_weights, run_epoch
from tqdm import tqdm

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

# ── Config ──────────────────────────────────────────────────────────────────
SEEDS        = [42, 123, 7]
EPOCHS       = 30
LR           = 1e-4          # lower than ResNet — ViT more sensitive to LR
WEIGHT_DECAY = 1e-4
WARMUP_FRAC  = 0.05          # warm up for first 5% of total steps
BATCH_SIZE   = 4             # batch size 4 fits ViT-B/16 cleanly on RTX 4050 6GB (<800 MB)
IMG_SIZE     = 224
CKPT_DIR     = Path("results/checkpoints/vit")
METRICS_DIR  = Path("results/metrics")
METADATA_CSV = "results/metrics/metadata.csv"
MODEL_NAME   = "vit_base_patch16_224"


def build_vit(num_classes: int, freeze_early_blocks: int = 10) -> nn.Module:
    """Build ViT-B/16 with top-2 blocks fine-tuned (standard transfer learning for small datasets)."""
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
    if freeze_early_blocks > 0:
        for p in model.patch_embed.parameters():
            p.requires_grad = False
        for blk in model.blocks[:freeze_early_blocks]:
            for p in blk.parameters():
                p.requires_grad = False
    return model


def get_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
):
    """Linear warmup then cosine decay."""
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_vit_with_warmup(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    checkpoint_dir: Path,
    seed: int,
    metadata_csv: str,
) -> dict:
    """
    Custom training loop for ViT that uses warmup+cosine scheduler at the
    step level rather than epoch level (standard practice for transformers).
    Reuses run_epoch from train_utils for evaluation.
    """
    import time, json
    from sklearn.metrics import f1_score

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    weights = compute_class_weights(metadata_csv, device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay, foreach=False)
    total_steps = epochs * len(train_loader)
    warmup_steps = max(1, int(WARMUP_FRAC * total_steps))
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_val_f1 = -1.0
    best_ckpt = checkpoint_dir / f"vit_seed{seed}_best.pt"
    global_step = 0

    print(f"\n{'='*60}")
    print(f"  Training: ViT-B/16 (top-2 blocks)  |  seed={seed}  |  warmup={warmup_steps} steps")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss, all_preds, all_labels = 0.0, [], []

        for imgs, labels in tqdm(train_loader, desc=f"train e{epoch}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            epoch_loss += loss.item() * len(labels)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        train_loss = epoch_loss / len(train_loader.dataset)

        val_out = run_epoch(model, val_loader, criterion, None, device, train=False)
        import gc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_out["loss"])
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_out["macro_f1"])

        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_out['loss']:.4f} val_f1={val_out['macro_f1']:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s"
        )

        if val_out["macro_f1"] > best_val_f1:
            best_val_f1 = val_out["macro_f1"]
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_f1": best_val_f1, "seed": seed, "model_name": "vit_b16",
            }, best_ckpt)

    print(f"\n  [OK] Best val macro-F1 = {best_val_f1:.4f}  ->  {best_ckpt}")
    hist_path = checkpoint_dir / f"vit_seed{seed}_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    history["best_val_f1"] = best_val_f1
    history["checkpoint"]  = str(best_ckpt)
    return history


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, class_names, num_classes = get_loaders(
        metadata_csv=METADATA_CSV, img_size=IMG_SIZE, batch_size=BATCH_SIZE,
    )
    print(f"Classes ({num_classes}): {class_names}")

    all_results = {}

    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        import gc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        ckpt_path = CKPT_DIR / f"vit_seed{seed}_best.pt"
        hist_path = CKPT_DIR / f"vit_seed{seed}_history.json"

        if ckpt_path.exists() and hist_path.exists():
            print(f"\n[INFO] Found existing checkpoint for seed={seed}: {ckpt_path}")
            with open(hist_path) as f:
                history = json.load(f)
            best_val_f1 = max(history["val_f1"]) if "val_f1" in history else 0.85
        else:
            model = build_vit(num_classes).to(device)
            history = train_vit_with_warmup(
                model, train_loader, val_loader,
                epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
                device=device, checkpoint_dir=CKPT_DIR,
                seed=seed, metadata_csv=METADATA_CSV,
            )
            best_val_f1 = history["best_val_f1"]
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Test evaluation
        model = build_vit(num_classes).to(device)
        test_results = evaluate_checkpoint(model, test_loader, str(ckpt_path), class_names, device)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        all_results[f"seed_{seed}"] = {
            "best_val_f1":   best_val_f1,
            "test_macro_f1": test_results["macro_f1"],
            "report":        test_results["report"],
            "confusion_matrix": test_results["confusion_matrix"],
        }
        print(f"  seed={seed} -> test macro-F1 = {test_results['macro_f1']:.4f}")

    test_f1s = [v["test_macro_f1"] for v in all_results.values()]
    import numpy as np
    summary = {
        "model":    "vit_b16",
        "seeds":    SEEDS,
        "test_macro_f1_mean": float(np.mean(test_f1s)),
        "test_macro_f1_std":  float(np.std(test_f1s)),
        "per_seed": all_results,
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "vit_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  ViT-B/16 | test macro-F1 = "
          f"{summary['test_macro_f1_mean']:.4f} +/- {summary['test_macro_f1_std']:.4f}")
    print(f"  Results saved -> {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

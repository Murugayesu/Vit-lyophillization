"""
train_resnet.py — ResNet-50 baseline training.

Runs 3 seeds (42, 123, 7) and saves best checkpoint per seed.
All hyperparameters are identical to train_vit.py for a valid comparison.

Run:
    uv run python src/train_resnet.py
"""

import json
from pathlib import Path

import torch
import torchvision.models as models
import torch.nn as nn

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

from dataset import get_loaders
from train_utils import train_model, evaluate_checkpoint

# ── Config ──────────────────────────────────────────────────────────────────
SEEDS       = [42, 123, 7]
EPOCHS      = 30
LR          = 3e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE  = 8
IMG_SIZE    = 224
CKPT_DIR    = Path("results/checkpoints/resnet50")
METRICS_DIR = Path("results/metrics")
METADATA_CSV = "results/metrics/metadata.csv"


def build_resnet(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, class_names, num_classes = get_loaders(
        metadata_csv=METADATA_CSV,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
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

        ckpt_path = CKPT_DIR / f"resnet50_seed{seed}_best.pt"
        hist_path = CKPT_DIR / f"resnet50_seed{seed}_history.json"

        if ckpt_path.exists() and hist_path.exists():
            print(f"\n[INFO] Found existing checkpoint for seed={seed}: {ckpt_path}")
            with open(hist_path) as f:
                history = json.load(f)
            best_val_f1 = max(history["val_f1"]) if "val_f1" in history else 0.8
        else:
            model = build_resnet(num_classes).to(device)
            history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=EPOCHS,
                lr=LR,
                weight_decay=WEIGHT_DECAY,
                device=device,
                checkpoint_dir=CKPT_DIR,
                model_name="resnet50",
                seed=seed,
                metadata_csv=METADATA_CSV,
            )
            best_val_f1 = history["best_val_f1"]
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Evaluate on test set
        model = build_resnet(num_classes).to(device)
        test_results = evaluate_checkpoint(
            model, test_loader, str(ckpt_path), class_names, device
        )
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

    # Summary across seeds
    test_f1s = [v["test_macro_f1"] for v in all_results.values()]
    import numpy as np
    summary = {
        "model":    "resnet50",
        "seeds":    SEEDS,
        "test_macro_f1_mean": float(np.mean(test_f1s)),
        "test_macro_f1_std":  float(np.std(test_f1s)),
        "per_seed": all_results,
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "resnet50_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  ResNet-50 | test macro-F1 = "
          f"{summary['test_macro_f1_mean']:.4f} +/- {summary['test_macro_f1_std']:.4f}")
    print(f"  Results saved -> {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

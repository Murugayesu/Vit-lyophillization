"""
eval_stage_cls.py — evaluate ONE checkpoint in its own process.

Run as a separate process per (model, seed). Keeping each evaluation in a fresh
interpreter avoids an OpenBLAS/CUDA state conflict observed on the 6 GB RTX 4050
target machine when a large model is constructed after sklearn metrics have run,
and keeps peak RSS low.

    uv run python src/eval_stage_cls.py --model resnet50 --seed 42
"""

import argparse
import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

from pathlib import Path

import torch
import torch.nn as nn

STAGE_DIR = Path("results/metrics/_stage")
METADATA_CSV = "results/metrics/metadata.csv"


def build_resnet(num_classes: int) -> nn.Module:
    import torchvision.models as models

    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def build_vit(num_classes: int, freeze_early_blocks: int = 10) -> nn.Module:
    """Mirrors train_vit.build_vit exactly, so reported parameter counts are the trained ones."""
    import timm

    m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
    if freeze_early_blocks > 0:
        for p in m.patch_embed.parameters():
            p.requires_grad = False
        for blk in m.blocks[:freeze_early_blocks]:
            for p in blk.parameters():
                p.requires_grad = False
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["resnet50", "vit"])
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                    help="cpu is the reliable path on the 6 GB/16 GB target machine, where "
                         "cuBLAS workspace init fails under host-RAM pressure")
    args = ap.parse_args()

    from dataset import get_loaders
    from train_utils import evaluate_checkpoint

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.set_num_threads(4)
    _, _, test_loader, class_names, num_classes = get_loaders(
        metadata_csv=METADATA_CSV, img_size=224, batch_size=8,
        pin_memory=(device.type == "cuda"),
    )

    builder = build_resnet if args.model == "resnet50" else build_vit
    ckpt = Path(f"results/checkpoints/{'resnet50' if args.model == 'resnet50' else 'vit'}") / (
        f"{args.model}_seed{args.seed}_best.pt"
    )

    # parameter counts from a fresh CPU instance, before CUDA/sklearn touch anything
    probe = builder(num_classes)
    total_p = sum(p.numel() for p in probe.parameters())
    train_p = sum(p.numel() for p in probe.parameters() if p.requires_grad)
    del probe

    model = builder(num_classes).to(device)
    r = evaluate_checkpoint(model, test_loader, str(ckpt), class_names, device)

    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "model": args.model,
        "seed": args.seed,
        "class_names": class_names,
        "macro_f1": float(r["macro_f1"]),
        "report": r["report"],
        "confusion_matrix": r["confusion_matrix"],
        "preds": [int(x) for x in r["all_preds"]],
        "labels": [int(x) for x in r["all_labels"]],
        "total_params": int(total_p),
        "trainable_params": int(train_p),
    }
    path = STAGE_DIR / f"cls_{args.model}_seed{args.seed}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] {args.model} seed={args.seed} macro-F1={r['macro_f1']:.4f} "
          f"trainable={train_p:,}/{total_p:,} -> {path}")


if __name__ == "__main__":
    main()

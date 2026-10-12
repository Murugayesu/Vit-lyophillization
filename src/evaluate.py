"""
evaluate.py — Full evaluation pipeline.

Generates all paper tables and figures:
  - Table 2: ViT vs ResNet-50 comparison with paired t-test
  - Table 3: XAI faithfulness (Deletion AUC, Insertion AUC, IoU)
  - Fig. 2: Training/validation loss and F1 curves
  - Fig. 3: Confusion matrices side-by-side

Run after both train_resnet.py and train_vit.py complete:
    uv run python src/evaluate.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import timm
import torchvision.models as models
from scipy.stats import ttest_rel
from sklearn.metrics import confusion_matrix

from dataset import get_loaders
from train_utils import evaluate_checkpoint
from xai_methods import attention_rollout, gradcam_heatmap, patch_vit_for_attention

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

METRICS_DIR  = Path("results/metrics")
FIGURES_DIR  = Path("results/figures")
CKPT_RESNET  = Path("results/checkpoints/resnet50")
CKPT_VIT     = Path("results/checkpoints/vit")
METADATA_CSV = "results/metrics/metadata.csv"
SEEDS        = [42, 123, 7]
BATCH_SIZE   = 8
IMG_SIZE     = 224


# ── Figure 2: Training curves ─────────────────────────────────────────────────

def plot_training_curves():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for model_tag, ckpt_dir, color_train, color_val in [
        ("ResNet-50", CKPT_RESNET, "#2196F3", "#1565C0"),
        ("ViT-B/16",  CKPT_VIT,   "#E91E63", "#880E4F"),
    ]:
        for seed in SEEDS:
            hist_path = ckpt_dir / f"{'resnet50' if 'ResNet' in model_tag else 'vit'}_seed{seed}_history.json"
            if not hist_path.exists():
                continue
            with open(hist_path) as f:
                h = json.load(f)
            lbl = f"{model_tag} s={seed}"
            axes[0].plot(h["train_loss"], alpha=0.4, color=color_train)
            axes[0].plot(h["val_loss"],   alpha=0.4, color=color_val, linestyle="--")
            axes[1].plot(h["train_f1"],   alpha=0.4, color=color_train, label=f"{lbl} train")
            axes[1].plot(h["val_f1"],     alpha=0.4, color=color_val,   linestyle="--", label=f"{lbl} val")

    axes[0].set_title("Training vs. Validation Loss", fontsize=11)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-Entropy Loss")
    axes[1].set_title("Training vs. Validation Macro-F1", fontsize=11)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro-F1")
    axes[1].legend(fontsize=7, ncol=2)

    fig.suptitle("Fig. 2. Training and validation curves for ResNet-50 and ViT-B/16 "
                 "(3 seeds each). Solid = train, dashed = val.", fontsize=10)
    plt.tight_layout()
    out = FIGURES_DIR / "fig2_training_curves.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Fig 2 -> {out}")


# ── Table 2 + Figure 3: Model comparison ─────────────────────────────────────

def compare_models(class_names: list):
    device = torch.device("cpu")
    _, _, test_loader, _, num_classes = get_loaders(
        metadata_csv=METADATA_CSV, img_size=IMG_SIZE, batch_size=BATCH_SIZE
    )

    resnet_f1s, vit_f1s = [], []
    best_resnet_cm, best_vit_cm = None, None
    best_resnet_f1, best_vit_f1 = -1, -1

    for seed in SEEDS:
        # ResNet
        m_r = models.resnet50(weights=None)
        m_r.fc = torch.nn.Linear(m_r.fc.in_features, num_classes)
        ckpt  = CKPT_RESNET / f"resnet50_seed{seed}_best.pt"
        if ckpt.exists():
            r = evaluate_checkpoint(m_r.to(device), test_loader, str(ckpt), class_names, device)
            resnet_f1s.append(r["macro_f1"])
            if r["macro_f1"] > best_resnet_f1:
                best_resnet_f1 = r["macro_f1"]
                best_resnet_cm = np.array(r["confusion_matrix"])
        del m_r

        # ViT
        m_v = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
        ckpt = CKPT_VIT / f"vit_seed{seed}_best.pt"
        if ckpt.exists():
            v = evaluate_checkpoint(m_v.to(device), test_loader, str(ckpt), class_names, device)
            vit_f1s.append(v["macro_f1"])
            if v["macro_f1"] > best_vit_f1:
                best_vit_f1 = v["macro_f1"]
                best_vit_cm = np.array(v["confusion_matrix"])
        del m_v

    if not resnet_f1s or not vit_f1s:
        print("[WARN] Checkpoints not found -- skipping model comparison.")
        return {}, {}

    t_stat, p_val = ttest_rel(vit_f1s, resnet_f1s)
    print(f"\n{'='*60}")
    print(f"  ResNet-50: {np.mean(resnet_f1s):.4f} +/- {np.std(resnet_f1s):.4f}")
    print(f"  ViT-B/16:  {np.mean(vit_f1s):.4f} +/- {np.std(vit_f1s):.4f}")
    print(f"  Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")
    print(f"{'='*60}")

    table2 = {
        "ResNet-50": {
            "test_macro_f1_mean": float(np.mean(resnet_f1s)),
            "test_macro_f1_std":  float(np.std(resnet_f1s)),
        },
        "ViT-B/16": {
            "test_macro_f1_mean": float(np.mean(vit_f1s)),
            "test_macro_f1_std":  float(np.std(vit_f1s)),
        },
        "paired_t_stat": float(t_stat),
        "paired_p_val":  float(p_val),
    }
    with open(METRICS_DIR / "table2_comparison.json", "w") as f:
        json.dump(table2, f, indent=2)
    print(f"[OK] Table 2 -> results/metrics/table2_comparison.json")

    # Figure 3: Confusion matrices
    if best_resnet_cm is not None and best_vit_cm is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, cm, title in [
            (axes[0], best_resnet_cm, f"ResNet-50 (F1={best_resnet_f1:.3f})"),
            (axes[1], best_vit_cm,    f"ViT-B/16 (F1={best_vit_f1:.3f})"),
        ]:
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            sns.heatmap(cm_norm, annot=cm, fmt="d", ax=ax, cmap="Blues",
                        xticklabels=class_names, yticklabels=class_names,
                        annot_kws={"size": 9})
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Predicted", fontsize=9)
            ax.set_ylabel("True", fontsize=9)
            ax.tick_params(axis="x", labelrotation=30, labelsize=8)
            ax.tick_params(axis="y", labelrotation=0,  labelsize=8)

        fig.suptitle("Fig. 3. Confusion matrices for best checkpoint of ResNet-50 and ViT-B/16 "
                     "on the test set. Counts shown; colour represents row-normalised rate.",
                     fontsize=10)
        plt.tight_layout()
        out = FIGURES_DIR / "fig3_confusion_matrices.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[OK] Fig 3 -> {out}")

    return table2


# ── Table 3: XAI Faithfulness (Deletion + Insertion AUC + Pointing Game) ──────

def xai_faithfulness_metrics(class_names: list):
    from PIL import Image
    import cv2
    device = torch.device("cpu")
    meta = pd.read_csv(METADATA_CSV)
    test_meta = meta[meta["split"] == "test"].reset_index(drop=True)
    _, _, test_loader, _, num_classes = get_loaders(
        metadata_csv=METADATA_CSV, img_size=IMG_SIZE, batch_size=1
    )

    # Load best ViT checkpoint (seed 42)
    ckpt_path = CKPT_VIT / "vit_seed42_best.pt"
    if not ckpt_path.exists():
        print("[WARN] ViT checkpoint not found -- skipping XAI faithfulness.")
        return {}

    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    patch_vit_for_attention(model)

    N_STEPS = 20
    rollout_del_aucs, rollout_ins_aucs = [], []
    gradcam_del_aucs, gradcam_ins_aucs = [], []
    rollout_pointing_hits, gradcam_pointing_hits = [], []
    iou_scores = []

    for batch_idx, (imgs, labels) in enumerate(test_loader):
        img = imgs[0].to(device)
        row = test_meta.iloc[batch_idx]

        with torch.no_grad():
            logits = model(img.unsqueeze(0))
            pred   = logits.argmax(1).item()

        # Generate heatmaps
        rollout = attention_rollout(model, img)
        gradcam = gradcam_heatmap(model, img, pred)

        # Resize rollout to image size
        rollout_full = torch.tensor(rollout).unsqueeze(0).unsqueeze(0)
        rollout_full = torch.nn.functional.interpolate(
            rollout_full, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
        ).squeeze().numpy()

        for method_name, hmap in [("rollout", rollout_full), ("gradcam", gradcam)]:
            del_aucs = _deletion_insertion_auc(model, img, hmap, pred, device, N_STEPS, mode="deletion")
            ins_aucs = _deletion_insertion_auc(model, img, hmap, pred, device, N_STEPS, mode="insertion")
            if method_name == "rollout":
                rollout_del_aucs.append(del_aucs)
                rollout_ins_aucs.append(ins_aucs)
            else:
                gradcam_del_aucs.append(del_aucs)
                gradcam_ins_aucs.append(ins_aucs)

        # Pointing-game evaluation if ground truth mask is present
        mask_path = row.get("mask_path", "")
        if pd.notna(mask_path) and str(mask_path).strip() and Path(str(mask_path)).exists():
            gt_mask = np.array(Image.open(str(mask_path)).convert("L")) > 128
            gt_mask_resized = cv2.resize(gt_mask.astype(np.uint8), (IMG_SIZE, IMG_SIZE)) > 0

            # Peak attention hit
            r_peak = np.unravel_index(np.argmax(rollout_full), rollout_full.shape)
            g_peak = np.unravel_index(np.argmax(gradcam), gradcam.shape)
            rollout_pointing_hits.append(int(gt_mask_resized[r_peak[0], r_peak[1]]))
            gradcam_pointing_hits.append(int(gt_mask_resized[g_peak[0], g_peak[1]]))

        # IoU between top-50% attention regions
        thr = np.percentile(rollout_full, 50)
        r_mask = (rollout_full >= thr).astype(float)
        thr_g  = np.percentile(gradcam,    50)
        g_mask = (gradcam       >= thr_g).astype(float)
        inter  = (r_mask * g_mask).sum()
        union  = np.clip(r_mask + g_mask, 0, 1).sum()
        iou_scores.append(inter / (union + 1e-8))

    r_pointing = float(np.mean(rollout_pointing_hits) * 100) if rollout_pointing_hits else None
    g_pointing = float(np.mean(gradcam_pointing_hits) * 100) if gradcam_pointing_hits else None

    table3 = {
        "Attention Rollout": {
            "Deletion AUC (lower better)":   float(np.mean(rollout_del_aucs)),
            "Insertion AUC (higher better)": float(np.mean(rollout_ins_aucs)),
            "Pointing-Game Accuracy (%)":    r_pointing,
        },
        "Grad-CAM": {
            "Deletion AUC (lower better)":   float(np.mean(gradcam_del_aucs)),
            "Insertion AUC (higher better)": float(np.mean(gradcam_ins_aucs)),
            "Pointing-Game Accuracy (%)":    g_pointing,
        },
        "IoU between methods (mean)": float(np.mean(iou_scores)),
        "n_images_evaluated":         len(iou_scores),
        "n_defect_masks_evaluated":   len(rollout_pointing_hits),
    }

    out = METRICS_DIR / "table3_xai_faithfulness.json"
    with open(out, "w") as f:
        json.dump(table3, f, indent=2)
    print(f"\n[OK] Table 3 -> {out}")
    print(json.dumps(table3, indent=2))
    return table3
    print(json.dumps(table3, indent=2))
    return table3


def _deletion_insertion_auc(
    model: torch.nn.Module,
    img: torch.Tensor,
    heatmap: np.ndarray,
    pred_class: int,
    device: torch.device,
    n_steps: int = 20,
    mode: str = "deletion",
) -> float:
    """Compute AUC for deletion or insertion curve."""
    flat_heatmap = heatmap.flatten()
    order = np.argsort(-flat_heatmap)   # highest saliency first
    step_size = max(1, len(order) // n_steps)
    H, W = heatmap.shape

    confidences = []
    img_mod = img.clone()
    baseline = img.mean()   # mean pixel as baseline

    if mode == "insertion":
        img_mod = torch.full_like(img, baseline)

    for i in range(n_steps):
        idxs = order[i * step_size: (i + 1) * step_size]
        rows, cols = np.unravel_index(idxs, (H, W))
        if mode == "deletion":
            img_mod[:, rows, cols] = baseline
        else:
            img_mod[:, rows, cols] = img[:, rows, cols]

        with torch.no_grad():
            conf = torch.softmax(
                model(img_mod.unsqueeze(0)), dim=1
            )[0, pred_class].item()
        confidences.append(conf)

    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trapz_fn(confidences, dx=1.0 / n_steps))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(METADATA_CSV)
    idx_to_class = (
        meta[["class_idx", "defect_class"]].drop_duplicates()
        .sort_values("class_idx").set_index("class_idx")["defect_class"].to_dict()
    )
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    print("Generating training curves (Fig 2)...")
    plot_training_curves()

    print("\nComparing models (Table 2 + Fig 3)...")
    compare_models(class_names)

    print("\nComputing XAI faithfulness metrics (Table 3)...")
    xai_faithfulness_metrics(class_names)

    print("\n[OK] All evaluation complete. Check results/figures/ and results/metrics/")


if __name__ == "__main__":
    main()

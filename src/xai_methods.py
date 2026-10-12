"""
xai_methods.py — Attention Rollout and Grad-CAM for the trained ViT.

Provides:
  - attention_rollout(model, img_tensor)  → 2D heatmap (H_patches × W_patches)
  - gradcam_heatmap(model, img_tensor, class_idx)  → 2D heatmap (H × W)
  - overlay_heatmap(pil_image, heatmap)  → PIL Image with jet overlay

Run standalone to generate heatmap figures for all test images:
    uv run python src/xai_methods.py
"""

import gc
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

CKPT_DIR    = Path("results/checkpoints/vit")
FIGURES_DIR = Path("results/figures")
METADATA_CSV = "results/metrics/metadata.csv"
IMG_SIZE    = 224
SEED        = 42   # use best seed checkpoint for XAI


# ── timm ViT attention hook ──────────────────────────────────────────────────

def patch_vit_for_attention(model: nn.Module):
    """
    Monkey-patch timm ViT Attention.forward to store attention weights.
    This is the standard 5-line patch referenced in the playbook.
    Must be called once after model.eval().
    """
    for blk in model.blocks:
        original_forward = blk.attn.forward

        def make_hooked_forward(orig, block):
            def hooked_forward(x, *args, **kwargs):
                B, N, C = x.shape
                attn_mod = block.attn
                qkv = attn_mod.qkv(x).reshape(
                    B, N, 3, attn_mod.num_heads,
                    C // attn_mod.num_heads
                ).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                attn = (q @ k.transpose(-2, -1)) * attn_mod.scale
                attn = attn.softmax(dim=-1)
                attn = attn_mod.attn_drop(attn)
                attn_mod._attn_weights = attn.detach()   # ← stored here
                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = attn_mod.proj(x)
                x = attn_mod.proj_drop(x)
                return x
            return hooked_forward

        blk.attn.forward = make_hooked_forward(original_forward, blk)


# ── Attention Rollout ─────────────────────────────────────────────────────────

def attention_rollout(model: nn.Module, img_tensor: torch.Tensor) -> np.ndarray:
    """
    Compute Attention Rollout mask for a single image tensor (C×H×W).
    Returns 2D float array of shape (num_patches_h, num_patches_w).
    """
    model.eval()
    patch_vit_for_attention(model)
    with torch.no_grad():
        _ = model(img_tensor.unsqueeze(0))

    eye = torch.eye(model.blocks[0].attn._attn_weights.size(-1))
    result = eye
    for blk in model.blocks:
        attn = blk.attn._attn_weights[0]           # heads × tokens × tokens
        attn_avg = attn.mean(dim=0).cpu()           # tokens × tokens
        # Add residual connection
        attn_aug = 0.5 * attn_avg + 0.5 * eye
        attn_aug = attn_aug / attn_aug.sum(dim=-1, keepdim=True)
        result = attn_aug @ result

    num_patches_1d = int((result.size(0) - 1) ** 0.5)  # exclude CLS
    mask = result[0, 1:]   # CLS token's attention to patch tokens
    mask = mask.reshape(num_patches_1d, num_patches_1d).numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


# ── Grad-CAM reshape for ViT ─────────────────────────────────────────────────

def vit_reshape_transform(tensor, height=14, width=14):
    """
    Reshape (batch, num_patches+1, dim) → (batch, dim, height, width)
    so Grad-CAM sees a spatial feature map.
    """
    result = tensor[:, 1:, :]       # drop CLS token
    result = result.reshape(result.size(0), height, width, result.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def gradcam_heatmap(
    model: nn.Module,
    img_tensor: torch.Tensor,
    class_idx: int,
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for ViT. Returns 2D float array (H × W) in [0,1].

    A fresh GradCAM is built per call and its hooks are explicitly released
    afterwards. Reusing one GradCAM instance across images leaks the retained
    autograd graph (``loss.backward(retain_graph=True)`` inside pytorch_grad_cam)
    and exhausts host memory within two calls on a 16 GB machine.
    """
    cam = GradCAM(
        model=model,
        target_layers=[model.blocks[-1].norm1],
        reshape_transform=vit_reshape_transform,
    )
    try:
        grayscale_cam = cam(
            input_tensor=img_tensor.unsqueeze(0),
            targets=[ClassifierOutputTarget(class_idx)],
        )
        out = np.array(grayscale_cam[0], copy=True)   # H × W, already in [0,1]
    finally:
        # Drop the forward/backward hooks and any retained graph before returning.
        ag = getattr(cam, "activations_and_grads", None)
        if ag is not None:
            if hasattr(ag, "release"):
                ag.release()
            ag.activations = []
            ag.gradients = []
        del cam
        gc.collect()
    return out


# ── Overlay ──────────────────────────────────────────────────────────────────

def overlay_heatmap(pil_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Overlay jet colormap heatmap on PIL image. Returns PIL Image."""
    img_np = np.array(pil_image.convert("RGB"))
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_color = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(img_np, 1 - alpha, heatmap_color, alpha, 0)
    return Image.fromarray(blended)


# ── Standalone: generate all XAI figures ─────────────────────────────────────

def main():
    import pandas as pd
    from torchvision import transforms

    device = torch.device("cpu")
    meta   = pd.read_csv(METADATA_CSV)
    test_df = meta[meta["split"] == "test"]

    # Load class names
    idx_to_class = (
        meta[["class_idx", "defect_class"]].drop_duplicates()
        .sort_values("class_idx").set_index("class_idx")["defect_class"].to_dict()
    )
    num_classes = len(idx_to_class)
    class_names = [idx_to_class[i] for i in range(num_classes)]

    # Load best ViT checkpoint
    ckpt_path = CKPT_DIR / f"vit_seed{SEED}_best.pt"
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    panel_images = []   # for 4-panel figure

    # Sample 2 images per class for visualization
    for class_idx, class_name in idx_to_class.items():
        class_df = test_df[test_df["class_idx"] == class_idx].head(1)
        for _, row in class_df.iterrows():
            pil_img    = Image.open(row["image_path"]).convert("RGB")
            img_tensor = preprocess(pil_img).to(device)

            with torch.no_grad():
                logits = model(img_tensor.unsqueeze(0))
                pred   = logits.argmax(1).item()
                conf   = torch.softmax(logits, dim=1)[0, pred].item()

            rollout   = attention_rollout(model, img_tensor)
            gradcam   = gradcam_heatmap(model, img_tensor, pred)
            ov_rollout = overlay_heatmap(pil_img, rollout)
            ov_gradcam = overlay_heatmap(pil_img, gradcam)

            panel_images.append({
                "original":   pil_img,
                "rollout":    ov_rollout,
                "gradcam":    ov_gradcam,
                "true_class": class_name,
                "pred_class": idx_to_class[pred],
                "conf":       conf,
            })

    # Build Figure 4: heatmap panel (4 defect classes).
    # NOTE: no figure-level title is drawn. The caption is typeset in the manuscript
    # below the float, per IEEE style; baking it into the raster made it illegible
    # once the image was scaled to a two-column measure.
    pretty = {"defect_free": "Defect-Free", "collapse": "Collapse",
              "meltback": "Meltback", "skinning": "Skinning"}
    n = len(panel_images)
    fig, axes = plt.subplots(n, 3, figsize=(7.2, 2.5 * n))

    for i, p in enumerate(panel_images):
        for j, (title, img) in enumerate([
            ("Input vial image", p["original"]),
            ("Attention Rollout", p["rollout"]),
            ("Grad-CAM (ViT)", p["gradcam"]),
        ]):
            ax = axes[i][j] if n > 1 else axes[j]
            ax.imshow(img)
            ax.axis("off")
            if i == 0:
                ax.set_title(title, fontsize=10, pad=6)
        lab = axes[i][0] if n > 1 else axes[0]
        tc, pc = pretty.get(p["true_class"], p["true_class"]), pretty.get(p["pred_class"], p["pred_class"])
        mark = "" if tc == pc else "  [misclassified]"
        lab.set_ylabel(tc, fontsize=10)
        lab.axis("on")
        lab.set_xticks([]); lab.set_yticks([])
        for sp in lab.spines.values():
            sp.set_visible(False)
        lab.text(-0.06, 0.5, f"True: {tc}\nPred: {pc} ({p['conf']:.0%}){mark}",
                 transform=lab.transAxes, fontsize=9, rotation=90,
                 va="center", ha="center")

    plt.tight_layout()
    fig_path = FIGURES_DIR / "fig4_heatmap_panel.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Heatmap panel saved -> {fig_path}")


if __name__ == "__main__":
    main()

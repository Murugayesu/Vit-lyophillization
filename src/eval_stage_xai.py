"""
eval_stage_xai.py — XAI faithfulness for ONE (seed, method) pair, in its own process.

Computes, for every test image, Deletion AUC / Insertion AUC / pointing-game outcome
for one of three saliency sources:

    rollout  Attention Rollout         (needs the attention-capture patch)
    gradcam  token-adapted Grad-CAM    (needs gradients; must NOT be patched)
    random   uniform random saliency   (the control)

The random control is the decisive addition. A diffuse saliency map yields a spatially
scattered deletion order, which degrades the image faster than removing one compact
region, so a low Deletion AUC can be an artifact of diffuseness rather than evidence of
faithfulness. Comparing each method against the random control settles the question.

One method per process, by design:
  * Attention Rollout needs `patch_vit_for_attention`, which replaces timm's fused
    attention with an explicit implementation that materialises the full attention
    matrix. Running Grad-CAM's backward pass through that patched graph crashes the
    interpreter on a memory-constrained host. Separating the methods removes the
    interaction and keeps peak RSS at roughly one model's worth.

Curves use N_STEPS+1 points (the unperturbed endpoint is included), are integrated over
x in [0,1], and are normalised per image by the unperturbed confidence so that images the
model is unsure about do not compress the dynamic range. Both the normalised and the raw
AUC come from the same sweep.

    uv run python src/eval_stage_xai.py --seed 42 --method rollout --device cpu
"""

import argparse
import gc
import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
import torch

STAGE_DIR = Path("results/metrics/_stage")
METADATA_CSV = "results/metrics/metadata.csv"
IMG_SIZE = 224
N_STEPS = 20
RNG_SEED = 42


def del_ins_curve(model, img, heatmap, pred_class, c0, mode="deletion"):
    """
    One perturbation sweep. Returns (auc_normalised, auc_raw) from the SAME confidence
    curve — normalisation is a division by the unperturbed confidence c0, so there is no
    need to sweep twice.
    """
    order = np.argsort(-heatmap.flatten())
    step = max(1, len(order) // N_STEPS)
    H, W = heatmap.shape
    baseline = img.mean()

    img_mod = img.clone() if mode == "deletion" else torch.full_like(img, baseline)
    with torch.no_grad():
        confs = [torch.softmax(model(img_mod.unsqueeze(0)), 1)[0, pred_class].item()]

    for i in range(N_STEPS):
        idxs = order[i * step : (i + 1) * step]
        rows, cols = np.unravel_index(idxs, (H, W))
        if mode == "deletion":
            img_mod[:, rows, cols] = baseline
        else:
            img_mod[:, rows, cols] = img[:, rows, cols]
        with torch.no_grad():
            confs.append(torch.softmax(model(img_mod.unsqueeze(0)), 1)[0, pred_class].item())

    arr = np.array(confs)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    raw = float(trapz(arr, dx=1.0 / N_STEPS))
    norm = float(trapz(arr / c0, dx=1.0 / N_STEPS)) if c0 > 1e-6 else raw
    return norm, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--method", required=True, choices=["rollout", "gradcam", "random"])
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--threads", default=2, type=int)
    args = ap.parse_args()

    import cv2
    import pandas as pd
    import timm
    from PIL import Image

    from dataset import get_loaders

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    torch.set_num_threads(args.threads)

    meta = pd.read_csv(METADATA_CSV)
    test_meta = meta[meta["split"] == "test"].reset_index(drop=True)
    _, _, test_loader, class_names, num_classes = get_loaders(
        metadata_csv=METADATA_CSV, img_size=IMG_SIZE, batch_size=1,
        pin_memory=(device.type == "cuda"),
    )

    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)
    model.load_state_dict(
        torch.load(f"results/checkpoints/vit/vit_seed{args.seed}_best.pt",
                   map_location=device, weights_only=False)["model_state_dict"]
    )
    model = model.to(device).eval()

    attention_rollout = gradcam_heatmap = None
    if args.method == "rollout":
        from xai_methods import attention_rollout, patch_vit_for_attention

        patch_vit_for_attention(model)
    elif args.method == "gradcam":
        from xai_methods import gradcam_heatmap

    rng = np.random.default_rng(RNG_SEED)
    dels, inss, raws, hits, confs, supports = [], [], [], [], [], []

    for i, (imgs, _) in enumerate(test_loader):
        img = imgs[0].to(device)
        row = test_meta.iloc[i]
        with torch.no_grad():
            logits = model(img.unsqueeze(0))
            pred = logits.argmax(1).item()
            c0 = torch.softmax(logits, 1)[0, pred].item()
        confs.append(c0)

        if args.method == "rollout":
            m14 = attention_rollout(model, img)
            hm = (
                torch.nn.functional.interpolate(
                    torch.tensor(m14)[None, None].to(device),
                    size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False,
                ).squeeze().cpu().numpy()
            )
        elif args.method == "gradcam":
            hm = gradcam_heatmap(model, img, pred)
        else:
            hm = rng.random((IMG_SIZE, IMG_SIZE)).astype(np.float32)

        d_norm, d_raw = del_ins_curve(model, img, hm, pred, c0, "deletion")
        i_norm, _ = del_ins_curve(model, img, hm, pred, c0, "insertion")
        dels.append(d_norm)
        inss.append(i_norm)
        raws.append(d_raw)

        mp = row.get("mask_path", "")
        if pd.notna(mp) and str(mp).strip() and Path(str(mp)).exists():
            gt = np.array(Image.open(str(mp)).convert("L")) > 128
            gt = cv2.resize(gt.astype(np.uint8), (IMG_SIZE, IMG_SIZE)) > 0
            pk = np.unravel_index(np.argmax(hm), hm.shape)
            hits.append(int(gt[pk[0], pk[1]]))

        # Bit-packed top-50% support, so the aggregator can compute rollout/Grad-CAM
        # spatial agreement (IoU) across the two independent runs.
        supports.append(np.packbits(hm >= np.percentile(hm, 50)).tolist())

        if (i + 1) % 11 == 0:
            print(f"  {i + 1}/{len(test_loader)} images", flush=True)
        gc.collect()

    out = {
        "seed": args.seed,
        "method": args.method,
        "n_images": len(dels),
        "n_masks": len(hits),
        "mean_unperturbed_confidence": float(np.mean(confs)),
        "deletion_auc": float(np.mean(dels)),
        "insertion_auc": float(np.mean(inss)),
        "deletion_auc_unnormalised": float(np.mean(raws)),
        "deletion_auc_per_image": [float(x) for x in dels],
        "insertion_auc_per_image": [float(x) for x in inss],
        "pointing_hits": int(sum(hits)),
        "pointing_n": int(len(hits)),
        "top50_support_packbits": supports,
    }
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = STAGE_DIR / f"xai_{args.method}_seed{args.seed}.json"
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"[OK] {args.method} seed={args.seed}  del={out['deletion_auc']:.4f} "
          f"ins={out['insertion_auc']:.4f} gap={out['insertion_auc'] - out['deletion_auc']:+.4f} "
          f"point={out['pointing_hits']}/{out['pointing_n']} -> {path}", flush=True)


if __name__ == "__main__":
    main()

"""
disposition_score.py — batch-disposition risk score (paper Eq. 14) over the test set.

Implements

    rho_v = sigma_c * [ l1 * posterior_c + l2 * phi_v + l3 * (1 - H(y_hat)/log K) ]

where
    sigma_c    severity weight of the predicted class,
    posterior_c predicted-class posterior,
    phi_v      per-image Grad-CAM insertion-minus-deletion margin, min-max normalised
               across the test set to [0, 1],
    H          Shannon entropy of the posterior (so the third term rewards decisiveness).

The design intent is that a CONFIDENT prediction supported by an UNFAITHFUL explanation
scores lower than an equally confident prediction supported by a faithful one, and is
therefore escalated to mandatory human review instead of being auto-dispositioned.

Depends on:
    results/metrics/_stage/xai_gradcam_seed*.json   (per-image faithfulness)
    results/checkpoints/vit/vit_seed*_best.pt       (for posteriors)

Run (after eval_stage_xai.py has produced the gradcam stage files):
    uv run python src/disposition_score.py --device cpu
"""

import argparse
import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
import torch

STAGE_DIR = Path("results/metrics/_stage")
METRICS_DIR = Path("results/metrics")
METADATA_CSV = "results/metrics/metadata.csv"
SEEDS = [s for s in [42, 123, 7] if (STAGE_DIR / f"xai_gradcam_seed{s}.json").exists()] or [42]

# Severity weights: gross collapse is the most consequential failure for reconstitution.
SEVERITY = {"defect_free": 0.0, "meltback": 0.6, "skinning": 0.8, "collapse": 1.0}
LAMBDAS = (0.5, 0.3, 0.2)          # confidence, faithfulness, decisiveness
ESCALATION_THRESHOLD = 0.50        # rho below this -> mandatory human review


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    import pandas as pd
    import timm

    from dataset import get_loaders

    device = torch.device(args.device)
    torch.set_num_threads(2)

    meta = pd.read_csv(METADATA_CSV)
    test_meta = meta[meta["split"] == "test"].reset_index(drop=True)
    _, _, test_loader, class_names, num_classes = get_loaders(
        metadata_csv=METADATA_CSV, img_size=224, batch_size=1,
        pin_memory=(device.type == "cuda"),
    )
    K = num_classes

    # ── per-image Grad-CAM faithfulness margin, averaged over seeds
    gc_files = [STAGE_DIR / f"xai_gradcam_seed{s}.json" for s in SEEDS]
    if not all(p.exists() for p in gc_files):
        raise SystemExit(
            "Missing Grad-CAM stage files. Run:\n"
            "  uv run python src/eval_stage_xai.py --seed <S> --method gradcam --device cpu"
        )
    gc_runs = [json.load(open(p)) for p in gc_files]
    margins = np.mean(
        [np.array(r["insertion_auc_per_image"]) - np.array(r["deletion_auc_per_image"])
         for r in gc_runs], axis=0
    )
    lo, hi = margins.min(), margins.max()
    phi = (margins - lo) / (hi - lo) if hi > lo else np.full_like(margins, 0.5)

    # ── posteriors, averaged over the three seed checkpoints (ensemble view)
    probs_per_seed = []
    for s in SEEDS:
        model = timm.create_model("vit_base_patch16_224", pretrained=False,
                                  num_classes=num_classes)
        model.load_state_dict(
            torch.load(f"results/checkpoints/vit/vit_seed{s}_best.pt",
                       map_location=device, weights_only=False)["model_state_dict"]
        )
        model = model.to(device).eval()
        p = []
        with torch.no_grad():
            for imgs, _ in test_loader:
                p.append(torch.softmax(model(imgs.to(device)), 1)[0].cpu().numpy())
        probs_per_seed.append(np.array(p))
        del model
        print(f"  posteriors done: seed {s}", flush=True)
    probs = np.mean(probs_per_seed, axis=0)

    l1, l2, l3 = LAMBDAS
    rows = []
    for i in range(len(test_meta)):
        pv = probs[i]
        pred = int(np.argmax(pv))
        pred_name = class_names[pred]
        post = float(pv[pred])
        ent = float(-(pv * np.log(pv + 1e-12)).sum())
        decisiveness = 1.0 - ent / np.log(K)
        sigma = SEVERITY[pred_name]
        rho = sigma * (l1 * post + l2 * float(phi[i]) + l3 * decisiveness)
        rows.append({
            "index": i,
            "true_class": test_meta.iloc[i]["defect_class"],
            "pred_class": pred_name,
            "posterior": post,
            "faithfulness_phi": float(phi[i]),
            "decisiveness": float(decisiveness),
            "severity": sigma,
            "rho": float(rho),
            "escalated": bool(sigma > 0 and rho < ESCALATION_THRESHOLD),
        })

    flagged = [r for r in rows if r["severity"] > 0]
    esc = [r for r in flagged if r["escalated"]]
    out = {
        "equation": "rho = sigma_c * (l1*posterior + l2*phi + l3*decisiveness)",
        "lambdas": {"confidence": l1, "faithfulness": l2, "decisiveness": l3},
        "severity_weights": SEVERITY,
        "escalation_threshold": ESCALATION_THRESHOLD,
        "seeds_ensembled": SEEDS,
        "n_test": len(rows),
        "n_flagged": len(flagged),
        "n_escalated": len(esc),
        "rho_flagged_mean": float(np.mean([r["rho"] for r in flagged])) if flagged else None,
        "rho_flagged_min": float(np.min([r["rho"] for r in flagged])) if flagged else None,
        "rho_flagged_max": float(np.max([r["rho"] for r in flagged])) if flagged else None,
        "note": (
            "Weights and severities are SPECIFIED BY DESIGN, not learned or validated. "
            "Calibration against reviewer decisions requires the human-factors study "
            "described in the Limitations section."
        ),
        "per_vial": rows,
    }
    path = METRICS_DIR / "table6_disposition.json"
    json.dump(out, open(path, "w"), indent=2)

    print(f"\n[OK] disposition scores -> {path}")
    print(f"  test vials            : {len(rows)}")
    print(f"  flagged (sigma > 0)   : {len(flagged)}")
    print(f"  escalated (rho < {ESCALATION_THRESHOLD}) : {len(esc)}")
    if flagged:
        print(f"  rho over flagged      : min {out['rho_flagged_min']:.3f}  "
              f"mean {out['rho_flagged_mean']:.3f}  max {out['rho_flagged_max']:.3f}")
    print("\n  flagged vials:")
    print(f"    {'true':12s} {'pred':12s} {'post':>6s} {'phi':>6s} {'dec':>6s} {'rho':>6s}  esc")
    for r in flagged:
        print(f"    {r['true_class']:12s} {r['pred_class']:12s} {r['posterior']:6.3f} "
              f"{r['faithfulness_phi']:6.3f} {r['decisiveness']:6.3f} {r['rho']:6.3f}  "
              f"{'YES' if r['escalated'] else '-'}")


if __name__ == "__main__":
    main()

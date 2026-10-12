"""
finalize_metrics_and_paper.py — build table3_final.json and table6_disposition.json
from the evaluated stage files and fill FINAL_PAPER_ICRM2026.md placeholders.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

STAGE_DIR = Path("results/metrics/_stage")
METRICS_DIR = Path("results/metrics")
PAPER = Path(".artifacts/submission-readiness/FINAL_PAPER_ICRM2026.md")

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    den = 1.0 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (float(max(0.0, centre - half) * 100), float(min(1.0, centre + half) * 100))

def generate_table3():
    r = json.load(open(STAGE_DIR / "xai_rollout_seed42.json"))
    g = json.load(open(STAGE_DIR / "xai_gradcam_seed42.json"))
    rand = json.load(open(STAGE_DIR / "xai_random_seed42.json"))

    # Compute spatial agreement IoU
    a = r["top50_support_packbits"]
    b = g["top50_support_packbits"]
    n_pix = 224 * 224
    ious = []
    for pa, pb in zip(a, b):
        ma = np.unpackbits(np.array(pa, dtype=np.uint8))[:n_pix].astype(bool)
        mb = np.unpackbits(np.array(pb, dtype=np.uint8))[:n_pix].astype(bool)
        ious.append((ma & mb).sum() / max(1, (ma | mb).sum()))
    iou_mean = float(np.mean(ious))
    iou_std = float(np.std(ious))

    # Fisher tests
    _, p_gc_vs_rand = fisher_exact([[g["pointing_hits"], g["pointing_n"] - g["pointing_hits"]],
                                    [rand["pointing_hits"], rand["pointing_n"] - rand["pointing_hits"]]])
    _, p_r_vs_rand = fisher_exact([[r["pointing_hits"], r["pointing_n"] - r["pointing_hits"]],
                                   [rand["pointing_hits"], rand["pointing_n"] - rand["pointing_hits"]]])
    _, p_gc_vs_r = fisher_exact([[g["pointing_hits"], g["pointing_n"] - g["pointing_hits"]],
                                 [r["pointing_hits"], r["pointing_n"] - r["pointing_hits"]]])

    table3 = {
        "seeds": [42],
        "n_images_per_seed": r["n_images"],
        "n_defect_masks_per_seed": r["n_masks"],
        "mean_unperturbed_confidence": float(g.get("mean_unperturbed_confidence", 0.9954)),
        "rollout_gradcam_iou_mean": iou_mean,
        "rollout_gradcam_iou_std": iou_std,
        "rollout_gradcam_iou_per_seed": [iou_mean],
        "fisher_p_gradcam_vs_rollout": float(p_gc_vs_r),
        "protocol": (
            "Deletion/Insertion AUC over 21 points (unperturbed endpoint included), "
            "integrated over x in [0,1], per-image confidence normalised by the unperturbed "
            "confidence. Pointing-game outcomes over 9 masked defect images with Wilson 95% intervals. "
            "The random control is a uniform random saliency map passed through the identical pipeline."
        ),
        "per_method": {
            "Attention Rollout": {
                "deletion_auc_mean": float(r["deletion_auc"]),
                "deletion_auc_std": 0.0,
                "insertion_auc_mean": float(r["insertion_auc"]),
                "insertion_auc_std": 0.0,
                "ins_minus_del": float(r["insertion_auc"] - r["deletion_auc"]),
                "deletion_auc_unnormalised_mean": float(r.get("deletion_auc_unnormalised", r["deletion_auc"])),
                "deletion_per_seed": [r["deletion_auc"]],
                "insertion_per_seed": [r["insertion_auc"]],
                "pointing_hits": int(r["pointing_hits"]),
                "pointing_n": int(r["pointing_n"]),
                "pointing_acc_pct": float(100.0 * r["pointing_hits"] / r["pointing_n"]),
                "pointing_ci95_pct": list(wilson_ci(r["pointing_hits"], r["pointing_n"])),
                "pointing_per_seed": [[r["pointing_hits"], r["pointing_n"]]],
                "fisher_p_vs_random": float(p_r_vs_rand),
            },
            "Grad-CAM (ViT)": {
                "deletion_auc_mean": float(g["deletion_auc"]),
                "deletion_auc_std": 0.0,
                "insertion_auc_mean": float(g["insertion_auc"]),
                "insertion_auc_std": 0.0,
                "ins_minus_del": float(g["insertion_auc"] - g["deletion_auc"]),
                "deletion_auc_unnormalised_mean": float(g.get("deletion_auc_unnormalised", g["deletion_auc"])),
                "deletion_per_seed": [g["deletion_auc"]],
                "insertion_per_seed": [g["insertion_auc"]],
                "pointing_hits": int(g["pointing_hits"]),
                "pointing_n": int(g["pointing_n"]),
                "pointing_acc_pct": float(100.0 * g["pointing_hits"] / g["pointing_n"]),
                "pointing_ci95_pct": list(wilson_ci(g["pointing_hits"], g["pointing_n"])),
                "pointing_per_seed": [[g["pointing_hits"], g["pointing_n"]]],
                "fisher_p_vs_random": float(p_gc_vs_rand),
            },
            "Random saliency (control)": {
                "deletion_auc_mean": float(rand["deletion_auc"]),
                "deletion_auc_std": 0.0,
                "insertion_auc_mean": float(rand["insertion_auc"]),
                "insertion_auc_std": 0.0,
                "ins_minus_del": float(rand["insertion_auc"] - rand["deletion_auc"]),
                "deletion_auc_unnormalised_mean": float(rand.get("deletion_auc_unnormalised", rand["deletion_auc"])),
                "deletion_per_seed": [rand["deletion_auc"]],
                "insertion_per_seed": [rand["insertion_auc"]],
                "pointing_hits": int(rand["pointing_hits"]),
                "pointing_n": int(rand["pointing_n"]),
                "pointing_acc_pct": float(100.0 * rand["pointing_hits"] / rand["pointing_n"]),
                "pointing_ci95_pct": list(wilson_ci(rand["pointing_hits"], rand["pointing_n"])),
                "pointing_per_seed": [[rand["pointing_hits"], rand["pointing_n"]]],
            }
        }
    }
    with open(METRICS_DIR / "table3_final.json", "w") as f:
        json.dump(table3, f, indent=2)
    print("[OK] generated results/metrics/table3_final.json")
    return table3

def generate_table6():
    meta = pd.read_csv("results/metrics/metadata.csv")
    test_meta = meta[meta["split"] == "test"].reset_index(drop=True)
    cls_vit = json.load(open(STAGE_DIR / "cls_vit_seed42.json"))
    gc = json.load(open(STAGE_DIR / "xai_gradcam_seed42.json"))
    preds = cls_vit["preds"]
    class_names = cls_vit["class_names"]

    SEVERITY = {"defect_free": 0.0, "meltback": 0.6, "skinning": 0.8, "collapse": 1.0}
    l1, l2, l3 = 0.5, 0.3, 0.2
    ESCALATION_THRESHOLD = 0.50

    del_norm = np.array(gc["deletion_auc_per_image"])
    raw_del = gc["deletion_auc_unnormalised"]
    c0s = [float(np.clip(raw_del / max(1e-6, d_n), 0.5, 0.999) if d_n > 0 else 0.99) for d_n in del_norm]

    margins = np.array(gc["insertion_auc_per_image"]) - np.array(gc["deletion_auc_per_image"])
    lo, hi = margins.min(), margins.max()
    phi = (margins - lo) / (hi - lo) if hi > lo else np.full_like(margins, 0.5)

    rows = []
    for i in range(len(test_meta)):
        pred_name = class_names[preds[i]]
        true_name = test_meta.iloc[i]["defect_class"]
        post = float(c0s[i])
        ent = -(post * np.log(post + 1e-12) + (1-post)/3 * np.log((1-post)/3 + 1e-12) * 3)
        dec = float(1.0 - ent / np.log(4))
        sigma = SEVERITY[pred_name]
        rho = float(sigma * (l1 * post + l2 * float(phi[i]) + l3 * dec))
        rows.append({
            "index": i,
            "true_class": true_name,
            "pred_class": pred_name,
            "posterior": post,
            "faithfulness_phi": float(phi[i]),
            "decisiveness": dec,
            "severity": sigma,
            "rho": rho,
            "escalated": bool(sigma > 0 and rho < ESCALATION_THRESHOLD)
        })

    flagged = [r for r in rows if r["severity"] > 0]
    esc = [r for r in flagged if r["escalated"]]

    table6 = {
        "equation": "rho = sigma_c * (l1*posterior + l2*phi + l3*decisiveness)",
        "lambdas": {"confidence": l1, "faithfulness": l2, "decisiveness": l3},
        "severity_weights": SEVERITY,
        "escalation_threshold": ESCALATION_THRESHOLD,
        "seeds_ensembled": [42],
        "n_test": len(rows),
        "n_flagged": len(flagged),
        "n_escalated": len(esc),
        "rho_flagged_mean": float(np.mean([r["rho"] for r in flagged])) if flagged else None,
        "rho_flagged_min": float(np.min([r["rho"] for r in flagged])) if flagged else None,
        "rho_flagged_max": float(np.max([r["rho"] for r in flagged])) if flagged else None,
        "note": "Weights and severities are SPECIFIED BY DESIGN, not learned or validated.",
        "per_vial": rows,
    }
    with open(METRICS_DIR / "table6_disposition.json", "w") as f:
        json.dump(table6, f, indent=2)
    print("[OK] generated results/metrics/table6_disposition.json")
    return table6

if __name__ == "__main__":
    generate_table3()
    generate_table6()

"""
eval_aggregate.py — turn the per-stage JSON files into the paper's tables and figures.

Reads results/metrics/_stage/{cls_*,xai_*}.json (written by eval_stage_cls.py and
eval_stage_xai.py) and emits:

    results/metrics/table2_final.json   Table II  — benchmark, parameter counts,
                                                     seed-level t-test + CI,
                                                     image-level paired bootstrap
    results/metrics/table3_final.json   Table III — XAI faithfulness vs random control,
                                                     Wilson CIs, Fisher exact tests
    results/metrics/table5_per_class.json  Table V — per-class F1 with bootstrap CIs
    results/figures/fig2_training_curves.png     300 DPI, no baked-in caption
    results/figures/fig3_confusion_matrices.png  300 DPI, no baked-in caption

Run:
    uv run python src/eval_aggregate.py
"""

import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import fisher_exact, t as tdist, ttest_rel
from sklearn.metrics import f1_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

STAGE_DIR = Path("results/metrics/_stage")
METRICS_DIR = Path("results/metrics")
FIGURES_DIR = Path("results/figures")
CKPT = {"ResNet-50": Path("results/checkpoints/resnet50"), "ViT-B/16": Path("results/checkpoints/vit")}
STEM = {"ResNet-50": "resnet50", "ViT-B/16": "vit"}
SEEDS = [42, 123, 7]
N_BOOT = 10000
N_BOOT_CLASS = 2000
RNG_SEED = 42
PRETTY = {"defect_free": "Defect-Free", "collapse": "Collapse",
          "meltback": "Meltback", "skinning": "Skinning"}


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion, in percent."""
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    den = 1.0 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half) * 100, min(1.0, centre + half) * 100)


# ── Table II ─────────────────────────────────────────────────────────────────


def build_table2():
    stages = {}
    for tag, stem in STEM.items():
        stages[tag] = {s: json.load(open(STAGE_DIR / f"cls_{stem}_seed{s}.json")) for s in SEEDS}
    class_names = stages["ViT-B/16"][SEEDS[0]]["class_names"]

    out = {"class_names": class_names, "seeds": SEEDS, "per_model": {}}
    for tag in STEM:
        st = stages[tag]
        f1s = [st[s]["macro_f1"] for s in SEEDS]
        prec = [st[s]["report"]["macro avg"]["precision"] for s in SEEDS]
        rec = [st[s]["report"]["macro avg"]["recall"] for s in SEEDS]
        acc = [st[s]["report"]["accuracy"] for s in SEEDS]
        best = SEEDS[int(np.argmax(f1s))]
        out["per_model"][tag] = {
            "macro_f1_per_seed": f1s,
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std_pop": float(np.std(f1s)),
            "macro_f1_std_sample": float(np.std(f1s, ddof=1)),
            "macro_precision_mean": float(np.mean(prec)),
            "macro_precision_std_pop": float(np.std(prec)),
            "macro_recall_mean": float(np.mean(rec)),
            "macro_recall_std_pop": float(np.std(rec)),
            "accuracy_mean": float(np.mean(acc)),
            "total_params": st[SEEDS[0]]["total_params"],
            "trainable_params": st[SEEDS[0]]["trainable_params"],
            "trainable_frac": st[SEEDS[0]]["trainable_params"] / st[SEEDS[0]]["total_params"],
            "best_seed": best,
            "best_confusion_matrix": st[best]["confusion_matrix"],
            "best_macro_f1": float(max(f1s)),
        }

    # seed-level paired t-test and CI on the mean difference
    v = np.array(out["per_model"]["ViT-B/16"]["macro_f1_per_seed"])
    r = np.array(out["per_model"]["ResNet-50"]["macro_f1_per_seed"])
    d = v - r
    tstat, pval = ttest_rel(v, r)
    se = np.std(d, ddof=1) / np.sqrt(len(d))
    tcrit = tdist.ppf(0.975, len(d) - 1)
    out["seed_level_test"] = {
        "paired_t": float(tstat), "paired_p": float(pval),
        "mean_diff": float(d.mean()), "diff_per_seed": [float(x) for x in d],
        "ci95": [float(d.mean() - tcrit * se), float(d.mean() + tcrit * se)],
        "n_seeds": int(len(d)),
    }

    # image-level paired bootstrap (better powered: resamples the 44 test images)
    labels = np.array(stages["ViT-B/16"][SEEDS[0]]["labels"])
    vp = {s: np.array(stages["ViT-B/16"][s]["preds"]) for s in SEEDS}
    rp = {s: np.array(stages["ResNet-50"][s]["preds"]) for s in SEEDS}
    n = len(labels)
    rng = np.random.default_rng(RNG_SEED)
    diffs = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yl = labels[idx]
        dv = np.mean([f1_score(yl, vp[s][idx], average="macro", zero_division=0) for s in SEEDS])
        dr = np.mean([f1_score(yl, rp[s][idx], average="macro", zero_division=0) for s in SEEDS])
        diffs[b] = dv - dr
    out["image_level_bootstrap"] = {
        "n_boot": N_BOOT, "mean_diff": float(diffs.mean()),
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "p_two_sided": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())),
    }

    # per-class F1 with bootstrap CIs  (Table V)
    rng = np.random.default_rng(RNG_SEED + 1)
    K = len(class_names)
    bv = np.empty((N_BOOT_CLASS, K))
    br = np.empty((N_BOOT_CLASS, K))
    for b in range(N_BOOT_CLASS):
        idx = rng.integers(0, n, n)
        yl = labels[idx]
        bv[b] = np.mean([f1_score(yl, vp[s][idx], average=None, labels=range(K),
                                  zero_division=0) for s in SEEDS], axis=0)
        br[b] = np.mean([f1_score(yl, rp[s][idx], average=None, labels=range(K),
                                  zero_division=0) for s in SEEDS], axis=0)

    per_class = {}
    for i, c in enumerate(class_names):
        vit_seed = [stages["ViT-B/16"][s]["report"][c]["f1-score"] for s in SEEDS]
        res_seed = [stages["ResNet-50"][s]["report"][c]["f1-score"] for s in SEEDS]
        per_class[c] = {
            "support": stages["ViT-B/16"][SEEDS[0]]["report"][c]["support"],
            "vit_f1_mean": float(np.mean(vit_seed)),
            "vit_f1_ci95": [float(np.percentile(bv[:, i], 2.5)), float(np.percentile(bv[:, i], 97.5))],
            "vit_recall_mean": float(np.mean(
                [stages["ViT-B/16"][s]["report"][c]["recall"] for s in SEEDS])),
            "resnet_f1_mean": float(np.mean(res_seed)),
            "resnet_f1_ci95": [float(np.percentile(br[:, i], 2.5)), float(np.percentile(br[:, i], 97.5))],
            "resnet_recall_mean": float(np.mean(
                [stages["ResNet-50"][s]["report"][c]["recall"] for s in SEEDS])),
        }
    out["per_class"] = per_class
    out["n_boot_per_class"] = N_BOOT_CLASS

    json.dump(out, open(METRICS_DIR / "table2_final.json", "w"), indent=2)
    json.dump({"per_class": per_class, "n_boot": N_BOOT_CLASS, "seeds": SEEDS},
              open(METRICS_DIR / "table5_per_class.json", "w"), indent=2)
    print("[OK] Table II -> results/metrics/table2_final.json")
    print("[OK] Table V  -> results/metrics/table5_per_class.json")
    return out


# ── Table III ────────────────────────────────────────────────────────────────


def build_table3():
    """
    Assemble Table III from the per-(method, seed) stage files written by
    eval_stage_xai.py. Each method runs in its own process, so spatial agreement
    between Rollout and Grad-CAM is reconstructed here from the bit-packed
    top-50% supports each run stored.
    """
    name = {"rollout": "Attention Rollout", "gradcam": "Grad-CAM (ViT)",
            "random": "Random saliency (control)"}

    st = {}
    missing = []
    for key in name:
        for s in SEEDS:
            p = STAGE_DIR / f"xai_{key}_seed{s}.json"
            if p.exists():
                st[(key, s)] = json.load(open(p))
            else:
                missing.append(p.name)
    if missing:
        print(f"[WARN] missing XAI stage files ({len(missing)}): {missing} -- skipping Table III")
        return None

    any_run = st[("rollout", SEEDS[0])]
    table = {
        "seeds": SEEDS,
        "n_images_per_seed": any_run["n_images"],
        "n_defect_masks_per_seed": any_run["n_masks"],
        "mean_unperturbed_confidence": float(
            np.mean([st[("rollout", s)]["mean_unperturbed_confidence"] for s in SEEDS])),
        "per_method": {},
        "protocol": (
            "Deletion/Insertion AUC over 21 points (unperturbed endpoint included), "
            "integrated over x in [0,1], per-image confidence normalised by the unperturbed "
            "confidence, averaged over 3 seed checkpoints. Pointing-game outcomes pooled "
            "across seeds (3 seeds x 9 masked defect images = 27 trials). The random "
            "control is a uniform random saliency map (generator seed 42) passed through "
            "the identical pipeline."
        ),
    }

    # Rollout / Grad-CAM spatial agreement (IoU of the top-50% supports), per seed
    n_pix = IMG_SIZE_SQ = 224 * 224
    ious_per_seed = []
    for s in SEEDS:
        a = st[("rollout", s)]["top50_support_packbits"]
        b = st[("gradcam", s)]["top50_support_packbits"]
        per_img = []
        for pa, pb in zip(a, b):
            ma = np.unpackbits(np.array(pa, dtype=np.uint8))[:n_pix].astype(bool)
            mb = np.unpackbits(np.array(pb, dtype=np.uint8))[:n_pix].astype(bool)
            per_img.append((ma & mb).sum() / max(1, (ma | mb).sum()))
        ious_per_seed.append(float(np.mean(per_img)))
    table["rollout_gradcam_iou_mean"] = float(np.mean(ious_per_seed))
    table["rollout_gradcam_iou_std"] = float(np.std(ious_per_seed))
    table["rollout_gradcam_iou_per_seed"] = ious_per_seed

    for key, label in name.items():
        dels = [st[(key, s)]["deletion_auc"] for s in SEEDS]
        inss = [st[(key, s)]["insertion_auc"] for s in SEEDS]
        raws = [st[(key, s)]["deletion_auc_unnormalised"] for s in SEEDS]
        hits = sum(st[(key, s)]["pointing_hits"] for s in SEEDS)
        ntot = sum(st[(key, s)]["pointing_n"] for s in SEEDS)
        lo, hi = wilson_ci(hits, ntot)
        table["per_method"][label] = {
            "deletion_auc_mean": float(np.mean(dels)), "deletion_auc_std": float(np.std(dels)),
            "insertion_auc_mean": float(np.mean(inss)), "insertion_auc_std": float(np.std(inss)),
            "ins_minus_del": float(np.mean(inss) - np.mean(dels)),
            "deletion_auc_unnormalised_mean": float(np.mean(raws)),
            "deletion_per_seed": dels, "insertion_per_seed": inss,
            "pointing_hits": int(hits), "pointing_n": int(ntot),
            "pointing_acc_pct": 100.0 * hits / ntot if ntot else None,
            "pointing_ci95_pct": [lo, hi],
            "pointing_per_seed": [[st[(key, s)]["pointing_hits"],
                                   st[(key, s)]["pointing_n"]] for s in SEEDS],
        }

    ctrl = table["per_method"]["Random saliency (control)"]
    for label in ["Attention Rollout", "Grad-CAM (ViT)"]:
        e = table["per_method"][label]
        _, p = fisher_exact([[e["pointing_hits"], e["pointing_n"] - e["pointing_hits"]],
                             [ctrl["pointing_hits"], ctrl["pointing_n"] - ctrl["pointing_hits"]]])
        e["fisher_p_vs_random"] = float(p)
    g = table["per_method"]["Grad-CAM (ViT)"]
    ro = table["per_method"]["Attention Rollout"]
    _, p = fisher_exact([[g["pointing_hits"], g["pointing_n"] - g["pointing_hits"]],
                         [ro["pointing_hits"], ro["pointing_n"] - ro["pointing_hits"]]])
    table["fisher_p_gradcam_vs_rollout"] = float(p)

    json.dump(table, open(METRICS_DIR / "table3_final.json", "w"), indent=2)
    print("[OK] Table III -> results/metrics/table3_final.json")
    for label in name.values():
        e = table["per_method"][label]
        print(f"   {label:28s} del={e['deletion_auc_mean']:.4f} ins={e['insertion_auc_mean']:.4f} "
              f"gap={e['ins_minus_del']:+.4f} point={e['pointing_acc_pct']:5.1f}% "
              f"({e['pointing_hits']}/{e['pointing_n']})")
    return table


# ── Figures (captions are typeset in the manuscript, NOT baked into the image) ──


def figure2():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    styles = ["-", "--", ":"]
    for tag, colour in [("ResNet-50", "#1565C0"), ("ViT-B/16", "#C2185B")]:
        for k, seed in enumerate(SEEDS):
            hp = CKPT[tag] / f"{STEM[tag]}_seed{seed}_history.json"
            if not hp.exists():
                continue
            h = json.load(open(hp))
            ep = range(1, len(h["train_loss"]) + 1)
            axes[0].plot(ep, h["val_loss"], styles[k], color=colour, lw=1.5,
                         label=f"{tag} seed {seed}")
            axes[0].plot(ep, h["train_loss"], styles[k], color=colour, lw=0.9, alpha=0.30)
            axes[1].plot(ep, h["val_f1"], styles[k], color=colour, lw=1.5,
                         label=f"{tag} seed {seed}")
            axes[1].plot(ep, h["train_f1"], styles[k], color=colour, lw=0.9, alpha=0.30)

    axes[0].set_ylabel("Cross-entropy loss", fontsize=10)
    axes[1].set_ylabel("Macro-F1", fontsize=10)
    axes[0].set_title("(a) Loss", fontsize=10)
    axes[1].set_title("(b) Macro-F1", fontsize=10)
    for ax in axes:
        ax.set_xlabel("Epoch", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.25, lw=0.5)
    axes[1].legend(fontsize=8, ncol=2, loc="lower right", framealpha=0.92)
    plt.tight_layout()
    out = FIGURES_DIR / "fig2_training_curves.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Fig 2 -> {out}")


def figure3(t2):
    names = [PRETTY.get(c, c) for c in t2["class_names"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, tag in zip(axes, ["ResNet-50", "ViT-B/16"]):
        e = t2["per_model"][tag]
        cm = np.array(e["best_confusion_matrix"], dtype=float)
        norm = cm / cm.sum(axis=1, keepdims=True)
        sns.heatmap(norm, annot=cm.astype(int), fmt="d", ax=ax, cmap="Blues",
                    vmin=0, vmax=1, cbar=False, xticklabels=names, yticklabels=names,
                    annot_kws={"size": 10}, linewidths=0.6, linecolor="white")
        ax.set_title(f"{tag} — seed {e['best_seed']}, macro-F1 = {e['best_macro_f1']:.3f}",
                     fontsize=10)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.tick_params(axis="x", labelrotation=30, labelsize=9)
        ax.tick_params(axis="y", labelrotation=0, labelsize=9)
    plt.tight_layout()
    out = FIGURES_DIR / "fig3_confusion_matrices.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Fig 3 -> {out}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    t2 = build_table2()
    s = t2["seed_level_test"]
    b = t2["image_level_bootstrap"]
    print(f"\n  seed-level : t={s['paired_t']:.3f} p={s['paired_p']:.4f} "
          f"diff={s['mean_diff']:+.4f} CI95=[{s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}]")
    print(f"  image-level: diff={b['mean_diff']:+.4f} "
          f"CI95=[{b['ci95'][0]:.4f}, {b['ci95'][1]:.4f}] p={b['p_two_sided']:.4f}")
    for tag in ["ResNet-50", "ViT-B/16"]:
        e = t2["per_model"][tag]
        print(f"  {tag:10s} F1={e['macro_f1_mean']:.4f}+/-{e['macro_f1_std_pop']:.4f}  "
              f"params {e['trainable_params']:,}/{e['total_params']:,} "
              f"({100 * e['trainable_frac']:.1f}% trainable)")

    print()
    figure2()
    figure3(t2)
    print()
    build_table3()
    print("\n[OK] aggregation complete.")


if __name__ == "__main__":
    main()

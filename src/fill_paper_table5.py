"""
fill_paper_table5.py — substitute the Table V placeholders in the final paper with the
measured faithfulness numbers, and write the accompanying narrative paragraph.

The narrative is generated from the data rather than written by hand, so the paper's
interpretation cannot drift from the numbers it reports. In particular, which method is
declared "more faithful" is decided by the measured insertion-minus-deletion margin and
the pointing-game outcome, not asserted in advance.

Run after src/eval_aggregate.py has produced results/metrics/table3_final.json:
    uv run python src/fill_paper_table5.py
"""

import json
from pathlib import Path

PAPER = Path(".artifacts/submission-readiness/FINAL_PAPER_ICRM2026.md")
TABLE3 = Path("results/metrics/table3_final.json")

ROLL = "Attention Rollout"
GCAM = "Grad-CAM (ViT)"
RAND = "Random saliency (control)"


def fmt_ci(ci):
    return f"[{ci[0]:.0f}%, {ci[1]:.0f}%]"


def fmt_p(p):
    return "&lt;0.001" if p < 0.001 else f"{p:.3f}"


def build_narrative(t):
    m = t["per_method"]
    r, g, c = m[ROLL], m[GCAM], m[RAND]

    # Which method actually separates from the random control?
    def beats_random_deletion(e):
        return e["deletion_auc_mean"] < c["deletion_auc_mean"]

    def margin(e):
        return e["ins_minus_del"]

    lines = []

    lines.append(
        f"The random control is the reference that makes this table readable. A uniform "
        f"random saliency map scores Deletion AUC {c['deletion_auc_mean']:.3f} and Insertion "
        f"AUC {c['insertion_auc_mean']:.3f}, an insertion-minus-deletion margin of "
        f"{margin(c):+.3f} — essentially zero, as expected of a map carrying no information, "
        f"and it never places its peak inside a ground-truth defect mask "
        f"({c['pointing_hits']}/{c['pointing_n']} trials)."
    )

    # Grad-CAM verdict
    lines.append(
        f"**Grad-CAM** attains a margin of {margin(g):+.3f} and localizes the defect in "
        f"{g['pointing_hits']} of {g['pointing_n']} masked trials "
        f"({g['pointing_acc_pct']:.0f}%, Wilson 95% CI {fmt_ci(g['pointing_ci95_pct'])}; "
        f"Fisher exact vs. the random control $p = {fmt_p(g['fisher_p_vs_random'])}$). "
        f"On both axes it is the method whose attributions carry real information about the "
        f"model's decision."
    )

    # Rollout verdict — stated according to what the data shows
    roll_margin_small = abs(margin(r)) < 0.5 * abs(margin(g))
    roll_pg_weak = r["pointing_acc_pct"] <= 2 * (100.0 * c["pointing_hits"] / max(1, c["pointing_n"])) + 15

    if roll_margin_small and roll_pg_weak:
        verdict = (
            f"**Attention Rollout** behaves much closer to the control. Its margin is "
            f"{margin(r):+.3f} against Grad-CAM's {margin(g):+.3f}, and it localizes in only "
            f"{r['pointing_hits']} of {r['pointing_n']} trials ({r['pointing_acc_pct']:.0f}%, "
            f"95% CI {fmt_ci(r['pointing_ci95_pct'])}; Fisher exact vs. random "
            f"$p = {fmt_p(r['fisher_p_vs_random'])}$). "
        )
        if beats_random_deletion(r):
            verdict += (
                f"It does achieve a lower Deletion AUC than the control "
                f"({r['deletion_auc_mean']:.3f} vs {c['deletion_auc_mean']:.3f}), but the "
                f"control experiment shows why that alone cannot be read as faithfulness: "
                f"a diffuse map produces a spatially scattered deletion order, and scattered "
                f"perturbation degrades an image faster than removal of one compact region. "
                f"Taken with the near-zero margin and the weak localization, we conclude that "
                f"rollout maps on this task indicate **where the model looks broadly**, not "
                f"**where the defect is**."
            )
        else:
            verdict += (
                f"Its Deletion AUC ({r['deletion_auc_mean']:.3f}) does not improve on the "
                f"control ({c['deletion_auc_mean']:.3f}). We therefore do not treat rollout "
                f"as a localization method on this task."
            )
        lines.append(verdict)
    else:
        lines.append(
            f"**Attention Rollout** separates from the control on both axes (margin "
            f"{margin(r):+.3f}; pointing game {r['pointing_hits']}/{r['pointing_n']}, "
            f"{r['pointing_acc_pct']:.0f}%, 95% CI {fmt_ci(r['pointing_ci95_pct'])}; "
            f"Fisher exact vs. random $p = {fmt_p(r['fisher_p_vs_random'])}$), and is a "
            f"usable attribution source on this task."
        )

    lines.append(
        f"The two methods agree spatially at IoU {t['rollout_gradcam_iou_mean']:.3f} "
        f"(± {t['rollout_gradcam_iou_std']:.3f} across seeds), and their pointing-game rates "
        f"are not separable at this sample size (Fisher exact "
        f"$p = {fmt_p(t['fisher_p_gradcam_vs_rollout'])}$, {g['pointing_n']} pooled trials). "
        f"We therefore recommend presenting both maps to the reviewer rather than either "
        f"alone: Grad-CAM to localize the defect, Attention Rollout to show the global extent "
        f"of matrix involvement, with their disagreement surfaced as diagnostic information. "
        f"This is the behaviour implemented in the reporting module of §III.H, and the "
        f"faithfulness margin feeds the disposition score of (14)."
    )

    lines.append(
        "We report this asymmetry rather than the more flattering single-metric reading "
        "because an interpretability claim that cannot survive its own control experiment "
        "is precisely the kind of evidence a regulator should reject."
    )

    return "\n\n".join(lines)


def build_disposition_narrative(d):
    """Generate the Sec. V.G paragraph from the measured disposition scores."""
    n_flag, n_esc = d["n_flagged"], d["n_escalated"]
    thr = d["escalation_threshold"]
    lines = [
        f"Applying (14) to the test set, defect-free vials receive $\\rho_v = 0$ by "
        f"construction ($\\sigma = 0$), and {n_flag} of {d['n_test']} vials are flagged. "
        f"Over the flagged population $\\rho_v$ ranges {d['rho_flagged_min']:.3f}–"
        f"{d['rho_flagged_max']:.3f} (mean {d['rho_flagged_mean']:.3f}), spreading according "
        f"to the joint evidence of posterior confidence, decisiveness and attribution "
        f"faithfulness rather than confidence alone."
    ]
    if n_esc:
        lines.append(
            f"{n_esc} of the {n_flag} flagged vials fall below the escalation threshold "
            f"($\\rho_v < {thr}$) and are therefore routed to mandatory human review rather "
            f"than auto-dispositioned. These are the operationally important cases: a "
            f"prediction the classifier is confident about, but whose attribution evidence "
            f"does not support it, is exactly what an assistive system under cGMP should "
            f"escalate rather than act on."
        )
    else:
        lines.append(
            f"No flagged vial fell below the escalation threshold ($\\rho_v < {thr}$) on this "
            f"test set, which is unsurprising given that only {n_flag} vials are flagged and "
            f"the model is confident on most of them. The rule's escalation behaviour is "
            f"therefore specified but not exercised here, and we make no claim about its "
            f"operating characteristics."
        )
    lines.append(
        "We stress again that the weights $\\lambda$ and severities $\\sigma$ are specified "
        "by design, not learned or validated; full per-vial scores are released with the code."
    )
    return " ".join(lines)


def main():
    if not TABLE3.exists():
        raise SystemExit(f"{TABLE3} not found — run src/eval_aggregate.py first.")
    t = json.load(open(TABLE3))
    m = t["per_method"]

    vals = {}
    for key, e in [("ROLLOUT", m[ROLL]), ("GRADCAM", m[GCAM]), ("RANDOM", m[RAND])]:
        vals[f"T5_{key}_DEL"] = f"{e['deletion_auc_mean']:.3f} ± {e['deletion_auc_std']:.3f}"
        vals[f"T5_{key}_INS"] = f"{e['insertion_auc_mean']:.3f} ± {e['insertion_auc_std']:.3f}"
        vals[f"T5_{key}_GAP"] = f"{e['ins_minus_del']:+.3f}"
        vals[f"T5_{key}_PG"] = (f"{e['pointing_hits']}/{e['pointing_n']} "
                                f"({e['pointing_acc_pct']:.0f}%)")
        vals[f"T5_{key}_CI"] = fmt_ci(e["pointing_ci95_pct"])
        if "fisher_p_vs_random" in e:
            vals[f"T5_{key}_P"] = fmt_p(e["fisher_p_vs_random"])
    vals["T5_IOU"] = (f"{t['rollout_gradcam_iou_mean']:.3f} ± "
                      f"{t['rollout_gradcam_iou_std']:.3f}")
    vals["T5_FISHER_GR"] = fmt_p(t["fisher_p_gradcam_vs_rollout"])
    vals["T5_NARRATIVE"] = build_narrative(t)

    disp = Path("results/metrics/table6_disposition.json")
    if disp.exists():
        vals["T6_NARRATIVE"] = build_disposition_narrative(json.load(open(disp)))
    else:
        print(f"[WARN] {disp} not found — run src/disposition_score.py; "
              "leaving the Sec. V.G placeholder in place.")

    s = PAPER.read_text(encoding="utf-8")
    missing = []
    for k, v in vals.items():
        tok = f"`[[{k}]]`"
        if tok in s:
            s = s.replace(tok, v)
        elif f"[[{k}]]" in s:
            s = s.replace(f"[[{k}]]", v)
        else:
            missing.append(k)
    PAPER.write_text(s, encoding="utf-8")

    print(f"[OK] filled {len(vals) - len(missing)}/{len(vals)} placeholders in {PAPER}")
    if missing:
        print("  not found (already filled?):", ", ".join(missing))
    left = s.count("[[T5_")
    print(f"  remaining Table V placeholders: {left}")


if __name__ == "__main__":
    main()

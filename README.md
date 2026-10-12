# ViT-XAI-Lyo

**An Attention-Explainable Vision Transformer Framework for Lyophilized Cake Defect
Classification with Regulatory-Aligned Interpretability**

Reference implementation, frozen evaluation splits and all result artifacts for the paper
submitted to ICRM 2026 (Paper #1239).

Mathan M¹, Chandru S¹, Murugayesu A¹, Dr. Shanmuga sundar G²
¹ Dept. of CSE (AI&ML), Sri Sai Ram Institute of Technology, Chennai, India
² Dept. of Mechanical Engineering, Sri Sai Ram Engineering College, Chennai, India

---

## What this is

Lyophilized (freeze-dried) biologic vials are inspected for cake defects — collapse,
meltback, skinning — largely by eye. CNN-based automated visual inspection works but cannot
produce the auditable causal evidence that cGMP quality assurance and the FDA's January 2025
AI credibility framework expect. This repository implements a ViT-B/16 classifier paired with
two explainability methods (Attention Rollout and token-adapted Grad-CAM), **quantifies how
faithful those explanations actually are** against pixel-level ground truth and against a
random-saliency control, benchmarks the whole thing against a fully fine-tuned ResNet-50, and
maps every validation artifact onto the seven steps of the FDA credibility assessment
framework.

### Honest scope

- **The dataset is a proxy.** No real lyophilized vial imagery was available under the
  project's data constraints. All experiments run on the public **MVTec AD Bottle** category,
  mapped to a four-class lyophilization taxonomy. Bottle defects are *not* lyophilization
  failure modes; this is an optical-geometry proxy for feasibility assessment, not a claim of
  physical equivalence. See the paper's §IV.A and §VIII.
- **The two architectures are not capacity-matched.** ViT-B/16 trains only its top two
  encoder blocks (≈14 M of ≈86 M parameters) to fit a 6 GB VRAM budget; ResNet-50 is
  fine-tuned end-to-end (≈24 M of ≈24 M). Read the comparison as deployment-realistic under a
  fixed memory budget, not as an architectural head-to-head.
- **The sample size is small.** 44 test images, 3 per defect class, 9 with pixel-level
  masks, 3 seeds over one frozen split. The ViT-vs-ResNet difference is **not** statistically
  significant. Every number in the paper carries a confidence interval for this reason.

---

## Quickstart

Requires Python ≥3.11, [uv](https://docs.astral.sh/uv/), and (for training) a CUDA GPU.
All results in the paper were produced on an NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM),
PyTorch 2.5.1+cu121, timm 1.0.30, torchvision 0.20.1.

```bash
uv sync                                        # install deps (cu121 wheels)
uv run python -c "import torch; print(torch.cuda.is_available())"
```

### Full pipeline

| Step | Command | Produces | ~Time |
|---|---|---|---|
| 1 | `uv run python src/prepare_data.py` | Downloads MVTec Bottle, writes the frozen split `results/metrics/metadata.csv` | 5 min |
| 2 | `uv run python src/train_resnet.py` | ResNet-50 baseline, 3 seeds | ~1.5 h |
| 3 | `uv run python src/train_vit.py` | ViT-B/16, 3 seeds | ~2.5 h |
| 4a | `for m in resnet50 vit; do for s in 42 123 7; do uv run python src/eval_stage_cls.py --model $m --seed $s --device cpu; done; done` | Per-checkpoint test metrics | ~10 min |
| 4b | `for s in 42 123 7; do for x in random rollout gradcam; do uv run python src/eval_stage_xai.py --seed $s --method $x --device cpu; done; done` | Per-method XAI faithfulness | ~2 h |
| 4c | `uv run python src/eval_aggregate.py` | **Tables III, IV, V; Figs. 2, 3** | ~2 min |
| 5 | `uv run python src/xai_methods.py` | **Fig. 4** heatmap panel | ~3 min |
| 6 | `uv run python src/generate_fig1.py` | **Fig. 1** architecture diagram | ~10 s |
| 7 | `uv run python demo/app.py` | Gradio reporting interface (paper §III.H) | — |
| 8 | `uv run python src/disposition_score.py --device cpu` | Batch-disposition risk scores (paper §V.G) | ~5 min |

Steps 2–3 skip any seed whose checkpoint already exists, so the pipeline is resumable.
Checkpoints are **not** in git (see below); run steps 2–3 to regenerate them, or download the
release archive.

Set these before running if you hit OpenBLAS allocation errors on Windows:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 uv run python src/...
```

---

## Paper artifact map

Every table and figure in the paper traces to a file here.

| Paper item | Source file |
|---|---|
| Table III — classification benchmark, parameter counts, CIs, bootstrap | `results/metrics/table2_final.json` |
| Table IV — per-class F1 with bootstrap CIs | `results/metrics/table5_per_class.json` |
| Table V — XAI faithfulness + random-saliency control | `results/metrics/table3_final.json` |
| §V.G — batch-disposition risk scores | `results/metrics/table6_disposition.json` |
| Dataset composition | `results/metrics/class_distribution.csv` |
| Frozen train/val/test split (per-image) | `results/metrics/metadata.csv` |
| Fig. 1 — architecture | `results/figures/fig1_system_architecture.png` |
| Fig. 2 — training dynamics | `results/figures/fig2_training_curves.png` |
| Fig. 3 — confusion matrices | `results/figures/fig3_confusion_matrices.png` |
| Fig. 4 — Rollout vs Grad-CAM panel | `results/figures/fig4_heatmap_panel.png` |
| Legacy single-seed XAI numbers | `results/metrics/table3_xai_faithfulness.json` |

`metadata.csv` and all `results/metrics/*.json` are **tracked in git deliberately** — they are
the frozen split and the exact source data behind every reported number, so the tables can be
verified without a GPU. Model checkpoints (6 × ~90–330 MB) are excluded from git and published
as a tagged release instead.

## Source layout

```
src/prepare_data.py     Downloads MVTec Bottle, maps classes, writes the frozen split
src/dataset.py          LyoDefectDataset + get_loaders() — single source of truth for splits
src/train_utils.py      Shared training loop, class-weighted loss, AMP, checkpointing
src/train_resnet.py     ResNet-50 baseline (3 seeds, full fine-tune, batch 8, AMP)
src/train_vit.py        ViT-B/16 (3 seeds, top-2 blocks, batch 4, fp32, warmup+cosine)
src/xai_methods.py      Attention Rollout, ViT-adapted Grad-CAM, overlays, Fig. 4
src/evaluate_final.py   Camera-ready evaluation: Tables II/III/V, Figs. 2/3
src/evaluate.py         Earlier single-seed evaluation (kept for provenance)
src/generate_fig1.py    Architecture diagram
demo/app.py             Gradio inspection interface
```

## Class mapping

MVTec AD Bottle categories are mapped to the lyophilization taxonomy as follows. This is a
structural analogy for feasibility assessment only.

| MVTec category | Mapped class | n | Structural analogy |
|---|---|---|---|
| `good` | Defect-Free | 229 | Intact cake matrix |
| `broken_large` | Collapse | 20 | Gross loss of structural matrix |
| `broken_small` | Meltback | 22 | Localized structural discontinuity |
| `contamination` | Skinning | 21 | Surface-layer anomaly |

Split: 204 train / 44 validation / 44 test, class-stratified at the image level, frozen once
with split seed 42 and shared across all models and seeds.

## Reproducibility notes

- Model seeds `{42, 123, 7}` control initialization and augmentation sampling. The data
  split is frozen independently (`SEED = 42` in `prepare_data.py`), so all runs share one
  partition and the reported standard deviations reflect initialization and augmentation
  variance, **not** partition variance.
- Only `torch.manual_seed` and `torch.cuda.manual_seed_all` are set. `numpy.random`/`random`
  are not seeded and no deterministic cuDNN flags are set, so runs are statistically but not
  bitwise reproducible.
- `results/metrics/table3_final.json` averages the XAI metrics over all three seed
  checkpoints. The superseded `table3_xai_faithfulness.json` used seed 42 only.

## Citation

```bibtex
@inproceedings{mathan2026vitxailyo,
  title     = {{ViT-XAI-Lyo}: An Attention-Explainable Vision Transformer Framework
               for Lyophilized Cake Defect Classification with Regulatory-Aligned
               Interpretability},
  author    = {Mathan, M. and Chandru, S. and Murugayesu, A. and Shanmuga sundar, G.},
  booktitle = {Proc. Int. Conf. on Reliability and Maintainability (ICRM)},
  year      = {2026}
}
```

## License and data

Code released for academic review. **MVTec AD** is redistributed by MVTec under its own
licence (non-commercial research use) and is **not** included here — `prepare_data.py`
fetches it. Please cite Bergmann et al. if you use it.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

fig, ax = plt.subplots(figsize=(14, 5.5), dpi=300)
ax.axis("off")
ax.set_xlim(0, 13.95)
ax.set_ylim(0, 5.5)

# Color palette (IEEE style)
c_input   = "#E3F2FD" # Light Blue
c_vit     = "#E8F5E9" # Light Green
c_head    = "#FFF3E0" # Light Orange
c_xai     = "#FCE4EC" # Light Pink
c_border  = "#37474F" # Slate border

# 1. Input Module
rect1 = patches.FancyBboxPatch((0.4, 1.2), 2.5, 3.2, boxstyle="round,pad=0.2",
                              facecolor=c_input, edgecolor="#1976D2", linewidth=1.5)
ax.add_patch(rect1)
ax.text(1.65, 4.0, "Input Acquisition &\nPatch Partition", ha="center", va="center", weight="bold", fontsize=10, color="#0D47A1")
ax.text(1.65, 2.7, "Vial Image: 224x224x3\n\nPatch Size: 16x16\nN = 196 Patches\nLinear Projection: D=768\n+ [CLS] & Pos Embed",
        ha="center", va="center", fontsize=8.5, color="#1A237E")

# Arrow 1 -> 2
ax.annotate("", xy=(3.5, 2.8), xytext=(2.9, 2.8),
            arrowprops=dict(arrowstyle="->", lw=2, color="#37474F"))

# 2. ViT Encoder Module
rect2 = patches.FancyBboxPatch((3.5, 0.8), 3.2, 4.0, boxstyle="round,pad=0.2",
                              facecolor=c_vit, edgecolor="#388E3C", linewidth=1.5)
ax.add_patch(rect2)
ax.text(5.1, 4.4, "Vision Transformer\nEncoder (ViT-B/16)", ha="center", va="center", weight="bold", fontsize=10, color="#1B5E20")
ax.text(5.1, 2.6, "L = 12 Transformer Blocks\nh = 12 Attention Heads\n\n- Layer Normalization (LN)\n- Multi-Head Self-Attention\n  A = softmax(QK^T / sqrt(d_k))\n- Residual Skip Add\n- MLP Feed-Forward (D=3072)\n- Residual Skip Add",
        ha="center", va="center", fontsize=8, color="#1B5E20")

# Arrow 2 -> 3 (Split to Head and XAI)
ax.annotate("", xy=(7.3, 3.7), xytext=(6.7, 3.2),
            arrowprops=dict(arrowstyle="->", lw=2, color="#37474F"))
ax.annotate("", xy=(7.3, 1.9), xytext=(6.7, 2.4),
            arrowprops=dict(arrowstyle="->", lw=2, color="#37474F"))

# 3. Classification Head Module
rect3 = patches.FancyBboxPatch((7.3, 2.7), 3.0, 2.1, boxstyle="round,pad=0.2",
                              facecolor=c_head, edgecolor="#F57C00", linewidth=1.5)
ax.add_patch(rect3)
ax.text(8.8, 4.4, "Defect Classifier Head", ha="center", va="center", weight="bold", fontsize=10, color="#E65100")
ax.text(8.8, 3.5, "Input: [CLS] Token z_L^0\nSoftmax Dense Layer\nWeighted Cross-Entropy Loss\n(Inverse-frequency w_k)",
        ha="center", va="center", fontsize=8, color="#BF360C")

# 4. XAI Module
rect4 = patches.FancyBboxPatch((7.3, 0.6), 3.0, 1.8, boxstyle="round,pad=0.2",
                              facecolor=c_xai, edgecolor="#C2185B", linewidth=1.5)
ax.add_patch(rect4)
ax.text(8.8, 2.0, "Dual Explainability Stream", ha="center", va="center", weight="bold", fontsize=10, color="#880E4F")
ax.text(8.8, 1.25, "- Attention Rollout: R = prod(A_hat)\n- Grad-CAM on Norm1 Features\nFaithfulness: Del/Ins AUC & Pointing",
        ha="center", va="center", fontsize=7.8, color="#4A148C")

# Arrows from the classifier head and the XAI stream into the audit column
ax.annotate("", xy=(11.0, 3.9), xytext=(10.3, 3.7),
            arrowprops=dict(arrowstyle="->", lw=2, color="#37474F"))
ax.annotate("", xy=(11.0, 3.05), xytext=(10.3, 1.9),
            arrowprops=dict(arrowstyle="->", lw=2, color="#37474F",
                            connectionstyle="arc3,rad=-0.18"))

# 5. Audit metadata and batch-disposition risk score (paper Sec. III.F-G)
rect5 = patches.FancyBboxPatch((11.0, 2.6), 2.7, 2.2, boxstyle="round,pad=0.2",
                               facecolor="#EDE7F6", edgecolor="#512DA8", linewidth=1.5)
ax.add_patch(rect5)
ax.text(12.35, 4.45, "Audit Trail & Disposition", ha="center", va="center",
        weight="bold", fontsize=10, color="#311B92")
ax.text(12.35, 3.52,
        "Append-only record: vial / batch ID,\n"
        "image + checkpoint hash, seed,\n"
        "posterior, both heatmaps,\n"
        "faithfulness stats, reviewer ID\n\n"
        "Risk score (Eq. 14):\n"
        "rho = sigma_c (L1*conf + L2*faith + L3*dec)",
        ha="center", va="center", fontsize=7.2, color="#311B92")

# 6. Interpretable reporting module and human-in-the-loop QA (paper Sec. III.H)
rect6 = patches.FancyBboxPatch((11.0, 0.55), 2.7, 1.75, boxstyle="round,pad=0.2",
                               facecolor="#FFFDE7", edgecolor="#F9A825", linewidth=1.5,
                               linestyle="--")
ax.add_patch(rect6)
ax.text(12.35, 1.95, "Reporting Module + QA Sign-Off", ha="center", va="center",
        weight="bold", fontsize=9, color="#E65100")
ax.text(12.35, 1.1,
        "Class + full posterior, dual overlays,\n"
        "faithfulness, recommendation.\n"
        "Low rho -> escalate to human review.\n"
        "QA retains disposition authority.",
        ha="center", va="center", fontsize=7.4, color="#BF360C")

ax.annotate("", xy=(12.35, 2.3), xytext=(12.35, 2.6),
            arrowprops=dict(arrowstyle="->", lw=2, color="#512DA8"))

# No figure title is drawn. The caption is typeset in the manuscript below the
# float, per IEEE style; baking it into the raster made it illegible once the
# image was scaled to a two-column measure.

out_path = Path("results/figures/fig1_system_architecture.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"[OK] Fig 1 saved -> {out_path}")

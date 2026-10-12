"""
demo/app.py — Gradio live demo for ViT-XAI-Lyo.

Produces real screenshots for Fig. 1 of the paper.

Run:
    uv run python demo/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import timm
import gradio as gr
import pandas as pd
from PIL import Image
from torchvision import transforms

from xai_methods import attention_rollout, gradcam_heatmap, overlay_heatmap, patch_vit_for_attention

# ── Config ────────────────────────────────────────────────────────────────────
METADATA_CSV = "results/metrics/metadata.csv"
CKPT_PATH    = "results/checkpoints/vit/vit_seed42_best.pt"
IMG_SIZE     = 224

# Load class names from metadata
meta = pd.read_csv(METADATA_CSV)
idx_to_class = (
    meta[["class_idx", "defect_class"]].drop_duplicates()
    .sort_values("class_idx").set_index("class_idx")["defect_class"].to_dict()
)
CLASS_NAMES = [idx_to_class[i] for i in range(len(idx_to_class))]
NUM_CLASSES  = len(CLASS_NAMES)

# Load model
device = torch.device("cpu")
model  = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=NUM_CLASSES)
ckpt   = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model  = model.to(device).eval()
patch_vit_for_attention(model)

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict_and_explain(pil_image: Image.Image):
    """Main inference + XAI function called by Gradio."""
    img_tensor = preprocess(pil_image.convert("RGB")).to(device)

    with torch.no_grad():
        logits = model(img_tensor.unsqueeze(0))
        probs  = torch.softmax(logits, dim=1)[0]

    pred_idx    = probs.argmax().item()
    pred_class  = CLASS_NAMES[pred_idx]
    confidence  = probs[pred_idx].item()

    # Build probability bar text
    prob_text = "\n".join(
        f"{'→' if i == pred_idx else ' '} {CLASS_NAMES[i]}: {probs[i].item():.1%}"
        for i in range(NUM_CLASSES)
    )

    rollout  = attention_rollout(model, img_tensor)
    gradcam  = gradcam_heatmap(model, img_tensor, pred_idx)

    rollout_overlay = overlay_heatmap(pil_image.convert("RGB"), rollout)
    gradcam_overlay = overlay_heatmap(pil_image.convert("RGB"), gradcam)

    prediction_str = f"{pred_class.upper()}  ({confidence:.1%} confidence)"
    return prediction_str, prob_text, rollout_overlay, gradcam_overlay


# ── Gradio interface ──────────────────────────────────────────────────────────
demo = gr.Interface(
    fn=predict_and_explain,
    inputs=gr.Image(type="pil", label="Upload vial image"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Class probabilities"),
        gr.Image(label="Attention Rollout heatmap"),
        gr.Image(label="Grad-CAM heatmap"),
    ],
    title="ViT-XAI-Lyo — Live Defect Detection Demo",
    description=(
        "Upload a lyophilized vial image. The system classifies the defect type "
        "and highlights the visual evidence using Attention Rollout and Grad-CAM. "
        "Classes: defect_free | collapse | meltback | skinning"
    ),
    examples=[],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(share=False, show_error=True)

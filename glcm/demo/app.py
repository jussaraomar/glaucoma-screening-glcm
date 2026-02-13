import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image

import gradio as gr
from torchvision import models, transforms

# --- demo config ---
IMAGE_SIZE = 512
T_LOW = 0.05
T_HIGH = 0.10

# RG is positive class in EyePACS
CLASS_NAMES = ["NRG", "RG"]
MODEL_PATH = Path(__file__).resolve().parent / "weights" / "resnet50_eyepacs_best.pt"

# match your training preprocessing (resize + ToTensor)
preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

def build_model(device: str):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def risk_bucket(prob_rg: float) -> str:
    if prob_rg >= T_HIGH:
        return "HIGH risk (refer for review)"
    if prob_rg <= T_LOW:
        return "LOW risk"
    return "MID / review suggested"

def predict(image: Image.Image):
    if image is None:
        return "No image provided", "0%", "N/A"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(device)

    if image.mode != "RGB":
        image = image.convert("RGB")

    x = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    prob_nrg = float(probs[0])
    prob_rg = float(probs[1])
    pred_idx = int(np.argmax(probs))

    # Human-readable labels
    if pred_idx == 1:
        pred_label = "Referable Glaucoma"
    else:
        pred_label = "Non-Referable Glaucoma"

    bucket = risk_bucket(prob_rg)

    # Format probability nicely as percentage
    prob_percent = f"{prob_rg * 100:.2f}%"

    return pred_label, prob_percent, bucket


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload fundus image"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Probability of Glaucoma"),
        gr.Textbox(label="Risk Category"),
    ],
    title="Glaucoma Screening Demo (ResNet50 + EyePACS AIROGS)",
    description="Research demo only — not a medical diagnosis.",
)

if __name__ == "__main__":
    demo.launch()

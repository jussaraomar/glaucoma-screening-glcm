# glcm/src/eval_eyepacs_resnet50.py

import torch
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from torchvision import models
import torch.nn as nn

from glcm.utils.data_loader_eyepacs import get_dataloaders_eyepacs
from glcm.config.params import T_HIGH, T_LOW


def load_model(device):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load("glcm/models/resnet50_eyepacs_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def screening_metrics(y_true_bin, y_prob_pos, thr):
    y_pred_bin = (y_prob_pos >= thr).astype(int)
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return sensitivity, specificity, cm


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    _, _, test_loader, class_to_idx = get_dataloaders_eyepacs()
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print("class_to_idx:", class_to_idx)

    # positive class = RG (referable glaucoma)
    rg_idx = class_to_idx.get("RG", class_to_idx.get("rg"))
    if rg_idx is None:
        rg_idx = 1  # fallback

    model = load_model(device)

    y_true, y_pred, y_prob_pos = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)

            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            prob_rg = probs[:, rg_idx]
            pred = probs.argmax(axis=1)

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(pred.tolist())
            y_prob_pos.extend(prob_rg.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob_pos = np.array(y_prob_pos)

    # --- Standard metrics (argmax) ---
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score((y_true == rg_idx).astype(int), y_prob_pos)

    cm_default = confusion_matrix(y_true, y_pred)

    print("\nAccuracy:", round(acc, 4))
    print("ROC-AUC:", round(auc, 4))
    print("\nConfusion matrix (rows=true, cols=pred):\n", cm_default)
    print("\nReport:\n", classification_report(y_true, y_pred, target_names=[idx_to_class[0], idx_to_class[1]]))

    # --- Risk grading ---
    high = int((y_prob_pos >= T_HIGH).sum())
    low = int((y_prob_pos <= T_LOW).sum())
    mid = int(len(y_prob_pos) - high - low)
    print(f"\nRisk grading using T_LOW={T_LOW}, T_HIGH={T_HIGH}: high={high} mid={mid} low={low}")

    # --- Threshold sweep for screening mode ---
    y_true_bin = (y_true == rg_idx).astype(int)

   
    best_thr = None
    best_sens = -1.0
    best_spec = -1.0
    best_cm = None

    thresholds = [0.01, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20,
                  0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90]

 
    for thr in [0.50, 0.30, 0.20, 0.10]:
        sens, spec, cm = screening_metrics(y_true_bin, y_prob_pos, thr)
        print(f"\nScreening @ threshold = {thr:.2f}")
        print("Confusion matrix (tn fp / fn tp):")
        print(cm)
        print("Sensitivity (recall RG):", round(sens, 4))
        print("Specificity (recall NRG):", round(spec, 4))

    # Find best threshold
    for thr in thresholds:
        sens, spec, cm = screening_metrics(y_true_bin, y_prob_pos, thr)
        if (sens > best_sens) or (sens == best_sens and spec > best_spec):
            best_sens = sens
            best_spec = spec
            best_thr = thr
            best_cm = cm

    assert best_thr is not None and best_cm is not None, "Best threshold was not computed"

    print("\n==============================")
    print("Best screening threshold (maximize sensitivity, then specificity):")
    print("  threshold:", best_thr)
    print("  sensitivity:", round(best_sens, 4))
    print("  specificity:", round(best_spec, 4))
    print("  confusion (tn fp / fn tp):")
    print(best_cm)
    print("==============================\n")

    # --- Save report ---
    report = {
        "dataset": "EyePACS Glaucoma",
        "model": "ResNet50",
        "image_size": 512,
        "accuracy": float(acc),
        "roc_auc": float(auc),
        "confusion_matrix_default": cm_default.tolist(),
        "best_screening_threshold": float(best_thr),
        "best_sensitivity": float(best_sens),
        "best_specificity": float(best_spec),
        "confusion_matrix_screening": best_cm.tolist(),
        "risk_grading": {
            "T_LOW": float(T_LOW),
            "T_HIGH": float(T_HIGH),
            "low": int(low),
            "mid": int(mid),
            "high": int(high),
        },
    }

    reports_dir = Path("glcm/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    out_path = reports_dir / "eval_resnet50_eyepacs.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved evaluation report to {out_path}")


if __name__ == "__main__":
    main()

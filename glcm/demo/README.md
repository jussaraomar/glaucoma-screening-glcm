---
title: Glaucoma Screening Demo
emoji: 👁️
colorFrom: blue
colorTo: green
sdk: gradio
python_version: "3.10"
app_file: app.py
---

# Glaucoma Screening Demo (ResNet50 + EyePACS)

This demo allows users to upload a retinal fundus image and receive:

- A binary classification (Referable Glaucoma / Non-Referable Glaucoma)
- A confidence score (probability of glaucoma)
- A simple risk category based on model confidence

## Model

- Architecture: ResNet50
- Pretrained on: ImageNet
- Fine-tuned on: balanced subset of the EyePACS AIROGS dataset 
- Image resolution: 512 × 512

## Disclaimer

This application is a research prototype and **does not provide medical diagnosis**.
It is intended for educational and research demonstration purposes only.

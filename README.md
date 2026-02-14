# Automated Glaucoma Screening using Deep Learning (ResNet-50)
================= 

This repository contains the full source code for a computer science project focused on automated glaucoma screening from retinal fundus images using deep learning.

The system performs binary image-level classification between:

Referable Glaucoma (RG)
Non-Referable Glaucoma (NRG)

The project also includes a confidence-based risk grading mechanism and an interactive demo application.


## Model Overview
* __Architecture:__ ResNet50
* __Pretrained on:__ ImageNet (via torchvision)
* __Fine-tuned on:__ Balanced subset of the EyePACS AIROGS glaucoma dataset
* __Image size:__ 512 × 512
* __Framework:__ PyTorch
* __Optimizer:__ AdamW
* Learning rate scheduler + early stopping
* Mixed-precision training for efficiency

##Dataset
The model was trained on a balanced subset of the EyePACS AIROGS glaucoma dataset by [Riley Kiefer](https://www.kaggle.com/datasets/deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2 "Riley Kiefer"), containing:
* ~4,000 RG and ~4,000 NRG images for training 
* ~385 RG and ~385 NRG images for validation
* ~385 RG and ~385 NRG images for testing
The dataset is not included in this repository.

## Repository Structure
glcm/
├── config/        # Parameters and paths
├── src/           # Training and evaluation scripts
├── utils/         # Data loading utilities
├── demo/          # Gradio demo application
├── models/        # Saved weights (not tracked)
└── reports/       # Evaluation outputs

## Training
Train the model:
`python -m glcm.src.train_eyepacs_resnet50`

Evaluate the model:
`python -m glcm.src.eval_eyepacs_resnet50`

Evaluation includes accuracy, ROC-AUC, confusion matrices, and threshold-based screening analysis.

## Live Demo

An interactive demo application has been deployed using Gradio and Hugging Face Spaces:

👉 [Demo App](https://huggingface.co/spaces/juuuu0/glaucoma-detection-app "Here")

The demo allows users to upload a fundus image and receive:
* A predicted class
* Probability of glaucoma
* A simple risk category
This demo serves as a proof-of-concept deployment of the trained model.

## Disclaimer
This project is intended for academic and research purposes only and does not provide medical diagnosis.
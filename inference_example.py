"""
Inference example for the SCAET-AI model.

Demonstrates how to load the trained checkpoint and run prediction on a single sample from the public dataset.

Usage:
    python inference_example.py

Requirements:
    torch, torchvision, Pillow, numpy, webdataset
"""

from pathlib import Path
import json
import io

import torch
import numpy as np
from PIL import Image
import webdataset as wds

from models import NMIClassifier
from preprocessing import preprocess


def load_model(checkpoint_path, device):
    """Load the trained model from a checkpoint file."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = NMIClassifier(n_eds_features=checkpoint["n_eds_features"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint["best_threshold"]


def predict(model, img, eds, threshold, device):
    """Run prediction on a single preprocessed sample."""

    img = img.unsqueeze(0).to(device)
    eds = eds.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img, eds)
        probability = torch.sigmoid(logits).item()

    prediction = "true NMI" if probability >= threshold else "false NMI"

    return probability, prediction


def main():

    # ── Paths ──
    checkpoint_path = Path("checkpoints/best_model.pt")
    dataset_path    = Path("data/features_test_public.tar")
    metadata_path   = Path("data/metadata_public.json")

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # ── Load model ──
    model, threshold = load_model(checkpoint_path, device)
    print(f"Model loaded. Classification threshold: {threshold:.3f}")

    # ── Load metadata (needed for EDS standardization) ──
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    eds_cols = list(metadata["eds_mean"].keys())
    eds_mean = metadata["eds_mean"]
    eds_std  = metadata["eds_std"]

    # ── Load one sample from the dataset ──
    dataset = wds.WebDataset(str(dataset_path))

    for sample in dataset:
        img, eds, true_label = preprocess(sample, eds_cols, eds_mean, eds_std)
        break

    # ── Run prediction ──
    probability, prediction = predict(model, img, eds, threshold, device)

    # ── Display result ──
    truth = "true NMI" if true_label.item() == 1 else "false NMI"

    print()
    print(f"Sample prediction:")
    print(f"  Probability:     {probability:.3f}")
    print(f"  Prediction:      {prediction}")
    print(f"  Ground truth:    {truth}")
    print(f"  Classification:  {'correct' if prediction == truth else 'incorrect'}")


if __name__ == "__main__":
    main()
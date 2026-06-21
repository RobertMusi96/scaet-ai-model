"""
Inference example for the SCAET-AI model on a user's own data.

Demonstrates how to load the trained model and run prediction on a single BSE image file and EDS composition dictionary, without going
through the webdataset format.

Usage:
    python inference_own_data.py

Requirements:
    torch, torchvision, Pillow, numpy
"""

from pathlib import Path
import json

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from models import NMIClassifier


def load_model(checkpoint_path, device):
    """Load the trained model from a checkpoint file."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = NMIClassifier(n_eds_features=checkpoint["n_eds_features"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint["best_threshold"]


def preprocess_image(image_path):
    """
    Load and preprocess a BSE image from a file path.

    The image should be a 15-bit grayscale PNG showing the local patch around
    the feature, with a margin included for morphological context.
    """

    img = np.array(Image.open(image_path))
    img = img.astype(np.float32)

    # Scale from 15-bit grayscale [0, 32767] to [0, 1], then center to [-1, 1]
    img = img / 32767.0
    img = (img - 0.5) / 0.5

    # Add channel dimension (H, W) -> (1, H, W)
    if img.ndim == 2:
        img = img[np.newaxis, :]

    # Resize to 224 x 224 for the model
    transform = transforms.Resize((224, 224))
    img = transform(torch.tensor(img))

    return img


def preprocess_eds(eds_dict, eds_cols, eds_mean, eds_std):
    """
    Standardize an EDS composition dictionary.

    Parameters
    ----------
    eds_dict : dict
        Element symbol to weight percentage, e.g. {"Fe (Wt%)": 53.9, "Al (Wt%)": 22.2, ...}.
        Missing elements are assumed to be zero.
    eds_cols : list of str
        Element order expected by the model.
    eds_mean : dict
        Per-element training-set means in wt%.
    eds_std : dict
        Per-element training-set standard deviations.
    """

    eds_vec = np.array([(eds_dict.get(col, 0.0) - eds_mean[col]) / eds_std[col] for col in eds_cols], dtype=np.float32)

    return torch.tensor(eds_vec)


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
    metadata_path = Path("data/metadata_public.json")

    # User-provided inputs
    image_path = Path("path/to/your/bse_image.png")
    eds_dict = {
        "Fe": 53.88,
        "Al": 22.23,
        "O": 18.38,
        "C": 5.51,
        # Elements not listed are assumed to be 0 wt%
    }

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
    eds_std = metadata["eds_std"]

    # ── Preprocess inputs ──
    img = preprocess_image(image_path)
    eds = preprocess_eds(eds_dict, eds_cols, eds_mean, eds_std)

    # ── Run prediction ──
    probability, prediction = predict(model, img, eds, threshold, device)

    # ── Display result ──
    print()
    print(f"Prediction:")
    print(f"  Image:        {image_path}")
    print(f"  Probability:  {probability:.3f}")
    print(f"  Classification: {prediction}")


if __name__ == "__main__":
    main()
from torchvision import transforms
import json
import numpy as np
import torch


def preprocess(sample):
    # Decode shape first
    shape = json.loads(sample["shape.json"])
    dtype = np.dtype(sample["dtype.txt"].decode("utf-8"))

    # Decode and reshape image
    img = np.frombuffer(sample["image.npy"], dtype=dtype).copy()
    img = img.reshape(shape)

    img = img.reshape(shape).astype(np.float32)
    img = img / 32767.0  # scale to [0, 1]
    img = (img - 0.5) / 0.5  # center to [-1, 1]

    # Add channel dim if grayscale (H, W) -> (1, H, W)
    if img.ndim == 2:
        img = img[np.newaxis, :]

    # Decode EDS
    eds = np.frombuffer(sample["eds.npy"], dtype=np.float32).copy()

    # Decode label
    label = float(sample["label.cls"].decode("utf-8"))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])

    # To tensor and resize
    img = transform(torch.tensor(img))
    eds = torch.tensor(eds)
    label = torch.tensor(label)

    return img, eds, label
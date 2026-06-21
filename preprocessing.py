from torchvision import transforms
from PIL import Image
import json
import io
import numpy as np
import torch


def preprocess(sample, eds_cols, eds_mean, eds_std):
    """
    Decode a sample from the public webdataset and prepare it for the model.

    Parameters
    ----------
    sample : dict
        Webdataset sample dictionary with keys 'image.png', 'eds.json', 'label.cls'.
    eds_cols : list of str
        Element symbols in the order expected by the model.
    eds_mean : dict
        Per-element training-set means in wt% (from metadata_public.json).
    eds_std : dict
        Per-element training-set standard deviations (from metadata_public.json).

    Returns
    -------
    img : torch.Tensor   shape (1, 224, 224), float32
    eds : torch.Tensor   shape (n_eds_features,), standardized
    label : torch.Tensor scalar float
    """

    # ── Decode image ──
    img = np.array(Image.open(io.BytesIO(sample["image.png"])))
    img = img.astype(np.float32)
    img = img / 32767.0
    img = (img - 0.5) / 0.5

    if img.ndim == 2:
        img = img[np.newaxis, :]

    # ── Decode EDS and apply standardization ──
    eds_raw = json.loads(sample["eds.json"])
    eds_vec = np.array(
        [(eds_raw[col] - eds_mean[col]) / eds_std[col] for col in eds_cols],
        dtype=np.float32
    )

    # ── Decode label ──
    label = float(sample["label.cls"].decode("utf-8"))

    # ── Resize image ──
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])
    img = transform(torch.tensor(img))

    eds = torch.tensor(eds_vec)
    label = torch.tensor(label)

    return img, eds, label
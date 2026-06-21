import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

class NMIClassifier(nn.Module):
    def __init__(self, n_eds_features, embedding_dim=128, eds_dim=64, dropout=0.3):
        super().__init__()

        # Image branch — EfficientNet-B0 pretrained
        self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Replace first conv to accept 1-channel grayscale instead of 3-channel RGB
        self.backbone.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )

        # Remove the built-in classifier — we only want the backbone features
        self.backbone.classifier = nn.Identity()

        # Projection: 1280 -> embedding_dim
        self.image_projector = nn.Sequential(
            nn.Linear(1280, embedding_dim),
            nn.ReLU()
        )

        # EDS branch
        self.eds_encoder = nn.Sequential(
            nn.Linear(n_eds_features, 128),
            nn.ReLU(),
            nn.Linear(128, eds_dim),
            nn.ReLU()
        )

        # Fusion head
        fusion_input_dim = embedding_dim + eds_dim  # 128 + 64 = 192
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, img, eds):
        # Image branch
        x_img = self.backbone(img)          # (B, 1280)
        x_img = self.image_projector(x_img) # (B, 128)

        # EDS branch
        x_eds = self.eds_encoder(eds)       # (B, 64)

        # Concatenate and classify
        x = torch.cat([x_img, x_eds], dim=1)  # (B, 192)
        return self.fusion_head(x).squeeze(1)  # (B,) raw logits
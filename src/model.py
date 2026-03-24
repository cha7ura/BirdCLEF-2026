"""Model definitions for BirdCLEF+ 2026."""

import timm
import torch
import torch.nn as nn

from .config import NUM_CLASSES


class BirdCLEFModel(nn.Module):
    """Mel-spectrogram classifier using a pretrained CNN backbone."""

    def __init__(self, backbone: str = "efficientnet_b0", num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, in_chans=1, num_classes=0)
        feature_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

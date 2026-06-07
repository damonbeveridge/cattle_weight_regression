"""Bespoke CNN regression model — placeholder for future PyTorch development."""

import torch
import torch.nn as nn
from torchvision import models


class CattleWeightCNN(nn.Module):
    """Transfer-learning backbone with a regression head."""

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        self.backbone = models.get_model(backbone, weights=weights)

        # Replace classifier with a single-output regression head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).squeeze(1)

"""PyTorch regression models for cattle weight estimation."""

import torch
import torch.nn as nn
from torchvision import models


class CattleWeightCNN(nn.Module):
    """Single-image regression model.

    Transfer-learning backbone with a regression head. Used for the baseline
    where each of the 4 views is treated as an independent training sample.
    At inference, average predictions across all views of the same cow.
    """

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        base = models.get_model(backbone, weights=weights)
        feature_dim = base.fc.in_features
        base.fc = nn.Identity()  # strip the classification head
        self.encoder = base
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)).squeeze(1)


class MultiViewCattleWeightCNN(nn.Module):
    """Multi-view late-fusion regression model.

    All n views share a single encoder. Features are mean-pooled across views,
    then passed through a regression head to predict a single weight value.

    This is the natural next step after establishing a CattleWeightCNN baseline:
    the encoder architecture is identical, but the model now sees all views at once
    and can learn that different angles carry complementary body-dimension information.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        n_views: int = 4,
        pretrained: bool = True,
    ):
        super().__init__()
        self.n_views = n_views

        weights = "IMAGENET1K_V1" if pretrained else None
        base = models.get_model(backbone, weights=weights)
        feature_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.encoder = base  # shared across all views

        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, views: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            views: list of n_views tensors, each (B, C, H, W)
        Returns:
            weight predictions of shape (B,)
        """
        features = torch.stack([self.encoder(v) for v in views], dim=1)  # (B, n_views, D)
        pooled = features.mean(dim=1)                                      # (B, D)
        return self.head(pooled).squeeze(1)

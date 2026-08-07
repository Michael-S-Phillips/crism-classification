import torch
import torch.nn as nn


class SpectralSpatialCNN(nn.Module):
    """
    2D CNN over a (patch_size x patch_size x n_bands) spatial-spectral patch.
    Input shape: (batch, n_bands, patch_size, patch_size)
    Returns logits of shape (batch, n_classes).
    """

    def __init__(
        self,
        n_bands: int = 60,
        n_classes: int = 6,
        patch_size: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_bands, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.classifier(x)

"""
1D Spectral CNN for per-pixel mineral classification.
Treats the 59-band mrral spectrum as a 1D signal and applies convolutional
feature extraction along the spectral dimension.
"""
import torch
import torch.nn as nn


class SpectralCNN1D(nn.Module):
    """
    1D CNN operating on a single pixel's reflectance spectrum.

    Input:  (batch, n_bands)  — e.g. (batch, 59) for mrral
    Output: (batch, n_classes) — raw logits

    Architecture:
        spectrum → unsqueeze → Conv1d stack → global avg pool → dropout → linear
    """

    def __init__(self, n_bands: int = 59, n_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: local spectral patterns (kernel 5 covers ~100nm)
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(dropout),
            # Block 2: medium-range absorption features
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(dropout),
            # Block 3: broad spectral shape
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_bands)
        x = x.unsqueeze(1)           # (batch, 1, n_bands)
        x = self.features(x)         # (batch, 256, n_bands)
        x = self.pool(x).squeeze(2)  # (batch, 256)
        return self.classifier(x)    # (batch, n_classes)

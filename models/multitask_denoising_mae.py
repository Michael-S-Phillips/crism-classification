# models/multitask_denoising_mae.py
"""Denoising MAE + supervised 5-class auxiliary head (multi-task pretraining).

Inherits the full denoising-recon path from DenoisingSpatialSpectralMAE unchanged.
Adds a single Linear(embed_dim, n_classes) head reading the full-visibility
center-token embedding (MAE.encode) — the exact token the downstream classifier
uses. forward() is the recon path; forward_aux() runs the supervised path.

Checkpoints save encoder_state in the standard format so the encoder loads into
SpatialSpectralClassifier with no changes.

Spec: docs/superpowers/specs/2026-05-26-plag-aware-pretraining-design.md
"""
from typing import Tuple

import torch
import torch.nn as nn

from models.denoising_spatial_mae import DenoisingSpatialSpectralMAE


class MultiTaskDenoisingMAE(DenoisingSpatialSpectralMAE):
    def __init__(self, *args, n_classes: int = 5, embed_dim: int = 128, **kwargs):
        super().__init__(*args, embed_dim=embed_dim, **kwargs)
        self.n_classes = n_classes
        self.aux_head = nn.Linear(embed_dim, n_classes)

    def forward_aux(self, x_labeled: torch.Tensor) -> torch.Tensor:
        """Full-visibility center-token classification logits. Shape: (B, n_classes)."""
        center = self.encode(x_labeled)        # (B, embed_dim), inherited, no masking
        return self.aux_head(center)

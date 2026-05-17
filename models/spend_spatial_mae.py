"""
SPEND-style spatial-spectral MAE (v4).

Pretraining objective: Noise2Noise between two random spectral-band views
of the same CRISM patch. Adjacent bands image the same surface but have
independent detector-noise realizations, so predicting one view from the
other forces the model to learn the underlying clean spectrum without any
synthetic-noise assumptions.

Spec: docs/superpowers/specs/2026-05-16-spend-spatial-mae-design.md
"""
from __future__ import annotations

import torch

from models.spatial_mae import SpatialSpectralMAE


def compute_spectral_mask_ratio(
    epoch: int,
    anneal_start_epoch: int = 161,
    anneal_end_epoch: int = 181,
    base: float = 0.5,
) -> float:
    """Schedule for the fraction of bands zeroed during SPEND pretraining.

    Three phases:
      - epoch < anneal_start_epoch: returns `base` (SPEND phase A)
      - anneal_start_epoch <= epoch < anneal_end_epoch: linearly decreases
        from `base` toward 0 (SPEND anneal phase B)
      - epoch >= anneal_end_epoch: returns 0.0 (plain MAE phase C)
    """
    if epoch < anneal_start_epoch:
        return base
    if epoch >= anneal_end_epoch:
        return 0.0
    return base * (anneal_end_epoch - epoch) / (anneal_end_epoch - anneal_start_epoch)


class SpendSpatialSpectralMAE(SpatialSpectralMAE):
    """SpatialSpectralMAE + SPEND spectral-partition Noise2Noise objective.

    The architecture (encoder + decoder + projections + mask token) is
    inherited from SpatialSpectralMAE unchanged. The forward pass is
    overridden to (1) sample a random per-batch band partition, (2) zero
    target-half bands in the encoder input, and (3) compute MSE loss only on
    the target-half bands of the reconstruction.
    """

    def __init__(
        self,
        n_bands: int = 59,
        patch_size: int = 7,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        decoder_dim: int = 64,
        decoder_layers: int = 2,
        mask_ratio: float = 0.75,
        dropout: float = 0.0,
        spectral_mask_ratio: float = 0.5,
    ):
        super().__init__(
            n_bands=n_bands, patch_size=patch_size,
            embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers,
            decoder_dim=decoder_dim, decoder_layers=decoder_layers,
            mask_ratio=mask_ratio, dropout=dropout,
        )
        # Mutable attribute; the training loop updates it each epoch.
        self.spectral_mask_ratio: float = spectral_mask_ratio

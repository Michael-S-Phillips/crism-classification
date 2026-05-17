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

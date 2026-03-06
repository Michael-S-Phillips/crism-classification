import torch
import torch.nn as nn


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Binary cross-entropy with logits, weighted per sample by confidence weight.
    Averages over classes first, then takes confidence-weighted mean over samples.
    """

    def forward(
        self,
        logits: torch.Tensor,   # (batch, n_classes)
        targets: torch.Tensor,  # (batch, n_classes)
        weights: torch.Tensor,  # (batch,)
    ) -> torch.Tensor:
        # Per-sample, per-class BCE: shape (batch, n_classes)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        # Mean over classes: shape (batch,)
        bce_per_sample = bce.mean(dim=1)
        # Weighted mean over samples
        return (bce_per_sample * weights).sum() / (weights.sum() + 1e-8)

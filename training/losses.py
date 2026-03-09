import torch
import torch.nn as nn
from typing import Optional


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Binary cross-entropy with logits, weighted per sample by confidence weight.
    Averages over classes first, then takes confidence-weighted mean over samples.

    Optional pos_weight: (n_classes,) tensor of positive class weights.
    pos_weight[c] = n_neg[c] / n_pos[c] upweights rare positive classes.
    """

    def forward(
        self,
        logits: torch.Tensor,                       # (batch, n_classes)
        targets: torch.Tensor,                      # (batch, n_classes)
        weights: torch.Tensor,                      # (batch,)
        pos_weight: Optional[torch.Tensor] = None,  # (n_classes,)
    ) -> torch.Tensor:
        # Per-sample, per-class BCE: shape (batch, n_classes)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction='none'
        )
        # Mean over classes: shape (batch,)
        bce_per_sample = bce.mean(dim=1)
        # Weighted mean over samples
        return (bce_per_sample * weights).sum() / (weights.sum() + 1e-8)


class FocalBCEWithLogitsLoss(nn.Module):
    """
    Focal binary cross-entropy, weighted per sample.

    Applies focal modulation (1 - p_t)^gamma to standard BCE, down-weighting
    easy examples and focusing training on hard/rare ones (e.g. plagioclase).

    gamma=2.0 is standard; gamma=0.0 reduces to weighted BCE.
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,                       # (batch, n_classes)
        targets: torch.Tensor,                      # (batch, n_classes)
        weights: torch.Tensor,                      # (batch,)
        pos_weight: Optional[torch.Tensor] = None,  # (n_classes,)
    ) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction='none'
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        loss = (focal_weight * bce).mean(dim=1)
        return (loss * weights).sum() / (weights.sum() + 1e-8)

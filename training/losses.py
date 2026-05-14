import torch
import torch.nn as nn
from typing import Optional


def _apply_class_weights(
    loss_per_class: torch.Tensor,
    class_weights: Optional[torch.Tensor],
) -> torch.Tensor:
    """Multiply (batch, n_classes) loss by per-class weight and reduce to (batch,).

    Uses the weighted-mean reduction (sum(w*l) / sum(w)) so the absolute scale
    is comparable to the unweighted mean — only the relative weight between
    classes changes the gradient direction.
    """
    if class_weights is None:
        return loss_per_class.mean(dim=1)
    w = class_weights.to(loss_per_class.device, dtype=loss_per_class.dtype)
    return (loss_per_class * w).sum(dim=1) / (w.sum() + 1e-8)


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Binary cross-entropy with logits, weighted per sample by confidence weight
    and (optionally) per class by `class_weights`.

    Averages over classes first (with optional class_weights), then takes
    confidence-weighted mean over samples.

    Optional pos_weight: (n_classes,) tensor of positive class weights —
        pos_weight[c] = n_neg[c] / n_pos[c] upweights rare positive samples.
    Optional class_weights: (n_classes,) tensor of per-class gradient multipliers,
        applied symmetrically to positive and negative samples in each class —
        useful when whole classes are under-represented overall.
    """

    def forward(
        self,
        logits: torch.Tensor,                          # (batch, n_classes)
        targets: torch.Tensor,                         # (batch, n_classes)
        weights: torch.Tensor,                         # (batch,)
        pos_weight: Optional[torch.Tensor] = None,     # (n_classes,)
        class_weights: Optional[torch.Tensor] = None,  # (n_classes,)
    ) -> torch.Tensor:
        # Per-sample, per-class BCE: shape (batch, n_classes)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction='none'
        )
        bce_per_sample = _apply_class_weights(bce, class_weights)
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
        logits: torch.Tensor,                          # (batch, n_classes)
        targets: torch.Tensor,                         # (batch, n_classes)
        weights: torch.Tensor,                         # (batch,)
        pos_weight: Optional[torch.Tensor] = None,     # (n_classes,)
        class_weights: Optional[torch.Tensor] = None,  # (n_classes,)
    ) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction='none'
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        loss = _apply_class_weights(focal_weight * bce, class_weights)
        return (loss * weights).sum() / (weights.sum() + 1e-8)


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification (Wang et al. 2021).

    Decouples positive/negative focusing via separate gamma values.
    A probability margin (clip) hard-zeros very easy negatives before
    computing the log, strongly suppressing trivial negative samples.

    Typical settings: gamma_neg=4, gamma_pos=0, clip=0.05.
    Setting gamma_neg=0, gamma_pos=0, clip=0 recovers standard BCE.

    API matches WeightedBCEWithLogitsLoss (pos_weight accepted but not used —
    ASL handles imbalance via the asymmetric focusing terms instead).
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(
        self,
        logits: torch.Tensor,                          # (batch, n_classes)
        targets: torch.Tensor,                         # (batch, n_classes)
        weights: torch.Tensor,                         # (batch,)
        pos_weight: Optional[torch.Tensor] = None,     # accepted for API compat, not used
        class_weights: Optional[torch.Tensor] = None,  # (n_classes,)
    ) -> torch.Tensor:
        p = torch.sigmoid(logits)

        # Asymmetric clip: shift negative probability down by margin before log
        p_neg = (p - self.clip).clamp(min=0) if self.clip > 0 else p

        # Log probabilities (clamped for numerical stability)
        log_p_pos = torch.log(p.clamp(min=1e-8))
        log_p_neg = torch.log((1 - p_neg).clamp(min=1e-8))

        # BCE with asymmetric log probs
        bce = targets * log_p_pos + (1 - targets) * log_p_neg

        # Asymmetric focal weights
        p_t = p * targets + p_neg * (1 - targets)
        focal_weight = torch.where(
            targets.bool(),
            (1 - p_t) ** self.gamma_pos,
            p_t ** self.gamma_neg,
        )

        loss = _apply_class_weights(-focal_weight * bce, class_weights)  # (batch,)
        return (loss * weights).sum() / (weights.sum() + 1e-8)

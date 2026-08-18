"""ASL on probabilities, plus gate supervision, for the gated head.

training.losses.AsymmetricLoss consumes logits and applies its own sigmoid. The
gated head produces PROBABILITIES (g * c), and logit(g * c) is unstable at both
ends, so the loss is restated to take p directly. The formula is otherwise
identical -- same clip, gammas, class weights and per-sample weights -- so this
arm differs from its comparator in the head alone.

Spec: docs/superpowers/specs/2026-08-17-hierarchical-mineral-gate-design.md
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gated_classifier import compose_gated_probs
from training.losses import _apply_class_weights

EPS = 1e-8


class AsymmetricLossFromProb(nn.Module):
    """AsymmetricLoss, taking probabilities instead of logits."""

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0,
                 clip: float = 0.05):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip = gamma_neg, gamma_pos, clip

    def forward(self, p, targets, weights, class_weights=None):
        p_neg = (p - self.clip).clamp(min=0) if self.clip > 0 else p
        log_p_pos = torch.log(p.clamp(min=EPS))
        log_p_neg = torch.log((1 - p_neg).clamp(min=EPS))
        bce = targets * log_p_pos + (1 - targets) * log_p_neg
        p_t = p * targets + p_neg * (1 - targets)
        focal_weight = torch.where(
            targets.bool(),
            (1 - p_t) ** self.gamma_pos,
            p_t ** self.gamma_neg,
        )
        loss = _apply_class_weights(-focal_weight * bce, class_weights)
        return (loss * weights).sum() / (weights.sum() + EPS)


class GatedAsymmetricLoss(nn.Module):
    """Main ASL on composed probabilities + lambda_gate * BCE on the gate.

    The gate would receive gradient implicitly through p_k alone, but
    implicit-only training lets it drift to a constant while the conditionals
    absorb everything. y_gate is DERIVED from the mineral labels -- the
    mineral/non-mineral partition is contradiction-free in the data.
    """

    def __init__(self, mineral_idx, non_mineral_idx, gamma_neg: float = 4.0,
                 gamma_pos: float = 0.0, clip: float = 0.0,
                 lambda_gate: float = 1.0):
        super().__init__()
        self.mineral_idx = list(mineral_idx)
        self.non_mineral_idx = list(non_mineral_idx)
        self.lambda_gate = lambda_gate
        self.main = AsymmetricLossFromProb(gamma_neg, gamma_pos, clip)

    def forward(self, logits, targets, weights, class_weights=None):
        probs, gate = compose_gated_probs(
            logits, self.mineral_idx, self.non_mineral_idx)
        loss = self.main(probs, targets, weights, class_weights)
        if self.lambda_gate:
            y_gate = (targets[:, self.mineral_idx].amax(dim=1) > 0).float()
            gate_bce = F.binary_cross_entropy(
                gate.clamp(EPS, 1 - EPS), y_gate, reduction='none')
            loss = loss + self.lambda_gate * (
                (gate_bce * weights).sum() / (weights.sum() + EPS))
        return loss

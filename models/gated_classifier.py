"""Hierarchical mineral-present gate.

Seven independent sigmoids let the model assert "this is lcp" and "this is
featureless" about the same pixel: on Nili t1321 the e87 model puts mean
p_lcp 0.996 and mean p_bland 0.067 on the SAME pixels, and 35.4% of valid pixels
have max(p_mineral) + max(p_non-mineral) > 1, peaking at 1.935.

    g   = sigmoid(z_gate)                  P(any mineral present)
    p_k = g * sigmoid(z_k)                 minerals
    p_k = (1 - g) * sigmoid(z_k)           bland, junk

so max(p_mineral) + max(p_non-mineral) <= 1 by construction. Conditionals stay
INDEPENDENT sigmoids within each branch: 18.8% of training rows carry 2+ minerals
and a softmax over the seven classes would destroy assemblages like olivine+hcp.

Spec: docs/superpowers/specs/2026-08-17-hierarchical-mineral-gate-design.md
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.spatial_spectral_classifier_aux import SpatialSpectralClassifierAux

MINERAL_NAMES_7 = ('olivine', 'lcp', 'hcp', 'plagioclase', 'alteration')
NON_MINERAL_NAMES_7 = ('bland', 'junk')


def class_partition(class_names):
    """(mineral_idx, non_mineral_idx) for a vocabulary. Raises on anything else."""
    names = list(class_names)
    unknown = set(names) - set(MINERAL_NAMES_7) - set(NON_MINERAL_NAMES_7)
    if unknown:
        raise ValueError(
            f'gate partition undefined for {sorted(unknown)}; known classes are '
            f'{MINERAL_NAMES_7 + NON_MINERAL_NAMES_7}')
    mineral = [i for i, n in enumerate(names) if n in MINERAL_NAMES_7]
    non_mineral = [i for i, n in enumerate(names) if n in NON_MINERAL_NAMES_7]
    if not mineral or not non_mineral:
        raise ValueError(f'both branches must be non-empty; got {names}')
    return mineral, non_mineral


def compose_gated_probs(logits, mineral_idx, non_mineral_idx):
    """(B, 8) logits -> ((B, 7) probabilities, (B,) gate).

    Computed in log space: exp(logsigmoid(z_g) + logsigmoid(z_k)) stays accurate
    in the small-p tail where the naive product loses precision.
    """
    if logits.shape[-1] != len(mineral_idx) + len(non_mineral_idx) + 1:
        raise ValueError(
            f'expected {len(mineral_idx) + len(non_mineral_idx) + 1} logits '
            f'(1 gate + conditionals), got {logits.shape[-1]}')
    z_gate, z_cond = logits[:, 0:1], logits[:, 1:]
    log_g, log_1mg = F.logsigmoid(z_gate), F.logsigmoid(-z_gate)
    log_c = F.logsigmoid(z_cond)
    branch = torch.empty_like(log_c)
    branch[:, mineral_idx] = log_g
    branch[:, non_mineral_idx] = log_1mg
    return torch.exp(branch + log_c), torch.sigmoid(z_gate).squeeze(-1)


class GatedSpatialSpectralClassifierAux(SpatialSpectralClassifierAux):
    """Identical to its parent but for one extra head output: the gate logit.

    Returns raw logits so it stays a plain nn.Module; callers compose
    probabilities with compose_gated_probs. Keeping composition OUT of the model
    is what lets training and inference share one implementation.
    """

    def __init__(self, *args, n_classes: int = 7, **kwargs):
        super().__init__(*args, n_classes=n_classes + 1, **kwargs)
        self.n_real_classes = n_classes

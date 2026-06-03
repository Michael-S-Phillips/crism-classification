"""Tests for the weighted InfoNCE loss."""
import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.contrastive_train import info_nce_loss


def _norm(x):
    return F.normalize(x, dim=-1)


def test_identical_anchor_positive_with_orthogonal_negatives_low_loss():
    """When anchor==positive and negatives are orthogonal, loss should be small."""
    torch.manual_seed(0)
    B, D, N_h, N_s = 4, 8, 3, 3
    # Anchor and positive are the same e_0 direction
    e0 = torch.zeros(B, D); e0[:, 0] = 1.0
    z_anchor = _norm(e0)
    z_pos = _norm(e0.clone())
    # Negatives orthogonal: pick directions e_1, e_2, e_3
    e_neg = torch.zeros(B, N_h, D); e_neg[:, :, 1] = 1.0
    e_neg2 = torch.zeros(B, N_s, D); e_neg2[:, :, 2] = 1.0
    z_hard = _norm(e_neg)
    z_soft = _norm(e_neg2)
    loss = info_nce_loss(z_anchor, z_pos, z_hard, z_soft,
                         tau=0.07, hard_weight=1.0, soft_weight=1.0)
    # Lower bound: -(1/0.07 - log(sum exp)). With equal-norm orthogonals and
    # tau=0.07, the positive term dominates by huge margin → loss near 0.
    assert loss.item() < 0.01, f'expected ~0 loss, got {loss.item()}'


def test_anchor_equals_hard_negative_large_loss():
    """When anchor==hard_neg, denominator is dominated by negatives → loss large."""
    torch.manual_seed(0)
    B, D, N_h, N_s = 2, 8, 3, 3
    e0 = torch.zeros(B, D); e0[:, 0] = 1.0
    z_anchor = _norm(e0)
    # positive: orthogonal direction → tiny numerator
    z_pos = torch.zeros(B, D); z_pos[:, 1] = 1.0; z_pos = _norm(z_pos)
    # hard negatives: ALL equal to anchor → big numerator-killer
    z_hard = z_anchor.unsqueeze(1).expand(B, N_h, D).contiguous()
    z_soft = torch.zeros(B, N_s, D); z_soft[:, :, 2] = 1.0; z_soft = _norm(z_soft)
    loss = info_nce_loss(z_anchor, z_pos, z_hard, z_soft,
                         tau=0.07, hard_weight=2.0, soft_weight=1.0)
    # Numerator sim: 0/tau == 0; denominator dominated by 1/tau >> 0 → loss
    # should be approximately 1/tau ≈ 14.28.
    assert loss.item() > 5.0, f'expected large loss, got {loss.item()}'


def test_hard_weight_increases_loss_when_hard_negs_are_close():
    """Raising hard_weight on close hard negatives should monotonically increase loss."""
    torch.manual_seed(0)
    B, D, N_h, N_s = 4, 8, 3, 3
    e0 = torch.zeros(B, D); e0[:, 0] = 1.0
    z_anchor = _norm(e0)
    z_pos = _norm(e0 + 0.01 * torch.randn(B, D))
    # hard_negs: somewhat aligned with anchor (cosine ~0.6)
    z_hard = e0.unsqueeze(1).expand(B, N_h, D).clone()
    z_hard = z_hard + 0.5 * torch.randn(B, N_h, D)
    z_hard = _norm(z_hard)
    # soft_negs: orthogonal
    z_soft = torch.zeros(B, N_s, D); z_soft[:, :, 1] = 1.0; z_soft = _norm(z_soft)
    losses = [
        info_nce_loss(z_anchor, z_pos, z_hard, z_soft,
                      tau=0.07, hard_weight=w, soft_weight=1.0).item()
        for w in (0.5, 1.0, 2.0, 4.0)
    ]
    # Monotonically increasing
    for a, b in zip(losses, losses[1:]):
        assert b > a, f'loss not monotone in hard_weight: {losses}'


def test_gradients_flow_to_all_inputs():
    torch.manual_seed(0)
    B, D, N_h, N_s = 4, 8, 3, 3
    z_anchor = _norm(torch.randn(B, D, requires_grad=True))
    z_pos = _norm(torch.randn(B, D, requires_grad=True))
    z_hard = _norm(torch.randn(B, N_h, D, requires_grad=True))
    z_soft = _norm(torch.randn(B, N_s, D, requires_grad=True))
    # Re-build with requires_grad on raw tensors (normalize keeps grad)
    raw_anchor = torch.randn(B, D, requires_grad=True)
    raw_pos = torch.randn(B, D, requires_grad=True)
    raw_hard = torch.randn(B, N_h, D, requires_grad=True)
    raw_soft = torch.randn(B, N_s, D, requires_grad=True)
    z_a = F.normalize(raw_anchor, dim=-1)
    z_p = F.normalize(raw_pos, dim=-1)
    z_h = F.normalize(raw_hard, dim=-1)
    z_s = F.normalize(raw_soft, dim=-1)
    loss = info_nce_loss(z_a, z_p, z_h, z_s, tau=0.07)
    loss.backward()
    assert raw_anchor.grad is not None and raw_anchor.grad.abs().sum() > 0
    assert raw_pos.grad is not None and raw_pos.grad.abs().sum() > 0
    assert raw_hard.grad is not None and raw_hard.grad.abs().sum() > 0
    assert raw_soft.grad is not None and raw_soft.grad.abs().sum() > 0


def test_invalid_args_rejected():
    z = F.normalize(torch.randn(2, 4), dim=-1)
    zn = F.normalize(torch.randn(2, 3, 4), dim=-1)
    with pytest.raises(ValueError):
        info_nce_loss(z, z, zn, zn, tau=0.0)
    with pytest.raises(ValueError):
        info_nce_loss(z, z, zn, zn, hard_weight=0.0)
    with pytest.raises(ValueError):
        info_nce_loss(z, z, zn, zn, soft_weight=-1.0)


def test_finite_loss_on_random_input():
    torch.manual_seed(42)
    B, D, N_h, N_s = 16, 32, 4, 4
    z_anchor = F.normalize(torch.randn(B, D), dim=-1)
    z_pos = F.normalize(torch.randn(B, D), dim=-1)
    z_hard = F.normalize(torch.randn(B, N_h, D), dim=-1)
    z_soft = F.normalize(torch.randn(B, N_s, D), dim=-1)
    loss = info_nce_loss(z_anchor, z_pos, z_hard, z_soft)
    assert torch.isfinite(loss)

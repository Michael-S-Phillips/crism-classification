"""The gated loss must match ASL exactly on probabilities, plus gate supervision."""
from __future__ import annotations

import pytest
import torch

from training.gated_losses import AsymmetricLossFromProb, GatedAsymmetricLoss
from training.losses import AsymmetricLoss

MIN_IDX, NON_IDX = [0, 1, 2, 3, 5], [4, 6]


def test_prob_form_matches_the_logit_form_exactly():
    """If these diverge, this arm is not comparable to any other -- the loss
    would differ as well as the head."""
    torch.manual_seed(0)
    logits = torch.randn(32, 7)
    targets = (torch.rand(32, 7) > 0.7).float()
    w = torch.ones(32)
    ref = AsymmetricLoss(4.0, 0.0, 0.05)(logits, targets, w)
    got = AsymmetricLossFromProb(4.0, 0.0, 0.05)(torch.sigmoid(logits), targets, w)
    assert got.item() == pytest.approx(ref.item(), rel=1e-5)


def test_prob_form_matches_at_clip_zero():
    torch.manual_seed(1)
    logits = torch.randn(32, 7)
    targets = (torch.rand(32, 7) > 0.5).float()
    w = torch.ones(32)
    ref = AsymmetricLoss(4.0, 0.0, 0.0)(logits, targets, w)
    got = AsymmetricLossFromProb(4.0, 0.0, 0.0)(torch.sigmoid(logits), targets, w)
    assert got.item() == pytest.approx(ref.item(), rel=1e-5)


def test_gate_target_is_derived_from_the_mineral_labels():
    """A pixel with any mineral positive should drive the gate open; the
    AUXILIARY gate term specifically (not the main ASL term, which already
    prefers the agreeing gate state on its own and would mask a wrong y_gate)
    must be lower when the gate agrees than when it disagrees.

    Isolate the auxiliary term by differencing against lambda_gate=0, whose
    main-term loss is identical (same logits/targets/partition) and so
    cancels exactly, leaving only lambda_gate * gate_bce.
    """
    targets = torch.zeros(2, 7)
    targets[0, 1] = 1.0          # lcp positive -> gate should open
    targets[1, 4] = 1.0          # bland positive -> gate should shut
    gated = GatedAsymmetricLoss(MIN_IDX, NON_IDX, 4.0, 0.0, 0.0, lambda_gate=1.0)
    ungated = GatedAsymmetricLoss(MIN_IDX, NON_IDX, 4.0, 0.0, 0.0, lambda_gate=0.0)
    agree, disagree = torch.zeros(2, 8), torch.zeros(2, 8)
    agree[0, 0], agree[1, 0] = 6.0, -6.0
    disagree[0, 0], disagree[1, 0] = -6.0, 6.0
    w = torch.ones(2)
    agree_gate_term = gated(agree, targets, w).item() - ungated(agree, targets, w).item()
    disagree_gate_term = (
        gated(disagree, targets, w).item() - ungated(disagree, targets, w).item())
    assert agree_gate_term < disagree_gate_term


def test_lambda_gate_zero_removes_the_auxiliary_term():
    torch.manual_seed(2)
    logits = torch.randn(16, 8)
    targets = (torch.rand(16, 7) > 0.6).float()
    w = torch.ones(16)
    a = GatedAsymmetricLoss(MIN_IDX, NON_IDX, 4.0, 0.0, 0.0, lambda_gate=0.0)
    b = GatedAsymmetricLoss(MIN_IDX, NON_IDX, 4.0, 0.0, 0.0, lambda_gate=5.0)
    assert a(logits, targets, w).item() != pytest.approx(b(logits, targets, w).item())


def test_loss_is_finite_at_saturation():
    """A large-magnitude gate logit drives composed probabilities to exactly 0
    in float32; log(0) must not produce NaN and silently poison training.

    +-30 (float32 min prob ~4.7e-14, still representable and finite under
    log) does NOT actually underflow to exact 0.0 -- exact-zero underflow
    only starts around +-120 for this composition. Verified by removing the
    EPS clamp in AsymmetricLossFromProb: at +-30 the loss stays finite either
    way (that value would not catch a missing clamp); at +-200 it goes NaN
    without the clamp and stays finite with it, so +-200 is used here to
    actually exercise the guard.
    """
    for z in (-200.0, 200.0):
        logits = torch.zeros(4, 8)
        logits[:, 0] = z
        targets = (torch.rand(4, 7) > 0.5).float()
        out = GatedAsymmetricLoss(MIN_IDX, NON_IDX, 4.0, 0.0, 0.0, 1.0)(
            logits, targets, torch.ones(4))
        assert torch.isfinite(out), f'non-finite loss at gate logit {z}'


def test_gradients_reach_the_gate_logit():
    torch.manual_seed(3)
    logits = torch.randn(8, 8, requires_grad=True)
    targets = (torch.rand(8, 7) > 0.5).float()
    GatedAsymmetricLoss(MIN_IDX, NON_IDX, 4.0, 0.0, 0.0, 1.0)(
        logits, targets, torch.ones(8)).backward()
    assert logits.grad[:, 0].abs().sum().item() > 0

"""ASL's probability clip kills the gradient on confidently-wrong negatives.

Discovered 2026-08-17 while investigating why the dual-CR model fires lcp >= 0.99
on bright dust (Nili t1321: 35% of 125,757 confident-lcp px have LCPINDEX2 ~ 0)
and never corrects.

ASL scores a negative with log(1 - (p - clip)). As p -> 1 that saturates at
1/clip instead of diverging as 1/(1-p), so it no longer cancels the sigmoid
derivative p(1-p) -> 0 and the gradient decays to nothing. Plain BCE keeps
|dL/dlogit| = p because the cancellation is exact.

These tests pin the behaviour in both directions: that the defect is real at the
project's default clip, and that clip=0 fixes it WITHOUT reintroducing the
trivial-negative noise the clip was there to suppress (gamma_neg does that job).
They are the executable form of the note in MODELS.md.
"""
from __future__ import annotations

import pytest
import torch

from training.losses import AsymmetricLoss

DEFAULT_GAMMA_NEG = 4.0     # scripts/train.py --asl_gamma_neg default
DEFAULT_CLIP = 0.05         # scripts/train.py --asl_clip default


def neg_grad(loss, p: float) -> float:
    """|dL/dlogit| for a single NEGATIVE example predicted at probability p."""
    z = torch.tensor([[torch.logit(torch.tensor(p))]], requires_grad=True)
    loss(z, torch.zeros(1, 1), torch.ones(1)).backward()
    return z.grad.abs().item()


def bce_grad(p: float) -> float:
    """Plain BCE reference: analytically exactly p for a negative."""
    return neg_grad(AsymmetricLoss(0.0, 0.0, 0.0), p)


def test_bce_reference_is_p():
    """Anchors the harness: if this drifts, every ratio below is meaningless."""
    for p in (0.1, 0.5, 0.99, 0.999):
        assert bce_grad(p) == pytest.approx(p, rel=1e-4)


def test_default_clip_all_but_eliminates_correction_of_confident_errors():
    """The defect. At the project default, a negative the model scores 0.999
    receives under 5% of BCE's correction, and under 1% at 0.9999."""
    asl = AsymmetricLoss(DEFAULT_GAMMA_NEG, 0.0, DEFAULT_CLIP)
    assert neg_grad(asl, 0.999) / bce_grad(0.999) < 0.05
    assert neg_grad(asl, 0.9999) / bce_grad(0.9999) < 0.01


def test_removing_the_clip_restores_it():
    """clip=0 brings confident-error correction back to BCE strength or better."""
    asl = AsymmetricLoss(DEFAULT_GAMMA_NEG, 0.0, 0.0)
    assert neg_grad(asl, 0.999) / bce_grad(0.999) > 0.95
    assert neg_grad(asl, 0.9999) / bce_grad(0.9999) > 0.95


def test_clip_zero_still_suppresses_easy_negatives():
    """The clip's stated purpose is discarding trivial negatives. gamma_neg=4
    already does that, so removing the clip does not undo it -- this is why
    clip=0 is a targeted fix and not a reversion to BCE."""
    asl = AsymmetricLoss(DEFAULT_GAMMA_NEG, 0.0, 0.0)
    assert neg_grad(asl, 0.05) / bce_grad(0.05) < 0.01
    assert neg_grad(asl, 0.20) / bce_grad(0.20) < 0.05


def test_clip_zero_keeps_hard_negative_focus():
    """gamma_neg should still UPweight mid-high negatives relative to BCE;
    otherwise clip=0 would just be BCE with extra steps."""
    asl = AsymmetricLoss(DEFAULT_GAMMA_NEG, 0.0, 0.0)
    assert neg_grad(asl, 0.90) / bce_grad(0.90) > 1.0


def test_gamma_is_not_the_cause():
    """Halving gamma_neg does not rescue the high-confidence gradient while the
    clip remains -- attributing the defect to gamma would send the fix at the
    wrong knob."""
    for gamma in (2.0, 4.0):
        asl = AsymmetricLoss(gamma, 0.0, DEFAULT_CLIP)
        assert neg_grad(asl, 0.999) / bce_grad(0.999) < 0.06


def test_gradient_is_non_monotonic_in_p_under_the_default_clip():
    """The shape of the defect: correction PEAKS in the middle and falls off at
    high confidence, so the loss pushes hardest on pixels it is least wrong
    about. A monotone-increasing curve (BCE) has no such blind spot."""
    asl = AsymmetricLoss(DEFAULT_GAMMA_NEG, 0.0, DEFAULT_CLIP)
    assert neg_grad(asl, 0.90) > neg_grad(asl, 0.999)
    assert bce_grad(0.90) < bce_grad(0.999)

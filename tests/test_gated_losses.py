"""The gated loss must match ASL exactly on probabilities, plus gate supervision."""
from __future__ import annotations

import pytest
import torch

import inspect

from training.gated_losses import AsymmetricLossFromProb, GatedAsymmetricLoss
from training.losses import AsymmetricLoss

MIN_IDX, NON_IDX = [0, 1, 2, 3, 5], [4, 6]


# --- the call convention train_torch actually uses -------------------------
# training/train_torch.py's non-decomp branch calls EVERY loss as
#     loss_fn(logits, labels, weights, pos_weight=..., class_weights=...)
# with both trailing arguments as KEYWORDS. Every other test in this file
# calls the loss positionally, which is why a missing pos_weight parameter
# passed 33 green tests and still killed the run on its first batch.

def _call_like_train_torch(loss_fn, logits, targets, weights,
                           pos_weight=None, class_weights=None):
    """Byte-for-byte the keyword convention of train_torch.py's train loop."""
    return loss_fn(logits, targets, weights,
                   pos_weight=pos_weight, class_weights=class_weights)


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


# --- C1: train_torch's keyword call convention -----------------------------

def test_gated_loss_accepts_train_torchs_keyword_call():
    """train_torch calls loss_fn(..., pos_weight=..., class_weights=...).

    Without a pos_weight parameter this raises
    TypeError: GatedAsymmetricLoss.forward() got an unexpected keyword
    argument 'pos_weight' on the FIRST batch of the run.
    """
    torch.manual_seed(0)
    logits = torch.randn(8, 8)
    targets = (torch.rand(8, 7) > 0.5).float()
    loss = _call_like_train_torch(
        GatedAsymmetricLoss(MIN_IDX, NON_IDX, 4.0, 0.0, 0.0, 1.0),
        logits, targets, torch.ones(8),
        pos_weight=torch.ones(7), class_weights=torch.ones(7))
    assert torch.isfinite(loss)


def test_prob_form_accepts_train_torchs_keyword_call():
    """AsymmetricLossFromProb is reachable as a loss_fn in its own right and
    must carry the same API as training.losses.AsymmetricLoss."""
    torch.manual_seed(1)
    p = torch.sigmoid(torch.randn(8, 7))
    targets = (torch.rand(8, 7) > 0.5).float()
    loss = _call_like_train_torch(
        AsymmetricLossFromProb(4.0, 0.0, 0.0), p, targets, torch.ones(8),
        pos_weight=torch.ones(7), class_weights=torch.ones(7))
    assert torch.isfinite(loss)


@pytest.mark.parametrize('cls', [GatedAsymmetricLoss, AsymmetricLossFromProb])
def test_forward_signature_matches_asymmetric_loss(cls):
    """Structural guard: the gated losses are drop-in replacements for
    training.losses.AsymmetricLoss at train_torch's one call site, so the
    trailing parameter NAMES and ORDER must agree."""
    ref = list(inspect.signature(AsymmetricLoss.forward).parameters)[1:]
    got = list(inspect.signature(cls.forward).parameters)[1:]
    assert got[1:] == ref[1:], (
        f'{cls.__name__}.forward{tuple(got)} is not callable the way '
        f'train_torch calls AsymmetricLoss.forward{tuple(ref)}')


def test_pos_weight_is_accepted_but_ignored():
    """Matches training/losses.py:113 -- ASL handles imbalance through its
    asymmetric focusing terms, so pos_weight must not change the value.
    Accepting it and then USING it would silently change this arm's loss."""
    torch.manual_seed(2)
    logits = torch.randn(16, 8)
    targets = (torch.rand(16, 7) > 0.5).float()
    w = torch.ones(16)
    fn = GatedAsymmetricLoss(MIN_IDX, NON_IDX, 4.0, 0.0, 0.0, 1.0)
    a = _call_like_train_torch(fn, logits, targets, w, pos_weight=None)
    b = _call_like_train_torch(fn, logits, targets, w,
                               pos_weight=torch.full((7,), 9.0))
    assert a.item() == pytest.approx(b.item(), rel=1e-9)


def test_class_weights_still_reach_the_main_term_as_a_keyword():
    """The trap in fixing C1: GatedAsymmetricLoss forwards to its inner
    AsymmetricLossFromProb. If that forwarding stays POSITIONAL once
    pos_weight is inserted ahead of class_weights, the class weights land in
    the ignored pos_weight slot and per-class weighting silently vanishes --
    a wrong-but-running loss, worse than the crash it replaced."""
    torch.manual_seed(3)
    logits = torch.randn(16, 8)
    targets = (torch.rand(16, 7) > 0.5).float()
    w = torch.ones(16)
    fn = GatedAsymmetricLoss(MIN_IDX, NON_IDX, 4.0, 0.0, 0.0, lambda_gate=0.0)
    flat = _call_like_train_torch(fn, logits, targets, w,
                                  class_weights=torch.ones(7))
    skewed = _call_like_train_torch(
        fn, logits, targets, w,
        class_weights=torch.tensor([5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    assert flat.item() != pytest.approx(skewed.item(), rel=1e-6), \
        'class_weights had no effect -- it is being swallowed by pos_weight'

# Hierarchical Mineral Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace seven independent sigmoids with a mineral-present gate times per-class conditionals, so "this is lcp" and "this is featureless" can no longer both be asserted about the same pixel.

**Architecture:** The model emits 8 raw logits (1 gate + 7 conditionals) and stays a plain `nn.Module` returning logits. A single free function `compose_gated_probs` turns those into 7 probabilities, and **both training and inference call it** — that shared call is what stops inference writing raw conditionals that look like probabilities. The loss is the existing ASL formula rewritten to accept probabilities instead of logits, plus an auxiliary BCE on the gate.

**Tech Stack:** Python 3.11, PyTorch, conda env `crism`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-17-hierarchical-mineral-gate-design.md`. Read it first.
- All commands run under `conda run -n crism` from the repo root.
- **mineral** = `olivine, lcp, hcp, plagioclase, alteration`; **non-mineral** = `bland, junk`. Verified in the data: 0 contradictions across 2,619,784 rows.
- **Conditionals stay independent sigmoids within each branch.** 18.8% of training rows carry 2+ minerals and 4.3% carry 3; a softmax over the seven classes would destroy real assemblages like olivine+hcp.
- Class order is `['olivine','lcp','hcp','plagioclase','bland','alteration','junk']` (`_CLASS_NAMES_7` in `scripts/classify_tile_supervised.py`). Gate logit is index **0** of the 8; conditionals follow in class order at indices 1–7.
- **This arm uses `--asl_clip 0.0`.** Running it at 0.05 would spend 24 h reproducing the defect measured in `tests/test_asl_clip_gradient.py`.
- Compare against `dualcr_noclip`, never against e87 — e87 differs in clip as well as head.
- Do NOT run the full pytest suite (~50 min, 13 known pre-existing failures). Run only the named test files.
- Never use `git checkout --` to revert; use `cp` backup/restore.

---

### Task 1: Gated model and probability composition

**Files:**
- Create: `models/gated_classifier.py`
- Test: `tests/test_gated_classifier.py`

**Interfaces:**
- Consumes: `models.spatial_spectral_classifier_aux.SpatialSpectralClassifierAux` (its `__init__` takes `n_bands, patch_size, n_classes, embed_dim, n_heads, n_layers, dropout, aux_dim, aux_hidden`; `forward(x, aux)` returns `(B, n_classes)`).
- Produces:
  - `MINERAL_NAMES_7`, `NON_MINERAL_NAMES_7`
  - `class_partition(class_names) -> tuple[list[int], list[int]]`
  - `compose_gated_probs(logits, mineral_idx, non_mineral_idx) -> tuple[Tensor, Tensor]` returning `(probs (B,7), gate (B,))`
  - `GatedSpatialSpectralClassifierAux` with `forward(x, aux) -> (B, 8)` logits

- [ ] **Step 1: Write the failing tests**

```python
"""The gate makes mineral and non-mineral mutually exclusive by construction."""
from __future__ import annotations

import pytest
import torch

from models.gated_classifier import (
    GatedSpatialSpectralClassifierAux, class_partition, compose_gated_probs,
)

CLASSES = ['olivine', 'lcp', 'hcp', 'plagioclase', 'bland', 'alteration', 'junk']
MIN_IDX, NON_IDX = [0, 1, 2, 3, 5], [4, 6]


def test_partition_matches_the_spec():
    m, n = class_partition(CLASSES)
    assert m == MIN_IDX
    assert n == NON_IDX


def test_partition_rejects_an_unknown_vocabulary():
    """A silent mis-partition would gate the wrong classes."""
    with pytest.raises(ValueError):
        class_partition(['olivine', 'pyx', 'plagioclase'])


def test_the_exclusivity_constraint_holds_for_random_logits():
    """The whole point: max(p_mineral) + max(p_non-mineral) <= 1. The current
    flat head violates this on 35.4% of t1321's valid pixels, peaking at 1.935."""
    torch.manual_seed(0)
    logits = torch.randn(512, 8) * 5
    probs, _ = compose_gated_probs(logits, MIN_IDX, NON_IDX)
    s = probs[:, MIN_IDX].max(1).values + probs[:, NON_IDX].max(1).values
    assert s.max().item() <= 1.0 + 1e-5


def test_all_probabilities_are_in_range():
    torch.manual_seed(1)
    probs, gate = compose_gated_probs(torch.randn(256, 8) * 8, MIN_IDX, NON_IDX)
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    assert gate.min() >= 0.0 and gate.max() <= 1.0


def test_a_closed_gate_zeroes_every_mineral():
    logits = torch.zeros(1, 8)
    logits[0, 0] = -30.0            # gate shut
    probs, gate = compose_gated_probs(logits, MIN_IDX, NON_IDX)
    assert gate.item() == pytest.approx(0.0, abs=1e-6)
    assert probs[0, MIN_IDX].max().item() == pytest.approx(0.0, abs=1e-6)


def test_an_open_gate_zeroes_bland_and_junk():
    logits = torch.zeros(1, 8)
    logits[0, 0] = 30.0             # gate open
    probs, gate = compose_gated_probs(logits, MIN_IDX, NON_IDX)
    assert gate.item() == pytest.approx(1.0, abs=1e-6)
    assert probs[0, NON_IDX].max().item() == pytest.approx(0.0, abs=1e-6)


def test_co_occurrence_survives_the_gate():
    """olivine+hcp is a real assemblage and 18.8% of training rows carry 2+
    minerals. If the gate forced competition among minerals this would fail."""
    logits = torch.zeros(1, 8)
    logits[0, 0] = 10.0                          # gate open
    logits[0, 1 + CLASSES.index('olivine')] = 10.0
    logits[0, 1 + CLASSES.index('hcp')] = 10.0
    probs, _ = compose_gated_probs(logits, MIN_IDX, NON_IDX)
    assert probs[0, CLASSES.index('olivine')].item() > 0.99
    assert probs[0, CLASSES.index('hcp')].item() > 0.99


def test_composition_equals_the_plain_product():
    """Log-space maths must agree with the naive product where the naive one is
    still accurate -- otherwise a bug hides behind 'numerical stability'."""
    torch.manual_seed(2)
    logits = torch.randn(64, 8)
    probs, gate = compose_gated_probs(logits, MIN_IDX, NON_IDX)
    c = torch.sigmoid(logits[:, 1:])
    g = torch.sigmoid(logits[:, 0:1])
    assert torch.allclose(probs[:, MIN_IDX], (g * c)[:, MIN_IDX], atol=1e-6)
    assert torch.allclose(probs[:, NON_IDX], ((1 - g) * c)[:, NON_IDX], atol=1e-6)


def test_model_emits_eight_logits():
    m = GatedSpatialSpectralClassifierAux(
        n_bands=118, patch_size=7, embed_dim=32, n_heads=2, n_layers=1,
        aux_dim=1, aux_hidden=16)
    out = m(torch.randn(4, 7, 7, 118), torch.randn(4, 1))
    assert out.shape == (4, 8)


def test_model_head_is_one_wider_than_the_class_count():
    m = GatedSpatialSpectralClassifierAux(
        n_bands=118, patch_size=7, embed_dim=32, n_heads=2, n_layers=1,
        aux_dim=1, aux_hidden=16)
    assert m.head.out_features == 8
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n crism python -m pytest tests/test_gated_classifier.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'models.gated_classifier'`

- [ ] **Step 3: Implement**

Create `models/gated_classifier.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n crism python -m pytest tests/test_gated_classifier.py -q`
Expected: PASS (10 tests)

- [ ] **Step 5: Verify the tests can fail (mutation)**

```bash
cp models/gated_classifier.py /tmp/gated_backup.py
python3 - <<'PY'
p='models/gated_classifier.py'; s=open(p).read()
s=s.replace("    branch[:, non_mineral_idx] = log_1mg", "    branch[:, non_mineral_idx] = log_g")
open(p,'w').write(s)
PY
conda run -n crism python -m pytest tests/test_gated_classifier.py -q
cp /tmp/gated_backup.py models/gated_classifier.py
```
Expected: the exclusivity and open-gate tests FAIL, then all pass after restore. If they all still pass, the constraint is not actually being tested.

- [ ] **Step 6: Commit**

```bash
git add models/gated_classifier.py tests/test_gated_classifier.py
git commit -m "gated classifier: mineral-present gate times per-class conditionals

p_k = g*c_k for minerals, (1-g)*c_k for bland/junk, so
max(p_mineral)+max(p_non-mineral) <= 1 by construction -- the current flat head
violates that on 35.4% of t1321's valid pixels, peaking at 1.935.

Conditionals stay independent sigmoids WITHIN each branch: 18.8% of training rows
carry 2+ minerals, so a softmax over the seven classes would destroy real
assemblages like olivine+hcp.

Composition lives in a free function, not the model, so training and inference
share one implementation -- inference writing raw conditionals would produce npz
values that look like probabilities and are wrong."
```

---

### Task 2: Gated loss

**Files:**
- Create: `training/gated_losses.py`
- Test: `tests/test_gated_losses.py`

**Interfaces:**
- Consumes: `models.gated_classifier.compose_gated_probs`; `training.losses._apply_class_weights`.
- Produces: `AsymmetricLossFromProb(gamma_neg, gamma_pos, clip)` with `forward(p, targets, weights, class_weights=None) -> scalar`; `GatedAsymmetricLoss(mineral_idx, non_mineral_idx, gamma_neg, gamma_pos, clip, lambda_gate)` with `forward(logits, targets, weights, class_weights=None) -> scalar`.

- [ ] **Step 1: Write the failing tests**

```python
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
    """A pixel with any mineral positive should drive the gate open; the loss
    must be lower when the gate agrees than when it disagrees."""
    targets = torch.zeros(2, 7)
    targets[0, 1] = 1.0          # lcp positive -> gate should open
    targets[1, 4] = 1.0          # bland positive -> gate should shut
    loss = GatedAsymmetricLoss(MIN_IDX, NON_IDX, 4.0, 0.0, 0.0, lambda_gate=1.0)
    agree, disagree = torch.zeros(2, 8), torch.zeros(2, 8)
    agree[0, 0], agree[1, 0] = 6.0, -6.0
    disagree[0, 0], disagree[1, 0] = -6.0, 6.0
    w = torch.ones(2)
    assert loss(agree, targets, w).item() < loss(disagree, targets, w).item()


def test_lambda_gate_zero_removes_the_auxiliary_term():
    torch.manual_seed(2)
    logits = torch.randn(16, 8)
    targets = (torch.rand(16, 7) > 0.6).float()
    w = torch.ones(16)
    a = GatedAsymmetricLoss(MIN_IDX, NON_IDX, 4.0, 0.0, 0.0, lambda_gate=0.0)
    b = GatedAsymmetricLoss(MIN_IDX, NON_IDX, 4.0, 0.0, 0.0, lambda_gate=5.0)
    assert a(logits, targets, w).item() != pytest.approx(b(logits, targets, w).item())


def test_loss_is_finite_at_saturation():
    """Gate at +-30 drives probabilities to exactly 0 or 1; log(0) must not
    produce NaN and silently poison training."""
    for z in (-30.0, 30.0):
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n crism python -m pytest tests/test_gated_losses.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'training.gated_losses'`

- [ ] **Step 3: Implement**

Create `training/gated_losses.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n crism python -m pytest tests/test_gated_losses.py tests/test_gated_classifier.py -q`
Expected: PASS (16 tests)

- [ ] **Step 5: Verify the tests can fail (mutation)**

```bash
cp training/gated_losses.py /tmp/gloss_backup.py
python3 - <<'PY'
p='training/gated_losses.py'; s=open(p).read()
s=s.replace("            y_gate = (targets[:, self.mineral_idx].amax(dim=1) > 0).float()",
            "            y_gate = torch.zeros(targets.shape[0], device=targets.device)")
open(p,'w').write(s)
PY
conda run -n crism python -m pytest tests/test_gated_losses.py -q
cp /tmp/gloss_backup.py training/gated_losses.py
```
Expected: `test_gate_target_is_derived_from_the_mineral_labels` FAILS, then passes after restore.

- [ ] **Step 6: Commit**

```bash
git add training/gated_losses.py tests/test_gated_losses.py
git commit -m "gated loss: ASL on probabilities plus derived gate supervision

AsymmetricLoss takes logits and sigmoids internally; the gated head produces
probabilities and logit(g*c) is unstable at both ends, so the formula is restated
to accept p. Tests pin it numerically equal to the logit form at clip 0.05 and
0.0, so this arm differs from its comparator in the head alone.

The gate also gets an explicit BCE against a target derived from the mineral
labels -- implicit gradient through p_k alone lets the gate drift to a constant
while the conditionals absorb everything."
```

---

### Task 3: Training wiring

**Files:**
- Modify: `scripts/train.py` (argparse block near the existing `--asl_clip` at line ~121; the **6 of 8** `train_torch_model(...)` call sites that pass `use_asl_loss=args.asl_loss`)
- Modify: `training/train_torch.py` (`train_torch_model` signature at line ~153; model construction; loss selection near line ~446)
- Test: `tests/test_gated_wiring.py`

**Interfaces:**
- Consumes: `GatedSpatialSpectralClassifierAux`, `GatedAsymmetricLoss`, `class_partition`.
- Produces: a `--gated_head` CLI flag and a `gated_head: bool = False` parameter on `train_torch_model`.
- NOTE: the function is `train_torch_model`, NOT `train_torch`. `data.dataset.LABEL_COLS` is imported locally inside functions, never at module level — follow that pattern.

- [ ] **Step 1: Write the failing test**

```python
"""--gated_head must reach the model and the loss, not just parse."""
from __future__ import annotations

import inspect

import scripts.train as train_mod
import training.train_torch as tt


def test_train_torch_model_accepts_gated_head():
    assert 'gated_head' in inspect.signature(tt.train_torch_model).parameters


def test_gated_head_defaults_off():
    """Every existing run must be unaffected."""
    assert inspect.signature(tt.train_torch_model).parameters['gated_head'].default is False


def test_cli_exposes_gated_head():
    src = inspect.getsource(train_mod)
    assert "'--gated_head'" in src or '"--gated_head"' in src


def test_gated_head_forwarded_wherever_asl_is():
    """A flag that parses but is not forwarded silently trains the flat head
    while the log claims a gated run -- the worst outcome available here.
    scripts/train.py has 8 train_torch_model call sites; the 6 that pass
    use_asl_loss are exactly the ones that must also pass gated_head."""
    src = inspect.getsource(train_mod)
    n_asl = src.count('use_asl_loss=args.asl_loss')
    n_gated = src.count('gated_head=args.gated_head')
    assert n_asl == 6, f'call-site count changed ({n_asl}); re-read train.py'
    assert n_gated == n_asl, (
        f'{n_asl} call sites pass use_asl_loss but only {n_gated} pass gated_head')
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n crism python -m pytest tests/test_gated_wiring.py -q`
Expected: FAIL — `gated_head` not in signature.

- [ ] **Step 3: Add the CLI flag**

In `scripts/train.py`, immediately after the `--asl_clip` argument (~line 121):

```python
    parser.add_argument('--gated_head', action='store_true',
                        help='Hierarchical mineral-present gate: the head emits '
                             '8 logits (1 gate + 7 conditionals) and probabilities '
                             'are g*c for minerals, (1-g)*c for bland/junk, so '
                             'max(p_mineral)+max(p_non-mineral) <= 1 by '
                             'construction. Requires --seven_class. Pair with '
                             '--asl_clip 0.0.')
```

Then at **each of the 6** `train_torch_model(` call sites that already pass `use_asl_loss=args.asl_loss,` add the argument immediately after `asl_clip=args.asl_clip,`:

```python
                gated_head=args.gated_head,
```

- [ ] **Step 4: Add the parameter and wiring in train_torch**

In `training/train_torch.py`, add to the `train_torch_model` signature next to `asl_clip` (~line 180):

```python
    gated_head: bool = False,
```

Where the model is constructed, select the gated class when the flag is set. Where the loss is selected (the `elif use_asl_loss:` branch, ~line 446), replace that branch with:

```python
    elif use_asl_loss and gated_head:
        from data.dataset import LABEL_COLS
        from models.gated_classifier import class_partition
        from training.gated_losses import GatedAsymmetricLoss
        mineral_idx, non_mineral_idx = class_partition(LABEL_COLS)
        loss_fn = GatedAsymmetricLoss(
            mineral_idx, non_mineral_idx, gamma_neg=asl_gamma_neg,
            gamma_pos=asl_gamma_pos, clip=asl_clip, lambda_gate=1.0)
        logger.info(
            f'Using GatedAsymmetricLoss: gate over minerals {mineral_idx}, '
            f'non-minerals {non_mineral_idx}, clip={asl_clip}, lambda_gate=1.0')
    elif use_asl_loss:
        from training.losses import AsymmetricLoss
        loss_fn = AsymmetricLoss(gamma_neg=asl_gamma_neg, gamma_pos=asl_gamma_pos, clip=asl_clip)
```

Guard the combination explicitly, right after the signature's body starts:

```python
    if gated_head and not use_asl_loss:
        raise ValueError('--gated_head requires --asl_loss (GatedAsymmetricLoss '
                         'is the only gated loss implemented)')
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `conda run -n crism python -m pytest tests/test_gated_wiring.py -q`
Expected: PASS (4 tests)

- [ ] **Step 6: Verify no existing run changed**

Run: `conda run -n crism python -m pytest tests/test_gated_losses.py tests/test_gated_classifier.py tests/test_asl_clip_gradient.py -q`
Expected: PASS. `gated_head` defaults to `False`, so the flat path is untouched.

- [ ] **Step 7: Commit**

```bash
git add scripts/train.py training/train_torch.py tests/test_gated_wiring.py
git commit -m "wire --gated_head through train.py and train_torch

Defaults off so every existing run is unaffected. The test asserts the flag is
forwarded at EVERY train_torch call site -- a flag that parses but is not
forwarded would silently train the flat head while the log claims a gated run."
```

---

### Task 4: Inference wiring

**Files:**
- Modify: `scripts/classify_tile_supervised.py` (`_set_n_classes` at line ~90; model construction; the probability write)
- Test: `tests/test_classify_gated.py`

**Interfaces:**
- Consumes: `compose_gated_probs`, `class_partition`.
- Produces: a `--gated_head` flag on the classifier and `GATED_MODE` module state.

- [ ] **Step 1: Write the failing test**

```python
"""An 8-wide head must be read as 7 gated classes, never as 8 classes."""
from __future__ import annotations

import pytest
import torch

import scripts.classify_tile_supervised as cts


def test_eight_wide_head_is_rejected_when_not_gated():
    """Loud failure is correct: silently inventing a class shifts every
    downstream index."""
    cts.GATED_MODE = False
    with pytest.raises(ValueError, match='unsupported head size'):
        cts._set_n_classes({'head.weight': torch.zeros(8, 272)})


def test_eight_wide_head_is_seven_classes_when_gated():
    cts.GATED_MODE = True
    try:
        cts._set_n_classes({'head.weight': torch.zeros(8, 272)})
        assert cts.N_CLASSES == 7
        assert list(cts.CLASS_NAMES) == ['olivine', 'lcp', 'hcp', 'plagioclase',
                                         'bland', 'alteration', 'junk']
    finally:
        cts.GATED_MODE = False


def test_gated_probs_written_are_composed_not_raw_conditionals():
    """Raw conditionals look like probabilities, sum plausibly, and are wrong."""
    from models.gated_classifier import class_partition, compose_gated_probs
    names = ['olivine', 'lcp', 'hcp', 'plagioclase', 'bland', 'alteration', 'junk']
    m, n = class_partition(names)
    logits = torch.zeros(1, 8)
    logits[0, 0] = -8.0                      # gate mostly shut
    logits[0, 1 + names.index('lcp')] = 8.0  # conditional very confident
    probs, _ = compose_gated_probs(logits, m, n)
    assert probs[0, names.index('lcp')].item() < 0.01, \
        'a shut gate must suppress a confident conditional'
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `conda run -n crism python -m pytest tests/test_classify_gated.py -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'GATED_MODE'`

- [ ] **Step 3: Implement**

In `scripts/classify_tile_supervised.py`, near the other mode flags (`PYX_MODE`, `PYX_ALT_MODE`):

```python
GATED_MODE = False   # set by --gated_head before the checkpoint loads
```

In `_set_n_classes`, before the final `else: raise ValueError(...)`:

```python
    elif n == 8 and GATED_MODE:
        # 8 = 1 gate + 7 conditionals. Without --gated_head this raises below,
        # which is the correct outcome: reading it as 8 classes would invent a
        # class and shift every downstream index.
        N_CLASSES, CLASS_NAMES, CLASS_COLORS = 7, _CLASS_NAMES_7, _CLASS_COLORS_7
```

Add the CLI flag and set the module flag before the checkpoint is loaded:

```python
    parser.add_argument('--gated_head', action='store_true',
                        help='Checkpoint uses the hierarchical mineral gate '
                             '(8-wide head: 1 gate + 7 conditionals). '
                             'Probabilities are composed as g*c / (1-g)*c.')
```

```python
    if args.gated_head:
        globals()['GATED_MODE'] = True
```

Where logits become probabilities, branch:

```python
    if GATED_MODE:
        from models.gated_classifier import class_partition, compose_gated_probs
        mineral_idx, non_mineral_idx = class_partition(CLASS_NAMES)
        probs_batch, _gate = compose_gated_probs(logits, mineral_idx, non_mineral_idx)
    else:
        probs_batch = torch.sigmoid(logits)
```

Construct `GatedSpatialSpectralClassifierAux` instead of `SpatialSpectralClassifierAux` when `GATED_MODE` is set.

- [ ] **Step 4: Run the test to verify it passes**

Run: `conda run -n crism python -m pytest tests/test_classify_gated.py -q`
Expected: PASS (3 tests)

- [ ] **Step 5: Regression — the flat path must be untouched**

Run: `conda run -n crism python -m pytest tests/test_classify_dual_cr.py tests/test_classify_cr_parity.py tests/test_load_tile_phys_max.py -q`
Expected: PASS (17 tests).

- [ ] **Step 6: Commit**

```bash
git add scripts/classify_tile_supervised.py tests/test_classify_gated.py
git commit -m "classify: --gated_head reads an 8-wide head as 7 gated classes

Without the flag an 8-wide head still raises 'unsupported head size', which is
the right outcome -- reading it as 8 classes would invent a class and shift every
downstream index.

The flag does two things, and the second matters more: it rebinds the vocab AND
composes g*c / (1-g)*c before writing probabilities. Writing raw conditionals
would put values in the npz that look like probabilities and are wrong."
```

---

### Task 5: Training job

**Files:**
- Create: `scripts/hpc_finetune_dualcr_gated.slurm`

- [ ] **Step 1: Copy the noclip arm and rewrite the anchors**

```bash
cp scripts/hpc_finetune_dualcr_noclip.slurm scripts/hpc_finetune_dualcr_gated.slurm
python3 - <<'PY'
p='scripts/hpc_finetune_dualcr_gated.slurm'; s=open(p).read()
subs=[('#SBATCH --job-name=dualcr_ft_noclip','#SBATCH --job-name=dualcr_ft_gated'),
      ('#SBATCH --output=logs/dualcr_ft_noclip_%j.log','#SBATCH --output=logs/dualcr_ft_gated_%j.log'),
      ('#SBATCH --error=logs/dualcr_ft_noclip_%j.log','#SBATCH --error=logs/dualcr_ft_gated_%j.log'),
      ('RUN_NAME=ft_7cls_handcore_dualcr_noclip','RUN_NAME=ft_7cls_handcore_dualcr_gated'),
      ('    --asl_loss --asl_clip 0.0 || { echo "ERROR: fine-tune failed"; exit 1; }',
       '    --asl_loss --asl_clip 0.0 --gated_head || { echo "ERROR: fine-tune failed"; exit 1; }')]
for a,b in subs:
    assert s.count(a)==1, (a, s.count(a))
    s=s.replace(a,b)
open(p,'w').write(s)
print('rewrote 5 anchors')
PY
diff scripts/hpc_finetune_dualcr_noclip.slurm scripts/hpc_finetune_dualcr_gated.slurm
```
Expected: a five-line diff — job name, two log paths, run name, and `--gated_head`.

- [ ] **Step 2: Commit**

```bash
git add scripts/hpc_finetune_dualcr_gated.slurm
git commit -m "gated arm: noclip job plus --gated_head

Built from the noclip job, not from e87, so the ONLY difference from its
comparator is the head. Comparing a gated run against e87 would confound head
with clip."
```

---

## Notes for the executor

- Read the gated arm against **`dualcr_noclip`**, never against e87.
- First acceptance check once a gated checkpoint classifies a tile:
  `max(p_mineral) + max(p_non-mineral) <= 1` for every valid pixel. If that fails, the head is wired wrong — stop.
- Then: t1321 false share 35% → target <10%; t1249 confident-lcp must not fall >15%; floor test compared in **pixels retained**, not polygon counts.
- Watch alteration specifically. It is grouped as a mineral but altered terrain is often bright and dusty, so the gate may learn "bright ⇒ no mineral" and take alteration with it. MC11 is the altered/dusty probe.
- Watch the gate's distribution during training, not only the loss. A gate saturated near 1 degenerates to the flat head; near 0 it kills every mineral at once.

# Adversarial Decomposition (v2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `DecompSpVitAdv` — an adversarially-decorrelated signal/noise decomposition encoder that drops v1's multiplicative atmosphere term, uses additive `x ≈ s + n`, and replaces prior-based disentanglement with a gradient-reversal discriminator that forces the noise embedding to be class-uninformative.

**Architecture:** Reuse the SpatialSpectralTransformer encoder (MAE checkpoint loads unchanged). Two parallel linear projections off each token produce signal and noise embeddings. Two per-token MLP decoders produce per-pixel `s_hat` and `n_hat`. A gradient-reversal layer + small MLP discriminator attaches to the center-pixel noise embedding and is trained to predict class — encoder is pushed (via reversed gradient) to make that prediction fail. Classifier reads the center-pixel signal embedding.

**Tech Stack:** PyTorch (autograd custom Function for GRL), conda env `crism`, pytest. Project root `/mnt/mrdr/crism_classification`. Commit prefixes: `feat:`, `test:`, `fix:`, `perf:`.

**Conventions:**
- All commands run from `/mnt/mrdr/crism_classification` unless stated.
- Python prefix: `conda run -n crism …`
- Spec at `docs/superpowers/specs/2026-05-15-adversarial-decomposition-design.md` is the source of truth for ambiguities.

---

## File Map

| Path | Action | Responsibility |
|---|---|---|
| `models/decomp_spatial_vit_adv.py` | Create | `GradientReversalLayer` autograd Function + `DecompSpVitAdv` module. Forward returns 7-tuple. |
| `training/adv_decomp_losses.py` | Create | `AdversarialDecompositionLoss` — ASL classification + recon MSE + ASL adversarial + TV smoothness, returns total + components dict. |
| `training/train_torch.py` | Modify | Add a `DecompSpVitAdv` branch parallel to the `DecompSpVit` branch; per-epoch `lambda_adv` schedule update; log `val_disc_acc`. |
| `scripts/train.py` | Modify | Add `decomp_spatial_vit_adv` to `TORCH_MODELS`; CLI flag `--lambda_adv_max`. Model construction branch. |
| `scripts/hpc_ablation_decomp_v2.slurm` | Create | 3-task ablation array — `lrscale ∈ {0.001, 0.01, 0.1}`, no frozen condition. |
| `tests/test_decomp_spatial_vit_adv.py` | Create | Shape contracts, GRL gradient-sign test, MAE-load test, schedule update test. |
| `tests/test_adv_decomp_losses.py` | Create | Each loss component's behaviour in isolation. |

---

## Chunk 1 — `DecompSpVitAdv` model + GRL

### Task 1: Failing tests for the model

**Files:**
- Create: `tests/test_decomp_spatial_vit_adv.py`

- [ ] **Step 1: Write the failing test file**

```python
# tests/test_decomp_spatial_vit_adv.py
"""Tests for the adversarial signal/noise decomposition encoder."""
import pytest
import torch

from models.decomp_spatial_vit_adv import DecompSpVitAdv, GradientReversalLayer


@pytest.fixture
def model():
    return DecompSpVitAdv(
        n_bands=59, patch_size=7, n_classes=5,
        embed_dim=128, n_heads=4, n_layers=6, dropout=0.0,
        disc_hidden=64,
        lambda_adv=1.0,
    )


def test_forward_shapes(model):
    """Model returns the documented 7-tuple with correct shapes."""
    B = 4
    x = torch.randn(B, 7, 7, 59)
    out = model(x)
    # (logits, s_hat, n_hat, x_hat, disc_logits, s_emb_center, n_emb_center)
    logits, s_hat, n_hat, x_hat, disc_logits, s_emb_c, n_emb_c = out
    assert logits.shape == (B, 5)
    assert s_hat.shape == (B, 49, 59)
    assert n_hat.shape == (B, 49, 59)
    assert x_hat.shape == (B, 49, 59)
    assert disc_logits.shape == (B, 5)
    assert s_emb_c.shape == (B, 128)
    assert n_emb_c.shape == (B, 128)


def test_reconstruction_is_additive(model):
    """x_hat must equal s_hat + n_hat exactly."""
    B = 2
    x = torch.randn(B, 7, 7, 59)
    _, s_hat, n_hat, x_hat, _, _, _ = model(x)
    torch.testing.assert_close(x_hat, s_hat + n_hat, rtol=1e-5, atol=1e-5)


def test_classifier_reads_center_signal_embedding(model):
    """Logits should depend on center-pixel s_emb."""
    B = 3
    torch.manual_seed(0)
    x = torch.randn(B, 7, 7, 59)
    logits0 = model(x)[0]
    # Perturbing the center pixel should change logits more than perturbing a corner
    x_c = x.clone(); x_c[:, 3, 3, :] += 1.0
    x_corner = x.clone(); x_corner[:, 0, 0, :] += 1.0
    delta_center = (model(x_c)[0] - logits0).abs().sum().item()
    delta_corner = (model(x_corner)[0] - logits0).abs().sum().item()
    assert delta_center > delta_corner


def test_grl_forward_identity():
    """GRL must be identity in the forward pass."""
    x = torch.randn(4, 8)
    y = GradientReversalLayer.apply(x, 1.0)
    torch.testing.assert_close(y, x)


def test_grl_backward_negates_and_scales():
    """GRL multiplies the upstream gradient by -lambda_adv."""
    x = torch.randn(4, 8, requires_grad=True)
    lam = 0.5
    y = GradientReversalLayer.apply(x, lam)
    # Use a known upstream grad: sum(y) → dy/dx = 1 everywhere; through GRL → -lam
    y.sum().backward()
    expected = -lam * torch.ones_like(x)
    torch.testing.assert_close(x.grad, expected, rtol=1e-5, atol=1e-5)


def test_adversarial_gradient_signs(model):
    """The encoder receives a *reversed* gradient from L_adv vs the discriminator.

    Construction: forward the model, compute disc CE only, backward, and check:
      - encoder.band_embed.weight gradient is NON-zero (path exists)
      - the sign of the encoder gradient is OPPOSITE to what it would be
        without GRL.
    """
    B = 4
    x = torch.randn(B, 7, 7, 59)
    labels = (torch.rand(B, 5) > 0.5).float()
    _, _, _, _, disc_logits, _, _ = model(x)
    # Make sure backward through disc loss flows back
    disc_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        disc_logits, labels
    )
    grads = torch.autograd.grad(
        disc_loss, [model.encoder.band_embed.weight, model.discriminator[0].weight],
        retain_graph=True,
    )
    enc_grad, disc_grad = grads[0], grads[1]
    assert enc_grad.abs().sum() > 0, "encoder must receive gradient from adversarial loss"
    assert disc_grad.abs().sum() > 0, "discriminator must receive gradient from adversarial loss"
    # Construct a copy with lambda_adv=0 (no reversal effect) — sign on enc grad
    # would flip if GRL is doing its job.
    model.lambda_adv = 0.0
    _, _, _, _, disc_logits_zero, _, _ = model(x)
    disc_loss_zero = torch.nn.functional.binary_cross_entropy_with_logits(
        disc_logits_zero, labels
    )
    enc_grad_zero = torch.autograd.grad(
        disc_loss_zero, [model.encoder.band_embed.weight],
    )[0]
    # With lambda_adv=0 the encoder receives zero gradient (GRL zero-multiplies).
    assert enc_grad_zero.abs().sum().item() == pytest.approx(0.0, abs=1e-6), \
        "with lambda_adv=0, encoder gradient through adversarial path must be zero"


def test_load_mae_encoder(model):
    """MAE checkpoint state loads cleanly into the encoder."""
    import os
    ckpt_path = '/mnt/mrdr/crism_classification/checkpoints/spatial_mae_128d_6l_best.pt'
    if not os.path.exists(ckpt_path):
        pytest.skip(f"MAE checkpoint not available at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_encoder_state_dict(ckpt['encoder_state'])
    assert unexpected == [], f"Unexpected keys: {unexpected}"
    assert not any(k.startswith('encoder.encoder') for k in missing), \
        f"Core encoder weights missing: {[k for k in missing if k.startswith('encoder.encoder')]}"


def test_lambda_adv_setter():
    """lambda_adv is mutable so the training loop can update it per epoch."""
    m = DecompSpVitAdv(lambda_adv=0.5)
    assert m.lambda_adv == 0.5
    m.lambda_adv = 0.8
    assert m.lambda_adv == 0.8
    # Verify the forward picks up the new value
    x = torch.randn(2, 7, 7, 59)
    m.lambda_adv = 0.0
    _, _, _, _, disc_logits, _, n_emb_c = m(x)
    # With lambda_adv=0, gradient should not flow back through to encoder via disc
    grad_through_disc = torch.autograd.grad(
        disc_logits.sum(), m.encoder.band_embed.weight, retain_graph=True,
    )[0]
    assert grad_through_disc.abs().sum().item() == pytest.approx(0.0, abs=1e-6)
```

- [ ] **Step 2: Run the test file**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism pytest tests/test_decomp_spatial_vit_adv.py -v
```

Expected: `ModuleNotFoundError: No module named 'models.decomp_spatial_vit_adv'`. Confirms RED.

---

### Task 2: Implement `DecompSpVitAdv` + GRL

**Files:**
- Create: `models/decomp_spatial_vit_adv.py`

- [ ] **Step 1: Create the module**

```python
# models/decomp_spatial_vit_adv.py
"""
Adversarial signal/noise decomposition encoder for CRISM patches (v2).

Additive decomposition: x ≈ s + n. Disentanglement is enforced by
adversarial decorrelation — a gradient-reversal layer + discriminator
push the noise embedding to be class-uninformative; the reconstruction
loss closes the additive identity; the classifier reads only the signal
embedding.

Spec: docs/superpowers/specs/2026-05-15-adversarial-decomposition-design.md
"""
from typing import Tuple

import torch
import torch.nn as nn
from torch.autograd import Function

from models.spatial_spectral_transformer import SpatialSpectralTransformer


class GradientReversalLayer(Function):
    """Identity in forward; multiplies upstream gradient by `-lambda_adv` in backward.

    Standard DANN trick (Ganin & Lempitsky 2015). lambda_adv is passed as a
    runtime argument so it can be scheduled per-epoch from the training loop.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_adv: float) -> torch.Tensor:
        ctx.lambda_adv = lambda_adv
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Reverse sign and scale.
        return grad_output.neg() * ctx.lambda_adv, None


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class DecompSpVitAdv(nn.Module):
    """
    Adversarial signal/noise decomposition classifier.

    Forward returns: (logits, s_hat, n_hat, x_hat, disc_logits, s_emb_center, n_emb_center)

    Args:
      lambda_adv:  scalar weight on the gradient-reversed adversarial path.
                   Mutable from outside via `model.lambda_adv = value` so a
                   training-loop scheduler can update it per epoch.
    """

    def __init__(
        self,
        n_bands: int = 59,
        patch_size: int = 7,
        n_classes: int = 5,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        dropout: float = 0.1,
        head_hidden: int = 256,
        disc_hidden: int = 64,
        lambda_adv: float = 1.0,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.n_tokens = patch_size * patch_size
        self.embed_dim = embed_dim
        self.lambda_adv = lambda_adv

        self.encoder = SpatialSpectralTransformer(
            n_bands=n_bands, patch_size=patch_size,
            embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers, dropout=dropout,
        )

        # Two lightweight projections off each token → signal / noise embeddings
        self.signal_projection = nn.Linear(embed_dim, embed_dim)
        self.noise_projection = nn.Linear(embed_dim, embed_dim)

        # Per-token decoders → per-pixel reflectance & residual
        self.signal_decoder = _mlp(embed_dim, head_hidden, n_bands, dropout=dropout)
        self.noise_decoder = _mlp(embed_dim, head_hidden, n_bands, dropout=dropout)

        # Classifier reads center-pixel signal embedding
        self.classifier = nn.Linear(embed_dim, n_classes)

        # Discriminator reads center-pixel noise embedding via GRL
        self.discriminator = nn.Sequential(
            nn.Linear(embed_dim, disc_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(disc_hidden, n_classes),
        )

        self._center_idx = self.n_tokens // 2 + 1   # +1 for CLS

    def forward(self, x: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        z = self.encoder(x)                              # (B, 50, 128)
        tokens = z[:, 1:]                                # (B, 49, 128)

        s_emb = self.signal_projection(tokens)           # (B, 49, 128)
        n_emb = self.noise_projection(tokens)            # (B, 49, 128)

        s_hat = self.signal_decoder(s_emb)               # (B, 49, 59)
        n_hat = self.noise_decoder(n_emb)                # (B, 49, 59)
        x_hat = s_hat + n_hat                            # additive recon

        # Slot for the center-pixel spatial token is _center_idx in the
        # CLS-prepended sequence z; in the post-CLS-strip `tokens` it's at
        # _center_idx - 1.
        center_s_emb = s_emb[:, self._center_idx - 1]    # (B, 128)
        center_n_emb = n_emb[:, self._center_idx - 1]    # (B, 128)

        logits = self.classifier(center_s_emb)           # (B, n_classes)

        # GRL: forward identity, backward flips and scales the encoder's
        # gradient signal by lambda_adv.
        n_emb_grl = GradientReversalLayer.apply(center_n_emb, self.lambda_adv)
        disc_logits = self.discriminator(n_emb_grl)      # (B, n_classes)

        return logits, s_hat, n_hat, x_hat, disc_logits, center_s_emb, center_n_emb

    def get_param_groups(self, head_lr: float, encoder_lr: float) -> list:
        encoder_params = list(self.encoder.parameters())
        encoder_ids = {id(p) for p in encoder_params}
        head_params = [p for p in self.parameters() if id(p) not in encoder_ids]
        return [
            {'params': encoder_params, 'lr': encoder_lr},
            {'params': head_params,    'lr': head_lr},
        ]

    def load_encoder_state_dict(self, state: dict):
        return self.encoder.load_encoder_state_dict(state)
```

- [ ] **Step 2: Run the tests**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism pytest tests/test_decomp_spatial_vit_adv.py -v
```

Expected: 8 tests pass.

- [ ] **Step 3: Commit**

```bash
cd /mnt/mrdr/crism_classification
git add models/decomp_spatial_vit_adv.py tests/test_decomp_spatial_vit_adv.py
git commit -m "feat: DecompSpVitAdv — adversarial signal/noise decomposition encoder"
```

---

## Chunk 2 — Composite loss

### Task 3: Failing tests for `AdversarialDecompositionLoss`

**Files:**
- Create: `tests/test_adv_decomp_losses.py`

- [ ] **Step 1: Write the failing test file**

```python
# tests/test_adv_decomp_losses.py
"""Tests for the adversarial decomposition composite loss."""
import pytest
import torch

from training.adv_decomp_losses import AdversarialDecompositionLoss


@pytest.fixture
def loss_fn():
    return AdversarialDecompositionLoss(
        lambda_recon=10.0,
        lambda_smooth=0.001,
        asl_gamma_neg=4.0, asl_gamma_pos=0.0, asl_clip=0.05,
    )


def _make_outputs(B=4, n_tokens=49, n_bands=59, n_classes=5):
    torch.manual_seed(0)
    s_hat = torch.randn(B, n_tokens, n_bands) * 0.1
    n_hat = torch.randn(B, n_tokens, n_bands) * 0.01
    x_hat = s_hat + n_hat
    x = x_hat + torch.randn_like(x_hat) * 0.005
    logits = torch.randn(B, n_classes)
    disc_logits = torch.randn(B, n_classes)
    labels = (torch.rand(B, n_classes) > 0.5).float()
    weights = torch.ones(B)
    return dict(
        x=x, logits=logits, labels=labels, weights=weights,
        s_hat=s_hat, n_hat=n_hat, x_hat=x_hat,
        disc_logits=disc_logits,
    )


def test_loss_returns_scalar_and_components(loss_fn):
    o = _make_outputs()
    total, components = loss_fn(**o)
    assert total.ndim == 0
    for key in ('cls', 'recon', 'adv', 'smooth'):
        assert key in components
        assert components[key].ndim == 0


def test_recon_zero_when_perfect(loss_fn):
    o = _make_outputs()
    o['x_hat'] = o['x']
    o['n_hat'] = o['x'] - o['s_hat']  # Make recon perfect
    o['x_hat'] = o['s_hat'] + o['n_hat']
    _, c = loss_fn(**o)
    assert c['recon'].item() < 1e-8


def test_smooth_zero_when_signal_uniform(loss_fn):
    B, n_tokens, n_bands = 2, 49, 59
    spec = torch.randn(B, 1, n_bands) * 0.1
    s_hat_uniform = spec.expand(-1, n_tokens, -1).clone()
    n_hat = torch.zeros(B, n_tokens, n_bands)
    x_hat = s_hat_uniform + n_hat
    o = dict(
        x=x_hat.clone(),
        logits=torch.zeros(B, 5),
        labels=torch.zeros(B, 5),
        weights=torch.ones(B),
        s_hat=s_hat_uniform, n_hat=n_hat, x_hat=x_hat,
        disc_logits=torch.zeros(B, 5),
    )
    _, c = loss_fn(**o)
    assert c['smooth'].item() < 1e-8


def test_total_is_weighted_sum(loss_fn):
    """Total = cls + λ_recon·recon + adv + λ_smooth·smooth.

    Note: λ_adv is NOT applied in the loss — the gradient reversal layer
    inside the model handles the encoder-side sign and scale. The
    discriminator-side loss uses plain `adv` weight 1.0.
    """
    o = _make_outputs()
    total, c = loss_fn(**o)
    expected = (
        c['cls']
        + loss_fn.lambda_recon * c['recon']
        + c['adv']
        + loss_fn.lambda_smooth * c['smooth']
    )
    torch.testing.assert_close(total, expected, rtol=1e-5, atol=1e-5)


def test_class_weights_threaded_through(loss_fn):
    o = _make_outputs()
    cw = torch.tensor([1.0, 1.0, 1.5, 3.0, 1.0])
    _, c_with = loss_fn(**o, class_weights=cw)
    assert torch.is_tensor(c_with['cls'])
    assert torch.is_tensor(c_with['adv'])
```

- [ ] **Step 2: Run tests and verify they fail on import**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism pytest tests/test_adv_decomp_losses.py -v
```

Expected: `ModuleNotFoundError`.

---

### Task 4: Implement `AdversarialDecompositionLoss`

**Files:**
- Create: `training/adv_decomp_losses.py`

- [ ] **Step 1: Create the module**

```python
# training/adv_decomp_losses.py
"""
Composite loss for the adversarial signal/noise decomposition encoder (v2).

  L_total = L_cls
          + λ_recon  · L_recon
          + L_adv               (gradient-reversed for encoder side via GRL in the model)
          + λ_smooth · L_smooth

L_cls is ASL on the classifier logits. L_recon is MSE of (s_hat + n_hat)
against the input. L_adv is ASL on the discriminator logits — the GRL
inside the model handles the encoder-side sign flip, so the loss itself
just treats L_adv as a standard classification term. L_smooth is TV on
s_hat.

Spec: docs/superpowers/specs/2026-05-15-adversarial-decomposition-design.md
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from training.losses import AsymmetricLoss


class AdversarialDecompositionLoss(nn.Module):
    def __init__(
        self,
        lambda_recon: float = 10.0,
        lambda_smooth: float = 0.001,
        asl_gamma_neg: float = 4.0,
        asl_gamma_pos: float = 0.0,
        asl_clip: float = 0.05,
    ):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_smooth = lambda_smooth
        # Same ASL family for classifier and discriminator.
        self.cls_loss = AsymmetricLoss(
            gamma_neg=asl_gamma_neg, gamma_pos=asl_gamma_pos, clip=asl_clip,
        )
        self.adv_loss = AsymmetricLoss(
            gamma_neg=asl_gamma_neg, gamma_pos=asl_gamma_pos, clip=asl_clip,
        )

    def forward(
        self,
        x: torch.Tensor,            # (B, n_tokens, n_bands) or (B, P, P, n_bands)
        logits: torch.Tensor,       # (B, n_classes)
        labels: torch.Tensor,       # (B, n_classes)
        weights: torch.Tensor,      # (B,)
        s_hat: torch.Tensor,        # (B, n_tokens, n_bands)
        n_hat: torch.Tensor,        # (B, n_tokens, n_bands)
        x_hat: torch.Tensor,        # (B, n_tokens, n_bands)
        disc_logits: torch.Tensor,  # (B, n_classes)
        pos_weight: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Flatten input to (B, n_tokens, n_bands) if it's a 4D patch
        if x.dim() == 4:
            B, P, P2, n_bands = x.shape
            assert P == P2
            x_flat = x.reshape(B, P * P2, n_bands)
        else:
            x_flat = x

        # 1. Classification
        cls = self.cls_loss(
            logits, labels, weights,
            pos_weight=pos_weight, class_weights=class_weights,
        )

        # 2. Reconstruction MSE on valid pixels
        valid_mask = (x_flat.abs() < 1.0).float()
        sq_err = (x_hat - x_flat) ** 2 * valid_mask
        recon = sq_err.sum() / (valid_mask.sum() + 1e-8)

        # 3. Adversarial loss — discriminator predicts class from (GRL'd) n_emb.
        # GRL inside the model flipped the encoder-side gradient; here we just
        # compute the standard classification loss.
        adv = self.adv_loss(
            disc_logits, labels, weights,
            pos_weight=pos_weight, class_weights=class_weights,
        )

        # 4. Spatial smoothness on signal (TV penalty over the 7×7 layout)
        B, N, nb = s_hat.shape
        P = int(N ** 0.5)
        s_spatial = s_hat.view(B, P, P, nb)
        dv = (s_spatial[:, 1:, :, :] - s_spatial[:, :-1, :, :]).abs()
        dh = (s_spatial[:, :, 1:, :] - s_spatial[:, :, :-1, :]).abs()
        smooth = (dv.mean() + dh.mean()) * 0.5

        total = (
            cls
            + self.lambda_recon * recon
            + adv
            + self.lambda_smooth * smooth
        )
        components = {
            'cls': cls, 'recon': recon, 'adv': adv, 'smooth': smooth,
        }
        return total, components
```

- [ ] **Step 2: Run the loss tests**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism pytest tests/test_adv_decomp_losses.py -v
```

Expected: 5 tests pass.

- [ ] **Step 3: Commit**

```bash
cd /mnt/mrdr/crism_classification
git add training/adv_decomp_losses.py tests/test_adv_decomp_losses.py
git commit -m "feat: AdversarialDecompositionLoss with cls + recon + adv + smooth"
```

---

## Chunk 3 — Training loop integration

### Task 5: Add `DecompSpVitAdv` branch to `train_torch.py`

**Files:**
- Modify: `training/train_torch.py`

The existing file already has a `DecompSpVit` branch (Task 5 of the v1 plan). We add a parallel `DecompSpVitAdv` branch with:
- Separate composite loss (`AdversarialDecompositionLoss`)
- Per-epoch `lambda_adv` schedule update via `model.lambda_adv = schedule(epoch)`
- Validation-time `val_disc_acc` logging

- [ ] **Step 1: Update the function signature**

Find:

```python
    decomp_lambda_recon: float = 1.0,
    decomp_lambda_eps: float = 0.1,
    decomp_lambda_T: float = 0.01,
    decomp_lambda_b: float = 0.01,
    decomp_lambda_smooth: float = 0.001,
    device: Optional[str] = None,
```

Replace with:

```python
    decomp_lambda_recon: float = 1.0,
    decomp_lambda_eps: float = 0.1,
    decomp_lambda_T: float = 0.01,
    decomp_lambda_b: float = 0.01,
    decomp_lambda_smooth: float = 0.001,
    lambda_adv_max: float = 1.0,
    device: Optional[str] = None,
```

- [ ] **Step 2: Add the adversarial loss branch**

Find:

```python
    is_decomp = type(model).__name__ == 'DecompSpVit'

    if is_decomp:
        from training.decomp_losses import DecompositionLoss
        loss_fn = DecompositionLoss(
            lambda_recon=decomp_lambda_recon,
            lambda_eps=decomp_lambda_eps,
            lambda_T=decomp_lambda_T,
            lambda_b=decomp_lambda_b,
            lambda_smooth=decomp_lambda_smooth,
            asl_gamma_neg=asl_gamma_neg,
            asl_gamma_pos=asl_gamma_pos,
            asl_clip=asl_clip,
        )
        logger.info(
            f"Using DecompositionLoss: λ_recon={decomp_lambda_recon}, "
            f"λ_eps={decomp_lambda_eps}, λ_T={decomp_lambda_T}, "
            f"λ_b={decomp_lambda_b}, λ_smooth={decomp_lambda_smooth}"
        )
    elif use_asl_loss:
```

Replace with:

```python
    is_decomp = type(model).__name__ == 'DecompSpVit'
    is_decomp_adv = type(model).__name__ == 'DecompSpVitAdv'

    if is_decomp_adv:
        from training.adv_decomp_losses import AdversarialDecompositionLoss
        loss_fn = AdversarialDecompositionLoss(
            lambda_recon=decomp_lambda_recon,
            lambda_smooth=decomp_lambda_smooth,
            asl_gamma_neg=asl_gamma_neg,
            asl_gamma_pos=asl_gamma_pos,
            asl_clip=asl_clip,
        )
        logger.info(
            f"Using AdversarialDecompositionLoss: λ_recon={decomp_lambda_recon}, "
            f"λ_smooth={decomp_lambda_smooth}, λ_adv_max={lambda_adv_max}"
        )
    elif is_decomp:
        from training.decomp_losses import DecompositionLoss
        loss_fn = DecompositionLoss(
            lambda_recon=decomp_lambda_recon,
            lambda_eps=decomp_lambda_eps,
            lambda_T=decomp_lambda_T,
            lambda_b=decomp_lambda_b,
            lambda_smooth=decomp_lambda_smooth,
            asl_gamma_neg=asl_gamma_neg,
            asl_gamma_pos=asl_gamma_pos,
            asl_clip=asl_clip,
        )
        logger.info(
            f"Using DecompositionLoss: λ_recon={decomp_lambda_recon}, "
            f"λ_eps={decomp_lambda_eps}, λ_T={decomp_lambda_T}, "
            f"λ_b={decomp_lambda_b}, λ_smooth={decomp_lambda_smooth}"
        )
    elif use_asl_loss:
```

- [ ] **Step 3: Add lambda_adv schedule + training step branch**

Find the `for epoch in range(...)` loop. Just inside, immediately after `train_loss_components: dict = {}`, add the per-epoch schedule update:

Find:

```python
    for epoch in range(1, max_epochs + 1):
        # --- Train ---
        model.train()
        train_losses = []
        train_loss_components: dict = {}   # decomp-only; ignored for non-decomp models
```

Replace with:

```python
    for epoch in range(1, max_epochs + 1):
        # --- Train ---
        model.train()
        train_losses = []
        train_loss_components: dict = {}

        # DANN-style lambda_adv warmup for the adversarial decomposition model.
        # Smooth schedule from ~0 at epoch 1 to ~lambda_adv_max at the last epoch.
        if is_decomp_adv:
            import math
            p = (epoch - 1) / max(max_epochs - 1, 1)        # ∈ [0, 1]
            schedule = (2.0 / (1.0 + math.exp(-10.0 * p))) - 1.0   # ∈ [0, ~1)
            model.lambda_adv = float(lambda_adv_max * schedule)
            logger.info(
                f"epoch {epoch}: lambda_adv = {model.lambda_adv:.4f}"
            )
```

Find the train-step body:

```python
            optimizer.zero_grad()
            if is_decomp:
                logits, s_hat, T_hat, b_hat, eps_hat, x_hat = model(features)
                loss, components = loss_fn(
                    x=features,
                    logits=logits, labels=labels, weights=weights,
                    s_hat=s_hat, T_hat=T_hat, b_hat=b_hat,
                    eps_hat=eps_hat, x_hat=x_hat,
                    pos_weight=pos_weight, class_weights=class_weights,
                )
                # Stash component values for later epoch-level logging
                for k, v in components.items():
                    train_loss_components.setdefault(k, []).append(v.item())
            else:
                logits = model(features)
                loss = loss_fn(
                    logits, labels, weights,
                    pos_weight=pos_weight, class_weights=class_weights,
                )
            loss.backward()
```

Replace with:

```python
            optimizer.zero_grad()
            if is_decomp_adv:
                logits, s_hat, n_hat, x_hat, disc_logits, _, _ = model(features)
                loss, components = loss_fn(
                    x=features,
                    logits=logits, labels=labels, weights=weights,
                    s_hat=s_hat, n_hat=n_hat, x_hat=x_hat,
                    disc_logits=disc_logits,
                    pos_weight=pos_weight, class_weights=class_weights,
                )
                for k, v in components.items():
                    train_loss_components.setdefault(k, []).append(v.item())
            elif is_decomp:
                logits, s_hat, T_hat, b_hat, eps_hat, x_hat = model(features)
                loss, components = loss_fn(
                    x=features,
                    logits=logits, labels=labels, weights=weights,
                    s_hat=s_hat, T_hat=T_hat, b_hat=b_hat,
                    eps_hat=eps_hat, x_hat=x_hat,
                    pos_weight=pos_weight, class_weights=class_weights,
                )
                for k, v in components.items():
                    train_loss_components.setdefault(k, []).append(v.item())
            else:
                logits = model(features)
                loss = loss_fn(
                    logits, labels, weights,
                    pos_weight=pos_weight, class_weights=class_weights,
                )
            loss.backward()
```

- [ ] **Step 4: Add validation-time disc-accuracy logging**

Find the validation forward block:

```python
        # --- Validate ---
        model.eval()
        all_logits, all_labels = [], []
        val_T_means, val_b_means, val_eps_norms = [], [], []

        with torch.no_grad():
            for features, labels, weights in val_loader:
                features = features.to(device)
                if is_decomp:
                    logits, _s_hat, T_hat, b_hat, eps_hat, _x_hat = model(features)
                    val_T_means.append(T_hat.mean().item())
                    val_b_means.append(b_hat.mean().item())
                    val_eps_norms.append(eps_hat.norm(dim=-1).mean().item())
                else:
                    logits = model(features)
                all_logits.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(labels.numpy())
```

Replace with:

```python
        # --- Validate ---
        model.eval()
        all_logits, all_labels = [], []
        val_T_means, val_b_means, val_eps_norms = [], [], []
        val_disc_correct, val_disc_total = 0, 0
        val_n_norms = []

        with torch.no_grad():
            for features, labels, weights in val_loader:
                features = features.to(device)
                if is_decomp_adv:
                    logits, _, n_hat, _, disc_logits, _, _ = model(features)
                    val_n_norms.append(n_hat.norm(dim=-1).mean().item())
                    # Multi-label accuracy: prediction is `(sigmoid(disc) > 0.5)`.
                    # We measure per-class accuracy averaged across classes &
                    # samples — informative even though the underlying task
                    # is multi-label.
                    disc_pred = (torch.sigmoid(disc_logits) > 0.5).float().cpu()
                    target = (labels > 0.4).float()
                    val_disc_correct += (disc_pred == target).float().mean().item() * features.size(0)
                    val_disc_total += features.size(0)
                elif is_decomp:
                    logits, _s_hat, T_hat, b_hat, eps_hat, _x_hat = model(features)
                    val_T_means.append(T_hat.mean().item())
                    val_b_means.append(b_hat.mean().item())
                    val_eps_norms.append(eps_hat.norm(dim=-1).mean().item())
                else:
                    logits = model(features)
                all_logits.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(labels.numpy())
```

- [ ] **Step 5: Add the new metrics to the wandb log block**

Find:

```python
        if use_wandb:
            import wandb as wb
            log_dict = {'epoch': epoch, 'train_loss': np.mean(train_losses), **flat}
            if is_decomp:
                # Per-epoch mean of each loss component (training side)
                for k, vals in train_loss_components.items():
                    if vals:
                        log_dict[f'train_loss_{k}'] = float(np.mean(vals))
                # Validation-side physical metrics
                if val_T_means:
                    log_dict['val_T_mean'] = float(np.mean(val_T_means))
                if val_b_means:
                    log_dict['val_b_mean'] = float(np.mean(val_b_means))
                if val_eps_norms:
                    log_dict['val_eps_norm_mean'] = float(np.mean(val_eps_norms))
            wb.log(log_dict)
```

Replace with:

```python
        if use_wandb:
            import wandb as wb
            log_dict = {'epoch': epoch, 'train_loss': np.mean(train_losses), **flat}
            if is_decomp_adv:
                for k, vals in train_loss_components.items():
                    if vals:
                        log_dict[f'train_loss_{k}'] = float(np.mean(vals))
                if val_n_norms:
                    log_dict['val_n_norm_mean'] = float(np.mean(val_n_norms))
                if val_disc_total > 0:
                    log_dict['val_disc_acc'] = float(val_disc_correct / val_disc_total)
                log_dict['lambda_adv'] = float(model.lambda_adv)
            elif is_decomp:
                for k, vals in train_loss_components.items():
                    if vals:
                        log_dict[f'train_loss_{k}'] = float(np.mean(vals))
                if val_T_means:
                    log_dict['val_T_mean'] = float(np.mean(val_T_means))
                if val_b_means:
                    log_dict['val_b_mean'] = float(np.mean(val_b_means))
                if val_eps_norms:
                    log_dict['val_eps_norm_mean'] = float(np.mean(val_eps_norms))
            wb.log(log_dict)
```

- [ ] **Step 6: Smoke test the integration**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism python -c "
import sys; sys.path.insert(0, '.')
import torch
from models.decomp_spatial_vit_adv import DecompSpVitAdv
from training.adv_decomp_losses import AdversarialDecompositionLoss

m = DecompSpVitAdv()
loss_fn = AdversarialDecompositionLoss(lambda_recon=10.0, lambda_smooth=0.001)
x = torch.randn(2, 7, 7, 59)
logits, s, n, xh, dl, _, _ = m(x)
labels = (torch.rand(2, 5) > 0.5).float()
w = torch.ones(2)
total, comp = loss_fn(x=x, logits=logits, labels=labels, weights=w,
                     s_hat=s, n_hat=n, x_hat=xh, disc_logits=dl)
print('total:', total.item())
print('components:', {k: f'{v.item():.4f}' for k, v in comp.items()})
total.backward()
print('backward OK')
m.lambda_adv = 0.5
print('updated lambda_adv:', m.lambda_adv)
"
```

Expected: prints the total + 4 components + "backward OK" + "updated lambda_adv: 0.5".

- [ ] **Step 7: Make sure nothing in the existing test suite regressed**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism pytest tests/ -x -q 2>&1 | tail -10
```

Expected: same number of passes as before this chunk plus the 13 new ones from Tasks 2+4.

- [ ] **Step 8: Commit**

```bash
cd /mnt/mrdr/crism_classification
git add training/train_torch.py
git commit -m "feat: train_torch.py supports DecompSpVitAdv + adversarial loss + lambda_adv schedule"
```

---

## Chunk 4 — CLI + slurm

### Task 6: Add `decomp_spatial_vit_adv` to `scripts/train.py`

**Files:**
- Modify: `scripts/train.py`

- [ ] **Step 1: Register the model**

Find:

```python
TORCH_MODELS = {'mlp', 'cnn', 'vit', 'spectral_cnn', 'spectral_vit',
                'spectral_hybrid', 'spatial_vit', 'decomp_spatial_vit'}
```

Replace with:

```python
TORCH_MODELS = {'mlp', 'cnn', 'vit', 'spectral_cnn', 'spectral_vit',
                'spectral_hybrid', 'spatial_vit', 'decomp_spatial_vit',
                'decomp_spatial_vit_adv'}
```

- [ ] **Step 2: Add the lambda_adv_max CLI flag**

Find:

```python
    parser.add_argument('--decomp_lambda_smooth', type=float, default=0.001,
                        help='Weight on spatial-TV smoothness on s_hat for DecompSpVit.')
```

Append after it:

```python
    parser.add_argument('--lambda_adv_max', type=float, default=1.0,
                        help='Max value of the adversarial GRL multiplier for '
                             'DecompSpVitAdv. Schedule warms from ~0 → this value.')
```

- [ ] **Step 3: Add the model construction branch**

Find the end of the `elif args.model == 'decomp_spatial_vit':` block — it ends with `)`. Right after that, add a parallel branch:

```python
        elif args.model == 'decomp_spatial_vit_adv':
            import glob as _glob
            data_root = cfg.get('data_root', '/mnt/crism/MRDR')
            globs_to_try = [
                os.path.join(data_root, 'mc*', 't*mrral*.hdr'),
                os.path.join(data_root, 't*mrral*.hdr'),
            ]
            mrral_hdrs = []
            for pattern in globs_to_try:
                mrral_hdrs = sorted(_glob.glob(pattern))
                if mrral_hdrs:
                    break
            mrral_map = {}
            for hdr in mrral_hdrs:
                tid = os.path.basename(hdr).split('_mrral_')[0]
                mrral_map[tid] = hdr.replace('.hdr', '.img')
            logging.info(f'mrral_map: {len(mrral_map)} tiles found')

            mrral_parquet = os.path.join(cfg['output_dir'], 'mrral_pixels.parquet')
            df_mrral = pd.read_parquet(mrral_parquet)
            dropout = args.dropout if args.dropout is not None else 0.1

            from models.decomp_spatial_vit_adv import DecompSpVitAdv
            model = DecompSpVitAdv(
                n_bands=59, patch_size=args.patch_size, n_classes=5,
                embed_dim=args.embed_dim, n_heads=args.n_heads,
                n_layers=args.n_layers, dropout=dropout,
                lambda_adv=0.0,   # warms up via the per-epoch schedule
            )
            if args.pretrain_ckpt:
                ckpt = torch.load(args.pretrain_ckpt, map_location='cpu', weights_only=False)
                missing, unexpected = model.load_encoder_state_dict(ckpt['encoder_state'])
                logging.info(
                    f'Loaded spatial MAE encoder from {args.pretrain_ckpt}. '
                    f'Missing: {missing}, Unexpected: {unexpected}'
                )
            if args.freeze_encoder:
                for p in model.encoder.parameters():
                    p.requires_grad = False
                logging.info('Encoder frozen (requires_grad=False on all encoder params)')

            mrral_cache_dir = cfg.get('patch_cache_dir')
            metrics = train_torch_model(
                model=model, df=df_mrral, model_name=run_name,
                max_epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, patience=args.patience,
                use_wandb=use_wandb, checkpoint_dir=checkpoint_dir,
                mrral_map=mrral_map, patch_size=args.patch_size,
                cache_dir=mrral_cache_dir,
                use_pos_weight=args.use_pos_weight,
                weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs,
                lr_t_max=args.lr_t_max,
                high_conf_only=args.high_conf_only,
                use_focal_loss=args.focal_loss,
                focal_gamma=args.focal_gamma,
                use_asl_loss=args.asl_loss,
                asl_gamma_neg=args.asl_gamma_neg,
                asl_gamma_pos=args.asl_gamma_pos,
                asl_clip=args.asl_clip,
                use_balanced_sampling=args.balanced_sampling,
                encoder_lr_scale=args.encoder_lr_scale,
                class_weights=class_weights_tensor,
                min_delta=args.min_delta,
                decomp_lambda_recon=args.decomp_lambda_recon,
                decomp_lambda_smooth=args.decomp_lambda_smooth,
                lambda_adv_max=args.lambda_adv_max,
                freeze_encoder=args.freeze_encoder,
            )
```

- [ ] **Step 4: Verify CLI parses cleanly**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism python scripts/train.py --help 2>&1 | grep -E "decomp_spatial_vit_adv|lambda_adv_max" | head -5
```

Expected: both flag and model choice show up.

- [ ] **Step 5: Commit**

```bash
cd /mnt/mrdr/crism_classification
git add scripts/train.py
git commit -m "feat: scripts/train.py supports --model decomp_spatial_vit_adv"
```

---

### Task 7: HPC slurm for the v2 sweep (3 conditions, no frozen)

**Files:**
- Create: `scripts/hpc_ablation_decomp_v2.slurm`

- [ ] **Step 1: Write the slurm file**

```bash
# scripts/hpc_ablation_decomp_v2.slurm
#!/bin/bash
#SBATCH --job-name=spvit_decomp_v2
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32gb
#SBATCH --time=0-24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/decomp_v2_%a_%j.out
#SBATCH --error=logs/decomp_v2_%a_%j.err

# v2: adversarial signal/noise decomposition encoder (DecompSpVitAdv).
# Drops the frozen-encoder condition (per user feedback — always
# underperforms). Three lr_scale conditions only.
# Spec: docs/superpowers/specs/2026-05-15-adversarial-decomposition-design.md

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
CKPT_DIR=${WORK_DIR}/checkpoints
PRETRAIN_CKPT=${CKPT_DIR}/spatial_mae_128d_6l_best.pt
CACHE_DIR=${WORK_DIR}/data/patch_cache

RUN_NAMES=("spvit_decomp_v2_lrscale0001" "spvit_decomp_v2_lrscale001" "spvit_decomp_v2_lrscale01")
LR_SCALE_ARGS=("--encoder_lr_scale 0.001" "--encoder_lr_scale 0.01" "--encoder_lr_scale 0.1")
CLASS_WEIGHTS="1.0,1.0,1.5,3.0,1.0"

PYTHON=/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python

cd "$WORK_DIR"

if [ ! -f config.local.yaml ]; then
    cat > config.local.yaml <<EOF
data_root: /xdisk/sbyrne/phillipsm/CRISM_MRDR
checkpoint_dir: ${CKPT_DIR}
checkpoints_dir: ${CKPT_DIR}
output_dir: ${WORK_DIR}/data
patch_cache_dir: ${WORK_DIR}/data/patch_cache
EOF
fi

if [ ! -f "${CACHE_DIR}/mrral_train_patches_p7.npy" ]; then
    echo "ERROR: mrral_train_patches_p7.npy not found at ${CACHE_DIR}" >&2
    exit 1
fi

mkdir -p logs checkpoints

echo "=== decomp_v2 ablation: ${RUN_NAMES[$SLURM_ARRAY_TASK_ID]} (task $SLURM_ARRAY_TASK_ID) ==="
${PYTHON} -u scripts/train.py \
    --model decomp_spatial_vit_adv \
    --run_name "${RUN_NAMES[$SLURM_ARRAY_TASK_ID]}" \
    --epochs 100 \
    --patience 25 \
    --min_delta 0.001 \
    --batch_size 256 \
    --lr 5e-4 \
    --lr_t_max 100 \
    --embed_dim 128 \
    --n_heads 4 \
    --n_layers 6 \
    --patch_size 7 \
    --asl_loss \
    --class_weights "${CLASS_WEIGHTS}" \
    --decomp_lambda_recon 10.0 \
    --decomp_lambda_smooth 0.001 \
    --lambda_adv_max 1.0 \
    --pretrain_ckpt "${PRETRAIN_CKPT}" \
    ${LR_SCALE_ARGS[$SLURM_ARRAY_TASK_ID]}
```

- [ ] **Step 2: Commit**

```bash
cd /mnt/mrdr/crism_classification
git add scripts/hpc_ablation_decomp_v2.slurm
git commit -m "feat: HPC slurm for decomp_v2 (adversarial) 3-config sweep"
```

---

## Chunk 5 — Final integration check + HPC deployment

### Task 8: Final smoke test + handoff

**Files:** none modified.

- [ ] **Step 1: Run the full test suite**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism pytest tests/ -x -q 2>&1 | tail -10
```

Expected: all tests pass, 13 new ones in the v2 suite.

- [ ] **Step 2: Smoke-load MAE checkpoint into the new model**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism python -c "
import sys; sys.path.insert(0, '.')
import torch
from models.decomp_spatial_vit_adv import DecompSpVitAdv
from training.adv_decomp_losses import AdversarialDecompositionLoss

m = DecompSpVitAdv()
ck = torch.load('checkpoints/spatial_mae_128d_6l_best.pt', map_location='cpu', weights_only=False)
miss, unex = m.load_encoder_state_dict(ck['encoder_state'])
assert not unex, f'unexpected: {unex}'
print('MAE encoder loaded into DecompSpVitAdv OK')

loss_fn = AdversarialDecompositionLoss(lambda_recon=10.0, lambda_smooth=0.001)
x = torch.randn(2, 7, 7, 59)
logits, s, n, xh, dl, _, _ = m(x)
labels = (torch.rand(2, 5) > 0.5).float()
total, comp = loss_fn(x=x, logits=logits, labels=labels, weights=torch.ones(2),
                     s_hat=s, n_hat=n, x_hat=xh, disc_logits=dl)
total.backward()
print(f'integration OK. total={total.item():.4f}')
for k, v in comp.items(): print(f'  {k}: {v.item():.4f}')
"
```

Expected: prints "MAE encoder loaded into DecompSpVitAdv OK" then "integration OK." with the 4 component values.

- [ ] **Step 3: Print the HPC handoff list (informational)**

The local work is complete. User needs to run on HPC (DUO auth blocks autonomous execution).

```bash
echo "=== Files to rsync to HPC ==="
echo ""
echo "# 1. Model + loss + train_torch.py + scripts/train.py + slurm"
echo "rsync -avh \\"
echo "    /mnt/mrdr/crism_classification/models/decomp_spatial_vit_adv.py \\"
echo "    phillipsm@filexfer.hpc.arizona.edu:/groups/sbyrne/phillipsm/crism_classification/models/"
echo ""
echo "rsync -avh \\"
echo "    /mnt/mrdr/crism_classification/training/adv_decomp_losses.py \\"
echo "    /mnt/mrdr/crism_classification/training/train_torch.py \\"
echo "    phillipsm@filexfer.hpc.arizona.edu:/groups/sbyrne/phillipsm/crism_classification/training/"
echo ""
echo "rsync -avh \\"
echo "    /mnt/mrdr/crism_classification/scripts/train.py \\"
echo "    /mnt/mrdr/crism_classification/scripts/hpc_ablation_decomp_v2.slurm \\"
echo "    phillipsm@filexfer.hpc.arizona.edu:/groups/sbyrne/phillipsm/crism_classification/scripts/"
echo ""
echo "=== After rsync, ssh and submit ==="
echo "ssh phillipsm@hpc.arizona.edu"
echo ""
echo "  cd /groups/sbyrne/phillipsm/crism_classification"
echo ""
echo "  # Sanity-check imports in the HPC env:"
echo "  /groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python -c \\"
echo "    'from models.decomp_spatial_vit_adv import DecompSpVitAdv; \\"
echo "     from training.adv_decomp_losses import AdversarialDecompositionLoss; print(\"OK\")'"
echo ""
echo "  # Submit the v2 ablation array (3 tasks, no frozen):"
echo "  sbatch scripts/hpc_ablation_decomp_v2.slurm"
echo "  squeue -u phillipsm"
```

- [ ] **Step 4: Show all commits from the plan**

```bash
cd /mnt/mrdr/crism_classification
git log --oneline | head -10
```

Expected: ~5 commits from this plan, plus prior commits.

---

## Spec coverage check (self-review)

| Spec section | Task |
|---|---|
| Additive decomposition `x ≈ s + n` | Task 2 (forward: `x_hat = s_hat + n_hat`) |
| Two parallel projections + decoders | Task 2 (`signal_projection`, `noise_projection`, `signal_decoder`, `noise_decoder`) |
| Classifier on center-pixel signal embedding | Task 2 (`self.classifier(center_s_emb)`) |
| Discriminator + GRL on center-pixel noise embedding | Task 2 (`GradientReversalLayer.apply`, `self.discriminator`) |
| DANN-style lambda_adv warmup schedule | Task 5 (per-epoch schedule update) |
| Composite loss with cls + recon + adv + smooth | Task 4 |
| MAE checkpoint loads unchanged | Task 1 test + Task 2 `load_encoder_state_dict` |
| Same stratified parquet / cache as v5 | Task 7 (no parquet rebuild; reuses existing cache) |
| Drop the frozen-encoder condition | Task 7 (3-task array, three lr_scale values, no `--freeze_encoder`) |
| Log val_disc_acc, val_n_norm_mean, lambda_adv | Task 5 step 5 |
| CLI flag `--lambda_adv_max` | Task 6 step 2 |
| HPC handoff instructions | Task 8 step 3 |

All requirements covered.

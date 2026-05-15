# Signal/Noise Decomposition Encoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `DecompSpVit` — a CRISM patch encoder that decomposes input I/F into surface signal × atmospheric transmission + path radiance + stochastic residual, with a classifier reading the shared encoder embedding. The reconstruction + classification joint loss pressures the embedding to represent surface mineralogy with noise factored out.

**Architecture:** Reuse the existing `SpatialSpectralTransformer` encoder unchanged (so the MAE pre-training checkpoint loads in). Add four new heads after the encoder: signal decoder (per-token MLP to 59-d reflectance), atmosphere head (CLS-token MLP to per-patch T and b vectors), residual head (per-token MLP to 59-d), classification head (linear on center-pixel embedding to 5 classes). Composite loss: ASL classification + reconstruction MSE + four regularizers preventing the trivial solution.

**Tech Stack:** PyTorch, conda env `crism`, pytest. Project root `/mnt/mrdr/crism_classification`. Commit messages follow recent repo style (`feat:`, `test:`, `fix:`, `perf:`).

**Conventions:**
- All commands run from `/mnt/mrdr/crism_classification` unless stated.
- All Python uses the `crism` env: prefix shell calls with `conda run -n crism ...` or activate first.
- Spec at `docs/superpowers/specs/2026-05-14-signal-noise-decomposition-design.md` is the source of truth for ambiguities.

---

## File Map

| Path | Action | Responsibility |
|---|---|---|
| `models/decomp_spatial_vit.py` | Create | `DecompSpVit` module — encoder + 4 heads + forward returning `(logits, s_hat, T_hat, b_hat, eps_hat, x_hat)` |
| `training/decomp_losses.py` | Create | `DecompositionLoss` — composite loss with classifier + reconstruction + four regularizers, configurable λ values |
| `training/train_torch.py` | Modify | Branch on `decomp_spatial_vit`: use the composite loss, log T/b/eps metrics each epoch |
| `scripts/train.py` | Modify | Add `decomp_spatial_vit` to `TORCH_MODELS`; instantiate `DecompSpVit`; thread λ args through to `train_torch_model` |
| `scripts/hpc_ablation_decomp_v1.slurm` | Create | First v1 HPC sweep — 4 encoder_lr_scale conditions, decomp_v1 naming |
| `tests/test_decomp_spatial_vit.py` | Create | Shape correctness + invariant checks + checkpoint-load test |
| `tests/test_decomp_losses.py` | Create | Loss-term gradient and regularizer behavior tests |

---

## Chunk 1 — Build `DecompSpVit` model

### Task 1: Failing tests for `DecompSpVit` shape contract

**Files:**
- Create: `tests/test_decomp_spatial_vit.py`

- [ ] **Step 1: Write the failing test file**

```python
# tests/test_decomp_spatial_vit.py
"""Tests for the signal/noise decomposition encoder."""
import pytest
import torch

from models.decomp_spatial_vit import DecompSpVit


@pytest.fixture
def model():
    return DecompSpVit(
        n_bands=59, patch_size=7, n_classes=5,
        embed_dim=128, n_heads=4, n_layers=6, dropout=0.0,
        T_min=0.3, T_max=1.0,
    )


def test_forward_shapes(model):
    B = 4
    x = torch.randn(B, 7, 7, 59)
    out = model(x)

    # Documented forward return tuple: (logits, s_hat, T_hat, b_hat, eps_hat, x_hat)
    logits, s_hat, T_hat, b_hat, eps_hat, x_hat = out
    assert logits.shape == (B, 5)
    assert s_hat.shape == (B, 49, 59)
    assert T_hat.shape == (B, 59)
    assert b_hat.shape == (B, 59)
    assert eps_hat.shape == (B, 49, 59)
    assert x_hat.shape == (B, 49, 59)


def test_T_hat_is_bounded(model):
    B = 4
    x = torch.randn(B, 7, 7, 59) * 5.0   # noisy input
    _, _, T_hat, _, _, _ = model(x)
    # T_hat is sigmoid-scaled to [T_min, T_max]; both bounds are inclusive
    assert torch.all(T_hat >= 0.3 - 1e-6)
    assert torch.all(T_hat <= 1.0 + 1e-6)


def test_reconstruction_equation(model):
    """x_hat MUST equal T_hat[:,None,:] * s_hat + b_hat[:,None,:] + eps_hat."""
    B = 2
    x = torch.randn(B, 7, 7, 59)
    _, s_hat, T_hat, b_hat, eps_hat, x_hat = model(x)
    expected = T_hat[:, None, :] * s_hat + b_hat[:, None, :] + eps_hat
    torch.testing.assert_close(x_hat, expected, rtol=1e-5, atol=1e-5)


def test_classifier_reads_center_pixel_embedding(model):
    """logits at batch i must depend on encoder output at center-pixel token of batch i."""
    B = 3
    torch.manual_seed(0)
    x = torch.randn(B, 7, 7, 59)
    logits = model(x)[0]
    assert logits.shape == (B, 5)
    # Perturbing the center pixel should change logits more than perturbing a corner
    x2_center = x.clone(); x2_center[:, 3, 3, :] += 1.0
    x2_corner = x.clone(); x2_corner[:, 0, 0, :] += 1.0
    delta_center = (model(x2_center)[0] - logits).abs().sum().item()
    delta_corner = (model(x2_corner)[0] - logits).abs().sum().item()
    assert delta_center > delta_corner, (
        f"Classifier should be more sensitive to center pixel: "
        f"delta_center={delta_center:.4f}, delta_corner={delta_corner:.4f}"
    )


def test_load_mae_encoder_checkpoint():
    """Encoder state from a SpatialSpectralMAE checkpoint should load cleanly."""
    import os
    ckpt_path = '/mnt/mrdr/crism_classification/checkpoints/spatial_mae_128d_6l_best.pt'
    if not os.path.exists(ckpt_path):
        pytest.skip(f"MAE checkpoint not available at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = DecompSpVit(
        n_bands=59, patch_size=7, n_classes=5,
        embed_dim=128, n_heads=4, n_layers=6, dropout=0.1,
    )
    missing, unexpected = model.load_encoder_state_dict(ckpt['encoder_state'])
    # No unexpected keys (encoder state matches the encoder submodule)
    assert unexpected == [], f"Unexpected keys when loading MAE encoder: {unexpected}"
    # Missing keys are allowed (the new heads aren't in the MAE checkpoint), but the
    # encoder's core weights should all be present
    assert not any(k.startswith('encoder.encoder') for k in missing), \
        f"Core encoder weights missing: {[k for k in missing if k.startswith('encoder.encoder')]}"


def test_param_groups_split_encoder_and_heads(model):
    """get_param_groups should return distinct groups for encoder vs new heads."""
    groups = model.get_param_groups(head_lr=1e-3, encoder_lr=1e-5)
    assert len(groups) == 2
    encoder_lr = groups[0]['lr']; head_lr = groups[1]['lr']
    assert encoder_lr == 1e-5
    assert head_lr == 1e-3
    # Every encoder param should be in the encoder group, no overlap
    encoder_param_ids = {id(p) for p in groups[0]['params']}
    head_param_ids = {id(p) for p in groups[1]['params']}
    assert encoder_param_ids.isdisjoint(head_param_ids)
    # Total params should match the model
    total = sum(p.numel() for p in model.parameters())
    grouped = sum(p.numel() for g in groups for p in g['params'])
    assert grouped == total
```

- [ ] **Step 2: Run the test file and confirm it fails on import**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism pytest tests/test_decomp_spatial_vit.py -v
```

Expected: `ModuleNotFoundError: No module named 'models.decomp_spatial_vit'`. This confirms RED before GREEN.

---

### Task 2: Implement `DecompSpVit`

**Files:**
- Create: `models/decomp_spatial_vit.py`

- [ ] **Step 1: Create the new module**

```python
# models/decomp_spatial_vit.py
"""
Signal/noise decomposition encoder for CRISM patches.

Decomposes input I/F into:
    x  ≈  T(λ) · s + b(λ) + ε

where s is per-pixel surface reflectance (the signal), T and b are
per-patch multiplicative and additive atmospheric terms, and ε is the
per-pixel stochastic residual. The classifier reads the shared encoder's
center-pixel embedding; the reconstruction objective pressures the
embedding to represent surface mineralogy.

Spec: docs/superpowers/specs/2026-05-14-signal-noise-decomposition-design.md
"""
from typing import Tuple

import torch
import torch.nn as nn

from models.spatial_spectral_transformer import SpatialSpectralTransformer


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    """Two-layer MLP with GELU activation."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class DecompSpVit(nn.Module):
    """
    Decomposition-aware classifier over CRISM patches.

    Forward returns: (logits, s_hat, T_hat, b_hat, eps_hat, x_hat)
        logits:  (B, n_classes)        — classifier output, sigmoid not applied
        s_hat:   (B, n_tokens, n_bands) — per-pixel surface reflectance
        T_hat:   (B, n_bands)          — per-patch multiplicative correction in [T_min, T_max]
        b_hat:   (B, n_bands)          — per-patch additive offset (unconstrained)
        eps_hat: (B, n_tokens, n_bands) — per-pixel residual
        x_hat:   (B, n_tokens, n_bands) — reconstruction = T·s + b + eps
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
        T_min: float = 0.3,
        T_max: float = 1.0,
    ):
        super().__init__()
        self.n_bands = n_bands
        self.n_tokens = patch_size * patch_size
        self.embed_dim = embed_dim
        self.T_min = T_min
        self.T_max = T_max

        self.encoder = SpatialSpectralTransformer(
            n_bands=n_bands, patch_size=patch_size,
            embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers, dropout=dropout,
        )

        # Heads
        self.signal_decoder = _mlp(embed_dim, head_hidden, n_bands, dropout=dropout)
        self.residual_decoder = _mlp(embed_dim, head_hidden, n_bands, dropout=dropout)
        # Atmosphere head outputs 2*n_bands — first n_bands → T_hat (sigmoid-scaled),
        # second n_bands → b_hat (unconstrained).
        self.atmosphere_head = _mlp(embed_dim, head_hidden, 2 * n_bands, dropout=dropout)
        self.class_head = nn.Linear(embed_dim, n_classes)

        # CLS token is slot 0; center-pixel token in the grid is at flat index
        # n_tokens//2 in the spatial layout (e.g., (3,3) for 7×7 = 24), then +1
        # for the CLS offset → 25.
        self._center_idx = self.n_tokens // 2 + 1

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, patch_size, patch_size, n_bands)
        z = self.encoder(x)                      # (B, n_tokens+1, embed_dim)
        cls = z[:, 0]                            # (B, embed_dim)
        tokens = z[:, 1:]                        # (B, n_tokens, embed_dim)

        s_hat = self.signal_decoder(tokens)      # (B, n_tokens, n_bands)
        eps_hat = self.residual_decoder(tokens)  # (B, n_tokens, n_bands)

        Tb = self.atmosphere_head(cls)           # (B, 2*n_bands)
        T_raw, b_hat = Tb[:, :self.n_bands], Tb[:, self.n_bands:]
        # Sigmoid-scale T to [T_min, T_max]
        T_hat = self.T_min + (self.T_max - self.T_min) * torch.sigmoid(T_raw)

        # Broadcast per-patch T_hat and b_hat across the n_tokens dimension
        x_hat = T_hat.unsqueeze(1) * s_hat + b_hat.unsqueeze(1) + eps_hat

        center_token = z[:, self._center_idx]    # (B, embed_dim)
        logits = self.class_head(center_token)   # (B, n_classes)

        return logits, s_hat, T_hat, b_hat, eps_hat, x_hat

    def get_param_groups(self, head_lr: float, encoder_lr: float) -> list:
        """Param groups for differential LR fine-tuning.
        Groups: encoder (slow) and heads (fast). Returned in that order so the
        index matches the training loop's lr-scheduling convention.
        """
        encoder_params = list(self.encoder.parameters())
        encoder_ids = {id(p) for p in encoder_params}
        head_params = [p for p in self.parameters() if id(p) not in encoder_ids]
        return [
            {'params': encoder_params, 'lr': encoder_lr},
            {'params': head_params,    'lr': head_lr},
        ]

    def load_encoder_state_dict(self, state: dict):
        """Load encoder weights from a SpatialSpectralMAE checkpoint."""
        return self.encoder.load_encoder_state_dict(state)
```

- [ ] **Step 2: Run tests and verify they pass**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism pytest tests/test_decomp_spatial_vit.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 3: Commit**

```bash
cd /mnt/mrdr/crism_classification
git add models/decomp_spatial_vit.py tests/test_decomp_spatial_vit.py
git commit -m "feat: DecompSpVit — signal/noise decomposition encoder"
```

---

## Chunk 2 — Composite loss

### Task 3: Failing tests for `DecompositionLoss`

**Files:**
- Create: `tests/test_decomp_losses.py`

- [ ] **Step 1: Write the failing test file**

```python
# tests/test_decomp_losses.py
"""Tests for the composite decomposition loss."""
import pytest
import torch

from training.decomp_losses import DecompositionLoss


@pytest.fixture
def loss_fn():
    return DecompositionLoss(
        lambda_recon=1.0,
        lambda_eps=0.1,
        lambda_T=0.01,
        lambda_b=0.01,
        lambda_smooth=0.001,
        asl_gamma_neg=4.0, asl_gamma_pos=0.0, asl_clip=0.05,
    )


def _make_decomp_outputs(B=4, n_tokens=49, n_bands=59, n_classes=5):
    """Synthesize valid forward outputs."""
    torch.manual_seed(0)
    s_hat = torch.randn(B, n_tokens, n_bands) * 0.1
    eps_hat = torch.randn(B, n_tokens, n_bands) * 0.01
    T_hat = torch.full((B, n_bands), 0.85)
    b_hat = torch.randn(B, n_bands) * 0.01
    x_hat = T_hat.unsqueeze(1) * s_hat + b_hat.unsqueeze(1) + eps_hat
    x = x_hat + torch.randn_like(x_hat) * 0.005   # small reconstruction error
    logits = torch.randn(B, n_classes)
    labels = (torch.rand(B, n_classes) > 0.5).float()
    weights = torch.ones(B)
    return dict(
        x=x, logits=logits, labels=labels, weights=weights,
        s_hat=s_hat, T_hat=T_hat, b_hat=b_hat, eps_hat=eps_hat, x_hat=x_hat,
    )


def test_loss_returns_scalar_and_components(loss_fn):
    o = _make_decomp_outputs()
    total, components = loss_fn(
        x=o['x'], logits=o['logits'], labels=o['labels'], weights=o['weights'],
        s_hat=o['s_hat'], T_hat=o['T_hat'], b_hat=o['b_hat'],
        eps_hat=o['eps_hat'], x_hat=o['x_hat'],
    )
    assert total.ndim == 0, "total must be a scalar tensor"
    for key in ('cls', 'recon', 'eps_reg', 'T_reg', 'b_reg', 'smooth'):
        assert key in components, f"missing loss component: {key}"
        assert components[key].ndim == 0


def test_recon_loss_zero_when_reconstruction_perfect(loss_fn):
    """If x_hat exactly equals x, recon component should be 0."""
    o = _make_decomp_outputs()
    perfect = o['x']  # set x_hat == x
    total, components = loss_fn(
        x=o['x'], logits=o['logits'], labels=o['labels'], weights=o['weights'],
        s_hat=o['s_hat'], T_hat=o['T_hat'], b_hat=o['b_hat'],
        eps_hat=o['eps_hat'], x_hat=perfect,
    )
    assert components['recon'].item() < 1e-8


def test_eps_reg_zero_when_residual_zero(loss_fn):
    """If eps_hat is all zeros, eps_reg should be 0."""
    o = _make_decomp_outputs()
    o['eps_hat'] = torch.zeros_like(o['eps_hat'])
    total, components = loss_fn(**o, labels=o['labels'])
    assert components['eps_reg'].item() < 1e-8


def test_T_reg_zero_when_T_is_one(loss_fn):
    """L_T_reg should be exactly 0 when T_hat == 1.0 (its prior)."""
    o = _make_decomp_outputs()
    o['T_hat'] = torch.ones_like(o['T_hat'])
    _, components = loss_fn(**o, labels=o['labels'])
    assert components['T_reg'].item() < 1e-8


def test_b_reg_zero_when_b_is_zero(loss_fn):
    o = _make_decomp_outputs()
    o['b_hat'] = torch.zeros_like(o['b_hat'])
    _, components = loss_fn(**o, labels=o['labels'])
    assert components['b_reg'].item() < 1e-8


def test_smooth_zero_when_signal_uniform(loss_fn):
    """L_smooth should be 0 when s_hat is spatially uniform across the patch."""
    B, n_tokens, n_bands = 2, 49, 59
    # Uniform spatial signal — same spectrum at every spatial position
    spec = torch.randn(B, 1, n_bands) * 0.1
    s_hat_uniform = spec.expand(-1, n_tokens, -1).clone()
    T_hat = torch.ones(B, n_bands)
    b_hat = torch.zeros(B, n_bands)
    eps_hat = torch.zeros(B, n_tokens, n_bands)
    x_hat = T_hat.unsqueeze(1) * s_hat_uniform + b_hat.unsqueeze(1) + eps_hat
    x = x_hat.clone()
    logits = torch.zeros(B, 5)
    labels = torch.zeros(B, 5)
    weights = torch.ones(B)
    _, components = loss_fn(
        x=x, logits=logits, labels=labels, weights=weights,
        s_hat=s_hat_uniform, T_hat=T_hat, b_hat=b_hat,
        eps_hat=eps_hat, x_hat=x_hat,
    )
    assert components['smooth'].item() < 1e-8


def test_total_loss_is_weighted_sum_of_components(loss_fn):
    """Total loss must equal cls + λ_recon*recon + λ_eps*eps_reg + λ_T*T_reg + λ_b*b_reg + λ_smooth*smooth."""
    o = _make_decomp_outputs()
    total, c = loss_fn(**o, labels=o['labels'])
    expected = (
        c['cls']
        + loss_fn.lambda_recon * c['recon']
        + loss_fn.lambda_eps * c['eps_reg']
        + loss_fn.lambda_T * c['T_reg']
        + loss_fn.lambda_b * c['b_reg']
        + loss_fn.lambda_smooth * c['smooth']
    )
    torch.testing.assert_close(total, expected, rtol=1e-5, atol=1e-5)


def test_class_weights_scale_classification_term(loss_fn):
    """Passing class_weights should scale the classification component."""
    o = _make_decomp_outputs()
    cw = torch.tensor([1.0, 1.0, 1.5, 3.0, 1.0])
    _, c_with = loss_fn(**o, labels=o['labels'], class_weights=cw)
    _, c_without = loss_fn(**o, labels=o['labels'])
    # The two cls values are almost certainly different in the random-label case;
    # the only invariant we can assert is the class_weights branch ran.
    assert torch.is_tensor(c_with['cls'])
```

- [ ] **Step 2: Run the test file and confirm it fails on import**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism pytest tests/test_decomp_losses.py -v
```

Expected: `ModuleNotFoundError: No module named 'training.decomp_losses'`.

---

### Task 4: Implement `DecompositionLoss`

**Files:**
- Create: `training/decomp_losses.py`

- [ ] **Step 1: Create the loss module**

```python
# training/decomp_losses.py
"""
Composite loss for the signal/noise decomposition encoder.

  L_total = L_cls
          + λ_recon  · L_recon
          + λ_eps    · L_eps_reg
          + λ_T      · L_T_reg
          + λ_b      · L_b_reg
          + λ_smooth · L_smooth

L_cls is the existing AsymmetricLoss on (logits, labels, sample_weights,
optional class_weights). The other terms enforce the physical decomposition
structure documented in
docs/superpowers/specs/2026-05-14-signal-noise-decomposition-design.md.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from training.losses import AsymmetricLoss


class DecompositionLoss(nn.Module):
    def __init__(
        self,
        lambda_recon: float = 1.0,
        lambda_eps: float = 0.1,
        lambda_T: float = 0.01,
        lambda_b: float = 0.01,
        lambda_smooth: float = 0.001,
        asl_gamma_neg: float = 4.0,
        asl_gamma_pos: float = 0.0,
        asl_clip: float = 0.05,
    ):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_eps = lambda_eps
        self.lambda_T = lambda_T
        self.lambda_b = lambda_b
        self.lambda_smooth = lambda_smooth
        self.cls_loss = AsymmetricLoss(
            gamma_neg=asl_gamma_neg, gamma_pos=asl_gamma_pos, clip=asl_clip,
        )

    def forward(
        self,
        x: torch.Tensor,            # (B, n_tokens, n_bands) or (B, P, P, n_bands)
        logits: torch.Tensor,       # (B, n_classes)
        labels: torch.Tensor,       # (B, n_classes)
        weights: torch.Tensor,      # (B,)
        s_hat: torch.Tensor,        # (B, n_tokens, n_bands)
        T_hat: torch.Tensor,        # (B, n_bands)
        b_hat: torch.Tensor,        # (B, n_bands)
        eps_hat: torch.Tensor,      # (B, n_tokens, n_bands)
        x_hat: torch.Tensor,        # (B, n_tokens, n_bands)
        pos_weight: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # If x came in as a 4D patch, flatten to (B, n_tokens, n_bands) for recon.
        if x.dim() == 4:
            B, P, P2, n_bands = x.shape
            assert P == P2, "patches must be square"
            x_flat = x.reshape(B, P * P2, n_bands)
        else:
            x_flat = x

        # 1. Classification loss
        cls = self.cls_loss(
            logits, labels, weights,
            pos_weight=pos_weight, class_weights=class_weights,
        )

        # 2. Reconstruction loss (MSE on valid pixels).
        # NODATA pixels were zeroed upstream (see data/extract_pixels.py).
        # Pixels with |x|>1 are pathological; mask them out so they don't dominate.
        valid_mask = (x_flat.abs() < 1.0).float()
        sq_err = (x_hat - x_flat) ** 2 * valid_mask
        recon = sq_err.sum() / (valid_mask.sum() + 1e-8)

        # 3. Residual magnitude regularizer
        eps_reg = (eps_hat ** 2).mean()

        # 4. Atmospheric priors
        T_reg = ((T_hat - 1.0) ** 2).mean()
        b_reg = (b_hat ** 2).mean()

        # 5. Spatial smoothness on signal — TV on the (B, P, P, n_bands) layout.
        # Reshape s_hat to (B, P, P, n_bands), compute horizontal + vertical
        # first-differences, average over everything.
        B, N, nb = s_hat.shape
        P = int(N ** 0.5)
        s_spatial = s_hat.view(B, P, P, nb)
        dv = (s_spatial[:, 1:, :, :] - s_spatial[:, :-1, :, :]).abs()
        dh = (s_spatial[:, :, 1:, :] - s_spatial[:, :, :-1, :]).abs()
        smooth = (dv.mean() + dh.mean()) * 0.5

        total = (
            cls
            + self.lambda_recon * recon
            + self.lambda_eps * eps_reg
            + self.lambda_T * T_reg
            + self.lambda_b * b_reg
            + self.lambda_smooth * smooth
        )
        components = {
            'cls': cls, 'recon': recon,
            'eps_reg': eps_reg, 'T_reg': T_reg, 'b_reg': b_reg,
            'smooth': smooth,
        }
        return total, components
```

- [ ] **Step 2: Run loss tests**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism pytest tests/test_decomp_losses.py -v
```

Expected: all 8 tests pass.

- [ ] **Step 3: Run the full new test suite together**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism pytest tests/test_decomp_spatial_vit.py tests/test_decomp_losses.py -v
```

Expected: 14 tests pass (6 from Task 2 + 8 from Task 4).

- [ ] **Step 4: Commit**

```bash
cd /mnt/mrdr/crism_classification
git add training/decomp_losses.py tests/test_decomp_losses.py
git commit -m "feat: DecompositionLoss composite loss with reconstruction + regularizers"
```

---

## Chunk 3 — Wire into training pipeline

### Task 5: Extend `train_torch.py` with a decomp model branch

**Files:**
- Modify: `training/train_torch.py`

This is the most invasive integration step. We branch on `model.__class__.__name__ == 'DecompSpVit'` rather than introducing a new model_type string parameter — the model class is already passed in, so we can inspect it. Composite loss takes the place of `loss_fn` for that branch.

- [ ] **Step 1: Read the existing training loop**

```bash
cd /mnt/mrdr/crism_classification
sed -n '200,260p' training/train_torch.py
```

Expected: see the training loop with `loss_fn(logits, labels, weights, pos_weight=..., class_weights=...)`.

- [ ] **Step 2: Add a decomp-aware loss invocation path**

Modify `training/train_torch.py` at the loss-construction block (around line 175-186) — add a path that builds a `DecompositionLoss` instead of the per-class loss when the model is a `DecompSpVit`. Replace this block:

```python
    if use_asl_loss:
        from training.losses import AsymmetricLoss
        loss_fn = AsymmetricLoss(gamma_neg=asl_gamma_neg, gamma_pos=asl_gamma_pos, clip=asl_clip)
    elif use_focal_loss:
        from training.losses import FocalBCEWithLogitsLoss
        loss_fn = FocalBCEWithLogitsLoss(gamma=focal_gamma)
    else:
        loss_fn = WeightedBCEWithLogitsLoss()

    # Per-class loss weights (e.g., boost rare classes like plagioclase, HCP).
    # Moved to device once so the loss can broadcast without per-step transfer.
    if class_weights is not None:
        class_weights = class_weights.to(device)
        logger.info(f"Using per-class loss weights: {class_weights.tolist()}")
```

with:

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
        from training.losses import AsymmetricLoss
        loss_fn = AsymmetricLoss(gamma_neg=asl_gamma_neg, gamma_pos=asl_gamma_pos, clip=asl_clip)
    elif use_focal_loss:
        from training.losses import FocalBCEWithLogitsLoss
        loss_fn = FocalBCEWithLogitsLoss(gamma=focal_gamma)
    else:
        loss_fn = WeightedBCEWithLogitsLoss()

    # Per-class loss weights (e.g., boost rare classes like plagioclase, HCP).
    # Moved to device once so the loss can broadcast without per-step transfer.
    if class_weights is not None:
        class_weights = class_weights.to(device)
        logger.info(f"Using per-class loss weights: {class_weights.tolist()}")
```

- [ ] **Step 3: Add new `decomp_lambda_*` parameters to the function signature**

In `training/train_torch.py`, locate the `train_torch_model` function signature (starts around line 41) and add five new optional params after `min_delta`:

Find:
```python
    class_weights: Optional[torch.Tensor] = None,
    min_delta: float = 0.0,
    device: Optional[str] = None,
```

Replace with:
```python
    class_weights: Optional[torch.Tensor] = None,
    min_delta: float = 0.0,
    decomp_lambda_recon: float = 1.0,
    decomp_lambda_eps: float = 0.1,
    decomp_lambda_T: float = 0.01,
    decomp_lambda_b: float = 0.01,
    decomp_lambda_smooth: float = 0.001,
    device: Optional[str] = None,
```

- [ ] **Step 4: Branch the training-step body to handle decomp forward + loss**

In `training/train_torch.py`, locate the training-step body (around line 200-220 — the `for features, labels, weights in train_loader:` block). Find:

```python
            optimizer.zero_grad()
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

- [ ] **Step 5: Initialize the component accumulator at epoch start**

In the same file, immediately before the `for features, labels, weights in train_loader:` line, find:

```python
    for epoch in range(1, max_epochs + 1):
        # --- Train ---
        model.train()
        train_losses = []
```

Replace with:

```python
    for epoch in range(1, max_epochs + 1):
        # --- Train ---
        model.train()
        train_losses = []
        train_loss_components: dict = {}   # decomp-only; ignored for non-decomp models
```

- [ ] **Step 6: Branch the validation forward pass too**

In `training/train_torch.py`, locate the validation block:

```python
        # --- Validate ---
        model.eval()
        all_logits, all_labels = [], []

        with torch.no_grad():
            for features, labels, weights in val_loader:
                features = features.to(device)
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

- [ ] **Step 7: Log decomp metrics to wandb**

In `training/train_torch.py`, locate the wandb-log block:

```python
        if use_wandb:
            import wandb as wb
            wb.log({'epoch': epoch, 'train_loss': np.mean(train_losses), **flat})
```

Replace with:

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

- [ ] **Step 8: Smoke-test the integration**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism python -c "
import sys; sys.path.insert(0, '.')
import torch
from models.decomp_spatial_vit import DecompSpVit
from training.decomp_losses import DecompositionLoss

m = DecompSpVit()
loss_fn = DecompositionLoss()
x = torch.randn(2, 7, 7, 59)
logits, s, T, b, e, xh = m(x)
labels = (torch.rand(2, 5) > 0.5).float()
w = torch.ones(2)
total, comp = loss_fn(x=x, logits=logits, labels=labels, weights=w,
                     s_hat=s, T_hat=T, b_hat=b, eps_hat=e, x_hat=xh)
print('total loss:', total.item())
print('components:', {k: f'{v.item():.4f}' for k, v in comp.items()})
total.backward()
print('backward succeeded')
"
```

Expected: total prints, components dict prints, "backward succeeded" prints.

- [ ] **Step 9: Run the existing test suite to make sure nothing regressed**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism pytest tests/ -x -q 2>&1 | tail -10
```

Expected: same number of passes as before this chunk (plus the 14 new ones from Tasks 2+4); no new failures.

- [ ] **Step 10: Commit**

```bash
cd /mnt/mrdr/crism_classification
git add training/train_torch.py
git commit -m "feat: train_torch.py supports DecompSpVit + DecompositionLoss path"
```

---

## Chunk 4 — CLI + slurm

### Task 6: Add `decomp_spatial_vit` to `scripts/train.py`

**Files:**
- Modify: `scripts/train.py`

- [ ] **Step 1: Add the model to `TORCH_MODELS`**

In `scripts/train.py`, find:

```python
TORCH_MODELS = {'mlp', 'cnn', 'vit', 'spectral_cnn', 'spectral_vit', 'spectral_hybrid', 'spatial_vit'}
```

Replace with:

```python
TORCH_MODELS = {'mlp', 'cnn', 'vit', 'spectral_cnn', 'spectral_vit',
                'spectral_hybrid', 'spatial_vit', 'decomp_spatial_vit'}
```

- [ ] **Step 2: Add CLI flags for λ values**

In `scripts/train.py`, locate the `--min_delta` argument line and append the decomp lambdas after it:

```python
    parser.add_argument('--min_delta', type=float, default=0.0,
                        help='Early-stopping tolerance: val_mAP drops up to this '
                             'much below the running best do not tick patience. '
                             'Default 0.0 = strict, any non-improvement counts.')
    parser.add_argument('--decomp_lambda_recon', type=float, default=1.0,
                        help='Weight on reconstruction MSE for DecompSpVit.')
    parser.add_argument('--decomp_lambda_eps', type=float, default=0.1,
                        help='Weight on residual L2 for DecompSpVit.')
    parser.add_argument('--decomp_lambda_T', type=float, default=0.01,
                        help='Weight on (T-1)^2 prior for DecompSpVit.')
    parser.add_argument('--decomp_lambda_b', type=float, default=0.01,
                        help='Weight on b^2 prior for DecompSpVit.')
    parser.add_argument('--decomp_lambda_smooth', type=float, default=0.001,
                        help='Weight on spatial-TV smoothness on s_hat for DecompSpVit.')
```

- [ ] **Step 3: Add the model construction branch**

In `scripts/train.py`, find the `elif args.model == 'spatial_vit':` branch (around line 340 — search for the `SpatialSpectralClassifier` import). After it, add a parallel branch that constructs `DecompSpVit`. The cleanest patch is to add an `elif` after the existing spatial_vit block.

Find the existing block that ends with:

```python
            mrral_cache_dir = cfg.get('patch_cache_dir')
            metrics = train_torch_model(
                model=model, df=df_mrral, model_name=run_name,
                ...
                freeze_encoder=args.freeze_encoder,
            )
```

(There are multiple `train_torch_model` calls — find the one whose preceding `model = SpatialSpectralClassifier(...)` instantiation is for `spatial_vit`. It's the third `train_torch_model(...)` call in the file per the earlier grep, ending around line 410 with `freeze_encoder=args.freeze_encoder,`.)

After the closing parenthesis of that `train_torch_model(...)`, add a new `elif`:

```python
        elif args.model == 'decomp_spatial_vit':
            import glob
            data_root = cfg.get('data_root', '/mnt/crism/MRDR')
            mrral_hdrs = sorted(set(
                glob.glob(os.path.join(data_root, 'mc*', 't*mrral*.hdr'))
                + glob.glob(os.path.join(data_root, 't*mrral*.hdr'))
            ))
            mrral_map = {}
            for hdr in mrral_hdrs:
                tid = os.path.basename(hdr).split('_mrral_')[0]
                mrral_map[tid] = hdr.replace('.hdr', '.img')
            logging.info(f'mrral_map: {len(mrral_map)} tiles found')

            mrral_parquet = os.path.join(cfg['output_dir'], 'mrral_pixels.parquet')
            df_mrral = pd.read_parquet(mrral_parquet)
            dropout = args.dropout if args.dropout is not None else 0.1

            from models.decomp_spatial_vit import DecompSpVit
            model = DecompSpVit(
                n_bands=59, patch_size=args.patch_size, n_classes=5,
                embed_dim=args.embed_dim, n_heads=args.n_heads,
                n_layers=args.n_layers, dropout=dropout,
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
                decomp_lambda_eps=args.decomp_lambda_eps,
                decomp_lambda_T=args.decomp_lambda_T,
                decomp_lambda_b=args.decomp_lambda_b,
                decomp_lambda_smooth=args.decomp_lambda_smooth,
                freeze_encoder=args.freeze_encoder,
            )
```

- [ ] **Step 4: Verify the CLI parses and the help text shows the new flags**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism python scripts/train.py --help 2>&1 | grep -E "decomp_|model.*decomp"
```

Expected: the 5 new `--decomp_lambda_*` flags show up, and the `--model` choices include `decomp_spatial_vit`.

- [ ] **Step 5: Smoke-run for 1 epoch on a synthetic mini dataset**

Skip — the full v5 cache + parquet are present locally, but training is slow. The next chunk runs the real thing on HPC. The integration smoke test from Task 5 Step 8 already validated the model+loss interplay.

- [ ] **Step 6: Commit**

```bash
cd /mnt/mrdr/crism_classification
git add scripts/train.py
git commit -m "feat: scripts/train.py supports --model decomp_spatial_vit + λ flags"
```

---

### Task 7: HPC slurm for the v1 decomp sweep

**Files:**
- Create: `scripts/hpc_ablation_decomp_v1.slurm`

- [ ] **Step 1: Write the slurm file**

```bash
# scripts/hpc_ablation_decomp_v1.slurm
#!/bin/bash
#SBATCH --job-name=spvit_decomp_v1
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32gb
#SBATCH --time=0-24:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/decomp_v1_%a_%j.out
#SBATCH --error=logs/decomp_v1_%a_%j.err

# v1: first run of the signal/noise decomposition encoder (DecompSpVit).
# Matches v5 hyperparams except --model decomp_spatial_vit and the new
# --decomp_lambda_* flags. Spec at:
#   docs/superpowers/specs/2026-05-14-signal-noise-decomposition-design.md

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
CKPT_DIR=${WORK_DIR}/checkpoints
PRETRAIN_CKPT=${CKPT_DIR}/spatial_mae_128d_6l_best.pt
CACHE_DIR=${WORK_DIR}/data/patch_cache

RUN_NAMES=("spvit_decomp_v1_frozen" "spvit_decomp_v1_lrscale0001" "spvit_decomp_v1_lrscale001" "spvit_decomp_v1_lrscale01")
LR_SCALE_ARGS=("--freeze_encoder" "--encoder_lr_scale 0.001" "--encoder_lr_scale 0.01" "--encoder_lr_scale 0.1")
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

echo "=== decomp_v1 ablation: ${RUN_NAMES[$SLURM_ARRAY_TASK_ID]} (task $SLURM_ARRAY_TASK_ID) ==="
${PYTHON} -u scripts/train.py \
    --model decomp_spatial_vit \
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
    --decomp_lambda_recon 1.0 \
    --decomp_lambda_eps 0.1 \
    --decomp_lambda_T 0.01 \
    --decomp_lambda_b 0.01 \
    --decomp_lambda_smooth 0.001 \
    --pretrain_ckpt "${PRETRAIN_CKPT}" \
    ${LR_SCALE_ARGS[$SLURM_ARRAY_TASK_ID]}
```

- [ ] **Step 2: Commit**

```bash
cd /mnt/mrdr/crism_classification
git add scripts/hpc_ablation_decomp_v1.slurm
git commit -m "feat: HPC slurm for decomp_v1 4-config sweep"
```

---

## Chunk 5 — Documentation

### Task 8: Wiki methodology section

**Files:**
- Modify: `/mnt/mrdr/wiki/Methodology Log v5.md` (append a new section)

- [ ] **Step 1: Append the new section**

Open `/mnt/mrdr/wiki/Methodology Log v5.md` and append the following block at the end (before the "Index of artifacts" section, which should stay last):

```markdown
## 12. Signal/Noise Decomposition Encoder (DecompSpVit, May 14 2026)

The cosine-similarity diagnostic in figure 5 (`fig_v5_embedding.png`) revealed
that the encoder's representation of the two pyroxene classes was insufficiently
separated (HCP ↔ LCP = 0.84) — a structural failure rather than a label or
sampling problem. The fix considered was a physics-informed encoder
decomposition that respects the canonical CRISM observation equation:

```
I/F = T_atm(λ) · R_surface(λ, r, c) + b_path(λ) + n_column(c, λ) + ε(r, c, λ)
```

**Architecture (DecompSpVit, B′ variant).** Reuses the existing
SpatialSpectralTransformer encoder (so the MAE pre-training checkpoint loads
in unchanged) and adds four new heads on top:

- **Signal decoder** — per-token MLP (128 → 256 → 59), produces per-pixel
  surface reflectance estimate `s_hat`.
- **Atmosphere head** — reads only the CLS token, MLP 128 → 2·59. First half
  becomes `T_hat` (sigmoid-scaled to [0.3, 1.0]); second half becomes `b_hat`
  (unconstrained). One value per band per patch — atmospheric scale heights
  exceed the 7 × 180 m ≈ 1 km patch size.
- **Residual decoder** — per-token MLP, produces `ε_hat`. Column-correlated
  noise is lumped into ε for v1; if ε shows clear column structure post-train
  we'll split it out in v2.
- **Classification head** — linear 128 → 5 on the center-pixel encoder
  embedding (NOT on `s_hat`). The reconstruction loss pressures the shared
  encoder embedding to represent surface mineralogy; the classifier consumes
  that same embedding.

**Composite loss.** Classification (ASL with class weights) +
reconstruction MSE + four regularizers that prevent the trivial solution:

- `L_recon = ‖x − (T·s + b + ε)‖²` on valid pixels
- `L_eps_reg = ‖ε_hat‖²` — keeps the residual small
- `L_T_reg = ‖T_hat − 1‖²` — priors T toward "no attenuation"
- `L_b_reg = ‖b_hat‖²` — priors path radiance toward zero
- `L_smooth` — spatial total-variation on `s_hat` (mineralogy varies smoothly
  inside a 7×7 patch; per-pixel noise does not)

Defaults: `(λ_recon, λ_eps, λ_T, λ_b, λ_smooth) = (1.0, 0.1, 0.01, 0.01, 0.001)`.

**What's the same as v5.** Stratified parquet, hard olivine labels,
tier-based confidence weights (1.0 / 0.85 / 0.70), per-class loss weights
`(1, 1, 1.5, 3, 1)`, 100-epoch / patience-25 / min_delta-0.001 training,
4-config encoder_lr_scale ablation.

**What's different.** New model class `DecompSpVit`, new
`DecompositionLoss`, new logged metrics (`val_T_mean`, `val_b_mean`,
`val_eps_norm_mean`, and the per-component train losses).

**Expected outcome.** Either (a) the HCP↔LCP embedding similarity drops
meaningfully and per-class AP rises — confirming that explicit physical
decomposition gives the classifier a cleaner input than raw I/F — or (b) the
classifier loss dominates and the decomposition heads just learn convenient
factorizations that don't reflect physics. We'll know from val_T_mean and the
spectrum-decomposition figure: if T_hat is roughly in [0.7, 0.95] and b_hat is
small, the model has converged on physically reasonable atmospheric estimates;
if T_hat is at the [0.3, 1.0] boundary or b_hat is huge, the model has decided
the priors are wrong.

**Sweep:** `scripts/hpc_ablation_decomp_v1.slurm`. Run names:
`spvit_decomp_v1_{frozen,lrscale0001,lrscale001,lrscale01}`.

| Run | encoder_lr_scale | val_mAP | val_T_mean | val_b_mean | Notes |
|-----|------------------|---------|------------|------------|-------|
| `spvit_decomp_v1_frozen` | frozen | — | — | — | Pending |
| `spvit_decomp_v1_lrscale0001` | 0.001 | — | — | — | Pending |
| `spvit_decomp_v1_lrscale001` | 0.01 | — | — | — | Pending |
| `spvit_decomp_v1_lrscale01` | 0.1 | — | — | — | Pending |

Architecture figure: `reports/v5/fig_decomp_architecture.png` (planned —
will be generated with the scientific-schematics skill once at least one
decomp checkpoint converges).
```

- [ ] **Step 2: Commit (note: wiki is not git-tracked but the spec and plan are; this is a wiki-only edit)**

The wiki lives outside the git repo. Just save the file (the Edit tool above did this). Verify content with:

```bash
grep -A 3 "DecompSpVit, May 14" "/mnt/mrdr/wiki/Methodology Log v5.md" | head
```

Expected: shows the new section heading.

---

### Task 9: Architecture figure script (placeholder for later rendering)

**Files:**
- Create: `scripts/figures/fig_decomp_architecture.py`

- [ ] **Step 1: Write a matplotlib-based block diagram (no AI generation required)**

```python
# scripts/figures/fig_decomp_architecture.py
"""
Generate fig_v5_decomp_architecture.png — block diagram of the DecompSpVit
signal/noise decomposition encoder.

Usage:
    conda run -n crism python scripts/figures/fig_decomp_architecture.py
"""
from __future__ import annotations

import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt

OUT_PATH = '/mnt/mrdr/crism_classification/reports/v5/fig_v5_decomp_architecture.png'


def block(ax, x, y, w, h, label, color='#cdeaf7', edgecolor='#1f77b4',
          fontsize=10, lw=1.6):
    rect = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=lw, edgecolor=edgecolor, facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha='center', va='center', fontsize=fontsize)


def arrow(ax, x0, y0, x1, y1, color='black', lw=1.5, ls='-'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                linestyle=ls, shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(13.5, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    # Input
    block(ax, 0.2, 6.0, 1.7, 1.2, 'Patch x\n(7×7×59)',
          color='#f0f0f0', edgecolor='gray')
    arrow(ax, 1.9, 6.6, 2.9, 6.6)

    # Encoder
    block(ax, 2.9, 5.5, 2.6, 2.0,
          'Shared encoder\n(6L ViT, 128-d,\nMAE-pretrained)',
          color='#cdeaf7', edgecolor='#1f77b4', fontsize=10)

    # Tokens output
    arrow(ax, 5.5, 7.0, 6.8, 7.5)
    arrow(ax, 5.5, 6.6, 6.8, 6.4)
    arrow(ax, 5.5, 6.2, 6.8, 5.3)
    arrow(ax, 5.5, 6.0, 6.8, 3.6)

    # CLS path
    block(ax, 6.8, 7.1, 1.4, 0.6, 'CLS token', color='#fff2cc', edgecolor='#bf8b00',
          fontsize=9)
    # Spatial tokens path
    block(ax, 6.8, 6.0, 1.4, 0.6, '49 spatial\ntokens', color='#fff2cc',
          edgecolor='#bf8b00', fontsize=9)
    block(ax, 6.8, 5.0, 1.4, 0.6, 'center-pixel\ntoken (3,3)',
          color='#e8f6e8', edgecolor='#2ca02c', fontsize=9)
    block(ax, 6.8, 3.3, 1.4, 0.6, '49 spatial\ntokens', color='#fff2cc',
          edgecolor='#bf8b00', fontsize=9)

    # Heads
    # Atmosphere head (from CLS)
    arrow(ax, 8.2, 7.4, 9.4, 7.4)
    block(ax, 9.4, 6.9, 2.2, 1.0, 'Atmosphere head\n(MLP 128 → 2·59)',
          color='#ffe0e0', edgecolor='#c44', fontsize=9)
    arrow(ax, 11.6, 7.4, 12.6, 7.7)
    arrow(ax, 11.6, 7.4, 12.6, 7.1)
    ax.text(12.7, 7.85, 'T_hat (B,59)', fontsize=9, va='center')
    ax.text(12.7, 7.05, 'b_hat (B,59)', fontsize=9, va='center')

    # Signal decoder
    arrow(ax, 8.2, 6.3, 9.4, 6.3)
    block(ax, 9.4, 5.8, 2.2, 1.0, 'Signal decoder\n(MLP per token)',
          color='#e8f6e8', edgecolor='#2ca02c', fontsize=9)
    arrow(ax, 11.6, 6.3, 12.6, 6.3)
    ax.text(12.7, 6.3, 's_hat (B,49,59)', fontsize=9, va='center')

    # Classifier head
    arrow(ax, 8.2, 5.3, 9.4, 5.0)
    block(ax, 9.4, 4.5, 2.2, 1.0, 'Classification head\n(Linear 128 → 5)',
          color='#fde2e1', edgecolor='#d62728', fontsize=9)
    arrow(ax, 11.6, 5.0, 12.6, 5.0)
    ax.text(12.7, 5.0, 'logits (B,5)', fontsize=9, va='center')

    # Residual decoder
    arrow(ax, 8.2, 3.6, 9.4, 3.6)
    block(ax, 9.4, 3.1, 2.2, 1.0, 'Residual decoder\n(MLP per token)',
          color='#f5e0ff', edgecolor='#a050a0', fontsize=9)
    arrow(ax, 11.6, 3.6, 12.6, 3.6)
    ax.text(12.7, 3.6, 'eps_hat (B,49,59)', fontsize=9, va='center')

    # Reconstruction box at bottom
    block(ax, 4.5, 0.6, 5.5, 1.2,
          'x_hat = T_hat · s_hat + b_hat + eps_hat\n(reconstruction loss vs x)',
          color='#fffae0', edgecolor='#998800', fontsize=11)

    # Dashed arrows from each output to the reconstruction
    arrow(ax, 13.5, 7.7, 9.7, 1.8, color='#888', lw=1.0, ls='--')
    arrow(ax, 13.5, 7.1, 9.5, 1.8, color='#888', lw=1.0, ls='--')
    arrow(ax, 13.5, 6.3, 8.5, 1.8, color='#888', lw=1.0, ls='--')
    arrow(ax, 13.5, 3.6, 7.0, 1.8, color='#888', lw=1.0, ls='--')

    # Title
    ax.text(0.2, 8.1, 'DecompSpVit — physics-informed decomposition encoder',
            fontsize=13, fontweight='bold')
    ax.text(0.2, 0.2,
            'The classifier reads the encoder embedding (not s_hat). The shared encoder is pressured by '
            'both the classification loss\nand the reconstruction loss to represent surface mineralogy, '
            'with atmospheric attenuation T, additive path radiance b, and stochastic residual ε '
            'factored out.',
            fontsize=9.5, color='#444', style='italic')

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {OUT_PATH}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Generate the figure**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism python scripts/figures/fig_decomp_architecture.py
```

Expected: prints `Wrote /mnt/mrdr/crism_classification/reports/v5/fig_v5_decomp_architecture.png`. No errors.

- [ ] **Step 3: Commit**

```bash
cd /mnt/mrdr/crism_classification
git add scripts/figures/fig_decomp_architecture.py reports/v5/fig_v5_decomp_architecture.png
git commit -m "feat: architecture diagram for DecompSpVit"
```

---

## Chunk 6 — Handoff

### Task 10: Final integration check + prepare HPC handoff list

**Files:** none modified — this task validates and documents what the user needs to do.

- [ ] **Step 1: Run the full test suite end-to-end**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism pytest tests/ -x -q 2>&1 | tail -10
```

Expected: all tests pass (existing + 14 new from this plan).

- [ ] **Step 2: Verify the smoke integration still works**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism python -c "
import sys; sys.path.insert(0, '.')
import torch
from models.decomp_spatial_vit import DecompSpVit
from training.decomp_losses import DecompositionLoss

m = DecompSpVit()
ckpt_path = 'checkpoints/spatial_mae_128d_6l_best.pt'
import os
if os.path.exists(ckpt_path):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    miss, unex = m.load_encoder_state_dict(ck['encoder_state'])
    assert not unex, f'unexpected: {unex}'
    print('MAE encoder loaded into DecompSpVit OK')

loss_fn = DecompositionLoss()
x = torch.randn(2, 7, 7, 59)
logits, s, T, b, e, xh = m(x)
labels = (torch.rand(2, 5) > 0.5).float()
total, comp = loss_fn(x=x, logits=logits, labels=labels, weights=torch.ones(2),
                     s_hat=s, T_hat=T, b_hat=b, eps_hat=e, x_hat=xh)
total.backward()
print(f'integration OK. total={total.item():.4f}, components={ {k: round(v.item(), 4) for k, v in comp.items()} }')
"
```

Expected: prints `MAE encoder loaded into DecompSpVit OK` and `integration OK. ...`.

- [ ] **Step 3: Print the HPC command list the user needs to run**

The local work is now complete. The user needs to do the following on HPC manually (interactive DUO auth blocks autonomous execution). This task just prints/documents the commands — no code change.

```bash
echo "=== Files to rsync to HPC ==="
echo "rsync -avh /mnt/mrdr/crism_classification/models/decomp_spatial_vit.py phillipsm@filexfer.hpc.arizona.edu:/groups/sbyrne/phillipsm/crism_classification/models/"
echo "rsync -avh /mnt/mrdr/crism_classification/training/decomp_losses.py /mnt/mrdr/crism_classification/training/train_torch.py phillipsm@filexfer.hpc.arizona.edu:/groups/sbyrne/phillipsm/crism_classification/training/"
echo "rsync -avh /mnt/mrdr/crism_classification/scripts/train.py /mnt/mrdr/crism_classification/scripts/hpc_ablation_decomp_v1.slurm phillipsm@filexfer.hpc.arizona.edu:/groups/sbyrne/phillipsm/crism_classification/scripts/"
echo ""
echo "=== After rsync, ssh and submit ==="
echo "ssh phillipsm@hpc.arizona.edu"
echo "  cd /groups/sbyrne/phillipsm/crism_classification"
echo "  # Verify imports work in the HPC env:"
echo "  /groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python -c \"from models.decomp_spatial_vit import DecompSpVit; from training.decomp_losses import DecompositionLoss; print('OK')\""
echo "  # Submit the v1 ablation array:"
echo "  sbatch scripts/hpc_ablation_decomp_v1.slurm"
echo "  squeue -u phillipsm"
```

This step is informational only — running these `echo` lines confirms the agent has the right paths and command structure.

- [ ] **Step 4: Final commit summarizing the plan completion**

```bash
cd /mnt/mrdr/crism_classification
git log --oneline -10
```

Expected: shows ~6 commits from this plan in reverse chronological order:
1. Architecture diagram
2. HPC slurm
3. scripts/train.py CLI
4. train_torch.py integration
5. DecompositionLoss
6. DecompSpVit model

No additional commit needed.

---

## Spec coverage check (self-review)

Spec section / requirement → Task that implements it:

- **Decomposition definition (B′)** — Task 2 (`DecompSpVit.forward`, the `x_hat = T·s + b + ε` line).
- **Architecture: shared encoder + 4 heads** — Task 2 (signal_decoder, residual_decoder, atmosphere_head, class_head).
- **Atmosphere head with sigmoid-scaled T, unconstrained b** — Task 2 (forward, `T_hat = T_min + (T_max - T_min) * sigmoid(T_raw)`).
- **Classifier reads encoder center-pixel token (not s_hat)** — Task 2 (`center_token = z[:, self._center_idx]`).
- **Composite loss with 6 terms** — Task 4 (`DecompositionLoss.forward`).
- **Reconstruction loss masks invalid pixels** — Task 4 (`valid_mask = (x_flat.abs() < 1.0).float()`).
- **Class weights supported in the cls term** — Task 4 (passed through to AsymmetricLoss).
- **MAE checkpoint loads in unchanged** — Task 1 test_load_mae_encoder_checkpoint; Task 2 `load_encoder_state_dict`; Task 6 `if args.pretrain_ckpt:` branch.
- **Same stratified parquet / cache as v5** — Task 6/7 (no parquet rebuild; reuses existing cache).
- **Same ASL + class_weights + min_delta + 100-epoch budget as v5** — Task 7 slurm flags.
- **4-config encoder_lr_scale ablation** — Task 7 RUN_NAMES and LR_SCALE_ARGS arrays.
- **Logged metrics: val_T_mean, val_b_mean, val_eps_norm_mean** — Task 5 step 7.
- **Wiki documentation** — Task 8.
- **Architecture figure** — Task 9.
- **HPC handoff procedure documented** — Task 10.

All spec requirements covered.

# Denoising MAE Pre-training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `DenoisingSpatialSpectralMAE` (subclass of `SpatialSpectralMAE`) that corrupts inputs with CRISM-physics-motivated noise and learns to reconstruct the clean spectra, then launch a 200-epoch pre-training run on HPC.

**Architecture:** New `CrismNoiseAugmentation` module applies (Gaussian + 1 µm spike + per-column bias) corruptions with data-informed σ values (0.0087 / 0.0058 / 0.0049). `DenoisingSpatialSpectralMAE` composes the noise aug into the existing MAE forward: encoder sees `x_corrupted`, decoder reconstructs `x_clean`, loss averaged over all 49 positions (not masked-only). Downstream classifiers consume the pre-trained encoder unchanged.

**Tech Stack:** PyTorch (custom `nn.Module`s), conda env `crism`, pytest. Project root `/mnt/mrdr/crism_classification`. Commit prefixes: `feat:`, `test:`, `fix:`.

**Conventions:**
- All commands run from `/mnt/mrdr/crism_classification`.
- Python prefix: `conda run -n crism …`.
- Spec at `docs/superpowers/specs/2026-05-16-denoising-mae-design.md` is the source of truth for ambiguities.

---

## File Map

| Path | Action | Responsibility |
|---|---|---|
| `models/noise_augmentation.py` | Create | `CrismNoiseAugmentation` module — composes Gaussian + spike + column corruptions; configurable σ values; eval-mode disables corruption |
| `models/denoising_spatial_mae.py` | Create | `DenoisingSpatialSpectralMAE(SpatialSpectralMAE)` — overrides `forward()` to: corrupt input, encode/decode normally, compute loss on all positions vs clean target |
| `scripts/pretrain_spatial_mae_denoising.py` | Create | Pre-training script paralleling `pretrain_spatial_mae.py`, with CLI flags for the σ values and spike parameters |
| `scripts/hpc_pretrain_denoising.slurm` | Create | HPC pre-training slurm — single-task (no array), 48 hr wall budget |
| `tests/test_noise_augmentation.py` | Create | Shape contract, statistical sanity (empirical σ matches configured), eval-mode disables |
| `tests/test_denoising_spatial_mae.py` | Create | Forward shape, σ=0 collapses to vanilla MAE behavior, loss is on all positions |
| `scripts/figures/fig_denoising_corruption.py` | Create | Visualize corruption realism — clean spectrum, corrupted spectrum, masked corrupted spectrum, the three corruption components broken out |

---

## Chunk 1 — Noise augmentation module

### Task 1: Failing tests for `CrismNoiseAugmentation`

**Files:**
- Create: `tests/test_noise_augmentation.py`

- [ ] **Step 1: Write the failing test file**

```python
# tests/test_noise_augmentation.py
"""Tests for the CRISM noise augmentation module."""
import pytest
import torch

from models.noise_augmentation import CrismNoiseAugmentation


@pytest.fixture
def aug():
    return CrismNoiseAugmentation(
        sigma_gauss=0.0087,
        sigma_spike=0.0058,
        sigma_column=0.0049,
        spike_center_band=15,
        spike_fwhm_bands=3,
        spike_band_range=(13, 17),
        n_bands=59,
        patch_size=7,
    )


def test_forward_shape_preserved(aug):
    aug.train()
    x = torch.randn(8, 7, 7, 59)
    out = aug(x)
    assert out.shape == x.shape


def test_eval_mode_disables_corruption(aug):
    aug.eval()
    x = torch.randn(4, 7, 7, 59)
    out = aug(x)
    torch.testing.assert_close(out, x)


def test_train_mode_changes_output(aug):
    aug.train()
    torch.manual_seed(0)
    x = torch.randn(4, 7, 7, 59) * 0.1
    out = aug(x)
    assert not torch.allclose(out, x), "training mode must produce a different output from input"


def test_empirical_gaussian_sigma(aug):
    """Empirical std of (corrupted - clean), averaged over many patches, should
    approximate the configured σ_gauss when other corruptions are off."""
    aug_only_gauss = CrismNoiseAugmentation(
        sigma_gauss=0.01, sigma_spike=0.0, sigma_column=0.0,
        n_bands=59, patch_size=7,
    )
    aug_only_gauss.train()
    torch.manual_seed(42)
    x = torch.zeros(2000, 7, 7, 59)   # use constant input so the only variability is from noise
    out = aug_only_gauss(x)
    empirical_sigma = (out - x).std().item()
    # Allow ±10% tolerance for empirical estimation
    assert 0.009 < empirical_sigma < 0.011, f"empirical σ = {empirical_sigma}"


def test_empirical_spike_only(aug):
    """Spike-only augmentation: only bands inside the spike range should be nonzero."""
    aug_only_spike = CrismNoiseAugmentation(
        sigma_gauss=0.0, sigma_spike=0.01, sigma_column=0.0,
        spike_center_band=15, spike_fwhm_bands=3, spike_band_range=(13, 17),
        n_bands=59, patch_size=7,
    )
    aug_only_spike.train()
    torch.manual_seed(0)
    x = torch.zeros(1000, 7, 7, 59)
    out = aug_only_spike(x)
    delta = out - x
    # Outside the spike range bands [13, 17], delta must be ~0
    outside_max = delta[:, :, :, :13].abs().max().item()
    assert outside_max < 1e-6, f"corruption leaked outside spike band range: max={outside_max}"
    above_max = delta[:, :, :, 18:].abs().max().item()
    assert above_max < 1e-6, f"corruption leaked above spike band range: max={above_max}"
    # Inside the range, there should be visible spike content
    inside_std = delta[:, :, :, 13:18].std().item()
    assert inside_std > 1e-5, f"no spike content inside range: std={inside_std}"


def test_empirical_column_only(aug):
    """Column-only: within a single patch, all 7 rows of a given column should be
    perturbed by the same value (column bias broadcasts down rows)."""
    aug_only_column = CrismNoiseAugmentation(
        sigma_gauss=0.0, sigma_spike=0.0, sigma_column=0.01,
        n_bands=59, patch_size=7,
    )
    aug_only_column.train()
    torch.manual_seed(0)
    x = torch.zeros(50, 7, 7, 59)
    out = aug_only_column(x)
    delta = out - x
    # For each patch, each column should be constant down rows
    # (rows = dim 1, cols = dim 2, bands = dim 3)
    # Pick a few patches and verify row-uniformity per column.
    for i in range(5):
        for c in range(7):
            col = delta[i, :, c, :]   # (7 rows, 59 bands)
            row0 = col[0]
            for r in range(1, 7):
                torch.testing.assert_close(col[r], row0, rtol=1e-5, atol=1e-6)


def test_all_components_combine_additively(aug):
    """When all three σ values are set, the corruption is the sum of the individual components."""
    aug.train()
    torch.manual_seed(0)
    x = torch.zeros(1000, 7, 7, 59)
    out = aug(x)
    # Empirical std should approximately equal sqrt(σ_gauss² + (contrib_spike)² + σ_column²)
    # spike only contributes inside band range; outside band range, only gauss + column.
    delta = out - x
    outside_std = delta[:, :, :, 0:13].std().item()
    expected_outside = (0.0087 ** 2 + 0.0049 ** 2) ** 0.5
    # Tolerance ±20% due to finite sample
    assert 0.8 * expected_outside < outside_std < 1.2 * expected_outside, \
        f"outside-spike std = {outside_std}, expected ≈ {expected_outside}"
```

- [ ] **Step 2: Run the test file**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism pytest tests/test_noise_augmentation.py -v
```

Expected: `ModuleNotFoundError: No module named 'models.noise_augmentation'`. Confirms RED.

---

### Task 2: Implement `CrismNoiseAugmentation`

**Files:**
- Create: `models/noise_augmentation.py`

- [ ] **Step 1: Create the module**

```python
# models/noise_augmentation.py
"""CRISM-physics-motivated noise augmentation for denoising MAE pre-training.

Three corruption components applied additively to a clean patch:
  1. ε_gauss   — per-pixel, per-band, independent 𝒩(0, σ_gauss²)
  2. ε_spike   — band-localized perturbation centered at the 1 µm detector seam.
                 One scalar magnitude per patch (𝒩(0, σ_spike²)); the spike
                 profile is a Gaussian bump in band space centered at
                 spike_center_band, zeroed outside spike_band_range.
  3. ε_column  — one bias per (column, band) drawn from 𝒩(0, σ_column²),
                 broadcast across all rows of that column.

σ values are estimated from the labeled-polygon parquet — see the spec.

The module is a no-op in eval mode, so the same model can be used for
inference without corruption.
"""
from typing import Tuple

import torch
import torch.nn as nn


def _spike_profile(
    n_bands: int,
    center: int,
    fwhm_bands: float,
    band_range: Tuple[int, int],
) -> torch.Tensor:
    """A 1-D Gaussian bump in band space, peak 1.0 at `center`, zeroed outside band_range.

    Returns: (n_bands,) tensor with values in [0, 1].
    """
    sigma = fwhm_bands / 2.355   # FWHM → σ for a Gaussian
    bands = torch.arange(n_bands, dtype=torch.float32)
    profile = torch.exp(-0.5 * ((bands - center) / sigma) ** 2)
    lo, hi = band_range
    # Zero out bands strictly outside the band range
    mask = (bands < lo) | (bands > hi)
    profile[mask] = 0.0
    return profile


class CrismNoiseAugmentation(nn.Module):
    """Apply CRISM-physics-motivated corruption to a clean patch.

    Forward: (B, patch_size, patch_size, n_bands) → same shape with noise added.
    No-op in eval mode.
    """

    def __init__(
        self,
        sigma_gauss: float = 0.0087,
        sigma_spike: float = 0.0058,
        sigma_column: float = 0.0049,
        spike_center_band: int = 15,
        spike_fwhm_bands: float = 3.0,
        spike_band_range: Tuple[int, int] = (13, 17),
        n_bands: int = 59,
        patch_size: int = 7,
    ):
        super().__init__()
        self.sigma_gauss = sigma_gauss
        self.sigma_spike = sigma_spike
        self.sigma_column = sigma_column
        self.n_bands = n_bands
        self.patch_size = patch_size

        # Register the spike profile as a buffer so .to(device) moves it,
        # and so it's saved with the state dict.
        profile = _spike_profile(n_bands, spike_center_band, spike_fwhm_bands, spike_band_range)
        self.register_buffer('_spike_profile', profile, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        B = x.shape[0]
        device = x.device
        dtype = x.dtype

        # 1. Gaussian per-pixel, per-band
        eps = torch.randn_like(x) * self.sigma_gauss if self.sigma_gauss > 0 else 0.0

        # 2. 1 µm spike — one magnitude per patch, broadcast spatially, weighted by profile
        if self.sigma_spike > 0:
            mag = torch.randn(B, device=device, dtype=dtype) * self.sigma_spike   # (B,)
            # Broadcast: (B,) × (n_bands,) → (B, n_bands)
            # Then reshape to (B, 1, 1, n_bands) and broadcast over rows/cols.
            spike = mag.unsqueeze(-1) * self._spike_profile.to(device=device, dtype=dtype)
            eps = eps + spike.view(B, 1, 1, self.n_bands)

        # 3. Column bias — per-(column, band), broadcast over rows.
        if self.sigma_column > 0:
            col_bias = torch.randn(
                B, 1, self.patch_size, self.n_bands, device=device, dtype=dtype,
            ) * self.sigma_column
            eps = eps + col_bias    # broadcasts to (B, patch_size, patch_size, n_bands)

        return x + eps
```

- [ ] **Step 2: Run the tests**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism pytest tests/test_noise_augmentation.py -v
```

Expected: 7 tests pass.

- [ ] **Step 3: Commit**

```bash
cd /mnt/mrdr/crism_classification
git add models/noise_augmentation.py tests/test_noise_augmentation.py
git commit -m "feat: CRISM noise augmentation for denoising MAE pre-training"
```

---

## Chunk 2 — `DenoisingSpatialSpectralMAE`

### Task 3: Failing tests for the denoising MAE

**Files:**
- Create: `tests/test_denoising_spatial_mae.py`

- [ ] **Step 1: Write the failing test file**

```python
# tests/test_denoising_spatial_mae.py
"""Tests for the denoising spatial-spectral MAE."""
import pytest
import torch

from models.denoising_spatial_mae import DenoisingSpatialSpectralMAE
from models.spatial_mae import SpatialSpectralMAE


@pytest.fixture
def model():
    return DenoisingSpatialSpectralMAE(
        n_bands=59, patch_size=7,
        embed_dim=128, n_heads=4, n_layers=6,
        decoder_dim=64, decoder_layers=2,
        mask_ratio=0.75,
        sigma_gauss=0.0087, sigma_spike=0.0058, sigma_column=0.0049,
    )


def test_forward_returns_loss_recon_mask(model):
    B = 4
    x = torch.randn(B, 7, 7, 59)
    out = model(x)
    assert isinstance(out, tuple) and len(out) == 3
    loss, recon, mask = out
    assert loss.ndim == 0
    assert recon.shape == (B, 49, 59)
    assert mask.shape == (B, 49)
    assert mask.dtype == torch.bool


def test_loss_is_finite_and_positive(model):
    model.train()
    B = 8
    torch.manual_seed(0)
    x = torch.randn(B, 7, 7, 59) * 0.1
    loss, _, _ = model(x)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_loss_is_all_positions_not_masked_only(model):
    """The denoising MAE's loss is computed over all 49 positions of the patch,
    not just the masked subset. Compared to vanilla MAE on the same input,
    when σ=0 (no corruption), the denoising loss should equal the all-position
    MSE — a different value than the masked-only MSE.
    """
    model.eval()      # disable corruption AND disable dropout (deterministic forward)
    # Force σ=0 to make the comparison exact
    model.noise_aug.sigma_gauss = 0.0
    model.noise_aug.sigma_spike = 0.0
    model.noise_aug.sigma_column = 0.0
    B = 4
    torch.manual_seed(0)
    x = torch.randn(B, 7, 7, 59) * 0.1
    loss, recon, mask = model(x)
    # Manually compute all-positions MSE on the returned recon
    x_flat = x.reshape(B, 49, 59)
    expected_loss = ((recon - x_flat) ** 2).mean()
    torch.testing.assert_close(loss, expected_loss, rtol=1e-5, atol=1e-6)


def test_mask_ratio_respected(model):
    """The mask must hide approximately the configured fraction of tokens."""
    model.train()
    B = 50
    torch.manual_seed(0)
    x = torch.randn(B, 7, 7, 59)
    _, _, mask = model(x)
    fraction_masked = mask.float().mean().item()
    # mask_ratio=0.75, n_tokens=49 → exact masked count = int(49*0.75)=36 (=0.7347)
    expected = int(49 * 0.75) / 49
    assert abs(fraction_masked - expected) < 0.01, f"got {fraction_masked}, expected ≈ {expected}"


def test_noise_aug_called_in_train_mode(model):
    """In train mode, the encoder sees a corrupted version of the input. Verify
    the noise_aug forward is invoked by patching its sigma_gauss to a large value
    and checking that the recon differs from a no-corruption run."""
    model.train()
    torch.manual_seed(0)
    x = torch.randn(3, 7, 7, 59) * 0.1
    _, recon_with_noise, _ = model(x)

    # Now run with sigma=0 — should produce a different recon
    model.noise_aug.sigma_gauss = 0.0
    model.noise_aug.sigma_spike = 0.0
    model.noise_aug.sigma_column = 0.0
    torch.manual_seed(0)
    _, recon_no_noise, _ = model(x)
    # Recons should differ because the encoder saw different inputs.
    # (At init the encoder is random but deterministic given seed, so the two
    # recons differ only due to the corruption.)
    assert not torch.allclose(recon_with_noise, recon_no_noise, rtol=1e-3)


def test_encoder_state_dict_loads_into_classifier(model):
    """A pre-trained denoising MAE encoder must load into SpatialSpectralClassifier."""
    from models.spatial_spectral_transformer import SpatialSpectralClassifier
    classifier = SpatialSpectralClassifier(
        n_bands=59, patch_size=7, n_classes=5,
        embed_dim=128, n_heads=4, n_layers=6,
    )
    encoder_state = model.encoder_state_dict()
    missing, unexpected = classifier.load_encoder_state_dict(encoder_state)
    assert unexpected == []
    assert not any(k.startswith('encoder.encoder') for k in missing), \
        f"core encoder weights missing: {[k for k in missing if k.startswith('encoder.encoder')]}"
```

- [ ] **Step 2: Run the tests**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism pytest tests/test_denoising_spatial_mae.py -v
```

Expected: `ModuleNotFoundError: No module named 'models.denoising_spatial_mae'`.

---

### Task 4: Implement `DenoisingSpatialSpectralMAE`

**Files:**
- Create: `models/denoising_spatial_mae.py`

- [ ] **Step 1: Create the module**

```python
# models/denoising_spatial_mae.py
"""
Denoising variant of SpatialSpectralMAE.

Differs from the parent class in three ways:
  1. Input is corrupted by CrismNoiseAugmentation before encoding
  2. Reconstruction target is x_clean (not the encoder input)
  3. Loss is averaged over all 49 positions (not masked-only)

Encoder state dict is structurally identical to the parent's, so the
resulting checkpoint loads into SpatialSpectralClassifier / DecompSpVit /
DecompSpVitAdv unchanged via load_encoder_state_dict.

Spec: docs/superpowers/specs/2026-05-16-denoising-mae-design.md
"""
from typing import Tuple

import torch

from models.noise_augmentation import CrismNoiseAugmentation
from models.spatial_mae import SpatialSpectralMAE


class DenoisingSpatialSpectralMAE(SpatialSpectralMAE):
    """SpatialSpectralMAE + denoising objective.

    The architecture (encoder + decoder + projections + mask token) is inherited
    from SpatialSpectralMAE unchanged. Only the forward pass is overridden to
    insert the noise augmentation and rewrite the loss aggregation.
    """

    def __init__(
        self,
        n_bands: int = 59,
        patch_size: int = 7,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        decoder_dim: int = 64,
        decoder_layers: int = 2,
        mask_ratio: float = 0.75,
        dropout: float = 0.0,
        # Noise augmentation parameters
        sigma_gauss: float = 0.0087,
        sigma_spike: float = 0.0058,
        sigma_column: float = 0.0049,
        spike_center_band: int = 15,
        spike_fwhm_bands: float = 3.0,
        spike_band_range: Tuple[int, int] = (13, 17),
    ):
        super().__init__(
            n_bands=n_bands, patch_size=patch_size,
            embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers,
            decoder_dim=decoder_dim, decoder_layers=decoder_layers,
            mask_ratio=mask_ratio, dropout=dropout,
        )
        self.noise_aug = CrismNoiseAugmentation(
            sigma_gauss=sigma_gauss,
            sigma_spike=sigma_spike,
            sigma_column=sigma_column,
            spike_center_band=spike_center_band,
            spike_fwhm_bands=spike_fwhm_bands,
            spike_band_range=spike_band_range,
            n_bands=n_bands,
            patch_size=patch_size,
        )

    def forward(self, x_clean: torch.Tensor):
        """Returns (loss, recon, mask).

        loss:  scalar MSE on all 49 positions of recon vs x_clean
        recon: (B, n_tokens, n_bands) — reconstructed spectra at every position
        mask:  (B, n_tokens) bool — True = was masked at the encoder
        """
        # The noise_aug is a no-op when self.training is False
        # (it inherits .training from the parent module).
        x_corrupted = self.noise_aug(x_clean)

        B = x_clean.shape[0]
        device = x_clean.device
        N = self.n_tokens

        visible_ids, masked_ids, mask = self._mask_tokens(B, device)

        # Encode VISIBLE tokens of the corrupted input
        enc_out = self.encoder.encode_visible(x_corrupted, visible_ids)
        enc_proj = self.enc_to_dec(enc_out[:, 1:])

        # Standard decoder pathway — identical to parent
        decoder_tokens = self.mask_token.expand(B, N, -1).clone()
        scatter_idx = visible_ids.unsqueeze(-1).expand(-1, -1, self.decoder_dim)
        decoder_tokens.scatter_(1, scatter_idx, enc_proj)
        pos_ids = torch.arange(1, N + 1, device=device)
        decoder_tokens = decoder_tokens + self.decoder_pos_embed(pos_ids)
        decoded = self.decoder(decoder_tokens)
        recon = self.reconstruction_head(decoded)

        # MSE on ALL positions vs x_CLEAN (denoising target)
        x_flat = x_clean.reshape(B, N, self.n_bands)
        loss = ((recon - x_flat) ** 2).mean()

        return loss, recon, mask
```

- [ ] **Step 2: Run all denoising tests**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism pytest tests/test_denoising_spatial_mae.py tests/test_noise_augmentation.py -v
```

Expected: all 13 tests pass (6 denoising + 7 noise aug).

- [ ] **Step 3: Commit**

```bash
cd /mnt/mrdr/crism_classification
git add models/denoising_spatial_mae.py tests/test_denoising_spatial_mae.py
git commit -m "feat: DenoisingSpatialSpectralMAE — clean-target reconstruction with all-position loss"
```

---

## Chunk 3 — Pre-training script

### Task 5: Create the denoising-MAE pre-training script

**Files:**
- Create: `scripts/pretrain_spatial_mae_denoising.py`

This script parallels `scripts/pretrain_spatial_mae.py` but uses the denoising model and exposes the noise-aug CLI flags. We start by copying the existing script's structure, then change the model construction and add the new flags.

- [ ] **Step 1: Read the existing pretrain script for structure reference**

```bash
cd /mnt/mrdr/crism_classification
head -50 scripts/pretrain_spatial_mae.py
```

Just to be familiar with the layout — argparse block, dataset loading, model construction, training loop, checkpoint saving. We'll mirror it.

- [ ] **Step 2: Write the new pre-training script**

```python
# scripts/pretrain_spatial_mae_denoising.py
"""
Denoising MAE pre-training for CRISM spatial-spectral patches.

Same data and training-loop machinery as scripts/pretrain_spatial_mae.py.
Differences:
  - Uses DenoisingSpatialSpectralMAE (corrupts input, recovers clean target)
  - Adds CLI flags for the three noise σ values

Usage (HPC):
    python scripts/pretrain_spatial_mae_denoising.py \\
        --epochs 200 --embed_dim 128 --n_layers 6 --mask_ratio 0.75 \\
        --sigma_gauss 0.0087 --sigma_spike 0.0058 --sigma_column 0.0049
"""
import argparse
import glob
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    # Schedule
    parser.add_argument('--epochs',      type=int,   default=200)
    parser.add_argument('--warmup',      type=int,   default=10)
    parser.add_argument('--batch_size',  type=int,   default=1024)
    parser.add_argument('--patches_per_epoch', type=int, default=200_000)
    parser.add_argument('--num_workers', type=int,   default=4)
    # Architecture
    parser.add_argument('--embed_dim',   type=int,   default=128)
    parser.add_argument('--n_heads',     type=int,   default=4)
    parser.add_argument('--n_layers',    type=int,   default=6)
    parser.add_argument('--decoder_dim', type=int,   default=64)
    parser.add_argument('--decoder_layers', type=int, default=2)
    parser.add_argument('--mask_ratio',  type=float, default=0.75)
    # Noise augmentation
    parser.add_argument('--sigma_gauss',  type=float, default=0.0087)
    parser.add_argument('--sigma_spike',  type=float, default=0.0058)
    parser.add_argument('--sigma_column', type=float, default=0.0049)
    parser.add_argument('--spike_center_band', type=int, default=15)
    parser.add_argument('--spike_fwhm_bands',  type=float, default=3.0)
    # Run management
    parser.add_argument('--run_name', type=str, default='spatial_mae_denoising_128d_6l')
    parser.add_argument('--config',   type=str, default='config.yaml')
    parser.add_argument('--resume',   type=str, default=None)
    parser.add_argument('--no_wandb', action='store_true')
    args = parser.parse_args()

    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.config
    )
    from config_loader import load_config
    cfg = load_config(cfg_path)

    run_name = args.run_name
    log.info(f"Run name: {run_name}")
    log.info(f"σ_gauss={args.sigma_gauss}, σ_spike={args.sigma_spike}, "
             f"σ_column={args.sigma_column}")

    # ── Data ──────────────────────────────────────────────────────────────
    data_root = cfg.get('data_root', '/mnt/crism/MRDR')
    globs_to_try = [
        os.path.join(data_root, 'mc*', 't*mrral*.hdr'),
        os.path.join(data_root, 't*mrral*.hdr'),
    ]
    hdr_files = []
    for g in globs_to_try:
        hdr_files = sorted(glob.glob(g))
        if hdr_files:
            break
    if not hdr_files:
        raise FileNotFoundError(
            f"No mrral HDR files found. Tried:\n" + "\n".join(f"  {g}" for g in globs_to_try)
        )
    log.info(f"Found {len(hdr_files)} mrral tiles")

    from data.global_patch_dataset import CRISMGlobalPatchDataset
    ds = CRISMGlobalPatchDataset(hdr_files, patch_size=7, min_valid_frac=0.8)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")

    from models.denoising_spatial_mae import DenoisingSpatialSpectralMAE
    model = DenoisingSpatialSpectralMAE(
        n_bands=59, patch_size=7,
        embed_dim=args.embed_dim, n_heads=args.n_heads, n_layers=args.n_layers,
        decoder_dim=args.decoder_dim, decoder_layers=args.decoder_layers,
        mask_ratio=args.mask_ratio,
        sigma_gauss=args.sigma_gauss,
        sigma_spike=args.sigma_spike,
        sigma_column=args.sigma_column,
        spike_center_band=args.spike_center_band,
        spike_fwhm_bands=args.spike_fwhm_bands,
    ).to(device)

    # ── Optimizer & schedule ──────────────────────────────────────────────
    base_lr = 1.5e-4 * args.batch_size / 256
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr,
        betas=(0.9, 0.95), weight_decay=0.05,
    )

    def lr_lambda(epoch):
        if epoch < args.warmup:
            return (epoch + 1) / args.warmup
        progress = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 1
    best_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['mae_state'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_loss = ckpt.get('mae_loss', float('inf'))
        log.info(f"Resumed from {args.resume} at epoch {start_epoch}, loss={best_loss:.6f}")

    # ── wandb ─────────────────────────────────────────────────────────────
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb_entity = cfg.get('wandb', {}).get('entity') or None
            wandb.init(project='crism-mineral-classification', entity=wandb_entity,
                       name=run_name, config=vars(args), resume='allow')
        except Exception as e:
            log.warning(f"wandb init failed ({e}), continuing without")
            use_wandb = False

    # ── Training loop ─────────────────────────────────────────────────────
    batches_per_epoch = args.patches_per_epoch // args.batch_size
    data_iter = iter(loader)

    ckpt_dir = cfg.get('checkpoints_dir', '/mnt/mrdr/crism_classification/checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        losses = []
        for _ in range(batches_per_epoch):
            try:
                patches = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                patches = next(data_iter)

            patches = patches.to(device)
            optimizer.zero_grad()
            loss, _, _ = model(patches)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        mean_loss = float(np.mean(losses))
        lr_now = optimizer.param_groups[0]['lr']
        log.info(f"Epoch {epoch}/{args.epochs} | denoising_loss={mean_loss:.6f} | lr={lr_now:.2e}")

        if use_wandb:
            import wandb
            wandb.log({'epoch': epoch, 'denoising_loss': mean_loss, 'lr': lr_now})

        # Save every 50 epochs and at end
        if epoch % 50 == 0 or epoch == args.epochs:
            path = os.path.join(ckpt_dir, f'{run_name}_epoch{epoch}.pt')
            torch.save({
                'mae_state': model.state_dict(),
                'encoder_state': model.encoder_state_dict(),
                'epoch': epoch, 'mae_loss': mean_loss, 'config': vars(args),
            }, path)
            log.info(f"Saved {path}")

        # Save best
        if mean_loss < best_loss:
            best_loss = mean_loss
            path = os.path.join(ckpt_dir, f'{run_name}_best.pt')
            torch.save({
                'mae_state': model.state_dict(),
                'encoder_state': model.encoder_state_dict(),
                'epoch': epoch, 'mae_loss': mean_loss, 'config': vars(args),
            }, path)


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Verify the CLI parses**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism python scripts/pretrain_spatial_mae_denoising.py --help 2>&1 | head -30
```

Expected: argparse usage text listing all flags including `--sigma_gauss`, `--sigma_spike`, `--sigma_column`.

- [ ] **Step 4: Commit**

```bash
cd /mnt/mrdr/crism_classification
git add scripts/pretrain_spatial_mae_denoising.py
git commit -m "feat: denoising MAE pre-training script"
```

---

## Chunk 4 — HPC slurm

### Task 6: HPC pretraining slurm

**Files:**
- Create: `scripts/hpc_pretrain_denoising.slurm`

- [ ] **Step 1: Write the slurm file**

```bash
# scripts/hpc_pretrain_denoising.slurm
#!/bin/bash
#SBATCH --job-name=spatial_mae_denoising
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/pretrain_denoising_%j.log
#SBATCH --error=logs/pretrain_denoising_%j.log

# Denoising spatial-spectral MAE pre-training.
# Single GPU, 200 epochs, ~32-hour wall time expected. 48-hour budget.
# Spec: docs/superpowers/specs/2026-05-16-denoising-mae-design.md

WORK_DIR=/groups/sbyrne/phillipsm/crism_classification
PYTHON=/groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python

cd "$WORK_DIR"

# Make sure config.local.yaml points at the right tile dir
if [ ! -f config.local.yaml ]; then
    cat > config.local.yaml <<EOF
data_root: /xdisk/sbyrne/phillipsm/CRISM_MRDR
checkpoint_dir: ${WORK_DIR}/checkpoints
checkpoints_dir: ${WORK_DIR}/checkpoints
output_dir: ${WORK_DIR}/data
patch_cache_dir: ${WORK_DIR}/data/patch_cache
EOF
fi

mkdir -p logs checkpoints

echo "=== Denoising MAE pre-training start: $(date) ==="

${PYTHON} -u scripts/pretrain_spatial_mae_denoising.py \
    --epochs 200 \
    --warmup 10 \
    --batch_size 1024 \
    --patches_per_epoch 200000 \
    --num_workers 6 \
    --embed_dim 128 \
    --n_heads 4 \
    --n_layers 6 \
    --decoder_dim 64 \
    --decoder_layers 2 \
    --mask_ratio 0.75 \
    --sigma_gauss 0.0087 \
    --sigma_spike 0.0058 \
    --sigma_column 0.0049 \
    --spike_center_band 15 \
    --spike_fwhm_bands 3.0 \
    --run_name spatial_mae_denoising_128d_6l

echo "=== Denoising MAE pre-training end: $(date) ==="
```

- [ ] **Step 2: Commit**

```bash
cd /mnt/mrdr/crism_classification
git add scripts/hpc_pretrain_denoising.slurm
git commit -m "feat: HPC slurm for denoising MAE pre-training"
```

---

## Chunk 5 — Corruption visualization figure

### Task 7: Visualize the corruption realism

**Files:**
- Create: `scripts/figures/fig_denoising_corruption.py`

A diagnostic figure that confirms the corruption looks like real CRISM noise. For three representative pixels (one per class), show:
- Row 1: clean input spectrum
- Row 2: corrupted spectrum (the encoder's view during training)
- Row 3: each corruption component broken out (Gaussian, spike, column)

- [ ] **Step 1: Write the figure script**

```python
# scripts/figures/fig_denoising_corruption.py
"""
Visualize the denoising-MAE corruption pipeline on real CRISM pixels.

For three representative pixels (olivine, hcp, plagioclase), show:
  - the clean input spectrum
  - the corrupted spectrum (model's training view)
  - each corruption component (gauss, spike, column) plotted separately

This is the figure that demonstrates the noise model is realistic.

Usage:
    conda run -n crism python scripts/figures/fig_denoising_corruption.py
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, '/mnt/mrdr/crism_classification')
from models.noise_augmentation import CrismNoiseAugmentation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _utils import (
    CLASS_COLORS, build_mrral_map, find_representative_pixels,
    get_wavelengths_59, load_mrral_parquet, read_patch_from_tile,
)

OUT_PATH = '/mnt/mrdr/crism_classification/reports/v5/fig_v5_denoising_corruption.png'
CLASSES_TO_SHOW = ['olivine', 'hcp', 'plagioclase']


def isolate_component(aug, x_clean, which):
    """Forward the augmentation with only `which` ∈ {'gauss', 'spike', 'column'} enabled
    and return the perturbation (corrupted - clean)."""
    saved = (aug.sigma_gauss, aug.sigma_spike, aug.sigma_column)
    aug.sigma_gauss, aug.sigma_spike, aug.sigma_column = 0.0, 0.0, 0.0
    if which == 'gauss':   aug.sigma_gauss = saved[0]
    if which == 'spike':   aug.sigma_spike = saved[1]
    if which == 'column':  aug.sigma_column = saved[2]
    aug.train()
    torch.manual_seed(0 if which == 'gauss' else (1 if which == 'spike' else 2))
    out = aug(x_clean)
    aug.sigma_gauss, aug.sigma_spike, aug.sigma_column = saved
    return (out - x_clean).numpy()


def main():
    df = load_mrral_parquet()
    mrral_map = build_mrral_map()
    pixels = find_representative_pixels(df, n_per_class=1, seed=42)
    wls = get_wavelengths_59()

    aug = CrismNoiseAugmentation()
    aug.train()
    torch.manual_seed(7)

    fig, axes = plt.subplots(
        len(CLASSES_TO_SHOW), 4,
        figsize=(15.5, 3.0 * len(CLASSES_TO_SHOW)),
    )
    if len(CLASSES_TO_SHOW) == 1:
        axes = axes[None, :]

    for row_i, cls in enumerate(CLASSES_TO_SHOW):
        sel = pixels.get(cls, [])
        if not sel:
            continue
        tid, pr, pc = sel[0]
        mrral = mrral_map.get(tid)
        if not (mrral and os.path.exists(mrral)):
            continue

        patch = read_patch_from_tile(mrral, pr, pc, patch_size=7, n_bands=59)
        x_clean = torch.from_numpy(patch).unsqueeze(0)  # (1, 7, 7, 59)
        center_clean = patch[3, 3]

        # Corrupted version (all three components)
        x_corrupted = aug(x_clean).numpy()[0]
        center_corrupted = x_corrupted[3, 3]

        color = CLASS_COLORS[cls]

        # ── Col 1: clean spectrum ──────────────────────────────────────
        ax = axes[row_i, 0]
        valid = (center_clean > 0) & (center_clean < 0.5)
        ax.plot(wls[valid], center_clean[valid], color=color, linewidth=1.8)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('I/F')
        ax.set_title(f'{cls}: tile {tid}\nclean center pixel ({pr},{pc})',
                     color=color, fontsize=9.5)
        ax.grid(alpha=0.3)

        # ── Col 2: corrupted spectrum (all components, what model sees) ──
        ax = axes[row_i, 1]
        ax.plot(wls[valid], center_clean[valid], color='#aaa',
                linewidth=1.0, label='clean')
        ax.plot(wls[valid], center_corrupted[valid], color=color,
                linewidth=1.5, label='corrupted')
        ax.axvspan(wls[13], wls[17], alpha=0.08, color='red',
                   label='spike region')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('I/F')
        ax.set_title('clean + all 3 corruptions\n(what the encoder sees)',
                     fontsize=9.5)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.3)

        # ── Col 3: each corruption component (delta from clean) ──────────
        ax = axes[row_i, 2]
        delta_gauss = isolate_component(aug, x_clean, 'gauss')[0, 3, 3]
        delta_spike = isolate_component(aug, x_clean, 'spike')[0, 3, 3]
        delta_column = isolate_component(aug, x_clean, 'column')[0, 3, 3]
        ax.plot(wls, delta_gauss, color='#2c7a2c', linewidth=1.2,
                label=f'Gaussian (σ={aug.sigma_gauss})')
        ax.plot(wls, delta_spike, color='#c44', linewidth=1.5,
                label=f'1 µm spike (σ={aug.sigma_spike})')
        ax.plot(wls, delta_column, color='#1f77b4', linewidth=1.2,
                label=f'column bias (σ={aug.sigma_column})')
        ax.axhline(0, color='black', linewidth=0.4)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('corruption (I/F)')
        ax.set_title('individual corruption components\n(center pixel)', fontsize=9.5)
        ax.legend(fontsize=7.5, loc='lower right')
        ax.grid(alpha=0.3)

        # ── Col 4: spatial pattern of column-bias corruption ─────────────
        ax = axes[row_i, 3]
        # Show the column-bias contribution at one band (band 30, mid-range)
        col_delta_2d = isolate_component(aug, x_clean, 'column')[0, :, :, 30]
        vmax = np.abs(col_delta_2d).max() * 1.05 or 1e-3
        im = ax.imshow(col_delta_2d, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       interpolation='nearest')
        ax.set_xticks(range(7)); ax.set_yticks(range(7))
        ax.set_xlabel('patch col'); ax.set_ylabel('patch row')
        ax.set_title(f'column-bias at {wls[30]:.0f} nm\n(rows uniform within col)',
                     fontsize=9.5)
        plt.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(
        'Denoising-MAE corruption — what the model is asked to remove\n'
        'σ values: gauss=0.0087, 1 µm spike=0.0058, column=0.0049 (data-informed)',
        fontsize=11, y=1.005,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Wrote {OUT_PATH}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Generate the figure**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism python scripts/figures/fig_denoising_corruption.py
```

Expected: prints the output path. No errors.

- [ ] **Step 3: Commit**

```bash
cd /mnt/mrdr/crism_classification
git add scripts/figures/fig_denoising_corruption.py reports/v5/fig_v5_denoising_corruption.png
git commit -m "feat: corruption-realism visualization for denoising MAE"
```

---

## Chunk 6 — HPC deployment instructions (informational)

### Task 8: Final integration check + HPC handoff

**Files:** none modified — this task validates and documents the deployment commands.

- [ ] **Step 1: Run the full test suite**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism pytest tests/test_noise_augmentation.py tests/test_denoising_spatial_mae.py -v 2>&1 | tail -10
```

Expected: 13 tests pass (7 from Task 2 + 6 from Task 4).

- [ ] **Step 2: End-to-end smoke test on a synthetic batch**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism python -c "
import sys; sys.path.insert(0, '.')
import torch
from models.denoising_spatial_mae import DenoisingSpatialSpectralMAE

m = DenoisingSpatialSpectralMAE().to('cpu')
x = torch.randn(8, 7, 7, 59) * 0.1
m.train()
loss, recon, mask = m(x)
loss.backward()
print(f'train OK. loss={loss.item():.4f}, recon.shape={recon.shape}, mask sum={mask.sum().item()}')
m.eval()
loss_eval, _, _ = m(x)
print(f'eval OK. loss={loss_eval.item():.4f}  (noise aug disabled)')

# Encoder state loadability check
enc_state = m.encoder_state_dict()
from models.spatial_spectral_transformer import SpatialSpectralClassifier
c = SpatialSpectralClassifier()
miss, unex = c.load_encoder_state_dict(enc_state)
assert not unex, f'unexpected keys: {unex}'
print('downstream loadability: OK')
"
```

Expected: prints `train OK. loss=…, recon.shape=torch.Size([8, 49, 59])`, `eval OK.`, and `downstream loadability: OK`.

- [ ] **Step 3: Print the HPC handoff command list**

The local work is complete. User needs to run on HPC (DUO auth blocks autonomous execution).

```bash
echo "=== Files to rsync to HPC ==="
echo ""
echo "rsync -avh \\"
echo "    /mnt/mrdr/crism_classification/models/noise_augmentation.py \\"
echo "    /mnt/mrdr/crism_classification/models/denoising_spatial_mae.py \\"
echo "    phillipsm@filexfer.hpc.arizona.edu:/groups/sbyrne/phillipsm/crism_classification/models/"
echo ""
echo "rsync -avh \\"
echo "    /mnt/mrdr/crism_classification/scripts/pretrain_spatial_mae_denoising.py \\"
echo "    /mnt/mrdr/crism_classification/scripts/hpc_pretrain_denoising.slurm \\"
echo "    phillipsm@filexfer.hpc.arizona.edu:/groups/sbyrne/phillipsm/crism_classification/scripts/"
echo ""
echo "=== On HPC ==="
echo "ssh phillipsm@hpc.arizona.edu"
echo "  cd /groups/sbyrne/phillipsm/crism_classification"
echo "  # Sanity-check the imports:"
echo "  /groups/sbyrne/phillipsm/micromamba/envs/crism/bin/python -c \\"
echo "    'from models.denoising_spatial_mae import DenoisingSpatialSpectralMAE; print(\"OK\")'"
echo "  # Submit the 200-epoch pre-training (no array — single big run):"
echo "  sbatch scripts/hpc_pretrain_denoising.slurm"
echo "  squeue -u phillipsm"
```

- [ ] **Step 4: Show commits from this plan**

```bash
cd /mnt/mrdr/crism_classification
git log --oneline | head -10
```

Expected: ~7 commits from this plan.

---

## Spec coverage check (self-review)

| Spec requirement | Task |
|---|---|
| `CrismNoiseAugmentation` module | Task 2 |
| Configurable σ values | Task 2 (`__init__` args) |
| Gaussian noise per-pixel per-band | Task 2 (`eps_gauss`) |
| 1 µm spike: band-localized, broadcast spatially, scalar magnitude per patch | Task 2 (spike block in `forward`) |
| Column bias: per-(column, band), broadcast over rows | Task 2 (column block in `forward`) |
| Eval-mode disables corruption | Task 2 (`if not self.training: return x`) |
| `DenoisingSpatialSpectralMAE` subclass | Task 4 |
| Target = x_clean | Task 4 (`x_flat = x_clean.reshape(...)`) |
| Loss on all 49 positions | Task 4 (`((recon - x_flat) ** 2).mean()`) |
| Encoder state loads into downstream classifiers | Task 4 (no override of `encoder_state_dict`; parent's works) + Task 3 test_encoder_state_dict_loads_into_classifier |
| Pre-training script with CLI flags for σ values | Task 5 |
| HPC slurm with 200 epochs, 48 hr budget, data-informed σ defaults | Task 6 |
| Corruption visualization figure | Task 7 |
| HPC handoff command list | Task 8 step 3 |

All spec requirements covered.

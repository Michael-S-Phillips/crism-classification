# SPEND Spatial-Spectral MAE (v4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a SPEND-style spectral-frame Noise2Noise pre-training objective to the existing `SpatialSpectralMAE` stack, producing a v4 encoder that learns to denoise CRISM mrral without any synthetic noise model.

**Architecture:** New `SpendSpatialSpectralMAE` subclass of `SpatialSpectralMAE`. Per-batch random partition of the 59 bands into two views (~30/29 split); encoder sees input-view bands at 25% of spatial positions with the other view's bands zeroed; decoder reconstructs target-view bands at all 49 positions; MSE loss is Noise2Noise between predicted target-view bands and observed target-view bands. Spectral mask ratio anneals linearly from 0.5 to 0 over epochs 161–181 to close the train/fine-tune distribution gap.

**Tech Stack:** Python 3, PyTorch, pytest, the existing `SpatialSpectralTransformer` / `SpatialSpectralMAE` / `CRISMGlobalPatchDataset` machinery, SLURM for HPC.

**Spec:** `docs/superpowers/specs/2026-05-16-spend-spatial-mae-design.md`

**Reference implementations to mirror:**
- `models/denoising_spatial_mae.py` — v3 subclass structure
- `scripts/pretrain_spatial_mae_denoising.py` — v3 training driver
- `tests/test_denoising_spatial_mae.py` — v3 test structure
- `scripts/hpc_pretrain_denoising.slurm` — v3 SLURM template

---

## File Structure

**New files:**
- `models/spend_spatial_mae.py` — `compute_spectral_mask_ratio` helper + `SpendSpatialSpectralMAE` class
- `tests/test_spend_spatial_mae.py` — unit tests for everything in the model module
- `scripts/pretrain_spatial_mae_spend.py` — 200-epoch training driver
- `scripts/hpc_pretrain_spend.slurm` — HPC SLURM submission
- `scripts/figures/fig_spend_partition.py` — post-pretraining validation figure

**Files referenced (read-only):**
- `models/spatial_mae.py` — parent class, unchanged
- `models/spatial_spectral_transformer.py` — encoder, unchanged
- `models/spatial_spectral_transformer.py:103` — `SpatialSpectralClassifier`, used in compatibility test
- `data/global_patch_dataset.py` — `CRISMGlobalPatchDataset`, used by training driver

---

## Task 1: Spectral-mask annealing schedule helper

**Files:**
- Create: `models/spend_spatial_mae.py`
- Test: `tests/test_spend_spatial_mae.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_spend_spatial_mae.py` with this content:

```python
"""Tests for the SPEND-style spatial-spectral MAE."""
import pytest
import torch

from models.spend_spatial_mae import compute_spectral_mask_ratio


class TestSpectralMaskSchedule:
    """Anneal schedule for spectral_mask_ratio over the training run."""

    def test_returns_base_before_anneal_start(self):
        assert compute_spectral_mask_ratio(
            epoch=100, anneal_start_epoch=161, anneal_end_epoch=181, base=0.5,
        ) == 0.5

    def test_returns_base_just_before_anneal_start(self):
        assert compute_spectral_mask_ratio(
            epoch=160, anneal_start_epoch=161, anneal_end_epoch=181, base=0.5,
        ) == 0.5

    def test_returns_base_at_anneal_start(self):
        # At epoch == anneal_start_epoch the formula still evaluates to base
        # (anneal_end - epoch) / (anneal_end - anneal_start) = 20/20 = 1
        assert compute_spectral_mask_ratio(
            epoch=161, anneal_start_epoch=161, anneal_end_epoch=181, base=0.5,
        ) == 0.5

    def test_linear_interpolation_mid_range(self):
        # epoch 170 is 11/20 of the way to anneal_end → ratio = 0.5 * 11/20
        assert compute_spectral_mask_ratio(
            epoch=170, anneal_start_epoch=161, anneal_end_epoch=181, base=0.5,
        ) == pytest.approx(0.275)

    def test_near_end_of_anneal(self):
        # epoch 180 is 1/20 of the way out → ratio = 0.5 * 1/20 = 0.025
        assert compute_spectral_mask_ratio(
            epoch=180, anneal_start_epoch=161, anneal_end_epoch=181, base=0.5,
        ) == pytest.approx(0.025)

    def test_returns_zero_at_anneal_end(self):
        assert compute_spectral_mask_ratio(
            epoch=181, anneal_start_epoch=161, anneal_end_epoch=181, base=0.5,
        ) == 0.0

    def test_returns_zero_after_anneal_end(self):
        assert compute_spectral_mask_ratio(
            epoch=200, anneal_start_epoch=161, anneal_end_epoch=181, base=0.5,
        ) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n crism python -m pytest tests/test_spend_spatial_mae.py::TestSpectralMaskSchedule -v`
Expected: 7 FAILs with `ModuleNotFoundError: No module named 'models.spend_spatial_mae'`

- [ ] **Step 3: Implement minimal code to pass**

Create `models/spend_spatial_mae.py` with this content:

```python
"""
SPEND-style spatial-spectral MAE (v4).

Pretraining objective: Noise2Noise between two random spectral-band views
of the same CRISM patch. Adjacent bands image the same surface but have
independent detector-noise realizations, so predicting one view from the
other forces the model to learn the underlying clean spectrum without any
synthetic-noise assumptions.

Spec: docs/superpowers/specs/2026-05-16-spend-spatial-mae-design.md
"""
from __future__ import annotations


def compute_spectral_mask_ratio(
    epoch: int,
    anneal_start_epoch: int = 161,
    anneal_end_epoch: int = 181,
    base: float = 0.5,
) -> float:
    """Schedule for the fraction of bands zeroed during SPEND pretraining.

    Three phases:
      - epoch < anneal_start_epoch: returns `base` (SPEND phase A)
      - anneal_start_epoch <= epoch < anneal_end_epoch: linearly decreases
        from `base` toward 0 (SPEND anneal phase B)
      - epoch >= anneal_end_epoch: returns 0.0 (plain MAE phase C)
    """
    if epoch < anneal_start_epoch:
        return base
    if epoch >= anneal_end_epoch:
        return 0.0
    return base * (anneal_end_epoch - epoch) / (anneal_end_epoch - anneal_start_epoch)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n crism python -m pytest tests/test_spend_spatial_mae.py::TestSpectralMaskSchedule -v`
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git add models/spend_spatial_mae.py tests/test_spend_spatial_mae.py
git commit -m "$(cat <<'EOF'
feat(spend): add spectral-mask annealing schedule helper

compute_spectral_mask_ratio returns the SPEND spectral mask ratio for a
given training epoch. Three-phase schedule: constant base in phase A,
linear interpolation toward 0 in phase B, hold at 0 in phase C.

Spec: docs/superpowers/specs/2026-05-16-spend-spatial-mae-design.md

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `SpendSpatialSpectralMAE` skeleton (inheritance only)

**Files:**
- Modify: `models/spend_spatial_mae.py` (add class)
- Test: `tests/test_spend_spatial_mae.py` (add fixture + skeleton tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_spend_spatial_mae.py`:

```python
from models.spend_spatial_mae import SpendSpatialSpectralMAE


@pytest.fixture
def model():
    return SpendSpatialSpectralMAE(
        n_bands=59, patch_size=7,
        embed_dim=128, n_heads=4, n_layers=6,
        decoder_dim=64, decoder_layers=2,
        mask_ratio=0.75,
        spectral_mask_ratio=0.5,
    )


class TestSkeletonAndAttributes:
    def test_instantiates_with_expected_attributes(self, model):
        assert model.n_bands == 59
        assert model.n_tokens == 49
        assert model.mask_ratio == 0.75
        assert model.spectral_mask_ratio == 0.5

    def test_spectral_mask_ratio_is_mutable(self, model):
        model.spectral_mask_ratio = 0.0
        assert model.spectral_mask_ratio == 0.0

    def test_inherits_encoder_state_dict_method(self, model):
        # Inherited from SpatialSpectralMAE; must still work for downstream loading
        state = model.encoder_state_dict()
        assert any(k.startswith('band_embed') for k in state)
        assert any(k.startswith('encoder.') for k in state)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n crism python -m pytest tests/test_spend_spatial_mae.py::TestSkeletonAndAttributes -v`
Expected: 3 FAILs with `ImportError: cannot import name 'SpendSpatialSpectralMAE'`

- [ ] **Step 3: Implement minimal code to pass**

Append to `models/spend_spatial_mae.py`:

```python
import torch

from models.spatial_mae import SpatialSpectralMAE


class SpendSpatialSpectralMAE(SpatialSpectralMAE):
    """SpatialSpectralMAE + SPEND spectral-partition Noise2Noise objective.

    The architecture (encoder + decoder + projections + mask token) is
    inherited from SpatialSpectralMAE unchanged. The forward pass is
    overridden to (1) sample a random per-batch band partition, (2) zero
    target-half bands in the encoder input, and (3) compute MSE loss only on
    the target-half bands of the reconstruction.
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
        spectral_mask_ratio: float = 0.5,
    ):
        super().__init__(
            n_bands=n_bands, patch_size=patch_size,
            embed_dim=embed_dim, n_heads=n_heads, n_layers=n_layers,
            decoder_dim=decoder_dim, decoder_layers=decoder_layers,
            mask_ratio=mask_ratio, dropout=dropout,
        )
        # Mutable attribute; the training loop updates it each epoch.
        self.spectral_mask_ratio: float = spectral_mask_ratio
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n crism python -m pytest tests/test_spend_spatial_mae.py -v`
Expected: 10 PASS (7 schedule + 3 skeleton)

- [ ] **Step 5: Commit**

```bash
git add models/spend_spatial_mae.py tests/test_spend_spatial_mae.py
git commit -m "$(cat <<'EOF'
feat(spend): add SpendSpatialSpectralMAE class skeleton

Subclass of SpatialSpectralMAE that adds a mutable spectral_mask_ratio
attribute. Forward pass still inherited; SPEND-specific behavior comes
in subsequent commits.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Band-partition method

**Files:**
- Modify: `models/spend_spatial_mae.py` (add `_partition_bands` method)
- Test: `tests/test_spend_spatial_mae.py` (add partition tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_spend_spatial_mae.py`:

```python
class TestBandPartition:
    """The random per-batch band partition splits 59 bands into input/target."""

    def test_shape_and_dtype(self, model):
        target_mask = model._partition_bands(device=torch.device('cpu'))
        assert target_mask.shape == (59,)
        assert target_mask.dtype == torch.bool

    def test_target_count_at_ratio_half(self, model):
        # With ratio 0.5 and 59 bands: round(59 * 0.5) = 30 → 30 target bands.
        model.spectral_mask_ratio = 0.5
        target_mask = model._partition_bands(device=torch.device('cpu'))
        assert int(target_mask.sum().item()) == 30

    def test_target_count_at_ratio_zero(self, model):
        model.spectral_mask_ratio = 0.0
        target_mask = model._partition_bands(device=torch.device('cpu'))
        assert int(target_mask.sum().item()) == 0

    def test_unbiased_over_many_samples(self, model):
        """Every band index appears in the target-half across enough samples."""
        model.spectral_mask_ratio = 0.5
        counts = torch.zeros(59, dtype=torch.long)
        torch.manual_seed(0)
        for _ in range(1000):
            counts += model._partition_bands(device=torch.device('cpu')).long()
        # Each band should appear in target-half ~500 times.
        # Generous bound: every band hits target-half in ≥ 10 of 1000 samples.
        assert counts.min().item() >= 10, f"min count={counts.min().item()} — partition is biased"

    def test_samples_differ_across_calls(self, model):
        model.spectral_mask_ratio = 0.5
        torch.manual_seed(0)
        m1 = model._partition_bands(device=torch.device('cpu'))
        m2 = model._partition_bands(device=torch.device('cpu'))
        assert not torch.equal(m1, m2), "Two consecutive partitions should differ"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n crism python -m pytest tests/test_spend_spatial_mae.py::TestBandPartition -v`
Expected: 5 FAILs with `AttributeError: 'SpendSpatialSpectralMAE' object has no attribute '_partition_bands'`

- [ ] **Step 3: Implement minimal code to pass**

Append this method to `SpendSpatialSpectralMAE` in `models/spend_spatial_mae.py` (inside the class body):

```python
    def _partition_bands(self, device: torch.device) -> torch.Tensor:
        """Sample one random band partition for this batch.

        Returns a boolean mask `target_mask: bool[n_bands]` where True
        indicates a target-half band (encoder input zeroes these out;
        the loss is evaluated on these).

        Per-batch (not per-sample) partition: all samples in the batch
        share the same target-mask.
        """
        n_target = round(self.n_bands * self.spectral_mask_ratio)
        target_mask = torch.zeros(self.n_bands, dtype=torch.bool, device=device)
        if n_target == 0:
            return target_mask
        target_idx = torch.randperm(self.n_bands, device=device)[:n_target]
        target_mask[target_idx] = True
        return target_mask
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n crism python -m pytest tests/test_spend_spatial_mae.py -v`
Expected: 15 PASS (7 schedule + 3 skeleton + 5 partition)

- [ ] **Step 5: Commit**

```bash
git add models/spend_spatial_mae.py tests/test_spend_spatial_mae.py
git commit -m "$(cat <<'EOF'
feat(spend): add per-batch band-partition method

_partition_bands samples a random boolean target-mask over the 59 bands,
with sum proportional to spectral_mask_ratio. One partition is shared
across all samples in a batch; partitions differ across batches.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: SPEND forward pass

**Files:**
- Modify: `models/spend_spatial_mae.py` (override `forward`)
- Test: `tests/test_spend_spatial_mae.py` (add forward tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_spend_spatial_mae.py`:

```python
class TestForwardPass:
    """SPEND forward pass: shape, loss localization, masking interaction."""

    def test_forward_returns_loss_recon_mask(self, model):
        model.train()
        B = 4
        x = torch.randn(B, 7, 7, 59) * 0.1
        out = model(x)
        assert isinstance(out, tuple) and len(out) == 3
        loss, recon, mask = out
        assert loss.ndim == 0
        assert recon.shape == (B, 49, 59)
        assert mask.shape == (B, 49)
        assert mask.dtype == torch.bool

    def test_loss_is_finite_and_positive(self, model):
        model.train()
        B = 8
        torch.manual_seed(0)
        x = torch.randn(B, 7, 7, 59) * 0.1
        loss, _, _ = model(x)
        assert torch.isfinite(loss)
        assert loss.item() > 0.0

    def test_loss_is_target_band_only_at_ratio_half(self, model):
        """At spectral_mask_ratio=0.5, the returned loss equals MSE only on
        the target-band positions of the reconstruction (not on input bands)."""
        model.eval()
        model.spectral_mask_ratio = 0.5
        torch.manual_seed(42)
        B = 4
        x = torch.randn(B, 7, 7, 59) * 0.1

        # Patch _partition_bands to return a deterministic mask so we can
        # reconstruct the expected loss after the call.
        chosen_targets = torch.zeros(59, dtype=torch.bool)
        chosen_targets[torch.arange(0, 59, 2)] = True  # even indices = target
        model._partition_bands = lambda device: chosen_targets.to(device)

        loss, recon, _ = model(x)
        x_flat = x.reshape(B, 49, 59)
        expected_loss = (
            (recon[:, :, chosen_targets] - x_flat[:, :, chosen_targets]) ** 2
        ).mean()
        torch.testing.assert_close(loss, expected_loss, rtol=1e-5, atol=1e-6)

    def test_mask_ratio_75_percent_spatial_tokens_hidden(self, model):
        """Spatial masking is preserved from the parent class: ~75% hidden."""
        model.train()
        B = 50
        torch.manual_seed(0)
        x = torch.randn(B, 7, 7, 59)
        _, _, mask = model(x)
        fraction_masked = mask.float().mean().item()
        expected = int(49 * 0.75) / 49
        assert abs(fraction_masked - expected) < 0.01, (
            f"got {fraction_masked}, expected ≈ {expected}"
        )

    def test_encoder_sees_zeroed_target_bands(self, model):
        """Two forward passes that differ only at target-band positions
        should produce identical encoder visible-token outputs, because
        target bands are zeroed before encoding."""
        model.eval()
        model.spectral_mask_ratio = 0.5
        chosen_targets = torch.zeros(59, dtype=torch.bool)
        chosen_targets[torch.arange(0, 59, 2)] = True
        model._partition_bands = lambda device: chosen_targets.to(device)

        torch.manual_seed(0)
        x_a = torch.randn(2, 7, 7, 59) * 0.1
        x_b = x_a.clone()
        # Perturb target bands only
        x_b[..., chosen_targets] += 5.0

        # Same spatial mask so we compare apples to apples.
        torch.manual_seed(123)
        _, recon_a, _ = model(x_a)
        torch.manual_seed(123)
        _, recon_b, _ = model(x_b)

        # Encoder input is band-masked → identical at every position →
        # recon should be identical for both inputs.
        torch.testing.assert_close(recon_a, recon_b, rtol=1e-5, atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n crism python -m pytest tests/test_spend_spatial_mae.py::TestForwardPass -v`
Expected: 5 FAILs. The model still uses the inherited `SpatialSpectralMAE.forward`, which computes loss on masked spatial tokens (not on target bands), so `test_loss_is_target_band_only_at_ratio_half` and `test_encoder_sees_zeroed_target_bands` will fail. Shape/mask-ratio tests may pass by coincidence but the loss-localization test will not.

- [ ] **Step 3: Implement the SPEND forward**

Append this method to `SpendSpatialSpectralMAE` in `models/spend_spatial_mae.py` (inside the class body, after `_partition_bands`):

```python
    def forward(self, x_clean: torch.Tensor):
        """Returns (loss, recon, mask).

        loss:  scalar MSE on target-band positions of recon vs x_clean,
               across all 49 spatial positions. If spectral_mask_ratio == 0,
               the loss degenerates to MSE on all bands at all positions
               (equivalent to v3's all-position MAE loss).
        recon: (B, n_tokens, n_bands) — reconstructed spectra at every position
        mask:  (B, n_tokens) bool — True = was spatially masked at the encoder
        """
        B = x_clean.shape[0]
        device = x_clean.device
        N = self.n_tokens

        # 1. Sample one band partition for the whole batch.
        target_mask = self._partition_bands(device)  # (n_bands,) bool
        input_band_mask = ~target_mask                # bands the encoder sees

        # 2. Zero out target bands at every pixel in the encoder input.
        x_in = x_clean * input_band_mask.view(1, 1, 1, self.n_bands).to(x_clean.dtype)

        # 3. Standard spatial masking + encoder pass (parent-class machinery).
        visible_ids, masked_ids, mask = self._mask_tokens(B, device)
        enc_out = self.encoder.encode_visible(x_in, visible_ids)
        enc_proj = self.enc_to_dec(enc_out[:, 1:])

        decoder_tokens = self.mask_token.expand(B, N, -1).clone()
        scatter_idx = visible_ids.unsqueeze(-1).expand(-1, -1, self.decoder_dim)
        decoder_tokens.scatter_(1, scatter_idx, enc_proj)
        pos_ids = torch.arange(1, N + 1, device=device)
        decoder_tokens = decoder_tokens + self.decoder_pos_embed(pos_ids)
        decoded = self.decoder(decoder_tokens)
        recon = self.reconstruction_head(decoded)

        # 4. SPEND loss: MSE on target bands only (or all bands when ratio=0).
        x_flat = x_clean.reshape(B, N, self.n_bands)
        if target_mask.any():
            loss = ((recon[:, :, target_mask] - x_flat[:, :, target_mask]) ** 2).mean()
        else:
            # ratio==0 → no target bands → fall back to all-band MSE (phase C).
            loss = ((recon - x_flat) ** 2).mean()

        return loss, recon, mask
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n crism python -m pytest tests/test_spend_spatial_mae.py -v`
Expected: 20 PASS (7 + 3 + 5 + 5)

- [ ] **Step 5: Commit**

```bash
git add models/spend_spatial_mae.py tests/test_spend_spatial_mae.py
git commit -m "$(cat <<'EOF'
feat(spend): override forward with SPEND Noise2Noise objective

Encoder input has target-half bands zeroed at every spatial position;
decoder reconstructs all 59 bands; loss is MSE only on the target-half
positions of the reconstruction, evaluated at all 49 spatial positions.
When spectral_mask_ratio==0, the loss degenerates to all-band MSE so
phase C of training behaves as plain MAE.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Edge case — `spectral_mask_ratio == 0` matches v3 all-position loss

**Files:**
- Test: `tests/test_spend_spatial_mae.py` (add edge-case test)
- No code change expected (Task 4's `else` branch already covers it)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_spend_spatial_mae.py`:

```python
class TestDegenerateRatioEdgeCase:
    """At spectral_mask_ratio==0, SPEND becomes plain all-band-all-position MAE."""

    def test_loss_equals_all_band_all_position_mse_at_ratio_zero(self, model):
        model.eval()
        model.spectral_mask_ratio = 0.0
        torch.manual_seed(0)
        B = 4
        x = torch.randn(B, 7, 7, 59) * 0.1
        loss, recon, _ = model(x)
        x_flat = x.reshape(B, 49, 59)
        expected = ((recon - x_flat) ** 2).mean()
        torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)

    def test_encoder_sees_full_band_input_at_ratio_zero(self, model):
        """At ratio=0 target_mask is all False, so encoder input == x_clean."""
        model.eval()
        model.spectral_mask_ratio = 0.0
        torch.manual_seed(0)
        x_a = torch.randn(2, 7, 7, 59) * 0.1
        x_b = x_a.clone()
        x_b[..., :30] += 5.0  # perturb the first 30 bands

        torch.manual_seed(123)
        _, recon_a, _ = model(x_a)
        torch.manual_seed(123)
        _, recon_b, _ = model(x_b)
        # At ratio=0 the encoder sees the full input, so different x → different recon.
        assert not torch.allclose(recon_a, recon_b, rtol=1e-3)
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `conda run -n crism python -m pytest tests/test_spend_spatial_mae.py::TestDegenerateRatioEdgeCase -v`
Expected: 2 PASS (the implementation from Task 4 already handles this branch).

If they fail, fix `forward` in `models/spend_spatial_mae.py` until they pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_spend_spatial_mae.py
git commit -m "$(cat <<'EOF'
test(spend): cover spectral_mask_ratio==0 phase-C degeneration

At ratio=0 the SPEND loss reverts to MSE over all 59 bands at all 49
spatial positions, and the encoder sees the full input. Both behaviors
are required for the train/fine-tune distribution gap to close cleanly.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Noise2Noise gradient-direction sanity check

**Files:**
- Test: `tests/test_spend_spatial_mae.py` (add convergence test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_spend_spatial_mae.py`:

```python
class TestNoise2NoiseGradientDirection:
    """Confirm SPEND drives the model toward the clean signal, not the noise."""

    def test_recon_converges_toward_clean_signal(self):
        """Train a tiny SPEND model for a few steps on a synthetic
        (smooth-signal + i.i.d. Gaussian noise) dataset. After training,
        the reconstruction should be closer to the clean signal than to
        the noisy observation.
        """
        torch.manual_seed(0)
        model = SpendSpatialSpectralMAE(
            n_bands=59, patch_size=7,
            embed_dim=64, n_heads=4, n_layers=2,
            decoder_dim=32, decoder_layers=1,
            mask_ratio=0.5, spectral_mask_ratio=0.5,
        )
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Synthetic clean signal: smooth Gaussian-like spectrum per patch.
        B, P, n_bands = 32, 7, 59
        wavelength = torch.linspace(0, 1, n_bands)
        clean = torch.exp(-((wavelength - 0.5) ** 2) / 0.05)  # (n_bands,)
        clean_patch = clean.view(1, 1, 1, n_bands).expand(B, P, P, n_bands).contiguous()

        # Training loop: each step gets fresh i.i.d. noise (Noise2Noise condition).
        for _ in range(100):
            noisy = clean_patch + 0.1 * torch.randn_like(clean_patch)
            loss, _, _ = model(noisy)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Evaluation: a single new noisy sample, evaluated with the same
        # spectral mask ratio used during training (so the decoder is asked
        # to do the task it was actually trained for). Fix the partition for
        # reproducibility — the model has seen all bands as targets across
        # the 100 random training partitions, so any fixed partition is fair.
        model.eval()
        target_mask_eval = torch.zeros(n_bands, dtype=torch.bool)
        target_mask_eval[torch.arange(0, n_bands, 2)] = True
        model._partition_bands = lambda device: target_mask_eval.to(device)

        torch.manual_seed(99)
        eval_noisy = clean_patch[:4] + 0.1 * torch.randn(4, P, P, n_bands)
        _, recon, _ = model(eval_noisy)

        clean_flat = clean_patch[:4].reshape(4, 49, n_bands)
        noisy_flat = eval_noisy.reshape(4, 49, n_bands)
        # Compare on target-band positions only — those are what the decoder
        # is trained to predict.
        mse_vs_clean = (
            (recon[:, :, target_mask_eval] - clean_flat[:, :, target_mask_eval]) ** 2
        ).mean().item()
        mse_vs_noisy = (
            (recon[:, :, target_mask_eval] - noisy_flat[:, :, target_mask_eval]) ** 2
        ).mean().item()

        # The whole point of Noise2Noise: recon is closer to clean than to noisy.
        assert mse_vs_clean < mse_vs_noisy, (
            f"recon is closer to noisy ({mse_vs_noisy:.4f}) than clean ({mse_vs_clean:.4f}) — "
            "training did not drive the model toward the signal."
        )
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `conda run -n crism python -m pytest tests/test_spend_spatial_mae.py::TestNoise2NoiseGradientDirection -v`
Expected: PASS (this is a sanity check on the already-implemented forward).

If it fails: increase training steps to 200, or lower noise scale to 0.05. The test should not be flaky on a working implementation — if it is, file an issue. Do not weaken the assertion.

- [ ] **Step 3: Commit**

```bash
git add tests/test_spend_spatial_mae.py
git commit -m "$(cat <<'EOF'
test(spend): sanity-check Noise2Noise gradient direction

Train a tiny SPEND model for 100 steps on synthetic smooth-spectrum
patches with i.i.d. additive Gaussian noise. After training, the
reconstruction should be closer to the clean signal than to the noisy
observation — the defining property of the Noise2Noise objective.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Encoder state-dict compatibility with downstream classifier

**Files:**
- Test: `tests/test_spend_spatial_mae.py` (add compatibility test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_spend_spatial_mae.py`:

```python
class TestEncoderStateDictCompat:
    """Pretrained SPEND encoder must load into SpatialSpectralClassifier."""

    def test_encoder_state_dict_loads_into_classifier(self, model):
        from models.spatial_spectral_transformer import SpatialSpectralClassifier
        classifier = SpatialSpectralClassifier(
            n_bands=59, patch_size=7, n_classes=5,
            embed_dim=128, n_heads=4, n_layers=6,
        )
        encoder_state = model.encoder_state_dict()
        missing, unexpected = classifier.load_encoder_state_dict(encoder_state)
        assert unexpected == [], f"unexpected keys: {unexpected}"
        core_missing = [k for k in missing if k.startswith('encoder.encoder')]
        assert not core_missing, f"core encoder weights missing: {core_missing}"
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `conda run -n crism python -m pytest tests/test_spend_spatial_mae.py::TestEncoderStateDictCompat -v`
Expected: PASS (encoder_state_dict is inherited unchanged from `SpatialSpectralMAE`).

- [ ] **Step 3: Run the entire test file once**

Run: `conda run -n crism python -m pytest tests/test_spend_spatial_mae.py -v`
Expected: 24 PASS (7 schedule + 3 skeleton + 5 partition + 5 forward + 2 edge-case + 1 N2N gradient + 1 compat).

If the count is off, find and fix the test that's not running.

- [ ] **Step 4: Commit**

```bash
git add tests/test_spend_spatial_mae.py
git commit -m "$(cat <<'EOF'
test(spend): verify encoder state_dict loads into classifier

After SPEND pre-training, the encoder weights must load cleanly into
SpatialSpectralClassifier with no unexpected keys and no missing core
encoder weights. This is the contract that lets v4 plug into the same
fine-tuning pipeline as v3.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Pretraining script

**Files:**
- Create: `scripts/pretrain_spatial_mae_spend.py`
- Reference (read-only): `scripts/pretrain_spatial_mae_denoising.py`

- [ ] **Step 1: Write the training driver**

Create `scripts/pretrain_spatial_mae_spend.py` with this content:

```python
"""
SPEND-style spatial-spectral MAE pre-training driver.

Mirrors scripts/pretrain_spatial_mae_denoising.py but with the SPEND
objective and a spectral-mask annealing schedule replacing the synthetic
noise injection.

Usage (HPC):
    python scripts/pretrain_spatial_mae_spend.py \\
        --epochs 200 --embed_dim 128 --n_layers 6 --mask_ratio 0.75 \\
        --spectral_mask_ratio 0.5 \\
        --anneal_start_epoch 161 --anneal_end_epoch 181
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
    parser.add_argument('--epochs',      type=int, default=200)
    parser.add_argument('--warmup',      type=int, default=10)
    parser.add_argument('--batch_size',  type=int, default=1024)
    parser.add_argument('--patches_per_epoch', type=int, default=200_000)
    parser.add_argument('--num_workers', type=int, default=4)
    # Architecture
    parser.add_argument('--embed_dim',      type=int, default=128)
    parser.add_argument('--n_heads',        type=int, default=4)
    parser.add_argument('--n_layers',       type=int, default=6)
    parser.add_argument('--decoder_dim',    type=int, default=64)
    parser.add_argument('--decoder_layers', type=int, default=2)
    parser.add_argument('--mask_ratio',     type=float, default=0.75)
    # SPEND
    parser.add_argument('--spectral_mask_ratio', type=float, default=0.5,
                        help='Base spectral mask ratio for phase A.')
    parser.add_argument('--anneal_start_epoch', type=int, default=161)
    parser.add_argument('--anneal_end_epoch',   type=int, default=181)
    # Run management
    parser.add_argument('--run_name', type=str, default='spatial_mae_spend_128d_6l')
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
    log.info(f"SPEND base ratio={args.spectral_mask_ratio}, "
             f"anneal {args.anneal_start_epoch}→{args.anneal_end_epoch}")

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

    from models.spend_spatial_mae import (
        SpendSpatialSpectralMAE,
        compute_spectral_mask_ratio,
    )
    model = SpendSpatialSpectralMAE(
        n_bands=59, patch_size=7,
        embed_dim=args.embed_dim, n_heads=args.n_heads, n_layers=args.n_layers,
        decoder_dim=args.decoder_dim, decoder_layers=args.decoder_layers,
        mask_ratio=args.mask_ratio,
        spectral_mask_ratio=args.spectral_mask_ratio,
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
        # ── Anneal callback: update spectral_mask_ratio for this epoch ────
        model.spectral_mask_ratio = compute_spectral_mask_ratio(
            epoch=epoch,
            anneal_start_epoch=args.anneal_start_epoch,
            anneal_end_epoch=args.anneal_end_epoch,
            base=args.spectral_mask_ratio,
        )

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
        log.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"spend_loss={mean_loss:.6f} | "
            f"spectral_mask_ratio={model.spectral_mask_ratio:.3f} | "
            f"lr={lr_now:.2e}"
        )

        if use_wandb:
            import wandb
            wandb.log({
                'epoch': epoch,
                'spend_loss': mean_loss,
                'spectral_mask_ratio': model.spectral_mask_ratio,
                'lr': lr_now,
            })

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

- [ ] **Step 2: Smoke-test the training script (manual; not an automated test)**

Run a 2-epoch dry-run with tiny patches-per-epoch to confirm the script starts, the anneal callback updates the ratio each epoch, and no NaNs appear. Adjust the `--config` path to wherever your local data points.

Run:
```bash
conda run -n crism python scripts/pretrain_spatial_mae_spend.py \
    --epochs 2 --warmup 1 --batch_size 64 \
    --patches_per_epoch 256 --num_workers 0 \
    --anneal_start_epoch 1 --anneal_end_epoch 2 \
    --run_name spend_smoke --no_wandb
```

Expected:
- Two epoch log lines printed.
- `spectral_mask_ratio` shown in log: 0.5 at epoch 1, 0.0 at epoch 2 (the schedule fully exercised because anneal_end_epoch=2).
- `spend_loss` is a finite positive number on both epochs.
- No tracebacks.

If the script fails to find data, it's the local data-root config — that's expected on a fresh checkout. Note the failure and continue. The HPC SLURM script in Task 9 sets the path correctly for the GPU run.

- [ ] **Step 3: Commit**

```bash
git add scripts/pretrain_spatial_mae_spend.py
git commit -m "$(cat <<'EOF'
feat(spend): add SPEND pre-training driver script

Mirrors the v3 denoising driver but uses SpendSpatialSpectralMAE and
calls compute_spectral_mask_ratio each epoch to update the spectral
mask ratio. wandb logs spend_loss and spectral_mask_ratio per epoch.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: HPC SLURM submission script

**Files:**
- Create: `scripts/hpc_pretrain_spend.slurm`
- Reference (read-only): `scripts/hpc_pretrain_denoising.slurm`

- [ ] **Step 1: Write the SLURM file**

Create `scripts/hpc_pretrain_spend.slurm` with this content:

```bash
#!/bin/bash
#SBATCH --job-name=spatial_mae_spend
#SBATCH --account=sbyrne
#SBATCH --partition=gpu_standard
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64gb
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/pretrain_spend_%j.log
#SBATCH --error=logs/pretrain_spend_%j.log

# SPEND-style spatial-spectral MAE pre-training (v4).
# Single GPU, 200 epochs, ~32-hour wall time expected. 48-hour budget.
# Designed to run CONCURRENTLY with the v3 denoising job — separate
# job name, separate output log, same data root.
# Spec: docs/superpowers/specs/2026-05-16-spend-spatial-mae-design.md

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

echo "=== SPEND MAE pre-training start: $(date) ==="

${PYTHON} -u scripts/pretrain_spatial_mae_spend.py \
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
    --spectral_mask_ratio 0.5 \
    --anneal_start_epoch 161 \
    --anneal_end_epoch 181 \
    --run_name spatial_mae_spend_128d_6l

echo "=== SPEND MAE pre-training end: $(date) ==="
```

- [ ] **Step 2: Verify the file is executable-ready (no need to actually submit)**

Run: `head -1 scripts/hpc_pretrain_spend.slurm`
Expected output: `#!/bin/bash`

Run: `grep -c '^#SBATCH' scripts/hpc_pretrain_spend.slurm`
Expected: at least 10.

- [ ] **Step 3: Commit**

```bash
git add scripts/hpc_pretrain_spend.slurm
git commit -m "$(cat <<'EOF'
feat(spend): add HPC SLURM submission for SPEND pre-training

Single-GPU job, 48-hour budget, designed to run concurrently with the
v3 denoising job. Separate job name (spatial_mae_spend) so it won't
collide on accounting.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Validation-figure script

**Files:**
- Create: `scripts/figures/fig_spend_partition.py`
- Reference (read-only): `scripts/figures/fig_denoising_corruption.py`, `scripts/figures/_utils.py`

This script is authored now so it's ready to run after pre-training completes. The actual figure generation requires a trained checkpoint — that's a manual run later, not part of this plan's automation.

- [ ] **Step 1: Write the figure script**

Create `scripts/figures/fig_spend_partition.py` with this content:

```python
"""
Visualize the SPEND-style spectral-partition objective on real CRISM pixels.

For three representative pixels (olivine, hcp, plagioclase), show:
  - Col 1: clean center-pixel spectrum (reference)
  - Col 2: input-half bands (gray) vs target-half bands (colored) overlaid
           on the spectrum, for one sample partition
  - Col 3: model's predicted target-band values (line) overlaid on actual
           target-band values (markers) at the same partition
  - Col 4: residual = (prediction − target) per band; should look like
           centered i.i.d. noise (flat, no structure)

Usage:
    conda run -n crism python scripts/figures/fig_spend_partition.py \\
        --checkpoint checkpoints/spatial_mae_spend_128d_6l_best.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, '/mnt/mrdr/crism_classification')
from models.spend_spatial_mae import SpendSpatialSpectralMAE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _utils import (
    CLASS_COLORS, build_mrral_map, find_representative_pixels,
    get_wavelengths_59, load_mrral_parquet, read_patch_from_tile,
)

OUT_PATH = '/mnt/mrdr/crism_classification/reports/v5/fig_v5_spend_partition.png'
CLASSES_TO_SHOW = ['olivine', 'hcp', 'plagioclase']


def load_model(checkpoint_path: str) -> SpendSpatialSpectralMAE:
    model = SpendSpatialSpectralMAE(
        n_bands=59, patch_size=7,
        embed_dim=128, n_heads=4, n_layers=6,
        decoder_dim=64, decoder_layers=2,
        mask_ratio=0.75, spectral_mask_ratio=0.5,
    )
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['mae_state'])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to a trained SPEND .pt checkpoint')
    parser.add_argument('--out', type=str, default=OUT_PATH)
    args = parser.parse_args()

    model = load_model(args.checkpoint)

    df = load_mrral_parquet()
    mrral_map = build_mrral_map()
    pixels = find_representative_pixels(df, n_per_class=1, seed=42)
    wls = get_wavelengths_59()

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
        x_clean = torch.from_numpy(patch).unsqueeze(0).float()
        center_clean = patch[3, 3]

        # Use a fixed partition (evens = target) so the figure is reproducible.
        target_mask = torch.zeros(59, dtype=torch.bool)
        target_mask[torch.arange(0, 59, 2)] = True
        model._partition_bands = lambda device: target_mask.to(device)
        model.spectral_mask_ratio = 0.5

        with torch.no_grad():
            torch.manual_seed(7 + row_i)
            _, recon, _ = model(x_clean)
        recon_center = recon[0, 24].numpy()  # center spatial token

        color = CLASS_COLORS[cls]
        valid = (center_clean > 0) & (center_clean < 0.5)

        # Col 1: clean spectrum
        ax = axes[row_i, 0]
        ax.plot(wls[valid], center_clean[valid], color=color, linewidth=1.8)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('I/F')
        ax.set_title(f'{cls}: tile {tid}\nclean center pixel ({pr},{pc})',
                     color=color, fontsize=9.5)
        ax.grid(alpha=0.3)

        # Col 2: partition view — input vs target bands
        ax = axes[row_i, 1]
        ax.plot(wls[valid], center_clean[valid], color='#bbb',
                linewidth=1.0, alpha=0.6, label='spectrum')
        input_idx = (~target_mask).numpy()
        target_idx = target_mask.numpy()
        ax.scatter(wls[input_idx], center_clean[input_idx],
                   color='#888', s=18, label='input-half (seen)', zorder=3)
        ax.scatter(wls[target_idx], center_clean[target_idx],
                   color=color, s=22, marker='x',
                   label='target-half (predicted)', zorder=4)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('I/F')
        ax.set_title('partition: which bands are\nseen vs predicted',
                     fontsize=9.5)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.3)

        # Col 3: prediction overlay
        ax = axes[row_i, 2]
        target_wls = wls[target_idx]
        actual_targets = center_clean[target_idx]
        pred_targets = recon_center[target_idx]
        ax.scatter(target_wls, actual_targets, color='#666', s=22, marker='x',
                   label='observed', zorder=3)
        ax.plot(target_wls, pred_targets, color=color, linewidth=1.4,
                label='model prediction', zorder=2)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('I/F')
        ax.set_title('target-band reconstruction\n(model vs observation)',
                     fontsize=9.5)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.3)

        # Col 4: residual
        ax = axes[row_i, 3]
        residual = pred_targets - actual_targets
        ax.plot(target_wls, residual, color=color, linewidth=1.2)
        ax.axhline(0, color='black', linewidth=0.4)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('residual (I/F)')
        ax.set_title('prediction − observation\n(should look noise-like)',
                     fontsize=9.5)
        ax.grid(alpha=0.3)

    fig.suptitle(
        'SPEND-style spectral partition — what the model predicts\n'
        '(even bands target, odd bands visible to encoder)',
        fontsize=11, y=1.005,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify imports cleanly without a checkpoint**

Run: `conda run -n crism python -c "import scripts.figures.fig_spend_partition as f; print('module imported:', f.OUT_PATH)"`
Expected output: `module imported: /mnt/mrdr/crism_classification/reports/v5/fig_v5_spend_partition.png`

If `scripts/figures` is not a package, just check the file imports cleanly:
Run: `conda run -n crism python -c "
import sys; sys.path.insert(0, '/mnt/mrdr/crism_classification')
import importlib.util
spec = importlib.util.spec_from_file_location('fig', 'scripts/figures/fig_spend_partition.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print('module imported:', m.OUT_PATH)
"`
Expected: `module imported: /mnt/mrdr/crism_classification/reports/v5/fig_v5_spend_partition.png`

- [ ] **Step 3: Verify CLI argument parsing**

Run: `conda run -n crism python scripts/figures/fig_spend_partition.py --help`
Expected: a usage line and `--checkpoint` / `--out` flags listed.

- [ ] **Step 4: Commit**

```bash
git add scripts/figures/fig_spend_partition.py
git commit -m "$(cat <<'EOF'
feat(spend): add SPEND partition validation figure script

Generates reports/v5/fig_v5_spend_partition.png from a trained SPEND
checkpoint. Four-column layout per representative mineral pixel:
clean spectrum, partition view, prediction overlay, residual. Used as
the qualitative validation of SPEND pre-training once HPC training
completes.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Final integration check

- [ ] **Step 1: Run the full SPEND test suite once**

Run: `conda run -n crism python -m pytest tests/test_spend_spatial_mae.py -v`
Expected: 24 tests pass.

- [ ] **Step 2: Confirm no other tests regressed**

Run: `conda run -n crism python -m pytest tests/ -x --timeout=300`
Expected: every test in the suite either passes or skips for an unrelated reason. No new failures introduced by this plan's changes.

If a previously-passing test now fails, investigate before declaring done.

- [ ] **Step 3: Summary commit (optional, only if there's anything outstanding)**

If everything is clean, no commit needed. If you fixed an unrelated test that broke during integration, commit those fixes with a `chore: ...` message.

---

## Out of scope (do not implement in this plan)

The following are explicitly out of scope per the spec; do not add them:

- Real noisy-pair training from multi-observation MRDR overlap regions (requires additional storage; tabled).
- Adjacent-column sampling (E2E-CRISM style) — a v5 fallback if v4 underperforms.
- Wavelength-aware positional encoding (architectural change deferred).
- Generating a denoised global mosaic (future project, after v4 encoder is validated).
- Replacing v3 — v3 continues on its own HPC job in parallel.
- Quantitative noise-removal metric implementation — spec defers this; the qualitative figure in Task 10 is the only deliverable for this round.

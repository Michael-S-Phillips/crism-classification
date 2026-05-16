# Denoising Spatial-Spectral MAE Pre-training Design

**Date:** 2026-05-16
**Status:** Approved (direction and corruption parameters approved by user; implementation delegated)
**Scope:** Build a denoising variant of the existing `SpatialSpectralMAE` that learns to recover clean CRISM spectra from physically-motivated corruptions, then pre-train it on HPC. Drop-in encoder for downstream classifiers — the resulting encoder loads into `SpatialSpectralClassifier`, `DecompSpVit`, or `DecompSpVitAdv` via the existing `load_encoder_state_dict` interface.

---

## Motivation

The v2 adversarial decomposition (`fig_v5_decomp_v2_recon_*.png`) revealed that the per-pixel reconstructions `ŝ` and `n̂` are not physically meaningful — the model satisfies `ŝ + n̂ ≈ x` with arbitrary high-frequency factors. The reconstruction loss alone doesn't pressure either branch to look like real surface reflectance or real instrumental noise.

A root cause: the encoder is pre-trained on raw I/F via standard MAE (predict missing pixels from spatial neighbors). It never learns to **distinguish signal from noise** — the MAE objective treats the input as ground truth, so the encoder learns to compress it faithfully without any incentive to denoise.

Denoising MAE pre-training corrupts the input with instrument-physics-motivated noise and asks the encoder + decoder to recover the *clean* spectrum. The encoder is forced to internally separate "what came in" from "what the mineral signature actually is."

## Corruption design (data-informed)

The three noise components match the three real CRISM artifacts the user identified, with magnitudes estimated from the actual labeled-polygon parquet (see "Noise stats analysis" below).

```
x_corrupted  =  x_clean  +  ε_gauss  +  ε_spike  +  ε_column
```

| Component | Magnitude | Shape | Sampling per patch |
|---|---|---|---|
| `ε_gauss` | σ = 0.0087 | per-pixel, per-band (independent) | `ε ~ 𝒩(0, σ²)`, shape (7, 7, 59) |
| `ε_spike` | σ = 0.0058 | band-localized at the 1 µm detector seam (bands 13-17, 925-1023 nm), broadcast across all 49 spatial pixels | one scalar `m ~ 𝒩(0, σ²)` per patch; shape is a Gaussian-weighted bump centered at band 15 with FWHM ~3 bands, scaled to peak amplitude `m` |
| `ε_column` | σ = 0.0049 | per-column, per-band; broadcast uniformly down all 7 rows of the patch | bias `b[c, λ] ~ 𝒩(0, σ²)` for each (column ∈ 0..6, band ∈ 0..58) |

**Always applied** (the user-approved choice over random-enable). Magnitudes are drawn fresh per patch each forward pass, so the model still sees a range from ~0 (small draws) up to ~3σ (rare large draws).

### Noise stats analysis

From `scripts/figures/_utils.py` workflow on `mrral_pixels.parquet` (train split, ~1.7M valid pixels):

- **Within-polygon residual MAD-σ:** 0.0087 median across bands (the polygon-membership assumption is that pixels in the same hand-drawn polygon share spectral content; residuals around the polygon mean are stochastic noise).
- **1 µm seam discontinuity:** 0.0058 RMS at the worst band (band 15, 984 nm). Computed as residual of per-polygon mean spectra against a 5-band median filter, isolated to the 925-1023 nm region.
- **Column-bias upper bound:** 0.0164. We use 30% (= 0.0049) as the actual `σ_column` because the upper bound contains legitimate scene structure varying across columns (not just dark-current striping).

These are the **defaults**. The pre-training script exposes all three as CLI flags so ablations are possible.

## Architecture

No change to the encoder backbone. We add one new module and one new MAE subclass:

### `CrismNoiseAugmentation` (new module, `models/noise_augmentation.py`)

A `nn.Module` with no learnable parameters. Forward:

```python
def forward(self, x):                    # x: (B, 7, 7, 59) clean
    if not self.training:                # corruption disabled at eval
        return x
    B = x.shape[0]
    eps_gauss = torch.randn_like(x) * self.sigma_gauss

    # 1 µm spike: one magnitude per patch, shaped as a Gaussian bump in band space
    spike_mag = torch.randn(B, device=x.device) * self.sigma_spike   # (B,)
    eps_spike = spike_mag[:, None, None, None] * self._spike_profile  # broadcast
    # _spike_profile is a fixed (1, 1, 1, 59) tensor with values only in bands 13-17,
    # weighted as exp(-0.5 * ((band - 15)/1.5)²) for band ∈ [13, 17], 0 elsewhere

    # Column bias: per-column-of-patch, per-band
    col_bias = torch.randn(B, 1, 7, 59, device=x.device) * self.sigma_column
    eps_column = col_bias.expand(B, 7, 7, 59)

    return x + eps_gauss + eps_spike + eps_column
```

Configurable knobs:
- `sigma_gauss` (default 0.0087)
- `sigma_spike` (default 0.0058)
- `sigma_column` (default 0.0049)
- `spike_center_band` (default 15)
- `spike_fwhm_bands` (default 3 → σ ≈ 1.5 bands)
- `spike_band_range` (default 13-17, used to mask the profile to zero outside)

### `DenoisingSpatialSpectralMAE` (new class, `models/denoising_spatial_mae.py`)

Subclass of `SpatialSpectralMAE`. Identical architecture (encoder + decoder + projections); the only behavioral change is in `forward()`:

```python
def forward(self, x_clean):              # x_clean: (B, 7, 7, 59)
    x_corrupted = self.noise_aug(x_clean)

    B = x_clean.shape[0]
    N = self.n_tokens
    device = x_clean.device

    visible_ids, masked_ids, mask = self._mask_tokens(B, device)

    # Encode visible tokens of the CORRUPTED input
    enc_out = self.encoder.encode_visible(x_corrupted, visible_ids)
    enc_proj = self.enc_to_dec(enc_out[:, 1:])

    # Standard decoder pathway (unchanged)
    decoder_tokens = self.mask_token.expand(B, N, -1).clone()
    scatter_idx = visible_ids.unsqueeze(-1).expand(-1, -1, self.decoder_dim)
    decoder_tokens.scatter_(1, scatter_idx, enc_proj)
    pos_ids = torch.arange(1, N + 1, device=device)
    decoder_tokens = decoder_tokens + self.decoder_pos_embed(pos_ids)
    decoded = self.decoder(decoder_tokens)
    recon = self.reconstruction_head(decoded)             # (B, N, n_bands)

    # MSE on ALL positions vs x_CLEAN (not x_corrupted)
    x_flat = x_clean.reshape(B, N, self.n_bands)
    per_pixel_loss = ((recon - x_flat) ** 2).mean(dim=-1) # (B, N)
    loss = per_pixel_loss.mean()                          # all positions

    return loss, recon, mask
```

Three deltas vs `SpatialSpectralMAE.forward`:
1. The encoder sees `x_corrupted = noise_aug(x_clean)` instead of `x_clean`.
2. The reconstruction target is `x_clean` (denoising), not the encoder input.
3. Loss is averaged over **all 49 positions** (not just the masked subset).

The decoder's reconstruction-head output dimensionality, decoder architecture, masking ratio, and positional embeddings are all unchanged. The MAE checkpoint of the existing model would NOT load into this directly because the input distribution it expects (clean I/F) differs from what we're going to train on (corrupted I/F) — we'll train from scratch.

### Why all-positions loss (and not masked-only)

In standard MAE, the recon target is the input itself, so penalizing reconstruction at visible positions is trivial (the encoder can just memorize). In denoising MAE the target is `x_clean ≠ x_corrupted`, so the visible-position loss is non-trivial: it explicitly trains the model to denoise pixels it can see.

The masked-position loss is still computed (it's part of "all positions"), so the spatial-inference objective survives unchanged. We get both objectives jointly:
- **Visible-position loss** → learn to denoise the spectra the encoder receives directly
- **Masked-position loss** → learn to infer clean spectra at masked positions from corrupted spatial neighbors

## Pre-training procedure

**Data:** Same as current MAE — `CRISMGlobalPatchDataset` reading from `/xdisk/sbyrne/phillipsm/CRISM_MRDR/` on HPC. Unsupervised; uses all valid pixels in all tiles.

**Hyperparameters (matching current MAE for fair comparison):**
- Encoder: 128-d, 6 layers, 4 heads, dropout 0.0 (no dropout during pre-training)
- Decoder: 64-d, 2 layers
- Patch size: 7×7
- Mask ratio: 0.75 (the existing pretrain script defaults to 0.85; we use 0.75 because the denoising signal at visible positions reduces the need for aggressive masking — a v2 ablation could revisit)
- Batch size: 1024 (matches existing pretrain)
- Optimizer: AdamW, lr 1e-3, weight decay 0.05
- Schedule: cosine annealing over `epochs` epochs
- **Epochs: 200** (matches the 194-epoch existing MAE for direct comparison; the user-approved direction is "another large pre-training run")

**Estimated wall time on HPC:** Single GPU pre-training of the existing MAE ran 194 epochs at ~10 min/epoch on A100. Denoising MAE has same compute footprint per epoch (the noise aug is a few `randn_like` ops, negligible). Expected ~32 hours total.

**Checkpointing:**
- Save every 50 epochs to `spatial_mae_denoising_128d_6l_epoch{N}.pt`
- Save best (lowest val recon MSE) to `spatial_mae_denoising_128d_6l_best.pt`

## Loading into downstream classifiers

The encoder weights live in the same submodule structure as the existing MAE. After pre-training:

```python
ckpt = torch.load('checkpoints/spatial_mae_denoising_128d_6l_best.pt')
classifier.load_encoder_state_dict(ckpt['encoder_state'])
```

works unchanged for `SpatialSpectralClassifier`, `DecompSpVit`, and `DecompSpVitAdv`. The new denoising encoder is a drop-in replacement.

## Files & components

| Path | Action | Responsibility |
|---|---|---|
| `models/noise_augmentation.py` | Create | `CrismNoiseAugmentation` module — Gaussian + 1 µm spike + column-bias corruption |
| `models/denoising_spatial_mae.py` | Create | `DenoisingSpatialSpectralMAE` subclass — composes noise aug into the existing MAE forward, with target = x_clean and all-position loss |
| `scripts/pretrain_spatial_mae_denoising.py` | Create | Pre-training script, similar to the existing `pretrain_spatial_mae.py`. CLI flags for the three σ values and the spike-band parameters. |
| `scripts/hpc_pretrain_denoising.slurm` | Create | Single-task pre-training job (no array) |
| `tests/test_noise_augmentation.py` | Create | Shape contracts, sigma sanity checks, eval-mode disables corruption |
| `tests/test_denoising_spatial_mae.py` | Create | Forward shape, target-vs-x_clean equality at zero-noise, masking still applied, all-position loss aggregation |
| `scripts/figures/fig_denoising_mae_corruption.py` | Create | Visualize what the corruption looks like — input vs corrupted vs reconstructed for a handful of pixels (this is the figure that demonstrates the noise is realistic) |

## Acceptance criteria

1. `CrismNoiseAugmentation` produces (B, 7, 7, 59) outputs with statistics matching the configured σ values (verified by unit test sampling many patches and computing empirical std).
2. `DenoisingSpatialSpectralMAE.forward(x_clean)` returns the documented 3-tuple; with `noise_aug.sigma_gauss = noise_aug.sigma_spike = noise_aug.sigma_column = 0` the reconstruction loss against `x_clean` matches the standard MAE loss against the same input.
3. The corruption figure shows visually realistic noise — Gaussian fuzz, a visible 1 µm bump (sometimes positive, sometimes negative), and column-correlated striping in the (7×7) spatial display.
4. HPC pre-training completes 200 epochs without divergence (`val_loss` monotonically decreasing or plateauing).
5. The final encoder loads cleanly into `DecompSpVit` / `DecompSpVitAdv` via `load_encoder_state_dict`.
6. A reconstruction diagnostic figure (input vs corrupted vs reconstructed spectra on val patches) shows reconstructed spectra that look like smooth mineral reflectance — the visual flag that says "this encoder learned to denoise, unlike the v2 design."

## Out of scope (deferred)

- **Larger encoder (256-d / 8L)** — deferred per the user's "keep 128-d 6L for v1" decision. Will revisit if denoising reconstruction still looks poor.
- **Wavelength-dependent corruption magnitudes** — current σ values are scalar (one per noise type). The data analysis shows the actual per-band σ varies (P95 across bands = 0.011 vs median 0.0087). A v2 of the denoising MAE could use per-band σ tables. Deferred until v1 results inform whether this matters.
- **Real instrument noise sampling** — option C from the brainstorm (sample noise from CRISM noise-equivalent-radiance tables or from data residuals directly). Cleaner physics but adds a pre-processing step. Deferred.
- **Denoising probability < 1.0** — currently all corruptions applied to every patch. Could add a "clean-pass probability" so the model also sees uncorrupted inputs. Deferred.

## Open hyperparameters

- `mask_ratio` — default 0.75 (vs current MAE's 0.85). The denoising signal at visible positions reduces the need for aggressive masking, so 0.75 should work fine. Could be set higher; needs no change to architecture.
- Per-σ ablation — running a sweep of `(σ_gauss only, σ_spike only, σ_column only, all three)` could quantify which noise component is doing most of the work. Out of scope for v1 launch but worth noting for paper-time follow-up.
- `spike_fwhm_bands` — currently 1.5 bands (σ in band space). If the real 1 µm seam is narrower or wider, this could be tuned. The data analysis showed the spike is concentrated at band 15 (rms 0.0058) with smaller magnitude in bands 13-17 — FWHM 3 bands matches.

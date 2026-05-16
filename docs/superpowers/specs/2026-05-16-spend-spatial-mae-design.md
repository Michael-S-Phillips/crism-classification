# SPEND-style Spatial-Spectral MAE (v4) — Design Spec

**Date:** 2026-05-16
**Status:** Design (not yet implemented)
**Related:** [Denoising MAE (v3) spec](2026-05-16-denoising-mae-design.md), [Methodology Log v5](../../../../wiki/Methodology Log v5.md) section 14

## 1. Goal

Build a self-supervised pre-trained encoder for CRISM mrral patches that learns to denoise *without* assuming any synthetic noise model. The encoder shares architecture with the existing `SpatialSpectralTransformer` (128d, 6L, 4H ViT, 7×7 patches, 59 mrral bands). Pretrain it with a Noise2Noise objective derived from spectral-frame partitioning (SPEND-style; Wu et al., *Newton* 2025). Output is a general-purpose embedding model:

- **Primary use:** fine-tune the downstream `SpatialSpectralClassifier` head for primary-rock-forming mineral mapping.
- **Secondary uses (future):** produce a denoised global mrral mosaic; run unsupervised analyses (clustering, similarity search) on the encoder embeddings.

This is v4 in the pre-training-objective series:
- v1: multiplicative decomposition (`DecompSpVit`) — collapsed to no-op.
- v2: adversarial signal/noise decomposition (`DecompSpVitAdv`) — discriminator fooled but reconstructions uninterpretable.
- v3: denoising MAE with synthetic CRISM-physics noise injection (`DenoisingSpatialSpectralMAE`) — currently pre-training on HPC; noise model is a load-bearing assumption.
- **v4 (this spec):** Noise2Noise via spectral-frame partitioning. No synthetic noise. Adjacent CRISM bands provide independent-noise pairs of the same signal.

## 2. Background and rationale

### Why SPEND fits CRISM mrral

The Noise2Noise theorem (Lehtinen et al., 2018) states that training a regression network with the loss `MSE(f(x), y)` where `x` and `y` are independent noisy observations of the same underlying clean target converges (in expectation) to the same optimum as training against the clean target. The only requirement is that the noise in `y` be conditionally zero-mean and independent of the noise in `x`.

SPEND (Wu et al., 2025) realizes this for hyperspectral imaging by exploiting *spectral redundancy*: adjacent spectral bands at a single spatial position image essentially the same physical signal but the detector noise on each band is realized independently (each band corresponds to a different read of the detector). Partitioning the bands into two interleaved views yields a Noise2Noise pair from a *single* observation.

For CRISM mrral specifically:
- 59 bands span 410–2457 nm with a median spacing of 32.7 nm.
- Mineral absorption features (olivine 1050 nm, pyroxene 1000/2000 nm, etc.) span 100–300 nm — wider than the band spacing, so adjacent-band signals are highly correlated.
- CRISM detector noise per-band is largely independent (different detector reads, different column biases, different smile/keystone realizations).

So the SPEND assumption holds: predicting one band-half from the other requires learning the underlying smooth spectrum, which is the denoising goal.

### Why not stick with v3

v3 injects synthetic Gaussian + 1 µm spike + per-column-bias noise at data-informed σ values. If the true CRISM noise distribution diverges from this synthetic model — different tails, structured artifacts we haven't characterized, residual atmospheric correlations — the encoder learns to denoise *the synthetic distribution* rather than the real one. SPEND removes this assumption entirely: the supervisory signal is the data's own structure.

v3 and v4 are complementary: v3 runs to completion on HPC alongside this work, giving us a comparison point and the calibrated-σ baseline. v4 is the parallel research arm.

## 3. Architecture

### Encoder (unchanged)

`SpatialSpectralTransformer` at `models/spatial_spectral_transformer.py`:
- `n_bands=59, patch_size=7, embed_dim=128, n_heads=4, n_layers=6, dropout=0.1`
- `band_embed = nn.Linear(59, 128)` projects each pixel's full 59-band spectrum to embed_dim.
- Learned positional embeddings, CLS token, standard pre-norm transformer.
- Two forward paths: `forward(x)` (all positions visible) and `encode_visible(x, visible_ids)` (MAE pretraining).

**No architectural changes.** SPEND is expressed entirely by manipulating the input tensor: bands not in the current view are zeroed at every spatial position. The `band_embed` linear layer learns to handle sparse (mostly-zero) input columns naturally. (We are not introducing a learned per-band mask token; the band-zeroing is sufficient because each band index has a fixed column in `band_embed` and the linear projection can learn that a zero in column `k` means "band `k` is masked".)

### MAE wrapper (new subclass)

`SpendSpatialSpectralMAE(SpatialSpectralMAE)` at `models/spend_spatial_mae.py`. Inherits all spatial-masking, decoder, and pos-embed machinery from `SpatialSpectralMAE`. Overrides `forward` to:
1. Sample a per-batch random band partition (`m_band: bool[59]`, ~30 True).
2. Zero out non-input bands in `x_clean`.
3. Apply standard spatial masking (75% hidden) on the band-masked input.
4. Decode to per-position 59-band reconstruction.
5. Compute MSE loss only on target-band positions (`~m_band`), across all 49 spatial positions.

An attribute `spectral_mask_ratio: float` (default 0.5) controls the fraction of bands zeroed; the training script mutates this attribute over epochs to implement the anneal schedule.

### Annealing schedule

Three phases over 200 epochs:
- **Phase A (epochs 1–160):** `spectral_mask_ratio = 0.5`. Full SPEND.
- **Phase B (epochs 161–180):** linear interpolation from `0.5` → `0.0`.
- **Phase C (epochs 181–200):** `spectral_mask_ratio = 0.0`. Pure spatial MAE (all 59 bands visible at unmasked positions; loss is MAE reconstruction across all bands at all positions).

By the end of training, the encoder has seen full-band input and the band-zeroing trick is no longer in play — closing the train/fine-tune distribution gap.

When `spectral_mask_ratio == 0`, the SPEND loss formulation degenerates to standard MAE recon loss (target bands = all bands; loss is MSE over all bands at all positions). This degeneration is intentional and means no code branching is needed in the loss for phase C.

## 4. Data flow per training step

```
x_clean: (B, 7, 7, 59) — patch from CRISMGlobalPatchDataset

# 1. Random band partition
ratio = self.spectral_mask_ratio           # set by training-loop callback
n_target = round(59 * ratio)
target_idx = torch.randperm(59)[:n_target] # shape: (n_target,)
m_band = ones(59); m_band[target_idx] = 0  # shape: (59,)

# 2. Spatial mask (unchanged from SpatialSpectralMAE)
n_visible = round(49 * (1 - 0.75))         # = 12
visible_ids = torch.randperm(49)[:n_visible]  # shape (B, n_visible) in practice

# 3. Encoder input
x_in = x_clean * m_band.view(1, 1, 1, 59)  # zeros out target bands at every pixel

# 4. Encode visible positions
latents = self.encoder.encode_visible(x_in, visible_ids)
# → (B, n_visible+1, 128)  — CLS + visible spatial tokens

# 5. Decoder (standard MAE decoder, unchanged)
recon = self.decoder(latents, visible_ids)
# → (B, 49, 59)

# 6. Loss — target bands only, all 49 spatial positions
x_flat = x_clean.reshape(B, 49, 59)
target_band_mask = (m_band == 0)           # bool[59], True for target bands
diff = (recon[:, :, target_band_mask]
        - x_flat[:, :, target_band_mask])
loss = (diff ** 2).mean()
return loss, recon, mask
```

**Edge case — `spectral_mask_ratio == 0`:** `n_target = 0`, `target_idx` is empty, `m_band` is all-ones, `target_band_mask` is all-False. The above code would produce a zero-element tensor for `diff` and a NaN/zero loss. The implementation special-cases this: if `n_target == 0`, set `target_band_mask = ones(59, dtype=bool)` so the loss reverts to MSE over all 59 bands at all 49 spatial positions (the v3 / standard MAE all-position loss).

## 5. Loss & training configuration

| Parameter | Value | Notes |
|---|---|---|
| Loss | MSE | No SAM, no L1, no auxiliary. Matches Noise2Noise theorem. |
| Optimizer | AdamW | betas (0.9, 0.95), wd 0.05 |
| LR (peak) | 6e-4 | = 1.5e-4 × (1024 / 256) |
| Schedule | 10-epoch linear warmup + cosine decay to 0 | Identical to v3 |
| Batch size | 1024 | Identical to v3 |
| Epochs | 200 | 160 SPEND + 20 anneal + 20 plain MAE |
| Patches per epoch | 200,000 | Identical to v3 |
| Spatial mask ratio | 0.75 | Identical to v3 |
| Spectral mask ratio | 0.5 (epochs 1–160), linearly anneals 0.5→0 (161–180), 0 (181–200) | New; see `compute_spectral_mask_ratio` in section 7 |
| Spectral mask schedule | linear | New |
| Gradient clipping | norm 1.0 | Identical to v3 |
| Checkpoints | best loss + every 50 epochs + final | Identical to v3 |

## 6. File structure

### New files

- **`models/spend_spatial_mae.py`**: `SpendSpatialSpectralMAE(SpatialSpectralMAE)` subclass; per-batch band-partition logic, anneal-aware `spectral_mask_ratio` attribute, SPEND loss.
- **`scripts/pretrain_spatial_mae_spend.py`**: 200-epoch training driver. CLI: `--spectral_mask_ratio` (default 0.5), `--anneal_start_epoch` (default 161), `--anneal_end_epoch` (default 181). Mirrors `pretrain_spatial_mae_denoising.py` structure including wandb integration and `--resume` support.
- **`scripts/hpc_pretrain_spend.slurm`**: HPC job; runs *concurrently* with the existing v3 denoising job. Same data root (`/xdisk/sbyrne/phillipsm/CRISM_MRDR`), 48-hour budget, separate `--job-name=spatial_mae_spend`.
- **`tests/test_spend_spatial_mae.py`**: unit tests (see section 7).
- **`scripts/figures/fig_spend_partition.py`**: post-pretraining validation figure (see section 8).

### Reused (unchanged)

- `models/spatial_spectral_transformer.py` (encoder).
- `models/spatial_mae.py` (base MAE; decoder, spatial masking, encoder state-dict extraction).
- `data/global_patch_dataset.py` (dataset and dataloader).
- `config_loader.py`, `config.local.yaml` (config plumbing).

### Why a subclass, not a flag on `DenoisingSpatialSpectralMAE`

The SPEND objective changes both the input transform (band partition replaces noise injection) and the loss target (target-half bands replace clean signal). Expressing both modes via a flag would tangle two distinct objectives in one class and obscure the Noise2Noise math. Subclassing keeps each pretraining objective self-contained and independently testable.

## 7. Testing strategy

Tests in `tests/test_spend_spatial_mae.py`:

1. **Partition validity:** For `spectral_mask_ratio=0.5` and 1000 sampled partitions, every `m_band` has exactly `59 - round(59*0.5) = 30` ones (input bands) and `29` zeros (target bands); across the 1000 samples, every band index 0..58 appears in the target half in at least 10 of them (i.e., the partition is unbiased across band positions).
2. **Shape correctness:** `forward(x)` returns `(loss: scalar, recon: (B,49,59), mask: (B,49) bool)`.
3. **Loss localization at fixed partition:** with a fixed `m_band` and known `recon = x_clean + δ` for known `δ`, the returned loss equals `(δ[..., target_bands]**2).mean()`.
4. **Anneal schedule:** a free function `compute_spectral_mask_ratio(epoch, anneal_start_epoch=161, anneal_end_epoch=181, base=0.5)` returns `base` for `epoch < anneal_start_epoch`, `0.0` for `epoch >= anneal_end_epoch`, and `base * (anneal_end_epoch - epoch) / (anneal_end_epoch - anneal_start_epoch)` in between. With defaults this gives: epoch 160 → 0.5, epoch 161 → 0.5, epoch 170 → 0.275, epoch 180 → 0.025, epoch 181 → 0.0.
5. **Degenerate-ratio edge case:** at `spectral_mask_ratio=0`, the loss equals MSE over all 59 bands at all 49 positions (matches the v3 all-position loss).
6. **N2N gradient direction (sanity check):** synthetic test on Gaussian-clean signal + i.i.d. noise; after 50 optimizer steps, `MSE(recon, clean_signal) < MSE(recon, noisy_signal)`. Confirms training pushes toward the signal, not toward memorizing the noise.
7. **Encoder state-dict compatibility:** `SpendSpatialSpectralMAE.encoder_state_dict()` loads into `SpatialSpectralClassifier` with no unexpected keys and no missing `encoder.encoder.*` weights.

**Smoke test (not a unit test):** a 5-epoch `--patches_per_epoch 1000` dry-run on local CPU/GPU to confirm the loss curve descends and no NaNs appear.

## 8. Validation figure (post-pretraining)

`scripts/figures/fig_spend_partition.py` → `reports/v5/fig_v5_spend_partition.png`. For each of olivine / HCP / plagioclase center pixels:

- **Col 1:** clean center-pixel spectrum (reference).
- **Col 2:** one sample partition — input-half bands (gray markers) vs target-half bands (colored markers) overlaid on the spectrum.
- **Col 3:** model's predicted target-band values (line) overlaid on actual target-band values (markers), at the same partition. Demonstrates the model interpolates the underlying spectrum.
- **Col 4:** residual = `(prediction − target)` per band. Should look like centered i.i.d. noise — if it has structure (a bias, a periodic component, a spike near 1 µm), the model is failing to denoise the structured part.

Figure is generated after pretraining completes, added to `wiki/Methodology Log v5.md` as a new section 15.

## 9. Success criteria

This pretraining run is considered successful if **either** of the following holds, evaluated on the existing classifier benchmark (5-class mineral mAP on the stratified-tile validation set):

- **(a) Classifier transfer:** with the SPEND-pretrained encoder loaded into `SpatialSpectralClassifier` and fine-tuned with `lrscale=0.01` (current best v3+v4 setting), val_mAP ≥ 0.7175 (the v5 current best). Stretch: ≥ 0.72.
- **(b) Reconstruction quality:** the post-pretraining `fig_v5_spend_partition.png` Col 3 shows visually clean interpolation (no high-frequency residual structure) on all three mineral classes; and Col 4 residuals look centered and unstructured.

We also track:
- HCP/LCP confusion: cosine similarity between HCP and LCP class-mean embeddings should not be higher than v3's value.
- Reconstruction loss curve: final epoch loss < v3's final epoch loss (different objective so the values are not directly comparable, but both should descend monotonically modulo cosine-decay artifacts).

## 10. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Band-spacing too wide (median 33 nm) makes the signal-prediction task too hard, model never converges. | Medium | Smoke test catches this in 5 epochs. If loss plateaus high, fall back to stratified-by-spectral-region partition (Q2 option 3) before further investment. |
| Annealing too abrupt; encoder degrades when switching from 50% mask to 0% in 20 epochs. | Low | Use linear interpolation, not step. Track loss continuity at the phase transitions (epochs 161 and 181). |
| Encoder zeroing trick interacts badly with `band_embed = Linear(59, 128)` — the model could specialize the linear weights to the always-on band positions, failing to generalize when full bands appear in phase C. | Medium | Random per-batch partition (vs fixed odd/even) is specifically chosen to spread "always-on" probability evenly across all 59 bands during phase A. Phase C then exposes the encoder to full-band input. |
| Downstream transfer is worse than v3 despite cleaner pretraining. | Medium | v3 remains as the alternative pretrained encoder. We pick whichever wins on val_mAP. |
| Wavelength offset in target prediction (median 33 nm between adjacent bands) leaks signal information through smooth-spectrum interpolation rather than denoising. | Low | This is the whole point — the model *is* supposed to learn the smooth-spectrum prior; that's how it denoises. As long as Col 4 residuals look unstructured, the objective is working. |

## 11. Out of scope

- Real noisy-pair training from multi-observation MRDR overlap regions. (Tabled per user — would require additional disk capacity for original mapping-strip data.)
- Adjacent-column sampling (E2E-CRISM style). Available as a v5 alternative if v4 underperforms.
- Wavelength-aware positional encoding at the spectral level (would require encoder architectural changes).
- Producing a denoised global mosaic from the trained encoder. (Future project once v4 encoder is validated.)
- Replacing v3. v3 runs to completion on HPC in parallel and remains a baseline.

## 12. Open questions resolved during brainstorming

1. **Replace v3 or run in parallel?** → Run in parallel on separate HPC job.
2. **Partition strategy?** → Random per-batch (~30/29 split), not strict odd/even.
3. **Spatial-MAE masking composition?** → Keep 75% spatial mask + SPEND spectral mask = 12.5% of patch data visible to encoder.
4. **Inference/fine-tune distribution gap?** → Anneal spectral mask to 0 over epochs 161–180; phase C of training is pure spatial MAE.

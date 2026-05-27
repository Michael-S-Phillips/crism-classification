# mrrsu RPEAK1/BD1300 Auxiliary Injection — Design

> Status: approved design, pending implementation plan
> Date: 2026-05-27

## Problem & Motivation

Plagioclase classification is encoder-limited at val_AP ≈ 0.13–0.14: the 59-band mrral
encoder cannot derive the plag-vs-olivine discriminant from raw spectra, and plag-aware
re-pretraining barely moved it (0.127 → 0.14 on the honest val split). The user identifies
plagioclase in practice with two mrrsu summary parameters:

- **BD1300** (mrrsu band 17): 1.3 µm band depth — fires for *both* plag and olivine.
- **RPEAK1** (mrrsu band 8): reflectance-peak position — **plag's peak is at longer
  wavelengths than olivine's**, the actual discriminant.

### Signal check (de-risking, 2026-05-27)

A 2-feature logistic regression on `pixels.parquet` established that the discriminant is
**regional, not per-pixel**:

| level | plag RPEAK1 | olivine RPEAK1 | plag-vs-oli AUC |
|---|---|---|---|
| per-pixel | 0.730 (lower) | 0.738 | ~0.43 (noise — direction even reverses) |
| **polygon-mean** | **0.771 (higher ✓)** | **0.746** | **0.745** |

Per-pixel RPEAK1 is buried in noise (~0.3σ separation); averaging over a neighborhood
recovers the user's expected separation (√N denoising). RPEAK1 carries the signal
(polygon AUC 0.745); BD1300 alone is AUC 0.41 (fires for both, as expected).

**Conclusion:** inject a **spatially-smoothed** RPEAK1 (+ BD1300 as a weak add-on), not
per-pixel values. This targets the plag↔olivine confusion specifically (AUC ~0.75 ceiling
for that sub-problem).

## Goal

Give the classifier the smoothed RPEAK1/BD1300 discriminant directly, to push plag AP above
the ~0.14 encoder ceiling. Realistic framing: it addresses plag↔olivine confusion and should
move plag AP, paired with (not replacing) the encoder. Publishable target remains plag
AP ≈ 0.6; any clear, honest gain over the encoder-only baseline is progress.

## Decisions (locked during brainstorming)

1. **Features:** `[mean_7×7(RPEAK1, band 8), mean_7×7(BD1300, band 17)]` — 2 scalars per
   pixel, NODATA-excluded, z-scored with train-set mean/std.
2. **Smoothing scale:** 7×7 patch-mean (matches the encoder's patch size).
3. **Fusion:** late fusion — preserve the warm-startable 59-band encoder; concat an aux
   embedding to the center-token embedding before the head.
4. **Scope:** classifier + inference only. Pretraining stays mrral-only (the global patch
   cache has no mrrsu).

## Architecture

### Model — `SpatialSpectralClassifierAux`

Wraps the existing `SpatialSpectralTransformer` encoder unchanged:
- `encoder(x)` → center-token embedding `(B, 128)` (identical to `SpatialSpectralClassifier`).
- `aux_mlp`: `Linear(2, 16) → ReLU → Linear(16, 16)` on the 2 z-scored smoothed params.
- `head`: `Linear(embed_dim + 16, n_classes)` on `concat(center, aux_mlp(aux2))`.

`forward(x, aux2)` where `x` is `(B, 7, 7, 59)` and `aux2` is `(B, 2)`.

Encoder loads from any MAE checkpoint via `load_encoder_state_dict` with no shape change;
`aux_mlp` + `head` are fresh. `get_param_groups(head_lr, encoder_lr)` puts encoder params in
the encoder group and `aux_mlp` + `head` in the head group (so encoder_lr_scale still works).

### Data flow

- **Build step** — new `scripts/build_mrrsu_aux.py`:
  - For each split, for each labeled pixel in `mrral_pixels.parquet`, open the paired mrrsu
    tile (via a mrral→mrrsu `mrrsu_map`), compute the 7×7 mean of band 8 and band 17 at the
    pixel center (excluding NODATA 65535; if all-NODATA in the window, fall back to 0 after
    z-scoring → mean), and write an aligned `data/patch_cache/mrrsu_aux_{split}.npy` of shape
    `(n_split, 2)` in parquet-row order (column 0 = RPEAK1 mean, column 1 = BD1300 mean).
  - Compute train-split mean/std per feature and write `data/patch_cache/mrrsu_aux_stats.json`
    (`{"mean": [...], "std": [...]}`). Stats are computed on raw (pre-z-score) values.
- **Dataset** — new `MrrsuAuxPatchDataset` (or a thin wrapper) that pairs the existing
  mrral patch with the aligned aux-2 vector and applies z-scoring with the stored stats:
  `__getitem__ → (patch (7,7,59), aux2 (2,), label (5,), weight)`.
- **Inference** — extend `classify_tile_supervised.py`: locate the paired mrrsu `.img`,
  read bands 8 & 17, compute 7×7-mean rasters (NODATA-aware), z-score with the stored stats,
  and feed `aux2` per pixel alongside the mrral patch to `SpatialSpectralClassifierAux`.

### Training & eval

- Fine-tune from the best available encoder (`--pretrain_ckpt`, currently the plag-aware
  encoder / its continuation) with the cont1 recipe (encoder_lr_scale 0.001, ASL, lr 5e-4,
  batch 256, 6L/128d/4H, patch 7).
- Clean ablation: **same encoder, with vs without the aux injection**, plag AP on the
  official val split. The "without" arm is the existing `ft_plag_aware_real_only` number.

## Files

New:
- `models/spatial_spectral_classifier_aux.py` — `SpatialSpectralClassifierAux`.
- `scripts/build_mrrsu_aux.py` — aligned aux cache + stats builder.
- `data/dataset.py` — add `MrrsuAuxPatchDataset` (or wrapper).
- `scripts/hpc_finetune_mrrsu_aux.slurm` — HPC fine-tune.
- tests: `tests/test_spatial_spectral_classifier_aux.py`, `tests/test_mrrsu_aux_dataset.py`.

Modify:
- `scripts/train.py` — add a dedicated `--model spatial_vit_aux` branch (mirrors the
  `spatial_vit` branch) that builds `SpatialSpectralClassifierAux` and passes the aux cache
  paths (`--mrrsu_aux_cache`, `--mrrsu_aux_stats`) through to `train_torch_model`.
- `training/train_torch.py` — when aux cache paths are provided, build the aux dataset and
  unpack the `(patch, aux2, label, weight)` batch (aux model's forward takes `(x, aux2)`).
- `scripts/classify_tile_supervised.py` — paired-mrrsu reading + aux feed at inference.

## Testing

- **Unit:** `SpatialSpectralClassifierAux.forward((B,7,7,59),(B,2))` → `(B,5)`; encoder_state
  loads from a plain MAE checkpoint with empty missing/unexpected; `get_param_groups` splits
  encoder vs aux_mlp+head correctly.
- **Unit:** `MrrsuAuxPatchDataset.__getitem__` returns the 4-tuple with z-scored aux2 of the
  right shape; alignment assertion (cache rows == parquet rows) holds.
- **Unit:** 7×7-mean helper excludes NODATA and matches a hand-computed value on a small
  synthetic raster.
- **Smoke:** build the aux cache for one tile / small subset locally; a 1-step forward+loss
  on random patches + aux runs and the checkpoint reloads its encoder into the plain
  classifier.
- **Integration:** inference on one tile produces a probability map without shape errors.

## Out of Scope

- Feeding params into pretraining (global cache has no mrrsu).
- Larger smoothing radii / learned pooling (later knob if 7×7 underperforms).
- Per-pixel (unsmoothed) param injection (the signal check showed it carries no signal).

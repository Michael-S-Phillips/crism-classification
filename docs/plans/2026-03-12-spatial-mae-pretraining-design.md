# Spatial Spectral MAE Pre-training Design

**Date:** 2026-03-12
**Status:** Approved

## Problem

The current `SpectralMAE` pre-trains on ~1.97M labeled pixels from mineral polygon regions
(`data/mrral_pixels.parquet`). This is a tiny, geographically biased sample of the global
Mars surface. The resulting checkpoint (best loss: 1.81e-5) has almost certainly converged
on this small dataset rather than learning general spectral-spatial representations. The
full global MRDR dataset contains ~3.87B pixels across 1,764 mrral tiles — roughly 2,000×
more data the model has never seen.

## Solution

Replace the per-pixel 1D spectral MAE with a **spatial patch MAE** that:
1. Learns spatial geological context (not just per-pixel spectral patterns)
2. Pre-trains on the **full global dataset** by streaming 7×7 patches directly from tiles

## Architecture

### `SpatialSpectralTransformer` (`models/spatial_spectral_transformer.py`)

Encoder used for both pre-training and downstream classification.

```
Input: (B, 7, 7, 59) → reshape to (B, 49, 59)
band_embed: Linear(59 → embed_dim)        # per-pixel spectral projection
pos_embed:  Embedding(50, embed_dim)      # 0=CLS, 1–49=spatial positions (learned 2D)
CLS token prepended → (B, 50, embed_dim)
n_layers × TransformerEncoderLayer(Pre-LN, nhead=4, ffn_dim=embed_dim*4)
```

For downstream classification: center-pixel token (position 25 for 7×7) → LayerNorm → head.

### `SpatialSpectralMAE` (`models/spatial_mae.py`)

```
Pre-training forward:
  1. Flatten to (B, 49, 59); mask 75% of spatial positions (37/49 masked)
  2. Encode 12 visible tokens + CLS through SpatialSpectralTransformer
  3. Decoder: learnable mask token + positional embed at masked positions
             → 2-layer transformer decoder (decoder_dim=64)
             → Linear(decoder_dim → 59) per masked token
  4. Loss: MSE on masked pixel spectra only
```

Decoder uses its own learnable mask token (not zero-fill) to prevent encoder from encoding
position information to help reconstruction — standard MAE design.

### Hyperparameters

| Param | Value | Rationale |
|---|---|---|
| patch_size | 7 | matches existing config |
| embed_dim | 128 | same as current, fits 17GB GPU |
| n_heads | 4 | same as current |
| n_layers | 6 | increased from 4; spatial task is richer |
| mask_ratio | 0.75 | standard for spatial MAE (He et al.) |
| decoder_dim | 64 | lightweight asymmetric decoder |
| decoder_layers | 2 | standard |

## Data Pipeline

### `CRISMGlobalPatchDataset` (`data/global_patch_dataset.py`)

`torch.utils.data.IterableDataset` that streams patches from all 1,764 mrral tiles.

**Sampling**: Two-level random sampling per worker:
1. Sample a tile uniformly at random (weighted by tile pixel area)
2. Sample a random valid (row, col) within that tile

Avoids costly full-scan pre-indexing; training starts immediately.

**Validity**: A patch is valid if ≥80% of its 49 pixels have all 59 bands non-NaN and ≠ 65535.
Invalid patches are resampled. Residual NaN values within a valid patch are set to 0.0.

**Normalization**: Clip to `[0.0, 0.5]` — covers P99 of valid reflectance (~0.28) with headroom,
eliminates artifact outliers. No per-band z-scoring; MAE reconstructs physically interpretable
reflectance values consistent across tiles.

**DataLoader**: `num_workers=8`, `prefetch_factor=4`, `pin_memory=True`. Workers each hold an
independent shard of the tile list. File handles cached per tile, pid-safe (same pattern as
`CRISMPatchDataset`).

**Epoch definition**: 1 epoch = 1M patches. Dataset is effectively infinite.

## Training Setup

**Script**: `scripts/pretrain_spatial_mae.py`

**Optimizer**: AdamW, β1=0.9, β2=0.95, weight_decay=0.05
**LR**: `1.5e-4 × batch_size / 256` (linear scaling), warmup 40 epochs → cosine decay to 0
**Batch size**: 512
**Total epochs**: 400 (400M patches total)

**Checkpoints**:
- Best (lowest loss): `checkpoints/spatial_mae_{embed_dim}d_{n_layers}l_best.pt`
- Periodic: every 50 epochs to `checkpoints/spatial_mae_{embed_dim}d_{n_layers}l_epoch{N}.pt`
- Format: `{'encoder_state': ..., 'mae_state': ..., 'mae_loss': ..., 'epoch': ..., 'config': {...}}`

**Logging**: wandb + stdout per epoch. Per-epoch visual sanity check: 4 fixed patches,
original vs. reconstructed spectra.

## Downstream Integration

### `SpatialSpectralClassifier`

Defined in `models/spatial_spectral_transformer.py` alongside the encoder.

```
Input:  (B, patch_size, patch_size, 59)
Encoder: SpatialSpectralTransformer (loaded from checkpoint)
Center-pixel token (position 25) → LayerNorm → Linear(embed_dim → n_classes)
Output: (B, n_classes) logits
```

**Fine-tuning**:
```bash
conda run -n crism python scripts/train.py \
  --model spatial_vit \
  --n_layers 6 --embed_dim 128 \
  --pretrain_ckpt checkpoints/spatial_mae_128d_6l_best.pt \
  --encoder_lr_scale 0.1
```

**Differential LR**: `get_param_groups()` returns slow LR for pretrained encoder, fast LR
for randomly-initialized classification head — same pattern as `SpectralTransformer`.

**Data**: `CRISMPatchDataset` extended to read mrral tiles (currently reads mrrsu). The
labeled parquet already has `tile_id`, `pixel_row`, `pixel_col` — patch extraction at
fine-tuning time is straightforward.

## New Files

| File | Purpose |
|---|---|
| `models/spatial_spectral_transformer.py` | `SpatialSpectralTransformer` + `SpatialSpectralClassifier` |
| `models/spatial_mae.py` | `SpatialSpectralMAE` |
| `data/global_patch_dataset.py` | `CRISMGlobalPatchDataset` (IterableDataset) |
| `scripts/pretrain_spatial_mae.py` | Pre-training script |

## Modified Files

| File | Change |
|---|---|
| `models/__init__.py` | Export new model classes |
| `scripts/train.py` | Add `spatial_vit` model option |
| `data/dataset.py` | Add mrral patch reading to `CRISMPatchDataset` |

## Critical Constraint

When fine-tuning, `--n_layers` and `--embed_dim` passed to `train.py` **must match** the
pre-training checkpoint config. A mismatch leaves some encoder layers randomly initialized
while receiving the slow pretrained encoder LR — a silent bug that degrades performance.

# Spatial MAE Model Overview

## Pre-training: SpatialSpectralMAE

The pre-training objective is a **Masked Autoencoder (MAE)** over spatial patches of CRISM mrral data, adapted from He et al. (2022).

**Input:** A 7×7 spatial patch of pixels, each pixel being a 59-band mrral spectrum — shape `(batch, 7, 7, 59)`.

**Task:** Randomly mask 85% of the 49 pixels in each patch. Train the encoder to reconstruct the missing spectra from the visible ones.

This forces the encoder to learn both **spectral shape** (what a mineral spectrum looks like) and **spatial context** (how neighboring pixels relate to each other). After pre-training on all 1,764 global CRISM mrral tiles (~100K patches/epoch, 52 epochs on HPC), the encoder weights are saved.

### Architecture

```
SpatialSpectralMAE
├── encoder: SpatialSpectralTransformer   ← the part we keep
│   ├── band_embed: Linear(59 → 128)      project each pixel's spectrum to embed_dim
│   ├── pos_embed: Embedding(50, 128)     learned position per spatial slot (+ CLS)
│   ├── cls_token: Parameter(128)
│   └── encoder: TransformerEncoder       6 layers, 4 heads, dim_feedforward=512
│
├── enc_to_dec: Linear(128 → 64)          project encoder output to decoder space
├── mask_token: Parameter(64)             learned placeholder for masked positions
├── decoder_pos_embed: Embedding(50, 64)
├── decoder: TransformerEncoder           2 layers (lightweight)
└── reconstruction_head: Linear(64 → 59) reconstruct masked pixel spectra
```

The decoder and reconstruction head are **discarded** after pre-training. Only the `SpatialSpectralTransformer` encoder is carried forward.

---

## Fine-tuning: SpatialSpectralClassifier

The classifier wraps the pre-trained encoder with a single linear classification head.

### Architecture

```
SpatialSpectralClassifier
├── encoder: SpatialSpectralTransformer   ← loaded from MAE checkpoint
│   (same architecture as above)
└── head: Linear(128 → 5)                 mineral classification head
```

### Forward pass

```
Input patch (7×7×59)
       ↓
SpatialSpectralTransformer
       ↓  outputs 50 tokens: [CLS, pixel_0, ..., pixel_48]
Take center pixel token (pixel_24 = center of 7×7 grid, index 25 with CLS offset)
       ↓  shape: (batch, 128)
Linear(128 → 5)
       ↓
5-class logits: olivine | lcp | hcp | plagioclase | other
```

**Why the center pixel token, not CLS?** The task is per-pixel classification — we want the label for the center pixel of the patch. The center pixel's token has attended to all 48 surrounding pixels via the transformer, so it encodes "this pixel's spectrum in the context of its spatial neighborhood." CLS would aggregate the whole patch, which is appropriate for patch-level tasks but not per-pixel labeling.

**The head is a single linear layer** (no hidden layers). The embedding is expected to be rich enough from pre-training that a linear probe suffices. This also keeps fine-tuning fast and reduces overfitting risk on the limited labeled dataset.

### Differential learning rates

When loading a pre-trained encoder, the encoder and head use different learning rates:

```python
# --encoder_lr_scale 0.1 means:
encoder_lr = 0.1 × base_lr   # encoder trained slowly to preserve pre-trained weights
head_lr    = 1.0 × base_lr   # head trained at full rate (randomly initialized)
```

---

## Experimental runs

Two runs are being compared to isolate the contribution of spatial MAE pre-training:

| Run | Encoder init | Base LR | Encoder LR scale | Purpose |
|-----|-------------|---------|-----------------|---------|
| `spvit_spatial_mae` | HPC checkpoint (epoch 52, mae_loss=0.01793) | 5e-4 | 0.1× | Does pre-training help? |
| `spvit_baseline` | Random | 5e-4 | 1.0× | How well does the architecture alone do? |

Both runs use: ASL loss, 50 epochs, patience=10, batch_size=256, embed_dim=128, 6 layers, 4 heads, patch_size=7, mrral patch cache.

### Pre-training checkpoint

`checkpoints/spatial_mae_128d_6l_best.pt` — trained on HPC cluster.

```
epoch:     52
mae_loss:  0.01793
config:    embed_dim=128, n_layers=6, n_heads=4, mask_ratio=0.85,
           patches_per_epoch=100_000, batch_size=512
```

---

## Code locations

| File | Role |
|------|------|
| `models/spatial_spectral_transformer.py` | `SpatialSpectralTransformer` (encoder) and `SpatialSpectralClassifier` |
| `models/spatial_mae.py` | `SpatialSpectralMAE` (pre-training wrapper) |
| `scripts/pretrain_spatial_mae.py` | Pre-training entry point |
| `scripts/train.py` | Fine-tuning entry point (`--model spatial_vit`) |
| `config.local.yaml` | Local path overrides (data_root, checkpoints_dir, etc.) |
| `checkpoints/spatial_mae_128d_6l_best.pt` | Pre-trained encoder weights |
| `data/patch_cache/mrral_*_patches_p7.npy` | Cached 7×7 patches (train=20GB, val=1.1GB, test=834MB) |

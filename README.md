# CRISM Mineral Classification Pipeline

Multi-label pixel classification for Mars CRISM MRDR mrral hyperspectral tiles (59 spectral bands, ~410–3900 nm).

**Classes:** olivine · lcp · hcp · plagioclase · other (5-class multi-label)

---

## Pipeline Overview

```
mrral tiles + GeoPackage labels
        │
        ▼
build_mrral_dataset.py      →  data/mrral_pixels.parquet  (1.97M labeled pixels)
cache_mrral_patches.py      →  data/patch_cache/mrral_{split}_patches_p7.npy
        │
        ├──  pretrain_spatial_mae.py   →  checkpoints/spatial_mae_128d_6l_best.pt
        │
        └──  train.py                 →  checkpoints/spvit_lrscale0005_best.pt
                │
                ├── Supervised inference
                │     classify_tile_supervised.py   →  (H,W,5) probs .npz
                │
                └── Prototype inference (cosine similarity)
                      build_prototypes.py            →  data/prototypes/*.npz
                      classify_tile_prototype.py     →  (H,W,5) similarity .npz
                              │
                              ▼
                 compute_global_thresholds.py  →  config/vectroscopy_thresholds.json
                 vectorize_tile_minerals.py    →  data/vector/*.gpkg
                 plot_vector_mineral_maps.py   →  reports/fig_vector_mineral_maps.png
```

---

## Setup

```bash
conda run -n crism pip install scikit-learn xgboost lightgbm torch torchvision \
    wandb pyarrow tqdm pyyaml pytest rasterio geopandas shapely fiona
```

---

## Data Preparation

```bash
# Build pixel dataset from geopackages + mrral tiles (~5-10 min)
conda run -n crism python scripts/build_mrral_dataset.py

# Pre-extract 7×7 patches into memory-mapped cache (required for training + prototypes)
conda run -n crism python scripts/cache_mrral_patches.py
```

Produces:
- `data/mrral_pixels.parquet` — 1.97M pixels × columns (labels + metadata + split)
- `data/patch_cache/mrral_{train,val,test}_patches_p7.npy` — `(N, 7, 7, 59)` float32 memmaps

---

## MAE Pre-training

```bash
conda run -n crism python scripts/pretrain_spatial_mae.py \
    --epochs 50 --mask_ratio 0.85 \
    --out checkpoints/spatial_mae_128d_6l_best.pt
```

Trains a `SpatialSpectralMAE` (75% spatial masking) on unlabeled mrral tiles. The encoder (`SpatialSpectralTransformer`, 128-dim, 6 layers) is reused for fine-tuning and prototype building.

---

## Fine-tuning

```bash
conda run -n crism python scripts/train.py \
    --model spatial_vit \
    --ckpt checkpoints/spatial_mae_128d_6l_best.pt \
    --lr_scale 0.0005

# W&B hyperparameter sweep
conda run -n crism wandb sweep config/sweep_v6.yaml
conda run -n crism wandb agent <sweep_id>
```

---

## Supervised Tile Classification

```bash
conda run -n crism python scripts/classify_tile_supervised.py \
    --tile /mnt/mrdr/mc26/t0435_mrral_40s323_0327_4.img \
    --ckpt checkpoints/spvit_lrscale0005_best.pt \
    --save_probs /tmp/t0435_probs.npz
```

Outputs `(H, W, 5)` sigmoid probability `.npz`.

---

## Prototype Classifier (cosine similarity)

An alternative to the trained linear head that uses labeled pixels as anchor embeddings. Avoids class-imbalance miscalibration.

```bash
# Step 1: Build per-class prototypes from labeled pixels
conda run -n crism python scripts/build_prototypes.py \
    --ckpt checkpoints/spvit_lrscale0005_best.pt \
    --out data/prototypes/proto_finetuned_all.npz

# MAE encoder variant
conda run -n crism python scripts/build_prototypes.py \
    --ckpt checkpoints/spatial_mae_128d_6l_best.pt \
    --out data/prototypes/proto_mae_all.npz

# High-confidence pixels only
conda run -n crism python scripts/build_prototypes.py \
    --ckpt checkpoints/spvit_lrscale0005_best.pt \
    --confidence_tiers High \
    --out data/prototypes/proto_finetuned_high.npz

# Step 2: Classify tile + optional comparison figure
conda run -n crism python scripts/classify_tile_prototype.py \
    --tile /mnt/mrdr/mc26/t0435_mrral_40s323_0327_4.img \
    --proto_a data/prototypes/proto_finetuned_all.npz \
    --proto_b data/prototypes/proto_mae_all.npz \
    --supervised_probs /tmp/t0435_probs.npz \
    --save_probs /tmp/t0435_proto_probs.npz \
    --out reports/fig_prototype_t0435.png
```

Output is the same `(H, W, 5)` `.npz` schema as supervised classification — drop-in for the downstream pipeline.

---

## Vectroscopy Pipeline

Converts per-pixel probability maps to GeoPackage vector mineral maps.

```bash
# 1. Compute global percentile thresholds across tiles
conda run -n crism python scripts/compute_global_thresholds.py \
    --probs /tmp/t0435_probs.npz /tmp/t0434_probs.npz \
    --out config/vectroscopy_thresholds.json

# 2. Vectorize one tile
conda run -n crism python scripts/vectorize_tile_minerals.py \
    --tile /mnt/mrdr/mc26/t0435_mrral_40s323_0327_4.img \
    --probs /tmp/t0435_probs.npz \
    --thresholds config/vectroscopy_thresholds.json \
    --out data/vector/t0435_mineral_map.gpkg

# 3. Plot all tiles
conda run -n crism python scripts/plot_vector_mineral_maps.py
```

Outputs `reports/fig_vector_mineral_maps.png` — 5×2 grid (minerals × tiles), coloured by confidence tier.

---

## Tests

```bash
conda run -n crism python -m pytest tests/ -v
```

---

## Model Architecture

| Component | Details |
|---|---|
| Encoder | `SpatialSpectralTransformer` — 128-dim, 6 layers, 4 heads, 7×7 patches, 59 bands |
| Pre-training | `SpatialSpectralMAE` — 75% spatial masking |
| Classifier | `SpatialSpectralClassifier` — encoder + linear head, multi-label sigmoid |
| Prototype | Cosine similarity to per-class mean embeddings (L2-normalized) |

Center token index: `PATCH_SIZE² // 2 + 1 = 25` (slot 0 = CLS, slots 1–49 = spatial).

---

## Label Encoding

- **Confidence tiers:** High / Moderate / Low (935K / 870K / 162K pixels)
- **Olivine:** collapsed from `olivine_t1` + `olivine_t2` via max
- **Multi-label:** pixels can be positive for multiple classes
- **Hard positives:** class value == 1.0 (used for prototype building)

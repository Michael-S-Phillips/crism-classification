# CRISM Mineral Classification Pipeline — Design Document

**Date:** 2026-03-06
**Status:** Approved

---

## Overview

A machine learning pipeline to classify Mars CRISM MRDR pixels into primary rock-forming mineral classes using the 60-band summary parameter image cubes (`mrrsu`). Training data comes from 38 geopackage files containing manually mapped polygon ROIs with mineral category labels and confidence levels.

---

## Problem Statement

- **Input at inference:** a full CRISM mrrsu tile (60 spectral parameters × H × W pixels)
- **Output:** per-pixel probability maps for 6 classes, plus a single best-class prediction map
- **Task type:** multi-label classification (pixels can belong to multiple mineral classes simultaneously)
- **Classes:** `olivine_t1`, `olivine_t2`, `lcp`, `hcp`, `plagioclase`, `other`

---

## Data

### Source
- 38 paired (gpkg, mrrsu) tiles — all in `/mnt/crism/MRDR/mc25` and `/mnt/crism/MRDR/mc26`
- Geopackages in `/mnt/crism/MRDR/categorized_mineral_units/`
- Pairing by observation ID: `T0434.gpkg` ↔ `t0434_mrrsu_*.img`
- 3,260 total labeled polygons; ~60 spectral parameter bands per pixel

### Feature Extraction
- **Per-pixel extraction:** rasterize each polygon ROI onto the mrrsu grid, extract all pixel values (60 float32 bands per pixel)
- **NaN handling:** drop pixels where any band = 65535 or NaN (CRISM no-data value)
- Output stored as `data/pixels.parquet`

### Parquet Schema
```
tile_id, polygon_id, pixel_row, pixel_col,
b0..b59,                              # 60 spectral parameters (float32)
olivine_t1, olivine_t2, lcp, hcp, plagioclase, other,  # float: 0, 0.5, or 1.0
confidence_weight,                    # 0.25 / 0.5 / 1.0
confidence_tier,                      # "Low" / "Moderate" / "High"
split                                 # "train" / "val" / "test"
```

### Train/Val/Test Split
Split at **tile level** to prevent spatial leakage: ~26 train / 6 val / 6 test tiles (≈70/15/15), stratified so all mineral classes appear in each split.

### "Other" Class
The "Other" category is a spectral denominator class (spectrally neutral/bland). Downsample to ~400 randomly selected "Other" polygons across all tiles before pixel extraction, preserving tile-level split proportions.

---

## Label Encoding

Category strings are parsed by `data/label_parser.py` into multi-hot float vectors:

| Category string | `[t1, t2, lcp, hcp, plag, other]` | Notes |
|---|---|---|
| `Type 1 olivine (*)` | `[1, 0, 0, 0, 0, 0]` | |
| `Type 2 olivine (*)` | `[0, 1, 0, 0, 0, 0]` | |
| `lcp (*)` | `[0, 0, 1, 0, 0, 0]` | |
| `hcp (*)` | `[0, 0, 0, 1, 0, 0]` | |
| `plagioclase (*)` | `[0, 0, 0, 0, 1, 0]` | |
| `Other (*)` | `[0, 0, 0, 0, 0, 1]` | downsampled |
| `hcp + olivine (*)` | `[0.5, 0.5, 0, 1, 0, 0]` | untyped olivine → soft split |
| `olivine + plagioclase (*)` | `[0.5, 0.5, 0, 0, 1, 0]` | |
| `hcp + lcp (*)` | `[0, 0, 1, 1, 0, 0]` | |
| `alteration + olivine (*)` | `[0.5, 0.5, 0, 0, 0, 0]` | alteration not a target |
| `alteration + plagioclase (*)` | `[0, 0, 0, 0, 1, 0]` | |

### Confidence → Sample Weight
- `High` → 1.0
- `Moderate` → 0.5
- `Low` → 0.25

Sklearn models use `sample_weight` parameter. PyTorch models use `WeightedBCEWithLogitsLoss`.

---

## Model Architectures

All models share the same train/val/test pixel dataset and report identical metrics.

### Linear Baselines (`models/linear.py`)
- `LogisticRegression` + `MultiOutputClassifier`
- `LinearSVC` + `MultiOutputClassifier`

### Tree Ensembles (`models/tree_ensemble.py`)
- `RandomForestClassifier` + `MultiOutputClassifier`
- `XGBoostClassifier` (native multi-output + sample weights)
- `LightGBMClassifier` (native multi-output + sample weights)

### MLP (`models/mlp.py`, PyTorch)
- 60 → BN → 256 → ReLU → Dropout(0.3) → 128 → ReLU → Dropout(0.3) → 6
- Loss: `WeightedBCEWithLogitsLoss`

### Spatial-Spectral CNN (`models/cnn.py`, PyTorch)
- Input: 7×7 pixel patch × 60 bands extracted from mrrsu cube
- Conv2d(60→128, 3×3) → BN → ReLU → Conv2d(128→256, 3×3) → BN → ReLU → GlobalAvgPool → FC(256→6)

### Vision Transformer (`models/vit.py`, PyTorch)
- Input: 7×7×60 patch; each pixel = one token with 60-dim spectral embedding
- Linear embedding (60→128) + 2D positional encoding → 4-layer Transformer (4 heads, 128 dim) → CLS token → FC(128→6)

---

## Training

- **PyTorch models:** `training/train_torch.py` — epoch loop, early stopping on val mAP, W&B metric logging
- **Sklearn models:** `training/train_sklearn.py` — fit + predict, W&B metric logging
- **Loss:** `training/losses.py` — `WeightedBCEWithLogitsLoss` applying per-sample confidence weights
- **Entry point:** `scripts/train.py --model [logreg|svc|rf|xgb|lgbm|mlp|cnn|vit]`

---

## Evaluation & Metrics

Reported for all models on the test split:

- Overall mAP (mean Average Precision across 6 classes)
- Per-class AP: olivine_t1, olivine_t2, lcp, hcp, plagioclase, other
- All metrics broken out by confidence tier: High / Moderate / Low
- Per-class confusion matrices
- Precision-recall curves per class

---

## Weights & Biases Integration

- **Project:** `crism-mineral-classification`
- **Setup:** `scripts/setup_wandb.py` — prompts for API key, initializes project, writes to `config.yaml`
- **Per-run logging:** hyperparams, per-epoch metrics (PyTorch) or single-entry (sklearn), model checkpoint artifacts, W&B Tables with per-pixel predictions
- **Sweeps:** `config/sweep_mlp.yaml`, `config/sweep_xgb.yaml`, etc. — one per model family

---

## Inference

`scripts/predict_tile.py` — applies a trained model to a full mrrsu tile:
1. Load tile, mask no-data pixels
2. Extract patches (CNN/ViT) or flatten (other models)
3. Run model → sigmoid probabilities (H×W×6)
4. Write one GeoTIFF per class + argmax best-class map to `predictions/{tile_id}/`

---

## Project Structure

```
crism_classification/
├── README.md
├── environment.yml
├── config.yaml
├── config/
│   ├── sweep_mlp.yaml
│   ├── sweep_xgb.yaml
│   ├── sweep_lgbm.yaml
│   ├── sweep_cnn.yaml
│   └── sweep_vit.yaml
├── data/
│   ├── extract_pixels.py
│   ├── label_parser.py
│   ├── dataset.py
│   └── augmentations.py
├── models/
│   ├── __init__.py
│   ├── linear.py
│   ├── tree_ensemble.py
│   ├── mlp.py
│   ├── cnn.py
│   └── vit.py
├── training/
│   ├── train_sklearn.py
│   ├── train_torch.py
│   └── losses.py
├── evaluation/
│   ├── metrics.py
│   └── visualize.py
├── scripts/
│   ├── setup_wandb.py
│   ├── build_dataset.py
│   ├── train.py
│   └── predict_tile.py
└── docs/
    └── plans/
        └── 2026-03-06-crism-classification-design.md
```

---

## Dependencies to Install in `crism` Conda Env

```
scikit-learn, xgboost, lightgbm, torch, torchvision, wandb, pyarrow, fastparquet, tqdm, pyyaml, openpyxl
```

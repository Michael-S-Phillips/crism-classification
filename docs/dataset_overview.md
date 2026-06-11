# Dataset Overview

## Source

Labeled pixels extracted from CRISM MRDR mrral tiles (59-band hyperspectral reflectance, ~410–2457 nm). Labels come from manually mapped mineral unit polygons (GeoPackages) covering five mineral classes: olivine, lcp (low-calcium pyroxene), hcp (high-calcium pyroxene), plagioclase, other.

Labels are confidence-weighted (0 / 0.5 / 1.0) based on mapping confidence tier. The dataset is **multi-label** — a pixel can belong to multiple classes simultaneously.

## Files

| File | Description |
|------|-------------|
| `data/mrral_pixels.parquet` | mrral 59-band spectra + labels (used by spatial_vit, spectral_cnn, spectral_vit) |
| `data/pixels.parquet` | mrrsu 60-band summary parameters + labels (used by CNN/ViT spatial models, sklearn baselines) |
| `data/patch_cache/mrral_*_patches_p7.npy` | Pre-cached 7×7 mrral patches for fast training (~20 GB train, ~1.1 GB val, ~834 MB test) |

## Schema (mrral_pixels.parquet)

| Column | Type | Description |
|--------|------|-------------|
| `m0`–`m58` | float32 | 59-band mrral reflectance |
| `olivine_t1`, `olivine_t2` | float32 | Olivine type 1/2 confidence weights (collapsed to `olivine` during training) |
| `lcp`, `hcp`, `plagioclase`, `other` | float32 | Confidence weights (0 / 0.5 / 1.0) |
| `confidence_weight` | float32 | Per-pixel sample weight |
| `confidence_tier` | str | High / Moderate / Low |
| `split` | str | train / val / test |
| `tile_id`, `polygon_id`, `pixel_row`, `pixel_col` | — | Provenance keys |

## Size

**~1.97 million labeled pixels total.**

| Split | Count |
|-------|-------|
| Train | 1,794,293 |
| Val | 98,016 |
| Test | 75,562 |

## Confidence tiers

| Tier | Count |
|------|-------|
| High | 935,980 |
| Moderate | 870,014 |
| Low | 161,877 |

## Class distribution

Positive pixels per class (label > 0.4). Percentages sum to >100% because the dataset is multi-label.

| Class | Count | % of dataset |
|-------|-------|-------------|
| olivine | 822,923 | 41.8% |
| lcp | 436,131 | 22.2% |
| hcp | 282,500 | 14.4% |
| plagioclase | 351,926 | 17.9% |
| other | 471,095 | 23.9% |

Olivine dominates (~42% of pixels). HCP is the rarest class. This imbalance, combined with the multi-label structure, motivated use of Asymmetric Loss (ASL) over standard BCE.

# Prototype Classifier Design

## Goal

Replace the over-predicting linear head with a prototype-based cosine similarity classifier that uses labeled pixels as anchor points, comparing two encoder variants (fine-tuned vs MAE) to find which gives better mineral separation.

## Background

The current `SpatialSpectralClassifier` (fine-tuned `spvit_lrscale0005_best.pt`) collapses to predicting LCP across most of the tile. The embedding space itself separates minerals well (confirmed by PCA/k-means analysis on T0435), so the problem is the trained linear head — likely due to class imbalance and miscalibration during fine-tuning. The prototype approach removes the head entirely and uses embedding geometry directly.

## Architecture

Two new scripts, fully compatible with the existing downstream pipeline:

```
build_prototypes.py
  → data/prototypes/<encoder_tag>_<tiers>.npz   (5, 128) prototype matrix

classify_tile_prototype.py
  → (H, W, 5) similarity .npz                   same format as supervised probs
  → reports/fig_prototype_<tile>.png             optional comparison figure

Downstream (unchanged):
  compute_global_thresholds.py → vectorize_tile_minerals.py → plot_vector_mineral_maps.py
```

## Component: `scripts/build_prototypes.py`

**Inputs:**

| Flag | Default | Description |
|------|---------|-------------|
| `--ckpt` | required | Encoder checkpoint path |
| `--confidence_tiers` | `High Moderate Low` | Filter labeled pixels by tier |
| `--splits` | `train val` | Data splits to use (never test) |
| `--out` | `data/prototypes/proto_<tag>_<tiers>.npz` | Output path |

**Processing:**

1. Load `data/mrral_pixels.parquet`, filter by `split` ∈ `--splits` and `confidence_tier` ∈ `--confidence_tiers`
2. Collapse olivine labels: `olivine = max(olivine_t1, olivine_t2)` — same logic as `_collapse_labels()` in `data/dataset.py`. Hard-positive threshold: class value == 1.0.
3. Detect encoder type from checkpoint keys: if `encoder_state` present → MAE checkpoint; if `model_state` / `state_dict` present → fine-tuned `SpatialSpectralClassifier`.
4. Load encoder:
   - **Fine-tuned**: load `SpatialSpectralClassifier`, use `model.encoder` (drops the linear head)
   - **MAE**: load `encoder_state` key directly into a bare `SpatialSpectralTransformer` — do NOT load `mae_state` (full autoencoder)
5. For each of the 5 classes (olivine, lcp, hcp, plagioclase, other):
   - Select rows where that class label == 1.0 (hard positive only)
   - If zero pixels found after filtering: raise `ValueError(f"No hard-positive pixels for class '{name}' with tiers={tiers}, splits={splits}. Widen --confidence_tiers or --splits.")`
   - Group by split; for each split subset: reset integer index to look up rows in the per-split memmap `data/patch_cache/mrral_{split}_patches_p7.npy` (row order in memmap matches parquet row order within each split)
   - Load 7×7 patches, apply `normalize_patches()` (per-patch mean/std, same as `classify_tile_supervised.py`)
   - Run encoder in batches → extract center token at index `patch_size**2 // 2 + 1` (= 25 for 7×7) → `(N, 128)`
   - L2-normalize each embedding → mean → L2-normalize again = prototype
6. Save `prototypes (5, 128)`, `class_names`, `encoder_ckpt`, `confidence_tiers_used`, `splits_used`, `n_pixels_per_class`

**Example invocations:**

```bash
# Fine-tuned encoder, all confidence tiers
python scripts/build_prototypes.py \
    --ckpt checkpoints/spvit_lrscale0005_best.pt \
    --out data/prototypes/proto_finetuned_all.npz

# MAE encoder, all tiers
python scripts/build_prototypes.py \
    --ckpt checkpoints/spatial_mae_128d_6l_best.pt \
    --out data/prototypes/proto_mae_all.npz

# High-confidence only (for comparison)
python scripts/build_prototypes.py \
    --ckpt checkpoints/spvit_lrscale0005_best.pt \
    --confidence_tiers High \
    --out data/prototypes/proto_finetuned_high.npz
```

## Component: `scripts/classify_tile_prototype.py`

**Inputs:**

| Flag | Default | Description |
|------|---------|-------------|
| `--tile` | required | mrral `.img` tile path |
| `--proto_a` | required | Primary prototype `.npz` (used for `--save_probs` output) |
| `--proto_b` | None | Second prototype `.npz` for comparison figure only |
| `--save_probs` | None | Output `(H,W,5)` similarity `.npz` |
| `--supervised_probs` | None | Existing supervised `.npz` for 3rd argmax panel in figure |
| `--out` | `reports/fig_prototype_<tile>.png` | Figure output path |
| `--batch_size` | 512 | Inference batch size |

**Inference:**

1. Load tile raster + valid mask via `load_tile()` from `classify_tile_supervised.py`
2. Load encoder from proto_a metadata checkpoint (same detection logic as `build_prototypes.py`)
3. Embed all valid tile pixels in batches: apply `normalize_patches()`, run encoder, extract center token at index `patch_size**2 // 2 + 1` → `(H*W, 128)`
4. L2-normalize → cosine similarity: `(H*W, 128) · prototypes(5, 128)ᵀ` → `(H*W, 5)`
5. Clip to `[0, 1]`, reshape to `(H, W, 5)`, set invalid pixels to 0.0
6. Save as `.npz` with keys `probs`, `valid_mask`, `transform`, `crs_wkt` (proto_a only)
7. If proto_b supplied: repeat steps 2–5 with proto_b encoder

**Comparison figure** (when `--proto_b` supplied), using matplotlib GridSpec:

- **Top section** — 1 row × n_panels: argmax dominant-class maps (proto_a | proto_b | supervised if provided), colored by `CLASS_COLORS` from `fig_style.py`
- **Bottom section** — 2 rows × 5 columns: per-class cosine similarity heatmaps, row 0 = proto_a, row 1 = proto_b
- Row labels from `encoder_ckpt` basename stored in each prototype's metadata; column headers = mineral class names

## Data

- `data/prototypes/` — added to `.gitignore` (derived from gitignored checkpoints)
- Prototype `.npz` schema: `prototypes (5, 128)`, `class_names`, `encoder_ckpt`, `confidence_tiers_used`, `splits_used`, `n_pixels_per_class`
- Similarity `.npz` output: `probs (H,W,5)`, `valid_mask`, `transform`, `crs_wkt` — identical schema to supervised probs

## Tests

**`tests/test_build_prototypes.py`:**

- `test_prototype_shape` — output `prototypes` array is `(5, 128)`
- `test_prototype_l2_normalized` — each prototype has unit L2 norm (within 1e-6)
- `test_confidence_tier_filtering` — High-only prototypes differ from all-tiers prototypes given synthetic data with different per-tier embeddings
- `test_n_pixels_per_class_metadata` — `n_pixels_per_class` counts match the filtered hard-positive rows in synthetic parquet
- `test_zero_pixels_raises` — raises `ValueError` naming the class when no hard-positive pixels remain after filtering

**`tests/test_classify_tile_prototype.py`:**

- `test_cosine_similarity_shape` — output is `(H, W, 5)`
- `test_cosine_similarity_range` — all values in `[0, 1]` after clipping
- `test_perfect_similarity` — L2-normalized query == prototype → similarity = 1.0
- `test_invalid_pixels_masked` — pixels outside valid_mask → 0.0 in output

## Downstream Integration

The `(H, W, 5)` similarity `.npz` from `--save_probs` is a drop-in replacement for the supervised probs. Thresholds must be recomputed from the new rasters:

```bash
python scripts/compute_global_thresholds.py \
    --probs /tmp/t0434_proto_probs.npz /tmp/t0435_proto_probs.npz \
    --out config/vectroscopy_thresholds_proto.json

python scripts/vectorize_tile_minerals.py \
    --tile /mnt/mrdr/mc26/t0435_mrral_40s323_0327_4.img \
    --probs /tmp/t0435_proto_probs.npz \
    --thresholds config/vectroscopy_thresholds_proto.json \
    --out data/vector/t0435_proto_mineral_map.gpkg
```

For `plot_vector_mineral_maps.py`, update the `TILES` list to point to the proto GeoPackages.

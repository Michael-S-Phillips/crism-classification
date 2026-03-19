# Vectroscopy Integration Design

**Date:** 2026-03-19
**Status:** Approved
**Scope:** Test on T0435 and T0434; foundation for Mars Chart–scale global mapping

---

## Goal

Produce vector mineral map products from the fine-tuned SpatialSpectralClassifier by feeding
per-class probability rasters into Vectroscopy. Output is one GeoPackage per tile with four
mineral layers (olivine, LCP, HCP, plagioclase), each polygon carrying a model-driven confidence
tier and zonal statistics. Intended for a scientific publication.

---

## Pipeline Overview

Three sequential stages:

```
mrral tile
    │
    ▼
[Inference]  classify_tile_supervised.py
    │  → probs .npy  (H, W, 4) per tile
    ▼
[Calibration]  compute_global_thresholds.py
    │  → config/vectroscopy_thresholds.json  (percentile thresholds per class)
    ▼
[Vectorization]  vectorize_tile_minerals.py
    │  → data/vector/{tile_id}_mineral_map.gpkg  (4 layers, confidence + zonal stats)
```

---

## Stage 1 — Inference

**Script:** `scripts/classify_tile_supervised.py` (existing, extended)

- Input: mrral `.img` tile path, classifier checkpoint
- Output: `(H, W, 4)` float32 probability raster saved as `.npy` alongside the tile results
- Classes in order: olivine (0), lcp (1), hcp (2), plagioclase (3)
- The "other" class (index 4) is excluded from vector output
- Per-patch normalization is applied before inference (zero-mean unit-variance per 7×7 patch)
- Saves `{tile_id}_probs.npy` to a configurable output directory

**Test tiles:**
- T0435: `/mnt/mrdr/mc26/t0435_mrral_40s323_0327_4.img`
- T0434: `/mnt/mrdr/mc26/t0434_mrral_40s318_0327_4.img`

---

## Stage 2 — Global Threshold Calibration

**Script:** `scripts/compute_global_thresholds.py`

- Input: list of tile prob `.npy` files (or directory glob)
- For each mineral class, pool all valid-pixel probabilities across all input tiles
- Compute the 33rd, 67th, and 90th percentiles of each class's pooled distribution
- These define three confidence tiers:
  - **Tier 1 (low):** prob ≥ 33rd percentile
  - **Tier 2 (medium):** prob ≥ 67th percentile
  - **Tier 3 (high):** prob ≥ 90th percentile
- Output: `config/vectroscopy_thresholds.json`

```json
{
  "generated": "2026-03-19",
  "tiles_used": ["T0434", "T0435"],
  "percentiles": [33, 67, 90],
  "thresholds": {
    "olivine":     [0.28, 0.41, 0.57],
    "lcp":         [0.82, 0.91, 0.96],
    "hcp":         [0.04, 0.09, 0.18],
    "plagioclase": [0.03, 0.08, 0.15]
  },
  "morphology": {
    "median_filter_size": 3,
    "median_filter_iterations": 1,
    "sieve_min_pixels": 9,
    "majority_filter_iterations": 3,
    "simplify_tolerance_pixels": 1
  }
}
```

The thresholds JSON is the sole calibration artefact — re-running with more tiles updates it without
changing any other code.

---

## Stage 3 — Vectorization

**Script:** `scripts/vectorize_tile_minerals.py`

- Input: tile `.img` path, thresholds JSON, pre-saved probs `.npy` (optional; re-runs inference if absent)
- For each mineral class (olivine, lcp, hcp, plagioclase):
  1. Load the class probability raster `(H, W)`
  2. Call `Vectroscopy.from_array(array, thresholds, crs, transform, name=mineral)`
  3. Apply morphological pipeline: median filter → threshold → majority filter → sieve → simplify
  4. Attach attributes to output polygons:
     - `confidence`: int 1/2/3 (low/medium/high)
     - `mineral`: string class name
     - Zonal statistics: `mean_prob`, `std_prob`, `min_prob`, `max_prob`, `median_prob`, `count_px`
  5. Write layer `{mineral}` to the output GeoPackage
- Output: `data/vector/{tile_id}_mineral_map.gpkg` with four layers

### Morphological Parameters (defaults, all CLI-tunable)

| Parameter | Default | Rationale |
|---|---|---|
| Median filter size | 3×3 | Smooths single-pixel noise pre-threshold |
| Median filter iterations | 1 | |
| Sieve min pixels | 9 | Removes sub-patch-size (<1.8 km²) speckle |
| Majority filter iterations | 3 | Fills holes, smooths blocky boundaries |
| Simplify tolerance | 1 px | Reduces vertices without shape loss |

---

## Output Structure

```
data/vector/
  T0434_mineral_map.gpkg    # layers: olivine, lcp, hcp, plagioclase
  T0435_mineral_map.gpkg
config/
  vectroscopy_thresholds.json
```

Each layer in the GeoPackage contains polygon features with:

| Column | Type | Description |
|---|---|---|
| `geometry` | Polygon/MultiPolygon | Vector shape in tile CRS (Mars IAU 2000 equirectangular) |
| `confidence` | int (1–3) | Model-driven tier: 1=low, 2=medium, 3=high |
| `mineral` | str | Class name |
| `threshold` | float | Lower probability bound for this polygon's tier |
| `mean_prob` | float | Mean classifier probability within polygon |
| `std_prob` | float | Std dev of probability within polygon |
| `min_prob` | float | Min probability within polygon |
| `max_prob` | float | Max probability within polygon |
| `median_prob` | float | Median probability within polygon |
| `count_px` | int | Pixel count within polygon |

---

## Validation (post-implementation, not in scope here)

After the two test tiles are produced, compare against T0434's expert-labelled GeoPackage
(which carries High/Moderate/Low confidence tiers) to assess correspondence between
model-driven confidence and human-assessed confidence. This serves as a validation result
for the publication.

---

## Scaling to Mars Chart (future)

- Inference and vectorization run tile-by-tile unchanged
- Re-run `compute_global_thresholds.py` across all chart tiles to update thresholds
- Add `merge_chart_vectors.py` to concatenate per-tile GeoPackages into one per-chart
  per-mineral GeoPackage (one file per mineral globally, tile ID as attribute column)
- No tile-boundary reconciliation required for polygons; boundary handling deferred to
  publication figure post-processing

---

## Dependencies

- `Vectroscopy` (github.com/Tahn04/Vectroscopy) — must be installed in `crism` conda env
- `rasterio`, `geopandas`, `numpy`, `torch`, `tqdm` — already present
- `scipy` — for morphological ops (already present)

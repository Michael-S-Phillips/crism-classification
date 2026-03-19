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
[Inference]  classify_tile_supervised.py  (extended with --save_probs)
    │  → {out_dir}/{tile_id}_probs.npz  keys: probs(H,W,4), valid_mask(H,W), transform, crs_wkt
    ▼
[Calibration]  compute_global_thresholds.py
    │  → config/vectroscopy_thresholds.json
    ▼
[Vectorization]  vectorize_tile_minerals.py
    │  → data/vector/{tile_id}_mineral_map.gpkg  (4 layers)
```

---

## Stage 1 — Inference

**Script:** `scripts/classify_tile_supervised.py` (existing, extended)

**Extension:** add `--save_probs PATH.npz` argument that saves the four mineral class
probabilities as a compressed numpy archive. Existing behaviour (the classification figure) is
unchanged when `--save_probs` is omitted.

**CLI:**
```bash
python scripts/classify_tile_supervised.py \
    --tile /path/to/t0435_mrral_40s323_0327_4.img \
    --ckpt checkpoints/spvit_lrscale001_best.pt \
    --save_probs /tmp/t0435_probs.npz
```

**Output `.npz` keys:**
- `probs`: `(H, W, 4)` float32 — probabilities for olivine (0), lcp (1), hcp (2), plagioclase (3)
- `valid_mask`: `(H, W)` bool — True where all 59 bands are valid
- `transform`: `(6,)` float64 — rasterio Affine coefficients `(a, b, c, d, e, f)`
- `crs_wkt`: str — tile CRS as WKT, from `rasterio.open(tile_path).crs.to_wkt()`

The "other" class (index 4) is excluded from the saved array.

`tile_id` is derived as `os.path.splitext(os.path.basename(tile_path))[0]` (e.g., `t0435_mrral_40s323_0327_4`).

**Test tiles:**
- T0435: `/mnt/mrdr/mc26/t0435_mrral_40s323_0327_4.img`
- T0434: `/mnt/mrdr/mc26/t0434_mrral_40s318_0327_4.img`

---

## Stage 2 — Global Threshold Calibration

**Script:** `scripts/compute_global_thresholds.py`

**CLI:**
```bash
python scripts/compute_global_thresholds.py \
    --probs /tmp/t0434_probs.npz /tmp/t0435_probs.npz \
    --out config/vectroscopy_thresholds.json \
    --percentiles 33 67 90
```

**Logic:**
1. For each input `.npz`, load `probs` `(H, W, 4)` and `valid_mask` `(H, W)`
2. For each mineral class `c` in `[0, 1, 2, 3]`, collect `probs[:, :, c][valid_mask]` (1-D array of valid-pixel probabilities)
3. Pool these vectors across all input tiles into one array per class
4. Compute `np.percentile(pooled, [33, 67, 90])` → three threshold floats `[t1, t2, t3]`
5. Write output JSON

**Confidence tier mapping:** Vectroscopy generates one set of polygons per threshold value —
all pixels above `t1`, all above `t2`, all above `t3`. Polygons across tiers **overlap** (tier-3
areas are a strict subset of tier-2, which is a strict subset of tier-1). Each polygon's
`Threshold` column holds the float threshold used to generate it.

Tier assignment in Stage 3: rank the unique Threshold values ascending → rank 1 = `confidence 1`
(low, ≥33rd pctile), rank 2 = `confidence 2` (medium, ≥67th), rank 3 = `confidence 3` (high,
≥90th). Pixels that do not exceed `t1` produce no polygon and are absent from the output.

**Output JSON schema** (all values illustrative; script fills from data):
```json
{
  "generated": "2026-03-19",
  "tiles_used": ["t0434_mrral_40s318_0327_4", "t0435_mrral_40s323_0327_4"],
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
    "simplify_tolerance_meters": 200
  }
}
```

The `morphology` key is **documentation only** — it records the morphological parameter defaults
used during the calibration run for reproducibility. Stage 3 reads morphological parameters
exclusively from its own CLI arguments and does not read `morphology` from this JSON.

The JSON is the sole calibration artefact. Re-running with more tiles updates it without changing
any other code.

---

## Stage 3 — Vectorization

**Script:** `scripts/vectorize_tile_minerals.py`

**CLI:**
```bash
python scripts/vectorize_tile_minerals.py \
    --tile /path/to/t0435_mrral_40s323_0327_4.img \  # always required
    --probs /tmp/t0435_probs.npz \       # optional; if absent, runs inference inline
    --ckpt checkpoints/spvit_lrscale001_best.pt \  # required only when --probs is absent
    --thresholds config/vectroscopy_thresholds.json \
    --out data/vector/t0435_mrral_40s323_0327_4_mineral_map.gpkg \
    --median_size 3 \
    --median_iter 1 \
    --sieve_px 9 \
    --majority_iter 3
```

`--tile` is **always required** (argparse `required=True`). It provides both the tile path for
inline inference and the authoritative CRS/transform (falling back to the `.npz` fields only
if the `.img` file is unavailable). `--ckpt` is required if and only if `--probs` is absent;
the script validates this at startup and exits with a clear error message if violated.

If `--probs` is absent, the script imports and calls `load_tile`, `load_classifier`, and
`run_supervised` from `classify_tile_supervised` directly (not via subprocess). The inline
probs are not saved to disk.

**CRS and transform:**
Always loaded from `rasterio.open(tile_path)` at script startup:
```python
with rasterio.open(tile_path) as src:
    input_crs = src.crs                 # rasterio.crs.CRS
    input_transform = src.transform     # rasterio.transform.Affine
```
If `--probs` is supplied, the `.npz` `transform` array (6 coefficients) is only used as a
consistency check, not as the primary source. Reconstructing an Affine from the `.npz` if
needed: `from rasterio.transform import Affine; t = Affine(*npz['transform'])` (rasterio
coefficient order: `a, b, c, d, e, f` matching `(col_scale, col_shear, col_off, row_shear,
row_scale, row_off)`).

CRISM MRDR tiles use a projected equirectangular CRS in metres (Mars IAU 2000); simplify
tolerance of 200 m in CRS units is correct for all MRDR tiles.

**Vectroscopy API (verified from source at github.com/Tahn04/Vectroscopy):**

```python
# Install: git clone https://github.com/Tahn04/Vectroscopy.git /opt/Vectroscopy
# (no setup.py/pyproject.toml — pip install does not work)
import sys; sys.path.insert(0, '/opt/Vectroscopy/src')
import core.vectroscopy as vp_module

gdf = vp_module.Vectroscopy.from_array(
    array=prob_2d_nan,        # (H, W) float32, nodata pixels set to np.nan
    thresholds=[t1, t2, t3],  # list of pre-computed float absolute values
    crs=rasterio_crs,         # rasterio.crs.CRS object (converted to WKT internally)
    transform=rasterio_affine,# rasterio.transform.Affine object (converted to GDAL tuple)
    name=mineral_name,        # str, used as temp-file prefix
).vectorize()
# Returns GeoDataFrame in-memory (driver defaults to "pandas").
# 'Threshold' column contains the float threshold value for each polygon's level.
# IMPORTANT: Vectroscopy reprojects output to geographic CRS (degrees) by default.
# Must reproject back to tile projected CRS immediately after:
gdf = gdf.to_crs(input_crs)   # input_crs = rasterio.open(tile_path).crs
```

**Nodata handling:** Set `prob_2d[~valid_mask] = np.nan` before passing to `from_array`.

**Per-class processing loop:**
```python
CLASS_NAMES = ['olivine', 'lcp', 'hcp', 'plagioclase']
thresholds_cfg = json.load(open(thresholds_json))

for ci, mineral in enumerate(CLASS_NAMES):
    prob_2d = probs[:, :, ci].copy().astype(np.float32)
    prob_2d[~valid_mask] = np.nan
    t1, t2, t3 = thresholds_cfg['thresholds'][mineral]

    gdf = Vectroscopy.from_array(
        array=prob_2d, thresholds=[t1, t2, t3],
        crs=crs, transform=transform, name=mineral,
    ).vectorize()

    if gdf.empty:
        continue

    # Reproject from Vectroscopy's default geographic CRS back to tile projected CRS
    gdf = gdf.to_crs(input_crs)

    # Map Vectroscopy 'Threshold' float column → confidence tier integer.
    # 'Threshold' contains the actual float values from [t1, t2, t3].
    # Map by rank (sorted ascending) to avoid floating-point equality issues:
    unique_t = sorted(gdf['Threshold'].unique())
    tier_map = {v: i + 1 for i, v in enumerate(unique_t)}
    gdf['confidence'] = gdf['Threshold'].map(tier_map)
    gdf['mineral'] = mineral

    # Zonal statistics: compute from prob_2d over each polygon
    # Use rasterstats.zonal_stats with the prob_2d array and transform
    stats = rasterstats.zonal_stats(
        gdf.geometry, prob_2d,
        affine=transform, stats=['mean', 'std', 'min', 'max', 'median', 'count'],
        nodata=np.nan,
    )
    stats_df = pd.DataFrame(stats).rename(columns={
        'mean': 'mean_prob', 'std': 'std_prob', 'min': 'min_prob',
        'max': 'max_prob', 'median': 'median_prob', 'count': 'count_px',
    })
    gdf = pd.concat([gdf.reset_index(drop=True), stats_df], axis=1)

    # Drop Vectroscopy internal columns except geometry, keep our columns
    keep = ['geometry', 'confidence', 'mineral', 'Threshold',
            'mean_prob', 'std_prob', 'min_prob', 'max_prob', 'median_prob', 'count_px']
    gdf = gdf[[c for c in keep if c in gdf.columns]].rename(columns={'Threshold': 'threshold'})

    # Simplify geometry AFTER zonal stats so stored geometry matches the stats computed.
    # 200 m tolerance in tile projected CRS (metres).
    gdf['geometry'] = gdf['geometry'].simplify(tolerance=200, preserve_topology=True)

    gdf.to_file(out_gpkg, layer=mineral, driver='GPKG')
```

### Morphological Parameters (defaults, all CLI-tunable)

| Parameter | Default | Rationale |
|---|---|---|
| Median filter size | 3×3 | Smooths single-pixel noise before thresholding |
| Median filter iterations | 1 | One pass sufficient for patch-scale noise |
| Sieve min pixels | 9 | Removes patches < one 7×7 patch (~1.8 km²) |
| Majority filter iterations | 3 | Fills holes and smooths blocky boundaries |
| Simplify tolerance | 200 m | ~1 pixel; reduces vertices without shape loss |

Vectroscopy applies majority filter and sieve internally when configured via its pipeline.
The median filter is applied to `prob_2d` before passing to `from_array` using
`scipy.ndimage.median_filter`.

---

## Output Structure

```
data/vector/
  t0434_mrral_40s318_0327_4_mineral_map.gpkg   # layers: olivine, lcp, hcp, plagioclase
  t0435_mrral_40s323_0327_4_mineral_map.gpkg
config/
  vectroscopy_thresholds.json
```

Each layer contains polygon features with:

| Column | Type | Description |
|---|---|---|
| `geometry` | Polygon/MultiPolygon | Shape in tile CRS (Mars IAU 2000 equirectangular, metres) |
| `confidence` | int (1–3) | 1=low (>33rd pctile), 2=medium (>67th), 3=high (>90th) |
| `mineral` | str | Class name (e.g. `"olivine"`) |
| `threshold` | float | Lower probability bound for this polygon's tier |
| `mean_prob` | float | Mean classifier probability within polygon |
| `std_prob` | float | Std dev of probability within polygon |
| `min_prob` | float | Min probability within polygon |
| `max_prob` | float | Max probability within polygon |
| `median_prob` | float | Median probability within polygon |
| `count_px` | int | Pixel count within polygon |

---

## Validation (post-implementation, not in scope here)

Compare output against T0434's expert-labelled GeoPackage (High/Moderate/Low confidence
categories) to assess correspondence between model-driven tiers and human-assessed confidence.
This is a validation result for the publication.

---

## Scaling to Mars Chart (future)

- Inference and vectorization run tile-by-tile unchanged
- Re-run `compute_global_thresholds.py` across all chart tiles to update the JSON
- Add `merge_chart_vectors.py` to concatenate per-tile GeoPackages into one per-mineral
  file (tile ID as attribute column)

---

## Dependencies

- `Vectroscopy` — no `setup.py`/`pyproject.toml`; install via git clone:
  ```bash
  git clone https://github.com/Tahn04/Vectroscopy.git /opt/Vectroscopy
  # In scripts: sys.path.insert(0, '/opt/Vectroscopy/src')
  # Import: import core.vectroscopy as vp_module
  ```
- `rasterstats` — `pip install rasterstats` into `crism` env
- `rasterio`, `geopandas`, `numpy`, `torch`, `scipy`, `tqdm`, `pandas` — already in `crism` env

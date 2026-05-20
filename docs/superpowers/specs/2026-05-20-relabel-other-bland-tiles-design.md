# Relabel "Other" Using 8 Hand-Picked Bland Tiles — Design Spec

**Date:** 2026-05-20
**Status:** Design (not yet implemented)
**Author:** initiated by user; spec by Claude
**Related:** `data/label_parser.py`, `scripts/build_mrral_dataset.py`, `data/dataset.py:_collapse_labels`

## 1. Goal

Replace the existing 677K-pixel "other" label set — currently sourced from `"Other (...)"`-categorized GPKG polygons in mineral-bearing tiles — with ~900K pixels drawn from 8 hand-picked tiles known to be dust-covered / bland surface. This gives the v3 classifier a representative "non-mafic-surface" rejection class so it learns to distinguish mafic minerals from generic Martian dust.

The 8 tiles are:

| tile_id | location (lat/lon) | MC dir |
|---|---|---|
| t1241 | 20°N, 33°E | mc12 |
| t1242 | 20°N, 38°E | mc12 |
| t1243 | 20°N, 43°E | mc12 |
| t1280 | 20°N, 228°E | mc09 |
| t1313 | 25°N, 33°E | mc12 |
| t1314 | 25°N, 38°E | mc12 |
| t1315 | 25°N, 43°E | mc12 |
| t1336 | 25°N, 148°E | mc15 |

User selected these as "pretty much all dusty/bland". Six are in MC12 (a contiguous Arabia Terra block); the other two are spatially separated, providing geographic diversity.

## 2. Background and rationale

The current "other" pixel set has 677K rows in `data/mrral_pixels.parquet`. The top contributors are:

| tile_id | other-labeled pixels |
|---|---|
| t0435 | 69,564 |
| t0886 | 59,132 |
| t0576 | 38,763 |
| t0433 | 32,076 |
| t0818 | 31,949 |

All these are tiles that ALSO have mineral labels — meaning "other" pixels in the current set are typically adjacent to mafic mineral polygons within the same scene. They represent "mineral-scene-but-not-this-mineral" rather than "generic bland Martian surface". A classifier trained against this "other" class learns to discriminate within-scene boundaries, not to reject dust-covered terrain entirely.

The new label scheme tells the model: "if a pixel looks like Arabia-Terra dust, classify it as `other`, not as a mineral." This should improve the unclassified-vs-noisy-mafic boundary on global mosaic products like the recent MC13 run (where 16.8% of valid pixels currently end up "unclassified" because no mineral threshold is met).

## 3. Design summary

1. **New GPKGs:** Author 8 single-polygon GPKGs at `/mnt/mrdr/categorized_mineral_units/T{1241,1242,1243,1280,1313,1314,1315,1336}.gpkg`. Each contains one polygon covering the full tile extent in the tile's native equirectangular CRS, with `Category = "Other (High)"`, `Mineral ID 1 = "bland"`.

2. **Label parser change:** Modify the category → label mapping (in `data/label_parser.py`) so existing `"Other (...)"` polygons in *non-bland* tiles do NOT contribute the `other` label. The bland-tile whitelist (the 8 tile IDs above) is the gate. Polygons in other tiles whose Category starts with `"Other"` are simply dropped — no row added to the dataset for those pixels (unless they're independently mineral-labeled elsewhere in the same scene).

3. **Parquet rebuild:** Re-run `scripts/build_mrral_dataset.py` from scratch. The new parquet has ~13M rows tagged `other = 1` from the 8 bland tiles.

4. **Subsample:** Post-process the parquet to randomly subsample those 13M rows down to 113K per tile (seeded with `seed = 42` for reproducibility), yielding ~900K total "other" pixels. Other tiles' rows are untouched.

5. **Patch cache rebuild:** Re-run `scripts/cache_mrral_patches.py` to regenerate the labeled patch cache (`data/patch_cache/mrral_{train,val,test}_patches_p7.npy`) from the new parquet.

6. **(Out of scope) Finetune sweep:** The new patch cache feeds a future HPC finetune sweep against denoising/SPEND/March encoders.

## 4. New GPKG layout

### 4.1 Per-GPKG schema

Each of the 8 new GPKGs gets a single row matching the schema of existing `T*.gpkg` files in `/mnt/mrdr/categorized_mineral_units/`:

| column | value |
|---|---|
| Polygon Number | 0 |
| Color | `#aaaaaa` |
| Number of Points | (filled by GeoPandas) |
| Denominator | null |
| Template | null |
| Mineral ID 1 | `"bland"` |
| Mineral ID 2 | null |
| Mineral ID 3 | null |
| Mineral ID 4 | null |
| wvl | null |
| Spectrum Mean | null |
| params | null |
| Parameters Mean | null |
| Best Denom ID | null |
| Ratio Spectrum | null |
| **Category** | **`"Other (High)"`** |
| geometry | Polygon — full tile extent in the tile's native CRS |

The downstream label parser uses `Category` to assign label-column values. The other columns are preserved for parity with existing GPKGs but are unused.

### 4.2 Polygon geometry

For each tile, the polygon is a rectangle covering the full mrral tile extent in the tile's native equirectangular CRS:

```python
import rasterio
with rasterio.open(mrral_path) as src:
    bounds = src.bounds   # (left, bottom, right, top) in tile CRS
    crs = src.crs
poly = shapely.geometry.box(*bounds)
```

The polygon covers the entire pixel grid including nodata regions. The downstream `extract_mrral_pixels_from_pair` already drops nodata pixels at extract time, so this is fine.

Why the full extent vs the valid-pixel mask? Two reasons: (1) bounds + box() is simpler than computing a valid-pixel polygon; (2) it matches how existing GPKGs work — they cover whatever the analyst hand-drew, and the build pipeline filters nodata at pixel level.

### 4.3 Authoring script

A new one-shot script `scripts/build_bland_other_gpkgs.py` reads each mrral tile, builds the GPKG, and writes it to `/mnt/mrdr/categorized_mineral_units/`. Idempotent — skips files that already exist.

## 5. Label parser change

The label parser in `data/label_parser.py` builds a `Category → label-columns` mapping. The change:

- Existing categories `"Other (High|Moderate|Low)"` are no longer mapped to the `other` label column **by default**.
- New rule: `other = 1` is assigned only when EITHER `Mineral ID 1 == "bland"` (the new GPKG marker) OR the source tile_id is in the bland-tile whitelist:
  ```python
  BLAND_TILES = {'t1241', 't1242', 't1243', 't1280', 't1313', 't1314', 't1315', 't1336'}
  ```

Both conditions are belt-and-suspenders — either one suffices. Polygons in other tiles whose Category starts with `"Other"` contribute nothing to the label set (they're skipped during extraction).

This is the minimum invasive change. Edit happens in `data/label_parser.py` (or wherever the category-to-label mapping is centralized). The rest of the build pipeline is unchanged.

## 6. Parquet rebuild + subsampling

Run `scripts/build_mrral_dataset.py` after the GPKGs and label-parser change land. Expected output before subsampling:

- ~13M new rows tagged `other = 1` from the 8 bland tiles (~1.6M valid pixels each).
- Existing tiles' mineral labels unchanged (olivine_t1: 431K, lcp: 546K, hcp: 434K, plag: 353K).
- The previous 677K "other" rows are gone (or have `other = 0` after the parser change — depending on whether they had concurrent mineral labels).

Then run a subsampling step. Either as the tail of `build_mrral_dataset.py` or as a separate `scripts/subsample_other_labels.py`. The logic:

```python
SAMPLE_PER_TILE = 113_000
SEED = 42

bland_rows = df['tile_id'].isin(BLAND_TILES) & (df['other'] == 1)
sampled_idx = []
for tid in BLAND_TILES:
    tile_idx = df.index[bland_rows & (df['tile_id'] == tid)]
    rng = np.random.default_rng(SEED + hash(tid) % 1000)
    keep = rng.choice(tile_idx, size=min(SAMPLE_PER_TILE, len(tile_idx)),
                      replace=False)
    sampled_idx.extend(keep)

# Keep all non-bland-tile rows + subsampled bland rows
df_final = pd.concat([df[~bland_rows], df.loc[sampled_idx]], ignore_index=True)
```

Per-tile RNG seeding ensures reproducibility AND independence across tiles.

Target final counts (approximate):

| class | pixels |
|---|---|
| olivine (collapsed t1+t2) | ~900K |
| lcp | ~550K |
| hcp | ~430K |
| plagioclase | ~350K |
| other | ~900K (new) |

## 7. Patch cache rebuild

`scripts/cache_mrral_patches.py` regenerates `data/patch_cache/mrral_{train,val,test}_patches_p7.npy` from the new parquet. ~10 min on local disk. The patch cache is the actual input to the finetune trainer; until it's regenerated, downstream training would still use the OLD labels.

The 8 new bland tiles must be discoverable for patch extraction. The cache builder uses `find_mrral_pairs(gpkg_dir, data_root)` which crawls the GPKG directory — the new GPKGs from §4 will be picked up automatically.

## 8. Testing strategy

Lightweight — this is a data pipeline change, not a model change. Tests in `tests/test_relabel_other_bland.py`:

1. **GPKG schema test:** generate one bland-tile GPKG and assert it has the expected columns + a single row with `Category == "Other (High)"`, `Mineral ID 1 == "bland"`.
2. **Label parser whitelist test:** mock a row from a bland tile and a row from a non-bland tile, both with `Category = "Other (High)"`. Assert the bland-tile row produces `other = 1`; the non-bland row produces `other = 0`.
3. **Subsampling test:** small synthetic dataframe with > SAMPLE_PER_TILE rows in 2 bland tiles + some non-bland rows. After subsampling, each bland tile has exactly SAMPLE_PER_TILE rows; non-bland rows are untouched; seed produces deterministic indices.
4. **Smoke test:** end-to-end on a 1-tile subset (e.g., just t1241) — author its GPKG, run a 1-tile build, assert the parquet has ~1.6M rows with `tile_id == 't1241'` AND `other == 1`.

Manual / integration check (not automated):

5. After the full rebuild, run `python -c "import pandas as pd; df = pd.read_parquet('data/mrral_pixels.parquet'); print(df[['olivine_t1','olivine_t2','lcp','hcp','plagioclase','other']].sum())"` and confirm `other ≈ 900K`, other class counts unchanged.

## 9. Success criteria

- All 8 bland-tile GPKGs exist in `/mnt/mrdr/categorized_mineral_units/`.
- Rebuilt parquet has `other ≈ 900K` total (within 1% of `8 × 113K = 904K`).
- Existing tile rows: mineral class counts unchanged ± 1%.
- Patch cache rebuilds without errors.
- The four unit tests in `tests/test_relabel_other_bland.py` pass.

Downstream classifier improvement is a future hypothesis to test, not a success criterion of this work. The minimum bar: pipeline produces the expected data without errors.

## 10. Risks and mitigations

| risk | mitigation |
|---|---|
| One of the 8 tiles is not in fact uniformly bland — some pixels are mineral-bearing | User picked these by hand. If a downstream finetune shows the model now misclassifies some mineral pixels as "other", we can drop the offending tile from `BLAND_TILES` and rebuild. |
| Polygon-covers-everything causes the build pipeline to extract too many nodata-filtered "other" pixels | `extract_mrral_pixels_from_pair` already drops nodata. Verified in §4.2. |
| The label parser change inadvertently affects tier handling (Moderate / Low) for existing tiles | The parser change targets categories starting with `"Other"` specifically; mineral categories untouched. Unit test #2 covers this. |
| Subsampling is per-tile-rng but the build script might re-order rows non-deterministically | The subsampling explicitly seeds + selects by index, not order. Reproducible. |
| Patch cache `.npy` shapes change (new row count) — downstream slurm scripts assume specific sizes | The cache files use dynamic shape per built dataset; downstream training reads via shape inspection. No fixed-shape assumptions. |
| The existing 677K "other" pixels in `data/mrral_pixels.parquet` get dropped — old training runs are no longer reproducible from current parquet | Keep a backup: `cp data/mrral_pixels.parquet data/mrral_pixels.pre-bland.parquet` before rebuild. Allows reverting if the new labels regress downstream performance. |

## 11. Out of scope

- Modifying the existing source GPKGs in `/mnt/mrdr/categorized_mineral_units/`. Source data unchanged.
- Changing the 5-class architecture (still olivine / lcp / hcp / plagioclase / other).
- Running the finetune sweep against the new labels. That's a separate HPC job.
- Hellas / sup-GPKG re-integration (already in the parquet, will stay).
- Cleaning up stale `config.yaml` confidence-weight values (separate cleanup).
- Updating wiki / Methodology Log to reflect the new "other" definition (do that as a follow-up after the new labels are validated downstream).

## 12. Open questions resolved during brainstorming

1. **Replace or append existing "other"?** Replace entirely. The mineral-adjacent "Other" polygons don't represent dust; they represent "within-scene leftovers".
2. **How many "other" pixels?** Match the largest mineral class: ~113K per tile × 8 tiles ≈ 900K.
3. **Confidence tier?** High (weight 1.0). User hand-picked these tiles knowing they're dust-covered.
4. **Integration mechanism?** New GPKGs in `/mnt/mrdr/categorized_mineral_units/` + label parser whitelist — uses the existing build pipeline.

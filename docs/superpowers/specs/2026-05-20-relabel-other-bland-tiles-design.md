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

## 5. Pipeline change — gate at the extraction seam

A code review caught that `data/label_parser.py` is **not** the right seam: `parse_category()` is pure-functional and has no access to `tile_id`. The correct gate is the existing `other_polygon_ids` parameter on `extract_mrral_pixels_from_pair` (see `data/extract_pixels.py:97-131`), which already filters "Other" polygons by polygon index when set.

The change lives in **`scripts/build_mrral_dataset.py`** (which currently doesn't pass this parameter — see line 39). We add per-tile gating logic:

```python
BLAND_TILES = {'t1241', 't1242', 't1243', 't1280', 't1313', 't1314', 't1315', 't1336'}

for tile_id, gpkg_path, mrral_path in pairs:
    if tile_id in BLAND_TILES:
        # Allow ALL "Other" polygons (the whole tile is bland-labeled).
        other_polygon_ids = None
    else:
        # Block ALL "Other" polygons in non-bland tiles.
        other_polygon_ids = set()   # empty set → no "Other" polygons pass

    records = extract_mrral_pixels_from_pair(
        tile_id, mrral_path, gpkg_path,
        other_polygon_ids=other_polygon_ids,
    )
```

This is the entire label-policy change. `label_parser.py` is unchanged. The `Mineral ID 1 = "bland"` value in the new GPKGs is now purely descriptive metadata — it's not used as a gate. (Kept anyway because it helps a human reading the GPKG understand provenance.)

Effect on pre-existing rows with multi-label cells: a pixel that was previously labeled `(olivine=1, other=1)` from two overlapping polygons in t0435 will lose its `other=1` flag (because t0435 isn't in BLAND_TILES → empty `other_polygon_ids` → "Other" polygons drop). Its olivine label survives. This is the intended behavior.

## 6. Parquet rebuild + subsampling

Run `scripts/build_mrral_dataset.py` after the GPKGs and label-parser change land. Expected output before subsampling:

- ~13M new rows tagged `other = 1` from the 8 bland tiles (~1.6M valid pixels each).
- Existing tiles' mineral labels unchanged (olivine_t1: 431K, lcp: 546K, hcp: 434K, plag: 353K).
- The previous 677K "other" rows are gone (or have `other = 0` after the parser change — depending on whether they had concurrent mineral labels).

Then run a subsampling step. As the tail of `build_mrral_dataset.py` (before the parquet write). The logic:

```python
SAMPLE_PER_TILE = 113_000
SEED = 42
BLAND_TILES_ORDERED = [
    't1241', 't1242', 't1243', 't1280',
    't1313', 't1314', 't1315', 't1336',
]   # fixed enumeration order → deterministic per-tile seeds

bland_rows = df['tile_id'].isin(BLAND_TILES_ORDERED) & (df['other'] == 1)
sampled_idx = []
for i, tid in enumerate(BLAND_TILES_ORDERED):
    tile_idx = df.index[bland_rows & (df['tile_id'] == tid)]
    rng = np.random.default_rng(SEED + i)   # deterministic across Python runs
    keep = rng.choice(tile_idx, size=min(SAMPLE_PER_TILE, len(tile_idx)),
                      replace=False)
    sampled_idx.extend(keep)

# Keep all non-bland-tile rows + subsampled bland rows
df_final = pd.concat([df[~bland_rows], df.loc[sampled_idx]], ignore_index=True)
```

Per-tile RNG seeding uses fixed enumeration index (`SEED + i`) — reproducible across Python invocations regardless of `PYTHONHASHSEED`. (Using `hash(tid)` would have been salted and non-reproducible.)

Target final counts (approximate; exact unless a tile has < 113K valid pixels):

| class | pixels |
|---|---|
| olivine (collapsed t1+t2) | ~900K |
| lcp | ~550K |
| hcp | ~430K |
| plagioclase | ~350K |
| other | ~904K (new, = 8 × 113K) |

### 6.1 Split assignment for the new bland-tile rows

`build_mrral_dataset.py` assigns train/val/test splits via a **left-join** on `(tile_id, polygon_id, pixel_row, pixel_col)` against `data/pixels.parquet` (the mrrsu parquet), with `train` as the fallback for unmatched rows.

The 8 bland tiles have **zero rows in `data/pixels.parquet`** (confirmed) → all 904K "other" rows would default to `train`, leaving val/test free of the "other" class entirely. That would silently break per-class validation.

The fix: explicitly assign splits to the bland-tile rows BEFORE the merge (or immediately after, overwriting the `train` default for those rows). Random per-pixel within each tile, stratified to roughly 70/15/15 train/val/test matching the project convention (`config.yaml:split`). Same fixed-enumeration RNG seeding as §6:

```python
SPLIT_FRACS = {'train': 0.70, 'val': 0.15, 'test': 0.15}

for i, tid in enumerate(BLAND_TILES_ORDERED):
    mask = (df_final['tile_id'] == tid)
    n = int(mask.sum())
    rng = np.random.default_rng(SEED + 100 + i)
    splits = rng.choice(
        list(SPLIT_FRACS.keys()),
        size=n,
        p=list(SPLIT_FRACS.values()),
    )
    df_final.loc[mask, 'split'] = splits
```

Result: each bland tile contributes ~79K train + ~17K val + ~17K test = 113K rows. Aggregate val "other" pixel count: ~135K — comparable to mineral classes' val partitions.

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
| One of the 8 tiles is not in fact uniformly bland — some pixels are mineral-bearing | User picked these by hand. If a downstream finetune shows misclassification of mineral pixels as "other", drop the offending tile from `BLAND_TILES_ORDERED` and rebuild. |
| Polygon-covers-everything extracts nodata pixels | `extract_mrral_pixels_from_pair` already drops nodata at the pixel-read step. Confirmed in `data/extract_pixels.py`. |
| The extraction gate change inadvertently affects mineral labels for existing tiles | The gate (`other_polygon_ids=set()` for non-bland tiles) only blocks polygons whose category contains `"other"` (case-insensitive, see `extract_pixels.py:129`). Mineral-category polygons are not touched. Covered by unit test #2. |
| Subsampling reproducibility broken by Python's salted `hash()` | Use fixed enumeration order (`SEED + i`) instead of `hash(tid)`. Documented in §6. |
| `build_mrral_dataset.py` accumulates ~13M rows in memory before subsampling — ~7–10 GB peak RAM | If this causes OOM on the local box, modify the loop to subsample per-tile during extraction (cap at 113K per bland tile before extending `all_records`). Document as a follow-up fix; the local box has 64 GB so the first run should succeed. |
| The new GPKG CRS doesn't round-trip cleanly through GeoPandas → corrupt geometry on read | Author the GPKG with `crs = rasterio.open(mrral_path).crs` (string identity with what `extract_mrral_pixels_from_pair` expects). The `gdf.crs != raster_crs` check in extract_pixels.py becomes a no-op. Test #1 asserts `gdf.crs == expected_tile_crs` after read-back. |
| Existing patch cache `data/patch_cache/mrral_*_patches_p7.npy` overwritten — restoring just the parquet doesn't restore the cache | Back up both: `cp data/mrral_pixels.parquet data/mrral_pixels.pre-bland.parquet` AND `cp data/patch_cache/mrral_*_patches_p7.npy` to `*.pre-bland.npy`. Restore-from-backup procedure documented in §11. |
| New tiles get default `train` split because `pixels.parquet` doesn't have rows for them — val/test become "other"-free | Explicit split assignment for bland-tile rows BEFORE/AFTER the merge. See §6.1. |
| `find_mrral_pairs` doesn't pick up new GPKGs because of hard-coded list | Confirmed: `find_mrral_pairs` (extract_pixels.py:76-93) dynamically crawls the GPKG directory via glob. New `T*.gpkg` files are auto-discovered. |
| `build_bland_other_gpkgs.py` runs twice and silently re-uses bad first-run output (`skip if exists`) | Authoring script validates the file after write: reads it back via `gpd.read_file`, asserts row count = 1 and Category = "Other (High)" and CRS matches the source mrral. On re-run, validates the existing file the same way; only skips if it passes. |

## 10.1 Reversibility procedure

If the new labels regress downstream classifier performance and we need to revert:

```bash
cp data/mrral_pixels.pre-bland.parquet data/mrral_pixels.parquet
cp data/patch_cache/mrral_train_patches_p7.pre-bland.npy data/patch_cache/mrral_train_patches_p7.npy
cp data/patch_cache/mrral_val_patches_p7.pre-bland.npy   data/patch_cache/mrral_val_patches_p7.npy
cp data/patch_cache/mrral_test_patches_p7.pre-bland.npy  data/patch_cache/mrral_test_patches_p7.npy
# (Optional) Remove the 8 new GPKGs so a future rebuild reverts cleanly:
# rm /mnt/mrdr/categorized_mineral_units/T{1241,1242,1243,1280,1313,1314,1315,1336}.gpkg
```

Model checkpoints trained against the new labels are unaffected by the rollback — they continue to predict whatever they learned. If you want a "pre-bland" checkpoint, restore the labels and retrain.

## 11. Out of scope

- Modifying the existing source GPKGs in `/mnt/mrdr/categorized_mineral_units/`. Source data unchanged.
- Changing the 5-class architecture (still olivine / lcp / hcp / plagioclase / other).
- Running the finetune sweep against the new labels. That's a separate HPC job.
- Hellas / sup-GPKG re-integration (already in the parquet, will stay).
- Cleaning up stale `config.yaml` confidence-weight values (separate cleanup).
- Updating wiki / Methodology Log to reflect the new "other" definition (do that as a follow-up after the new labels are validated downstream).

## 12. Open questions resolved during brainstorming

1. **Replace or append existing "other"?** Replace entirely. The mineral-adjacent "Other" polygons don't represent dust; they represent "within-scene leftovers".
2. **How many "other" pixels?** Match the largest mineral class (collapsed olivine ≈ 900K): 113K per tile × 8 tiles ≈ 904K.
3. **Confidence tier?** High. Stored as `confidence_tier='High'` in the parquet. `data/dataset.py:_collapse_labels` maps this to a per-row weight of 1.0 at training time. (Note: the parquet's stored `confidence_weight` column is recomputed inside `_collapse_labels` and the in-code value 1.0/0.85/0.70 takes precedence over the parser's 1.0/0.5/0.25 — both end up correct for High tier; the discrepancy is a pre-existing inconsistency that's intentionally out of scope here.)
4. **Integration mechanism?** New GPKGs in `/mnt/mrdr/categorized_mineral_units/` + bland-tile gate in `build_mrral_dataset.py` via the existing `other_polygon_ids` parameter. Label parser unchanged.

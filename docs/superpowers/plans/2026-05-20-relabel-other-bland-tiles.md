# Relabel "Other" Using Bland Tiles — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 677K existing "other" pixels (from "Other"-categorized GPKG polygons in mineral-bearing tiles) with ~904K pixels sampled from 8 hand-picked dust-covered tiles, giving the classifier a real "non-mafic-surface" rejection class.

**Architecture:** Author 8 single-polygon GPKGs (one per bland tile, Category="Other (High)"). Modify `scripts/build_mrral_dataset.py` to (a) gate non-bland tiles' "Other" polygons via the existing `other_polygon_ids` parameter on `extract_mrral_pixels_from_pair`, (b) subsample bland-tile rows to 113K per tile, and (c) explicitly assign 70/15/15 train/val/test splits to bland-tile rows BEFORE the merge against the mrrsu parquet (otherwise all 904K rows default to `train`). Rebuild the labeled parquet + patch cache.

**Tech Stack:** Python, GeoPandas (for GPKG authoring), rasterio (tile metadata), pandas (parquet + subsampling), pytest.

**Spec:** `docs/superpowers/specs/2026-05-20-relabel-other-bland-tiles-design.md` (commit d30589c).

**Reference files:**
- `data/extract_pixels.py:97-131` — existing `other_polygon_ids` gate (no code change here; just consumed)
- `data/label_parser.py` — unchanged
- `scripts/build_mrral_dataset.py:18-65` — the build entry point that gets modified
- `data/dataset.py:_collapse_labels` — confidence-tier→weight mapping (unchanged; reads the parquet's `confidence_tier`)
- `/mnt/mrdr/categorized_mineral_units/T*.gpkg` — existing GPKG schema (reference only)

---

## File Structure

**New files:**
- `data/bland_tile_gpkg.py` — helper module that authors one GPKG given a single mrral tile path
- `scripts/build_bland_other_gpkgs.py` — CLI wrapper that loops over the 8 bland tiles + invokes the helper
- `tests/test_relabel_other_bland.py` — unit tests covering authoring, gating, subsampling, splits

**Modified files:**
- `scripts/build_mrral_dataset.py` — add `BLAND_TILES_ORDERED`, per-tile `other_polygon_ids` selection, post-extract subsampling, explicit split assignment for bland tiles

**Generated files (not committed to git):**
- `/mnt/mrdr/categorized_mineral_units/T{1241,1242,1243,1280,1313,1314,1315,1336}.gpkg` (8 new files, written outside the repo)
- `data/mrral_pixels.parquet` (rebuilt — replaces existing, NOT committed; large)
- `data/patch_cache/mrral_{train,val,test}_patches_p7.npy` (rebuilt — replaces existing, NOT committed)

**Backup files (created on disk, not committed):**
- `data/mrral_pixels.pre-bland.parquet`
- `data/patch_cache/mrral_{train,val,test}_patches_p7.pre-bland.npy`

---

## Task 1: `bland_tile_gpkg` helper module + tests

A pure function that takes a mrral tile path and returns a GeoDataFrame with one polygon covering the tile extent in the tile's native CRS. Separated from the CLI wrapper so it can be unit-tested.

**Files:**
- Create: `data/bland_tile_gpkg.py`
- Test: `tests/test_relabel_other_bland.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_relabel_other_bland.py` with this content:

```python
"""Tests for the relabel-other-bland data pipeline change.

Spec: docs/superpowers/specs/2026-05-20-relabel-other-bland-tiles-design.md
"""
from __future__ import annotations

import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.bland_tile_gpkg import build_bland_gpkg_for_tile


def _make_fake_mrral_tile(path, height=20, width=20, n_bands=59, crs_wkt=None):
    """Write a tiny synthetic mrral .img + .hdr pair for testing."""
    if crs_wkt is None:
        # Mars 2000 equirectangular, central meridian 0
        crs_wkt = (
            'PROJCS["MRO Mars Equirectangular [IAU 2000] [0.00N; 0.00E]",'
            'GEOGCS["GCS_Mars_2000",DATUM["D_Mars_2000",'
            'SPHEROID["Mars_2000_IAU_IAG",3396190,169.8944472]],'
            'PRIMEM["Reference_Meridian",0],UNIT["Degree",0.0174532925199433]],'
            'PROJECTION["Equirectangular"],PARAMETER["central_meridian",0],'
            'UNIT["metre",1]]'
        )
    transform = from_origin(0, 0, 200, 200)   # 200 m/pixel
    profile = {
        'driver': 'ENVI', 'dtype': 'float32', 'count': n_bands,
        'height': height, 'width': width, 'crs': crs_wkt, 'transform': transform,
    }
    data = np.random.uniform(0.0, 0.3, size=(n_bands, height, width)).astype(np.float32)
    with rasterio.open(path, 'w', **profile) as dst:
        for b in range(n_bands):
            dst.write(data[b], b + 1)


class TestBlandTileGpkg:
    def test_produces_single_polygon_with_other_high_category(self, tmp_path):
        mrral = tmp_path / 't9999_mrral_00n000_0327_4.img'
        _make_fake_mrral_tile(str(mrral))

        gdf = build_bland_gpkg_for_tile(str(mrral))

        assert len(gdf) == 1
        assert gdf.iloc[0]['Category'] == 'Other (High)'
        assert gdf.iloc[0]['Mineral ID 1'] == 'bland'

    def test_geometry_covers_tile_extent(self, tmp_path):
        mrral = tmp_path / 't9999_mrral_00n000_0327_4.img'
        _make_fake_mrral_tile(str(mrral), height=20, width=20)

        gdf = build_bland_gpkg_for_tile(str(mrral))

        with rasterio.open(mrral) as src:
            expected_bounds = src.bounds
        actual_bounds = gdf.total_bounds
        # Polygon should be the tile bounding box
        assert actual_bounds[0] == pytest.approx(expected_bounds.left)
        assert actual_bounds[1] == pytest.approx(expected_bounds.bottom)
        assert actual_bounds[2] == pytest.approx(expected_bounds.right)
        assert actual_bounds[3] == pytest.approx(expected_bounds.top)

    def test_crs_matches_source_tile(self, tmp_path):
        mrral = tmp_path / 't9999_mrral_00n000_0327_4.img'
        _make_fake_mrral_tile(str(mrral))

        gdf = build_bland_gpkg_for_tile(str(mrral))

        with rasterio.open(mrral) as src:
            tile_crs = src.crs
        # CRS should round-trip identically (matters for downstream extract step)
        assert gdf.crs.to_wkt() == tile_crs.to_wkt() or gdf.crs == tile_crs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/mrdr/crism_classification && conda run -n crism python -m pytest tests/test_relabel_other_bland.py::TestBlandTileGpkg -v`
Expected: 3 FAILs with `ModuleNotFoundError: No module named 'data.bland_tile_gpkg'`

- [ ] **Step 3: Implement the helper**

Create `data/bland_tile_gpkg.py`:

```python
"""
Author a single-polygon GeoPackage for a "bland" mrral tile.

Used by scripts/build_bland_other_gpkgs.py to create the 8 hand-picked
bland-tile labels (Category="Other (High)") that replace the existing
mineral-adjacent "Other" labels in the v3 classifier training set.

Spec: docs/superpowers/specs/2026-05-20-relabel-other-bland-tiles-design.md
"""
from __future__ import annotations

import geopandas as gpd
import rasterio
from shapely.geometry import box


def build_bland_gpkg_for_tile(mrral_path: str) -> gpd.GeoDataFrame:
    """Return a single-row GeoDataFrame representing the bland-tile labeling.

    The polygon covers the full tile extent (bounding box) in the tile's
    native CRS. `extract_mrral_pixels_from_pair` filters nodata at the
    pixel-read step, so it's safe to over-cover here.

    Schema mirrors the existing /mnt/mrdr/categorized_mineral_units/T*.gpkg
    files so the build pipeline ingests it without special-casing.
    """
    with rasterio.open(mrral_path) as src:
        bounds = src.bounds
        crs = src.crs

    geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    gdf = gpd.GeoDataFrame(
        {
            'Polygon Number': [0],
            'Color':           ['#aaaaaa'],
            'Number of Points': [None],
            'Denominator':     [None],
            'Template':        [None],
            'Mineral ID 1':    ['bland'],
            'Mineral ID 2':    [None],
            'Mineral ID 3':    [None],
            'Mineral ID 4':    [None],
            'wvl':             [None],
            'Spectrum Mean':   [None],
            'params':          [None],
            'Parameters Mean': [None],
            'Best Denom ID':   [None],
            'Ratio Spectrum':  [None],
            'Category':        ['Other (High)'],
        },
        geometry=[geom],
        crs=crs,
    )
    return gdf
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/mrdr/crism_classification && conda run -n crism python -m pytest tests/test_relabel_other_bland.py::TestBlandTileGpkg -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
cd /mnt/mrdr/crism_classification && git add data/bland_tile_gpkg.py tests/test_relabel_other_bland.py && git commit -m "$(cat <<'EOF'
feat(labels): add helper to author single-polygon bland-tile GPKGs

build_bland_gpkg_for_tile(mrral_path) returns a one-row GeoDataFrame
matching the existing T*.gpkg schema, with Category="Other (High)"
and Mineral ID 1="bland". Polygon covers the full tile extent in the
tile's native CRS; extract_mrral_pixels_from_pair drops nodata at
read time, so over-covering is safe.

Spec: docs/superpowers/specs/2026-05-20-relabel-other-bland-tiles-design.md

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: CLI wrapper script

A script that loops over the 8 bland tile paths and invokes the helper, writing each GPKG to `/mnt/mrdr/categorized_mineral_units/`. Idempotent + validated.

**Files:**
- Create: `scripts/build_bland_other_gpkgs.py`
- Test: `tests/test_relabel_other_bland.py` (add)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_relabel_other_bland.py`:

```python
import importlib.util


def _load_module(path):
    spec = importlib.util.spec_from_file_location('m', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class TestBuildBlandOtherGpkgsScript:
    def test_writes_gpkg_for_one_tile(self, tmp_path, monkeypatch):
        mrral = tmp_path / 't9999_mrral_00n000_0327_4.img'
        _make_fake_mrral_tile(str(mrral))
        out_dir = tmp_path / 'gpkgs'
        out_dir.mkdir()

        script_path = os.path.join(os.path.dirname(__file__), '..',
                                    'scripts', 'build_bland_other_gpkgs.py')
        m = _load_module(script_path)

        # Call the writer function directly so the test doesn't need to mock argparse
        out_path = m.write_one_bland_gpkg(str(mrral), 't9999', str(out_dir))
        assert os.path.isfile(out_path)
        gdf = gpd.read_file(out_path)
        assert len(gdf) == 1
        assert gdf.iloc[0]['Category'] == 'Other (High)'
        assert gdf.iloc[0]['Mineral ID 1'] == 'bland'

    def test_idempotent_re_run_validates_existing(self, tmp_path):
        mrral = tmp_path / 't9999_mrral_00n000_0327_4.img'
        _make_fake_mrral_tile(str(mrral))
        out_dir = tmp_path / 'gpkgs'
        out_dir.mkdir()

        script_path = os.path.join(os.path.dirname(__file__), '..',
                                    'scripts', 'build_bland_other_gpkgs.py')
        m = _load_module(script_path)

        path_a = m.write_one_bland_gpkg(str(mrral), 't9999', str(out_dir))
        mtime_a = os.path.getmtime(path_a)

        # Re-run: should detect existing valid file and skip
        path_b = m.write_one_bland_gpkg(str(mrral), 't9999', str(out_dir))
        mtime_b = os.path.getmtime(path_b)
        assert path_a == path_b
        assert mtime_a == mtime_b   # not overwritten

    def test_overwrites_invalid_existing_file(self, tmp_path):
        mrral = tmp_path / 't9999_mrral_00n000_0327_4.img'
        _make_fake_mrral_tile(str(mrral))
        out_dir = tmp_path / 'gpkgs'
        out_dir.mkdir()
        bad_path = out_dir / 'T9999.gpkg'
        bad_path.write_text('not a valid gpkg')

        script_path = os.path.join(os.path.dirname(__file__), '..',
                                    'scripts', 'build_bland_other_gpkgs.py')
        m = _load_module(script_path)
        out_path = m.write_one_bland_gpkg(str(mrral), 't9999', str(out_dir))
        # Should have overwritten the bad file with a real GPKG
        gdf = gpd.read_file(out_path)
        assert len(gdf) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/mrdr/crism_classification && conda run -n crism python -m pytest tests/test_relabel_other_bland.py::TestBuildBlandOtherGpkgsScript -v`
Expected: 3 FAILs with `FileNotFoundError` (script doesn't exist yet) or `AttributeError`.

- [ ] **Step 3: Implement the script**

Create `scripts/build_bland_other_gpkgs.py`:

```python
"""
Author single-polygon GPKGs for the 8 hand-picked bland tiles.

Each output GPKG ('T<tile_num>.gpkg') has one row with Category="Other (High)"
and a polygon covering the source mrral tile's full extent.

Output dir: /mnt/mrdr/categorized_mineral_units/ (existing repository of
labeled-tile GPKGs the build pipeline consumes via find_mrral_pairs).

Idempotent: skips files that read back as valid bland GPKGs; overwrites
invalid/garbage files. Run any time the bland-tile set changes.

Spec: docs/superpowers/specs/2026-05-20-relabel-other-bland-tiles-design.md
"""
from __future__ import annotations

import argparse
import os
import sys

import geopandas as gpd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.bland_tile_gpkg import build_bland_gpkg_for_tile


BLAND_TILES = [
    ('t1241', '/mnt/mrdr/mc12/t1241_mrral_20n033_0327_4.img'),
    ('t1242', '/mnt/mrdr/mc12/t1242_mrral_20n038_0327_4.img'),
    ('t1243', '/mnt/mrdr/mc12/t1243_mrral_20n043_0327_4.img'),
    ('t1280', '/mnt/mrdr/mc09/t1280_mrral_20n228_0327_4.img'),
    ('t1313', '/mnt/mrdr/mc12/t1313_mrral_25n033_0327_4.img'),
    ('t1314', '/mnt/mrdr/mc12/t1314_mrral_25n038_0327_4.img'),
    ('t1315', '/mnt/mrdr/mc12/t1315_mrral_25n043_0327_4.img'),
    ('t1336', '/mnt/mrdr/mc15/t1336_mrral_25n148_0327_4.img'),
]

DEFAULT_OUT_DIR = '/mnt/mrdr/categorized_mineral_units'


def _is_valid_bland_gpkg(path: str) -> bool:
    """Validate an existing file is a one-row bland GPKG."""
    try:
        gdf = gpd.read_file(path)
    except Exception:
        return False
    if len(gdf) != 1:
        return False
    if gdf.iloc[0].get('Category') != 'Other (High)':
        return False
    if gdf.iloc[0].get('Mineral ID 1') != 'bland':
        return False
    return True


def write_one_bland_gpkg(mrral_path: str, tile_id: str, out_dir: str) -> str:
    """Write the GPKG for one tile. Returns the output path."""
    out_path = os.path.join(out_dir, f'{tile_id.upper()}.gpkg')

    if os.path.exists(out_path) and _is_valid_bland_gpkg(out_path):
        return out_path

    gdf = build_bland_gpkg_for_tile(mrral_path)
    # Layer name matches the convention of existing T*.gpkg files
    gdf.to_file(out_path, layer=tile_id.upper(), driver='GPKG')
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR,
                        help=f'Output directory (default: {DEFAULT_OUT_DIR})')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for tid, mrral in BLAND_TILES:
        if not os.path.exists(mrral):
            print(f'  SKIP {tid}: source tile not found at {mrral}')
            continue
        out = write_one_bland_gpkg(mrral, tid, args.out_dir)
        print(f'  {tid} -> {out}')
    print('Done.')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/mrdr/crism_classification && conda run -n crism python -m pytest tests/test_relabel_other_bland.py::TestBuildBlandOtherGpkgsScript -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
cd /mnt/mrdr/crism_classification && git add scripts/build_bland_other_gpkgs.py tests/test_relabel_other_bland.py && git commit -m "$(cat <<'EOF'
feat(labels): script to author bland-tile GPKGs

scripts/build_bland_other_gpkgs.py loops over the 8 hand-picked bland
tiles (t1241/1242/1243/1280/1313/1314/1315/1336), authors a single-
polygon GPKG for each, and writes to /mnt/mrdr/categorized_mineral_units/.
Idempotent (skips files that validate; overwrites invalid ones).

Spec: docs/superpowers/specs/2026-05-20-relabel-other-bland-tiles-design.md

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Subsampling + split-assignment functions in `build_mrral_dataset.py`

Add the subsampling logic and explicit split-assignment as importable functions so they can be unit-tested separately from the full build pipeline.

**Files:**
- Modify: `scripts/build_mrral_dataset.py` (top of file)
- Test: `tests/test_relabel_other_bland.py` (add)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_relabel_other_bland.py`:

```python
class TestSubsampleBlandRows:
    def test_caps_each_bland_tile_to_sample_size(self):
        from scripts.build_mrral_dataset import (
            subsample_bland_other_rows, BLAND_TILES_ORDERED,
        )
        # 200K rows per bland tile + some non-bland rows
        rows = []
        for tid in BLAND_TILES_ORDERED:
            for r in range(200):
                rows.append({'tile_id': tid, 'pixel_row': r, 'pixel_col': 0,
                              'other': 1, 'olivine_t1': 0})
        rows.extend([
            {'tile_id': 't0435', 'pixel_row': r, 'pixel_col': 0,
             'other': 0, 'olivine_t1': 1}
            for r in range(50)
        ])
        df = pd.DataFrame(rows)
        out = subsample_bland_other_rows(df, sample_per_tile=100, seed=42)
        # Each bland tile reduced to 100 rows
        for tid in BLAND_TILES_ORDERED:
            n = int((out['tile_id'] == tid).sum())
            assert n == 100, f'{tid}: got {n}, expected 100'
        # Non-bland row count unchanged
        assert int((out['tile_id'] == 't0435').sum()) == 50

    def test_keeps_all_when_tile_has_fewer_than_sample(self):
        from scripts.build_mrral_dataset import subsample_bland_other_rows
        df = pd.DataFrame([
            {'tile_id': 't1241', 'pixel_row': r, 'pixel_col': 0,
             'other': 1, 'olivine_t1': 0}
            for r in range(50)
        ])
        out = subsample_bland_other_rows(df, sample_per_tile=100, seed=42)
        assert int((out['tile_id'] == 't1241').sum()) == 50

    def test_reproducible_seed(self):
        from scripts.build_mrral_dataset import subsample_bland_other_rows
        rows = [
            {'tile_id': 't1241', 'pixel_row': r, 'pixel_col': 0,
             'other': 1, 'olivine_t1': 0}
            for r in range(200)
        ]
        df = pd.DataFrame(rows)
        out_a = subsample_bland_other_rows(df.copy(), sample_per_tile=50, seed=42)
        out_b = subsample_bland_other_rows(df.copy(), sample_per_tile=50, seed=42)
        # Same rows in both runs (compare by pixel_row, since indices may differ)
        assert sorted(out_a['pixel_row'].tolist()) == sorted(out_b['pixel_row'].tolist())


class TestAssignBlandSplits:
    def test_assigns_70_15_15_split(self):
        from scripts.build_mrral_dataset import (
            assign_bland_tile_splits, BLAND_TILES_ORDERED,
        )
        # 10000 rows per bland tile (statistical test — need enough samples)
        rows = []
        for tid in BLAND_TILES_ORDERED:
            for r in range(10000):
                rows.append({'tile_id': tid, 'split': 'train'})
        df = pd.DataFrame(rows)
        out = assign_bland_tile_splits(df, seed=42)
        # Aggregate across all 8 tiles: 80000 rows total
        n = len(out)
        n_train = int((out['split'] == 'train').sum())
        n_val   = int((out['split'] == 'val').sum())
        n_test  = int((out['split'] == 'test').sum())
        # Allow ±2% tolerance per split (large samples)
        assert 0.68 * n <= n_train <= 0.72 * n, n_train / n
        assert 0.13 * n <= n_val   <= 0.17 * n, n_val / n
        assert 0.13 * n <= n_test  <= 0.17 * n, n_test / n

    def test_non_bland_rows_untouched(self):
        from scripts.build_mrral_dataset import assign_bland_tile_splits
        df = pd.DataFrame([
            {'tile_id': 't0435', 'split': 'val'},
            {'tile_id': 't1241', 'split': 'train'},
            {'tile_id': 't0886', 'split': 'test'},
        ])
        out = assign_bland_tile_splits(df, seed=42)
        # t0435 and t0886 are non-bland → keep original 'val'/'test'
        assert out.loc[out['tile_id'] == 't0435', 'split'].iloc[0] == 'val'
        assert out.loc[out['tile_id'] == 't0886', 'split'].iloc[0] == 'test'

    def test_reproducible_seed(self):
        from scripts.build_mrral_dataset import (
            assign_bland_tile_splits, BLAND_TILES_ORDERED,
        )
        rows = [
            {'tile_id': BLAND_TILES_ORDERED[0], 'split': 'train'}
            for _ in range(100)
        ]
        df = pd.DataFrame(rows)
        out_a = assign_bland_tile_splits(df.copy(), seed=42)
        out_b = assign_bland_tile_splits(df.copy(), seed=42)
        assert out_a['split'].tolist() == out_b['split'].tolist()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/mrdr/crism_classification && conda run -n crism python -m pytest tests/test_relabel_other_bland.py::TestSubsampleBlandRows tests/test_relabel_other_bland.py::TestAssignBlandSplits -v`
Expected: FAILs with `ImportError: cannot import name 'subsample_bland_other_rows' from 'scripts.build_mrral_dataset'`

- [ ] **Step 3: Modify `scripts/build_mrral_dataset.py`**

Open `scripts/build_mrral_dataset.py`. Replace the existing top of the file (everything down to the `def main():` line) with this:

```python
"""
Extract mrral (59-band, 410-2457 nm) spectra for all labeled polygons.
Writes data/mrral_pixels.parquet with columns m0..m58 plus standard metadata.

Per-tile gating:
  - Non-bland tiles: existing "Other"-categorized polygons are SKIPPED via
    the extract_mrral_pixels_from_pair `other_polygon_ids` parameter (set
    to an empty set). Their mineral-category polygons still contribute.
  - Bland tiles (BLAND_TILES_ORDERED): all polygons (which is just the
    single "Other (High)" polygon authored by build_bland_other_gpkgs.py)
    contribute via other_polygon_ids=None (no filter).
  - Bland-tile rows are subsampled to 113K per tile post-extraction.
  - Bland-tile split assignment is explicit (70/15/15) BEFORE the merge
    against the mrrsu parquet, since these tiles have no rows there.

Spec: docs/superpowers/specs/2026-05-20-relabel-other-bland-tiles-design.md

Usage:
    conda run -n crism python scripts/build_mrral_dataset.py
"""
import os
import sys
import logging

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Fixed enumeration order → reproducible per-tile RNG seeds (SEED + i).
# Using Python's hash(tid) instead would be PYTHONHASHSEED-salted and non-
# reproducible across invocations.
BLAND_TILES_ORDERED = [
    't1241', 't1242', 't1243', 't1280',
    't1313', 't1314', 't1315', 't1336',
]
BLAND_TILES = set(BLAND_TILES_ORDERED)

SAMPLE_PER_TILE = 113_000
SPLIT_FRACS = {'train': 0.70, 'val': 0.15, 'test': 0.15}
SEED = 42


def other_polygon_ids_for_tile(tile_id: str):
    """Return the `other_polygon_ids` argument for extract_mrral_pixels_from_pair.

    Bland tiles: None (no filter — all polygons contribute).
    Non-bland tiles: empty set (filter out "Other"-categorized polygons).
    """
    return None if tile_id in BLAND_TILES else set()


def subsample_bland_other_rows(df: pd.DataFrame, sample_per_tile: int = SAMPLE_PER_TILE,
                                seed: int = SEED) -> pd.DataFrame:
    """Randomly subsample bland-tile rows to ~sample_per_tile each.

    Reproducible: per-tile RNG seeded with `seed + i` where `i` is the
    tile's index in BLAND_TILES_ORDERED. Rows from non-bland tiles are
    untouched.
    """
    bland_mask = df['tile_id'].isin(BLAND_TILES)
    keep_idx = list(df.index[~bland_mask])

    for i, tid in enumerate(BLAND_TILES_ORDERED):
        tile_idx = df.index[bland_mask & (df['tile_id'] == tid)].to_numpy()
        if len(tile_idx) == 0:
            continue
        rng = np.random.default_rng(seed + i)
        n_keep = min(sample_per_tile, len(tile_idx))
        chosen = rng.choice(tile_idx, size=n_keep, replace=False)
        keep_idx.extend(chosen.tolist())

    return df.loc[keep_idx].reset_index(drop=True)


def assign_bland_tile_splits(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Overwrite the `split` column for bland-tile rows with a 70/15/15 random
    train/val/test assignment. Non-bland rows are left untouched.

    Per-tile RNG (seed + 100 + i) so the assignment is independent across
    tiles AND independent of the subsampling RNG.
    """
    splits_array = np.array(list(SPLIT_FRACS.keys()))
    probs = np.array(list(SPLIT_FRACS.values()))

    out = df.copy()
    for i, tid in enumerate(BLAND_TILES_ORDERED):
        mask = (out['tile_id'] == tid)
        n = int(mask.sum())
        if n == 0:
            continue
        rng = np.random.default_rng(seed + 100 + i)
        chosen = rng.choice(splits_array, size=n, p=probs)
        out.loc[mask, 'split'] = chosen

    return out


def main():
```

That replaces lines 1-19 (the original docstring + imports + `def main():` line). The existing `main()` body that follows it stays unchanged for now — Task 4 will modify the body.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/mrdr/crism_classification && conda run -n crism python -m pytest tests/test_relabel_other_bland.py::TestSubsampleBlandRows tests/test_relabel_other_bland.py::TestAssignBlandSplits -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
cd /mnt/mrdr/crism_classification && git add scripts/build_mrral_dataset.py tests/test_relabel_other_bland.py && git commit -m "$(cat <<'EOF'
feat(labels): subsample + split-assign helpers for bland-tile rows

Adds three importable functions to scripts/build_mrral_dataset.py:
  - other_polygon_ids_for_tile(tid)   — gate selector
  - subsample_bland_other_rows(df)    — 113K/tile cap, reproducible
                                        per-tile RNG (SEED + i)
  - assign_bland_tile_splits(df)      — explicit 70/15/15 for the
                                        8 bland tiles (which have no
                                        rows in mrrsu pixels.parquet)

main() body unchanged — Task 4 wires these into the build loop.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Wire the gate + subsample + split-assign into `main()`

Modify the `main()` body to call the new helpers in the correct order.

**Files:**
- Modify: `scripts/build_mrral_dataset.py` (`main()` body)

- [ ] **Step 1: Replace the existing `main()` body**

In `scripts/build_mrral_dataset.py`, replace the existing `main()` function (the part AFTER `def main():`) with this:

```python
    from config_loader import load_config
    cfg = load_config()
    from data.extract_pixels import find_mrral_pairs, extract_mrral_pixels_from_pair

    pairs = find_mrral_pairs(cfg['gpkg_dir'], cfg['data_root'])
    logging.info(f"Found {len(pairs)} mrral tile pairs")

    # Re-use same train/val/test split as mrrsu parquet — join on tile+polygon+row+col.
    mrrsu_parquet = os.path.join(cfg['output_dir'], 'pixels.parquet')
    mrrsu_splits = pd.read_parquet(
        mrrsu_parquet,
        columns=['tile_id', 'polygon_id', 'pixel_row', 'pixel_col', 'split'],
    )
    logging.info(f"Loaded {len(mrrsu_splits)} split entries from {mrrsu_parquet}")

    all_records = []
    for i, (tile_id, gpkg_path, mrral_path) in enumerate(pairs):
        gate = other_polygon_ids_for_tile(tile_id)
        gate_desc = 'BLAND (allow all)' if gate is None else 'block Other'
        logging.info(f"[{i+1}/{len(pairs)}] Processing {tile_id}  ({gate_desc})")
        records = extract_mrral_pixels_from_pair(
            tile_id, mrral_path, gpkg_path,
            other_polygon_ids=gate,
        )
        logging.info(f"  {len(records)} pixels extracted")
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    del all_records
    logging.info(f"Total pixels before subsample/merge: {len(df)}")

    # 1. Subsample bland-tile rows to 113K each.
    df = subsample_bland_other_rows(df)
    logging.info(f"After bland-tile subsample: {len(df)}")

    # 2. Merge in train/val/test splits from the mrrsu parquet.
    df = df.merge(
        mrrsu_splits,
        on=['tile_id', 'polygon_id', 'pixel_row', 'pixel_col'],
        how='left',
    )
    df['split'] = df['split'].fillna('train')

    # 3. Overwrite split for bland-tile rows — they have no rows in the mrrsu
    #    parquet so without this step ALL 904K would default to 'train', leaving
    #    val/test with no "other" pixels.
    df = assign_bland_tile_splits(df)

    out = os.path.join(cfg['output_dir'], 'mrral_pixels.parquet')
    df.to_parquet(out, index=False)
    logging.info(f"Wrote {len(df)} pixels to {out}")
    logging.info(f"Splits: {df['split'].value_counts().to_dict()}")
    logging.info(f"'other' label total: {int(df['other'].sum())}")
    logging.info(f"'other' by split: {df[df['other']==1]['split'].value_counts().to_dict()}")
    logging.info(f"Columns (first 10): {list(df.columns[:10])}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `cd /mnt/mrdr/crism_classification && conda run -n crism python -c "import ast; ast.parse(open('scripts/build_mrral_dataset.py').read()); print('syntax OK')"`
Expected: `syntax OK`

- [ ] **Step 3: Verify the helper-function tests still pass**

Run: `cd /mnt/mrdr/crism_classification && conda run -n crism python -m pytest tests/test_relabel_other_bland.py -v`
Expected: all previously-passing tests still PASS.

- [ ] **Step 4: Commit**

```bash
cd /mnt/mrdr/crism_classification && git add scripts/build_mrral_dataset.py && git commit -m "$(cat <<'EOF'
feat(labels): wire bland-tile gate + subsample + split into main()

Build loop now passes other_polygon_ids per-tile to
extract_mrral_pixels_from_pair (set() for non-bland, None for bland),
subsamples bland-tile rows to 113K each, then overwrites their split
column with an explicit 70/15/15 BEFORE the parquet write. Without
the explicit split overwrite, all 904K bland-tile rows would default
to 'train' because they have no rows in the mrrsu pixels.parquet.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Pre-build backup

Manual procedure — captured as a small shell script so it's repeatable.

**Files:**
- Create: `scripts/backup_pre_bland.sh`

- [ ] **Step 1: Write the script**

Create `scripts/backup_pre_bland.sh`:

```bash
#!/bin/bash
# Pre-bland-relabel backup. Run ONCE before scripts/build_mrral_dataset.py
# rewrites the parquet, and before scripts/cache_mrral_patches.py rewrites
# the patch cache.
#
# Spec: docs/superpowers/specs/2026-05-20-relabel-other-bland-tiles-design.md
set -e

PROJ=/mnt/mrdr/crism_classification

PARQUET="${PROJ}/data/mrral_pixels.parquet"
BACKUP_PARQUET="${PROJ}/data/mrral_pixels.pre-bland.parquet"

if [ -f "$BACKUP_PARQUET" ]; then
    echo "Backup already exists: ${BACKUP_PARQUET} (refusing to overwrite)"
else
    cp "$PARQUET" "$BACKUP_PARQUET"
    echo "Backed up parquet -> ${BACKUP_PARQUET}"
fi

CACHE_DIR="${PROJ}/data/patch_cache"
for split in train val test; do
    SRC="${CACHE_DIR}/mrral_${split}_patches_p7.npy"
    DST="${CACHE_DIR}/mrral_${split}_patches_p7.pre-bland.npy"
    if [ ! -f "$SRC" ]; then
        echo "  SKIP $split: source cache not found at $SRC"
        continue
    fi
    if [ -f "$DST" ]; then
        echo "  $split backup already exists (refusing to overwrite)"
        continue
    fi
    cp "$SRC" "$DST"
    echo "  Backed up $split cache -> $DST"
done
echo "Done."
```

- [ ] **Step 2: Make executable + commit**

```bash
cd /mnt/mrdr/crism_classification && chmod +x scripts/backup_pre_bland.sh && git add scripts/backup_pre_bland.sh && git commit -m "$(cat <<'EOF'
chore(labels): pre-bland backup script for parquet + patch cache

Refuses to overwrite existing backups so it's safe to run multiple
times. Run once before build_mrral_dataset.py rewrites the parquet,
and before cache_mrral_patches.py rewrites the patch cache.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Execute — author the 8 GPKGs

Manual run of the authoring script. No code change. Verifies the 8 source mrral tiles exist + writes the GPKGs to `/mnt/mrdr/categorized_mineral_units/`.

- [ ] **Step 1: Run the script**

```bash
cd /mnt/mrdr/crism_classification && conda run -n crism python scripts/build_bland_other_gpkgs.py
```

Expected output: 8 lines like `  t1241 -> /mnt/mrdr/categorized_mineral_units/T1241.gpkg`, followed by `Done.`

- [ ] **Step 2: Verify all 8 GPKGs exist**

```bash
ls -la /mnt/mrdr/categorized_mineral_units/T{1241,1242,1243,1280,1313,1314,1315,1336}.gpkg
```

Expected: 8 files listed, each ~10-50 KB.

- [ ] **Step 3: Spot-check one GPKG**

```bash
conda run -n crism python -c "
import geopandas as gpd
gdf = gpd.read_file('/mnt/mrdr/categorized_mineral_units/T1241.gpkg')
print(f'rows: {len(gdf)}')
print(f'category: {gdf.iloc[0][\"Category\"]}')
print(f'mineral_id_1: {gdf.iloc[0][\"Mineral ID 1\"]}')
print(f'bounds: {gdf.total_bounds}')
print(f'crs: {gdf.crs.name}')
"
```

Expected: 1 row, Category="Other (High)", Mineral ID 1="bland", bounds finite, CRS is the per-tile equirectangular.

No commit (these files live outside the repo at `/mnt/mrdr/categorized_mineral_units/`).

---

## Task 7: Execute — backup originals

- [ ] **Step 1: Run the backup script**

```bash
cd /mnt/mrdr/crism_classification && bash scripts/backup_pre_bland.sh
```

Expected output: 4 "Backed up" lines (1 parquet + 3 cache splits), then `Done.`

- [ ] **Step 2: Verify backups exist**

```bash
ls -la data/mrral_pixels.pre-bland.parquet data/patch_cache/mrral_*_patches_p7.pre-bland.npy
```

Expected: 4 files listed, each matching the original sizes.

---

## Task 8: Execute — rebuild the parquet

- [ ] **Step 1: Run the build**

```bash
cd /mnt/mrdr/crism_classification && conda run -n crism python scripts/build_mrral_dataset.py 2>&1 | tail -30
```

Expected: log lines showing "BLAND (allow all)" for the 8 bland tiles + "block Other" for the rest, total pixel count before subsample (~13M), after subsample (~3-4M), final `'other' label total: ~904000`, splits roughly `{'train': N, 'val': M, 'test': K}` with sensible proportions.

- [ ] **Step 2: Verify the parquet looks right**

```bash
conda run -n crism python -c "
import pandas as pd
df = pd.read_parquet('data/mrral_pixels.parquet')
print(f'Total rows: {len(df):,}')
print(f'Per-class totals:')
print(df[['olivine_t1','olivine_t2','lcp','hcp','plagioclase','other']].sum().to_string())
print()
print(f'Per-split row counts:')
print(df['split'].value_counts().to_string())
print()
print(f'Bland-tile rows by split (should be ~70/15/15):')
bland_tiles = {'t1241','t1242','t1243','t1280','t1313','t1314','t1315','t1336'}
sub = df[df['tile_id'].isin(bland_tiles)]
print(sub['split'].value_counts(normalize=True).round(2).to_string())
print()
print(f'Non-bland \"other\" rows: {int(((~df[\"tile_id\"].isin(bland_tiles)) & (df[\"other\"]==1)).sum())}')
print(f'(should be 0 — non-bland \"Other\" polygons gated out)')
"
```

Expected:
- Total `other ≈ 904K`
- Bland split fractions ≈ 0.70 / 0.15 / 0.15
- Non-bland "other" rows = 0

If any of these is off, STOP and investigate before proceeding.

---

## Task 9: Execute — rebuild the patch cache

- [ ] **Step 1: Run cache build**

```bash
cd /mnt/mrdr/crism_classification && conda run -n crism python scripts/cache_mrral_patches.py 2>&1 | tail -10
```

Expected: 3 cache files regenerated, sizes consistent with new parquet row counts.

- [ ] **Step 2: Verify cache shapes**

```bash
conda run -n crism python -c "
import numpy as np
for split in ['train', 'val', 'test']:
    a = np.load(f'data/patch_cache/mrral_{split}_patches_p7.npy', mmap_mode='r')
    print(f'  {split}: {a.shape}  dtype={a.dtype}')
"
```

Expected: shapes `(N, 7, 7, 59)` with N matching the per-split row counts from Task 8.

---

## Task 10: Final integration

- [ ] **Step 1: Run the full test suite to catch regressions**

```bash
cd /mnt/mrdr/crism_classification && conda run -n crism python -m pytest tests/test_relabel_other_bland.py tests/test_dataset.py -v 2>&1 | tail -15
```

Expected: all `test_relabel_other_bland` tests pass. `test_dataset.py` may have pre-existing failures from missing fixture data — those are not regressions of this work as long as they're the same failures as before.

- [ ] **Step 2: Compare label distribution to pre-bland baseline**

```bash
conda run -n crism python -c "
import pandas as pd
old = pd.read_parquet('data/mrral_pixels.pre-bland.parquet')
new = pd.read_parquet('data/mrral_pixels.parquet')
print('class            old        new      delta')
for c in ['olivine_t1','olivine_t2','lcp','hcp','plagioclase','other']:
    o = int(old[c].sum()); n = int(new[c].sum())
    print(f'  {c:<12}  {o:>9,}  {n:>9,}  {n-o:>+10,}')
print()
print(f'total rows:  {len(old):>9,}  {len(new):>9,}  {len(new)-len(old):>+10,}')
"
```

Expected: mineral class counts unchanged ± a few percent (very minor jitter from any "Other"-only-labeled pixels in old "other"-tiles being dropped). `other` count near 904K. Old total had ~2.4M rows; new total around 3.4-3.6M (mineral rows unchanged + 904K new "other" - 677K old "other").

- [ ] **Step 3: Verify the patch cache loads cleanly**

```bash
conda run -n crism python -c "
from data.dataset import CRISMSpectralPatchDataset
import pandas as pd
df = pd.read_parquet('data/mrral_pixels.parquet')
val_df = df[df['split'] == 'val'].head(1000)
# Build a tiny patch-cached dataset to verify the loader works against the new parquet + cache
print(f'val sample rows: {len(val_df)}')
print(f'other val rows: {int(val_df[\"other\"].sum())}')
print('Cache load: OK' if val_df['other'].sum() > 0 else 'WARN: no \"other\" pixels in val sample — check splits')
"
```

Expected: `Cache load: OK` (val partition has some "other" pixels now that bland tiles are explicitly split).

If everything looks right: the data pipeline change is complete. The next step (out of scope for this plan) is to launch an HPC finetune sweep against the new patch cache.

---

## Out of scope (do not implement in this plan)

- HPC finetune sweep against the new labels (separate slurm job).
- Modifying source GPKGs in `/mnt/mrdr/categorized_mineral_units/` for OTHER tiles.
- Reconciling the `label_parser.py` (1.0/0.5/0.25) vs `_collapse_labels` (1.0/0.85/0.70) confidence-weight discrepancy — pre-existing, out of scope.
- Updating wiki / Methodology Log entries with the new "other" definition — do after a finetune validates the change downstream.
- Removing the stale `config.yaml` confidence_weights stub.
- Adding additional bland tiles for geographic diversity.
- Re-introducing dropped multi-label `(mineral + other)` rows under the new schema.

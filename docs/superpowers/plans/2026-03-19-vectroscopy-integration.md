# Vectroscopy Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Feed SpatialSpectralClassifier probability rasters through Vectroscopy to produce per-mineral vector GeoPackage products with model-driven confidence tiers for T0435 and T0434.

**Architecture:** Three scripts — extend `classify_tile_supervised.py` to save prob rasters, add `compute_global_thresholds.py` to derive global percentile thresholds, add `vectorize_tile_minerals.py` to run Vectroscopy per class and write layered GeoPackages. All logic extracted into importable functions; scripts are thin CLI wrappers.

**Tech Stack:** PyTorch (inference), rasterio (CRS/transform), Vectroscopy (git clone, `core.vectroscopy`), rasterstats (zonal stats), geopandas, scipy (median filter), numpy, pytest.

---

## File Map

| Path | Action | Responsibility |
|---|---|---|
| `scripts/classify_tile_supervised.py` | Modify | Add `--save_probs` arg; extract `save_probs()` helper |
| `scripts/compute_global_thresholds.py` | Create | Pool valid-pixel probs, compute percentile thresholds, write JSON |
| `scripts/vectorize_tile_minerals.py` | Create | Load probs + thresholds, run per-class Vectroscopy pipeline, write GeoPackage |
| `tests/test_compute_global_thresholds.py` | Create | Unit tests for threshold computation logic |
| `tests/test_vectorize_tile_minerals.py` | Create | Unit tests for tier assignment, median filter ordering, zonal stats integration |
| `config/vectroscopy_thresholds.json` | Create (generated) | Calibrated thresholds for T0434+T0435 |
| `data/vector/` | Create (generated) | Output GeoPackages |

---

## Chunk 1: Environment Setup + Stage 1 Extension

### Task 1: Install Vectroscopy and rasterstats

**Files:** none (environment setup only)

- [ ] **Step 1: Clone Vectroscopy**

```bash
git clone https://github.com/Tahn04/Vectroscopy.git /opt/Vectroscopy
```

Expected: directory `/opt/Vectroscopy/src/core/vectroscopy.py` exists.

- [ ] **Step 2: Verify Vectroscopy import works**

```bash
conda run -n crism python3 -c "
import sys; sys.path.insert(0, '/opt/Vectroscopy/src')
import core.vectroscopy as vp
print('Vectroscopy OK:', vp.Vectroscopy)
"
```

Expected: prints `Vectroscopy OK: <class 'core.vectroscopy.Vectroscopy'>`.

- [ ] **Step 3: Install rasterstats**

```bash
conda run -n crism pip install rasterstats
```

- [ ] **Step 4: Verify rasterstats import**

```bash
conda run -n crism python3 -c "import rasterstats; print('rasterstats OK:', rasterstats.__version__)"
```

Expected: version string printed without error.

---

### Task 2: Extend classify_tile_supervised.py with --save_probs

**Files:**
- Modify: `scripts/classify_tile_supervised.py`
- Create: `tests/test_classify_tile_supervised_save_probs.py`

The existing `run_supervised()` returns `(H*W, 5)` probs including the "other" class at index 4.
We need to slice to `[:, :4]` (first 4 classes) before saving.

- [ ] **Step 1: Write the failing test**

Create `tests/test_classify_tile_supervised_save_probs.py`:

```python
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_save_probs_output_shape(tmp_path):
    """save_probs writes (H,W,4) probs + valid_mask + transform + crs_wkt to npz."""
    from scripts.classify_tile_supervised import save_probs

    H, W = 10, 12
    probs_hw4 = np.random.rand(H, W, 4).astype(np.float32)
    valid_mask = np.ones((H, W), dtype=bool)
    valid_mask[0, 0] = False
    # transform_arr: rasterio Affine order (a,b,c,d,e,f) = (col_scale, col_shear, col_off,
    #                                                         row_shear, row_scale, row_off)
    transform_arr = np.array([200.0, 0.0, 100000.0, 0.0, -200.0, -200000.0])
    crs_wkt = 'PROJCS["Mars_2000_Equidistant_Cylindrical"]'

    out = tmp_path / 'test_probs.npz'
    save_probs(str(out), probs_hw4, valid_mask, transform_arr, crs_wkt)

    data = np.load(str(out), allow_pickle=True)
    assert data['probs'].shape == (H, W, 4)
    assert data['probs'].dtype == np.float32
    assert data['valid_mask'].shape == (H, W)
    assert data['valid_mask'].dtype == bool
    assert data['transform'].shape == (6,)
    # crs_wkt must be a non-empty string matching the input
    assert isinstance(str(data['crs_wkt']), str)
    assert len(str(data['crs_wkt'])) > 0
    assert str(data['crs_wkt']) == crs_wkt


def test_save_probs_values_preserved(tmp_path):
    """Saved probs match input values exactly."""
    from scripts.classify_tile_supervised import save_probs

    probs = np.array([[[[0.1, 0.9, 0.2, 0.05]]]], dtype=np.float32)  # (1,1,4)
    mask = np.array([[True]])
    t = np.zeros(6)
    out = tmp_path / 'p.npz'
    save_probs(str(out), probs, mask, t, '')
    data = np.load(str(out))
    np.testing.assert_array_almost_equal(data['probs'], probs)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd /mnt/mrdr/crism_classification/.worktrees/spatial-mae
conda run -n crism pytest tests/test_classify_tile_supervised_save_probs.py -v 2>&1 | tail -10
```

Expected: `ImportError` or `AttributeError: module has no attribute 'save_probs'`.

- [ ] **Step 3: Add `save_probs()` function and `--save_probs` arg to classify_tile_supervised.py**

Add this function after the `run_supervised()` function (around line 128):

```python
def save_probs(path: str, probs_hw4: np.ndarray, valid_mask: np.ndarray,
               transform_arr: np.ndarray, crs_wkt: str) -> None:
    """Save (H,W,4) mineral probability raster to .npz for downstream vectorization.

    Args:
        path: output .npz path
        probs_hw4: (H, W, 4) float32 probabilities for olivine/lcp/hcp/plagioclase
        valid_mask: (H, W) bool, True = valid pixel
        transform_arr: (6,) float64 rasterio Affine coefficients (a,b,c,d,e,f)
        crs_wkt: CRS as WKT string
    """
    np.savez_compressed(
        path,
        probs=probs_hw4,
        valid_mask=valid_mask,
        transform=transform_arr,
        crs_wkt=crs_wkt,
    )
```

Then in `main()`, after the `parser.add_argument('--out', ...)` line, add:

```python
    parser.add_argument('--save_probs', default=None, metavar='PATH',
                        help='Save (H,W,4) mineral prob raster to .npz for vectorization')
```

And after `probs_flat = run_supervised(...)` and `probs = probs_flat.reshape(H, W, N_CLASSES)`, add:

```python
    if args.save_probs:
        import rasterio
        with rasterio.open(args.tile) as src:
            # Save rasterio Affine as (a,b,c,d,e,f) = (col_scale, col_shear, col_off,
            #                                            row_shear, row_scale, row_off)
            transform_arr = np.array([src.transform.a, src.transform.b, src.transform.c,
                                       src.transform.d, src.transform.e, src.transform.f],
                                      dtype=np.float64)
            crs_wkt = src.crs.to_wkt()
        probs_hw4 = probs[:, :, :4]  # drop "other" class (index 4)
        save_probs(args.save_probs, probs_hw4, valid_mask, transform_arr, crs_wkt)
        print(f'Saved probs → {args.save_probs}')
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
conda run -n crism pytest tests/test_classify_tile_supervised_save_probs.py -v 2>&1 | tail -10
```

Expected: `2 passed`.

- [ ] **Step 5: Run on T0435 to generate test probs file**

Use the full tile ID as the output filename so that `tiles_used` in the JSON reflects the tile ID.

```bash
conda run -n crism python scripts/classify_tile_supervised.py \
    --tile /mnt/mrdr/mc26/t0435_mrral_40s323_0327_4.img \
    --ckpt /mnt/mrdr/crism_classification/checkpoints/spvit_lrscale001_best.pt \
    --save_probs /tmp/t0435_mrral_40s323_0327_4_probs.npz \
    --out /dev/null 2>&1 | grep -E "(Saved|Device|Tile)"
```

Expected output includes `Saved probs → /tmp/t0435_mrral_40s323_0327_4_probs.npz`.

- [ ] **Step 6: Verify output shape**

```bash
conda run -n crism python3 -c "
import numpy as np
d = np.load('/tmp/t0435_mrral_40s323_0327_4_probs.npz')
print('probs:', d['probs'].shape, d['probs'].dtype)
print('valid_mask:', d['valid_mask'].shape)
print('transform:', d['transform'])
print('crs_wkt length:', len(str(d['crs_wkt'])))
assert d['probs'].shape[2] == 4, f'Expected 4 classes, got {d[\"probs\"].shape[2]}'
assert d['probs'].dtype == np.float32
assert d['valid_mask'].dtype == bool
assert d['transform'].shape == (6,)
print('All checks passed')
"
```

Expected: `shape[2] == 4`, `dtype == float32`. Actual H×W dimensions are printed (tile-dependent).

- [ ] **Step 7: Run on T0434 to generate second test file**

```bash
conda run -n crism python scripts/classify_tile_supervised.py \
    --tile /mnt/mrdr/mc26/t0434_mrral_40s318_0327_4.img \
    --ckpt /mnt/mrdr/crism_classification/checkpoints/spvit_lrscale001_best.pt \
    --save_probs /tmp/t0434_mrral_40s318_0327_4_probs.npz \
    --out /dev/null 2>&1 | grep -E "(Saved|Device|Tile)"
```

- [ ] **Step 8: Verify T0434 output shape**

```bash
conda run -n crism python3 -c "
import numpy as np
d = np.load('/tmp/t0434_mrral_40s318_0327_4_probs.npz')
print('probs:', d['probs'].shape, d['probs'].dtype)
assert d['probs'].shape[2] == 4
assert d['probs'].dtype == np.float32
print('All checks passed')
"
```

- [ ] **Step 9: Commit**

```bash
git add scripts/classify_tile_supervised.py \
        tests/test_classify_tile_supervised_save_probs.py
git commit -m "feat: add --save_probs to classify_tile_supervised for Vectroscopy pipeline"
```

---

## Chunk 2: Global Threshold Calibration Script

### Task 3: Write compute_global_thresholds.py

**Files:**
- Create: `scripts/compute_global_thresholds.py`
- Create: `tests/test_compute_global_thresholds.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_compute_global_thresholds.py`:

```python
import numpy as np
import json
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_npz(tmp_path, name, probs, valid_mask):
    """Helper: write a synthetic probs .npz."""
    path = tmp_path / name
    np.savez_compressed(
        str(path),
        probs=probs.astype(np.float32),
        valid_mask=valid_mask,
        transform=np.zeros(6),
        crs_wkt='',
    )
    return str(path)


def test_pool_valid_probs_single_tile(tmp_path):
    """pool_valid_probs returns correct valid-pixel probs for one tile."""
    from scripts.compute_global_thresholds import pool_valid_probs

    H, W = 4, 4
    probs = np.random.rand(H, W, 4).astype(np.float32)
    mask = np.ones((H, W), dtype=bool)
    mask[0, 0] = False  # one invalid pixel

    path = make_npz(tmp_path, 'tile.npz', probs, mask)
    result = pool_valid_probs([path])  # {0: array, 1: array, 2: array, 3: array}

    for ci in range(4):
        expected = probs[:, :, ci][mask]
        np.testing.assert_array_almost_equal(result[ci], expected)


def test_pool_valid_probs_two_tiles(tmp_path):
    """pool_valid_probs concatenates across tiles."""
    from scripts.compute_global_thresholds import pool_valid_probs

    probs1 = np.ones((3, 3, 4), dtype=np.float32) * 0.3
    probs2 = np.ones((3, 3, 4), dtype=np.float32) * 0.7
    mask = np.ones((3, 3), dtype=bool)

    p1 = make_npz(tmp_path, 't1.npz', probs1, mask)
    p2 = make_npz(tmp_path, 't2.npz', probs2, mask)
    result = pool_valid_probs([p1, p2])

    # 9 valid pixels × 2 tiles = 18 per class
    assert len(result[0]) == 18
    np.testing.assert_almost_equal(result[0].mean(), 0.5)


def test_compute_thresholds_values(tmp_path):
    """compute_thresholds returns correct percentiles per class."""
    from scripts.compute_global_thresholds import compute_thresholds

    # Class 0: uniform [0,1], class 1: all 0.9
    pooled = {
        0: np.linspace(0, 1, 100, dtype=np.float32),
        1: np.full(100, 0.9, dtype=np.float32),
        2: np.zeros(100, dtype=np.float32),
        3: np.zeros(100, dtype=np.float32),
    }
    CLASS_NAMES = ['olivine', 'lcp', 'hcp', 'plagioclase']
    result = compute_thresholds(pooled, CLASS_NAMES, percentiles=[33, 67, 90])

    # For uniform [0,1], 33rd pctile ≈ 0.33
    assert abs(result['olivine'][0] - 0.33) < 0.02
    assert abs(result['olivine'][1] - 0.67) < 0.02
    assert abs(result['olivine'][2] - 0.90) < 0.02
    # For all-0.9, all percentiles = 0.9
    assert result['lcp'][0] == pytest.approx(0.9, abs=0.01)
    assert result['lcp'][2] == pytest.approx(0.9, abs=0.01)


def test_write_thresholds_json(tmp_path):
    """write_thresholds_json produces valid JSON matching expected schema."""
    from scripts.compute_global_thresholds import write_thresholds_json

    thresholds = {
        'olivine': [0.28, 0.41, 0.57],
        'lcp': [0.82, 0.91, 0.96],
        'hcp': [0.04, 0.09, 0.18],
        'plagioclase': [0.03, 0.08, 0.15],
    }
    out = tmp_path / 'thresh.json'
    write_thresholds_json(
        str(out),
        thresholds=thresholds,
        tiles_used=['T0434', 'T0435'],
        percentiles=[33, 67, 90],
        morphology={'median_filter_size': 3, 'median_filter_iterations': 1,
                    'sieve_min_pixels': 9, 'majority_filter_iterations': 3,
                    'simplify_tolerance_meters': 200},
    )
    data = json.loads(out.read_text())
    assert 'generated' in data
    assert data['tiles_used'] == ['T0434', 'T0435']
    assert data['percentiles'] == [33, 67, 90]
    assert list(data['thresholds'].keys()) == ['olivine', 'lcp', 'hcp', 'plagioclase']
    assert len(data['thresholds']['olivine']) == 3
    assert 'morphology' in data
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
conda run -n crism pytest tests/test_compute_global_thresholds.py -v 2>&1 | tail -10
```

Expected: `ImportError: No module named 'scripts.compute_global_thresholds'`.

- [ ] **Step 3: Create scripts/compute_global_thresholds.py**

```python
"""
Compute global percentile-based probability thresholds for Vectroscopy vectorization.

Pools valid-pixel probabilities from multiple tile .npz files (produced by
classify_tile_supervised.py --save_probs) and computes percentile thresholds
per mineral class. Output JSON is consumed by vectorize_tile_minerals.py.

Usage:
    python scripts/compute_global_thresholds.py \
        --probs /tmp/t0434_probs.npz /tmp/t0435_probs.npz \
        --out config/vectroscopy_thresholds.json \
        --percentiles 33 67 90
"""
import argparse
import json
import os
import sys
from datetime import date
from typing import Dict, List

import numpy as np

CLASS_NAMES = ['olivine', 'lcp', 'hcp', 'plagioclase']
DEFAULT_MORPHOLOGY = {
    'median_filter_size': 3,
    'median_filter_iterations': 1,
    'sieve_min_pixels': 9,
    'majority_filter_iterations': 3,
    'simplify_tolerance_meters': 200,
}


def pool_valid_probs(npz_paths: List[str]) -> Dict[int, np.ndarray]:
    """Pool valid-pixel probabilities per class across all input tiles.

    Args:
        npz_paths: list of .npz paths produced by classify_tile_supervised --save_probs

    Returns:
        dict mapping class index (0-3) → 1-D float32 array of valid-pixel probs
    """
    pooled = {ci: [] for ci in range(4)}
    for path in npz_paths:
        data = np.load(path)
        probs = data['probs']          # (H, W, 4)
        valid_mask = data['valid_mask']  # (H, W) bool
        for ci in range(4):
            pooled[ci].append(probs[:, :, ci][valid_mask])
    return {ci: np.concatenate(pooled[ci]) for ci in range(4)}


def compute_thresholds(pooled: Dict[int, np.ndarray],
                       class_names: List[str],
                       percentiles: List[int]) -> Dict[str, List[float]]:
    """Compute percentile thresholds per mineral class.

    Args:
        pooled: dict from pool_valid_probs()
        class_names: list of class name strings in class-index order
        percentiles: list of 3 percentile values, e.g. [33, 67, 90]

    Returns:
        dict mapping mineral name → list of 3 float threshold values
    """
    thresholds = {}
    for ci, name in enumerate(class_names):
        vals = pooled[ci]
        t = [float(np.percentile(vals, p)) for p in percentiles]
        thresholds[name] = t
        print(f'  {name:12s}: {[f"{v:.4f}" for v in t]}  '
              f'(n={len(vals):,}, mean={vals.mean():.3f})')
    return thresholds


def write_thresholds_json(out_path: str, thresholds: Dict[str, List[float]],
                          tiles_used: List[str], percentiles: List[int],
                          morphology: dict) -> None:
    """Write calibrated thresholds to JSON.

    Args:
        out_path: output file path
        thresholds: dict from compute_thresholds()
        tiles_used: list of tile identifiers used for calibration
        percentiles: percentile values used
        morphology: morphological parameter defaults (documentation only)
    """
    payload = {
        'generated': str(date.today()),
        'tiles_used': tiles_used,
        'percentiles': percentiles,
        'thresholds': thresholds,
        'morphology': morphology,
    }
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Compute global percentile thresholds from tile prob rasters.')
    parser.add_argument('--probs', nargs='+', required=True, metavar='PATH',
                        help='.npz files from classify_tile_supervised --save_probs')
    parser.add_argument('--out', default='config/vectroscopy_thresholds.json',
                        help='Output JSON path (default: config/vectroscopy_thresholds.json)')
    parser.add_argument('--percentiles', type=int, nargs=3, default=[33, 67, 90],
                        metavar=('P1', 'P2', 'P3'),
                        help='Three percentile values for tiers 1/2/3 (default: 33 67 90)')
    args = parser.parse_args()

    print(f'Pooling probs from {len(args.probs)} tile(s)...')
    pooled = pool_valid_probs(args.probs)

    print(f'Computing {args.percentiles} percentile thresholds per class:')
    thresholds = compute_thresholds(pooled, CLASS_NAMES, args.percentiles)

    # Derive tile ID from npz filename: strip _probs suffix if present
    # e.g. /tmp/t0435_probs.npz → "t0435_probs" → "t0435"
    #      /tmp/t0435_mrral_40s323_0327_4_probs.npz → full tile ID
    def _npz_to_tile_id(p):
        stem = os.path.splitext(os.path.basename(p))[0]
        return stem[:-6] if stem.endswith('_probs') else stem

    tiles_used = [_npz_to_tile_id(p) for p in args.probs]
    write_thresholds_json(args.out, thresholds, tiles_used,
                          args.percentiles, DEFAULT_MORPHOLOGY)
    print(f'Saved → {args.out}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
conda run -n crism pytest tests/test_compute_global_thresholds.py -v 2>&1 | tail -15
```

Expected: `4 passed`.

- [ ] **Step 5: Run on T0434 + T0435 to generate real thresholds**

```bash
mkdir -p /mnt/mrdr/crism_classification/.worktrees/spatial-mae/config
conda run -n crism python scripts/compute_global_thresholds.py \
    --probs /tmp/t0434_mrral_40s318_0327_4_probs.npz /tmp/t0435_mrral_40s323_0327_4_probs.npz \
    --out config/vectroscopy_thresholds.json
```

The `tiles_used` field in the JSON will be `["t0434_mrral_40s318_0327_4", "t0435_mrral_40s323_0327_4"]` (full tile IDs, `_probs` suffix stripped).

Expected output: threshold table printed per class, then `Saved → config/vectroscopy_thresholds.json`.

- [ ] **Step 6: Verify JSON contents**

```bash
cat config/vectroscopy_thresholds.json
```

Expected: valid JSON with `thresholds` key containing 4 minerals each with 3 float values.

- [ ] **Step 7: Commit**

```bash
git add scripts/compute_global_thresholds.py \
        tests/test_compute_global_thresholds.py \
        config/vectroscopy_thresholds.json
git commit -m "feat: add compute_global_thresholds.py with percentile calibration"
```

---

## Chunk 3: Vectorization Script

### Task 4: Write vectorize_tile_minerals.py

**Files:**
- Create: `scripts/vectorize_tile_minerals.py`
- Create: `tests/test_vectorize_tile_minerals.py`

- [ ] **Step 1: Write the failing unit tests**

Create `tests/test_vectorize_tile_minerals.py`:

```python
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_apply_median_filter_no_nan():
    """Median filter runs on finite float array without NaN."""
    from scripts.vectorize_tile_minerals import apply_median_filter

    arr = np.random.rand(20, 20).astype(np.float32)
    result = apply_median_filter(arr, size=3, iterations=2)
    assert result.shape == arr.shape
    assert np.isfinite(result).all(), "median filter should not introduce NaN"


def test_apply_median_filter_preserves_uniform():
    """Median filter leaves a uniform array unchanged."""
    from scripts.vectorize_tile_minerals import apply_median_filter

    arr = np.full((10, 10), 0.5, dtype=np.float32)
    result = apply_median_filter(arr, size=3, iterations=1)
    np.testing.assert_array_almost_equal(result, arr)


def test_assign_confidence_tiers():
    """assign_confidence_tiers maps Threshold float values to int tiers by rank."""
    import geopandas as gpd
    from shapely.geometry import Point
    from scripts.vectorize_tile_minerals import assign_confidence_tiers

    gdf = gpd.GeoDataFrame({
        'geometry': [Point(0, 0), Point(1, 0), Point(2, 0)],
        'Threshold': [0.28, 0.41, 0.57],
    })
    result = assign_confidence_tiers(gdf)
    assert list(result['confidence']) == [1, 2, 3]


def test_assign_confidence_tiers_missing_levels():
    """Works when some tiers have no polygons (e.g. only tiers 1 and 3)."""
    import geopandas as gpd
    from shapely.geometry import Point
    from scripts.vectorize_tile_minerals import assign_confidence_tiers

    gdf = gpd.GeoDataFrame({
        'geometry': [Point(0, 0), Point(1, 0)],
        'Threshold': [0.28, 0.57],
    })
    result = assign_confidence_tiers(gdf)
    # Only two distinct levels; mapped to 1 and 2
    assert set(result['confidence']) == {1, 2}


def test_load_thresholds_json(tmp_path):
    """load_thresholds_json parses JSON and returns thresholds dict."""
    import json
    from scripts.vectorize_tile_minerals import load_thresholds_json

    payload = {
        'thresholds': {
            'olivine': [0.28, 0.41, 0.57],
            'lcp': [0.82, 0.91, 0.96],
            'hcp': [0.04, 0.09, 0.18],
            'plagioclase': [0.03, 0.08, 0.15],
        }
    }
    f = tmp_path / 'thresh.json'
    f.write_text(json.dumps(payload))
    result = load_thresholds_json(str(f))
    assert result['olivine'] == [0.28, 0.41, 0.57]
    assert len(result) == 4
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
conda run -n crism pytest tests/test_vectorize_tile_minerals.py -v 2>&1 | tail -10
```

Expected: `ImportError`.

- [ ] **Step 3: Create scripts/vectorize_tile_minerals.py**

```python
"""
Vectorize per-mineral classifier probability rasters using Vectroscopy.

Reads per-class probability rasters produced by classify_tile_supervised.py --save_probs,
applies global percentile thresholds from compute_global_thresholds.py, and writes
a GeoPackage with one layer per mineral class (olivine, lcp, hcp, plagioclase).

Each polygon carries:
  confidence (int 1-3): model-driven tier (1=low/33rd pctile, 2=medium/67th, 3=high/90th)
  mineral (str): class name
  threshold (float): lower probability bound for this polygon's tier
  mean_prob, std_prob, min_prob, max_prob, median_prob: zonal statistics
  count_px (int): pixel count within polygon

Usage:
    python scripts/vectorize_tile_minerals.py \\
        --tile /mnt/mrdr/mc26/t0435_mrral_40s323_0327_4.img \\
        --probs /tmp/t0435_probs.npz \\
        --thresholds config/vectroscopy_thresholds.json \\
        --out data/vector/t0435_mineral_map.gpkg
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import rasterstats
import rasterio
import scipy.ndimage
import geopandas as gpd

# Vectroscopy: no pip install available — loaded from git clone
_VECTROSCOPY_SRC = os.environ.get('VECTROSCOPY_SRC', '/opt/Vectroscopy/src')
sys.path.insert(0, _VECTROSCOPY_SRC)
try:
    import core.vectroscopy as _vp_module
except ImportError as e:
    raise ImportError(
        f"Cannot import Vectroscopy from {_VECTROSCOPY_SRC}. "
        "Clone the repo: git clone https://github.com/Tahn04/Vectroscopy.git /opt/Vectroscopy"
    ) from e

CLASS_NAMES = ['olivine', 'lcp', 'hcp', 'plagioclase']


# ---------------------------------------------------------------------------
# Helpers (importable for testing)
# ---------------------------------------------------------------------------

def load_thresholds_json(path: str) -> Dict[str, List[float]]:
    """Load and return the thresholds dict from a vectroscopy_thresholds.json file.

    Returns:
        dict mapping mineral name → [t1, t2, t3] float list
    """
    with open(path) as f:
        data = json.load(f)
    return data['thresholds']


def apply_median_filter(arr: np.ndarray, size: int, iterations: int) -> np.ndarray:
    """Apply scipy median filter N times on a finite float array.

    Must be called BEFORE applying NaN mask (scipy median_filter does not handle NaN).

    Args:
        arr: (H, W) float32 array with no NaN values
        size: filter kernel size (scalar, applied to both axes)
        iterations: number of times to apply the filter

    Returns:
        filtered (H, W) float32 array
    """
    result = arr.copy()
    for _ in range(iterations):
        result = scipy.ndimage.median_filter(result, size=size)
    return result


def assign_confidence_tiers(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Map Vectroscopy 'Threshold' float column to integer confidence tiers by rank.

    Vectroscopy stores the float threshold value in the 'Threshold' column.
    We rank unique values ascending: lowest float → tier 1, next → tier 2, etc.
    This avoids floating-point equality comparisons.

    Args:
        gdf: GeoDataFrame with 'Threshold' column (float values matching t1/t2/t3)

    Returns:
        gdf with new 'confidence' column (int 1-N, where N ≤ 3)
    """
    unique_t = sorted(gdf['Threshold'].unique())
    tier_map = {v: i + 1 for i, v in enumerate(unique_t)}
    result = gdf.copy()
    result['confidence'] = result['Threshold'].map(tier_map)
    return result


def vectorize_mineral(
    prob_2d: np.ndarray,
    valid_mask: np.ndarray,
    thresholds: List[float],
    mineral: str,
    input_crs,
    input_transform,
    median_size: int = 3,
    median_iter: int = 1,
) -> gpd.GeoDataFrame:
    """Run full per-mineral vectorization pipeline.

    Processing order:
      1. Median filter on finite prob_2d (before NaN masking)
      2. Apply NaN to invalid pixels
      3. Vectroscopy vectorize → GeoDataFrame in geographic CRS
      4. Reproject back to tile projected CRS
      5. Assign confidence tiers
      6. Compute zonal statistics
      7. Simplify geometry (200m tolerance, after zonal stats)

    Args:
        prob_2d: (H, W) float32 probability raster, finite (no NaN) on entry
        valid_mask: (H, W) bool, True = valid pixel
        thresholds: [t1, t2, t3] float values (33rd/67th/90th percentiles)
        mineral: class name string
        input_crs: rasterio.crs.CRS of the tile
        input_transform: rasterio.transform.Affine of the tile
        median_size: median filter kernel size
        median_iter: number of median filter iterations

    Returns:
        GeoDataFrame with columns: geometry, confidence, mineral, threshold,
        mean_prob, std_prob, min_prob, max_prob, median_prob, count_px.
        Empty GeoDataFrame if no pixels exceed thresholds[0].
    """
    # Step 1: median filter on finite array
    filtered = apply_median_filter(prob_2d, size=median_size, iterations=median_iter)

    # Step 2: mask nodata pixels
    filtered[~valid_mask] = np.nan

    # Step 3: vectorize
    gdf = _vp_module.Vectroscopy.from_array(
        array=filtered,
        thresholds=thresholds,
        crs=input_crs,
        transform=input_transform,
        name=mineral,
    ).vectorize()

    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame()

    # Step 4: reproject back to tile projected CRS
    # (Vectroscopy reprojects to geographic CRS by default)
    gdf = gdf.to_crs(input_crs)

    # Step 5: confidence tiers
    gdf = assign_confidence_tiers(gdf)
    gdf['mineral'] = mineral

    # Step 6: zonal statistics from median-filtered array (with NaN nodata)
    stats = rasterstats.zonal_stats(
        vectors=gdf.geometry,
        raster=filtered,
        affine=input_transform,
        stats=['mean', 'std', 'min', 'max', 'median', 'count'],
        nodata=np.nan,
        all_touched=False,
    )
    stats_df = pd.DataFrame(stats).rename(columns={
        'mean': 'mean_prob', 'std': 'std_prob', 'min': 'min_prob',
        'max': 'max_prob', 'median': 'median_prob', 'count': 'count_px',
    })
    gdf = pd.concat([gdf.reset_index(drop=True), stats_df], axis=1)

    # Step 7: simplify geometry AFTER zonal stats (stored geometry matches stats)
    gdf['geometry'] = gdf['geometry'].simplify(tolerance=200, preserve_topology=True)

    # Finalise column selection and rename Threshold → threshold
    keep = ['geometry', 'confidence', 'mineral', 'Threshold',
            'mean_prob', 'std_prob', 'min_prob', 'max_prob', 'median_prob', 'count_px']
    gdf = gdf[[c for c in keep if c in gdf.columns]].rename(
        columns={'Threshold': 'threshold'})

    return gdf


def load_probs_npz(path: str) -> Tuple[np.ndarray, np.ndarray, object, object]:
    """Load probs .npz; return (probs, valid_mask, crs, transform).

    Returns:
        probs: (H, W, 4) float32
        valid_mask: (H, W) bool
        crs: rasterio.crs.CRS
        transform: rasterio.transform.Affine
    """
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    data = np.load(path)
    probs = data['probs']
    valid_mask = data['valid_mask']
    crs = CRS.from_wkt(str(data['crs_wkt']))
    a, b, c, d, e, f = data['transform']
    transform = Affine(a, b, c, d, e, f)
    return probs, valid_mask, crs, transform


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Vectorize per-mineral probability rasters using Vectroscopy.')
    parser.add_argument('--tile', required=True,
                        help='Path to mrral .img tile (always required)')
    parser.add_argument('--probs', default=None, metavar='PATH',
                        help='.npz from classify_tile_supervised --save_probs (optional)')
    parser.add_argument('--ckpt', default=None, metavar='PATH',
                        help='Classifier checkpoint (required iff --probs absent)')
    parser.add_argument('--thresholds', required=True, metavar='JSON',
                        help='Path to vectroscopy_thresholds.json')
    parser.add_argument('--out', required=True, metavar='GPKG',
                        help='Output GeoPackage path')
    parser.add_argument('--median_size', type=int, default=3)
    parser.add_argument('--median_iter', type=int, default=1)
    parser.add_argument('--sieve_px', type=int, default=9)
    parser.add_argument('--majority_iter', type=int, default=3)
    args = parser.parse_args()

    # Validate probs/ckpt logic
    if args.probs is None and args.ckpt is None:
        parser.error('--ckpt is required when --probs is not supplied')

    # Load CRS and transform from tile (authoritative source)
    with rasterio.open(args.tile) as src:
        input_crs = src.crs
        input_transform = src.transform

    # Load or compute probs
    if args.probs:
        print(f'Loading probs from {args.probs}')
        probs, valid_mask, _, _ = load_probs_npz(args.probs)
    else:
        print('Running inference inline...')
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from scripts.classify_tile_supervised import (
            load_tile, load_classifier, run_supervised
        )
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tile, valid_mask, _, _ = load_tile(args.tile)
        H, W = valid_mask.shape
        model = load_classifier(args.ckpt, device)
        probs_flat = run_supervised(tile, model, device)  # (H*W, 5)
        probs = probs_flat.reshape(H, W, 5)[:, :, :4]    # (H, W, 4) drop "other"

    H, W = valid_mask.shape
    print(f'Tile: {H}×{W}, {valid_mask.sum():,} valid pixels')

    # Load thresholds
    thresholds_cfg = load_thresholds_json(args.thresholds)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    for ci, mineral in enumerate(CLASS_NAMES):
        t1, t2, t3 = thresholds_cfg[mineral]
        print(f'Vectorizing {mineral} (thresholds: {t1:.4f}/{t2:.4f}/{t3:.4f})...')

        prob_2d = probs[:, :, ci].copy().astype(np.float32)

        gdf = vectorize_mineral(
            prob_2d=prob_2d,
            valid_mask=valid_mask,
            thresholds=[t1, t2, t3],
            mineral=mineral,
            input_crs=input_crs,
            input_transform=input_transform,
            median_size=args.median_size,
            median_iter=args.median_iter,
        )

        if gdf.empty:
            print(f'  {mineral}: no polygons detected above threshold {t1:.4f}')
            continue

        print(f'  {mineral}: {len(gdf)} polygons '
              f'(tier 1: {(gdf["confidence"]==1).sum()}, '
              f'tier 2: {(gdf["confidence"]==2).sum()}, '
              f'tier 3: {(gdf["confidence"]==3).sum()})')

        gdf.to_file(args.out, layer=mineral, driver='GPKG')

    print(f'Saved → {args.out}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run unit tests to confirm they pass**

```bash
conda run -n crism pytest tests/test_vectorize_tile_minerals.py -v 2>&1 | tail -15
```

Expected: `4 passed` (or 5 if all tests run). If any Vectroscopy-dependent test fails due to import issues, verify `/opt/Vectroscopy/src` is correct.

- [ ] **Step 5: Commit**

```bash
git add scripts/vectorize_tile_minerals.py \
        tests/test_vectorize_tile_minerals.py
git commit -m "feat: add vectorize_tile_minerals.py with Vectroscopy integration"
```

---

## Chunk 4: End-to-End Pipeline Run

### Task 5: Run full pipeline on T0435 and T0434

**Files:**
- Create (generated): `data/vector/t0435_mrral_40s323_0327_4_mineral_map.gpkg`
- Create (generated): `data/vector/t0434_mrral_40s318_0327_4_mineral_map.gpkg`

- [ ] **Step 1: Ensure probs exist for both tiles**

```bash
ls -lh /tmp/t043{4,5}_mrral_*_probs.npz
```

If either is missing, re-run the classify step from Task 2, Step 5/7.

- [ ] **Step 2: Run vectorization on T0435**

```bash
mkdir -p /mnt/mrdr/crism_classification/.worktrees/spatial-mae/data/vector

conda run -n crism python scripts/vectorize_tile_minerals.py \
    --tile /mnt/mrdr/mc26/t0435_mrral_40s323_0327_4.img \
    --probs /tmp/t0435_mrral_40s323_0327_4_probs.npz \
    --thresholds config/vectroscopy_thresholds.json \
    --out data/vector/t0435_mrral_40s323_0327_4_mineral_map.gpkg
```

Expected: prints polygon counts per mineral per tier, then `Saved → data/vector/...`.

- [ ] **Step 3: Run vectorization on T0434**

```bash
conda run -n crism python scripts/vectorize_tile_minerals.py \
    --tile /mnt/mrdr/mc26/t0434_mrral_40s318_0327_4.img \
    --probs /tmp/t0434_mrral_40s318_0327_4_probs.npz \
    --thresholds config/vectroscopy_thresholds.json \
    --out data/vector/t0434_mrral_40s318_0327_4_mineral_map.gpkg
```

- [ ] **Step 4: Verify output GeoPackages**

```bash
conda run -n crism python3 -c "
import geopandas as gpd

for gpkg in [
    'data/vector/t0435_mrral_40s323_0327_4_mineral_map.gpkg',
    'data/vector/t0434_mrral_40s318_0327_4_mineral_map.gpkg',
]:
    print(f'\n=== {gpkg} ===')
    import fiona
    layers = fiona.listlayers(gpkg)
    print(f'Layers: {layers}')
    for layer in layers:
        gdf = gpd.read_file(gpkg, layer=layer)
        print(f'  {layer}: {len(gdf)} polygons, cols={list(gdf.columns)}')
        print(f'    confidence tiers: {dict(gdf[\"confidence\"].value_counts().sort_index())}')
        print(f'    mean_prob range: [{gdf[\"mean_prob\"].min():.3f}, {gdf[\"mean_prob\"].max():.3f}]')
"
```

Expected: 4 layers per GeoPackage, each with columns including `confidence`, `mineral`, `threshold`, `mean_prob`, `count_px`. Confidence values should be 1, 2, or 3.

- [ ] **Step 5: Spot-check CRS is correct (projected, not geographic)**

```bash
conda run -n crism python3 -c "
import geopandas as gpd
gdf = gpd.read_file(
    'data/vector/t0435_mrral_40s323_0327_4_mineral_map.gpkg',
    layer='olivine'
)
print('CRS:', gdf.crs)
print('Bounds:', gdf.total_bounds)
"
```

Expected: CRS should be a projected CRS (Mars equirectangular, not GEOGCS). Bounds should be in large metre values (~100000s), not degrees (-180 to 180).

- [ ] **Step 6: Commit outputs and all generated files**

```bash
git add scripts/vectorize_tile_minerals.py \
        scripts/compute_global_thresholds.py \
        config/vectroscopy_thresholds.json
git commit -m "feat: complete Vectroscopy integration pipeline — T0435 and T0434 vector products"
```

Note: do not commit the `data/vector/` GeoPackages to git (binary files, potentially large). Add to `.gitignore` if not already present:

```bash
echo 'data/vector/' >> .gitignore
git add .gitignore
git commit -m "chore: ignore data/vector output directory"
```

---

## Troubleshooting Reference

**Vectroscopy import fails:**
```bash
ls /opt/Vectroscopy/src/core/vectroscopy.py   # verify clone location
export VECTROSCOPY_SRC=/opt/Vectroscopy/src   # override default path
```

**`rasterstats` zonal_stats returns all None:**
Confirm `prob_2d` has the same spatial extent and resolution as `input_transform`. Check that valid_mask is correctly applied before passing.

**Output CRS is geographic (degrees) not projected:**
The `gdf.to_crs(input_crs)` step in `vectorize_mineral` is the reproject-back call. If it seems like a no-op, verify that `input_crs` is a projected CRS by printing `input_crs.is_projected`.

**Vectroscopy returns empty GeoDataFrame for all minerals:**
Check that the computed thresholds are not higher than all pixel values. Print `prob_2d[valid_mask].max()` and compare to `t1`. If LCP thresholds are very high (>0.99), no pixels may exceed them — this can happen if the test tile is the same as the calibration tiles.

**`Affine` reconstruction from .npz:**
If needed elsewhere: `from rasterio.transform import Affine; t = Affine(*data['transform'])` using the (a,b,c,d,e,f) rasterio ordering saved in Stage 1.

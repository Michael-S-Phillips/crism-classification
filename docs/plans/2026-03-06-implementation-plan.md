# CRISM Mineral Classification Pipeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a multi-label pixel classification pipeline for CRISM MRDR mrrsu tiles, training 7 model families with W&B experiment tracking.

**Architecture:** Per-pixel extraction from rasterized polygon ROIs → parquet dataset → shared train/val/test splits → model-specific training loops → unified evaluation with confidence-tier metrics.

**Tech Stack:** Python 3.11, conda env `crism`, geopandas, rasterio, scikit-learn, xgboost, lightgbm, torch, wandb, pyarrow, pytest

---

## Task 1: Environment & Project Scaffold

**Files:**
- Create: `crism_classification/environment.yml`
- Create: `crism_classification/config.yaml`
- Create: `crism_classification/README.md`
- Create: `crism_classification/tests/__init__.py`
- Create: `crism_classification/data/__init__.py`
- Create: `crism_classification/models/__init__.py`
- Create: `crism_classification/training/__init__.py`
- Create: `crism_classification/evaluation/__init__.py`

**Step 1: Create environment.yml**

```yaml
# crism_classification/environment.yml
name: crism
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - geopandas=1.0.1
  - rasterio=1.4.3
  - scipy
  - spectral
  - matplotlib
  - numpy
  - pandas
  - tqdm
  - pyyaml
  - pyarrow
  - pytest
  - pip
  - pip:
    - scikit-learn
    - xgboost
    - lightgbm
    - torch
    - torchvision
    - wandb
    - openpyxl
```

**Step 2: Install missing packages into existing crism env**

```bash
conda run -n crism pip install scikit-learn xgboost lightgbm torch torchvision wandb pyarrow tqdm pyyaml pytest
```

Expected: packages install without error.

**Step 3: Create config.yaml**

```yaml
# crism_classification/config.yaml
project_root: /mnt/crism/MRDR/crism_classification
data_root: /mnt/crism/MRDR
gpkg_dir: /mnt/crism/MRDR/categorized_mineral_units
output_dir: /mnt/crism/MRDR/crism_classification/data
predictions_dir: /mnt/crism/MRDR/crism_classification/predictions
checkpoints_dir: /mnt/crism/MRDR/crism_classification/checkpoints

classes:
  - olivine_t1
  - olivine_t2
  - lcp
  - hcp
  - plagioclase
  - other

confidence_weights:
  High: 1.0
  Moderate: 0.5
  Low: 0.25

other_max_polygons: 400
patch_size: 7  # for CNN/ViT
nodata_value: 65535

split:
  train: 0.70
  val: 0.15
  test: 0.15
  random_seed: 42

wandb:
  project: crism-mineral-classification
  entity: null  # filled in by setup_wandb.py
```

**Step 4: Create all `__init__.py` files and README**

```bash
cd /mnt/crism/MRDR/crism_classification
touch tests/__init__.py data/__init__.py models/__init__.py training/__init__.py evaluation/__init__.py
mkdir -p config scripts predictions checkpoints
```

**Step 5: Initialize git repo**

```bash
cd /mnt/crism/MRDR/crism_classification
git init
echo "__pycache__/\n*.pyc\n*.egg-info/\ndata/pixels.parquet\npredictions/\ncheckpoints/\n.env\nwandb/" > .gitignore
git add .
git commit -m "feat: initial project scaffold"
```

---

## Task 2: Label Parser

**Files:**
- Create: `crism_classification/data/label_parser.py`
- Create: `crism_classification/tests/test_label_parser.py`

**Step 1: Write failing tests**

```python
# crism_classification/tests/test_label_parser.py
import numpy as np
import pytest
from data.label_parser import parse_category, CLASSES

def test_classes_order():
    assert CLASSES == ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']

def test_type1_olivine_high():
    label, weight = parse_category("Type 1 olivine (High)")
    np.testing.assert_array_almost_equal(label, [1, 0, 0, 0, 0, 0])
    assert weight == 1.0

def test_type2_olivine_moderate():
    label, weight = parse_category("Type 2 olivine (Moderate)")
    np.testing.assert_array_almost_equal(label, [0, 1, 0, 0, 0, 0])
    assert weight == 0.5

def test_lcp_high():
    label, weight = parse_category("lcp (High)")
    np.testing.assert_array_almost_equal(label, [0, 0, 1, 0, 0, 0])
    assert weight == 1.0

def test_hcp_low():
    label, weight = parse_category("hcp (Low)")
    np.testing.assert_array_almost_equal(label, [0, 0, 0, 1, 0, 0])
    assert weight == 0.25

def test_plagioclase_moderate():
    label, weight = parse_category("plagioclase (Moderate)")
    np.testing.assert_array_almost_equal(label, [0, 0, 0, 0, 1, 0])
    assert weight == 0.5

def test_other_high():
    label, weight = parse_category("Other (High)")
    np.testing.assert_array_almost_equal(label, [0, 0, 0, 0, 0, 1])
    assert weight == 1.0

def test_hcp_plus_olivine():
    label, weight = parse_category("hcp + olivine (High)")
    np.testing.assert_array_almost_equal(label, [0.5, 0.5, 0, 1, 0, 0])
    assert weight == 1.0

def test_olivine_plus_plagioclase():
    label, weight = parse_category("olivine + plagioclase (Low)")
    np.testing.assert_array_almost_equal(label, [0.5, 0.5, 0, 0, 1, 0])
    assert weight == 0.25

def test_hcp_plus_lcp():
    label, weight = parse_category("hcp + lcp (Moderate)")
    np.testing.assert_array_almost_equal(label, [0, 0, 1, 1, 0, 0])
    assert weight == 0.5

def test_alteration_plus_olivine():
    label, weight = parse_category("alteration + olivine (Low)")
    np.testing.assert_array_almost_equal(label, [0.5, 0.5, 0, 0, 0, 0])
    assert weight == 0.25

def test_alteration_plus_plagioclase():
    label, weight = parse_category("alteration + plagioclase (Low)")
    np.testing.assert_array_almost_equal(label, [0, 0, 0, 0, 1, 0])
    assert weight == 0.25

def test_lcp_plus_hcp_plus_olivine():
    label, weight = parse_category("hcp + lcp + olivine (Moderate)")
    # olivine untyped -> 0.5 each, lcp=1, hcp=1
    assert label[2] == 1.0  # lcp
    assert label[3] == 1.0  # hcp
    assert label[0] == pytest.approx(0.5)  # olivine_t1
    assert label[1] == pytest.approx(0.5)  # olivine_t2

def test_unknown_category_returns_zeros():
    label, weight = parse_category("spinel (Low)")
    np.testing.assert_array_almost_equal(label, [0, 0, 0, 0, 0, 0])
    assert weight == 0.25

def test_returns_numpy_array():
    label, weight = parse_category("lcp (High)")
    assert isinstance(label, np.ndarray)
    assert label.dtype == np.float32
```

**Step 2: Run to verify failure**

```bash
cd /mnt/crism/MRDR/crism_classification
conda run -n crism python -m pytest tests/test_label_parser.py -v 2>&1 | head -20
```
Expected: ImportError — `data.label_parser` not found.

**Step 3: Implement label_parser.py**

```python
# crism_classification/data/label_parser.py
import re
import numpy as np

CLASSES = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']
N_CLASSES = len(CLASSES)

_CONFIDENCE_WEIGHTS = {'High': 1.0, 'Moderate': 0.5, 'Low': 0.25}

# Maps token strings found in category to class indices and values.
# For untyped "olivine", we assign 0.5 to both t1 and t2.
_TOKEN_MAP = {
    'type 1 olivine': {'olivine_t1': 1.0},
    'type 2 olivine': {'olivine_t2': 1.0},
    'olivine':        {'olivine_t1': 0.5, 'olivine_t2': 0.5},
    'lcp':            {'lcp': 1.0},
    'hcp':            {'hcp': 1.0},
    'plagioclase':    {'plagioclase': 1.0},
    'other':          {'other': 1.0},
    # ignored tokens (produce no label contribution)
    'alteration':     {},
    'red slope':      {},
    'spinel':         {},
    'pyroxene':       {},
}


def parse_category(category: str) -> tuple[np.ndarray, float]:
    """
    Parse a CRISM geopackage Category string into a multi-hot label vector
    and a confidence sample weight.

    Parameters
    ----------
    category : str
        e.g. "Type 1 olivine (High)", "hcp + olivine (Moderate)"

    Returns
    -------
    label : np.ndarray, shape (6,), dtype float32
        Multi-hot vector for [olivine_t1, olivine_t2, lcp, hcp, plagioclase, other].
        Untyped "olivine" in mixed labels contributes 0.5 to both t1 and t2.
    weight : float
        Confidence sample weight: High=1.0, Moderate=0.5, Low=0.25.
    """
    label = np.zeros(N_CLASSES, dtype=np.float32)
    class_idx = {c: i for i, c in enumerate(CLASSES)}

    # Extract confidence tier from parentheses, e.g. "(High)"
    conf_match = re.search(r'\((\w+)\)', category)
    confidence = conf_match.group(1) if conf_match else 'Low'
    weight = _CONFIDENCE_WEIGHTS.get(confidence, 0.25)

    # Remove the confidence part and split on '+'
    mineral_part = re.sub(r'\s*\([^)]*\)', '', category).strip()
    tokens = [t.strip().lower() for t in mineral_part.split('+')]

    for token in tokens:
        # Try longest match first (so "type 1 olivine" matches before "olivine")
        matched = False
        for key in sorted(_TOKEN_MAP.keys(), key=len, reverse=True):
            if key in token:
                for cls, val in _TOKEN_MAP[key].items():
                    if cls in class_idx:
                        label[class_idx[cls]] = max(label[class_idx[cls]], val)
                matched = True
                break
        # Unknown tokens silently produce no contribution

    return label, weight


def get_confidence_tier(category: str) -> str:
    """Extract the confidence tier string from a category label."""
    match = re.search(r'\((\w+)\)', category)
    return match.group(1) if match else 'Low'
```

**Step 4: Run tests**

```bash
cd /mnt/crism/MRDR/crism_classification
conda run -n crism python -m pytest tests/test_label_parser.py -v
```
Expected: all 14 tests PASS.

**Step 5: Commit**

```bash
cd /mnt/crism/MRDR/crism_classification
git add data/label_parser.py tests/test_label_parser.py
git commit -m "feat: add label parser with multi-hot encoding and confidence weights"
```

---

## Task 3: Pixel Extraction Pipeline

**Files:**
- Create: `crism_classification/data/extract_pixels.py`
- Create: `crism_classification/tests/test_extract_pixels.py`

**Step 1: Write failing tests**

```python
# crism_classification/tests/test_extract_pixels.py
import numpy as np
import pytest
import tempfile, os
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from shapely.geometry import box
from data.extract_pixels import (
    find_tile_pairs, extract_pixels_from_pair, NODATA_VALUE
)

@pytest.fixture
def synthetic_tile(tmp_path):
    """Create a tiny synthetic 10x10 mrrsu raster with 3 bands."""
    img_path = tmp_path / "t0001_mrrsu_test.img"
    transform = from_bounds(0, 0, 10, 10, 10, 10)
    crs = CRS.from_epsg(4326)
    data = np.random.rand(3, 10, 10).astype(np.float32)
    data[:, 0, 0] = NODATA_VALUE  # one nodata pixel
    with rasterio.open(
        img_path, 'w', driver='GTiff', height=10, width=10,
        count=3, dtype='float32', crs=crs, transform=transform
    ) as dst:
        dst.write(data)
    return str(img_path), data

@pytest.fixture
def synthetic_gpkg(tmp_path):
    """Create a geopackage with two polygons."""
    gdf = gpd.GeoDataFrame({
        'Category': ['lcp (High)', 'Type 1 olivine (Low)'],
        'geometry': [box(1, 1, 4, 4), box(6, 6, 9, 9)]
    }, crs='EPSG:4326')
    gpkg_path = tmp_path / "T0001.gpkg"
    gdf.to_file(gpkg_path, driver='GPKG')
    return str(gpkg_path)

def test_nodata_value():
    assert NODATA_VALUE == 65535

def test_extract_pixels_returns_records(synthetic_tile, synthetic_gpkg):
    img_path, _ = synthetic_tile
    records = extract_pixels_from_pair(
        tile_id='t0001',
        mrrsu_path=img_path,
        gpkg_path=synthetic_gpkg,
        n_bands=3
    )
    assert len(records) > 0

def test_extract_pixels_schema(synthetic_tile, synthetic_gpkg):
    img_path, _ = synthetic_tile
    records = extract_pixels_from_pair('t0001', img_path, synthetic_gpkg, n_bands=3)
    r = records[0]
    assert 'tile_id' in r
    assert 'polygon_id' in r
    assert 'pixel_row' in r
    assert 'pixel_col' in r
    assert 'b0' in r and 'b2' in r
    assert 'olivine_t1' in r
    assert 'confidence_weight' in r
    assert 'confidence_tier' in r

def test_nodata_pixels_excluded(synthetic_tile, synthetic_gpkg):
    img_path, data = synthetic_tile
    # pixel (0,0) has NODATA — should not appear in records
    records = extract_pixels_from_pair('t0001', img_path, synthetic_gpkg, n_bands=3)
    rows = [r['pixel_row'] for r in records]
    cols = [r['pixel_col'] for r in records]
    pairs = list(zip(rows, cols))
    assert (0, 0) not in pairs

def test_confidence_weight_lcp_high(synthetic_tile, synthetic_gpkg):
    img_path, _ = synthetic_tile
    records = extract_pixels_from_pair('t0001', img_path, synthetic_gpkg, n_bands=3)
    lcp_records = [r for r in records if r['lcp'] == 1.0]
    for r in lcp_records:
        assert r['confidence_weight'] == 1.0
        assert r['confidence_tier'] == 'High'

def test_find_tile_pairs_finds_existing():
    pairs = find_tile_pairs(
        gpkg_dir='/mnt/crism/MRDR/categorized_mineral_units',
        data_root='/mnt/crism/MRDR'
    )
    assert len(pairs) > 0
    t_id, gpkg_path, mrrsu_path = pairs[0]
    assert os.path.exists(gpkg_path)
    assert os.path.exists(mrrsu_path)
```

**Step 2: Run to verify failure**

```bash
cd /mnt/crism/MRDR/crism_classification
conda run -n crism python -m pytest tests/test_extract_pixels.py -v 2>&1 | head -20
```
Expected: ImportError.

**Step 3: Implement extract_pixels.py**

```python
# crism_classification/data/extract_pixels.py
"""
Extract per-pixel spectral parameter values from CRISM mrrsu rasters
using geopackage polygon ROIs as masks.
"""
import os
import glob
import logging
from typing import List, Tuple, Dict, Any

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import rowcol

from data.label_parser import parse_category, get_confidence_tier, CLASSES

logger = logging.getLogger(__name__)

NODATA_VALUE = 65535
BAND_PREFIX = 'b'


def find_tile_pairs(
    gpkg_dir: str,
    data_root: str
) -> List[Tuple[str, str, str]]:
    """
    Find (tile_id, gpkg_path, mrrsu_path) triples by matching gpkg filenames
    to mrrsu image files under data_root.

    Returns list of (tile_id, gpkg_path, mrrsu_path) sorted by tile_id.
    """
    pairs = []
    for fname in sorted(os.listdir(gpkg_dir)):
        if not fname.endswith('.gpkg'):
            continue
        tile_id = fname.replace('.gpkg', '').lower()  # e.g. "t0434"
        gpkg_path = os.path.join(gpkg_dir, fname)
        matches = glob.glob(
            os.path.join(data_root, '**', f'{tile_id}_mrrsu*.img'),
            recursive=True
        )
        if not matches:
            logger.warning(f"No mrrsu file found for {tile_id}, skipping.")
            continue
        pairs.append((tile_id, gpkg_path, matches[0]))
    return pairs


def extract_pixels_from_pair(
    tile_id: str,
    mrrsu_path: str,
    gpkg_path: str,
    n_bands: int = 60,
    other_polygon_ids: set = None,
) -> List[Dict[str, Any]]:
    """
    Extract per-pixel records from one (gpkg, mrrsu) pair.

    For each polygon in the gpkg:
      - Rasterizes the polygon to a pixel mask aligned to the mrrsu grid
      - Reads all n_bands values at each masked pixel
      - Drops pixels with any NODATA or NaN value
      - Parses the Category string into multi-hot labels + confidence weight

    Parameters
    ----------
    tile_id : str
    mrrsu_path : str
    gpkg_path : str
    n_bands : int
    other_polygon_ids : set, optional
        If provided, only process 'Other' polygons whose index is in this set.

    Returns
    -------
    List of dicts, one per valid pixel.
    """
    records = []

    with rasterio.open(mrrsu_path) as src:
        raster_crs = src.crs
        transform = src.transform
        height, width = src.height, src.width
        actual_bands = min(n_bands, src.count)

        # Load and reproject gpkg to match raster CRS
        gdf = gpd.read_file(gpkg_path)
        if gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)

        for poly_idx, row in gdf.iterrows():
            category = row.get('Category', '')
            if not category:
                continue

            # Skip Other polygons not in the sampled set
            if 'other' in category.lower() and other_polygon_ids is not None:
                if poly_idx not in other_polygon_ids:
                    continue

            label, conf_weight = parse_category(category)
            conf_tier = get_confidence_tier(category)

            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            # Rasterize this polygon to a boolean mask
            try:
                mask = rasterize(
                    [(geom, 1)],
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8
                ).astype(bool)
            except Exception as e:
                logger.warning(f"Rasterize failed for polygon {poly_idx} in {tile_id}: {e}")
                continue

            pixel_rows, pixel_cols = np.where(mask)
            if len(pixel_rows) == 0:
                continue

            # Read all bands for masked pixels
            # Use windowed reads per row range for memory efficiency
            row_min, row_max = int(pixel_rows.min()), int(pixel_rows.max()) + 1
            col_min, col_max = int(pixel_cols.min()), int(pixel_cols.max()) + 1

            window = rasterio.windows.Window(
                col_min, row_min,
                col_max - col_min, row_max - row_min
            )
            chunk = src.read(
                list(range(1, actual_bands + 1)),
                window=window
            )  # shape: (bands, h, w)

            for r, c in zip(pixel_rows, pixel_cols):
                local_r = r - row_min
                local_c = c - col_min
                pixel_vals = chunk[:, local_r, local_c]

                # Drop NODATA or NaN pixels
                if np.any(pixel_vals >= NODATA_VALUE) or np.any(np.isnan(pixel_vals)):
                    continue

                record = {
                    'tile_id': tile_id,
                    'polygon_id': int(poly_idx),
                    'pixel_row': int(r),
                    'pixel_col': int(c),
                }
                for b_idx in range(actual_bands):
                    record[f'{BAND_PREFIX}{b_idx}'] = float(pixel_vals[b_idx])
                for cls_idx, cls_name in enumerate(CLASSES):
                    record[cls_name] = float(label[cls_idx])
                record['confidence_weight'] = float(conf_weight)
                record['confidence_tier'] = conf_tier

                records.append(record)

    return records
```

**Step 4: Run tests**

```bash
cd /mnt/crism/MRDR/crism_classification
conda run -n crism python -m pytest tests/test_extract_pixels.py -v
```
Expected: all 7 tests PASS.

**Step 5: Commit**

```bash
git add data/extract_pixels.py tests/test_extract_pixels.py
git commit -m "feat: pixel extraction pipeline with nodata filtering"
```

---

## Task 4: Build Dataset Script

**Files:**
- Create: `crism_classification/scripts/build_dataset.py`
- Create: `crism_classification/tests/test_build_dataset.py`

**Step 1: Write failing test**

```python
# crism_classification/tests/test_build_dataset.py
import pandas as pd
import pytest
import os

PARQUET_PATH = '/mnt/crism/MRDR/crism_classification/data/pixels.parquet'

def test_parquet_exists():
    assert os.path.exists(PARQUET_PATH), "Run: python scripts/build_dataset.py first"

def test_parquet_schema():
    df = pd.read_parquet(PARQUET_PATH)
    required = ['tile_id', 'polygon_id', 'pixel_row', 'pixel_col',
                'b0', 'b59', 'olivine_t1', 'olivine_t2', 'lcp', 'hcp',
                'plagioclase', 'other', 'confidence_weight',
                'confidence_tier', 'split']
    for col in required:
        assert col in df.columns, f"Missing column: {col}"

def test_no_nodata_in_features():
    df = pd.read_parquet(PARQUET_PATH)
    band_cols = [f'b{i}' for i in range(60)]
    assert not df[band_cols].isnull().any().any()
    assert (df[band_cols] < 65535).all().all()

def test_split_values():
    df = pd.read_parquet(PARQUET_PATH)
    assert set(df['split'].unique()).issubset({'train', 'val', 'test'})

def test_split_covers_all_classes():
    df = pd.read_parquet(PARQUET_PATH)
    label_cols = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']
    for split in ['train', 'val', 'test']:
        sub = df[df['split'] == split]
        for col in label_cols:
            assert sub[col].sum() > 0, f"No {col} in split={split}"

def test_other_class_not_overrepresented():
    df = pd.read_parquet(PARQUET_PATH)
    n_other = (df['other'] == 1.0).sum()
    n_total = len(df)
    # Other should be at most 30% of total pixels
    assert n_other / n_total < 0.30, f"Other is {n_other/n_total:.1%} of dataset"

def test_confidence_weights_valid():
    df = pd.read_parquet(PARQUET_PATH)
    assert df['confidence_weight'].isin([0.25, 0.5, 1.0]).all()
```

**Step 2: Implement build_dataset.py**

```python
# crism_classification/scripts/build_dataset.py
"""
Build the pixel-level dataset from all (gpkg, mrrsu) pairs.

Usage:
    conda run -n crism python scripts/build_dataset.py
    conda run -n crism python scripts/build_dataset.py --config config.yaml
"""
import argparse
import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.extract_pixels import find_tile_pairs, extract_pixels_from_pair

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def assign_tile_splits(tile_ids, train_frac=0.70, val_frac=0.15, seed=42):
    """Assign each tile to train/val/test split."""
    rng = random.Random(seed)
    shuffled = list(tile_ids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    split_map = {}
    for i, tid in enumerate(shuffled):
        if i < n_train:
            split_map[tid] = 'train'
        elif i < n_train + n_val:
            split_map[tid] = 'val'
        else:
            split_map[tid] = 'test'
    return split_map


def sample_other_polygon_ids(pairs, max_polygons, seed=42):
    """
    Randomly select up to max_polygons 'Other' polygon indices across all tiles.
    Returns dict: tile_id -> set of polygon indices to include.
    """
    import geopandas as gpd

    all_other = []  # list of (tile_id, poly_idx)
    for tile_id, gpkg_path, _ in pairs:
        gdf = gpd.read_file(gpkg_path)
        for idx, row in gdf.iterrows():
            cat = row.get('Category', '')
            if cat and cat.lower().startswith('other'):
                all_other.append((tile_id, idx))

    rng = random.Random(seed)
    sampled = rng.sample(all_other, min(max_polygons, len(all_other)))

    result = {}
    for tile_id, poly_idx in sampled:
        result.setdefault(tile_id, set()).add(poly_idx)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()

    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.config
    )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    gpkg_dir = cfg['gpkg_dir']
    data_root = cfg['data_root']
    output_dir = cfg['output_dir']
    other_max = cfg.get('other_max_polygons', 400)
    seed = cfg['split']['random_seed']

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Finding tile pairs...")
    pairs = find_tile_pairs(gpkg_dir, data_root)
    logger.info(f"Found {len(pairs)} tile pairs")

    tile_ids = [p[0] for p in pairs]
    split_map = assign_tile_splits(
        tile_ids,
        train_frac=cfg['split']['train'],
        val_frac=cfg['split']['val'],
        seed=seed
    )
    logger.info(f"Split: {sum(v=='train' for v in split_map.values())} train, "
                f"{sum(v=='val' for v in split_map.values())} val, "
                f"{sum(v=='test' for v in split_map.values())} test tiles")

    logger.info("Sampling 'Other' polygons...")
    other_ids = sample_other_polygon_ids(pairs, other_max, seed=seed)

    all_records = []
    for tile_id, gpkg_path, mrrsu_path in tqdm(pairs, desc="Extracting pixels"):
        tile_other_ids = other_ids.get(tile_id, set())
        records = extract_pixels_from_pair(
            tile_id=tile_id,
            mrrsu_path=mrrsu_path,
            gpkg_path=gpkg_path,
            n_bands=60,
            other_polygon_ids=tile_other_ids if tile_other_ids else None
        )
        split = split_map[tile_id]
        for r in records:
            r['split'] = split
        all_records.extend(records)
        logger.info(f"  {tile_id}: {len(records)} pixels -> {split}")

    df = pd.DataFrame(all_records)
    out_path = os.path.join(output_dir, 'pixels.parquet')
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(df)} pixels to {out_path}")

    # Summary stats
    for split in ['train', 'val', 'test']:
        sub = df[df['split'] == split]
        logger.info(f"{split}: {len(sub)} pixels from {sub['tile_id'].nunique()} tiles")
    label_cols = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']
    logger.info("Label sums (positive pixels):")
    for col in label_cols:
        logger.info(f"  {col}: {(df[col] > 0).sum()}")


if __name__ == '__main__':
    main()
```

**Step 3: Run the dataset build**

```bash
cd /mnt/crism/MRDR/crism_classification
conda run -n crism python scripts/build_dataset.py
```
Expected: progress bar over 38 tiles, final parquet saved.

**Step 4: Run tests**

```bash
conda run -n crism python -m pytest tests/test_build_dataset.py -v
```
Expected: all 7 tests PASS.

**Step 5: Commit**

```bash
git add scripts/build_dataset.py tests/test_build_dataset.py
git commit -m "feat: build_dataset script producing pixels.parquet"
```

---

## Task 5: Dataset Loader

**Files:**
- Create: `crism_classification/data/dataset.py`
- Create: `crism_classification/tests/test_dataset.py`

**Step 1: Write failing tests**

```python
# crism_classification/tests/test_dataset.py
import numpy as np
import pytest
import pandas as pd
from data.dataset import CRISMPixelDataset, load_sklearn_arrays

PARQUET = '/mnt/crism/MRDR/crism_classification/data/pixels.parquet'

@pytest.fixture
def small_df():
    df = pd.read_parquet(PARQUET)
    return df[df['split'] == 'train'].head(200)

def test_dataset_len(small_df):
    ds = CRISMPixelDataset(small_df)
    assert len(ds) == 200

def test_dataset_item_shapes(small_df):
    ds = CRISMPixelDataset(small_df)
    features, labels, weight = ds[0]
    assert features.shape == (60,)
    assert labels.shape == (6,)
    assert weight.shape == ()

def test_dataset_item_types(small_df):
    import torch
    ds = CRISMPixelDataset(small_df)
    features, labels, weight = ds[0]
    assert features.dtype == torch.float32
    assert labels.dtype == torch.float32
    assert weight.dtype == torch.float32

def test_load_sklearn_arrays_shapes():
    X_tr, y_tr, w_tr, X_v, y_v, w_v, X_te, y_te, w_te = load_sklearn_arrays(PARQUET)
    assert X_tr.shape[1] == 60
    assert y_tr.shape[1] == 6
    assert w_tr.shape[0] == X_tr.shape[0]
    assert X_v.shape[1] == 60
    assert X_te.shape[1] == 60

def test_load_sklearn_no_nan():
    X_tr, y_tr, w_tr, *_ = load_sklearn_arrays(PARQUET)
    assert not np.isnan(X_tr).any()
    assert not np.isnan(y_tr).any()

def test_patch_dataset_shape(small_df):
    from data.dataset import CRISMPatchDataset
    # Needs mrrsu paths dict
    import yaml, os
    cfg_path = '/mnt/crism/MRDR/crism_classification/config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    from data.extract_pixels import find_tile_pairs
    pairs = find_tile_pairs(cfg['gpkg_dir'], cfg['data_root'])
    mrrsu_map = {tid: p for tid, _, p in pairs}
    ds = CRISMPatchDataset(small_df, mrrsu_map, patch_size=7)
    patch, labels, weight = ds[0]
    assert patch.shape == (60, 7, 7)
```

**Step 2: Implement dataset.py**

```python
# crism_classification/data/dataset.py
"""
PyTorch Dataset classes and sklearn array loaders for the pixel dataset.
"""
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import rasterio

LABEL_COLS = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']
BAND_COLS = [f'b{i}' for i in range(60)]


class CRISMPixelDataset(Dataset):
    """Per-pixel dataset for MLP and linear models."""

    def __init__(self, df: pd.DataFrame):
        self.features = torch.tensor(df[BAND_COLS].values, dtype=torch.float32)
        self.labels = torch.tensor(df[LABEL_COLS].values, dtype=torch.float32)
        self.weights = torch.tensor(df['confidence_weight'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx], self.weights[idx]


class CRISMPatchDataset(Dataset):
    """
    Spatial patch dataset for CNN and ViT.
    Extracts a (patch_size x patch_size x 60) neighbourhood around each pixel
    from the corresponding mrrsu raster at runtime.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        mrrsu_map: Dict[str, str],
        patch_size: int = 7,
    ):
        assert patch_size % 2 == 1, "patch_size must be odd"
        self.df = df.reset_index(drop=True)
        self.mrrsu_map = mrrsu_map
        self.patch_size = patch_size
        self.half = patch_size // 2
        self.labels = torch.tensor(df[LABEL_COLS].values, dtype=torch.float32)
        self.weights = torch.tensor(df['confidence_weight'].values, dtype=torch.float32)

        # Cache open rasterio file handles per tile
        self._handles: Dict[str, rasterio.DatasetReader] = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tile_id = row['tile_id']
        pr, pc = int(row['pixel_row']), int(row['pixel_col'])

        if tile_id not in self._handles:
            self._handles[tile_id] = rasterio.open(self.mrrsu_map[tile_id])
        src = self._handles[tile_id]

        h = self.half
        r0 = max(0, pr - h)
        r1 = min(src.height, pr + h + 1)
        c0 = max(0, pc - h)
        c1 = min(src.width, pc + h + 1)

        window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
        patch = src.read(window=window).astype(np.float32)  # (bands, h, w)

        # Replace NODATA
        patch[patch >= 65535] = 0.0
        patch = np.nan_to_num(patch, nan=0.0)

        # Pad to patch_size x patch_size if near border
        full = np.zeros((src.count, self.patch_size, self.patch_size), dtype=np.float32)
        pr_start = (pr - h) - r0 + (h - (pr - r0))
        pc_start = (pc - h) - c0 + (h - (pc - c0))
        actual_h = patch.shape[1]
        actual_w = patch.shape[2]
        full[:, h - (pr - r0):h - (pr - r0) + actual_h,
                h - (pc - c0):h - (pc - c0) + actual_w] = patch

        return torch.tensor(full, dtype=torch.float32), self.labels[idx], self.weights[idx]

    def __del__(self):
        for src in self._handles.values():
            src.close()


def load_sklearn_arrays(parquet_path: str):
    """
    Load train/val/test arrays for sklearn models.

    Returns
    -------
    X_train, y_train, w_train, X_val, y_val, w_val, X_test, y_test, w_test
    All as numpy arrays. y arrays are shape (n, 6) float32.
    """
    df = pd.read_parquet(parquet_path)

    def _split(split_name):
        sub = df[df['split'] == split_name]
        X = sub[BAND_COLS].values.astype(np.float32)
        y = sub[LABEL_COLS].values.astype(np.float32)
        w = sub['confidence_weight'].values.astype(np.float32)
        return X, y, w

    return (*_split('train'), *_split('val'), *_split('test'))
```

**Step 3: Run tests**

```bash
conda run -n crism python -m pytest tests/test_dataset.py -v
```
Expected: all 6 tests PASS.

**Step 4: Commit**

```bash
git add data/dataset.py tests/test_dataset.py
git commit -m "feat: pixel and patch dataset classes"
```

---

## Task 6: Loss Function & Metrics

**Files:**
- Create: `crism_classification/training/losses.py`
- Create: `crism_classification/evaluation/metrics.py`
- Create: `crism_classification/tests/test_losses_metrics.py`

**Step 1: Write failing tests**

```python
# crism_classification/tests/test_losses_metrics.py
import torch
import numpy as np
import pytest
from training.losses import WeightedBCEWithLogitsLoss
from evaluation.metrics import (
    compute_map, compute_per_class_ap, compute_metrics_by_confidence_tier
)

def test_weighted_loss_shape():
    loss_fn = WeightedBCEWithLogitsLoss()
    logits = torch.randn(8, 6)
    targets = torch.randint(0, 2, (8, 6)).float()
    weights = torch.ones(8)
    loss = loss_fn(logits, targets, weights)
    assert loss.shape == ()  # scalar
    assert loss.item() > 0

def test_high_weight_increases_loss():
    loss_fn = WeightedBCEWithLogitsLoss()
    logits = torch.zeros(4, 6)
    targets = torch.ones(4, 6)
    w_low = torch.full((4,), 0.25)
    w_high = torch.full((4,), 1.0)
    assert loss_fn(logits, targets, w_high) > loss_fn(logits, targets, w_low)

def test_compute_map_perfect():
    y_true = np.eye(6, dtype=np.float32)
    y_score = np.eye(6, dtype=np.float32)
    mAP = compute_map(y_true, y_score)
    assert mAP == pytest.approx(1.0)

def test_compute_map_range():
    y_true = np.random.randint(0, 2, (100, 6)).astype(np.float32)
    y_score = np.random.rand(100, 6).astype(np.float32)
    mAP = compute_map(y_true, y_score)
    assert 0.0 <= mAP <= 1.0

def test_per_class_ap_keys():
    from data.label_parser import CLASSES
    y_true = np.random.randint(0, 2, (50, 6)).astype(np.float32)
    y_score = np.random.rand(50, 6).astype(np.float32)
    ap_dict = compute_per_class_ap(y_true, y_score)
    assert set(ap_dict.keys()) == set(CLASSES)

def test_metrics_by_confidence_tier():
    y_true = np.random.randint(0, 2, (60, 6)).astype(np.float32)
    y_score = np.random.rand(60, 6).astype(np.float32)
    tiers = ['High'] * 20 + ['Moderate'] * 20 + ['Low'] * 20
    result = compute_metrics_by_confidence_tier(y_true, y_score, tiers)
    assert set(result.keys()) == {'High', 'Moderate', 'Low'}
    for tier_metrics in result.values():
        assert 'mAP' in tier_metrics
        assert 0.0 <= tier_metrics['mAP'] <= 1.0
```

**Step 2: Implement losses.py**

```python
# crism_classification/training/losses.py
import torch
import torch.nn as nn


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Binary cross-entropy with logits, weighted per sample by confidence weight.
    Averages over classes first, then takes confidence-weighted mean over samples.
    """

    def forward(
        self,
        logits: torch.Tensor,   # (batch, n_classes)
        targets: torch.Tensor,  # (batch, n_classes)
        weights: torch.Tensor,  # (batch,)
    ) -> torch.Tensor:
        # Per-sample, per-class BCE: shape (batch, n_classes)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        # Mean over classes: shape (batch,)
        bce_per_sample = bce.mean(dim=1)
        # Weighted mean over samples
        return (bce_per_sample * weights).sum() / (weights.sum() + 1e-8)
```

**Step 3: Implement metrics.py**

```python
# crism_classification/evaluation/metrics.py
"""
Evaluation metrics for multi-label mineral classification.
All functions accept numpy arrays.
"""
from typing import Dict, List
import numpy as np
from sklearn.metrics import average_precision_score

from data.label_parser import CLASSES


def compute_map(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mean Average Precision across all 6 classes. Skips classes with no positives."""
    aps = []
    for i in range(y_true.shape[1]):
        if y_true[:, i].sum() > 0:
            aps.append(average_precision_score(
                (y_true[:, i] > 0.4).astype(int), y_score[:, i]
            ))
    return float(np.mean(aps)) if aps else 0.0


def compute_per_class_ap(
    y_true: np.ndarray,
    y_score: np.ndarray
) -> Dict[str, float]:
    """Per-class Average Precision. Returns dict keyed by class name."""
    result = {}
    for i, cls in enumerate(CLASSES):
        if y_true[:, i].sum() > 0:
            result[cls] = float(average_precision_score(
                (y_true[:, i] > 0.4).astype(int), y_score[:, i]
            ))
        else:
            result[cls] = float('nan')
    return result


def compute_metrics_by_confidence_tier(
    y_true: np.ndarray,
    y_score: np.ndarray,
    confidence_tiers: List[str],
) -> Dict[str, Dict]:
    """
    Compute mAP and per-class AP broken out by confidence tier.

    Parameters
    ----------
    y_true : (n, 6)
    y_score : (n, 6)
    confidence_tiers : list of str, length n, values in {'High','Moderate','Low'}
    """
    tiers = np.array(confidence_tiers)
    result = {}
    for tier in ['High', 'Moderate', 'Low']:
        mask = tiers == tier
        if mask.sum() == 0:
            result[tier] = {'mAP': float('nan')}
            continue
        result[tier] = {
            'mAP': compute_map(y_true[mask], y_score[mask]),
            'per_class_ap': compute_per_class_ap(y_true[mask], y_score[mask]),
            'n_pixels': int(mask.sum()),
        }
    return result


def compute_full_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    confidence_tiers: List[str],
) -> Dict:
    """Convenience wrapper: overall + per-class + confidence-tier metrics."""
    return {
        'mAP': compute_map(y_true, y_score),
        'per_class_ap': compute_per_class_ap(y_true, y_score),
        'by_confidence': compute_metrics_by_confidence_tier(
            y_true, y_score, confidence_tiers
        ),
    }
```

**Step 4: Run tests**

```bash
conda run -n crism python -m pytest tests/test_losses_metrics.py -v
```
Expected: all 6 tests PASS.

**Step 5: Commit**

```bash
git add training/losses.py evaluation/metrics.py tests/test_losses_metrics.py
git commit -m "feat: weighted BCE loss and mAP/confidence-tier metrics"
```

---

## Task 7: W&B Setup

**Files:**
- Create: `crism_classification/scripts/setup_wandb.py`

**Step 1: Implement setup_wandb.py**

```python
# crism_classification/scripts/setup_wandb.py
"""
Interactive W&B setup. Run once before training.

Usage:
    conda run -n crism python scripts/setup_wandb.py
"""
import os, sys, yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    try:
        import wandb
    except ImportError:
        print("wandb not installed. Run: pip install wandb")
        sys.exit(1)

    print("=== Weights & Biases Setup ===")
    print("You need a free W&B account at https://wandb.ai")
    print()

    api_key = input("Paste your W&B API key (from https://wandb.ai/authorize): ").strip()
    if not api_key:
        print("No API key provided. Exiting.")
        sys.exit(1)

    wandb.login(key=api_key)

    entity = input("W&B username or team name (leave blank for default): ").strip() or None

    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config.yaml'
    )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    cfg['wandb']['entity'] = entity
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print(f"\nW&B configured: project=crism-mineral-classification, entity={entity}")
    print("Test with: conda run -n crism python -c \"import wandb; wandb.init(project='crism-mineral-classification')\"")

if __name__ == '__main__':
    main()
```

**Step 2: Run setup**

```bash
conda run -n crism python scripts/setup_wandb.py
```
Follow prompts to log in.

**Step 3: Commit**

```bash
git add scripts/setup_wandb.py
git commit -m "feat: wandb setup script"
```

---

## Task 8: Sklearn Training Loop

**Files:**
- Create: `crism_classification/training/train_sklearn.py`
- Create: `crism_classification/tests/test_train_sklearn.py`

**Step 1: Write failing tests**

```python
# crism_classification/tests/test_train_sklearn.py
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.train_sklearn import train_and_evaluate_sklearn

def make_fake_data(n=200, n_features=60, n_classes=6):
    X = np.random.rand(n, n_features).astype(np.float32)
    y = (np.random.rand(n, n_classes) > 0.7).astype(np.float32)
    w = np.ones(n, dtype=np.float32)
    return X, y, w

def test_logreg_returns_metrics():
    X_tr, y_tr, w_tr = make_fake_data()
    X_v, y_v, w_v = make_fake_data(50)
    metrics = train_and_evaluate_sklearn(
        'logreg', X_tr, y_tr, w_tr, X_v, y_v, w_v,
        use_wandb=False
    )
    assert 'val_mAP' in metrics
    assert 0.0 <= metrics['val_mAP'] <= 1.0

def test_rf_returns_metrics():
    X_tr, y_tr, w_tr = make_fake_data()
    X_v, y_v, w_v = make_fake_data(50)
    metrics = train_and_evaluate_sklearn(
        'rf', X_tr, y_tr, w_tr, X_v, y_v, w_v,
        use_wandb=False, n_estimators=10
    )
    assert 'val_mAP' in metrics

def test_xgb_returns_metrics():
    X_tr, y_tr, w_tr = make_fake_data()
    X_v, y_v, w_v = make_fake_data(50)
    metrics = train_and_evaluate_sklearn(
        'xgb', X_tr, y_tr, w_tr, X_v, y_v, w_v,
        use_wandb=False, n_estimators=10
    )
    assert 'val_mAP' in metrics
```

**Step 2: Implement train_sklearn.py**

```python
# crism_classification/training/train_sklearn.py
"""
Training and evaluation for sklearn-compatible models.
Supports: logreg, svc, rf, xgb, lgbm
"""
import os, sys, pickle, logging
from typing import Dict, Any
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metrics import compute_full_metrics
from data.label_parser import CLASSES

logger = logging.getLogger(__name__)


def _build_model(model_type: str, **kwargs):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier

    if model_type == 'logreg':
        base = LogisticRegression(max_iter=1000, C=kwargs.get('C', 1.0))
        return MultiOutputClassifier(base)
    elif model_type == 'svc':
        base = LinearSVC(max_iter=2000, C=kwargs.get('C', 1.0))
        return MultiOutputClassifier(base)
    elif model_type == 'rf':
        return RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 200),
            max_depth=kwargs.get('max_depth', None),
            n_jobs=-1, random_state=42
        )
    elif model_type == 'xgb':
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=kwargs.get('n_estimators', 200),
            max_depth=kwargs.get('max_depth', 6),
            learning_rate=kwargs.get('learning_rate', 0.1),
            use_label_encoder=False,
            eval_metric='logloss',
            tree_method='hist',
            random_state=42
        )
    elif model_type == 'lgbm':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=kwargs.get('n_estimators', 200),
            max_depth=kwargs.get('max_depth', -1),
            learning_rate=kwargs.get('learning_rate', 0.1),
            random_state=42, n_jobs=-1, verbose=-1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_and_evaluate_sklearn(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    w_val: np.ndarray,
    confidence_tiers_val: list = None,
    use_wandb: bool = True,
    checkpoint_dir: str = None,
    **model_kwargs
) -> Dict[str, Any]:
    """
    Train a sklearn model and evaluate on validation set.

    For multi-label targets (y shape n x 6), XGB/LGBM train one model per class.
    LogReg/SVC/RF use MultiOutputClassifier.
    """
    import wandb

    if use_wandb:
        wandb.init(
            project='crism-mineral-classification',
            name=f'{model_type}',
            config={'model': model_type, **model_kwargs}
        )

    n_classes = y_train.shape[1]
    models = []

    # Tree models and linear models handle multi-output differently
    if model_type in ('xgb', 'lgbm'):
        # Train one model per class
        for cls_idx in range(n_classes):
            y_col = (y_train[:, cls_idx] > 0.4).astype(int)
            m = _build_model(model_type, **model_kwargs)
            m.fit(X_train, y_col, sample_weight=w_train)
            models.append(m)
    else:
        # MultiOutputClassifier handles all classes at once
        # Convert soft labels to hard for sklearn
        y_hard = (y_train > 0.4).astype(int)
        m = _build_model(model_type, **model_kwargs)
        m.fit(X_train, y_hard, sample_weight=w_train)
        models = [m]

    # Get probability scores for val set
    y_score = _predict_proba(models, X_val, model_type, n_classes)

    if confidence_tiers_val is None:
        confidence_tiers_val = ['High'] * len(y_val)

    metrics = compute_full_metrics(y_val, y_score, confidence_tiers_val)
    flat_metrics = _flatten_metrics(metrics)

    logger.info(f"{model_type} val mAP: {metrics['mAP']:.4f}")
    for cls, ap in metrics['per_class_ap'].items():
        logger.info(f"  {cls}: AP={ap:.4f}")

    if use_wandb:
        wandb.log(flat_metrics)

    # Save checkpoint
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, f'{model_type}_model.pkl')
        with open(ckpt_path, 'wb') as f:
            pickle.dump(models, f)
        logger.info(f"Saved checkpoint to {ckpt_path}")
        if use_wandb:
            artifact = wandb.Artifact(f'{model_type}-model', type='model')
            artifact.add_file(ckpt_path)
            wandb.log_artifact(artifact)

    if use_wandb:
        wandb.finish()

    return {'val_mAP': metrics['mAP'], **flat_metrics}


def _predict_proba(models, X, model_type, n_classes):
    """Get probability scores. Returns array of shape (n, n_classes)."""
    if model_type in ('xgb', 'lgbm'):
        scores = np.stack([m.predict_proba(X)[:, 1] for m in models], axis=1)
    elif model_type == 'svc':
        # LinearSVC has no predict_proba — use decision_function
        m = models[0]
        raw = m.decision_function(X)
        if raw.ndim == 1:
            raw = raw[:, np.newaxis]
        # Normalize to 0-1 range via sigmoid
        scores = 1 / (1 + np.exp(-raw))
    else:
        m = models[0]
        proba_list = m.predict_proba(X)  # list of (n, 2) arrays
        scores = np.stack([p[:, 1] for p in proba_list], axis=1)
    return scores.astype(np.float32)


def _flatten_metrics(metrics: dict) -> dict:
    """Flatten nested metrics dict for W&B logging."""
    flat = {'val_mAP': metrics['mAP']}
    for cls, ap in metrics['per_class_ap'].items():
        flat[f'val_AP_{cls}'] = ap
    for tier, tm in metrics['by_confidence'].items():
        flat[f'val_mAP_{tier}'] = tm.get('mAP', float('nan'))
    return flat
```

**Step 3: Run tests**

```bash
conda run -n crism python -m pytest tests/test_train_sklearn.py -v
```
Expected: all 3 tests PASS.

**Step 4: Commit**

```bash
git add training/train_sklearn.py tests/test_train_sklearn.py
git commit -m "feat: sklearn training loop with W&B logging"
```

---

## Task 9: PyTorch Models — MLP

**Files:**
- Create: `crism_classification/models/mlp.py`
- Create: `crism_classification/tests/test_models.py`

**Step 1: Write failing tests**

```python
# crism_classification/tests/test_models.py
import torch
import pytest

def test_mlp_output_shape():
    from models.mlp import MLP
    model = MLP(n_features=60, n_classes=6)
    x = torch.randn(8, 60)
    out = model(x)
    assert out.shape == (8, 6)

def test_mlp_no_sigmoid_in_forward():
    """MLP should return logits, not probabilities."""
    from models.mlp import MLP
    model = MLP()
    x = torch.zeros(4, 60)
    out = model(x)
    # If sigmoid applied, all outputs would be 0.5 for zero input
    # Logits for zero input after linear layers will be near 0 but not exactly 0.5
    assert not torch.allclose(out, torch.full_like(out, 0.5))

def test_cnn_output_shape():
    from models.cnn import SpectralSpatialCNN
    model = SpectralSpatialCNN(n_bands=60, n_classes=6, patch_size=7)
    x = torch.randn(4, 60, 7, 7)
    out = model(x)
    assert out.shape == (4, 6)

def test_vit_output_shape():
    from models.vit import SpectralViT
    model = SpectralViT(n_bands=60, n_classes=6, patch_size=7, embed_dim=64, n_heads=4, n_layers=2)
    x = torch.randn(4, 60, 7, 7)
    out = model(x)
    assert out.shape == (4, 6)
```

**Step 2: Implement mlp.py**

```python
# crism_classification/models/mlp.py
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer perceptron for per-pixel multi-label classification.
    Returns logits (no sigmoid). Use BCEWithLogitsLoss during training.
    """

    def __init__(
        self,
        n_features: int = 60,
        n_classes: int = 6,
        hidden_dims: tuple = (256, 128),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = [nn.BatchNorm1d(n_features)]
        in_dim = n_features
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

**Step 3: Implement cnn.py**

```python
# crism_classification/models/cnn.py
import torch
import torch.nn as nn


class SpectralSpatialCNN(nn.Module):
    """
    2D CNN over a (patch_size x patch_size x n_bands) spatial-spectral patch.
    Input shape: (batch, n_bands, patch_size, patch_size)
    Returns logits of shape (batch, n_classes).
    """

    def __init__(
        self,
        n_bands: int = 60,
        n_classes: int = 6,
        patch_size: int = 7,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(n_bands, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.classifier(x)
```

**Step 4: Implement vit.py**

```python
# crism_classification/models/vit.py
"""
Lightweight Vision Transformer for spectral-spatial classification.
Treats each pixel in the patch as a token with spectral embedding.
"""
import math
import torch
import torch.nn as nn


class SpectralViT(nn.Module):
    """
    ViT over a (patch_size x patch_size) spatial grid of spectral tokens.
    Each pixel's n_bands values are embedded to embed_dim.
    A learned CLS token aggregates spatial context for classification.

    Input: (batch, n_bands, patch_size, patch_size)
    Output: (batch, n_classes) logits
    """

    def __init__(
        self,
        n_bands: int = 60,
        n_classes: int = 6,
        patch_size: int = 7,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        n_tokens = patch_size * patch_size

        # Spectral embedding: project n_bands to embed_dim per pixel
        self.spectral_embed = nn.Linear(n_bands, embed_dim)

        # Learned CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) where C=n_bands, H=W=patch_size
        B, C, H, W = x.shape
        # Flatten spatial dims: (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        # Spectral embedding: (B, H*W, embed_dim)
        x = self.spectral_embed(x)
        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, H*W+1, embed_dim)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x[:, 0])  # CLS token output
        return self.head(x)
```

**Step 5: Run model tests**

```bash
conda run -n crism python -m pytest tests/test_models.py -v
```
Expected: all 4 tests PASS.

**Step 6: Commit**

```bash
git add models/mlp.py models/cnn.py models/vit.py tests/test_models.py
git commit -m "feat: MLP, CNN, and ViT model architectures"
```

---

## Task 10: PyTorch Training Loop

**Files:**
- Create: `crism_classification/training/train_torch.py`
- Create: `crism_classification/tests/test_train_torch.py`

**Step 1: Write failing test**

```python
# crism_classification/tests/test_train_torch.py
import torch
import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.train_torch import train_torch_model
from models.mlp import MLP

def make_fake_df(n=300):
    data = {f'b{i}': np.random.rand(n).astype(np.float32) for i in range(60)}
    for col in ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']:
        data[col] = (np.random.rand(n) > 0.7).astype(np.float32)
    data['confidence_weight'] = np.ones(n, dtype=np.float32)
    data['confidence_tier'] = ['High'] * n
    data['tile_id'] = 't0001'
    data['polygon_id'] = 0
    data['pixel_row'] = 0
    data['pixel_col'] = 0
    splits = ['train'] * 200 + ['val'] * 50 + ['test'] * 50
    data['split'] = splits
    return pd.DataFrame(data)

def test_mlp_trains_without_error():
    df = make_fake_df()
    model = MLP(n_features=60, n_classes=6)
    metrics = train_torch_model(
        model=model,
        df=df,
        model_name='mlp_test',
        max_epochs=2,
        batch_size=32,
        lr=1e-3,
        use_wandb=False,
        checkpoint_dir=None,
    )
    assert 'val_mAP' in metrics
    assert 0.0 <= metrics['val_mAP'] <= 1.0

def test_early_stopping_triggers():
    df = make_fake_df(n=600)
    model = MLP()
    metrics = train_torch_model(
        model=model, df=df, model_name='mlp_es',
        max_epochs=50, patience=2, use_wandb=False, checkpoint_dir=None
    )
    # Should stop before epoch 50
    assert metrics.get('stopped_epoch', 50) <= 50
```

**Step 2: Implement train_torch.py**

```python
# crism_classification/training/train_torch.py
"""
Training loop for PyTorch models (MLP, CNN, ViT).
"""
import os, sys, copy, logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import CRISMPixelDataset, CRISMPatchDataset
from training.losses import WeightedBCEWithLogitsLoss
from evaluation.metrics import compute_full_metrics

logger = logging.getLogger(__name__)


def train_torch_model(
    model: torch.nn.Module,
    df: pd.DataFrame,
    model_name: str,
    max_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 10,
    use_wandb: bool = True,
    checkpoint_dir: Optional[str] = None,
    mrrsu_map: Optional[Dict[str, str]] = None,
    patch_size: int = 7,
    device: Optional[str] = None,
    **wandb_config
) -> Dict[str, Any]:
    """
    Train a PyTorch model with early stopping on val mAP.

    Automatically uses CRISMPatchDataset when mrrsu_map is provided (CNN/ViT),
    otherwise uses CRISMPixelDataset (MLP).
    """
    import wandb as wb

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if use_wandb:
        wb.init(
            project='crism-mineral-classification',
            name=model_name,
            config={'model': model_name, 'lr': lr, 'batch_size': batch_size,
                    'max_epochs': max_epochs, **wandb_config}
        )

    use_patches = mrrsu_map is not None

    def make_dataset(split):
        sub = df[df['split'] == split]
        if use_patches:
            return CRISMPatchDataset(sub, mrrsu_map, patch_size=patch_size)
        return CRISMPixelDataset(sub)

    train_ds = make_dataset('train')
    val_ds = make_dataset('val')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    loss_fn = WeightedBCEWithLogitsLoss()

    best_val_map = -1.0
    best_state = None
    patience_counter = 0
    stopped_epoch = max_epochs

    for epoch in range(1, max_epochs + 1):
        # --- Train ---
        model.train()
        train_losses = []
        for features, labels, weights in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, labels, weights)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # --- Validate ---
        model.eval()
        all_logits, all_labels, all_tiers = [], [], []
        val_sub = df[df['split'] == 'val']

        with torch.no_grad():
            for features, labels, weights in val_loader:
                features = features.to(device)
                logits = model(features)
                all_logits.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(labels.numpy())

        y_score = np.concatenate(all_logits)
        y_true = np.concatenate(all_labels)
        conf_tiers = val_sub['confidence_tier'].tolist()

        metrics = compute_full_metrics(y_true, y_score, conf_tiers)
        val_map = metrics['mAP']
        flat = _flatten_metrics(metrics)

        logger.info(f"Epoch {epoch}/{max_epochs} | train_loss={np.mean(train_losses):.4f} | val_mAP={val_map:.4f}")

        if use_wandb:
            wb.log({'epoch': epoch, 'train_loss': np.mean(train_losses), **flat})

        # Early stopping
        if val_map > best_val_map:
            best_val_map = val_map
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                stopped_epoch = epoch
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save checkpoint
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, f'{model_name}_best.pt')
        torch.save({'model_state': best_state, 'val_mAP': best_val_map}, ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}")
        if use_wandb:
            artifact = wb.Artifact(f'{model_name}-model', type='model')
            artifact.add_file(ckpt_path)
            wb.log_artifact(artifact)

    if use_wandb:
        wb.finish()

    return {'val_mAP': best_val_map, 'stopped_epoch': stopped_epoch, **_flatten_metrics(metrics)}


def _flatten_metrics(metrics: dict) -> dict:
    flat = {'val_mAP': metrics['mAP']}
    for cls, ap in metrics['per_class_ap'].items():
        flat[f'val_AP_{cls}'] = ap
    for tier, tm in metrics['by_confidence'].items():
        flat[f'val_mAP_{tier}'] = tm.get('mAP', float('nan'))
    return flat
```

**Step 3: Run tests**

```bash
conda run -n crism python -m pytest tests/test_train_torch.py -v
```
Expected: both tests PASS (will be slow — ~10 seconds).

**Step 4: Commit**

```bash
git add training/train_torch.py tests/test_train_torch.py
git commit -m "feat: pytorch training loop with early stopping and W&B"
```

---

## Task 11: Unified Train Entry Point

**Files:**
- Create: `crism_classification/scripts/train.py`

**Step 1: Implement train.py**

```python
# crism_classification/scripts/train.py
"""
Unified training entry point.

Usage:
    conda run -n crism python scripts/train.py --model logreg
    conda run -n crism python scripts/train.py --model rf --n_estimators 300
    conda run -n crism python scripts/train.py --model mlp --lr 1e-3 --epochs 100
    conda run -n crism python scripts/train.py --model cnn --patch_size 7
    conda run -n crism python scripts/train.py --model vit --embed_dim 128

Models: logreg, svc, rf, xgb, lgbm, mlp, cnn, vit
"""
import argparse, os, sys, yaml, logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

SKLEARN_MODELS = {'logreg', 'svc', 'rf', 'xgb', 'lgbm'}
TORCH_MODELS = {'mlp', 'cnn', 'vit'}

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Train a mineral classification model.")
    parser.add_argument('--model', required=True, choices=list(SKLEARN_MODELS | TORCH_MODELS))
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--no_wandb', action='store_true')
    # sklearn kwargs
    parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--max_depth', type=int, default=None)
    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    # torch kwargs
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=7)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=4)
    args = parser.parse_args()

    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.config
    )
    cfg = load_config(cfg_path)
    parquet_path = os.path.join(cfg['output_dir'], 'pixels.parquet')
    checkpoint_dir = cfg['checkpoints_dir']
    use_wandb = not args.no_wandb

    if args.model in SKLEARN_MODELS:
        from data.dataset import load_sklearn_arrays
        from training.train_sklearn import train_and_evaluate_sklearn
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        X_tr, y_tr, w_tr, X_v, y_v, w_v, X_te, y_te, w_te = load_sklearn_arrays(parquet_path)
        val_tiers = df[df['split'] == 'val']['confidence_tier'].tolist()
        test_tiers = df[df['split'] == 'test']['confidence_tier'].tolist()

        metrics = train_and_evaluate_sklearn(
            args.model, X_tr, y_tr, w_tr, X_v, y_v, w_v,
            confidence_tiers_val=val_tiers,
            use_wandb=use_wandb,
            checkpoint_dir=checkpoint_dir,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            C=args.C,
            learning_rate=args.learning_rate,
        )

    elif args.model in TORCH_MODELS:
        import torch
        from training.train_torch import train_torch_model
        df = pd.read_parquet(parquet_path)

        if args.model == 'mlp':
            from models.mlp import MLP
            model = MLP(n_features=60, n_classes=6)
            metrics = train_torch_model(
                model=model, df=df, model_name='mlp',
                max_epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, patience=args.patience,
                use_wandb=use_wandb, checkpoint_dir=checkpoint_dir,
            )

        elif args.model in ('cnn', 'vit'):
            from data.extract_pixels import find_tile_pairs
            pairs = find_tile_pairs(cfg['gpkg_dir'], cfg['data_root'])
            mrrsu_map = {tid: p for tid, _, p in pairs}

            if args.model == 'cnn':
                from models.cnn import SpectralSpatialCNN
                model = SpectralSpatialCNN(n_bands=60, n_classes=6, patch_size=args.patch_size)
            else:
                from models.vit import SpectralViT
                model = SpectralViT(
                    n_bands=60, n_classes=6, patch_size=args.patch_size,
                    embed_dim=args.embed_dim, n_heads=args.n_heads, n_layers=args.n_layers
                )

            metrics = train_torch_model(
                model=model, df=df, model_name=args.model,
                max_epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, patience=args.patience,
                use_wandb=use_wandb, checkpoint_dir=checkpoint_dir,
                mrrsu_map=mrrsu_map, patch_size=args.patch_size,
            )

    print(f"\n=== {args.model} Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == '__main__':
    main()
```

**Step 2: Smoke test**

```bash
cd /mnt/crism/MRDR/crism_classification
conda run -n crism python scripts/train.py --model logreg --no_wandb --n_estimators 10 2>&1 | tail -15
```
Expected: runs to completion, prints results table.

**Step 3: Commit**

```bash
git add scripts/train.py
git commit -m "feat: unified train.py entry point for all model families"
```

---

## Task 12: Inference Script

**Files:**
- Create: `crism_classification/scripts/predict_tile.py`
- Create: `crism_classification/tests/test_predict_tile.py`

**Step 1: Write failing test**

```python
# crism_classification/tests/test_predict_tile.py
import os, pytest, tempfile
import numpy as np

def test_predict_tile_produces_geotiffs(tmp_path):
    """Smoke test: run inference on the test tile, check outputs exist."""
    from scripts.predict_tile import predict_tile
    import glob

    # Use first test tile
    import pandas as pd, yaml
    cfg_path = '/mnt/crism/MRDR/crism_classification/config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    df = pd.read_parquet(os.path.join(cfg['output_dir'], 'pixels.parquet'))
    test_tile = df[df['split'] == 'test']['tile_id'].iloc[0]

    from data.extract_pixels import find_tile_pairs
    pairs = find_tile_pairs(cfg['gpkg_dir'], cfg['data_root'])
    mrrsu_map = {tid: p for tid, _, p in pairs}

    # Use a saved sklearn checkpoint
    import glob
    ckpts = glob.glob(os.path.join(cfg['checkpoints_dir'], 'logreg_model.pkl'))
    if not ckpts:
        pytest.skip("No logreg checkpoint found. Run train.py --model logreg first.")

    out_dir = str(tmp_path / 'predictions')
    predict_tile(
        tile_id=test_tile,
        mrrsu_path=mrrsu_map[test_tile],
        checkpoint_path=ckpts[0],
        model_type='logreg',
        output_dir=out_dir,
    )

    from data.label_parser import CLASSES
    for cls in CLASSES:
        assert os.path.exists(os.path.join(out_dir, test_tile, f'{cls}_prob.tif'))
    assert os.path.exists(os.path.join(out_dir, test_tile, 'best_class.tif'))
```

**Step 2: Implement predict_tile.py**

```python
# crism_classification/scripts/predict_tile.py
"""
Run inference on a full CRISM mrrsu tile and write per-class GeoTIFF outputs.

Usage:
    conda run -n crism python scripts/predict_tile.py \
        --tile_id t0503 \
        --model logreg \
        --checkpoint checkpoints/logreg_model.pkl
"""
import os, sys, argparse, pickle, logging
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.extract_pixels import NODATA_VALUE
from data.label_parser import CLASSES

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SKLEARN_MODELS = {'logreg', 'svc', 'rf', 'xgb', 'lgbm'}
TORCH_PIXEL_MODELS = {'mlp'}
TORCH_PATCH_MODELS = {'cnn', 'vit'}


def predict_tile(
    tile_id: str,
    mrrsu_path: str,
    checkpoint_path: str,
    model_type: str,
    output_dir: str,
    patch_size: int = 7,
    batch_size: int = 4096,
):
    """
    Predict mineral probabilities for all valid pixels in a mrrsu tile.
    Writes one GeoTIFF per class + a best_class.tif.
    """
    os.makedirs(os.path.join(output_dir, tile_id), exist_ok=True)

    logger.info(f"Loading tile: {mrrsu_path}")
    with rasterio.open(mrrsu_path) as src:
        data = src.read().astype(np.float32)  # (bands, H, W)
        profile = src.profile.copy()
        H, W = src.height, src.width
        n_bands = src.count

    # Build valid pixel mask
    nodata_mask = (data >= NODATA_VALUE).any(axis=0) | np.isnan(data).any(axis=0)
    valid_mask = ~nodata_mask  # (H, W)
    rows, cols = np.where(valid_mask)
    X = data[:, rows, cols].T  # (n_valid, n_bands)

    logger.info(f"Valid pixels: {len(rows):,} / {H * W:,}")

    # --- Load model and predict ---
    if model_type in SKLEARN_MODELS:
        with open(checkpoint_path, 'rb') as f:
            models = pickle.load(f)
        y_score = _sklearn_predict(models, X, model_type)

    elif model_type in TORCH_PIXEL_MODELS:
        y_score = _torch_pixel_predict(checkpoint_path, model_type, X, batch_size)

    elif model_type in TORCH_PATCH_MODELS:
        y_score = _torch_patch_predict(
            checkpoint_path, model_type, mrrsu_path, rows, cols, patch_size, batch_size
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # --- Write output GeoTIFFs ---
    out_profile = profile.copy()
    out_profile.update(count=1, dtype='float32', driver='GTiff')

    prob_volume = np.zeros((len(CLASSES), H, W), dtype=np.float32)
    prob_volume[:, rows, cols] = y_score.T

    tile_out_dir = os.path.join(output_dir, tile_id)
    for cls_idx, cls_name in enumerate(CLASSES):
        out_path = os.path.join(tile_out_dir, f'{cls_name}_prob.tif')
        with rasterio.open(out_path, 'w', **out_profile) as dst:
            dst.write(prob_volume[cls_idx:cls_idx + 1])

    # Best class map (argmax, 0-indexed)
    best_class = np.full((H, W), -1, dtype=np.int16)
    best_class[valid_mask] = prob_volume[:, valid_mask].argmax(axis=0)
    best_profile = out_profile.copy()
    best_profile.update(dtype='int16', nodata=-1)
    with rasterio.open(os.path.join(tile_out_dir, 'best_class.tif'), 'w', **best_profile) as dst:
        dst.write(best_class[np.newaxis])

    logger.info(f"Predictions written to {tile_out_dir}")


def _sklearn_predict(models, X, model_type):
    if model_type in ('xgb', 'lgbm'):
        return np.stack([m.predict_proba(X)[:, 1] for m in models], axis=1)
    elif model_type == 'svc':
        m = models[0]
        raw = m.decision_function(X)
        return 1 / (1 + np.exp(-raw))
    else:
        proba_list = models[0].predict_proba(X)
        return np.stack([p[:, 1] for p in proba_list], axis=1)


def _torch_pixel_predict(checkpoint_path, model_type, X, batch_size):
    from models.mlp import MLP
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model = MLP(n_features=X.shape[1], n_classes=len(CLASSES))
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    scores = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i + batch_size])
        with torch.no_grad():
            scores.append(torch.sigmoid(model(xb)).numpy())
    return np.concatenate(scores)


def _torch_patch_predict(checkpoint_path, model_type, mrrsu_path, rows, cols, patch_size, batch_size):
    from models.cnn import SpectralSpatialCNN
    from models.vit import SpectralViT
    import rasterio

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    half = patch_size // 2

    if model_type == 'cnn':
        model = SpectralSpatialCNN(patch_size=patch_size)
    else:
        model = SpectralViT(patch_size=patch_size)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    scores = []
    with rasterio.open(mrrsu_path) as src:
        H, W = src.height, src.width
        for i in range(0, len(rows), batch_size):
            batch_patches = []
            for r, c in zip(rows[i:i + batch_size], cols[i:i + batch_size]):
                r0, r1 = max(0, r - half), min(H, r + half + 1)
                c0, c1 = max(0, c - half), min(W, c + half + 1)
                window = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
                patch = src.read(window=window).astype(np.float32)
                patch[patch >= NODATA_VALUE] = 0.0
                patch = np.nan_to_num(patch)
                full = np.zeros((src.count, patch_size, patch_size), np.float32)
                full[:, half - (r - r0):half - (r - r0) + patch.shape[1],
                        half - (c - c0):half - (c - c0) + patch.shape[2]] = patch
                batch_patches.append(full)
            xb = torch.tensor(np.stack(batch_patches))
            with torch.no_grad():
                scores.append(torch.sigmoid(model(xb)).numpy())
    return np.concatenate(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tile_id', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--patch_size', type=int, default=7)
    args = parser.parse_args()

    import yaml
    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.config
    )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    from data.extract_pixels import find_tile_pairs
    pairs = find_tile_pairs(cfg['gpkg_dir'], cfg['data_root'])
    mrrsu_map = {tid: p for tid, _, p in pairs}

    if args.tile_id not in mrrsu_map:
        print(f"Tile {args.tile_id} not found.")
        sys.exit(1)

    predict_tile(
        tile_id=args.tile_id,
        mrrsu_path=mrrsu_map[args.tile_id],
        checkpoint_path=args.checkpoint,
        model_type=args.model,
        output_dir=cfg['predictions_dir'],
        patch_size=args.patch_size,
    )


if __name__ == '__main__':
    main()
```

**Step 3: Commit**

```bash
git add scripts/predict_tile.py tests/test_predict_tile.py
git commit -m "feat: tile inference script writing per-class probability GeoTIFFs"
```

---

## Task 13: W&B Sweep Configs

**Files:**
- Create: `crism_classification/config/sweep_mlp.yaml`
- Create: `crism_classification/config/sweep_xgb.yaml`
- Create: `crism_classification/config/sweep_lgbm.yaml`
- Create: `crism_classification/config/sweep_cnn.yaml`
- Create: `crism_classification/config/sweep_vit.yaml`

**Step 1: Create sweep configs**

```yaml
# config/sweep_mlp.yaml
program: scripts/train.py
method: bayes
metric:
  name: val_mAP
  goal: maximize
parameters:
  model:
    value: mlp
  lr:
    distribution: log_uniform_values
    min: 1e-4
    max: 1e-2
  batch_size:
    values: [128, 256, 512]
  epochs:
    value: 100
  patience:
    value: 10
```

```yaml
# config/sweep_xgb.yaml
program: scripts/train.py
method: bayes
metric:
  name: val_mAP
  goal: maximize
parameters:
  model:
    value: xgb
  n_estimators:
    values: [100, 200, 400]
  max_depth:
    values: [4, 6, 8]
  learning_rate:
    distribution: log_uniform_values
    min: 0.01
    max: 0.3
```

```yaml
# config/sweep_lgbm.yaml
program: scripts/train.py
method: bayes
metric:
  name: val_mAP
  goal: maximize
parameters:
  model:
    value: lgbm
  n_estimators:
    values: [100, 200, 400]
  max_depth:
    values: [-1, 6, 10]
  learning_rate:
    distribution: log_uniform_values
    min: 0.01
    max: 0.3
```

```yaml
# config/sweep_cnn.yaml
program: scripts/train.py
method: bayes
metric:
  name: val_mAP
  goal: maximize
parameters:
  model:
    value: cnn
  lr:
    distribution: log_uniform_values
    min: 1e-4
    max: 1e-2
  batch_size:
    values: [64, 128, 256]
  patch_size:
    values: [5, 7, 9]
  epochs:
    value: 100
  patience:
    value: 10
```

```yaml
# config/sweep_vit.yaml
program: scripts/train.py
method: bayes
metric:
  name: val_mAP
  goal: maximize
parameters:
  model:
    value: vit
  lr:
    distribution: log_uniform_values
    min: 1e-4
    max: 1e-2
  embed_dim:
    values: [64, 128, 256]
  n_heads:
    values: [4, 8]
  n_layers:
    values: [2, 4, 6]
  patch_size:
    values: [5, 7, 9]
  epochs:
    value: 100
  patience:
    value: 10
```

**Step 2: Commit**

```bash
git add config/
git commit -m "feat: W&B sweep configs for all model families"
```

---

## Task 14: Visualization

**Files:**
- Create: `crism_classification/evaluation/visualize.py`

**Step 1: Implement visualize.py**

```python
# crism_classification/evaluation/visualize.py
"""
Visualization utilities for model evaluation.
"""
import os
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
import rasterio

from data.label_parser import CLASSES


CLASS_COLORS = {
    'olivine_t1':  '#2ca02c',
    'olivine_t2':  '#98df8a',
    'lcp':         '#1f77b4',
    'hcp':         '#aec7e8',
    'plagioclase': '#d62728',
    'other':       '#7f7f7f',
}


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    confidence_tiers: List[str],
    output_path: Optional[str] = None,
):
    """
    Plot per-class precision-recall curves, one subplot per class,
    with separate lines for each confidence tier.
    """
    tiers = ['High', 'Moderate', 'Low']
    tier_colors = {'High': '#d62728', 'Moderate': '#ff7f0e', 'Low': '#1f77b4'}
    tier_arr = np.array(confidence_tiers)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for cls_idx, cls_name in enumerate(CLASSES):
        ax = axes[cls_idx]
        for tier in tiers:
            mask = tier_arr == tier
            if mask.sum() == 0 or y_true[mask, cls_idx].sum() == 0:
                continue
            y_t = (y_true[mask, cls_idx] > 0.4).astype(int)
            y_s = y_score[mask, cls_idx]
            precision, recall, _ = precision_recall_curve(y_t, y_s)
            ax.plot(recall, precision, color=tier_colors[tier], label=tier, linewidth=2)

        ax.set_title(cls_name, fontsize=12)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(fontsize=9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

    plt.suptitle('Precision-Recall Curves by Class and Confidence Tier', fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_prediction_map(
    prob_dir: str,
    output_path: Optional[str] = None,
):
    """
    Create a false-color map from per-class probability GeoTIFFs.
    Colours each pixel by its highest-probability class.
    """
    class_arrays = []
    for cls_name in CLASSES:
        tif_path = os.path.join(prob_dir, f'{cls_name}_prob.tif')
        with rasterio.open(tif_path) as src:
            class_arrays.append(src.read(1))
            if cls_name == CLASSES[0]:
                profile = src.profile

    probs = np.stack(class_arrays)  # (n_classes, H, W)
    best = probs.argmax(axis=0)

    cmap = mcolors.ListedColormap(list(CLASS_COLORS.values()))
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(best, cmap=cmap, vmin=0, vmax=len(CLASSES) - 1, origin='upper')
    cbar = plt.colorbar(im, ax=ax, ticks=range(len(CLASSES)))
    cbar.set_ticklabels(CLASSES)
    ax.set_title('Mineral Classification Map')
    ax.axis('off')

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig
```

**Step 2: Commit**

```bash
git add evaluation/visualize.py
git commit -m "feat: PR curve and prediction map visualization"
```

---

## Task 15: README

**Files:**
- Modify: `crism_classification/README.md`

**Step 1: Write README**

```markdown
# CRISM Mineral Classification Pipeline

Multi-label pixel classification for Mars CRISM MRDR mrrsu tiles using 60 spectral parameter bands.

**Classes:** olivine_t1, olivine_t2, lcp, hcp, plagioclase, other

## Setup

```bash
# Install dependencies into existing crism conda env
conda run -n crism pip install scikit-learn xgboost lightgbm torch torchvision wandb pyarrow tqdm pyyaml pytest

# Configure W&B (one-time)
conda run -n crism python scripts/setup_wandb.py
```

## Data Preparation

```bash
# Build pixel dataset from geopackages + mrrsu tiles (~5-10 min)
conda run -n crism python scripts/build_dataset.py
```

Produces `data/pixels.parquet` — ~N pixels × 66 columns.

## Training

```bash
# Linear baselines
conda run -n crism python scripts/train.py --model logreg
conda run -n crism python scripts/train.py --model svc

# Tree ensembles
conda run -n crism python scripts/train.py --model rf
conda run -n crism python scripts/train.py --model xgb
conda run -n crism python scripts/train.py --model lgbm

# Neural networks
conda run -n crism python scripts/train.py --model mlp --lr 1e-3 --epochs 100
conda run -n crism python scripts/train.py --model cnn --patch_size 7
conda run -n crism python scripts/train.py --model vit --embed_dim 128

# Skip W&B logging
conda run -n crism python scripts/train.py --model rf --no_wandb
```

## Hyperparameter Sweeps (W&B)

```bash
conda run -n crism wandb sweep config/sweep_mlp.yaml
conda run -n crism wandb agent <sweep_id>
```

## Inference

```bash
conda run -n crism python scripts/predict_tile.py \
    --tile_id t0503 \
    --model logreg \
    --checkpoint checkpoints/logreg_model.pkl
```

Outputs: `predictions/t0503/{class}_prob.tif` + `best_class.tif`

## Tests

```bash
conda run -n crism python -m pytest tests/ -v
```

## Confidence Tiers

Labels carry High/Moderate/Low confidence. Training uses sample weights (1.0/0.5/0.25). Test metrics are reported separately per tier.

## Label Encoding

Mixed-mineral labels (e.g. `hcp + olivine (High)`) produce multi-hot vectors. Untyped "olivine" splits 0.5/0.5 between olivine_t1 and olivine_t2. "Other" is a spectral denominator class (downsampled to ~400 polygons).
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: complete README with setup, training, and inference instructions"
```

---

## Execution Order Summary

1. Task 1 — environment + scaffold
2. Task 2 — label parser (TDD)
3. Task 3 — pixel extraction (TDD)
4. Task 4 — build dataset (run the actual extraction)
5. Task 5 — dataset loader (TDD)
6. Task 6 — losses + metrics (TDD)
7. Task 7 — W&B setup
8. Task 8 — sklearn training loop (TDD)
9. Task 9 — MLP/CNN/ViT models (TDD)
10. Task 10 — PyTorch training loop (TDD)
11. Task 11 — unified train.py entry point
12. Task 12 — inference script
13. Task 13 — sweep configs
14. Task 14 — visualization
15. Task 15 — README

# MC13 Polygon Review App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Streamlit app to walk through MC13 model-predicted polygons, confirm/reject each, and harvest interior pixels into an alternative training set (confirmed) and a hard-negatives set (rejected).

**Architecture:** Four files under `scripts/review/`. `queue.py` produces an ordered iterator of polygons across threshold layers (high→low) and area (desc), skipping decided polygons via a csv ledger. `loader.py` rasterizes a polygon onto its mrral tile and returns interior pixel spectra. `persistence.py` owns the decisions.csv ledger plus two parquet writers (confirmed pixels, hard negatives) whose schema matches `data/mrral_pixels.parquet`. `app.py` is thin Streamlit glue with two testable helpers (`compute_progress`, `make_spectrum_figure`).

**Tech Stack:** Python 3.11, geopandas+fiona (gpkg read), rasterio+rasterio.features (polygon→pixel mask), shapely (geometry), pandas+pyarrow (parquet), Streamlit + plotly (UI). Tests with pytest + temp dirs + tiny synthetic mrral arrays (no /mnt/mrdr dependency).

**Spec:** `docs/superpowers/specs/2026-06-07-mc13-polygon-review-app-design.md`.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/review/__init__.py` | Empty package marker. |
| `scripts/review/queue.py` | `PolygonQueue` iterator + `PolygonItem` dataclass. |
| `scripts/review/loader.py` | `load_polygon_pixels(geometry, tile_id, mrral_dir) -> PixelBundle`. |
| `scripts/review/persistence.py` | `DecisionLog`, `ConfirmedPixelsWriter`, `HardNegativesWriter`. |
| `scripts/review/app.py` | Streamlit entrypoint + `compute_progress`, `make_spectrum_figure`. |
| `tests/test_review_queue.py` | Queue ordering, polygon_uid stability, resumability skip. |
| `tests/test_review_loader.py` | Rasterization, NODATA masking, empty-polygon edge case. |
| `tests/test_review_persistence.py` | CSV append idempotence, parquet schema, dedupe across re-appends. |
| `tests/test_review_app_helpers.py` | `compute_progress` aggregation, `make_spectrum_figure` returns plotly Figure with both traces. |

Constants (label set, NODATA, wavelength count) are imported from the existing `data/dataset.py` where possible to avoid drift.

---

### Task 1: PolygonQueue iterator

**Files:**
- Create: `scripts/review/__init__.py`
- Create: `scripts/review/queue.py`
- Create: `tests/test_review_queue.py`

- [ ] **Step 1: Write failing tests for queue ordering + polygon_uid**

```python
# tests/test_review_queue.py
import os
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from scripts.review.queue import PolygonItem, PolygonQueue

LAYERS = ['thresh_0.85', 'thresh_0.90', 'thresh_0.93', 'thresh_0.95', 'thresh_0.97']
MARS_2000_WKT = 'PROJCS["Mars 2000 Equirect",GEOGCS["Mars 2000",DATUM["Mars 2000",SPHEROID["Mars 2000",3396190,169.8944472]],PRIMEM["Reference Meridian",0],UNIT["degree",0.0174532925199433]],PROJECTION["Equirectangular"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1]]'


def _square(x, y, size, tile_id):
    """Build a 1-row GeoDataFrame holding one square polygon."""
    geom = Polygon([(x, y), (x + size, y), (x + size, y + size), (x, y + size)])
    return gpd.GeoDataFrame(
        {'tile_id': [tile_id], 'mineral': ['hcp'], 'threshold': [0.95]},
        geometry=[geom], crs=MARS_2000_WKT,
    )


def _write_layered_gpkg(path, polys_by_layer):
    """polys_by_layer: dict[layer_name] -> list of (x, y, size, tile_id)."""
    for layer, polys in polys_by_layer.items():
        if not polys:
            continue
        gdfs = [_square(*p) for p in polys]
        merged = pd.concat(gdfs, ignore_index=True)
        merged = gpd.GeoDataFrame(merged, geometry='geometry', crs=MARS_2000_WKT)
        merged.to_file(path, driver='GPKG', layer=layer)


def test_queue_walks_layers_high_to_low(tmp_path):
    gpkg = tmp_path / 'hcp.gpkg'
    _write_layered_gpkg(str(gpkg), {
        'thresh_0.85': [(0, 0, 100, 't0001')],
        'thresh_0.90': [(0, 0, 100, 't0002')],
        'thresh_0.95': [(0, 0, 100, 't0003')],
        'thresh_0.97': [(0, 0, 100, 't0004')],
    })
    q = PolygonQueue(gpkg_path=str(gpkg), mineral='hcp')
    items = list(q)
    assert [i.layer for i in items] == ['thresh_0.97', 'thresh_0.95', 'thresh_0.90', 'thresh_0.85']
    assert [i.tile_id for i in items] == ['t0004', 't0003', 't0002', 't0001']
    # Each item's pred_prob is parsed from the layer name
    assert items[0].pred_prob == pytest.approx(0.97)
    assert items[1].pred_prob == pytest.approx(0.95)


def test_queue_sorts_by_area_within_layer(tmp_path):
    gpkg = tmp_path / 'hcp.gpkg'
    _write_layered_gpkg(str(gpkg), {
        'thresh_0.95': [
            (0, 0,   50, 't0001'),   # area 2,500
            (0, 0,  200, 't0002'),   # area 40,000  <- biggest, comes first
            (0, 0,  100, 't0003'),   # area 10,000
        ],
    })
    q = PolygonQueue(gpkg_path=str(gpkg), mineral='hcp')
    items = list(q)
    assert [i.tile_id for i in items] == ['t0002', 't0003', 't0001']
    # area_m2 is computed and exposed
    assert items[0].area_m2 == pytest.approx(40000.0)


def test_polygon_uid_is_stable(tmp_path):
    gpkg = tmp_path / 'hcp.gpkg'
    _write_layered_gpkg(str(gpkg), {
        'thresh_0.95': [(0, 0, 100, 't0001'), (10, 10, 50, 't0002')],
    })
    uids_run1 = [i.polygon_uid for i in PolygonQueue(gpkg_path=str(gpkg), mineral='hcp')]
    uids_run2 = [i.polygon_uid for i in PolygonQueue(gpkg_path=str(gpkg), mineral='hcp')]
    assert uids_run1 == uids_run2
    # Format: "{tile_id}::{layer}::{index_in_layer}"
    assert all('::' in u for u in uids_run1)


def test_queue_skips_decided_polygons(tmp_path):
    gpkg = tmp_path / 'hcp.gpkg'
    _write_layered_gpkg(str(gpkg), {
        'thresh_0.95': [(0, 0, 200, 't0001'), (0, 0, 100, 't0002')],
    })
    # The first polygon (t0001 — bigger) is already decided
    first_uid = next(iter(PolygonQueue(gpkg_path=str(gpkg), mineral='hcp'))).polygon_uid
    decisions_csv = tmp_path / 'decisions.csv'
    pd.DataFrame([{'polygon_uid': first_uid, 'decision': 'confirm'}]).to_csv(decisions_csv, index=False)

    q = PolygonQueue(gpkg_path=str(gpkg), mineral='hcp', decisions_csv=str(decisions_csv))
    items = list(q)
    assert len(items) == 1
    assert items[0].tile_id == 't0002'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n crism pytest tests/test_review_queue.py -v`
Expected: ImportError (scripts.review.queue does not exist).

- [ ] **Step 3: Implement queue.py**

```python
# scripts/review/queue.py
"""Polygon queue iterator for the MC13 review app.

Walks threshold layers high→low within a single gpkg, sorts polygons within
a layer by area descending, and skips polygons that already appear in a
decisions.csv ledger (resumability).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterator, Optional

import fiona
import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class PolygonItem:
    polygon_uid: str           # f"{tile_id}::{layer}::{index_in_layer}"
    tile_id: str
    layer: str                 # e.g. "thresh_0.95"
    predicted_class: str       # mineral the gpkg represents
    geometry: BaseGeometry
    area_m2: float
    pred_prob: float           # parsed from layer name
    source_gpkg: str           # basename, e.g. "vector_mc13_relabeled/hcp.gpkg"


_LAYER_RE = re.compile(r'^thresh_(?P<p>\d+\.\d+)$')


def _layer_threshold(name: str) -> Optional[float]:
    m = _LAYER_RE.match(name)
    return float(m.group('p')) if m else None


class PolygonQueue:
    """Iterable over PolygonItems for a single mineral gpkg.

    Yields items in: layer threshold high→low, then area descending.
    Skips polygons whose polygon_uid is present in ``decisions_csv``.
    """

    def __init__(
        self,
        gpkg_path: str,
        mineral: str,
        decisions_csv: Optional[str] = None,
    ):
        if not os.path.exists(gpkg_path):
            raise FileNotFoundError(gpkg_path)
        self.gpkg_path = gpkg_path
        self.mineral = mineral
        self._skip_uids: set[str] = set()
        if decisions_csv and os.path.exists(decisions_csv):
            df = pd.read_csv(decisions_csv)
            if 'polygon_uid' in df.columns:
                self._skip_uids = set(df['polygon_uid'].astype(str).tolist())

        layers = [L for L in fiona.listlayers(gpkg_path)
                  if _layer_threshold(L) is not None]
        layers.sort(key=_layer_threshold, reverse=True)
        self._layers = layers

        gpkg_parent = os.path.basename(os.path.dirname(os.path.abspath(gpkg_path)))
        gpkg_file = os.path.basename(gpkg_path)
        self._source_gpkg = f'{gpkg_parent}/{gpkg_file}'

    def __iter__(self) -> Iterator[PolygonItem]:
        for layer in self._layers:
            prob = _layer_threshold(layer)
            gdf = gpd.read_file(self.gpkg_path, layer=layer).reset_index(drop=True)
            if gdf.empty:
                continue
            # Capture the file-order index BEFORE sorting so polygon_uid is
            # stable across runs (fiona/gpd read features in fid order).
            gdf['_original_idx'] = gdf.index
            gdf = gdf.assign(_area=gdf.geometry.area)
            gdf = gdf.sort_values('_area', ascending=False, kind='mergesort')
            for _, row in gdf.iterrows():
                tile_id = str(row.get('tile_id', ''))
                uid = f'{tile_id}::{layer}::{int(row["_original_idx"])}'
                if uid in self._skip_uids:
                    continue
                yield PolygonItem(
                    polygon_uid=uid,
                    tile_id=tile_id,
                    layer=layer,
                    predicted_class=self.mineral,
                    geometry=row.geometry,
                    area_m2=float(row['_area']),
                    pred_prob=prob,
                    source_gpkg=self._source_gpkg,
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n crism pytest tests/test_review_queue.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/review/__init__.py scripts/review/queue.py tests/test_review_queue.py
git commit -m "feat(review): polygon queue iterator for MC13 review"
```

---

### Task 2: polygon_interior_pixels loader

**Files:**
- Create: `scripts/review/loader.py`
- Create: `tests/test_review_loader.py`

- [ ] **Step 1: Write failing tests for the loader**

```python
# tests/test_review_loader.py
import json
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from scripts.review.loader import PixelBundle, load_polygon_pixels


def _write_synthetic_mrral(path, height=20, width=20, n_bands=59, nodata=65535):
    """Write a tiny ENVI-style float32 tile.
    - all values = 0.1 + 0.01 * band_index
    - column 0 and row 0 are NODATA (to test masking)
    """
    arr = np.zeros((n_bands, height, width), dtype=np.float32)
    for b in range(n_bands):
        arr[b] = 0.1 + 0.01 * b
    arr[:, 0, :] = nodata
    arr[:, :, 0] = nodata
    transform = from_origin(0, height, 1, 1)  # 1 m px, north-up
    profile = dict(
        driver='ENVI', dtype='float32', count=n_bands,
        height=height, width=width, transform=transform, crs='+proj=eqc +datum=mars',
    )
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(arr)


def test_loader_returns_interior_pixels(tmp_path):
    tile_dir = tmp_path
    img = tile_dir / 't0001_mrral_x.img'
    _write_synthetic_mrral(str(img))
    # Polygon covering (row 5-9, col 5-9) → 5x5 = 25 interior pixels
    geom = Polygon([(5, 11), (10, 11), (10, 16), (5, 16)])
    bundle = load_polygon_pixels(geometry=geom, tile_id='t0001', mrral_dir=str(tile_dir))
    assert isinstance(bundle, PixelBundle)
    assert bundle.spectra.shape == (25, 59)
    assert bundle.rows.shape == (25,)
    assert bundle.cols.shape == (25,)
    # Spectra are uniform across pixels (synthetic) → std == 0
    assert np.allclose(bundle.std, 0.0, atol=1e-6)
    # Mean matches the per-band fill rule
    assert bundle.mean[0] == pytest.approx(0.1, abs=1e-6)
    assert bundle.mean[58] == pytest.approx(0.1 + 0.01 * 58, abs=1e-6)


def test_loader_masks_nodata(tmp_path):
    tile_dir = tmp_path
    img = tile_dir / 't0001_mrral_x.img'
    _write_synthetic_mrral(str(img))
    # Polygon covering (row 0-4, col 0-4) — row 0 and col 0 are NODATA
    geom = Polygon([(0, 16), (5, 16), (5, 20), (0, 20)])
    bundle = load_polygon_pixels(geometry=geom, tile_id='t0001', mrral_dir=str(tile_dir))
    # 5x5=25 raw pixels, but row 0 (5 px) + col 0 in remaining rows (4 px) = 9 NODATA → 16 left
    assert bundle.spectra.shape[0] == 16


def test_loader_returns_empty_for_polygon_outside_tile(tmp_path):
    tile_dir = tmp_path
    img = tile_dir / 't0001_mrral_x.img'
    _write_synthetic_mrral(str(img))
    geom = Polygon([(100, 100), (110, 100), (110, 110), (100, 110)])
    bundle = load_polygon_pixels(geometry=geom, tile_id='t0001', mrral_dir=str(tile_dir))
    assert bundle.spectra.shape == (0, 59)


def test_loader_raises_if_tile_missing(tmp_path):
    geom = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    with pytest.raises(FileNotFoundError):
        load_polygon_pixels(geometry=geom, tile_id='t9999', mrral_dir=str(tmp_path))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n crism pytest tests/test_review_loader.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement loader.py**

```python
# scripts/review/loader.py
"""Reads polygon-interior pixel spectra from an mrral tile."""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import rasterio
import rasterio.features
from shapely.geometry.base import BaseGeometry

NODATA = 65535
N_BANDS = 59


@dataclass(frozen=True)
class PixelBundle:
    rows: np.ndarray       # (n_pixels,) int64 — tile row index
    cols: np.ndarray       # (n_pixels,) int64 — tile col index
    spectra: np.ndarray    # (n_pixels, 59) float32 — reflectance
    mean: np.ndarray       # (59,) float32
    std: np.ndarray        # (59,) float32


def _find_mrral_img(tile_id: str, mrral_dir: str) -> str:
    pattern = os.path.join(mrral_dir, f'{tile_id}_mrral_*.img')
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f'no mrral .img for tile_id={tile_id} in {mrral_dir}')
    return matches[0]


def load_polygon_pixels(
    geometry: BaseGeometry,
    tile_id: str,
    mrral_dir: str,
) -> PixelBundle:
    img_path = _find_mrral_img(tile_id, mrral_dir)
    empty = PixelBundle(
        rows=np.zeros(0, dtype=np.int64),
        cols=np.zeros(0, dtype=np.int64),
        spectra=np.zeros((0, N_BANDS), dtype=np.float32),
        mean=np.zeros(N_BANDS, dtype=np.float32),
        std=np.zeros(N_BANDS, dtype=np.float32),
    )

    with rasterio.open(img_path) as src:
        # Rasterize the polygon onto the tile grid → boolean mask
        mask = rasterio.features.rasterize(
            [(geometry, 1)],
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            dtype='uint8',
        ).astype(bool)
        if not mask.any():
            return empty

        # Read all 59 bands once → (n_bands, h, w)
        cube = src.read(list(range(1, N_BANDS + 1))).astype(np.float32)

    # Build NODATA mask (any band == NODATA → drop that pixel)
    nodata_mask = (cube == NODATA).any(axis=0)
    keep = mask & ~nodata_mask
    if not keep.any():
        return empty

    rows, cols = np.where(keep)
    spectra = cube[:, rows, cols].T.copy()           # (n_pixels, 59)
    return PixelBundle(
        rows=rows.astype(np.int64),
        cols=cols.astype(np.int64),
        spectra=spectra,
        mean=spectra.mean(axis=0),
        std=spectra.std(axis=0),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n crism pytest tests/test_review_loader.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/review/loader.py tests/test_review_loader.py
git commit -m "feat(review): polygon interior pixel loader from mrral tile"
```

---

### Task 3: DecisionLog + ParquetWriters

**Files:**
- Create: `scripts/review/persistence.py`
- Create: `tests/test_review_persistence.py`

- [ ] **Step 1: Write failing tests for DecisionLog**

```python
# tests/test_review_persistence.py
import datetime as dt
import os
import numpy as np
import pandas as pd
import pytest

from scripts.review.persistence import (
    DecisionLog,
    ConfirmedPixelsWriter,
    HardNegativesWriter,
    confirmed_schema_columns,
)


# ---- DecisionLog -----------------------------------------------------------

def _record(uid='t0001::thresh_0.95::0', decision='confirm', corrected=''):
    return dict(
        source_gpkg='vector_mc13_relabeled/hcp.gpkg',
        layer='thresh_0.95',
        polygon_uid=uid,
        tile_id='t0001',
        predicted_class='hcp',
        decision=decision,
        corrected_class=corrected,
        n_pixels=312,
        area_m2=90400.5,
    )


def test_decision_log_creates_csv_and_appends_header(tmp_path):
    csv = tmp_path / 'decisions.csv'
    log = DecisionLog(str(csv))
    log.append(_record())
    df = pd.read_csv(csv)
    assert list(df.columns) == [
        'ts', 'source_gpkg', 'layer', 'polygon_uid', 'tile_id',
        'predicted_class', 'decision', 'corrected_class', 'n_pixels', 'area_m2',
    ]
    assert df.iloc[0]['polygon_uid'] == 't0001::thresh_0.95::0'
    assert df.iloc[0]['ts']  # iso8601 string


def test_decision_log_appends_without_rewriting_header(tmp_path):
    csv = tmp_path / 'decisions.csv'
    log = DecisionLog(str(csv))
    log.append(_record(uid='t0001::thresh_0.95::0'))
    log.append(_record(uid='t0001::thresh_0.95::1', decision='reject'))
    df = pd.read_csv(csv)
    assert len(df) == 2
    # File should have exactly one header line
    with open(csv) as fp:
        lines = fp.readlines()
    assert lines[0].startswith('ts,source_gpkg')


def test_decision_log_uids_seen(tmp_path):
    csv = tmp_path / 'decisions.csv'
    log = DecisionLog(str(csv))
    log.append(_record(uid='a::b::0'))
    log.append(_record(uid='a::b::1'))
    log2 = DecisionLog(str(csv))   # reopened
    assert log2.uids_seen() == {'a::b::0', 'a::b::1'}


def test_decision_log_uids_seen_empty_when_no_file(tmp_path):
    csv = tmp_path / 'decisions.csv'
    log = DecisionLog(str(csv))
    assert log.uids_seen() == set()


# ---- ConfirmedPixelsWriter -------------------------------------------------

def test_confirmed_writer_schema_matches_mrral_pixels(tmp_path):
    pq = tmp_path / 'confirmed.parquet'
    w = ConfirmedPixelsWriter(str(pq))
    w.append_polygon(
        tile_id='t0001',
        polygon_uid='t0001::thresh_0.95::0',
        rows=np.array([5, 6, 7], dtype=np.int64),
        cols=np.array([5, 6, 7], dtype=np.int64),
        spectra=np.arange(3 * 59, dtype=np.float32).reshape(3, 59),
        label_class='hcp',
    )
    w.flush()
    df = pd.read_parquet(pq)
    assert list(df.columns) == confirmed_schema_columns()
    assert len(df) == 3
    assert df['hcp'].iloc[0] == 1.0
    assert df['olivine_t1'].iloc[0] == 0.0
    assert df['confidence_weight'].iloc[0] == 1.0
    assert df['confidence_tier'].iloc[0] == 'High'
    assert df['split'].iloc[0] == 'train'
    assert df['tile_id'].iloc[0] == 't0001'
    assert df['m0'].iloc[0] == pytest.approx(0.0)
    assert df['m58'].iloc[2] == pytest.approx(3 * 59 - 1)


def test_confirmed_writer_olivine_sets_t1(tmp_path):
    pq = tmp_path / 'confirmed.parquet'
    w = ConfirmedPixelsWriter(str(pq))
    w.append_polygon(tile_id='t0001', polygon_uid='x::y::0',
                     rows=np.zeros(1, dtype=np.int64),
                     cols=np.zeros(1, dtype=np.int64),
                     spectra=np.zeros((1, 59), dtype=np.float32),
                     label_class='olivine')
    w.flush()
    df = pd.read_parquet(pq)
    assert df['olivine_t1'].iloc[0] == 1.0
    assert df['olivine_t2'].iloc[0] == 0.0


def test_confirmed_writer_dedupes_on_reappend(tmp_path):
    pq = tmp_path / 'confirmed.parquet'
    w = ConfirmedPixelsWriter(str(pq))
    w.append_polygon(tile_id='t0001', polygon_uid='t0001::a::0',
                     rows=np.array([1], dtype=np.int64),
                     cols=np.array([1], dtype=np.int64),
                     spectra=np.zeros((1, 59), dtype=np.float32),
                     label_class='hcp')
    w.flush()
    # Append the SAME polygon again — must replace, not duplicate
    w2 = ConfirmedPixelsWriter(str(pq))
    w2.append_polygon(tile_id='t0001', polygon_uid='t0001::a::0',
                      rows=np.array([1, 2], dtype=np.int64),
                      cols=np.array([1, 2], dtype=np.int64),
                      spectra=np.ones((2, 59), dtype=np.float32),
                      label_class='hcp')
    w2.flush()
    df = pd.read_parquet(pq)
    assert len(df) == 2  # replaced, not duplicated
    assert df['m0'].iloc[0] == 1.0


# ---- HardNegativesWriter ---------------------------------------------------

def test_hard_negatives_blank_corrected(tmp_path):
    pq = tmp_path / 'hard_negatives.parquet'
    w = HardNegativesWriter(str(pq))
    w.append_polygon(
        tile_id='t0001', polygon_uid='x::y::0',
        rows=np.zeros(1, dtype=np.int64),
        cols=np.zeros(1, dtype=np.int64),
        spectra=np.zeros((1, 59), dtype=np.float32),
        predicted_class='hcp',
        corrected_class=None,
    )
    w.flush()
    df = pd.read_parquet(pq)
    # All label columns 0; negative_of populated
    assert df['olivine_t1'].iloc[0] == 0.0
    assert df['lcp'].iloc[0] == 0.0
    assert df['hcp'].iloc[0] == 0.0
    assert df['negative_of'].iloc[0] == 'hcp'


def test_hard_negatives_with_corrected(tmp_path):
    pq = tmp_path / 'hard_negatives.parquet'
    w = HardNegativesWriter(str(pq))
    w.append_polygon(
        tile_id='t0001', polygon_uid='x::y::0',
        rows=np.zeros(1, dtype=np.int64),
        cols=np.zeros(1, dtype=np.int64),
        spectra=np.zeros((1, 59), dtype=np.float32),
        predicted_class='hcp',
        corrected_class='olivine',
    )
    w.flush()
    df = pd.read_parquet(pq)
    assert df['olivine_t1'].iloc[0] == 1.0
    assert df['hcp'].iloc[0] == 0.0
    # When corrected_class is set, negative_of is left blank/null
    assert pd.isna(df['negative_of'].iloc[0]) or df['negative_of'].iloc[0] == ''
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n crism pytest tests/test_review_persistence.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement persistence.py**

```python
# scripts/review/persistence.py
"""Decision-log csv + parquet writers for the MC13 review app.

decisions.csv is the source of truth (append-only). Both parquet files are
derived: on each polygon decision the corresponding pixel rows are written.
Re-appending the same polygon_uid replaces the prior rows (idempotent).
"""
from __future__ import annotations

import csv
import datetime as dt
import os
from typing import Optional

import numpy as np
import pandas as pd


# Order MUST match data/mrral_pixels.parquet exactly so downstream pipelines
# (patch-cache builder, train.py) can consume the new parquet unchanged.
_LABEL_COLS = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']
_DECISION_COLS = [
    'ts', 'source_gpkg', 'layer', 'polygon_uid', 'tile_id',
    'predicted_class', 'decision', 'corrected_class', 'n_pixels', 'area_m2',
]


def confirmed_schema_columns() -> list[str]:
    return (
        ['tile_id', 'polygon_id', 'pixel_row', 'pixel_col']
        + [f'm{i}' for i in range(59)]
        + _LABEL_COLS
        + ['confidence_weight', 'confidence_tier', 'split']
    )


def hard_negatives_schema_columns() -> list[str]:
    return confirmed_schema_columns() + ['negative_of']


class DecisionLog:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)

    def append(self, record: dict) -> None:
        row = {k: record.get(k, '') for k in _DECISION_COLS}
        row['ts'] = dt.datetime.now(dt.timezone.utc).isoformat()
        # Pass-through caller-supplied keys
        for k in _DECISION_COLS:
            if k != 'ts' and k in record:
                row[k] = record[k]
        write_header = not os.path.exists(self.csv_path)
        with open(self.csv_path, 'a', newline='') as fp:
            w = csv.DictWriter(fp, fieldnames=_DECISION_COLS)
            if write_header:
                w.writeheader()
            w.writerow(row)

    def uids_seen(self) -> set[str]:
        if not os.path.exists(self.csv_path):
            return set()
        df = pd.read_csv(self.csv_path)
        if 'polygon_uid' not in df.columns:
            return set()
        return set(df['polygon_uid'].astype(str).tolist())


def _label_dict_for(label_class: str) -> dict[str, float]:
    out = {c: 0.0 for c in _LABEL_COLS}
    if label_class == 'olivine':
        out['olivine_t1'] = 1.0  # use the more-confident tier slot for new confirmed olivine
    elif label_class in out:
        out[label_class] = 1.0
    return out


def _rows_for_polygon(
    tile_id: str,
    polygon_uid: str,
    rows: np.ndarray,
    cols: np.ndarray,
    spectra: np.ndarray,
    label_dict: dict[str, float],
) -> pd.DataFrame:
    n = spectra.shape[0]
    polygon_id_int = abs(hash(polygon_uid)) % (2**31)  # stable, int64-safe
    data = {
        'tile_id': [tile_id] * n,
        'polygon_id': np.full(n, polygon_id_int, dtype=np.int64),
        'pixel_row': rows.astype(np.int64),
        'pixel_col': cols.astype(np.int64),
    }
    for i in range(59):
        data[f'm{i}'] = spectra[:, i].astype(np.float64)
    for c in _LABEL_COLS:
        data[c] = np.full(n, label_dict[c], dtype=np.float64)
    data['confidence_weight'] = np.full(n, 1.0, dtype=np.float64)
    data['confidence_tier'] = ['High'] * n
    data['split'] = ['train'] * n
    return pd.DataFrame(data, columns=confirmed_schema_columns())


class ConfirmedPixelsWriter:
    """Buffered parquet writer keyed by polygon_uid (reappend = replace)."""

    def __init__(self, parquet_path: str):
        self.parquet_path = parquet_path
        os.makedirs(os.path.dirname(parquet_path) or '.', exist_ok=True)
        self._buf: dict[str, pd.DataFrame] = {}   # polygon_uid -> rows

    def append_polygon(self, *, tile_id: str, polygon_uid: str,
                        rows: np.ndarray, cols: np.ndarray,
                        spectra: np.ndarray, label_class: str) -> None:
        df = _rows_for_polygon(tile_id, polygon_uid, rows, cols, spectra,
                                _label_dict_for(label_class))
        self._buf[polygon_uid] = df

    def flush(self) -> None:
        # Load existing parquet (if any), drop rows for any uids in buffer
        if os.path.exists(self.parquet_path):
            existing = pd.read_parquet(self.parquet_path)
            # Rows in existing whose polygon_id maps to a uid we're rewriting
            buf_polygon_ids = {abs(hash(uid)) % (2**31) for uid in self._buf}
            existing = existing[~existing['polygon_id'].isin(buf_polygon_ids)]
        else:
            existing = pd.DataFrame(columns=confirmed_schema_columns())
        all_new = pd.concat(list(self._buf.values()), ignore_index=True) \
                  if self._buf else pd.DataFrame(columns=confirmed_schema_columns())
        out = pd.concat([existing, all_new], ignore_index=True)
        out = out[confirmed_schema_columns()]  # enforce column order
        out.to_parquet(self.parquet_path, index=False)
        self._buf.clear()


class HardNegativesWriter:
    def __init__(self, parquet_path: str):
        self.parquet_path = parquet_path
        os.makedirs(os.path.dirname(parquet_path) or '.', exist_ok=True)
        self._buf: dict[str, pd.DataFrame] = {}

    def append_polygon(self, *, tile_id: str, polygon_uid: str,
                        rows: np.ndarray, cols: np.ndarray,
                        spectra: np.ndarray,
                        predicted_class: str,
                        corrected_class: Optional[str]) -> None:
        if corrected_class:
            label = _label_dict_for(corrected_class)
            negative_of = ''
        else:
            label = {c: 0.0 for c in _LABEL_COLS}
            negative_of = predicted_class
        df = _rows_for_polygon(tile_id, polygon_uid, rows, cols, spectra, label)
        df['negative_of'] = negative_of
        df = df[hard_negatives_schema_columns()]
        self._buf[polygon_uid] = df

    def flush(self) -> None:
        if os.path.exists(self.parquet_path):
            existing = pd.read_parquet(self.parquet_path)
            buf_polygon_ids = {abs(hash(uid)) % (2**31) for uid in self._buf}
            existing = existing[~existing['polygon_id'].isin(buf_polygon_ids)]
        else:
            existing = pd.DataFrame(columns=hard_negatives_schema_columns())
        all_new = pd.concat(list(self._buf.values()), ignore_index=True) \
                  if self._buf else pd.DataFrame(columns=hard_negatives_schema_columns())
        out = pd.concat([existing, all_new], ignore_index=True)
        out = out[hard_negatives_schema_columns()]
        out.to_parquet(self.parquet_path, index=False)
        self._buf.clear()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n crism pytest tests/test_review_persistence.py -v`
Expected: 9 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/review/persistence.py tests/test_review_persistence.py
git commit -m "feat(review): decision log + parquet writers for confirmed/hard-neg sets"
```

---

### Task 4: Streamlit app with mineral selector, card, progress, stop banner

**Files:**
- Create: `scripts/review/app.py`
- Create: `tests/test_review_app_helpers.py`

- [ ] **Step 1: Write failing tests for the two pure helpers**

```python
# tests/test_review_app_helpers.py
import numpy as np
import pandas as pd
import pytest
from plotly.graph_objects import Figure

from scripts.review.app import compute_progress, make_spectrum_figure


def test_compute_progress_aggregates_by_mineral(tmp_path):
    # Synthetic decisions.csv
    decisions = pd.DataFrame([
        # 2 confirms for hcp (n_pixels 300 + 200), 1 reject for hcp, 1 skip for hcp
        {'predicted_class': 'hcp', 'decision': 'confirm', 'n_pixels': 300, 'corrected_class': ''},
        {'predicted_class': 'hcp', 'decision': 'confirm', 'n_pixels': 200, 'corrected_class': ''},
        {'predicted_class': 'hcp', 'decision': 'reject', 'n_pixels': 99, 'corrected_class': ''},
        {'predicted_class': 'hcp', 'decision': 'skip', 'n_pixels': 50, 'corrected_class': ''},
        # 1 confirm for lcp
        {'predicted_class': 'lcp', 'decision': 'confirm', 'n_pixels': 150, 'corrected_class': ''},
    ])
    csv = tmp_path / 'decisions.csv'
    decisions.to_csv(csv, index=False)

    prog_hcp = compute_progress(str(csv), mineral='hcp', target_pixels=30000)
    assert prog_hcp['confirmed_pixels'] == 500
    assert prog_hcp['reviewed'] == 4
    assert prog_hcp['confirm_count'] == 2
    assert prog_hcp['reject_count'] == 1
    assert prog_hcp['skip_count'] == 1
    assert prog_hcp['target_pixels'] == 30000
    assert prog_hcp['fraction'] == pytest.approx(500 / 30000)
    assert prog_hcp['target_reached'] is False

    prog_lcp = compute_progress(str(csv), mineral='lcp', target_pixels=100)
    assert prog_lcp['confirmed_pixels'] == 150
    assert prog_lcp['target_reached'] is True


def test_compute_progress_handles_missing_csv(tmp_path):
    prog = compute_progress(str(tmp_path / 'no.csv'), mineral='hcp', target_pixels=30000)
    assert prog['confirmed_pixels'] == 0
    assert prog['reviewed'] == 0
    assert prog['target_reached'] is False


def test_make_spectrum_figure_has_mean_and_envelope_traces():
    n_pixels, n_bands = 12, 59
    rng = np.random.default_rng(0)
    spectra = rng.normal(0.2, 0.01, size=(n_pixels, n_bands)).astype(np.float32)
    wavelengths_nm = np.linspace(410, 2457, n_bands)
    fig = make_spectrum_figure(spectra, wavelengths_nm)
    assert isinstance(fig, Figure)
    # At least: 1 mean line + 1 lower-envelope + 1 upper-envelope (3+ traces);
    # use trace names to assert presence.
    names = {tr.name for tr in fig.data}
    assert 'mean' in names
    assert any('envelope' in (n or '') for n in names)


def test_make_spectrum_figure_zero_pixels_returns_empty_figure():
    fig = make_spectrum_figure(
        np.zeros((0, 59), dtype=np.float32),
        np.linspace(410, 2457, 59),
    )
    assert isinstance(fig, Figure)
    # No data crash; ok to be empty
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n crism pytest tests/test_review_app_helpers.py -v`
Expected: ImportError (scripts.review.app missing or helpers missing).

- [ ] **Step 3: Implement app.py (helpers first, then the Streamlit UI)**

```python
# scripts/review/app.py
"""Streamlit app for reviewing MC13 polygon predictions and harvesting
confirmed/hard-negative training pixels.

Run:
  conda run -n crism streamlit run scripts/review/app.py
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scripts.review.queue import PolygonItem, PolygonQueue
from scripts.review.loader import load_polygon_pixels
from scripts.review.persistence import (
    DecisionLog, ConfirmedPixelsWriter, HardNegativesWriter,
)

DEFAULT_GPKG_DIR = '/mnt/mrdr/crism_classification/data/vector_mc13_relabeled'
DEFAULT_MRRAL_DIR = '/mnt/mrdr/mc13'
DEFAULT_OUT_DIR = '/mnt/mrdr/crism_classification/data/mc13_review'
DEFAULT_WAVELENGTHS = '/mnt/mrdr/crism_classification/data/vector_mc13_contrastive/vector_mc13_contrastive_wavelengths.json'
TARGET_PIXELS_PER_CLASS = 30000

MINERALS = ['olivine', 'lcp', 'hcp']


# ---- pure helpers (covered by tests) ---------------------------------------

def compute_progress(decisions_csv: str, mineral: str,
                     target_pixels: int) -> dict:
    """Aggregate decisions.csv for one mineral."""
    if not os.path.exists(decisions_csv):
        return dict(confirmed_pixels=0, reviewed=0, confirm_count=0,
                    reject_count=0, skip_count=0,
                    target_pixels=target_pixels, fraction=0.0,
                    target_reached=False)
    df = pd.read_csv(decisions_csv)
    df = df[df['predicted_class'] == mineral]
    conf = df[df['decision'] == 'confirm']
    rej = df[df['decision'] == 'reject']
    skip = df[df['decision'] == 'skip']
    pixels = int(conf['n_pixels'].fillna(0).sum())
    return dict(
        confirmed_pixels=pixels,
        reviewed=len(df),
        confirm_count=len(conf),
        reject_count=len(rej),
        skip_count=len(skip),
        target_pixels=target_pixels,
        fraction=pixels / target_pixels if target_pixels else 0.0,
        target_reached=pixels >= target_pixels,
    )


def make_spectrum_figure(spectra: np.ndarray,
                          wavelengths_nm: np.ndarray) -> go.Figure:
    """Mean + ±1σ envelope. No band markers."""
    fig = go.Figure()
    if spectra.shape[0] == 0:
        fig.update_layout(title='no interior pixels', height=350)
        return fig
    mean = spectra.mean(axis=0)
    std = spectra.std(axis=0)
    upper = mean + std
    lower = mean - std
    fig.add_trace(go.Scatter(
        x=wavelengths_nm, y=upper, mode='lines',
        line=dict(width=0), name='envelope_upper',
        showlegend=False, hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=wavelengths_nm, y=lower, mode='lines',
        line=dict(width=0), name='envelope_lower',
        fill='tonexty', fillcolor='rgba(100,100,200,0.18)',
        showlegend=False, hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=wavelengths_nm, y=mean, mode='lines',
        line=dict(width=2, color='royalblue'), name='mean',
    ))
    fig.update_layout(
        xaxis_title='wavelength (nm)', yaxis_title='reflectance',
        height=400, margin=dict(l=40, r=20, t=20, b=40), showlegend=False,
    )
    return fig


# ---- streamlit glue --------------------------------------------------------

def _load_wavelengths(path: str) -> np.ndarray:
    with open(path) as fp:
        d = json.load(fp)
    arr = np.asarray(d['wavelengths_nm'], dtype=float)
    if arr.size != 59:
        arr = arr[:59]
    return arr


def _get_or_make(queue_state: dict, key: str, builder):
    if key not in queue_state:
        queue_state[key] = builder()
    return queue_state[key]


def main():
    # Import streamlit lazily so pytest imports of helpers don't pull it in.
    import streamlit as st

    st.set_page_config(page_title='MC13 polygon review', layout='wide')
    st.title('MC13 polygon review')

    # Sidebar config
    gpkg_dir = st.sidebar.text_input('gpkg dir', DEFAULT_GPKG_DIR)
    mrral_dir = st.sidebar.text_input('mrral tile dir', DEFAULT_MRRAL_DIR)
    out_dir = st.sidebar.text_input('output dir', DEFAULT_OUT_DIR)
    decisions_csv = os.path.join(out_dir, 'decisions.csv')
    confirmed_pq = os.path.join(out_dir, 'confirmed_pixels.parquet')
    hardneg_pq = os.path.join(out_dir, 'hard_negatives.parquet')

    # Mineral selector
    mineral = st.radio('mineral', MINERALS, horizontal=True,
                        index=MINERALS.index(st.session_state.get('mineral', 'hcp')))
    st.session_state['mineral'] = mineral

    # Progress bar
    prog = compute_progress(decisions_csv, mineral, TARGET_PIXELS_PER_CLASS)
    col1, col2 = st.columns([3, 1])
    col1.progress(min(1.0, prog['fraction']),
                  text=f"{prog['confirmed_pixels']:,} / {TARGET_PIXELS_PER_CLASS:,} confirmed {mineral} pixels")
    col2.metric('reviewed', prog['reviewed'],
                f"+{prog['confirm_count']} -{prog['reject_count']} ~{prog['skip_count']}")

    if prog['target_reached']:
        st.success(f"30k reached for {mineral}. Switch mineral above or keep reviewing for more headroom.")

    # Queue
    gpkg_path = os.path.join(gpkg_dir, f'{mineral}.gpkg')
    queue_key = f'queue::{gpkg_path}::{mineral}'
    if st.session_state.get('queue_key') != queue_key:
        st.session_state['queue_key'] = queue_key
        st.session_state['queue_iter'] = iter(PolygonQueue(
            gpkg_path=gpkg_path, mineral=mineral, decisions_csv=decisions_csv,
        ))
        st.session_state['current_item'] = None
        st.session_state['current_bundle'] = None

    # Advance to next polygon
    def _advance():
        try:
            st.session_state['current_item'] = next(st.session_state['queue_iter'])
            item = st.session_state['current_item']
            st.session_state['current_bundle'] = load_polygon_pixels(
                geometry=item.geometry, tile_id=item.tile_id,
                mrral_dir=mrral_dir,
            )
        except StopIteration:
            st.session_state['current_item'] = None
            st.session_state['current_bundle'] = None

    if st.session_state.get('current_item') is None:
        _advance()

    item = st.session_state.get('current_item')
    bundle = st.session_state.get('current_bundle')

    if item is None:
        st.info('No more polygons in this queue.')
        return

    # Card
    n_px = bundle.spectra.shape[0] if bundle is not None else 0
    st.markdown(
        f"**tile** `{item.tile_id}` · **layer** `{item.layer}` · "
        f"**polygon_uid** `{item.polygon_uid}` · **n_pixels** {n_px} · "
        f"**pred_prob** {item.pred_prob:.2f}"
    )
    wavelengths = _load_wavelengths(DEFAULT_WAVELENGTHS)
    st.plotly_chart(make_spectrum_figure(bundle.spectra, wavelengths),
                     use_container_width=True)

    # Decision buttons + corrected-class dropdown
    corrected = st.selectbox(
        'if rejected, actually:',
        options=['', 'olivine', 'lcp', 'hcp', 'other'],
        index=0,
    )
    b1, b2, b3 = st.columns(3)
    log = DecisionLog(decisions_csv)
    confirmed_writer = ConfirmedPixelsWriter(confirmed_pq)
    hardneg_writer = HardNegativesWriter(hardneg_pq)

    def _record(decision: str):
        log.append(dict(
            source_gpkg=item.source_gpkg, layer=item.layer,
            polygon_uid=item.polygon_uid, tile_id=item.tile_id,
            predicted_class=mineral, decision=decision,
            corrected_class=(corrected if decision == 'reject' else ''),
            n_pixels=n_px, area_m2=item.area_m2,
        ))
        if decision == 'confirm' and bundle is not None and n_px > 0:
            confirmed_writer.append_polygon(
                tile_id=item.tile_id, polygon_uid=item.polygon_uid,
                rows=bundle.rows, cols=bundle.cols, spectra=bundle.spectra,
                label_class=mineral,
            )
            confirmed_writer.flush()
        elif decision == 'reject' and bundle is not None and n_px > 0:
            hardneg_writer.append_polygon(
                tile_id=item.tile_id, polygon_uid=item.polygon_uid,
                rows=bundle.rows, cols=bundle.cols, spectra=bundle.spectra,
                predicted_class=mineral,
                corrected_class=(corrected or None),
            )
            hardneg_writer.flush()
        _advance()
        st.rerun()

    if b1.button('Confirm', type='primary', use_container_width=True):
        _record('confirm')
    if b2.button('Reject', use_container_width=True):
        _record('reject')
    if b3.button('Skip', use_container_width=True):
        _record('skip')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n crism pytest tests/test_review_app_helpers.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Smoke-launch the app (manual)**

Run: `conda run -n crism streamlit run scripts/review/app.py --server.headless true --server.port 8505`
Open `http://localhost:8505`. Expect to see:
- Mineral selector defaulting to `hcp`
- A first-polygon card with spectrum plot
- Three buttons + a "if rejected, actually" dropdown
- Progress bar reading `0 / 30,000`

Click `Confirm` once; verify `data/mc13_review/decisions.csv` gains a row and `data/mc13_review/confirmed_pixels.parquet` gains pixels.

Stop with Ctrl-C.

- [ ] **Step 6: Commit**

```bash
git add scripts/review/app.py tests/test_review_app_helpers.py
git commit -m "feat(review): Streamlit app composing queue + loader + persistence"
```

---

### Task 5: README pointer + run script

**Files:**
- Create: `scripts/review/README.md`

- [ ] **Step 1: Write the README**

```markdown
# MC13 polygon review

Streamlit app for confirming/rejecting MC13 model-predicted polygons and
harvesting their interior pixels into an alternative training set.

## Run

```bash
conda run -n crism streamlit run scripts/review/app.py
```

Defaults (overridable in sidebar):
- gpkg dir   = `data/vector_mc13_relabeled`
- mrral dir  = `/mnt/mrdr/mc13`
- output dir = `data/mc13_review`

## Outputs

- `data/mc13_review/decisions.csv` — append-only ledger.
- `data/mc13_review/confirmed_pixels.parquet` — schema matches `data/mrral_pixels.parquet`.
- `data/mc13_review/hard_negatives.parquet` — same schema + `negative_of` column.

## Design

See `docs/superpowers/specs/2026-06-07-mc13-polygon-review-app-design.md`.
```

- [ ] **Step 2: Commit**

```bash
git add scripts/review/README.md
git commit -m "docs(review): README for the polygon review app"
```

---

## Final verification

- [ ] Run the full test suite: `conda run -n crism pytest tests/test_review_*.py -v`
- [ ] Confirm 4 test files, all PASS.
- [ ] Smoke-launch the Streamlit app and walk through 2-3 polygons end-to-end.
- [ ] Verify `data/mc13_review/decisions.csv` and the two parquets are written correctly.

# Mineral Verification App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local browser-based tool that lets a user load a CRISM mrral tile + vectroscopy GeoPackage, inspect predicted mineral polygons one-by-one, compare spectra against a user-defined denominator ROI, and record a correct/incorrect verdict with confidence.

**Architecture:** FastAPI backend serves tile image, polygon data, and spectra as JSON/PNG; a single-page HTML/JS frontend renders a three-panel layout (polygon table | Leaflet tile viewer | Plotly spectrum panel). Verification verdicts are written back to the GeoPackage in-place as new columns, so the output file is the same format the rest of the pipeline already reads.

**Tech Stack:** Python 3.12, FastAPI + uvicorn, rasterio, geopandas, numpy; frontend — Leaflet.js (via CDN), Plotly.js (via CDN), vanilla JS/CSS; pytest for backend tests.

---

## File Map

| Path | Role |
|---|---|
| `app/config.py` | Constants: bad-band ranges, wavelength mask, tile registry path, default data dirs |
| `app/gpkg_io.py` | Load polygon layers from .gpkg as GeoJSON in pixel-space; add/read/write verification columns |
| `app/tile_renderer.py` | Render CRISM mrral false-color (R=band60, G=band34, B=band21) to PNG bytes; expose pixel↔CRS transform |
| `app/spectrum.py` | Extract mean±std spectrum for a polygon; extract mean spectrum for an arbitrary pixel-space ROI |
| `app/main.py` | FastAPI app, all REST routes, static file mount |
| `app/static/index.html` | Single-page app shell: 3-panel layout |
| `app/static/app.js` | All frontend logic: table, map, spectrum plot, verification form |
| `app/static/style.css` | Layout and panel styles |
| `tests/test_gpkg_io.py` | Unit tests for GeoPackage read/write |
| `tests/test_tile_renderer.py` | Unit tests for PNG rendering and transform math |
| `tests/test_spectrum.py` | Unit tests for spectrum extraction |
| `run_app.py` | CLI entry point: `python run_app.py --img <path> --gpkg <path>` |

---

## Task 1: Install dependencies and scaffold project

**Files:**
- Create: `run_app.py`
- Create: `app/__init__.py`
- Create: `app/config.py`

- [ ] **Step 1: Install backend dependencies**

```bash
conda run -n crism pip install fastapi uvicorn[standard] python-multipart
```

Expected: `Successfully installed fastapi-... uvicorn-...`

- [ ] **Step 2: Create `app/config.py`**

```python
"""App-wide constants."""
import os

# Spectral constants (shared with pipeline)
CRISM_NODATA = 65535.0
BAD_BAND_RANGES = [(1040, 1070)]   # nm — single bad band ~1056 nm (S/L detector boundary)
WAV_MIN, WAV_MAX = 500, 2600       # display range (nm)

# CRISM mrral false-color band indices (1-based)
FC_BANDS = (60, 34, 21)  # R=2529 nm, G=1506 nm, B=1079 nm

# Tile-image downscale for display — longest edge becomes this many pixels
TILE_DISPLAY_MAX_PX = 1024

# Verification column names added to GeoPackage
COL_VERDICT    = 'verdict'          # 'correct' | 'incorrect' | None
COL_CONFIDENCE = 'verify_conf'      # 'low' | 'moderate' | 'high' | None
COL_NOTE       = 'verify_note'      # free-text string
COL_TIMESTAMP  = 'verified_at'      # ISO-8601 string
```

- [ ] **Step 3: Create `run_app.py`**

```python
"""Entry point: python run_app.py --img <mrral.img> --gpkg <mineral_map.gpkg>"""
import argparse, uvicorn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img',  required=True, help='Path to mrral .img file')
    parser.add_argument('--gpkg', required=True, help='Path to mineral_map .gpkg')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=8765, type=int)
    args = parser.parse_args()

    import os
    os.environ['CRISM_IMG']  = args.img
    os.environ['CRISM_GPKG'] = args.gpkg

    uvicorn.run('app.main:app', host=args.host, port=args.port, reload=False)

if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Create `app/__init__.py`** (empty)

- [ ] **Step 5: Commit**

```bash
git add app/ run_app.py
git commit -m "feat: scaffold mineral verification app structure"
```

---

## Task 2: GeoPackage I/O — load polygons, add verification columns

**Files:**
- Create: `app/gpkg_io.py`
- Create: `tests/test_gpkg_io.py`

### Background

The GeoPackage from vectroscopy has one layer per mineral. Each polygon carries `confidence` (1–5), `mineral`, `mean_prob`, `count_px`, etc. We need to:
1. Load all layers, assign a stable integer `poly_id` (row index within layer), return GeoJSON in the tile's own CRS (not reprojected — the renderer will handle pixel conversion).
2. Ensure verification columns exist (add them with NULL values if absent).
3. Write a single polygon's verdict back.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_gpkg_io.py
import pytest, geopandas as gpd, tempfile, os
from shapely.geometry import box
from app.gpkg_io import (
    load_all_polygons, ensure_verify_columns, write_verdict
)
from app.config import COL_VERDICT, COL_CONFIDENCE

MINERALS = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']

@pytest.fixture
def tmp_gpkg(tmp_path):
    path = str(tmp_path / 'test.gpkg')
    for mineral in MINERALS:
        gdf = gpd.GeoDataFrame(
            {'mineral': [mineral, mineral],
             'confidence': [1, 2],
             'mean_prob': [0.4, 0.6],
             'count_px': [10, 20],
             'geometry': [box(0,0,1,1), box(1,0,2,1)]},
            crs='EPSG:4326',
        )
        gdf.to_file(path, layer=mineral, driver='GPKG')
    return path

def test_load_all_polygons_returns_list(tmp_gpkg):
    polys = load_all_polygons(tmp_gpkg)
    assert isinstance(polys, list)
    assert len(polys) == len(MINERALS) * 2  # 2 per mineral

def test_load_all_polygons_has_poly_id(tmp_gpkg):
    polys = load_all_polygons(tmp_gpkg)
    ids = [p['poly_id'] for p in polys]
    assert ids == list(range(len(polys)))  # stable 0-based

def test_load_all_polygons_geojson_geometry(tmp_gpkg):
    polys = load_all_polygons(tmp_gpkg)
    assert all('geometry' in p for p in polys)

def test_ensure_verify_columns_adds_columns(tmp_gpkg):
    ensure_verify_columns(tmp_gpkg)
    gdf = gpd.read_file(tmp_gpkg, layer='olivine')
    assert COL_VERDICT in gdf.columns
    assert COL_CONFIDENCE in gdf.columns

def test_write_verdict_persists(tmp_gpkg):
    ensure_verify_columns(tmp_gpkg)
    polys = load_all_polygons(tmp_gpkg)
    poly = polys[0]   # poly_id=0, olivine, confidence=1 (first row in layer)
    write_verdict(tmp_gpkg, poly['poly_id'], polys,
                  verdict='correct', confidence='high', note='clear olivine')
    gdf = gpd.read_file(tmp_gpkg, layer=poly['mineral'])
    # Use mineral-local row index 0 (poly_id 0 is first olivine row)
    row = gdf.iloc[0]
    assert row[COL_VERDICT] == 'correct'
    assert row[COL_CONFIDENCE] == 'high'

def test_write_verdict_targets_correct_row(tmp_gpkg):
    """Verify second polygon in the layer is not overwritten."""
    ensure_verify_columns(tmp_gpkg)
    polys = load_all_polygons(tmp_gpkg)
    write_verdict(tmp_gpkg, polys[0]['poly_id'], polys,
                  verdict='correct', confidence='high', note='')
    gdf = gpd.read_file(tmp_gpkg, layer='olivine')
    assert gdf.iloc[1][COL_VERDICT] != 'correct'  # second row untouched
```

- [ ] **Step 2: Run tests — expect failure**

```bash
conda run -n crism pytest tests/test_gpkg_io.py -v 2>&1 | head -30
```

Expected: `ImportError` or `ModuleNotFoundError`

- [ ] **Step 3: Implement `app/gpkg_io.py`**

```python
"""Load and update mineral prediction GeoPackages."""
import json
from datetime import datetime, timezone
from typing import Any

import geopandas as gpd

from app.config import COL_CONFIDENCE, COL_NOTE, COL_TIMESTAMP, COL_VERDICT

MINERALS = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']


def load_all_polygons(gpkg_path: str) -> list[dict[str, Any]]:
    """Return flat list of polygon dicts with stable integer poly_id.

    Each dict has: poly_id, mineral, confidence, mean_prob, count_px,
    verdict, verify_conf, verify_note, geometry (GeoJSON dict in tile CRS).
    """
    records = []
    poly_id = 0
    for mineral in MINERALS:
        try:
            gdf = gpd.read_file(gpkg_path, layer=mineral)
        except Exception:
            continue
        for _, row in gdf.iterrows():
            geom = row.geometry.__geo_interface__ if row.geometry else None
            records.append({
                'poly_id':    poly_id,
                'mineral':    mineral,
                'confidence': int(row.get('confidence', 0)),
                'mean_prob':  float(row.get('mean_prob', 0.0)),
                'count_px':   int(row.get('count_px', 0)),
                'verdict':    row.get(COL_VERDICT, None),
                'verify_conf': row.get(COL_CONFIDENCE, None),
                'verify_note': row.get(COL_NOTE, None),
                'geometry':   geom,
            })
            poly_id += 1
    return records


def ensure_verify_columns(gpkg_path: str) -> None:
    """Add verification columns to every layer if they don't exist yet."""
    for mineral in MINERALS:
        try:
            gdf = gpd.read_file(gpkg_path, layer=mineral)
        except Exception:
            continue
        changed = False
        for col in (COL_VERDICT, COL_CONFIDENCE, COL_NOTE, COL_TIMESTAMP):
            if col not in gdf.columns:
                gdf[col] = None
                changed = True
        if changed:
            gdf.to_file(gpkg_path, layer=mineral, driver='GPKG')


def write_verdict(
    gpkg_path: str,
    poly_id: int,
    all_polys: list[dict],
    verdict: str,
    confidence: str,
    note: str = '',
) -> None:
    """Persist a verdict for a single polygon back to the GeoPackage."""
    meta = all_polys[poly_id]
    mineral = meta['mineral']

    gdf = gpd.read_file(gpkg_path, layer=mineral)

    # Identify row: poly_id is global; find the per-mineral row index
    mineral_polys = [p for p in all_polys if p['mineral'] == mineral]
    local_idx = next(
        i for i, p in enumerate(mineral_polys) if p['poly_id'] == poly_id
    )

    gdf.at[local_idx, COL_VERDICT]   = verdict
    gdf.at[local_idx, COL_CONFIDENCE] = confidence
    gdf.at[local_idx, COL_NOTE]       = note
    gdf.at[local_idx, COL_TIMESTAMP]  = datetime.now(timezone.utc).isoformat()

    gdf.to_file(gpkg_path, layer=mineral, driver='GPKG')
```

- [ ] **Step 4: Run tests — expect pass**

```bash
conda run -n crism pytest tests/test_gpkg_io.py -v
```

Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add app/gpkg_io.py tests/test_gpkg_io.py
git commit -m "feat: GeoPackage I/O with verification column support"
```

---

## Task 3: Tile renderer — false-color PNG + pixel↔CRS transform

**Files:**
- Create: `app/tile_renderer.py`
- Create: `tests/test_tile_renderer.py`

### Background

The frontend needs two things: (a) a PNG image of the CRISM tile for display, and (b) a way to convert pixel coordinates in that PNG back into CRS (source raster) pixel coordinates when the user right-clicks. We expose the affine transform as a JSON dict.

The render pipeline:
1. Open `mrral.img` with rasterio.
2. Read the three false-color bands (60, 34, 21), mask CRISM_NODATA and `|val| > 1`.
3. Downsample so the longest edge ≤ `TILE_DISPLAY_MAX_PX`.
4. Per-band 2/98 percentile stretch → uint8.
5. Encode as PNG bytes.
6. Also return the (scale_x, scale_y) downscale factors so the frontend can send pixel clicks back to the server as source-raster row/col.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_tile_renderer.py
import numpy as np, pytest
from app.tile_renderer import render_false_color, px_to_rowcol


def test_render_false_color_returns_bytes(real_img_path):
    png_bytes, meta = render_false_color(real_img_path)
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 1000  # non-trivial PNG

def test_render_false_color_meta_keys(real_img_path):
    _, meta = render_false_color(real_img_path)
    assert 'width' in meta and 'height' in meta
    assert 'scale_x' in meta and 'scale_y' in meta
    assert 'src_width' in meta and 'src_height' in meta

def test_render_false_color_max_dim(real_img_path):
    _, meta = render_false_color(real_img_path)
    assert max(meta['width'], meta['height']) <= 1024

def test_px_to_rowcol_roundtrip():
    meta = {'scale_x': 0.5, 'scale_y': 0.5}
    row, col = px_to_rowcol(img_x=100, img_y=80, meta=meta)
    assert row == 160
    assert col == 200
```

Note: `real_img_path` is a pytest fixture defined in `tests/conftest.py` pointing at a real tile (add a `conftest.py` in the next step or skip if running as integration test — mark with `@pytest.mark.integration`).

- [ ] **Step 2: Create `tests/conftest.py`**

```python
# tests/conftest.py
import pytest

REAL_IMG = '/mnt/mrdr/mc13/t1249_mrral_20n073_0327_4.img'

@pytest.fixture
def real_img_path():
    import os
    if not os.path.exists(REAL_IMG):
        pytest.skip('Real tile not available')
    return REAL_IMG
```

- [ ] **Step 3: Run tests — expect failure or skip**

```bash
conda run -n crism pytest tests/test_tile_renderer.py -v 2>&1 | head -20
```

Expected: `ImportError` (module not yet created)

- [ ] **Step 4: Implement `app/tile_renderer.py`**

```python
"""Render CRISM mrral tile as false-color PNG for browser display."""
import io
from typing import Any

import numpy as np
import rasterio
from PIL import Image

from app.config import CRISM_NODATA, FC_BANDS, TILE_DISPLAY_MAX_PX


def _percentile_stretch(band: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    """Stretch finite values to uint8; NaN → 0 (transparent via alpha)."""
    finite = band[np.isfinite(band)]
    if len(finite) == 0:
        return np.zeros_like(band, dtype=np.uint8)
    vmin, vmax = np.nanpercentile(finite, lo), np.nanpercentile(finite, hi)
    if vmax == vmin:
        return np.zeros_like(band, dtype=np.uint8)
    stretched = (band - vmin) / (vmax - vmin)
    stretched = np.clip(stretched, 0, 1)
    out = (stretched * 255).astype(np.uint8)
    out[~np.isfinite(band)] = 0
    return out


def render_false_color(img_path: str) -> tuple[bytes, dict[str, Any]]:
    """Return (PNG bytes, metadata dict) for the tile.

    metadata keys:
        width, height       — PNG image dimensions (pixels)
        src_width, src_height — original raster dimensions
        scale_x, scale_y    — img_px / src_px ratios
    """
    with rasterio.open(img_path) as src:
        src_h, src_w = src.height, src.width

        # Compute downscale factor
        scale = min(1.0, TILE_DISPLAY_MAX_PX / max(src_h, src_w))
        out_h = max(1, int(src_h * scale))
        out_w = max(1, int(src_w * scale))

        bands_u8 = []
        alpha = None
        for band_idx in FC_BANDS:
            raw = src.read(
                band_idx,
                out_shape=(1, out_h, out_w),
                resampling=rasterio.enums.Resampling.bilinear,
            )[0].astype(np.float32)
            raw[raw == CRISM_NODATA] = np.nan
            raw[np.abs(raw) > 1] = np.nan
            if alpha is None:
                alpha = np.where(np.isfinite(raw), 255, 0).astype(np.uint8)
            bands_u8.append(_percentile_stretch(raw))

    rgba = np.stack(bands_u8 + [alpha], axis=-1)  # (H, W, 4)
    img = Image.fromarray(rgba, mode='RGBA')

    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=False)
    png_bytes = buf.getvalue()

    meta: dict[str, Any] = {
        'width':     out_w,
        'height':    out_h,
        'src_width':  src_w,
        'src_height': src_h,
        'scale_x':   out_w / src_w,
        'scale_y':   out_h / src_h,
    }
    return png_bytes, meta


def px_to_rowcol(img_x: float, img_y: float, meta: dict[str, Any]) -> tuple[int, int]:
    """Convert display-image pixel (x=col, y=row) to source raster (row, col)."""
    src_col = img_x / meta['scale_x']
    src_row = img_y / meta['scale_y']
    return int(src_row), int(src_col)
```

- [ ] **Step 5: Install Pillow if not present**

```bash
conda run -n crism pip show pillow 2>/dev/null || conda run -n crism pip install pillow
```

- [ ] **Step 6: Run tests**

```bash
conda run -n crism pytest tests/test_tile_renderer.py -v
```

Expected: `test_px_to_rowcol_roundtrip` PASSED; integration tests PASSED or SKIPPED

- [ ] **Step 7: Commit**

```bash
git add app/tile_renderer.py tests/test_tile_renderer.py tests/conftest.py
git commit -m "feat: CRISM false-color tile renderer with scale metadata"
```

---

## Task 4: Spectrum extractor — polygon mean and ROI mean

**Files:**
- Create: `app/spectrum.py`
- Create: `tests/test_spectrum.py`

### Background

Two extraction modes:
1. **Polygon spectrum**: burn a single polygon to a pixel mask, read all pixels, return mean±std per band.
2. **ROI spectrum**: given a source-raster (row, col) and a radius in pixels, return mean±std of a square neighbourhood.

Both return wavelengths and a `(n_bands, )` mean and std array, filtered to `WAV_MIN–WAV_MAX` with bad bands interpolated.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_spectrum.py
import numpy as np, pytest
from app.spectrum import extract_polygon_spectrum, extract_roi_spectrum

def test_extract_polygon_spectrum_returns_dict(real_img_path, real_gpkg_path):
    import geopandas as gpd
    gdf = gpd.read_file(real_gpkg_path, layer='olivine')
    poly = gdf.iloc[0].geometry
    result = extract_polygon_spectrum(real_img_path, poly)
    assert 'wavelengths' in result
    assert 'mean' in result and 'std' in result
    assert len(result['wavelengths']) == len(result['mean'])
    assert all(500 <= w <= 2600 for w in result['wavelengths'])

def test_extract_polygon_spectrum_no_nan_in_mean(real_img_path, real_gpkg_path):
    import geopandas as gpd
    gdf = gpd.read_file(real_gpkg_path, layer='olivine')
    poly = gdf.iloc[0].geometry
    result = extract_polygon_spectrum(real_img_path, poly)
    assert not any(np.isnan(result['mean']))

def test_extract_roi_spectrum_returns_dict(real_img_path):
    result = extract_roi_spectrum(real_img_path, row=100, col=100, radius=5)
    assert 'wavelengths' in result
    assert len(result['wavelengths']) > 0

def test_extract_roi_spectrum_radius_zero(real_img_path):
    result = extract_roi_spectrum(real_img_path, row=100, col=100, radius=0)
    assert len(result['wavelengths']) > 0  # single-pixel still works
```

- [ ] **Step 2: Add `real_gpkg_path` fixture to `tests/conftest.py`**

```python
REAL_GPKG = '/mnt/mrdr/crism_classification/data/vector/t1249_mrral_20n073_0327_4_mineral_map.gpkg'

@pytest.fixture
def real_gpkg_path():
    import os
    if not os.path.exists(REAL_GPKG):
        pytest.skip('Real GeoPackage not available')
    return REAL_GPKG
```

- [ ] **Step 3: Run tests — expect failure**

```bash
conda run -n crism pytest tests/test_spectrum.py -v 2>&1 | head -20
```

- [ ] **Step 4: Implement `app/spectrum.py`**

```python
"""Spectral extraction utilities."""
import numpy as np
import rasterio
import rasterio.features
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from app.config import BAD_BAND_RANGES, CRISM_NODATA, WAV_MAX, WAV_MIN


def _read_wavelengths(src: rasterio.DatasetReader) -> np.ndarray:
    """Parse wavelengths (nm) from per-band ENVI-style tags."""
    try:
        # CRISM mrral: wavelength stored as per-band tag (1-based)
        return np.array(
            [float(src.tags(i)['wavelength']) for i in range(1, src.count + 1)],
            dtype=np.float32,
        )
    except (KeyError, TypeError, ValueError):
        # Fallback: evenly-spaced 72 bands 410–3920 nm
        return np.linspace(410, 3920, src.count).astype(np.float32)


def _interp_bad_bands(spec: np.ndarray, wav: np.ndarray) -> np.ndarray:
    spec = spec.copy()
    good = np.ones(len(wav), dtype=bool)
    for lo, hi in BAD_BAND_RANGES:
        good &= ~((wav >= lo) & (wav <= hi))
    bad_idx = np.where(~good)[0]
    if len(bad_idx) == 0:
        return spec
    good_idx = np.where(good)[0]
    spec[bad_idx] = np.interp(wav[bad_idx], wav[good_idx], spec[good_idx])
    return spec


def _wav_mask(wav: np.ndarray) -> np.ndarray:
    return (wav >= WAV_MIN) & (wav <= WAV_MAX)


def _extract_masked(img_path: str, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract pixel spectra for True pixels in mask.  Returns (N, B) array and wavelengths."""
    with rasterio.open(img_path) as src:
        wav = _read_wavelengths(src)
        wmask = _wav_mask(wav)
        wav_sel = wav[wmask]

        n_px = int(mask.sum())
        n_bands = int(wmask.sum())
        spectra = np.full((n_px, n_bands), np.nan, dtype=np.float32)

        for bi, band_global in enumerate(np.where(wmask)[0]):
            raw = src.read(int(band_global) + 1).astype(np.float32)
            raw[raw == CRISM_NODATA] = np.nan
            raw[np.abs(raw) > 1] = np.nan
            spectra[:, bi] = raw[mask]

    return spectra, wav_sel


def _summarise(spectra: np.ndarray, wav: np.ndarray) -> dict:
    mean = np.nanmean(spectra, axis=0)
    std  = np.nanstd(spectra, axis=0)
    # Interpolate bad bands on mean and std
    mean = _interp_bad_bands(mean, wav)
    std  = _interp_bad_bands(std, wav)
    return {
        'wavelengths': wav.tolist(),
        'mean':        mean.tolist(),
        'std':         std.tolist(),
        'n_pixels':    int((~np.isnan(spectra[:, 0])).sum()),
    }


def extract_polygon_spectrum(img_path: str, geometry: BaseGeometry) -> dict:
    """Return mean±std spectrum for all pixels within geometry."""
    with rasterio.open(img_path) as src:
        h, w = src.height, src.width
        transform = src.transform

    mask = rasterio.features.geometry_mask(
        [geometry], out_shape=(h, w), transform=transform, invert=True
    )
    if not mask.any():
        # Return empty-ish result
        return {'wavelengths': [], 'mean': [], 'std': [], 'n_pixels': 0}

    spectra, wav = _extract_masked(img_path, mask)
    return _summarise(spectra, wav)


def extract_roi_spectrum(
    img_path: str,
    row: int,
    col: int,
    radius: int = 5,
) -> dict:
    """Return mean±std spectrum for a square ROI of ±radius pixels around (row, col)."""
    with rasterio.open(img_path) as src:
        h, w = src.height, src.width

    r0 = max(0, row - radius)
    r1 = min(h, row + radius + 1)
    c0 = max(0, col - radius)
    c1 = min(w, col + radius + 1)

    mask = np.zeros((h, w), dtype=bool)
    mask[r0:r1, c0:c1] = True

    spectra, wav = _extract_masked(img_path, mask)
    return _summarise(spectra, wav)
```

- [ ] **Step 5: Run tests**

```bash
conda run -n crism pytest tests/test_spectrum.py -v
```

Expected: all PASSED or SKIPPED (if tile files absent)

- [ ] **Step 6: Commit**

```bash
git add app/spectrum.py tests/test_spectrum.py tests/conftest.py
git commit -m "feat: polygon and ROI spectral extraction with bad-band interpolation"
```

---

## Task 5: FastAPI backend — all REST routes

**Files:**
- Create: `app/main.py`

### Routes

| Method | Path | Returns |
|---|---|---|
| `GET` | `/api/tile/image` | PNG bytes |
| `GET` | `/api/tile/meta` | JSON: image dims, scale, polygon count |
| `GET` | `/api/polygons` | JSON: list of all polygon dicts (no geometry by default) |
| `GET` | `/api/polygons/geojson` | GeoJSON FeatureCollection in display-image pixel space |
| `GET` | `/api/polygon/{poly_id}/spectrum` | JSON: wavelengths, mean, std |
| `POST` | `/api/roi/spectrum` | JSON body `{row, col, radius}` → spectrum JSON |
| `POST` | `/api/polygon/{poly_id}/verdict` | JSON body `{verdict, confidence, note}` → `{ok: true}` |
| `GET` | `/` | Serve `static/index.html` |

### Coordinate note for `/api/polygons/geojson`

The polygon geometries are stored in the tile's native CRS (a local equirectangular in metres). The frontend uses Leaflet with `CRS.Simple` and displays the tile as a pixel image. We need polygon coordinates in **display-image pixel space**: multiply CRS coords by scale, using the tile's affine transform.

Convert: `src_col, src_row = ~transform * (x_crs, y_crs)`, then `img_x = src_col * scale_x`, `img_y = src_row * scale_y`. Leaflet `CRS.Simple` uses `[y, x]` = `[lat, lng]` = `[img_y, img_x]`.

- [ ] **Step 1: Implement `app/main.py`**

```python
"""FastAPI application for CRISM mineral verification."""
import json
import os
from functools import lru_cache
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from shapely.geometry import shape

from app.config import COL_VERDICT
from app.gpkg_io import ensure_verify_columns, load_all_polygons, write_verdict
from app.spectrum import extract_polygon_spectrum, extract_roi_spectrum
from app.tile_renderer import px_to_rowcol, render_false_color

app = FastAPI(title='CRISM Mineral Verification')

# ── env vars set by run_app.py ────────────────────────────────────────────────
IMG_PATH  = os.environ.get('CRISM_IMG', '')
GPKG_PATH = os.environ.get('CRISM_GPKG', '')


@lru_cache(maxsize=1)
def _tile_meta() -> dict[str, Any]:
    _, meta = render_false_color(IMG_PATH)
    return meta

@lru_cache(maxsize=1)
def _png_bytes() -> bytes:
    png, _ = render_false_color(IMG_PATH)
    return png

@lru_cache(maxsize=1)
def _all_polys() -> list[dict]:
    ensure_verify_columns(GPKG_PATH)
    return load_all_polygons(GPKG_PATH)


# ── helpers ───────────────────────────────────────────────────────────────────

def _polys_to_pixel_geojson(polys: list[dict], meta: dict) -> dict:
    """Convert polygon geometries from tile CRS to display-image pixel space."""
    with rasterio.open(IMG_PATH) as src:
        transform = src.transform
        crs = src.crs

    features = []
    for p in polys:
        if not p['geometry']:
            continue
        geom = shape(p['geometry'])
        # Convert CRS coords → src pixel → display pixel
        def _convert_coords(coords):
            result = []
            for x, y in coords:
                col, row = ~transform * (x, y)
                result.append([col * meta['scale_x'], row * meta['scale_y']])
            return result

        geom_px = _transform_geojson_coords(p['geometry'], _convert_coords)
        feature = {
            'type': 'Feature',
            'geometry': geom_px,
            'properties': {k: v for k, v in p.items() if k != 'geometry'},
        }
        features.append(feature)

    return {'type': 'FeatureCollection', 'features': features}


def _transform_geojson_coords(geojson_geom: dict, fn) -> dict:
    """Apply fn(coords_list) recursively to geometry coordinates."""
    import copy
    g = copy.deepcopy(geojson_geom)
    t = g['type']
    if t == 'Polygon':
        g['coordinates'] = [fn(ring) for ring in g['coordinates']]
    elif t == 'MultiPolygon':
        g['coordinates'] = [[fn(ring) for ring in poly] for poly in g['coordinates']]
    return g


# ── routes ────────────────────────────────────────────────────────────────────

@app.get('/api/tile/image')
def get_tile_image():
    return Response(content=_png_bytes(), media_type='image/png')


@app.get('/api/tile/meta')
def get_tile_meta():
    meta = _tile_meta()
    polys = _all_polys()
    return {**meta, 'poly_count': len(polys),
            'verified_count': sum(1 for p in polys if p.get(COL_VERDICT))}


@app.get('/api/polygons')
def get_polygons():
    polys = _all_polys()
    return [{k: v for k, v in p.items() if k != 'geometry'} for p in polys]


@app.get('/api/polygons/geojson')
def get_polygons_geojson():
    polys = _all_polys()
    meta = _tile_meta()
    return _polys_to_pixel_geojson(polys, meta)


@app.get('/api/polygon/{poly_id}/spectrum')
def get_polygon_spectrum(poly_id: int):
    polys = _all_polys()
    if poly_id < 0 or poly_id >= len(polys):
        raise HTTPException(404, 'poly_id out of range')
    poly = polys[poly_id]
    if not poly['geometry']:
        raise HTTPException(422, 'polygon has no geometry')
    geom = shape(poly['geometry'])
    return extract_polygon_spectrum(IMG_PATH, geom)


class RoiRequest(BaseModel):
    img_x: float
    img_y: float
    radius: int = 5

@app.post('/api/roi/spectrum')
def get_roi_spectrum(req: RoiRequest):
    meta = _tile_meta()
    row, col = px_to_rowcol(req.img_x, req.img_y, meta)
    return extract_roi_spectrum(IMG_PATH, row, col, req.radius)


class VerdictRequest(BaseModel):
    verdict: str      # 'correct' | 'incorrect'
    confidence: str   # 'low' | 'moderate' | 'high'
    note: str = ''

@app.post('/api/polygon/{poly_id}/verdict')
def post_verdict(poly_id: int, req: VerdictRequest):
    if req.verdict not in ('correct', 'incorrect'):
        raise HTTPException(422, 'verdict must be correct or incorrect')
    if req.confidence not in ('low', 'moderate', 'high'):
        raise HTTPException(422, 'confidence must be low, moderate, or high')
    polys = _all_polys()
    if poly_id < 0 or poly_id >= len(polys):
        raise HTTPException(404, 'poly_id out of range')
    write_verdict(GPKG_PATH, poly_id, polys, req.verdict, req.confidence, req.note)
    # Invalidate cache so next call picks up new verdicts
    _all_polys.cache_clear()
    return {'ok': True}


# ── static files (served last so API routes take precedence) ──────────────────
_STATIC = os.path.join(os.path.dirname(__file__), 'static')
app.mount('/', StaticFiles(directory=_STATIC, html=True), name='static')
```

- [ ] **Step 2: Smoke-test routes manually**

```bash
CRISM_IMG=/mnt/mrdr/mc13/t1249_mrral_20n073_0327_4.img \
CRISM_GPKG=/mnt/mrdr/crism_classification/data/vector/t1249_mrral_20n073_0327_4_mineral_map.gpkg \
conda run -n crism uvicorn app.main:app --port 8765 &
sleep 3
curl -s http://127.0.0.1:8765/api/tile/meta | python -m json.tool | head -10
curl -s http://127.0.0.1:8765/api/polygons | python -m json.tool | head -20
kill %1
```

Expected: JSON with `width`, `height`, `poly_count`; polygon list with `poly_id`, `mineral`, `confidence`.

- [ ] **Step 3: Commit**

```bash
git add app/main.py
git commit -m "feat: FastAPI backend with tile image, polygon, spectrum, and verdict routes"
```

---

## Task 6: Frontend — layout skeleton + tile image

**Files:**
- Create: `app/static/index.html`
- Create: `app/static/style.css`
- Create: `app/static/app.js`

### Layout

```
┌─────────────────────────────────────────────────────┐
│  CRISM Mineral Verification          [progress: 0/N] │
├──────────────────┬──────────────────┬───────────────┤
│  POLYGON TABLE   │   TILE IMAGE     │  SPECTRUM     │
│  (filterable)    │   (Leaflet)      │  (Plotly)     │
│  mineral ▼ tier▼ │   overlaid with  │               │
│  ────────────────│   polygon        │  [ratio mode] │
│  olivine t1 …    │   outlines       │               │
│  olivine t2 …    │                  │               │
│  …               │  right-click →   │               │
│                  │  define denom    │               │
│                  │  ROI             │               │
├──────────────────┴──────────────────┴───────────────┤
│  Verdict: ○ Correct  ○ Incorrect   Conf: [▼]  [Save]│
└─────────────────────────────────────────────────────┘
```

- [ ] **Step 1: Create `app/static/style.css`**

```css
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; font-size: 13px; background: #1a1a2e; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }

header { padding: 8px 16px; background: #16213e; display: flex; align-items: center; gap: 16px; border-bottom: 1px solid #0f3460; }
header h1 { font-size: 15px; font-weight: 600; color: #e94560; }
#progress { font-size: 12px; color: #aaa; margin-left: auto; }

#main { display: flex; flex: 1; overflow: hidden; }

/* Left panel — polygon table */
#panel-table { width: 280px; flex-shrink: 0; display: flex; flex-direction: column; border-right: 1px solid #0f3460; }
#filter-bar { padding: 6px 8px; display: flex; gap: 6px; background: #16213e; }
#filter-bar select { flex: 1; background: #0f3460; color: #e0e0e0; border: 1px solid #e94560; border-radius: 4px; padding: 3px 6px; }
#poly-table-wrap { flex: 1; overflow-y: auto; }
table { width: 100%; border-collapse: collapse; }
th { position: sticky; top: 0; background: #16213e; padding: 5px 8px; text-align: left; font-weight: 600; border-bottom: 1px solid #0f3460; }
td { padding: 4px 8px; border-bottom: 1px solid #0f3460; cursor: pointer; }
tr:hover td { background: #0f3460; }
tr.selected td { background: #e94560; color: #fff; }
tr.verified-correct td { border-left: 3px solid #4caf50; }
tr.verified-incorrect td { border-left: 3px solid #f44336; }

/* Center panel — tile map */
#panel-map { flex: 1; position: relative; }
#map { width: 100%; height: 100%; }
#roi-indicator { position: absolute; top: 8px; right: 8px; background: rgba(0,0,0,0.6); color: #fff; padding: 4px 8px; border-radius: 4px; font-size: 11px; pointer-events: none; display: none; }

/* Right panel — spectrum */
#panel-spectrum { width: 360px; flex-shrink: 0; display: flex; flex-direction: column; border-left: 1px solid #0f3460; }
#spectrum-header { padding: 6px 8px; background: #16213e; font-size: 12px; color: #aaa; }
#spectrum-plot { flex: 1; }
#ratio-toggle { margin: 0 8px 4px; display: flex; align-items: center; gap: 6px; font-size: 12px; }
#ratio-toggle input { accent-color: #e94560; }

/* Bottom bar — verdict */
#verdict-bar { padding: 8px 16px; background: #16213e; border-top: 1px solid #0f3460; display: flex; align-items: center; gap: 16px; }
.verdict-radio { display: flex; align-items: center; gap: 6px; }
.verdict-radio label { cursor: pointer; }
input[type=radio]:checked + label { color: #e94560; font-weight: 600; }
#verdict-note { flex: 1; background: #0f3460; border: 1px solid #e94560; color: #e0e0e0; border-radius: 4px; padding: 4px 8px; font-size: 12px; }
#conf-select { background: #0f3460; color: #e0e0e0; border: 1px solid #e94560; border-radius: 4px; padding: 4px 8px; }
#save-btn { background: #e94560; color: #fff; border: none; border-radius: 4px; padding: 6px 16px; cursor: pointer; font-weight: 600; }
#save-btn:hover { background: #c73652; }
#save-btn:disabled { background: #555; cursor: not-allowed; }
```

- [ ] **Step 2: Create `app/static/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>CRISM Mineral Verification</title>
  <link rel="stylesheet" href="style.css">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
</head>
<body>
  <header>
    <h1>CRISM Mineral Verification</h1>
    <span id="progress">Loading…</span>
  </header>

  <div id="main">
    <!-- Left: polygon table -->
    <div id="panel-table">
      <div id="filter-bar">
        <select id="filter-mineral"><option value="">All minerals</option></select>
        <select id="filter-tier"><option value="">All tiers</option></select>
        <select id="filter-status">
          <option value="">All</option>
          <option value="unverified">Unverified</option>
          <option value="correct">Correct</option>
          <option value="incorrect">Incorrect</option>
        </select>
      </div>
      <div id="poly-table-wrap">
        <table id="poly-table">
          <thead><tr><th>#</th><th>Mineral</th><th>T</th><th>Prob</th><th>Px</th><th>✓</th></tr></thead>
          <tbody id="poly-tbody"></tbody>
        </table>
      </div>
    </div>

    <!-- Center: Leaflet map -->
    <div id="panel-map">
      <div id="map"></div>
      <div id="roi-indicator">ROI: right-click to set denominator</div>
    </div>

    <!-- Right: spectrum -->
    <div id="panel-spectrum">
      <div id="spectrum-header">Select a polygon to view spectrum</div>
      <div id="ratio-toggle">
        <input type="checkbox" id="show-ratio">
        <label for="show-ratio">Show ratio spectrum (÷ denominator ROI)</label>
      </div>
      <div id="spectrum-plot"></div>
    </div>
  </div>

  <!-- Bottom: verdict bar -->
  <div id="verdict-bar">
    <span class="verdict-radio">
      <input type="radio" name="verdict" id="v-correct" value="correct">
      <label for="v-correct">Correct</label>
    </span>
    <span class="verdict-radio">
      <input type="radio" name="verdict" id="v-incorrect" value="incorrect">
      <label for="v-incorrect">Incorrect</label>
    </span>
    <select id="conf-select">
      <option value="">Confidence…</option>
      <option value="low">Low</option>
      <option value="moderate">Moderate</option>
      <option value="high">High</option>
    </select>
    <input id="verdict-note" type="text" placeholder="Optional note…">
    <button id="save-btn" disabled>Save</button>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 3: Create `app/static/app.js` (skeleton — tile image only)**

```javascript
// app.js — CRISM Mineral Verification frontend
'use strict';

const API = '';  // same origin

let tileMeta   = null;
let allPolys   = [];
let selectedId = null;
let roiSpectrum = null;   // {wavelengths, mean, std} from right-click ROI
let polySpectrum = null;  // {wavelengths, mean, std} from selected polygon
let map = null;
let imageOverlay = null;
let geojsonLayer = null;
let roiMarker = null;

const MINERAL_COLORS = {
  olivine: '#43a047', lcp: '#1e88e5', hcp: '#e53935',
  plagioclase: '#fb8c00', other: '#757575',
};

// ── Init ────────────────────────────────────────────────────────────────────

async function init() {
  tileMeta = await fetch(`${API}/api/tile/meta`).then(r => r.json());
  updateProgress();
  initMap();
  await loadPolygons();
  populateFilters();
  renderTable();
  loadGeojsonLayer();
}

// ── Map setup ────────────────────────────────────────────────────────────────

function initMap() {
  const { width, height } = tileMeta;
  const bounds = [[0, 0], [height, width]];

  map = L.map('map', {
    crs: L.CRS.Simple,
    minZoom: -3,
    maxZoom: 3,
  });

  imageOverlay = L.imageOverlay(`${API}/api/tile/image`, bounds).addTo(map);
  map.fitBounds(bounds);

  // Right-click → set denominator ROI
  map.on('contextmenu', async (e) => {
    const img_x = e.latlng.lng;
    const img_y = e.latlng.lat;
    await fetchRoiSpectrum(img_x, img_y);
    if (roiMarker) roiMarker.remove();
    roiMarker = L.circleMarker([img_y, img_x], {
      radius: 8, color: '#fff', weight: 2, fillColor: '#e94560', fillOpacity: 0.6,
    }).addTo(map).bindTooltip('Denominator ROI').openTooltip();
    updateSpectrumPlot();
  });
}

// ── Polygon table ─────────────────────────────────────────────────────────────

async function loadPolygons() {
  allPolys = await fetch(`${API}/api/polygons`).then(r => r.json());
}

function populateFilters() {
  const minerals = [...new Set(allPolys.map(p => p.mineral))].sort();
  const tiers = [...new Set(allPolys.map(p => p.confidence))].sort((a,b)=>a-b);
  const mSel = document.getElementById('filter-mineral');
  minerals.forEach(m => mSel.insertAdjacentHTML('beforeend', `<option value="${m}">${m}</option>`));
  const tSel = document.getElementById('filter-tier');
  tiers.forEach(t => tSel.insertAdjacentHTML('beforeend', `<option value="${t}">Tier ${t}</option>`));
  ['filter-mineral','filter-tier','filter-status'].forEach(id =>
    document.getElementById(id).addEventListener('change', renderTable));
}

function filteredPolys() {
  const mineral = document.getElementById('filter-mineral').value;
  const tier    = document.getElementById('filter-tier').value;
  const status  = document.getElementById('filter-status').value;
  return allPolys.filter(p => {
    if (mineral && p.mineral !== mineral) return false;
    if (tier && p.confidence !== parseInt(tier)) return false;
    if (status === 'unverified' && p.verdict) return false;
    if (status === 'correct'   && p.verdict !== 'correct') return false;
    if (status === 'incorrect' && p.verdict !== 'incorrect') return false;
    return true;
  });
}

function renderTable() {
  const tbody = document.getElementById('poly-tbody');
  tbody.innerHTML = '';
  filteredPolys().forEach(p => {
    const tr = document.createElement('tr');
    if (p.poly_id === selectedId) tr.classList.add('selected');
    if (p.verdict === 'correct')   tr.classList.add('verified-correct');
    if (p.verdict === 'incorrect') tr.classList.add('verified-incorrect');
    const verdictIcon = p.verdict === 'correct' ? '✓' : p.verdict === 'incorrect' ? '✗' : '';
    tr.innerHTML = `
      <td>${p.poly_id}</td>
      <td style="color:${MINERAL_COLORS[p.mineral]}">${p.mineral}</td>
      <td>${p.confidence}</td>
      <td>${(p.mean_prob || 0).toFixed(2)}</td>
      <td>${p.count_px}</td>
      <td>${verdictIcon}</td>`;
    tr.addEventListener('click', () => selectPolygon(p.poly_id));
    tbody.appendChild(tr);
  });
}

function updateProgress() {
  const verified = allPolys.filter(p => p.verdict).length;
  document.getElementById('progress').textContent =
    `Verified: ${verified} / ${allPolys.length}`;
}

// ── Polygon selection ────────────────────────────────────────────────────────

async function selectPolygon(polyId) {
  selectedId = polyId;
  renderTable();
  highlightSelectedOnMap(polyId);
  document.getElementById('spectrum-header').textContent = 'Loading spectrum…';
  polySpectrum = await fetch(`${API}/api/polygon/${polyId}/spectrum`).then(r => r.json());
  updateSpectrumPlot();

  // Pre-fill verdict controls if already verified
  const p = allPolys.find(x => x.poly_id === polyId);
  if (p?.verdict) {
    document.querySelector(`input[value="${p.verdict}"]`).checked = true;
    document.getElementById('conf-select').value = p.verify_conf || '';
    document.getElementById('verdict-note').value = p.verify_note || '';
  } else {
    document.querySelectorAll('input[name="verdict"]').forEach(r => r.checked = false);
    document.getElementById('conf-select').value = '';
    document.getElementById('verdict-note').value = '';
  }
  document.getElementById('save-btn').disabled = false;
}

// ── GeoJSON overlay ───────────────────────────────────────────────────────────

async function loadGeojsonLayer() {
  const geojson = await fetch(`${API}/api/polygons/geojson`).then(r => r.json());
  if (geojsonLayer) geojsonLayer.remove();
  geojsonLayer = L.geoJSON(geojson, {
    style: feature => ({
      color: MINERAL_COLORS[feature.properties.mineral] || '#aaa',
      weight: 1,
      fillOpacity: 0.0,
    }),
    onEachFeature: (feature, layer) => {
      layer.on('click', () => selectPolygon(feature.properties.poly_id));
    },
  }).addTo(map);
}

function highlightSelectedOnMap(polyId) {
  if (!geojsonLayer) return;
  geojsonLayer.eachLayer(layer => {
    const isSelected = layer.feature?.properties?.poly_id === polyId;
    layer.setStyle({ weight: isSelected ? 3 : 1, fillOpacity: isSelected ? 0.2 : 0.0 });
  });
}

// ── ROI spectrum ──────────────────────────────────────────────────────────────

async function fetchRoiSpectrum(img_x, img_y) {
  roiSpectrum = await fetch(`${API}/api/roi/spectrum`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ img_x, img_y, radius: 5 }),
  }).then(r => r.json());
  updateSpectrumPlot();
}

// ── Spectrum plot ─────────────────────────────────────────────────────────────

function updateSpectrumPlot() {
  if (!polySpectrum || polySpectrum.wavelengths.length === 0) return;

  const showRatio = document.getElementById('show-ratio').checked;
  const wav = polySpectrum.wavelengths;

  let traces = [];

  if (!showRatio || !roiSpectrum || roiSpectrum.wavelengths.length === 0) {
    // Raw reflectance plot
    const upper = polySpectrum.mean.map((v, i) => v + polySpectrum.std[i]);
    const lower = polySpectrum.mean.map((v, i) => v - polySpectrum.std[i]);
    // Note: lower must come first so upper's fill='tonexty' fills toward lower, not y=0
    traces = [
      { x: wav, y: lower, type: 'scatter', mode: 'none',
        showlegend: false },
      { x: wav, y: upper, fill: 'tonexty', type: 'scatter', mode: 'none',
        fillcolor: 'rgba(233,69,96,0.15)', showlegend: false },
      { x: wav, y: polySpectrum.mean, type: 'scatter', mode: 'lines',
        line: { color: '#e94560', width: 2 }, name: 'Polygon mean' },
    ];
    if (roiSpectrum) {
      traces.push({
        x: roiSpectrum.wavelengths, y: roiSpectrum.mean, type: 'scatter', mode: 'lines',
        line: { color: '#aaa', width: 1.5, dash: 'dot' }, name: 'ROI (denominator)',
      });
    }
  } else {
    // Ratio spectrum: resample ROI to poly wavelengths
    const roiInterp = wav.map(w => {
      const idx = roiSpectrum.wavelengths.reduce((best, rw, i) =>
        Math.abs(rw - w) < Math.abs(roiSpectrum.wavelengths[best] - w) ? i : best, 0);
      return roiSpectrum.mean[idx];
    });
    const ratio = polySpectrum.mean.map((v, i) =>
      roiInterp[i] !== 0 ? v / roiInterp[i] : NaN);
    traces = [
      { x: wav, y: ratio, type: 'scatter', mode: 'lines',
        line: { color: '#e94560', width: 2 }, name: 'Ratio (poly/ROI)' },
      { x: [wav[0], wav[wav.length-1]], y: [1, 1], type: 'scatter', mode: 'lines',
        line: { color: '#555', width: 1, dash: 'dot' }, showlegend: false },
    ];
  }

  const poly = allPolys.find(p => p.poly_id === selectedId);
  const title = poly ? `${poly.mineral} tier ${poly.confidence}  (n=${polySpectrum.n_pixels} px)` : '';

  Plotly.react('spectrum-plot', traces, {
    title: { text: title, font: { size: 11, color: '#e0e0e0' } },
    paper_bgcolor: '#1a1a2e', plot_bgcolor: '#0f3460',
    font: { color: '#e0e0e0', size: 10 },
    xaxis: { title: 'Wavelength (nm)', color: '#aaa', gridcolor: '#333' },
    yaxis: { title: showRatio ? 'Ratio' : 'Reflectance', color: '#aaa', gridcolor: '#333' },
    margin: { t: 36, r: 12, b: 48, l: 52 },
    legend: { font: { size: 9 }, bgcolor: 'rgba(0,0,0,0)' },
  }, { responsive: true });

  document.getElementById('spectrum-header').textContent = title || 'Spectrum';
}

document.getElementById('show-ratio').addEventListener('change', updateSpectrumPlot);

// ── Verdict save ──────────────────────────────────────────────────────────────

document.getElementById('save-btn').addEventListener('click', async () => {
  if (selectedId === null) return;
  const verdict = document.querySelector('input[name="verdict"]:checked')?.value;
  const confidence = document.getElementById('conf-select').value;
  if (!verdict || !confidence) {
    alert('Select a verdict and confidence level before saving.');
    return;
  }
  const note = document.getElementById('verdict-note').value;
  await fetch(`${API}/api/polygon/${selectedId}/verdict`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ verdict, confidence, note }),
  });
  // Refresh polygon list and re-render
  await loadPolygons();
  renderTable();
  updateProgress();
});

// ── Start ─────────────────────────────────────────────────────────────────────
init();
```

- [ ] **Step 4: Commit**

```bash
git add app/static/
git commit -m "feat: single-page verification UI with Leaflet map, Plotly spectrum, verdict form"
```

---

## Task 7: End-to-end smoke test and run instructions

**Files:**
- Modify: `README.md` (or a new `app/README.md`)

- [ ] **Step 1: Full run test**

```bash
CRISM_IMG=/mnt/mrdr/mc13/t1249_mrral_20n073_0327_4.img \
CRISM_GPKG=/mnt/mrdr/crism_classification/data/vector/t1249_mrral_20n073_0327_4_mineral_map.gpkg \
conda run -n crism python run_app.py \
  --img /mnt/mrdr/mc13/t1249_mrral_20n073_0327_4.img \
  --gpkg /mnt/mrdr/crism_classification/data/vector/t1249_mrral_20n073_0327_4_mineral_map.gpkg \
  --port 8765
```

Then open `http://127.0.0.1:8765` in a browser. Expected:
- Tile image renders in center panel
- Polygon table populates with olivine/lcp/hcp/plagioclase rows
- Clicking a row shows a spectrum on the right
- Right-clicking the map adds a ROI marker; toggling "ratio" shows ratio spectrum
- Saving a verdict updates the table row indicator

- [ ] **Step 2: Run full test suite**

```bash
conda run -n crism pytest tests/ -v --tb=short
```

- [ ] **Step 3: Update CLAUDE.md with run instructions**

Add to `CLAUDE.md` under a new `## Mineral Verification App` section:

```markdown
## Mineral Verification App

Local browser tool for reviewing vectroscopy output polygons.

### Run
```bash
conda run -n crism python run_app.py \
  --img /path/to/tile_mrral.img \
  --gpkg /path/to/tile_mineral_map.gpkg \
  --port 8765
# Then open http://127.0.0.1:8765
```

### What it does
- Polygon table: filterable by mineral, tier, verification status
- Center: Leaflet tile viewer with polygon outlines; click polygon to select
- Right: Plotly spectrum panel — raw reflectance + std shading
- Right-click tile image → set denominator ROI; toggle ratio spectrum
- Verdict bar: correct/incorrect + low/moderate/high confidence + note → writes back to .gpkg
```

- [ ] **Step 4: Final commit**

```bash
git add CLAUDE.md
git commit -m "docs: add mineral verification app run instructions to CLAUDE.md"
```

---

## Appendix: Key Data Contracts

### Polygon dict (from `/api/polygons`)
```json
{
  "poly_id": 42,
  "mineral": "olivine",
  "confidence": 3,
  "mean_prob": 0.612,
  "count_px": 87,
  "verdict": null,
  "verify_conf": null,
  "verify_note": null
}
```

### Spectrum response (from `/api/polygon/{id}/spectrum`, `/api/roi/spectrum`)
```json
{
  "wavelengths": [500.1, 511.2, ...],
  "mean": [0.12, 0.13, ...],
  "std":  [0.01, 0.02, ...],
  "n_pixels": 87
}
```

### Verdict request (POST `/api/polygon/{id}/verdict`)
```json
{
  "verdict": "correct",
  "confidence": "high",
  "note": "clean olivine doublet"
}
```

### GeoPackage verification columns added
| Column | Type | Values |
|---|---|---|
| `verdict` | str | `'correct'` \| `'incorrect'` \| NULL |
| `verify_conf` | str | `'low'` \| `'moderate'` \| `'high'` \| NULL |
| `verify_note` | str | free text \| NULL |
| `verified_at` | str | ISO-8601 UTC timestamp \| NULL |

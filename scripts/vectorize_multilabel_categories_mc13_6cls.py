"""
6-class vectorization for ft_6cls_mc11val_denoise_best.pt across all MC13 tiles.

Adds alteration as a 6th mineral class (5-bit bitmask, yellow in visualizations).
Source npz: /tmp/6cls_mc13/t*_probs.npz
Output:     data/vector_mc13_6cls/mc13_6cls_categories.gpkg

Usage:
    conda run -n crism python scripts/vectorize_multilabel_categories_mc13_6cls.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from itertools import combinations

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import scipy.ndimage
from affine import Affine
from pyproj import CRS
from shapely.geometry import shape as shapely_shape

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBS_DIR = '/tmp/6cls_mc13'
MC13_DIR  = '/mnt/mrdr/mc13'
OUT_PATH  = os.path.join(PROJ, 'data', 'vector_mc13_6cls',
                         'mc13_6cls_categories.gpkg')
WAVELENGTHS_SIDECAR = os.path.join(PROJ, 'data', 'vector_mc13_6cls',
                                   'mc13_6cls_wavelengths.json')

# Match classify_tile_supervised.py preprocessing
N_BANDS  = 59
NODATA   = 65535
CLIP_MAX = 0.5

# Mars 2000 geographic CRS (degrees lon/lat)
MARS_GEO_WKT = (
    'GEOGCS["GCS_Mars_2000",'
    'DATUM["D_Mars_2000",SPHEROID["Mars_2000_IAU_IAG",3396190,169.8944472]],'
    'PRIMEM["Reference_Meridian",0],'
    'UNIT["Degree",0.0174532925199433]]'
)
COMMON_CRS = CRS.from_wkt(MARS_GEO_WKT)

# Mineral order in saved probs (must match classify_tile_supervised.py 6-class order)
PROB_CHANNELS = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other', 'alteration']
# Which classes count toward the "category" string (drop 'other' — it's residual)
MINERAL_NAMES = ['olivine', 'lcp', 'hcp', 'plagioclase', 'alteration']

# Threshold scheme: max(val_AP-informed inverse-proportional, global floor)
# APs from ft_6cls_mc11val_denoise_best.pt; alteration from mc11val val split.
VAL_AP = {'olivine': 0.996, 'lcp': 0.895, 'hcp': 0.643,
           'plagioclase': 0.325, 'other': 1.000, 'alteration': 0.978}
GLOBAL_FLOOR = 0.80
INVERSE = {c: 0.99 - 0.49 * v for c, v in VAL_AP.items()}
THRESH  = {c: max(INVERSE[c], GLOBAL_FLOOR) for c in VAL_AP}

MEDIAN_SIZE  = 3
MEDIAN_ITERS = 1
MIN_PIXELS   = 9
SIMPLIFY_TOL_DEG = 200.0 / (3396190 * np.pi / 180)  # ~200 m in degrees


def discover_tiles() -> list[dict]:
    """Return sorted list of {tid, mrral} dicts for all tiles in MC13_DIR.

    Tile ID is the 't####' prefix extracted from the filename.
    Tiles whose _probs.npz is missing from PROBS_DIR are excluded (with warning).
    """
    pattern = os.path.join(MC13_DIR, 't*_mrral_*_0327_4.img')
    img_paths = sorted(glob.glob(pattern))
    if not img_paths:
        raise FileNotFoundError(f'No mrral tiles found at {pattern}')

    tiles = []
    skipped = []
    for img_path in img_paths:
        basename = os.path.basename(img_path)
        m = re.match(r'(t\d+)_mrral', basename)
        if not m:
            continue
        tid = m.group(1)
        npz = os.path.join(PROBS_DIR, f'{tid}_probs.npz')
        if not os.path.exists(npz):
            skipped.append(tid)
            continue
        tiles.append({'tid': tid, 'mrral': img_path})

    if skipped:
        print(f'WARNING: skipping {len(skipped)} tiles without probs.npz: '
              + ', '.join(skipped))
    print(f'Found {len(tiles)} tiles with probs.npz')
    return tiles


def category_string(bitmask: int) -> str:
    """Map a 4-bit mineral mask (bit i ↔ MINERAL_NAMES[i]) → human string."""
    if bitmask == 0:
        return 'unclassified'
    parts = [MINERAL_NAMES[i] for i in range(len(MINERAL_NAMES))
             if bitmask & (1 << i)]
    return ' + '.join(parts)


def vectorize_one_tile(tid: str, mrral_path: str) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame of category polygons for one tile."""
    npz_path = os.path.join(PROBS_DIR, f'{tid}_probs.npz')
    data = np.load(npz_path)
    probs      = data['probs'].astype(np.float32)      # (H, W, 5)
    valid_mask = data['valid_mask'].astype(bool)        # (H, W)
    src_transform = Affine(*[float(v) for v in data['transform']])
    src_crs    = CRS.from_wkt(str(data['crs_wkt']))

    H, W, C = probs.shape

    # Median-filter each prob channel (smooths speckle before thresholding)
    smooth = np.zeros_like(probs)
    for ci in range(C):
        ch = probs[:, :, ci].copy()
        for _ in range(MEDIAN_ITERS):
            ch = scipy.ndimage.median_filter(ch, size=MEDIAN_SIZE)
        smooth[:, :, ci] = ch

    # Per-mineral threshold pass → 4-bit bitmask per pixel
    bitmask = np.zeros((H, W), dtype=np.uint8)
    for bit_i, mineral in enumerate(MINERAL_NAMES):
        ci = PROB_CHANNELS.index(mineral)
        bitmask[(smooth[:, :, ci] >= THRESH[mineral]) & valid_mask] |= (1 << bit_i)

    # Polygonize: contiguous regions with identical bitmask value
    polygons = []
    for geom, value in rasterio.features.shapes(
        bitmask, mask=valid_mask, transform=src_transform, connectivity=4
    ):
        polygons.append((int(value), shapely_shape(geom)))

    _empty_cols = (
        ['tile_id', 'category', 'n_minerals']
        + [f'prob_{c}' for c in PROB_CHANNELS]
        + ['count_px', 'geometry']
    )
    if not polygons:
        return gpd.GeoDataFrame(columns=_empty_cols, crs=src_crs)

    cats, geoms = zip(*polygons)
    gdf = gpd.GeoDataFrame({'cat_bits': list(cats)},
                            geometry=list(geoms), crs=src_crs)

    # Drop tiny polygons below MIN_PIXELS
    pixel_area_m2 = abs(src_transform.a * src_transform.e)
    gdf['count_px'] = (gdf.geometry.area / pixel_area_m2).round().astype(int)
    gdf = gdf[gdf['count_px'] >= MIN_PIXELS].copy()

    if len(gdf) == 0:
        return gpd.GeoDataFrame(columns=_empty_cols, crs=src_crs)

    # Compute per-polygon mean probabilities via rasterized label IDs
    label_id_arr = np.zeros((H, W), dtype=np.int32)
    shapes = [(g, i + 1) for i, g in enumerate(gdf.geometry.values)]
    rasterio.features.rasterize(
        shapes=shapes, out=label_id_arr,
        transform=src_transform, fill=0,
    )
    n_polys = len(gdf)
    poly_idx = np.arange(1, n_polys + 1)

    polygon_means = {}
    for mineral in PROB_CHANNELS:
        ci = PROB_CHANNELS.index(mineral)
        means = scipy.ndimage.mean(probs[:, :, ci], labels=label_id_arr,
                                   index=poly_idx)
        polygon_means[f'prob_{mineral}'] = means.astype(np.float32)

    # Per-polygon mean reflectance spectrum (bands 1..N_BANDS from mrral)
    with rasterio.open(mrral_path) as src:
        mrral = src.read(list(range(1, N_BANDS + 1))).astype(np.float32)  # (59, H, W)
    nd_mask = (mrral == NODATA) | ~np.isfinite(mrral)
    mrral[nd_mask] = 0.0
    np.clip(mrral, 0.0, CLIP_MAX, out=mrral)

    band_means = np.zeros((n_polys, N_BANDS), dtype=np.float32)
    for b in range(N_BANDS):
        band_means[:, b] = scipy.ndimage.mean(
            mrral[b], labels=label_id_arr, index=poly_idx,
        )

    # Compose columns
    gdf['tile_id']   = tid
    gdf['category']  = gdf['cat_bits'].map(category_string)
    gdf['n_minerals'] = gdf['cat_bits'].apply(
        lambda x: bin(x).count('1') if x else 0
    )
    for k, v in polygon_means.items():
        gdf[k] = v
    for b in range(N_BANDS):
        gdf[f'band_{b:02d}'] = band_means[:, b]

    # Reproject to common geographic CRS and simplify slightly
    gdf = gdf.to_crs(COMMON_CRS)
    gdf['geometry'] = gdf.geometry.simplify(SIMPLIFY_TOL_DEG, preserve_topology=True)

    col_order = (
        ['tile_id', 'category', 'n_minerals']
        + [f'prob_{c}' for c in PROB_CHANNELS]
        + ['count_px']
        + [f'band_{b:02d}' for b in range(N_BANDS)]
        + ['geometry']
    )
    return gdf[col_order]


def read_wavelengths_nm(mrral_path: str) -> list:
    """Read the 59-band wavelength axis from the mrral tile's ENVI header."""
    import spectral.io.envi as envi
    hdr_path = mrral_path.replace('.img', '.hdr')
    hdr = envi.open(hdr_path)
    wl = [float(w) for w in hdr.metadata['wavelength']][:N_BANDS]
    return wl


def main():
    print('Thresholds applied (per class):')
    for c in PROB_CHANNELS:
        used = THRESH[c]
        inv  = INVERSE[c]
        gate = 'floor' if GLOBAL_FLOOR > inv else 'inverse'
        print(f'  {c:<12} val_AP={VAL_AP[c]:.3f}  inverse={inv:.3f}  '
              f'floor={GLOBAL_FLOOR:.2f}  → τ={used:.3f}  ({gate})')
    print()

    tiles = discover_tiles()
    if not tiles:
        print('No tiles to process — exiting.')
        sys.exit(1)

    frames = []
    for t in tiles:
        print(f'Vectorizing {t["tid"]} …')
        gdf = vectorize_one_tile(t['tid'], t['mrral'])
        print(f'  {len(gdf):,} polygons')
        frames.append(gdf)

    if not frames:
        print('No polygons generated — exiting.')
        sys.exit(1)

    merged = pd.concat(frames, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry='geometry', crs=COMMON_CRS)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    merged.to_file(OUT_PATH, layer='categories', driver='GPKG')
    print(f'\nWrote {len(merged):,} polygons → {OUT_PATH}')

    # Wavelength sidecar
    wl = read_wavelengths_nm(tiles[0]['mrral'])
    with open(WAVELENGTHS_SIDECAR, 'w') as f:
        json.dump({'wavelengths_nm': wl, 'n_bands': N_BANDS,
                   'column_prefix': 'band_',
                   'note': 'one entry per band_NN column; matches mrral band order'},
                  f, indent=2)
    print(f'Wrote wavelength sidecar → {WAVELENGTHS_SIDECAR}')

    # Category polygon counts (top 15 by total)
    print('\nCategory polygon counts (all tiles, top 15):')
    cat_counts = merged.groupby('category').size().sort_values(ascending=False)
    for cat, cnt in cat_counts.head(15).items():
        pct = 100 * cnt / len(merged)
        print(f'  {cat:<35}  {cnt:>8,d}  ({pct:.1f}%)')

    # Per-class valid-pixel coverage summary
    print('\nPer-class pixel coverage across MC13:')
    for mineral in MINERAL_NAMES:
        n_px = merged.loc[merged['category'].str.contains(mineral), 'count_px'].sum()
        print(f'  {mineral:<12}  {n_px:>12,d} px')

    # Tile summary
    print('\nPolygons per tile:')
    tile_counts = merged.groupby('tile_id').size().sort_values(ascending=False)
    for tid, cnt in tile_counts.items():
        print(f'  {tid}  {cnt:>6,d}')


if __name__ == '__main__':
    main()

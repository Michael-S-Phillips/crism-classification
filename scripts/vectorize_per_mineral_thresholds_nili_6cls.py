"""
6-class per-mineral, per-threshold vectorization for the 4 Nili Fossae tiles
(t1249, t1250, t1321, t1322) using probs from ft_6cls_mc11val_denoise_best.pt.

Produces 5 GeoPackages — one per mineral (olivine, lcp, hcp, plagioclase,
alteration) — each containing 8 threshold layers:
  [0.50, 0.60, 0.75, 0.85, 0.90, 0.95, 0.97, 0.99]

Output layout:
  data/vector_nili_6cls/
    olivine.gpkg
    lcp.gpkg
    hcp.gpkg
    plagioclase.gpkg
    alteration.gpkg

Source npz:  /tmp/6cls_mc13/t*_probs.npz  (from classify_tile_supervised.py, 6-class)

Usage:
    conda run -n crism python scripts/vectorize_per_mineral_thresholds_nili_6cls.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sqlite3
import sys

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

DEFAULT_PROBS_DIR = '/tmp/6cls_mc13'
DEFAULT_TILE_DIR  = '/mnt/mrdr/mc13'
DEFAULT_OUT_DIR   = os.path.join(PROJ, 'data', 'vector_nili_6cls')

NILI_TILES = ['t1249', 't1250', 't1321', 't1322']

# Filled in by main() from CLI args.
PROBS_DIR = DEFAULT_PROBS_DIR
TILE_DIR  = DEFAULT_TILE_DIR
OUT_DIR   = DEFAULT_OUT_DIR

N_BANDS  = 59
NODATA   = 65535
CLIP_MAX = 0.5

MARS_GEO_WKT = (
    'GEOGCS["GCS_Mars_2000",'
    'DATUM["D_Mars_2000",SPHEROID["Mars_2000_IAU_IAG",3396190,169.8944472]],'
    'PRIMEM["Reference_Meridian",0],'
    'UNIT["Degree",0.0174532925199433]]'
)
COMMON_CRS = CRS.from_wkt(MARS_GEO_WKT)

PROB_CHANNELS = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other', 'alteration']
MINERAL_NAMES = ['olivine', 'lcp', 'hcp', 'plagioclase', 'alteration']

UNIFORM_THRESHOLDS = [0.50, 0.60, 0.75, 0.85, 0.90, 0.95, 0.97, 0.99]
PER_MINERAL_THRESHOLDS = {m: list(UNIFORM_THRESHOLDS) for m in MINERAL_NAMES}

MEDIAN_SIZE  = 3
MEDIAN_ITERS = 1
MIN_PIXELS   = 9
SIMPLIFY_TOL_DEG = 200.0 / (3396190 * np.pi / 180)

MINERAL_BASE_RGB = {
    'olivine':     (1.00, 0.00, 0.00),
    'hcp':         (1.00, 0.00, 1.00),
    'lcp':         (0.00, 1.00, 1.00),
    'plagioclase': (1.00, 0.84, 0.00),
    'alteration':  (1.00, 0.89, 0.10),
}
N_LAYERS_PER_MINERAL = len(UNIFORM_THRESHOLDS)  # 8
MIN_SAT = 0.30
MAX_SAT = 1.00


def _check_npz_channels(data, expected_channels):
    if 'class_names' not in getattr(data, 'files', []):
        return
    names = [str(x) for x in data['class_names']]
    if names != list(expected_channels):
        raise SystemExit(
            f'npz class_names {names} != this script\'s channel order '
            f'{list(expected_channels)}.')


def discover_tiles() -> list[dict]:
    """Return sorted list of {tid, mrral} for the 4 Nili tiles with probs.npz."""
    tiles, missing = [], []
    for tid in NILI_TILES:
        hits = sorted(glob.glob(os.path.join(TILE_DIR, f'{tid}_mrral_*_0327_4.img')))
        if not hits:
            print(f'WARNING: no mrral img found for {tid}')
            continue
        mrral = hits[0]
        npz = os.path.join(PROBS_DIR, f'{tid}_probs.npz')
        if not os.path.exists(npz):
            missing.append(tid)
            continue
        tiles.append({'tid': tid, 'mrral': mrral})
    if missing:
        print(f'WARNING: skipping {len(missing)} tiles without probs.npz: '
              + ', '.join(missing))
    print(f'Found {len(tiles)} tiles with probs.npz')
    return tiles


def vectorize_one_tile_one_threshold(
    tid: str,
    smoothed_mineral_prob: np.ndarray,
    valid_mask: np.ndarray,
    src_transform: Affine,
    src_crs: CRS,
    mrral_cube: np.ndarray,
    mineral: str,
    threshold: float,
) -> gpd.GeoDataFrame:
    H, W = smoothed_mineral_prob.shape
    mask_bool = (smoothed_mineral_prob >= threshold) & valid_mask

    if not mask_bool.any():
        return gpd.GeoDataFrame(geometry=[], crs=src_crs)

    polygons = []
    for geom, value in rasterio.features.shapes(
        mask_bool.astype(np.uint8), mask=valid_mask,
        transform=src_transform, connectivity=4,
    ):
        if int(value) == 1:
            polygons.append(shapely_shape(geom))

    if not polygons:
        return gpd.GeoDataFrame(geometry=[], crs=src_crs)

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=src_crs)

    pixel_area_m2 = abs(src_transform.a * src_transform.e)
    gdf['count_px'] = (gdf.geometry.area / pixel_area_m2).round().astype(int)
    gdf = gdf[gdf['count_px'] >= MIN_PIXELS].copy()
    if len(gdf) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=src_crs)

    label_id_arr = np.zeros((H, W), dtype=np.int32)
    shapes = [(g, i + 1) for i, g in enumerate(gdf.geometry.values)]
    rasterio.features.rasterize(
        shapes=shapes, out=label_id_arr,
        transform=src_transform, fill=0,
    )
    n_polys = len(gdf)
    poly_idx = np.arange(1, n_polys + 1)

    mean_prob = scipy.ndimage.mean(
        smoothed_mineral_prob, labels=label_id_arr, index=poly_idx,
    ).astype(np.float32)

    band_means = np.zeros((n_polys, N_BANDS), dtype=np.float32)
    for b in range(N_BANDS):
        band_means[:, b] = scipy.ndimage.mean(
            mrral_cube[b], labels=label_id_arr, index=poly_idx,
        )

    gdf['tile_id']   = tid
    gdf['mineral']   = mineral
    gdf['threshold'] = threshold
    gdf['mean_prob'] = mean_prob
    for b in range(N_BANDS):
        gdf[f'band_{b:02d}'] = band_means[:, b]

    gdf = gdf.to_crs(COMMON_CRS)
    gdf['geometry'] = gdf.geometry.simplify(SIMPLIFY_TOL_DEG, preserve_topology=True)

    col_order = (
        ['tile_id', 'mineral', 'threshold', 'count_px', 'mean_prob']
        + [f'band_{b:02d}' for b in range(N_BANDS)]
        + ['geometry']
    )
    return gdf[col_order]


def process_tile_all_thresholds(
    tid: str,
    mrral_path: str,
) -> dict[str, dict[float, gpd.GeoDataFrame]]:
    npz_path = os.path.join(PROBS_DIR, f'{tid}_probs.npz')
    data = np.load(npz_path)
    _check_npz_channels(data, PROB_CHANNELS)
    probs      = data['probs'].astype(np.float32)
    valid_mask = data['valid_mask'].astype(bool)
    src_transform = Affine(*[float(v) for v in data['transform']])
    src_crs    = CRS.from_wkt(str(data['crs_wkt']))

    H, W, C = probs.shape

    smooth = np.zeros_like(probs)
    for ci in range(C):
        ch = probs[:, :, ci].copy()
        for _ in range(MEDIAN_ITERS):
            ch = scipy.ndimage.median_filter(ch, size=MEDIAN_SIZE)
        smooth[:, :, ci] = ch

    with rasterio.open(mrral_path) as src:
        mrral = src.read(list(range(1, N_BANDS + 1))).astype(np.float32)
    nd_mask = (mrral == NODATA) | ~np.isfinite(mrral)
    mrral[nd_mask] = 0.0
    np.clip(mrral, 0.0, CLIP_MAX, out=mrral)

    out: dict[str, dict[float, gpd.GeoDataFrame]] = {}
    for mineral in MINERAL_NAMES:
        ci = PROB_CHANNELS.index(mineral)
        smoothed_mineral = smooth[:, :, ci]
        out[mineral] = {}
        for thresh in PER_MINERAL_THRESHOLDS[mineral]:
            gdf = vectorize_one_tile_one_threshold(
                tid=tid,
                smoothed_mineral_prob=smoothed_mineral,
                valid_mask=valid_mask,
                src_transform=src_transform,
                src_crs=src_crs,
                mrral_cube=mrral,
                mineral=mineral,
                threshold=thresh,
            )
            out[mineral][thresh] = gdf
    return out


def _threshold_color_rgb255(mineral: str, layer_idx: int) -> tuple[int, int, int]:
    base = MINERAL_BASE_RGB[mineral]
    if N_LAYERS_PER_MINERAL <= 1:
        sat = MAX_SAT
    else:
        sat = MIN_SAT + (MAX_SAT - MIN_SAT) * layer_idx / (N_LAYERS_PER_MINERAL - 1)
    return tuple(int(round(c * sat * 255)) for c in base)


_QML_TEMPLATE = """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28" styleCategories="AllStyleCategories">
  <renderer-v2 forceraster="0" type="singleSymbol" enableorderby="0">
    <symbols>
      <symbol alpha="1" type="fill" name="0" clip_to_extent="1">
        <layer pass="0" class="SimpleFill" enabled="1" locked="0">
          <Option type="Map">
            <Option name="color" type="QString" value="{r},{g},{b},255"/>
            <Option name="outline_color" type="QString" value="35,35,35,0"/>
            <Option name="outline_style" type="QString" value="no"/>
            <Option name="outline_width" type="QString" value="0"/>
            <Option name="style" type="QString" value="solid"/>
            <Option name="joinstyle" type="QString" value="bevel"/>
          </Option>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
</qgis>"""


def add_qgis_layer_styles(gpkg_path: str, layer_colors: dict[str, tuple[int, int, int]]):
    conn = sqlite3.connect(gpkg_path)
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS layer_styles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                f_table_catalog TEXT(256),
                f_table_schema TEXT(256),
                f_table_name TEXT(256),
                f_geometry_column TEXT(256),
                styleName TEXT(30),
                styleQML TEXT,
                styleSLD TEXT,
                useAsDefault BOOLEAN,
                description TEXT,
                owner TEXT(30),
                ui TEXT(30),
                update_time TIMESTAMP DEFAULT (datetime('now'))
            )
        """)
        cur.execute(
            "DELETE FROM layer_styles WHERE f_table_name IN ({})".format(
                ','.join('?' * len(layer_colors))
            ),
            list(layer_colors.keys()),
        )
        for layer_name, (r, g, b) in layer_colors.items():
            qml = _QML_TEMPLATE.format(r=r, g=g, b=b)
            cur.execute(
                """INSERT INTO layer_styles
                   (f_table_catalog, f_table_schema, f_table_name,
                    f_geometry_column, styleName, styleQML,
                    useAsDefault, description, owner)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ('', '', layer_name, 'geom', layer_name, qml, 1,
                 f'Per-mineral threshold ladder color for {layer_name}', ''),
            )
        conn.commit()
    finally:
        conn.close()


def read_wavelengths_nm(mrral_path: str) -> list:
    import spectral.io.envi as envi
    hdr_path = mrral_path.replace('.img', '.hdr')
    hdr = envi.open(hdr_path)
    return [float(w) for w in hdr.metadata['wavelength']][:N_BANDS]


def main():
    global PROBS_DIR, TILE_DIR, OUT_DIR

    parser = argparse.ArgumentParser(
        description='Per-mineral, per-threshold vectorization for the 4 Nili '
                    'Fossae tiles (t1249, t1250, t1321, t1322), 8 thresholds '
                    'per mineral [0.50, 0.60, 0.75, 0.85, 0.90, 0.95, 0.97, 0.99].')
    parser.add_argument('--probs_dir', default=DEFAULT_PROBS_DIR,
                        help=f'Directory containing t*_probs.npz '
                             f'(default: {DEFAULT_PROBS_DIR})')
    parser.add_argument('--tile_dir', default=DEFAULT_TILE_DIR,
                        help=f'Directory containing t*_mrral_*.img tiles '
                             f'(default: {DEFAULT_TILE_DIR})')
    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR,
                        help=f'Output directory for per-mineral GPKGs '
                             f'(default: {DEFAULT_OUT_DIR})')
    args = parser.parse_args()

    PROBS_DIR = args.probs_dir
    TILE_DIR  = args.tile_dir
    OUT_DIR   = args.out_dir
    wavelengths_sidecar = os.path.join(OUT_DIR, 'vector_nili_6cls_wavelengths.json')

    print(f'probs_dir: {PROBS_DIR}')
    print(f'tile_dir:  {TILE_DIR}')
    print(f'out_dir:   {OUT_DIR}')
    print(f'tiles:     {NILI_TILES}')
    print()
    print(f'Threshold grid (all minerals): '
          + ', '.join(f'{t:.2f}' for t in UNIFORM_THRESHOLDS))
    print()

    tiles = discover_tiles()
    if not tiles:
        print('No tiles to process — exiting.')
        sys.exit(1)

    accum: dict[str, dict[float, list[gpd.GeoDataFrame]]] = {
        m: {t: [] for t in PER_MINERAL_THRESHOLDS[m]} for m in MINERAL_NAMES
    }

    for i, t in enumerate(tiles, start=1):
        print(f'[{i}/{len(tiles)}] Vectorizing {t["tid"]} …')
        tile_results = process_tile_all_thresholds(t['tid'], t['mrral'])
        total = 0
        for mineral in MINERAL_NAMES:
            for thresh in PER_MINERAL_THRESHOLDS[mineral]:
                gdf = tile_results[mineral][thresh]
                if len(gdf) > 0:
                    accum[mineral][thresh].append(gdf)
                    total += len(gdf)
        print(f'  {total:,} polygons across {len(MINERAL_NAMES)}×{N_LAYERS_PER_MINERAL} cells')

    os.makedirs(OUT_DIR, exist_ok=True)

    for mineral in MINERAL_NAMES:
        out_path = os.path.join(OUT_DIR, f'{mineral}.gpkg')
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except PermissionError:
                with open(out_path, 'wb'):
                    pass
        print(f'\nWriting {out_path} …')
        per_mineral_total = 0
        written_layer_colors: dict[str, tuple[int, int, int]] = {}
        for layer_idx, thresh in enumerate(PER_MINERAL_THRESHOLDS[mineral]):
            frames = accum[mineral][thresh]
            if not frames:
                print(f'  thresh_{thresh:.2f}: 0 polygons (skipping layer)')
                continue
            merged = pd.concat(frames, ignore_index=True)
            merged = gpd.GeoDataFrame(merged, geometry='geometry', crs=COMMON_CRS)
            layer_name = f'thresh_{thresh:.2f}'
            merged.to_file(out_path, layer=layer_name, driver='GPKG')
            per_mineral_total += len(merged)
            written_layer_colors[layer_name] = _threshold_color_rgb255(mineral, layer_idx)
            print(f'  {layer_name}: {len(merged):,} polygons')
        if written_layer_colors:
            add_qgis_layer_styles(out_path, written_layer_colors)
            print(f'  embedded QGIS layer_styles for {len(written_layer_colors)} layers')
        print(f'  → {per_mineral_total:,} polygons total in {os.path.basename(out_path)}')

    wl = read_wavelengths_nm(tiles[0]['mrral'])
    with open(wavelengths_sidecar, 'w') as f:
        json.dump({'wavelengths_nm': wl, 'n_bands': N_BANDS,
                   'column_prefix': 'band_',
                   'note': 'one entry per band_NN column; matches mrral band order'},
                  f, indent=2)
    print(f'\nWrote wavelength sidecar → {wavelengths_sidecar}')

    print('\nPer-mineral × threshold polygon counts:')
    hdr = '  ' + f'{"mineral":<14}' + ''.join(f' {t:.2f}' for t in UNIFORM_THRESHOLDS)
    print(hdr)
    for mineral in MINERAL_NAMES:
        row = f'  {mineral:<14}'
        for thresh in PER_MINERAL_THRESHOLDS[mineral]:
            n = sum(len(g) for g in accum[mineral][thresh])
            row += f' {n:>6,}'
        print(row)


if __name__ == '__main__':
    main()

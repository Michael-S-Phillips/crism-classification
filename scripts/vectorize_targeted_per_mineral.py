"""
Per-mineral, per-threshold vectorization for a single CRISM TARGETED
observation classified by scripts/classify_targeted_observation.py.

Mirrors the MC13 product (vectorize_per_mineral_thresholds_mc13.py):
  - Output: one GPKG per mineral (olivine/lcp/hcp/plagioclase), each with up
    to 5 nested-threshold layers (`thresh_X.XX`).
  - Per-polygon columns: obs_id, mineral, threshold, count_px, mean_prob,
    band_00 … band_58 (mean reflectance), geometry.
  - Same morphology: median filter (size=3), MIN_PIXELS, simplify.
  - Embeds QGIS layer_styles so the colors match the per_mineral_mc13
    figures (olivine red, hcp magenta, lcp cyan, plagioclase gold;
    saturation ramps 0.4 → 1.0 from lowest to highest threshold).

Default threshold grid matches the one currently used for MC13:
  olivine:     [0.80, 0.85, 0.90, 0.95, 0.97]
  lcp:         [0.85, 0.90, 0.93, 0.95, 0.97]
  hcp:         [0.85, 0.90, 0.93, 0.95, 0.97]
  plagioclase: [0.92, 0.94, 0.96, 0.97, 0.98]

Usage:
    conda run -n crism python scripts/vectorize_targeted_per_mineral.py \\
        --probs   /tmp/frt00009d44_probs.npz \\
        --tile    /path/to/frt00009d44_07_if165j_mtr3.img \\
        --out_dir data/vector_targeted/frt00009d44 \\
        --downsample 10        # must match what classify_targeted used

Spec note: the .npz transform reflects the *post-downsample* grid; pass the
same --downsample as the classify_targeted run so spectra are computed on
the same grid as the probs.
"""
from __future__ import annotations

import argparse
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
sys.path.insert(0, PROJ)

# Reuse the spectral-resampling helpers so the band axis matches mrral exactly
from scripts.classify_targeted_observation import (
    get_mrdr_wavelengths,
    read_targeted_wavelengths,
    find_band_mapping,
    load_targeted,
    N_BANDS,
)

PROB_CHANNELS = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']
MINERAL_NAMES = ['olivine', 'lcp', 'hcp', 'plagioclase']

def _check_npz_channels(data, expected_channels):
    """Fail loudly if the npz was produced by a checkpoint whose class list
    doesn't match this script's channel constants (e.g. a 6-class alteration
    model feeding a 5-class vectorizer would silently drop/mislabel layers)."""
    if 'class_names' not in getattr(data, 'files', []):
        return  # legacy 5-class npz, no metadata — assume constants are right
    names = [str(x) for x in data['class_names']]
    if names != list(expected_channels):
        raise SystemExit(
            f'npz class_names {names} != this script\'s channel order '
            f'{list(expected_channels)}. This script needs updating for that '
            f'checkpoint\'s class list (6-class alteration support pending).')


DEFAULT_THRESHOLDS = {
    'olivine':     [0.80, 0.85, 0.90, 0.95, 0.97],
    'lcp':         [0.85, 0.90, 0.93, 0.95, 0.97],
    'hcp':         [0.85, 0.90, 0.93, 0.95, 0.97],
    'plagioclase': [0.92, 0.94, 0.96, 0.97, 0.98],
}

# Per-mineral base color (matches plot_per_mineral_thresholds.py)
MINERAL_BASE_RGB = {
    'olivine':     (1.00, 0.00, 0.00),   # red
    'hcp':         (1.00, 0.00, 1.00),   # magenta
    'lcp':         (0.00, 1.00, 1.00),   # cyan
    'plagioclase': (1.00, 0.84, 0.00),   # gold
}
N_LAYERS_PER_MINERAL = 5
MIN_SAT = 0.40
MAX_SAT = 1.00

# Mars 2000 geographic CRS — same as MC13 product
MARS_GEO_WKT = (
    'GEOGCS["GCS_Mars_2000",'
    'DATUM["D_Mars_2000",SPHEROID["Mars_2000_IAU_IAG",3396190,169.8944472]],'
    'PRIMEM["Reference_Meridian",0],'
    'UNIT["Degree",0.0174532925199433]]'
)
COMMON_CRS = CRS.from_wkt(MARS_GEO_WKT)

MEDIAN_SIZE  = 3
MEDIAN_ITERS = 1
MIN_PIXELS   = 9
SIMPLIFY_TOL_DEG = 200.0 / (3396190 * np.pi / 180)


# ── QGIS styles (copied from vectorize_per_mineral_thresholds_mc13.py) ───────

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


def add_qgis_layer_styles(gpkg_path: str,
                           layer_colors: dict[str, tuple[int, int, int]]):
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


# ── vectorize one (mineral, threshold) cell for the single observation ───────

def vectorize_one_cell(
    obs_id: str,
    smoothed_mineral_prob: np.ndarray,
    valid_mask: np.ndarray,
    src_transform: Affine,
    src_crs: CRS,
    cube_hwc: np.ndarray,                # (H, W, 59) — mean spectrum source
    mineral: str,
    threshold: float,
) -> gpd.GeoDataFrame:
    H, W = smoothed_mineral_prob.shape
    pass_mask = (smoothed_mineral_prob >= threshold) & valid_mask
    if not pass_mask.any():
        return gpd.GeoDataFrame(geometry=[], crs=src_crs)

    polygons = []
    for geom, value in rasterio.features.shapes(
        pass_mask.astype(np.uint8), mask=valid_mask,
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
            cube_hwc[:, :, b], labels=label_id_arr, index=poly_idx,
        )

    gdf['obs_id']    = obs_id
    gdf['mineral']   = mineral
    gdf['threshold'] = threshold
    gdf['mean_prob'] = mean_prob
    for b in range(N_BANDS):
        gdf[f'band_{b:02d}'] = band_means[:, b]

    gdf = gdf.to_crs(COMMON_CRS)
    gdf['geometry'] = gdf.geometry.simplify(
        SIMPLIFY_TOL_DEG, preserve_topology=True,
    )

    col_order = (
        ['obs_id', 'mineral', 'threshold', 'count_px', 'mean_prob']
        + [f'band_{b:02d}' for b in range(N_BANDS)]
        + ['geometry']
    )
    return gdf[col_order]


# ── main ─────────────────────────────────────────────────────────────────────

def derive_obs_id(tile_path: str) -> str:
    """Pull the CRISM observation ID (e.g. 'frt00009d44') from the filename."""
    base = os.path.basename(tile_path)
    m = re.match(r'(frt|frs|hrl|hrs)([0-9a-f]+)', base, re.IGNORECASE)
    if m:
        return (m.group(1) + m.group(2)).lower()
    return os.path.splitext(base)[0]


def main():
    parser = argparse.ArgumentParser(
        description='Per-mineral vectorization of a targeted-observation '
                    'classifier output (.npz from classify_targeted_observation.py).'
    )
    parser.add_argument('--probs', required=True,
                        help='Path to probs.npz')
    parser.add_argument('--tile', required=True,
                        help='Path to source TRDR/MTRDR .img (used for mean spectra)')
    parser.add_argument('--out_dir', required=True,
                        help='Output directory for per-mineral GPKGs')
    parser.add_argument('--obs_id', default=None,
                        help='Observation ID (derived from tile filename if omitted)')
    parser.add_argument('--downsample', type=int, default=1,
                        help='Spatial downsample factor — must match the value '
                             'passed to classify_targeted_observation.py')
    args = parser.parse_args()

    obs_id = args.obs_id or derive_obs_id(args.tile)
    print(f'Observation ID: {obs_id}')

    # ── Load probs ───────────────────────────────────────────────────────────
    print(f'Loading probs: {args.probs}')
    npz = np.load(args.probs)
    _check_npz_channels(npz, PROB_CHANNELS)
    probs      = npz['probs'].astype(np.float32)         # (H, W, 5)
    valid_mask = npz['valid_mask'].astype(bool)           # (H, W)
    src_transform = Affine(*[float(v) for v in npz['transform']])
    src_crs    = CRS.from_wkt(str(npz['crs_wkt']))
    H, W, C    = probs.shape
    print(f'  probs shape: {probs.shape}, valid pixels: {int(valid_mask.sum()):,}')

    # ── Median-filter each prob channel (matches MC13 morphology) ─────────────
    smooth = np.zeros_like(probs)
    for ci in range(C):
        ch = probs[:, :, ci].copy()
        for _ in range(MEDIAN_ITERS):
            ch = scipy.ndimage.median_filter(ch, size=MEDIAN_SIZE)
        smooth[:, :, ci] = ch

    # ── Load + spectrally subsample the source cube to 59 bands ───────────────
    print(f'Loading source cube: {args.tile}')
    hdr_path = args.tile.replace('.img', '.hdr')
    target_wls = read_targeted_wavelengths(hdr_path)
    mrdr_wls   = get_mrdr_wavelengths()
    band_idx, _residuals = find_band_mapping(target_wls, mrdr_wls)
    print(f'  mapped {N_BANDS} mrral wavelengths into target bands '
          f'(max residual {np.max(_residuals):.2f} nm)')
    cube, cube_valid, profile = load_targeted(
        args.tile, [int(b + 1) for b in band_idx],
        downsample=args.downsample,
    )

    # Sanity: cube spatial shape must match probs after downsample
    if cube.shape[0] != H or cube.shape[1] != W:
        raise RuntimeError(
            f'Cube grid {cube.shape[:2]} does not match probs grid ({H}, {W}). '
            f'Re-check the --downsample value (passed: {args.downsample}).'
        )

    # ── Per-mineral × per-threshold polygonize ───────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    for mineral in MINERAL_NAMES:
        out_path = os.path.join(args.out_dir, f'{mineral}.gpkg')
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except PermissionError:
                with open(out_path, 'wb'):
                    pass

        print(f'\n{mineral} → {out_path}')
        smoothed_mineral = smooth[:, :, PROB_CHANNELS.index(mineral)]
        per_mineral_total = 0
        written_layer_colors: dict[str, tuple[int, int, int]] = {}

        for layer_idx, thresh in enumerate(DEFAULT_THRESHOLDS[mineral]):
            gdf = vectorize_one_cell(
                obs_id=obs_id,
                smoothed_mineral_prob=smoothed_mineral,
                valid_mask=valid_mask,
                src_transform=src_transform,
                src_crs=src_crs,
                cube_hwc=cube,
                mineral=mineral,
                threshold=thresh,
            )
            layer_name = f'thresh_{thresh:.2f}'
            if len(gdf) == 0:
                print(f'  {layer_name}: 0 polygons (skipping layer)')
                continue
            gdf.to_file(out_path, layer=layer_name, driver='GPKG')
            written_layer_colors[layer_name] = _threshold_color_rgb255(mineral, layer_idx)
            per_mineral_total += len(gdf)
            print(f'  {layer_name}: {len(gdf):,} polygons')

        if written_layer_colors:
            add_qgis_layer_styles(out_path, written_layer_colors)
            print(f'  embedded QGIS layer_styles for {len(written_layer_colors)} layers')
        print(f'  → {per_mineral_total:,} polygons total in {os.path.basename(out_path)}')

    # ── Wavelength sidecar ───────────────────────────────────────────────────
    sidecar = os.path.join(args.out_dir, f'{obs_id}_wavelengths.json')
    with open(sidecar, 'w') as f:
        json.dump({'wavelengths_nm': mrdr_wls.tolist(), 'n_bands': N_BANDS,
                   'column_prefix': 'band_',
                   'note': 'one entry per band_NN column; matches mrral band order'},
                  f, indent=2)
    print(f'\nWrote wavelength sidecar → {sidecar}')


if __name__ == '__main__':
    main()

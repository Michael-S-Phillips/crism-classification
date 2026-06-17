"""
6-class per-mineral, per-threshold vectorization for ft_6cls_mc11val_denoise_best.pt
across all MC13 quadrant tiles.

Produces 5 GeoPackages — one per mineral (olivine, lcp, hcp, plagioclase, alteration)
— each containing 5 threshold layers.

Output layout:
  data/vector_mc13_6cls/
    olivine.gpkg
    lcp.gpkg
    hcp.gpkg
    plagioclase.gpkg
    alteration.gpkg

Source npz:  /tmp/6cls_mc13/t*_probs.npz  (from classify_tile_supervised.py, 6-class)

Usage:
    conda run -n crism python scripts/vectorize_per_mineral_thresholds_mc13_6cls.py
"""
from __future__ import annotations

import glob
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
# Defaults match the original MC13 v3-denoising run; override at the CLI for
# other experiments (e.g. bland-v3 / bland-v4 / Nili-only).
DEFAULT_PROBS_DIR = '/tmp/6cls_mc13'
DEFAULT_TILE_DIR  = '/mnt/mrdr/mc13'
DEFAULT_OUT_DIR   = os.path.join(PROJ, 'data', 'vector_mc13_6cls')
DEFAULT_WAVELENGTHS_SIDECAR_NAME = 'mc13_6cls_per_mineral_wavelengths.json'

# Filled in by main() from CLI args.
PROBS_DIR = DEFAULT_PROBS_DIR
MC13_DIR  = DEFAULT_TILE_DIR
OUT_DIR   = DEFAULT_OUT_DIR
WAVELENGTHS_SIDECAR = os.path.join(OUT_DIR, DEFAULT_WAVELENGTHS_SIDECAR_NAME)

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
MINERAL_NAMES = ['olivine', 'lcp', 'hcp', 'plagioclase', 'alteration']

def _check_npz_channels(data, expected_channels):
    """Fail loudly if the npz class list doesn't match this script's constants."""
    if 'class_names' not in getattr(data, 'files', []):
        return  # legacy npz without metadata — trust constants
    names = [str(x) for x in data['class_names']]
    if names != list(expected_channels):
        raise SystemExit(
            f'npz class_names {names} != this script\'s channel order '
            f'{list(expected_channels)}.')


# Threshold grids.
# Alteration thresholds start lower because the model has high precision on it
# (val AP=0.978 on mc11val); plag still needs tight thresholds given AP=0.325.
PER_MINERAL_THRESHOLDS = {
    'olivine':     [0.80, 0.85, 0.90, 0.95, 0.97],
    'lcp':         [0.85, 0.90, 0.93, 0.95, 0.97],
    'hcp':         [0.85, 0.90, 0.93, 0.95, 0.97],
    'plagioclase': [0.88, 0.92, 0.95, 0.97, 0.98],
    'alteration':  [0.80, 0.85, 0.90, 0.93, 0.95],
}

MEDIAN_SIZE  = 3
MEDIAN_ITERS = 1
MIN_PIXELS   = 9
SIMPLIFY_TOL_DEG = 200.0 / (3396190 * np.pi / 180)  # ~200 m in degrees

# Per-mineral base color (matches the figures in plot_per_mineral_thresholds.py)
MINERAL_BASE_RGB = {
    'olivine':     (1.00, 0.00, 0.00),   # red
    'hcp':         (1.00, 0.00, 1.00),   # magenta
    'lcp':         (0.00, 1.00, 1.00),   # cyan
    'plagioclase': (1.00, 0.84, 0.00),   # gold
    'alteration':  (1.00, 0.89, 0.10),   # yellow
}
# Saturation ramp across the 5 threshold layers: lowest = MIN_SAT, highest = 1.0
N_LAYERS_PER_MINERAL = 5
MIN_SAT = 0.40
MAX_SAT = 1.00


def discover_tiles() -> list[dict]:
    """Return sorted list of {tid, mrral} dicts for tiles with probs.npz."""
    pattern = os.path.join(MC13_DIR, 't*_mrral_*_0327_4.img')
    img_paths = sorted(glob.glob(pattern))
    if not img_paths:
        raise FileNotFoundError(f'No mrral tiles found at {pattern}')

    tiles, skipped = [], []
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
    """Build polygons for one (tile, mineral, threshold) cell.

    `smoothed_mineral_prob` is the post-median-filter (H, W) array for ONE
    mineral; `mrral_cube` is the (N_BANDS, H, W) preprocessed reflectance cube
    for the tile (zeroed-nodata + clipped). Both are computed once per tile
    by the caller and reused across all 5 thresholds for the mineral.
    """
    H, W = smoothed_mineral_prob.shape
    mask_bool = (smoothed_mineral_prob >= threshold) & valid_mask

    if not mask_bool.any():
        return gpd.GeoDataFrame(geometry=[], crs=src_crs)

    # Polygonize the binary pass-mask (value=1 for pass, 0 for fail).
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

    # Drop tiny polygons (pixel area derived from native pixel size).
    pixel_area_m2 = abs(src_transform.a * src_transform.e)
    gdf['count_px'] = (gdf.geometry.area / pixel_area_m2).round().astype(int)
    gdf = gdf[gdf['count_px'] >= MIN_PIXELS].copy()
    if len(gdf) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=src_crs)

    # Rasterize each polygon with a unique 1-based id, then zonal-stat
    # the smoothed prob channel and each mrral band.
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
    gdf['geometry'] = gdf.geometry.simplify(
        SIMPLIFY_TOL_DEG, preserve_topology=True,
    )

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
    """For one tile, build polygons across every mineral × threshold.

    Returns: {mineral: {threshold: GeoDataFrame}}.
    """
    npz_path = os.path.join(PROBS_DIR, f'{tid}_probs.npz')
    data = np.load(npz_path)
    _check_npz_channels(data, PROB_CHANNELS)
    probs      = data['probs'].astype(np.float32)      # (H, W, 5)
    valid_mask = data['valid_mask'].astype(bool)        # (H, W)
    src_transform = Affine(*[float(v) for v in data['transform']])
    src_crs    = CRS.from_wkt(str(data['crs_wkt']))

    H, W, C = probs.shape

    # Per-channel median filter (computed once, reused across thresholds).
    smooth = np.zeros_like(probs)
    for ci in range(C):
        ch = probs[:, :, ci].copy()
        for _ in range(MEDIAN_ITERS):
            ch = scipy.ndimage.median_filter(ch, size=MEDIAN_SIZE)
        smooth[:, :, ci] = ch

    # Load and preprocess the mrral cube ONCE for this tile.
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
    """Return the (R, G, B) tuple in 0-255 range for one (mineral, layer_idx).

    Matches the saturation ramp used in plot_per_mineral_thresholds.py:
    layer_idx 0 → MIN_SAT × base, layer_idx N-1 → MAX_SAT × base.
    """
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
    """Insert a QGIS layer_styles table into the GPKG mapping each layer
    name to a solid-fill QML stylesheet with the given (R, G, B) color.

    QGIS auto-applies the QML when the layer is opened, so colleagues see
    the threshold ladder colors immediately without manual setup.
    """
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
        # Wipe any prior styles for the layers we're about to set, so re-runs
        # don't accumulate duplicate rows.
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
    wl = [float(w) for w in hdr.metadata['wavelength']][:N_BANDS]
    return wl


def main():
    global PROBS_DIR, MC13_DIR, OUT_DIR, WAVELENGTHS_SIDECAR

    parser = argparse.ArgumentParser(
        description='Per-mineral, per-threshold vectorization across all '
                    'tiles found in --tile_dir that have a matching probs.npz '
                    'in --probs_dir.')
    parser.add_argument('--probs_dir', default=DEFAULT_PROBS_DIR,
                        help=f'Directory containing t*_probs.npz '
                             f'(default: {DEFAULT_PROBS_DIR})')
    parser.add_argument('--tile_dir', default=DEFAULT_TILE_DIR,
                        help=f'Directory containing t*_mrral_*.img tiles '
                             f'(default: {DEFAULT_TILE_DIR})')
    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR,
                        help=f'Output directory for per-mineral GPKGs '
                             f'(default: {DEFAULT_OUT_DIR})')
    parser.add_argument('--wavelengths_sidecar', default=None,
                        help='Wavelength sidecar JSON path. '
                             'Defaults to <out_dir>/<basename of out_dir>_wavelengths.json')
    # Per-mineral threshold-grid overrides (comma-separated floats, exactly 5).
    for _m in MINERAL_NAMES:
        parser.add_argument(f'--{_m}_thresholds', default=None,
                            help=f'Override {_m} threshold grid (5 comma-sep '
                                 f'floats). Default: '
                                 f'{",".join(f"{t:.2f}" for t in PER_MINERAL_THRESHOLDS[_m])}')
    args = parser.parse_args()

    # Apply threshold overrides
    for _m in MINERAL_NAMES:
        raw = getattr(args, f'{_m}_thresholds', None)
        if raw:
            grid = [float(x) for x in raw.split(',')]
            if len(grid) != 5:
                parser.error(
                    f'--{_m}_thresholds must have exactly 5 comma-separated '
                    f'floats; got {len(grid)}: {raw}'
                )
            PER_MINERAL_THRESHOLDS[_m] = grid

    PROBS_DIR = args.probs_dir
    MC13_DIR  = args.tile_dir
    OUT_DIR   = args.out_dir
    if args.wavelengths_sidecar:
        WAVELENGTHS_SIDECAR = args.wavelengths_sidecar
    else:
        basename = os.path.basename(os.path.normpath(OUT_DIR))
        WAVELENGTHS_SIDECAR = os.path.join(OUT_DIR, f'{basename}_wavelengths.json')

    print(f'probs_dir:          {PROBS_DIR}')
    print(f'tile_dir:           {MC13_DIR}')
    print(f'out_dir:            {OUT_DIR}')
    print(f'wavelengths sidecar: {WAVELENGTHS_SIDECAR}')
    print()
    print('Threshold grid:')
    for mineral in MINERAL_NAMES:
        vals = ', '.join(f'{t:.2f}' for t in PER_MINERAL_THRESHOLDS[mineral])
        print(f'  {mineral:<12} [{vals}]')
    print()

    tiles = discover_tiles()
    if not tiles:
        print('No tiles to process — exiting.')
        sys.exit(1)

    # Per-mineral accumulators: {mineral: {threshold: list of per-tile GDFs}}
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
        print(f'  {total:,} polygons across {len(MINERAL_NAMES)}×5 cells')

    os.makedirs(OUT_DIR, exist_ok=True)

    # Write one GPKG per mineral, one layer per threshold.
    for mineral in MINERAL_NAMES:
        out_path = os.path.join(OUT_DIR, f'{mineral}.gpkg')
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except PermissionError:
                # WSL/NTFS mounts often forbid unlink but allow truncate-to-zero
                # when the file is world-writable; effectively the same outcome.
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
        # Embed QGIS layer_styles so colleagues see the ladder colors without
        # manually configuring symbology.
        if written_layer_colors:
            add_qgis_layer_styles(out_path, written_layer_colors)
            print(f'  embedded QGIS layer_styles for {len(written_layer_colors)} layers')
        print(f'  → {per_mineral_total:,} polygons total in {os.path.basename(out_path)}')

    # Wavelength sidecar (single file, since wavelengths match across MC13 tiles).
    wl = read_wavelengths_nm(tiles[0]['mrral'])
    with open(WAVELENGTHS_SIDECAR, 'w') as f:
        json.dump({'wavelengths_nm': wl, 'n_bands': N_BANDS,
                   'column_prefix': 'band_',
                   'note': 'one entry per band_NN column; matches mrral band order'},
                  f, indent=2)
    print(f'\nWrote wavelength sidecar → {WAVELENGTHS_SIDECAR}')

    # Summary
    print('\nPer-mineral × threshold polygon counts:')
    print(f'  {"mineral":<14} ' + '  '.join(
        f'{t:>10}' for t in ['0.85', '0.90', '0.93', '0.95', '0.97', '0.92', '0.94', '0.96', '0.98']
    ))
    for mineral in MINERAL_NAMES:
        row = f'  {mineral:<14} '
        for thresh in PER_MINERAL_THRESHOLDS[mineral]:
            n = sum(len(g) for g in accum[mineral][thresh])
            row += f' thresh_{thresh:.2f}={n:>6,}'
        print(row)


if __name__ == '__main__':
    main()

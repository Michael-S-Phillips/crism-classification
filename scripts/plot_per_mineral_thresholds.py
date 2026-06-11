"""
Per-mineral, threshold-ladder figures: MC13-scale + Nili-tile zooms.

For each mineral (olivine, lcp, hcp, plagioclase) and each scope (full MC13
or one of the 4 Nili Fossae tiles), produce a 2-panel PNG:

  Left  panel: MAF browse (R=OLINDEX3, G=LCPINDEX2, B=HCPINDEX2 from mrrsu).
  Right panel: same MAF browse with vector polygons overlain from that
               mineral's GPKG. All 5 threshold layers plotted; saturation
               ramps from 0.4 (lowest threshold) → 1.0 (strictest), with
               low-threshold layers drawn under high-threshold layers.

Colors:
  olivine     → red
  hcp         → magenta
  lcp         → cyan
  plagioclase → gold

Output: reports/per_mineral_mc13/
  mc13_<mineral>.png            (~2700 × 1800 px, downsample ×5)
  nili_<tid>_<mineral>.png      (native resolution per tile)

Usage (no args needed):
    conda run -n crism python scripts/plot_per_mineral_thresholds.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import sys

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.warp
from affine import Affine
from pyproj import CRS
from rasterio.enums import Resampling

PROJECT_ROOT = '/mnt/mrdr/crism_classification'

# Defaults match the v3-denoising MC13 product; overridable via CLI for
# bland-v3 / bland-v4 / Nili-only / etc. experiments.
DEFAULT_TILE_DIR   = '/mnt/mrdr/mc13'
DEFAULT_VECTOR_DIR = os.path.join(PROJECT_ROOT, 'data', 'vector_mc13_v3_denoising')
DEFAULT_OUT_DIR    = os.path.join(PROJECT_ROOT, 'reports', 'per_mineral_mc13')
DEFAULT_WAVELENGTHS_NAME = 'mc13_v3_denoising_per_mineral_wavelengths.json'

# Filled by main() from CLI args.
MC13_DIR     = DEFAULT_TILE_DIR
VECTOR_DIR   = DEFAULT_VECTOR_DIR
OUT_DIR      = DEFAULT_OUT_DIR
WAVELENGTHS_NAME = DEFAULT_WAVELENGTHS_NAME

# Mars 2000 geographic CRS — matches the GPKG products
MARS_GEO_WKT = (
    'GEOGCS["GCS_Mars_2000",'
    'DATUM["D_Mars_2000",SPHEROID["Mars_2000_IAU_IAG",3396190,169.8944472]],'
    'PRIMEM["Reference_Meridian",0],'
    'UNIT["Degree",0.0174532925199433]]'
)
COMMON_CRS = CRS.from_wkt(MARS_GEO_WKT)

# mrrsu band indexes (1-indexed for rasterio)
BAND_OLINDEX3  = 16
BAND_LCPINDEX2 = 19
BAND_HCPINDEX2 = 20

# Mineral → vector base color
MINERAL_BASE_RGB = {
    'olivine':     (1.00, 0.00, 0.00),   # red
    'hcp':         (1.00, 0.00, 1.00),   # magenta
    'lcp':         (0.00, 1.00, 1.00),   # cyan
    'plagioclase': (1.00, 0.84, 0.00),   # gold
}

# Threshold grid per mineral (must match vectorize_per_mineral_thresholds_mc13.py)
PER_MINERAL_THRESHOLDS = {
    'olivine':     [0.80, 0.85, 0.90, 0.95, 0.97],
    'lcp':         [0.85, 0.90, 0.93, 0.95, 0.97],
    'hcp':         [0.85, 0.90, 0.93, 0.95, 0.97],
    'plagioclase': [0.92, 0.94, 0.96, 0.97, 0.98],
}
MINERAL_ORDER = ['olivine', 'hcp', 'lcp', 'plagioclase']

# Saturation ramp: lowest layer → MIN_SAT, highest layer → MAX_SAT
N_LAYERS_PER_MINERAL = 5
MIN_SAT = 0.40
MAX_SAT = 1.00

# Downsample factor for MC13-scale figure (×5 → ~1 km/pixel)
DOWNSAMPLE_FACTOR = 5

# Nili Fossae 4-tile zoom-in set
NILI_TILES = ['t1249', 't1250', 't1321', 't1322']

DPI = 220


# ── tile discovery + grid utilities (mirrors plot_mc13_classifier_vs_params.py) ──

def discover_tiles(scope='mc13') -> list[dict]:
    """Return list of {tid, mrral, mrrsu} dicts.

    scope='mc13' → all MC13 tiles with mrrsu available.
    scope='nili' → only the 4 Nili tiles.
    """
    mrral_paths = sorted(glob.glob(
        os.path.join(MC13_DIR, 't*_mrral_*_0327_4.img')
    ))
    tiles, skipped = [], []
    for mrral_path in mrral_paths:
        basename = os.path.basename(mrral_path)
        m = re.match(r'(t\d+)_mrral_([\w]+)_0327_4\.img', basename)
        if not m:
            continue
        tid, latlon = m.group(1), m.group(2)
        if scope == 'nili' and tid not in NILI_TILES:
            continue
        mrrsu_path = os.path.join(MC13_DIR, f'{tid}_mrrsu_{latlon}_0327_4.img')
        if not os.path.exists(mrrsu_path):
            skipped.append(tid)
            continue
        tiles.append({'tid': tid, 'mrral': mrral_path, 'mrrsu': mrrsu_path})
    if skipped:
        print(f'  Skipped {len(skipped)} tiles without mrrsu: {", ".join(skipped)}')
    return tiles


def compute_target_grid(tiles, downsample=DOWNSAMPLE_FACTOR):
    """Build a common geographic raster grid covering all `tiles`."""
    bounds_list, src_res_list = [], []
    for t in tiles:
        with rasterio.open(t['mrral']) as src:
            b = rasterio.warp.transform_bounds(src.crs, COMMON_CRS, *src.bounds)
            bounds_list.append(b)
            src_res_list.append(abs(src.transform.a))
    xmin = min(b[0] for b in bounds_list)
    ymin = min(b[1] for b in bounds_list)
    xmax = max(b[2] for b in bounds_list)
    ymax = max(b[3] for b in bounds_list)
    native_res_m   = max(src_res_list)
    target_res_m   = native_res_m * downsample
    deg_per_m      = 1.0 / (3396190 * math.pi / 180)
    target_res_deg = target_res_m * deg_per_m
    n_x = int(math.ceil((xmax - xmin) / target_res_deg))
    n_y = int(math.ceil((ymax - ymin) / target_res_deg))
    target_transform = rasterio.transform.from_origin(
        xmin, ymax, target_res_deg, target_res_deg
    )
    return target_transform, (n_y, n_x), (xmin, ymin, xmax, ymax)


def reproject_continuous(src_path, band_idx, target_transform, target_shape):
    """Reproject one band to the common grid. NaN for nodata."""
    dst = np.full(target_shape, np.nan, dtype=np.float32)
    with rasterio.open(src_path) as src:
        src_arr = src.read(band_idx).astype(np.float32)
        nodata  = (src_arr == 65535) | ~np.isfinite(src_arr)
        src_arr[nodata] = np.nan
        rasterio.warp.reproject(
            source=src_arr, destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=target_transform, dst_crs=COMMON_CRS,
            resampling=Resampling.bilinear,
            src_nodata=np.nan, dst_nodata=np.nan,
        )
    return dst


def build_maf_browse(tiles, target_transform, target_shape):
    """OLINDEX3/LCPINDEX2/HCPINDEX2 composite, pooled-percentile stretched.

    Returns RGB array (H, W, 3) with light-gray fill outside coverage.
    """
    out = np.full((target_shape[0], target_shape[1], 3), np.nan, dtype=np.float32)
    for t in tiles:
        ol = reproject_continuous(t['mrrsu'], BAND_OLINDEX3,  target_transform, target_shape)
        lc = reproject_continuous(t['mrrsu'], BAND_LCPINDEX2, target_transform, target_shape)
        hc = reproject_continuous(t['mrrsu'], BAND_HCPINDEX2, target_transform, target_shape)
        for chan, src_band in enumerate([ol, lc, hc]):
            empty = np.isnan(out[:, :, chan])
            valid = np.isfinite(src_band)
            mask  = empty & valid
            out[mask, chan] = src_band[mask]
    rgb = np.zeros_like(out)
    for c in range(3):
        ch    = out[:, :, c]
        valid = np.isfinite(ch)
        if valid.any():
            p2, p98 = np.percentile(ch[valid], [2, 98])
            rgb[:, :, c] = np.clip((ch - p2) / max(1e-6, p98 - p2), 0, 1)
    invalid = np.isnan(out).any(axis=-1)
    for c in range(3):
        rgb[:, :, c][invalid] = 0.85
    return rgb


# ── color ramp helper ────────────────────────────────────────────────────────

def threshold_color(base_rgb, layer_idx):
    """Map (base_rgb, layer_idx 0..N-1) → darkened RGB.

    Saturation ramps linearly from MIN_SAT (idx=0) to MAX_SAT (idx=N-1).
    """
    if N_LAYERS_PER_MINERAL <= 1:
        sat = MAX_SAT
    else:
        sat = MIN_SAT + (MAX_SAT - MIN_SAT) * layer_idx / (N_LAYERS_PER_MINERAL - 1)
    return tuple(c * sat for c in base_rgb)


# ── figure builders ──────────────────────────────────────────────────────────

def render_two_panel(
    maf_rgb,
    extent_deg,
    overlay_gdf_layers,   # list of (layer_idx, threshold, gdf)
    mineral,
    title_left,
    title_right,
    out_path,
):
    """Save a 2-panel figure: MAF browse | MAF browse + vector overlay."""
    xmin, ymin, xmax, ymax = extent_deg
    base_rgb = MINERAL_BASE_RGB[mineral]

    # Aspect: mid-latitude foreshortening for figure size
    lat_mid = (ymin + ymax) / 2
    cos_mid = math.cos(math.radians(lat_mid))
    fig_w_per_panel = max(4.0, 8.0 * (xmax - xmin) * cos_mid / max(0.01, ymax - ymin))
    fig_h           = 8.0
    fig, axes = plt.subplots(
        1, 2, figsize=(2 * fig_w_per_panel + 1.0, fig_h),
        constrained_layout=True,
    )

    for ax in axes:
        ax.imshow(maf_rgb, extent=[xmin, xmax, ymin, ymax], origin='upper')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('lon (°E)')
        ax.set_ylabel('lat (°N)')
        ax.set_aspect(1.0 / cos_mid)

    axes[0].set_title(title_left, fontsize=10)
    axes[1].set_title(title_right, fontsize=10)

    # Draw overlay: lowest threshold at bottom, highest on top
    for layer_idx, thresh, gdf in overlay_gdf_layers:
        if gdf is None or len(gdf) == 0:
            continue
        color = threshold_color(base_rgb, layer_idx)
        gdf.plot(
            ax=axes[1], facecolor=color, edgecolor='none',
            zorder=10 + layer_idx, alpha=1.0,
        )

    # Legend with one swatch per threshold — keep on top of vector layers
    handles = []
    for layer_idx, thresh, _ in overlay_gdf_layers:
        color = threshold_color(base_rgb, layer_idx)
        handles.append(mpatches.Patch(facecolor=color,
                                       edgecolor='none',
                                       label=f'thresh ≥ {thresh:.2f}'))
    leg = axes[1].legend(handles=handles, loc='lower left', fontsize=7,
                          framealpha=0.95, title=mineral)
    leg.set_zorder(100)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {out_path}')


def load_wavelengths_nm() -> list[float]:
    """Read wavelength axis (59 entries, nm) from the per-mineral sidecar JSON.

    Searches VECTOR_DIR for either (a) the configured WAVELENGTHS_NAME, or
    (b) any single *_wavelengths.json — useful when the sidecar was written
    by the parameterized vectorize step under a different basename.
    """
    primary = os.path.join(VECTOR_DIR, WAVELENGTHS_NAME)
    if os.path.exists(primary):
        with open(primary) as f:
            return json.load(f)['wavelengths_nm']
    import glob as _glob
    matches = _glob.glob(os.path.join(VECTOR_DIR, '*_wavelengths.json'))
    if not matches:
        raise FileNotFoundError(
            f'No wavelength sidecar found in {VECTOR_DIR}. Looked for '
            f'{primary} and *_wavelengths.json.'
        )
    with open(matches[0]) as f:
        return json.load(f)['wavelengths_nm']


def render_three_panel_tile(
    maf_rgb,
    extent_deg,
    overlay_gdf_layers,
    mineral,
    tid,
    wavelengths_nm,
    out_path,
):
    """Save a 4-axis figure for one (tile, mineral):

      Top-left:    MAF browse
      Top-right:   MAF browse + vector overlay (low→high stacked)
      Bottom-left: per-threshold pixel-count bar chart
      Bottom-right: per-threshold mean reflectance spectrum
    """
    xmin, ymin, xmax, ymax = extent_deg
    base_rgb = MINERAL_BASE_RGB[mineral]

    lat_mid = (ymin + ymax) / 2
    cos_mid = math.cos(math.radians(lat_mid))
    fig_w_per_panel = max(4.0, 8.0 * (xmax - xmin) * cos_mid / max(0.01, ymax - ymin))
    fig = plt.figure(
        figsize=(2 * fig_w_per_panel + 1.0, 11.0),
        constrained_layout=True,
    )
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1.0])
    ax_maf      = fig.add_subplot(gs[0, 0])
    ax_overlay  = fig.add_subplot(gs[0, 1])
    ax_counts   = fig.add_subplot(gs[1, 0])
    ax_spectra  = fig.add_subplot(gs[1, 1])

    # ── Top row: MAF browse + overlay ────────────────────────────────────────
    for ax in (ax_maf, ax_overlay):
        ax.imshow(maf_rgb, extent=[xmin, xmax, ymin, ymax], origin='upper')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('lon (°E)')
        ax.set_ylabel('lat (°N)')
        ax.set_aspect(1.0 / cos_mid)
    ax_maf.set_title(f'{tid} — MAF browse', fontsize=10)
    ax_overlay.set_title(f'{tid} — {mineral} threshold ladder', fontsize=10)

    for layer_idx, thresh, gdf in overlay_gdf_layers:
        if gdf is None or len(gdf) == 0:
            continue
        color = threshold_color(base_rgb, layer_idx)
        gdf.plot(
            ax=ax_overlay, facecolor=color, edgecolor='none',
            zorder=10 + layer_idx, alpha=1.0,
        )

    handles = []
    for layer_idx, thresh, _ in overlay_gdf_layers:
        color = threshold_color(base_rgb, layer_idx)
        handles.append(mpatches.Patch(facecolor=color,
                                       edgecolor='none',
                                       label=f'thresh ≥ {thresh:.2f}'))
    leg = ax_overlay.legend(handles=handles, loc='lower left', fontsize=7,
                              framealpha=0.95, title=mineral)
    leg.set_zorder(100)

    # ── Bottom-left: pixel counts per threshold ──────────────────────────────
    # `tile_gdf` rows already filtered to this tile via clip_bbox in caller.
    bar_thresholds = []
    bar_counts     = []
    bar_colors     = []
    for layer_idx, thresh, gdf in overlay_gdf_layers:
        bar_thresholds.append(thresh)
        # Sum count_px across this tile's polygons in this threshold layer
        if gdf is None or len(gdf) == 0 or 'count_px' not in gdf.columns:
            bar_counts.append(0)
        else:
            bar_counts.append(int(gdf['count_px'].sum()))
        bar_colors.append(threshold_color(base_rgb, layer_idx))
    x_positions = np.arange(len(bar_thresholds))
    ax_counts.bar(x_positions, bar_counts, color=bar_colors, edgecolor='black',
                   linewidth=0.4)
    ax_counts.set_xticks(x_positions)
    ax_counts.set_xticklabels([f'{t:.2f}' for t in bar_thresholds], fontsize=8)
    ax_counts.set_xlabel('Threshold')
    ax_counts.set_ylabel('Pixel count (sum across polygons)')
    ax_counts.set_title(f'{tid} {mineral} — pixel count per threshold', fontsize=10)
    ax_counts.grid(True, axis='y', alpha=0.3)
    # Annotate counts above each bar
    if any(c > 0 for c in bar_counts):
        ymax_bar = max(bar_counts) * 1.08
        for x, c in zip(x_positions, bar_counts):
            if c > 0:
                ax_counts.text(x, c, f'{c:,}', ha='center', va='bottom', fontsize=7)
        ax_counts.set_ylim(0, ymax_bar)

    # ── Bottom-right: per-threshold mean reflectance spectrum ────────────────
    band_cols = [f'band_{b:02d}' for b in range(59)]
    for layer_idx, thresh, gdf in overlay_gdf_layers:
        if gdf is None or len(gdf) == 0:
            continue
        # Pixel-count-weighted mean spectrum across polygons
        bands = gdf[band_cols].to_numpy(dtype=np.float64)   # (n_polys, 59)
        weights = gdf['count_px'].to_numpy(dtype=np.float64)
        if weights.sum() == 0:
            continue
        mean_spec = (bands * weights[:, None]).sum(axis=0) / weights.sum()
        color = threshold_color(base_rgb, layer_idx)
        ax_spectra.plot(wavelengths_nm, mean_spec, color=color,
                        linewidth=1.4, label=f'thresh ≥ {thresh:.2f}')
    ax_spectra.set_xlabel('Wavelength (nm)')
    ax_spectra.set_ylabel('Mean reflectance (I/F)')
    ax_spectra.set_title(f'{tid} {mineral} — mean spectrum per threshold',
                         fontsize=10)
    ax_spectra.grid(True, alpha=0.3)
    leg_spec = ax_spectra.legend(loc='best', fontsize=7, framealpha=0.95)
    if leg_spec is not None:
        leg_spec.set_zorder(100)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {out_path}')


def load_mineral_layers(mineral, clip_bbox=None):
    """Load all threshold layers of one mineral GPKG into a list.

    Returns: [(layer_idx, threshold, GeoDataFrame), …]; missing layers skipped.
    If clip_bbox=(xmin, ymin, xmax, ymax) is given, polygons outside are dropped.
    """
    gpkg = os.path.join(VECTOR_DIR, f'{mineral}.gpkg')
    out = []
    for layer_idx, thresh in enumerate(PER_MINERAL_THRESHOLDS[mineral]):
        layer_name = f'thresh_{thresh:.2f}'
        try:
            gdf = gpd.read_file(gpkg, layer=layer_name)
        except Exception:
            print(f'    [warn] {mineral}: layer {layer_name} not present in {gpkg}')
            continue
        if gdf.crs != COMMON_CRS:
            gdf = gdf.to_crs(COMMON_CRS)
        if clip_bbox is not None:
            xmin, ymin, xmax, ymax = clip_bbox
            gdf = gdf.cx[xmin:xmax, ymin:ymax].copy()
        out.append((layer_idx, thresh, gdf))
    return out


def render_mc13(tiles):
    """One MC13-scale figure per mineral."""
    print(f'Building MC13-scale MAF browse (downsample ×{DOWNSAMPLE_FACTOR}) …')
    target_transform, target_shape, bounds = compute_target_grid(tiles)
    print(f'  grid: {target_shape[1]} × {target_shape[0]} pixels')
    maf_rgb = build_maf_browse(tiles, target_transform, target_shape)

    xmin, ymin, xmax, ymax = bounds
    for mineral in MINERAL_ORDER:
        print(f'  Rendering MC13 {mineral} …')
        layers = load_mineral_layers(mineral)
        out_path = os.path.join(OUT_DIR, f'mc13_{mineral}.png')
        render_two_panel(
            maf_rgb=maf_rgb,
            extent_deg=bounds,
            overlay_gdf_layers=layers,
            mineral=mineral,
            title_left='MAF browse (R: OLINDEX3, G: LCPINDEX2, B: HCPINDEX2)',
            title_right=f'{mineral} vectors — threshold ladder (low→high, dim→bright)',
            out_path=out_path,
        )


def render_nili_zoom(tiles):
    """One zoom figure per (Nili tile × mineral) = 4×4 = 16 figures.

    Each tile is rendered at native resolution (no downsample) so individual
    polygons are visible. Layout is 4-axis (2×2): top row = MAF + overlay,
    bottom row = pixel-count bars + mean spectra.
    """
    nili_tiles = [t for t in tiles if t['tid'] in NILI_TILES]
    if not nili_tiles:
        print('  No Nili tiles available, skipping zoom-ins.')
        return

    wavelengths = load_wavelengths_nm()

    for tile in nili_tiles:
        tid = tile['tid']
        print(f'Building Nili zoom for {tid} (native resolution) …')
        # Single-tile target grid at native resolution
        target_transform, target_shape, bounds = compute_target_grid(
            [tile], downsample=1,
        )
        print(f'  grid: {target_shape[1]} × {target_shape[0]} pixels')
        maf_rgb = build_maf_browse([tile], target_transform, target_shape)

        for mineral in MINERAL_ORDER:
            print(f'  Rendering {tid} {mineral} …')
            layers = load_mineral_layers(mineral, clip_bbox=bounds)
            out_path = os.path.join(OUT_DIR, f'nili_{tid}_{mineral}.png')
            render_three_panel_tile(
                maf_rgb=maf_rgb,
                extent_deg=bounds,
                overlay_gdf_layers=layers,
                mineral=mineral,
                tid=tid,
                wavelengths_nm=wavelengths,
                out_path=out_path,
            )


def main():
    global MC13_DIR, VECTOR_DIR, OUT_DIR, WAVELENGTHS_NAME

    import argparse
    parser = argparse.ArgumentParser(
        description='Render per-mineral threshold figures (MC13-scale + Nili '
                    'zooms) from per-mineral GPKGs.')
    parser.add_argument('--tile_dir', default=DEFAULT_TILE_DIR,
                        help=f'Directory with mrral/mrrsu .img tiles '
                             f'(default: {DEFAULT_TILE_DIR})')
    parser.add_argument('--vector_dir', default=DEFAULT_VECTOR_DIR,
                        help=f'Directory with {{olivine,lcp,hcp,plagioclase}}.gpkg '
                             f'(default: {DEFAULT_VECTOR_DIR})')
    parser.add_argument('--out_dir', default=DEFAULT_OUT_DIR,
                        help=f'Where to write PNGs '
                             f'(default: {DEFAULT_OUT_DIR})')
    parser.add_argument('--wavelengths_name', default=DEFAULT_WAVELENGTHS_NAME,
                        help='Sidecar filename inside --vector_dir. If not '
                             'found, any *_wavelengths.json in --vector_dir is used.')
    parser.add_argument('--skip_mc13', action='store_true',
                        help='Skip the 4 MC13-scale figures (useful when only '
                             'a small region was vectorized).')
    parser.add_argument('--skip_nili', action='store_true',
                        help='Skip the 16 Nili-tile zoom figures.')
    # Per-mineral threshold-grid overrides (must match what the vectorize step used)
    for _m in MINERAL_ORDER:
        parser.add_argument(f'--{_m}_thresholds', default=None,
                            help=f'Override {_m} threshold grid (5 comma-sep '
                                 f'floats). Must match the vectorize step '
                                 f'that produced --vector_dir. Default: '
                                 f'{",".join(f"{t:.2f}" for t in PER_MINERAL_THRESHOLDS[_m])}')
    args = parser.parse_args()

    # Apply threshold overrides
    for _m in MINERAL_ORDER:
        raw = getattr(args, f'{_m}_thresholds', None)
        if raw:
            grid = [float(x) for x in raw.split(',')]
            if len(grid) != 5:
                parser.error(
                    f'--{_m}_thresholds must have exactly 5 comma-separated '
                    f'floats; got {len(grid)}: {raw}'
                )
            PER_MINERAL_THRESHOLDS[_m] = grid

    MC13_DIR         = args.tile_dir
    VECTOR_DIR       = args.vector_dir
    OUT_DIR          = args.out_dir
    WAVELENGTHS_NAME = args.wavelengths_name

    print('Per-mineral threshold figures')
    print(f'  tile_dir:    {MC13_DIR}')
    print(f'  vector_dir:  {VECTOR_DIR}')
    print(f'  out_dir:     {OUT_DIR}')
    print(f'  skip_mc13:   {args.skip_mc13}')
    print(f'  skip_nili:   {args.skip_nili}')
    print()

    print('Discovering tiles …')
    all_tiles = discover_tiles(scope='mc13')
    print(f'  {len(all_tiles)} tiles included')
    if not all_tiles:
        sys.exit('No tiles available — exiting.')

    if not args.skip_mc13:
        print()
        render_mc13(all_tiles)

    if not args.skip_nili:
        print()
        render_nili_zoom(all_tiles)

    print('\nDone.')


if __name__ == '__main__':
    main()

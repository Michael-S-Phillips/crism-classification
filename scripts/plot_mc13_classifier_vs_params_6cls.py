"""
6-class version of the MC13 seamless mosaic (ft_6cls_mc11val_denoise_best.pt,
epoch 10, val_mAP=0.814).

Probabilities from /tmp/6cls_mc13/t*_probs.npz (6 channels: olivine/lcp/hcp/plag/other/alteration).
Alteration shown as yellow additive blend.

Usage:
    conda run -n crism python scripts/plot_mc13_classifier_vs_params_6cls.py
"""
from __future__ import annotations

import glob
import math
import os
import re
import sys

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
PROBS_DIR    = '/tmp/6cls_mc13'
MC13_DIR     = '/mnt/mrdr/mc13'

# Mars 2000 geographic CRS (degrees lon/lat)
MARS_GEO_WKT = (
    'GEOGCS["GCS_Mars_2000",'
    'DATUM["D_Mars_2000",SPHEROID["Mars_2000_IAU_IAG",3396190,169.8944472]],'
    'PRIMEM["Reference_Meridian",0],'
    'UNIT["Degree",0.0174532925199433]]'
)
COMMON_CRS = CRS.from_wkt(MARS_GEO_WKT)

# mrrsu band indexes (1-indexed for rasterio)
BAND_OLINDEX3  = 16   # rasterio band 16 (0-indexed 15)
BAND_LCPINDEX2 = 19
BAND_HCPINDEX2 = 20

# Multi-label color scheme: additive RGB, plagioclase is subtractive (darken)
CLASS_NAMES = ['olivine', 'lcp', 'hcp', 'plagioclase', 'alteration']
CLASS_COLORS_01 = {
    'olivine':     np.array([ 0.85,  0.00,  0.00], dtype=np.float32),  # red
    'lcp':         np.array([ 0.00,  0.00,  0.85], dtype=np.float32),  # blue
    'hcp':         np.array([ 0.00,  0.70,  0.00], dtype=np.float32),  # green
    'plagioclase': np.array([-0.50, -0.50, -0.50], dtype=np.float32),  # darken
    'alteration':  np.array([ 0.70,  0.70,  0.00], dtype=np.float32),  # yellow
}

# Threshold scheme: max(inverse-proportional, floor)
# Val APs from ft_6cls_mc11val_denoise_best.pt test-split (default parquet).
VAL_AP = {
    'olivine':     0.996,
    'lcp':         0.895,
    'hcp':         0.643,
    'plagioclase': 0.325,
    'other':       1.000,
    'alteration':  0.978,  # from mc11val val split; default parquet AP is contaminated
}
INVERSE_THRESH  = {c: 0.99 - 0.49 * VAL_AP[c] for c in VAL_AP}
GLOBAL_MIN_THRESH = 0.8
FINAL_THRESH    = {c: max(INVERSE_THRESH[c], GLOBAL_MIN_THRESH) for c in VAL_AP}

# Probability channel order in saved npz (matches classify_tile_supervised.py 6-class)
PROB_CHANNELS = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other', 'alteration']

# Downsample factor relative to native pixel size
DOWNSAMPLE_FACTOR = 5

OUT_PATH = os.path.join(
    PROJECT_ROOT, 'reports', 'v5',
    'fig_mc13_6cls_classifier_vs_params.png',
)


# ── tile discovery ────────────────────────────────────────────────────────────

def discover_tiles() -> list[dict]:
    """Return sorted list of {tid, mrral, mrrsu} dicts for tiles with probs.npz."""
    mrral_paths = sorted(glob.glob(
        os.path.join(MC13_DIR, 't*_mrral_*_0327_4.img')
    ))
    tiles = []
    missing_probs = []
    missing_mrrsu = []

    for mrral_path in mrral_paths:
        basename = os.path.basename(mrral_path)
        m = re.match(r'(t\d+)_mrral_([\w]+)_0327_4\.img', basename)
        if not m:
            continue
        tid     = m.group(1)
        latlon  = m.group(2)
        npz_path = os.path.join(PROBS_DIR, f'{tid}_probs.npz')
        mrrsu_path = os.path.join(MC13_DIR, f'{tid}_mrrsu_{latlon}_0327_4.img')

        if not os.path.exists(npz_path):
            missing_probs.append(tid)
            continue
        if not os.path.exists(mrrsu_path):
            missing_mrrsu.append(tid)
            continue

        tiles.append({'tid': tid, 'mrral': mrral_path, 'mrrsu': mrrsu_path})

    if missing_probs:
        print(f'  WARNING: {len(missing_probs)} tiles missing probs.npz (skipped): '
              + ', '.join(missing_probs))
    if missing_mrrsu:
        print(f'  WARNING: {len(missing_mrrsu)} tiles missing mrrsu (skipped): '
              + ', '.join(missing_mrrsu))
    print(f'  {len(tiles)} tiles included in figure')
    return tiles


# ── grid utilities ────────────────────────────────────────────────────────────

def compute_target_grid(tiles: list[dict], downsample: int = DOWNSAMPLE_FACTOR):
    """Build a common geographic raster grid covering all tiles.

    Resolution is native * downsample_factor (≈5×, so ~1 km/pixel) to keep
    the full MC13 mosaic memory-bounded (~2700×1800 pixels).
    """
    bounds_list = []
    src_res_list = []

    for t in tiles:
        with rasterio.open(t['mrral']) as src:
            b = rasterio.warp.transform_bounds(src.crs, COMMON_CRS, *src.bounds)
            bounds_list.append(b)
            src_res_list.append(abs(src.transform.a))  # native res in m/px

    xmin = min(b[0] for b in bounds_list)
    ymin = min(b[1] for b in bounds_list)
    xmax = max(b[2] for b in bounds_list)
    ymax = max(b[3] for b in bounds_list)

    native_res_m  = max(src_res_list)
    target_res_m  = native_res_m * downsample
    deg_per_m     = 1.0 / (3396190 * math.pi / 180)
    target_res_deg = target_res_m * deg_per_m

    n_x = int(math.ceil((xmax - xmin) / target_res_deg))
    n_y = int(math.ceil((ymax - ymin) / target_res_deg))
    target_transform = rasterio.transform.from_origin(
        xmin, ymax, target_res_deg, target_res_deg
    )
    return target_transform, (n_y, n_x), (xmin, ymin, xmax, ymax)


def reproject_continuous(src_path, band_idx, target_transform, target_shape):
    """Reproject one band to the common grid. Returns float32, NaN for nodata."""
    dst = np.full(target_shape, np.nan, dtype=np.float32)
    with rasterio.open(src_path) as src:
        src_arr = src.read(band_idx).astype(np.float32)
        nodata  = (src_arr == 65535) | ~np.isfinite(src_arr)
        src_arr[nodata] = np.nan
        rasterio.warp.reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=target_transform, dst_crs=COMMON_CRS,
            resampling=Resampling.bilinear,
            src_nodata=np.nan, dst_nodata=np.nan,
        )
    return dst


def reproject_rgb(rgb_hwc, src_transform, src_crs, target_transform, target_shape):
    """Reproject (H, W, 3) RGB float32 to common grid. NaN-aware per channel."""
    dst = np.full((target_shape[0], target_shape[1], 3), np.nan, dtype=np.float32)
    for c in range(3):
        rasterio.warp.reproject(
            source=rgb_hwc[:, :, c],
            destination=dst[:, :, c],
            src_transform=src_transform, src_crs=src_crs,
            dst_transform=target_transform, dst_crs=COMMON_CRS,
            resampling=Resampling.nearest,
            src_nodata=np.nan, dst_nodata=np.nan,
        )
    return dst


# ── panel builders ────────────────────────────────────────────────────────────

def build_band_param_mosaic(tiles, target_transform, target_shape):
    """Build OLINDEX3/LCPINDEX2/HCPINDEX2 composite, pooled-percentile stretched."""
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

    # Per-channel pooled percentile stretch
    rgb = np.zeros_like(out)
    for c in range(3):
        ch    = out[:, :, c]
        valid = np.isfinite(ch)
        if valid.any():
            p2, p98 = np.percentile(ch[valid], [2, 98])
            rgb[:, :, c] = np.clip((ch - p2) / max(1e-6, p98 - p2), 0, 1)
    # Outside-coverage → light gray
    invalid = np.isnan(out).any(axis=-1)
    for c in range(3):
        rgb[:, :, c][invalid] = 0.85
    return rgb, invalid


def build_classification_mosaic(tiles, target_transform, target_shape, thresholds):
    """Apply per-class thresholds, additive-blend multi-label RGB per tile, mosaic."""
    out = np.full((target_shape[0], target_shape[1], 3), np.nan, dtype=np.float32)
    per_tile_stats = []

    for t in tiles:
        npz_path = os.path.join(PROBS_DIR, f"{t['tid']}_probs.npz")
        npz       = np.load(npz_path)
        probs      = npz['probs'].astype(np.float32)   # (H, W, 5)
        valid_mask = npz['valid_mask'].astype(bool)     # (H, W)
        src_t      = Affine(*[float(v) for v in npz['transform']])
        src_crs    = CRS.from_wkt(str(npz['crs_wkt']))

        H, W, _ = probs.shape
        tile_rgb  = np.zeros((H, W, 3), dtype=np.float32)
        any_passed = np.zeros((H, W), dtype=bool)

        stats = {'tile': t['tid'], 'n_valid': int(valid_mask.sum())}
        for cname in CLASS_NAMES:
            ci      = PROB_CHANNELS.index(cname)
            passing = (probs[:, :, ci] >= thresholds[cname]) & valid_mask
            stats[cname] = int(passing.sum())
            tile_rgb[passing] += CLASS_COLORS_01[cname]
            any_passed |= passing
        np.clip(tile_rgb, 0.0, 1.0, out=tile_rgb)

        unclass_mask = valid_mask & ~any_passed
        tile_rgb[unclass_mask] = 1.0
        tile_rgb[~valid_mask] = np.nan
        stats['unclassified'] = int(unclass_mask.sum())
        per_tile_stats.append(stats)

        reproj = reproject_rgb(tile_rgb, src_t, src_crs, target_transform, target_shape)
        write_mask = np.isfinite(reproj).all(axis=-1) & np.isnan(out[:, :, 0])
        out[write_mask] = reproj[write_mask]

    invalid = np.isnan(out).any(axis=-1)
    for c in range(3):
        out[:, :, c][invalid] = 0.85
    return out, invalid, per_tile_stats


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print('Threshold scheme:')
    for c in PROB_CHANNELS:
        used = FINAL_THRESH[c]
        inv  = INVERSE_THRESH[c]
        gate = 'floor' if GLOBAL_MIN_THRESH > inv else 'inverse'
        print(f'  {c:<12} val_AP={VAL_AP[c]:.3f}  inverse={inv:.3f}  '
              f'floor={GLOBAL_MIN_THRESH:.2f}  → used={used:.3f}  ({gate})')
    print()

    print('Discovering tiles …')
    tiles = discover_tiles()
    if not tiles:
        print('No tiles available — exiting.')
        sys.exit(1)

    print(f'Computing target grid (downsample ×{DOWNSAMPLE_FACTOR}) …')
    target_transform, target_shape, bounds = compute_target_grid(tiles)
    print(f'  grid: {target_shape[1]} × {target_shape[0]} pixels  '
          f'(~{target_shape[1]*target_shape[0]/1e6:.1f} Mpx)')
    xmin, ymin, xmax, ymax = bounds
    print(f'  extent: lon [{xmin:.2f}, {xmax:.2f}]  lat [{ymin:.2f}, {ymax:.2f}]')

    print('Building band-parameter mosaic …')
    band_param_rgb, _ = build_band_param_mosaic(tiles, target_transform, target_shape)
    print(f'  band-param mosaic shape: {band_param_rgb.shape}')

    print('Building classification mosaic …')
    class_rgb, _, stats = build_classification_mosaic(
        tiles, target_transform, target_shape, FINAL_THRESH
    )
    print(f'  classification mosaic shape: {class_rgb.shape}')

    # ── figure layout ─────────────────────────────────────────────────────────
    # MC13 covers roughly 5°–35°N latitude (30° range) and 48°–93°E (45° range).
    # At mid-lat ~20°N, cos(20°) ≈ 0.94, so geographic aspect ≈ 30/(45*0.94) ≈ 0.71
    # i.e. slightly taller than wide → side-by-side panels.
    lat_mid = (ymin + ymax) / 2
    cos_mid = math.cos(math.radians(lat_mid))
    lon_range = xmax - xmin
    lat_range = ymax - ymin
    geo_aspect = lat_range / (lon_range * cos_mid)
    print(f'\nGeographic aspect (lat_range / (lon_range*cos_lat)): {geo_aspect:.3f}')

    if geo_aspect > 1.0:
        layout = 'side-by-side'
    else:
        layout = 'stacked'

    print(f'Layout: {layout}')

    extent = (xmin, xmax, ymin, ymax)

    if layout == 'side-by-side':
        # Each panel occupies full height, half the width
        panel_h = 12
        panel_w = panel_h / geo_aspect  # width ∝ 1/aspect
        fig, axes = plt.subplots(
            1, 2, figsize=(panel_w * 2 + 0.5, panel_h),
        )
    else:
        # stacked: full width, each panel half height
        panel_w = 14
        panel_h = panel_w * geo_aspect
        fig, axes = plt.subplots(
            2, 1, figsize=(panel_w, panel_h * 2 + 1.5),
        )

    ax0, ax1 = axes.flat if hasattr(axes, 'flat') else axes

    # Panel 1: band-parameter composite
    ax0.imshow(band_param_rgb, extent=extent, origin='upper', aspect='auto')
    ax0.set_xlim(xmin, xmax)
    ax0.set_ylim(ymin, ymax)
    ax0.set_title(
        'MAF band-parameter composite\n'
        'R = OLINDEX3  ·  G = LCPINDEX2  ·  B = HCPINDEX2',
        fontsize=10,
    )
    ax0.set_xlabel('Longitude (°E)')
    ax0.set_ylabel('Latitude (°N)')
    ax0.tick_params(labelsize=8)

    # Panel 2: classifier multi-label
    ax1.imshow(class_rgb, extent=extent, origin='upper', aspect='auto')
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax1.set_title(
        '6-class classifier (ft_6cls_mc11val_denoise) — multi-label\n'
        f'τ = max(inverse-proportional, {GLOBAL_MIN_THRESH:.2f});  '
        f'white = unclassified',
        fontsize=10,
    )
    ax1.set_xlabel('Longitude (°E)')
    ax1.set_ylabel('Latitude (°N)')
    ax1.tick_params(labelsize=8)

    # Legend: single-class + common mixes
    def _mix(*classes):
        c = np.zeros(3, dtype=np.float32)
        for name in classes:
            c = c + CLASS_COLORS_01[name]
        return np.clip(c, 0, 1)

    legend_handles = []
    for cname in ['olivine', 'lcp', 'hcp', 'alteration']:
        legend_handles.append(
            mpatches.Patch(facecolor=CLASS_COLORS_01[cname],
                           label=f'{cname} (τ={FINAL_THRESH[cname]:.2f})')
        )
    legend_handles.append(
        mpatches.Patch(facecolor=_mix('plagioclase'),
                       label=f'plagioclase (τ={FINAL_THRESH["plagioclase"]:.2f}; darkens)')
    )
    for a, b in [('olivine', 'hcp'), ('olivine', 'lcp'), ('lcp', 'hcp'),
                 ('olivine', 'alteration'), ('hcp', 'alteration')]:
        legend_handles.append(
            mpatches.Patch(facecolor=_mix(a, b), label=f'{a} + {b}')
        )
    legend_handles.append(
        mpatches.Patch(facecolor=_mix('olivine', 'lcp', 'hcp'),
                       label='olivine + lcp + hcp')
    )
    for a in ['olivine', 'lcp', 'hcp']:
        legend_handles.append(
            mpatches.Patch(facecolor=_mix(a, 'plagioclase'),
                           label=f'{a} + plagioclase')
        )
    legend_handles.append(
        mpatches.Patch(facecolor='white', edgecolor='black', label='unclassified')
    )
    ax1.legend(
        handles=legend_handles, loc='lower right', fontsize=7,
        ncol=2, framealpha=0.9, edgecolor='#888',
    )

    fig.suptitle(
        f'MC13 quadrant ({len(tiles)} tiles) — MAF band parameters vs 6-class classifier',
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\nSaved figure → {OUT_PATH}')
    print(f'Figure size: {target_shape[1]} × {target_shape[0]} target grid pixels '
          f'(rendered at 150 dpi)')

    # Summary stats
    total_valid = sum(s['n_valid'] for s in stats)
    total_unclass = sum(s['unclassified'] for s in stats)
    print(f'\nTotal valid pixels across MC13: {total_valid:,}')
    print(f'Unclassified: {total_unclass:,}  '
          f'({100*total_unclass/max(total_valid,1):.1f}%)')
    print('\nPer-class totals (% of valid pixels):')
    print(f'  {"tile":<6}  {"valid":>10}  {"oli":>7}  {"lcp":>7}  {"hcp":>7}  '
          f'{"plag":>7}  {"alt":>7}  {"unclass":>8}')
    grand = {k: 0 for k in ['olivine', 'lcp', 'hcp', 'plagioclase', 'alteration', 'unclassified']}
    for s in stats:
        nv = s['n_valid']
        for k in grand:
            grand[k] += s.get(k, 0)
        print(f'  {s["tile"]:<6}  {nv:>10,d}  '
              f'{s["olivine"]:>7,d}  {s["lcp"]:>7,d}  {s["hcp"]:>7,d}  '
              f'{s["plagioclase"]:>7,d}  {s.get("alteration", 0):>7,d}  '
              f'{s["unclassified"]:>8,d}  '
              f'({100*s["unclassified"]/max(nv,1):.1f}% unclass)')
    print('\nGrand totals:')
    for k, n in grand.items():
        print(f'  {k:<12}  {n:>12,d}  ({100*n/max(total_valid,1):.1f}%)')


if __name__ == '__main__':
    main()

"""
Seamless mosaic comparison over the Nili Fossae 4-tile area (t1249, t1250,
t1321, t1322):

  Panel 1 — MAF band-parameter composite (R=OLINDEX3, G=LCPINDEX2, B=HCPINDEX2)
            from each tile's mrrsu file. Tile-pooled percentile stretch.
  Panel 2 — v3 (denoising) classifier output with two-stage thresholding:
              (a) inverse-proportional thresholds derived from val_AP
              (b) global minimum floor (default 0.7)
            Pixels passing none → unclassified (white).
            Pixels confident in multiple classes → additive RGB blend.

Probabilities loaded from /tmp/v3_nili/t*_probs.npz (produced by
classify_tile_supervised.py --save_probs).

Output: reports/v5/fig_v5_nili_seamless_classifier_vs_params.png

Usage (no args):
    conda run -n crism python scripts/plot_nili_classifier_vs_params.py
"""
from __future__ import annotations

import math
import os
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
PROBS_DIR = '/tmp/v3_nili'

# Mars 2000 geographic CRS (degrees lon/lat) — matches plot_nili_seamless.py
MARS_GEO_WKT = (
    'GEOGCS["GCS_Mars_2000",'
    'DATUM["D_Mars_2000",SPHEROID["Mars_2000_IAU_IAG",3396190,169.8944472]],'
    'PRIMEM["Reference_Meridian",0],'
    'UNIT["Degree",0.0174532925199433]]'
)
COMMON_CRS = CRS.from_wkt(MARS_GEO_WKT)

TILES_INFO = [
    {
        'tid':   't1249',
        'mrral': '/mnt/mrdr/mc13/t1249_mrral_20n073_0327_4.img',
        'mrrsu': '/mnt/mrdr/mc13/t1249_mrrsu_20n073_0327_4.img',
    },
    {
        'tid':   't1250',
        'mrral': '/mnt/mrdr/mc13/t1250_mrral_20n078_0327_4.img',
        'mrrsu': '/mnt/mrdr/mc13/t1250_mrrsu_20n078_0327_4.img',
    },
    {
        'tid':   't1321',
        'mrral': '/mnt/mrdr/mc13/t1321_mrral_25n073_0327_4.img',
        'mrrsu': '/mnt/mrdr/mc13/t1321_mrrsu_25n073_0327_4.img',
    },
    {
        'tid':   't1322',
        'mrral': '/mnt/mrdr/mc13/t1322_mrral_25n078_0327_4.img',
        'mrrsu': '/mnt/mrdr/mc13/t1322_mrrsu_25n078_0327_4.img',
    },
]

# mrrsu band indexes (1-indexed for rasterio)
BAND_OLINDEX3   = 16   # 0-indexed 15 in CLAUDE.md
BAND_LCPINDEX2  = 19
BAND_HCPINDEX2  = 20

# Class color contributions for multi-label visualization.
#   olivine / lcp / hcp use additive RGB (primary colors).
#   plagioclase uses a SUBTRACTIVE contribution — it darkens whatever base
#   colors are already present. This avoids the yellow-collision problem
#   between olivine+hcp (additive yellow) and a yellow plagioclase color.
# Drop 'other' from the visualization (residual catch-all).
CLASS_NAMES = ['olivine', 'lcp', 'hcp', 'plagioclase']
CLASS_COLORS_01 = {
    'olivine':     np.array([ 0.85,  0.00,  0.00], dtype=np.float32),  # red
    'lcp':         np.array([ 0.00,  0.00,  0.85], dtype=np.float32),  # blue
    'hcp':         np.array([ 0.00,  0.70,  0.00], dtype=np.float32),  # green
    'plagioclase': np.array([-0.50, -0.50, -0.50], dtype=np.float32),  # darken
}

# Threshold scheme. val_AP for the best v3 denoising classifier.
VAL_AP = {
    'olivine':     0.860,
    'lcp':         0.742,
    'hcp':         0.440,
    'plagioclase': 0.093,
    'other':       0.787,
}
INVERSE_THRESH = {c: 0.99 - 0.49 * VAL_AP[c] for c in VAL_AP}
GLOBAL_MIN_THRESH = 0.8
FINAL_THRESH = {c: max(INVERSE_THRESH[c], GLOBAL_MIN_THRESH) for c in VAL_AP}

# Probability channel order in the saved npz (must match
# classify_tile_supervised.py's CLASS_NAMES)
PROB_CHANNELS = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']

OUT_PATH = os.path.join(
    PROJECT_ROOT, 'reports', 'v5',
    'fig_v5_nili_seamless_classifier_vs_params.png',
)


# ── grid utilities ────────────────────────────────────────────────────────────

def compute_target_grid(tile_paths):
    """Build a common geographic raster grid covering all tiles."""
    bounds_list = []
    src_res_list = []
    for path in tile_paths:
        with rasterio.open(path) as src:
            b = rasterio.warp.transform_bounds(src.crs, COMMON_CRS, *src.bounds)
            bounds_list.append(b)
            src_res_list.append(abs(src.transform.a))  # m/pixel

    xmin = min(b[0] for b in bounds_list)
    ymin = min(b[1] for b in bounds_list)
    xmax = max(b[2] for b in bounds_list)
    ymax = max(b[3] for b in bounds_list)

    # Convert source resolution from m to degrees at the central latitude
    src_res_m = max(src_res_list)
    deg_per_m = 1.0 / (3396190 * math.pi / 180)  # latitude degrees per meter
    target_res = src_res_m * deg_per_m

    n_x = int(math.ceil((xmax - xmin) / target_res))
    n_y = int(math.ceil((ymax - ymin) / target_res))
    target_transform = rasterio.transform.from_origin(xmin, ymax, target_res, target_res)
    return target_transform, (n_y, n_x), (xmin, ymin, xmax, ymax)


def reproject_continuous(src_path, band_idx, target_transform, target_shape):
    """Reproject one band of a source raster to the common grid. Returns float32."""
    dst = np.full(target_shape, np.nan, dtype=np.float32)
    with rasterio.open(src_path) as src:
        src_arr = src.read(band_idx).astype(np.float32)
        nodata = (src_arr == 65535) | ~np.isfinite(src_arr)
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
            resampling=Resampling.nearest,  # categorical-ish, avoid color mixing across tile edges
            src_nodata=np.nan, dst_nodata=np.nan,
        )
    return dst


# ── panel builders ────────────────────────────────────────────────────────────

def build_band_param_mosaic(tiles, target_transform, target_shape):
    """Read OLINDEX3 / LCPINDEX2 / HCPINDEX2 from each tile's mrrsu, reproject,
    combine into a single 3-channel mosaic, then percentile-stretch each
    channel using pooled valid pixels."""
    out = np.full((target_shape[0], target_shape[1], 3), np.nan, dtype=np.float32)
    for t in tiles:
        ol = reproject_continuous(t['mrrsu'], BAND_OLINDEX3,  target_transform, target_shape)
        lc = reproject_continuous(t['mrrsu'], BAND_LCPINDEX2, target_transform, target_shape)
        hc = reproject_continuous(t['mrrsu'], BAND_HCPINDEX2, target_transform, target_shape)
        for chan, src in enumerate([ol, lc, hc]):
            empty = np.isnan(out[:, :, chan])
            valid = np.isfinite(src)
            mask = empty & valid
            out[mask, chan] = src[mask]

    # Per-channel pooled percentile stretch
    rgb = np.zeros_like(out)
    for c in range(3):
        ch = out[:, :, c]
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
    """Apply per-class thresholds, additive-blend RGB across passing classes,
    reproject per tile to the common grid, and composite."""
    out = np.full((target_shape[0], target_shape[1], 3), np.nan, dtype=np.float32)

    per_tile_stats = []

    for t in tiles:
        npz = np.load(os.path.join(PROBS_DIR, f"{t['tid']}_probs.npz"))
        probs       = npz['probs'].astype(np.float32)   # (H, W, 5)
        valid_mask  = npz['valid_mask']                 # (H, W) bool
        src_t       = Affine(*[float(v) for v in npz['transform']])
        src_crs_wkt = str(npz['crs_wkt'])
        src_crs     = CRS.from_wkt(src_crs_wkt)

        H, W, _ = probs.shape
        tile_rgb = np.zeros((H, W, 3), dtype=np.float32)
        any_passed = np.zeros((H, W), dtype=bool)

        stats = {'tile': t['tid'], 'n_valid': int(valid_mask.sum())}
        for cname in CLASS_NAMES:
            ci = PROB_CHANNELS.index(cname)
            passing = (probs[:, :, ci] >= thresholds[cname]) & valid_mask
            stats[cname] = int(passing.sum())
            # Additive contribution of the class color
            tile_rgb[passing] += CLASS_COLORS_01[cname]
            any_passed |= passing
        # Clip so additive blending stays in [0, 1]
        np.clip(tile_rgb, 0.0, 1.0, out=tile_rgb)

        # Unclassified valid pixels → white
        unclass_mask = valid_mask & ~any_passed
        tile_rgb[unclass_mask] = 1.0
        # Invalid pixels (nodata) → NaN
        tile_rgb[~valid_mask] = np.nan
        stats['unclassified'] = int(unclass_mask.sum())
        per_tile_stats.append(stats)

        reproj = reproject_rgb(tile_rgb, src_t, src_crs, target_transform, target_shape)
        # Write into the mosaic where it isn't already filled by another tile
        write_mask = np.isfinite(reproj).all(axis=-1) & np.isnan(out[:, :, 0])
        out[write_mask] = reproj[write_mask]

    # Outside-coverage → light gray
    invalid = np.isnan(out).any(axis=-1)
    for c in range(3):
        out[:, :, c][invalid] = 0.85
    return out, invalid, per_tile_stats


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print('Threshold scheme:')
    for c in PROB_CHANNELS:
        used = FINAL_THRESH[c]
        inv = INVERSE_THRESH[c]
        gate = 'floor' if GLOBAL_MIN_THRESH > inv else 'inverse'
        print(f'  {c:<12} val_AP={VAL_AP[c]:.3f}  inverse={inv:.3f}  '
              f'floor={GLOBAL_MIN_THRESH:.2f}  → used={used:.3f}  ({gate})')
    print()

    print('Computing target grid …')
    target_transform, target_shape, bounds = compute_target_grid(
        [t['mrral'] for t in TILES_INFO]
    )
    print(f'  grid: {target_shape[1]} × {target_shape[0]} pixels')
    print(f'  extent: lon [{bounds[0]:.2f}, {bounds[2]:.2f}]  lat [{bounds[1]:.2f}, {bounds[3]:.2f}]')

    print('Building band-parameter mosaic …')
    band_param_rgb, _ = build_band_param_mosaic(TILES_INFO, target_transform, target_shape)

    print('Building classification mosaic …')
    class_rgb, _, stats = build_classification_mosaic(
        TILES_INFO, target_transform, target_shape, FINAL_THRESH
    )

    # ── figure ────────────────────────────────────────────────────────────────
    xmin, ymin, xmax, ymax = bounds
    lat_mid = (ymin + ymax) / 2
    cos_mid = math.cos(math.radians(lat_mid))
    aspect = (ymax - ymin) / ((xmax - xmin) * cos_mid)
    fig_w = 18
    panel_h = (fig_w / 2) * aspect

    fig, axes = plt.subplots(
        1, 2, figsize=(fig_w, panel_h + 1.5),
    )

    extent = (xmin, xmax, ymin, ymax)

    ax = axes[0]
    ax.imshow(band_param_rgb, extent=extent, origin='upper')
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_title('MAF band-parameter composite\n'
                 'R = OLINDEX3 · G = LCPINDEX2 · B = HCPINDEX2',
                 fontsize=10)
    ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
    ax.tick_params(labelsize=8)

    ax = axes[1]
    ax.imshow(class_rgb, extent=extent, origin='upper')
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_title('v3 (denoising) classifier — multi-label\n'
                 f'τ = max(inverse-proportional, {GLOBAL_MIN_THRESH:.2f}); '
                 f'white = unclassified',
                 fontsize=10)
    ax.set_xlabel('Longitude (°E)')
    ax.tick_params(labelsize=8, labelleft=False)

    # Build legend with single-class + mixed examples. plagioclase is
    # subtractive: it darkens whatever color is present, so its single-class
    # appearance is black, and its 2-class mixes are darkened primaries.
    def _mix(*classes):
        c = np.zeros(3, dtype=np.float32)
        for name in classes:
            c = c + CLASS_COLORS_01[name]
        return np.clip(c, 0, 1)

    legend_handles = []
    for cname in ['olivine', 'lcp', 'hcp']:
        legend_handles.append(
            mpatches.Patch(facecolor=CLASS_COLORS_01[cname],
                           label=f'{cname} (τ={FINAL_THRESH[cname]:.2f})')
        )
    legend_handles.append(
        mpatches.Patch(facecolor=_mix('plagioclase'),
                       label=f'plagioclase (τ={FINAL_THRESH["plagioclase"]:.2f}; darkens)')
    )
    # 2-way additive mixes among the three primaries
    for a, b in [('olivine', 'hcp'), ('olivine', 'lcp'), ('lcp', 'hcp')]:
        legend_handles.append(mpatches.Patch(facecolor=_mix(a, b),
                                             label=f'{a} + {b}'))
    legend_handles.append(mpatches.Patch(facecolor=_mix('olivine', 'lcp', 'hcp'),
                                         label='olivine + lcp + hcp'))
    # plagioclase darkening examples
    for a in ['olivine', 'lcp', 'hcp']:
        legend_handles.append(mpatches.Patch(facecolor=_mix(a, 'plagioclase'),
                                             label=f'{a} + plagioclase'))
    legend_handles.append(
        mpatches.Patch(facecolor='white', edgecolor='black', label='unclassified')
    )
    axes[1].legend(
        handles=legend_handles, loc='lower right', fontsize=7,
        ncol=2, framealpha=0.9, edgecolor='#888',
    )

    fig.suptitle(
        'Nili Fossae 2×2 mosaic — MAF band parameters vs v3 classifier output',
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\nSaved → {OUT_PATH}')

    # Per-tile detection summary
    print('\nPer-tile detection counts (under τ = max(inverse, floor)):')
    print(f'  {"tile":<6}  {"valid":>10}  {"oli":>7}  {"lcp":>7}  {"hcp":>7}  {"plag":>7}  {"unclass":>8}')
    for s in stats:
        nv = s['n_valid']
        print(f'  {s["tile"]:<6}  {nv:>10,d}  '
              f'{s["olivine"]:>7,d}  {s["lcp"]:>7,d}  {s["hcp"]:>7,d}  '
              f'{s["plagioclase"]:>7,d}  {s["unclassified"]:>8,d}'
              f'    ({100*s["unclassified"]/nv:.1f}% unclassified)')


if __name__ == '__main__':
    main()

"""
Three basemap versions of the seamless Nili Fossae mineral map.

Each version tiles the four Nili Fossae observations (T1249/T1250/T1321/T1322)
into a seamless mosaic in a common Mars geographic CRS, overlays predicted
mineral polygons, and saves an output figure.

Outputs:
  reports/fig_nili_seamless_crism_fc.png     — CRISM mrral false color
                                               R=2529 nm, G=1506 nm, B=1079 nm
  reports/fig_nili_seamless_themis.png       — THEMIS Day-IR 100 m/px (grayscale)
  reports/fig_nili_seamless_crism_params.png — CRISM mrrsu band parameters
                                               R=OLINDEX3, G=LCPINDEX2, B=HCPINDEX2

Mineral/tier filter (same as plot_nili_seamless.py):
  olivine     — all tiers (1–5)
  lcp         — tiers 4–5
  hcp         — all tiers (1–5)
  plagioclase — tier 5 only
  other       — not shown

Usage:
    conda run -n crism python scripts/plot_nili_basemaps.py
"""
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.warp
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds as window_from_bounds
from pyproj import CRS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fig_style import MINERAL_COLORS, DPI

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DIR = os.path.join(PROJ, 'data', 'vector')

TILES = [
    {
        'mrral': '/Volumes/Mars_GIS/CRISM/MRDR/mc13/t1249_mrral_20n073_0327_4.img',
        'mrrsu': '/Volumes/Mars_GIS/CRISM/MRDR/mc13/t1249_mrrsu_20n073_0327_4.img',
        'gpkg':  os.path.join(VECTOR_DIR, 't1249_mrral_20n073_0327_4_mineral_map.gpkg'),
    },
    {
        'mrral': '/Volumes/Mars_GIS/CRISM/MRDR/mc13/t1250_mrral_20n078_0327_4.img',
        'mrrsu': '/Volumes/Mars_GIS/CRISM/MRDR/mc13/t1250_mrrsu_20n078_0327_4.img',
        'gpkg':  os.path.join(VECTOR_DIR, 't1250_mrral_20n078_0327_4_mineral_map.gpkg'),
    },
    {
        'mrral': '/Volumes/Mars_GIS/CRISM/MRDR/mc13/t1321_mrral_25n073_0327_4.img',
        'mrrsu': '/Volumes/Mars_GIS/CRISM/MRDR/mc13/t1321_mrrsu_25n073_0327_4.img',
        'gpkg':  os.path.join(VECTOR_DIR, 't1321_mrral_25n073_0327_4_mineral_map.gpkg'),
    },
    {
        'mrral': '/Volumes/Mars_GIS/CRISM/MRDR/mc13/t1322_mrral_25n078_0327_4.img',
        'mrrsu': '/Volumes/Mars_GIS/CRISM/MRDR/mc13/t1322_mrrsu_25n078_0327_4.img',
        'gpkg':  os.path.join(VECTOR_DIR, 't1322_mrral_25n078_0327_4_mineral_map.gpkg'),
    },
]

THEMIS_PATH = (
    '/Volumes/phillipsm/Mars_GIS_Data/THEMIS/global/'
    'Mars_MO_THEMIS-IR-Day_mosaic_global_100m_v12.tif'
)

# CRISM mrral false color: 1-based band indices
# R=2529 nm (band 60), G=1506 nm (band 34), B=1079 nm (band 21)
CRISM_FC_BANDS = (60, 34, 21)

# CRISM mrrsu band parameters: 1-based
# R=OLINDEX3 (16), G=LCPINDEX2 (19), B=HCPINDEX2 (20)
CRISM_PARAM_BANDS = (16, 19, 20)

CRISM_NODATA = 65535.0

# Mineral display filter
SHOW = {
    'olivine':     [1, 2, 3, 4, 5],
    'lcp':         [4, 5],
    'hcp':         [1, 2, 3, 4, 5],
    'plagioclase': [5],
    'other':       [],
}
TIER_ALPHA = {1: 0.25, 2: 0.42, 3: 0.58, 4: 0.75, 5: 0.90}
RENDER_ORDER = ['hcp', 'lcp', 'plagioclase', 'olivine']

# Common CRS: Mars 2000 geographic (degrees)
MARS_GEO_WKT = (
    'GEOGCS["GCS_Mars_2000",'
    'DATUM["D_Mars_2000",SPHEROID["Mars_2000_IAU_IAG",3396190,169.8944472]],'
    'PRIMEM["Reference_Meridian",0],'
    'UNIT["Degree",0.0174532925199433]]'
)
COMMON_CRS = CRS.from_wkt(MARS_GEO_WKT)

# Target mosaic dimensions (pixels) — ~250 m/px at 22.5°N
TARGET_WIDTH  = 2000
TARGET_HEIGHT = 2000


# ── extent helpers ─────────────────────────────────────────────────────────────

def get_combined_extent():
    """Return (xmin, ymin, xmax, ymax) in COMMON_CRS covering all tiles."""
    all_bounds = []
    for tile in TILES:
        with rasterio.open(tile['mrral']) as src:
            b = rasterio.warp.transform_bounds(src.crs, COMMON_CRS, *src.bounds)
        all_bounds.append(b)
    return (
        min(b[0] for b in all_bounds),
        min(b[1] for b in all_bounds),
        max(b[2] for b in all_bounds),
        max(b[3] for b in all_bounds),
    )


def make_target_transform(xmin, ymin, xmax, ymax, width, height):
    return transform_from_bounds(xmin, ymin, xmax, ymax, width, height)


# ── basemap builders ───────────────────────────────────────────────────────────

def build_crism_mosaic(img_key, band_indices, target_transform, extent):
    """Mosaic CRISM tiles into (len(band_indices), H, W) float32 in COMMON_CRS."""
    xmin, ymin, xmax, ymax = extent
    n_bands = len(band_indices)
    mosaic = np.full((n_bands, TARGET_HEIGHT, TARGET_WIDTH), np.nan, dtype=np.float32)

    for tile in TILES:
        src_path = tile[img_key]
        with rasterio.open(src_path) as src:
            for bi, band_idx in enumerate(band_indices):
                raw = src.read(band_idx).astype(np.float32)
                raw[raw == CRISM_NODATA] = np.nan
                raw[np.abs(raw) > 1] = np.nan

                dst_band = np.full((TARGET_HEIGHT, TARGET_WIDTH), np.nan, dtype=np.float32)
                reproject(
                    source=raw,
                    destination=dst_band,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=target_transform,
                    dst_crs=COMMON_CRS,
                    resampling=Resampling.bilinear,
                    src_nodata=np.nan,
                    dst_nodata=np.nan,
                )
                valid = np.isfinite(dst_band)
                mosaic[bi][valid] = dst_band[valid]

    return mosaic


def percentile_stretch(arr, lo=5, hi=98):
    """Stretch array to [0, 1] using percentile clipping; NaN-safe."""
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return np.zeros_like(arr)
    vmin, vmax = np.nanpercentile(arr, lo), np.nanpercentile(arr, hi)
    if vmax == vmin:
        return np.zeros_like(arr)
    out = (arr - vmin) / (vmax - vmin)
    return np.clip(out, 0, 1)


def mosaic_to_rgb(mosaic):
    """Convert (3, H, W) float mosaic to (H, W, 4) RGBA with NaN → transparent."""
    r = percentile_stretch(mosaic[0])
    g = percentile_stretch(mosaic[1])
    b = percentile_stretch(mosaic[2])
    alpha = np.where(
        np.isfinite(mosaic[0]) & np.isfinite(mosaic[1]) & np.isfinite(mosaic[2]),
        1.0, 0.0,
    ).astype(np.float32)
    return np.stack([r, g, b, alpha], axis=-1)


def build_themis_basemap(target_transform, extent):
    """Read THEMIS Day IR at target extent using a windowed read; return (H, W) uint8."""
    xmin, ymin, xmax, ymax = extent
    dst = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.float32)

    with rasterio.open(THEMIS_PATH) as src:
        # Transform our geographic bounds to THEMIS CRS
        themis_bounds = rasterio.warp.transform_bounds(
            COMMON_CRS, src.crs, xmin, ymin, xmax, ymax
        )
        window = window_from_bounds(*themis_bounds, src.transform)
        raw = src.read(1, window=window, boundless=True, fill_value=0).astype(np.float32)
        raw_transform = src.window_transform(window)

        reproject(
            source=raw,
            destination=dst,
            src_transform=raw_transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=COMMON_CRS,
            resampling=Resampling.bilinear,
        )

    return dst


# ── mineral overlay ────────────────────────────────────────────────────────────

def overlay_minerals(ax):
    """Plot filtered mineral polygons on ax (all tiles, common CRS)."""
    for mineral in RENDER_ORDER:
        tiers_to_show = SHOW.get(mineral, [])
        if not tiers_to_show:
            continue
        color = MINERAL_COLORS[mineral]
        for tile in TILES:
            try:
                gdf = gpd.read_file(tile['gpkg'], layer=mineral)
            except Exception:
                continue
            if gdf.empty:
                continue
            gdf = gdf.to_crs(COMMON_CRS)
            for tier in sorted(tiers_to_show):
                subset = gdf[gdf['confidence'] == tier]
                if subset.empty:
                    continue
                subset.plot(ax=ax, color=color, edgecolor='none', alpha=TIER_ALPHA[tier])


def add_legend(ax):
    mineral_handles = [
        mpatches.Patch(facecolor=MINERAL_COLORS[m], label=m.capitalize())
        for m in RENDER_ORDER if SHOW.get(m)
    ]
    tier_handles = [
        mpatches.Patch(facecolor='#888', alpha=TIER_ALPHA[t], label=f'Tier {t}')
        for t in sorted(TIER_ALPHA)
    ]
    ax.legend(handles=mineral_handles + tier_handles,
              loc='lower right', fontsize=7, ncol=2,
              framealpha=0.75, edgecolor='none')


def setup_ax(ax, extent):
    xmin, ymin, xmax, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def fig_size(extent):
    xmin, ymin, xmax, ymax = extent
    cos_mid = math.cos(math.radians((ymin + ymax) / 2))
    w = 9
    h = w * ((ymax - ymin) / ((xmax - xmin) * cos_mid))
    return w, h


# ── figure generators ──────────────────────────────────────────────────────────

def make_crism_fc_figure(extent, target_transform):
    print('Building CRISM false-color mosaic ...')
    mosaic = build_crism_mosaic('mrral', CRISM_FC_BANDS, target_transform, extent)
    rgb = mosaic_to_rgb(mosaic)

    fig, ax = plt.subplots(figsize=fig_size(extent), constrained_layout=True)
    setup_ax(ax, extent)
    xmin, ymin, xmax, ymax = extent
    ax.imshow(rgb, extent=[xmin, xmax, ymin, ymax], origin='upper', aspect='auto')
    overlay_minerals(ax)
    add_legend(ax)
    ax.set_title(
        'Nili Fossae — CRISM False Color  (R=2529 nm, G=1506 nm, B=1079 nm)\n'
        'olivine all, lcp t4–5, hcp all, plag t5',
        fontsize=9, pad=5,
    )
    out = os.path.join(PROJ, 'reports', 'fig_nili_seamless_crism_fc.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out}')


def make_themis_figure(extent, target_transform):
    print('Building THEMIS Day-IR basemap ...')
    gray = build_themis_basemap(target_transform, extent)
    gray_stretched = percentile_stretch(gray, lo=1, hi=99)

    fig, ax = plt.subplots(figsize=fig_size(extent), constrained_layout=True)
    setup_ax(ax, extent)
    xmin, ymin, xmax, ymax = extent
    ax.imshow(gray_stretched, cmap='gray', extent=[xmin, xmax, ymin, ymax],
              origin='upper', aspect='auto', vmin=0, vmax=1)
    overlay_minerals(ax)
    add_legend(ax)
    ax.set_title(
        'Nili Fossae — THEMIS Day IR (100 m/px)\n'
        'olivine all, lcp t4–5, hcp all, plag t5',
        fontsize=9, pad=5,
    )
    out = os.path.join(PROJ, 'reports', 'fig_nili_seamless_themis.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out}')


def make_crism_params_figure(extent, target_transform):
    print('Building CRISM band-parameter mosaic (OLINDEX3 / LCPINDEX2 / HCPINDEX2) ...')
    mosaic = build_crism_mosaic('mrrsu', CRISM_PARAM_BANDS, target_transform, extent)
    rgb = mosaic_to_rgb(mosaic)

    fig, ax = plt.subplots(figsize=fig_size(extent), constrained_layout=True)
    setup_ax(ax, extent)
    xmin, ymin, xmax, ymax = extent
    ax.imshow(rgb, extent=[xmin, xmax, ymin, ymax], origin='upper', aspect='auto')
    ax.set_title(
        'Nili Fossae — CRISM Band Parameters  (R=OLINDEX3, G=LCPINDEX2, B=HCPINDEX2)',
        fontsize=9, pad=5,
    )
    out = os.path.join(PROJ, 'reports', 'fig_nili_seamless_crism_params.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out}')


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print('Computing combined tile extent ...')
    extent = get_combined_extent()
    xmin, ymin, xmax, ymax = extent
    print(f'  lon [{xmin:.2f}, {xmax:.2f}]  lat [{ymin:.2f}, {ymax:.2f}]')

    target_transform = make_target_transform(xmin, ymin, xmax, ymax,
                                             TARGET_WIDTH, TARGET_HEIGHT)

    make_crism_fc_figure(extent, target_transform)
    make_themis_figure(extent, target_transform)
    make_crism_params_figure(extent, target_transform)

    print('All done.')


if __name__ == '__main__':
    main()

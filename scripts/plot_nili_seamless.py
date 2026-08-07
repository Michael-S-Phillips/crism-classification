"""
Figure: Seamless mineral map over the Nili Fossae 2×2 tile area.

All four tiles (T1249/T1250/T1321/T1322) are reprojected to a common Mars
geographic CRS and plotted on a single axes, giving the appearance of one
continuous map with no tile boundaries.

Mineral / tier filter:
    olivine     — all tiers (1–5)
    lcp         — tiers 4–5 only
    hcp         — all tiers (1–5)
    plagioclase — tier 5 only
    other       — not shown

Output: reports/fig_nili_seamless.png

Usage:
    conda run -n crism python scripts/plot_nili_seamless.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import rasterio
import rasterio.warp
from pyproj import CRS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fig_style import MINERAL_COLORS, DPI

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DIR = os.path.join(PROJ, 'data', 'vector')

TILES = [
    {
        'img':  '/Volumes/Mars_GIS/CRISM/MRDR/mc13/t1249_mrral_20n073_0327_4.img',
        'gpkg': os.path.join(VECTOR_DIR, 't1249_mrral_20n073_0327_4_mineral_map.gpkg'),
    },
    {
        'img':  '/Volumes/Mars_GIS/CRISM/MRDR/mc13/t1250_mrral_20n078_0327_4.img',
        'gpkg': os.path.join(VECTOR_DIR, 't1250_mrral_20n078_0327_4_mineral_map.gpkg'),
    },
    {
        'img':  '/Volumes/Mars_GIS/CRISM/MRDR/mc13/t1321_mrral_25n073_0327_4.img',
        'gpkg': os.path.join(VECTOR_DIR, 't1321_mrral_25n073_0327_4_mineral_map.gpkg'),
    },
    {
        'img':  '/Volumes/Mars_GIS/CRISM/MRDR/mc13/t1322_mrral_25n078_0327_4.img',
        'gpkg': os.path.join(VECTOR_DIR, 't1322_mrral_25n078_0327_4_mineral_map.gpkg'),
    },
]

# Mineral → which tiers to show (empty list = skip entirely)
SHOW = {
    'olivine':     [1, 2, 3, 4, 5],
    'lcp':         [4, 5],
    'hcp':         [1, 2, 3, 4, 5],
    'plagioclase': [5],
    'other':       [],
}

TIER_ALPHA = {1: 0.25, 2: 0.42, 3: 0.58, 4: 0.75, 5: 0.90}

# Render order: less-dominant minerals first so higher-confidence ones sit on top
RENDER_ORDER = ['other', 'hcp', 'lcp', 'plagioclase', 'olivine']

# Common CRS: Mars 2000 geographic (degrees lon/lat)
MARS_GEO_WKT = (
    'GEOGCS["GCS_Mars_2000",'
    'DATUM["D_Mars_2000",SPHEROID["Mars_2000_IAU_IAG",3396190,169.8944472]],'
    'PRIMEM["Reference_Meridian",0],'
    'UNIT["Degree",0.0174532925199433]]'
)
COMMON_CRS = CRS.from_wkt(MARS_GEO_WKT)


def tile_bounds_geo(img_path):
    """Return (left, bottom, right, top) of the tile in COMMON_CRS degrees."""
    with rasterio.open(img_path) as src:
        left, bottom, right, top = rasterio.warp.transform_bounds(
            src.crs, COMMON_CRS, *src.bounds
        )
    return left, bottom, right, top


def main():
    # Compute combined extent across all tiles
    all_bounds = [tile_bounds_geo(t['img']) for t in TILES]
    xmin = min(b[0] for b in all_bounds)
    ymin = min(b[1] for b in all_bounds)
    xmax = max(b[2] for b in all_bounds)
    ymax = max(b[3] for b in all_bounds)
    print(f'Combined extent: lon [{xmin:.2f}, {xmax:.2f}]  lat [{ymin:.2f}, {ymax:.2f}]')

    # Aspect ratio: degrees lon / degrees lat (at ~22.5°N, cos correction ≈ 0.927)
    import math
    lon_span = xmax - xmin
    lat_span = ymax - ymin
    cos_mid = math.cos(math.radians((ymin + ymax) / 2))
    fig_w = 10
    fig_h = fig_w * (lat_span / (lon_span * cos_mid))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    ax.set_facecolor('#d8d8d8')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Plot minerals in render order, tiers 1→5 (lower confidence painted first)
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
                subset.plot(
                    ax=ax,
                    color=color,
                    edgecolor='none',
                    alpha=TIER_ALPHA[tier],
                )

    # Legend
    mineral_handles = [
        mpatches.Patch(facecolor=MINERAL_COLORS[m], label=m.capitalize())
        for m in RENDER_ORDER if SHOW.get(m)
    ]
    tier_handles = [
        mpatches.Patch(facecolor='#888', alpha=TIER_ALPHA[t], label=f'Tier {t}')
        for t in sorted(TIER_ALPHA)
    ]
    ax.legend(
        handles=mineral_handles + tier_handles,
        loc='lower right', fontsize=8, ncol=2,
        framealpha=0.85, edgecolor='none',
    )

    ax.set_title(
        'Nili Fossae — Predicted Minerals  '
        '(olivine all, lcp t4–5, hcp all, plag t5)',
        fontsize=10, pad=6,
    )

    out = os.path.join(PROJ, 'reports', 'fig_nili_seamless.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()

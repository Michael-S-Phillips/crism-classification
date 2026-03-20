"""
Figure: Vectroscopy vector mineral maps for T0435 and T0434.

4×2 grid (rows = minerals, columns = tiles). Each panel shows polygons
coloured by confidence tier (1=low, 2=medium, 3=high) over a grey
background showing the tile valid-mask extent.

Output: reports/fig_vector_mineral_maps.png

Usage:
    conda run -n crism python scripts/plot_vector_mineral_maps.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import rasterio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fig_style import MINERAL_COLORS, DPI, apply_style

PROJ      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DIR = os.path.join(PROJ, 'data', 'vector')
REPORTS   = os.path.join(PROJ, 'reports')
os.makedirs(REPORTS, exist_ok=True)

TILES = [
    {
        'id':   't0435_mrral_40s323_0327_4',
        'img':  '/mnt/mrdr/mc26/t0435_mrral_40s323_0327_4.img',
        'gpkg': os.path.join(VECTOR_DIR, 't0435_mrral_40s323_0327_4_mineral_map.gpkg'),
        'label': 'T0435',
    },
    {
        'id':   't0434_mrral_40s318_0327_4',
        'img':  '/mnt/mrdr/mc26/t0434_mrral_40s318_0327_4.img',
        'gpkg': os.path.join(VECTOR_DIR, 't0434_mrral_40s318_0327_4_mineral_map.gpkg'),
        'label': 'T0434',
    },
]

MINERALS = ['olivine', 'lcp', 'hcp', 'plagioclase']

# Confidence tier colours: light → dark (low → high)
TIER_COLORS = {
    1: (0.85, 0.85, 0.85),   # overridden per-mineral below
    2: (0.50, 0.50, 0.50),
    3: (0.20, 0.20, 0.20),
}

TIER_ALPHA = {1: 0.55, 2: 0.75, 3: 0.95}


def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


def tier_color(mineral, tier):
    """Lightness-scaled version of the mineral's base colour."""
    base = hex_to_rgb(MINERAL_COLORS[mineral])
    # tier 1 = 70% lightened, tier 2 = base, tier 3 = darkened
    factors = {1: 0.60, 2: 1.0, 3: 1.35}
    f = factors[tier]
    r, g, b = [min(1.0, c * f + (1 - f) * 0.95) if f < 1 else min(1.0, c * f) for c in base]
    return (r, g, b)


def get_valid_extent(img_path):
    """Return (xmin, xmax, ymin, ymax) of the tile in its projected CRS."""
    with rasterio.open(img_path) as src:
        b = src.bounds
        return b.left, b.right, b.bottom, b.top


def plot_panel(ax, tile, mineral):
    """Fill one subplot panel."""
    xmin, xmax, ymin, ymax = get_valid_extent(tile['img'])

    # Grey background for tile extent
    ax.set_facecolor('#e0e0e0')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    try:
        gdf = gpd.read_file(tile['gpkg'], layer=mineral)
    except Exception:
        ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                ha='center', va='center', color='#999', fontsize=8)
        return

    if gdf.empty:
        ax.text(0.5, 0.5, 'no polygons', transform=ax.transAxes,
                ha='center', va='center', color='#999', fontsize=8)
        return

    for tier in [1, 2, 3]:
        subset = gdf[gdf['confidence'] == tier]
        if subset.empty:
            continue
        subset.plot(
            ax=ax,
            color=tier_color(mineral, tier),
            edgecolor='none',
            alpha=TIER_ALPHA[tier],
        )

    # Polygon count annotation
    ax.text(0.02, 0.03, f'{len(gdf):,}', transform=ax.transAxes,
            fontsize=7, color='#333', va='bottom', ha='left',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))


def main():
    apply_style()

    n_rows = len(MINERALS)
    n_cols = len(TILES)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows),
                             constrained_layout=True)

    # Column headers (tile labels)
    for col, tile in enumerate(TILES):
        axes[0, col].set_title(tile['label'], fontsize=13, fontweight='bold', pad=6)

    # Row headers (mineral labels) + panels
    for row, mineral in enumerate(MINERALS):
        axes[row, 0].set_ylabel(mineral.upper(), fontsize=11, fontweight='bold',
                                rotation=90, labelpad=6)
        for col, tile in enumerate(TILES):
            plot_panel(axes[row, col], tile, mineral)

    # Shared confidence-tier legend
    legend_handles = [
        mpatches.Patch(facecolor='#aaaaaa', alpha=0.55, label='Tier 1 (≥33rd pctile)'),
        mpatches.Patch(facecolor='#666666', alpha=0.75, label='Tier 2 (≥67th pctile)'),
        mpatches.Patch(facecolor='#222222', alpha=0.95, label='Tier 3 (≥90th pctile)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=3, fontsize=9, framealpha=0.85,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Vectroscopy vector mineral maps — T0435 & T0434\n'
                 'SpatialSpectralClassifier (spvit_lrscale001) · global percentile thresholds',
                 fontsize=12, y=1.01)

    out = os.path.join(REPORTS, 'fig_vector_mineral_maps.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()

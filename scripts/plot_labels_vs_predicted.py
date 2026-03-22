"""
Figure: Label polygons vs. supervised vectroscopy output for T0435 and T0434.

2×2 grid: rows = tiles (T0435, T0434), columns = Labels | Predicted.
Labels come from /mnt/mrdr/categorized_mineral_units/.
Predicted comes from data/vector/ vectroscopy GeoPackages.
Colors: olivine=red, lcp=cyan, hcp=magenta, plagioclase=yellow, other=gray.
Mixed-mineral labels use an RGB blend of constituent classes.

Output: reports/fig_labels_vs_predicted.png

Usage:
    conda run -n crism python scripts/plot_labels_vs_predicted.py
"""
import os
import re
import sys
from typing import List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mc
import geopandas as gpd
import rasterio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fig_style import MINERAL_COLORS, DPI

PROJ      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL_DIR = '/mnt/mrdr/categorized_mineral_units'
VECTOR_DIR = os.path.join(PROJ, 'data', 'vector')
REPORTS   = os.path.join(PROJ, 'reports')
os.makedirs(REPORTS, exist_ok=True)

TILES = [
    {
        'label': 'T0435',
        'label_gpkg': os.path.join(LABEL_DIR, 'T0435.gpkg'),
        'label_layer': 'T0435',
        'pred_gpkg': os.path.join(VECTOR_DIR, 't0435_mrral_40s323_0327_4_mineral_map.gpkg'),
        'img': '/mnt/mrdr/mc26/t0435_mrral_40s323_0327_4.img',
    },
    {
        'label': 'T0434',
        'label_gpkg': os.path.join(LABEL_DIR, 'T0434.gpkg'),
        'label_layer': 'T0434',
        'pred_gpkg': os.path.join(VECTOR_DIR, 't0434_mrral_40s318_0327_4_mineral_map.gpkg'),
        'img': '/mnt/mrdr/mc26/t0434_mrral_40s318_0327_4.img',
    },
]

MINERALS = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']
TIER_ALPHA = {1: 0.25, 2: 0.42, 3: 0.58, 4: 0.75, 5: 0.90}
LABEL_TIER_ALPHA = {'Low': 0.40, 'Moderate': 0.65, 'High': 0.90}

# Canonical token mapping (lowercase keys)
_CANONICAL = {
    'olivine': 'olivine',
    'type 1 olivine': 'olivine',
    'type 2 olivine': 'olivine',
    'lcp': 'lcp',
    'hcp': 'hcp',
    'plagioclase': 'plagioclase',
}
_OTHER_TOKENS = {'alteration', 'red slope', 'other', 'bland', 'denom', 'uncertain'}


# ── pure helpers ──────────────────────────────────────────────────────────────

def parse_category(category: str) -> Tuple[List[str], str]:
    """Parse a Category string into (canonical_mineral_names, tier).

    Examples:
        "lcp (High)"               → (['lcp'], 'High')
        "hcp + olivine (Moderate)" → (['hcp', 'olivine'], 'Moderate')
        "red slope (Low)"          → (['other'], 'Low')
        "Other (High)"             → (['other'], 'High')
    """
    tier_match = re.search(r'\((\w+)\)\s*$', category)
    tier = tier_match.group(1) if tier_match else 'Low'
    mineral_str = category[:tier_match.start()].strip() if tier_match else category.strip()

    tokens = [t.strip() for t in mineral_str.split(' + ')]
    minerals = []
    for token in tokens:
        t_lower = token.lower()
        if t_lower in _CANONICAL:
            minerals.append(_CANONICAL[t_lower])
        elif t_lower in _OTHER_TOKENS:
            minerals.append('other')
        else:
            minerals.append('other')   # fail-safe for unknown tokens

    return minerals, tier


def blend_mineral_color(mineral_names: List[str]) -> Tuple[float, float, float]:
    """Return RGB (0-1) blend of the given minerals' MINERAL_COLORS.

    Single mineral returns its exact color. Multiple minerals return the
    component-wise average of their RGB values.
    """
    rgbs = [mc.to_rgb(MINERAL_COLORS[m]) for m in mineral_names]
    r = sum(c[0] for c in rgbs) / len(rgbs)
    g = sum(c[1] for c in rgbs) / len(rgbs)
    b = sum(c[2] for c in rgbs) / len(rgbs)
    return (r, g, b)


# ── panel renderers ───────────────────────────────────────────────────────────

def setup_ax(ax, img_path: str) -> None:
    """Set axes extent from tile raster bounds; grey background; no ticks/spines."""
    with rasterio.open(img_path) as src:
        b = src.bounds
    ax.set_facecolor('#e0e0e0')
    ax.set_xlim(b.left, b.right)
    ax.set_ylim(b.bottom, b.top)
    ax.set_aspect('equal')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_label_panel(ax, tile: dict) -> None:
    """Plot label polygons coloured by mineral class (blended for mixed labels)."""
    setup_ax(ax, tile['img'])
    try:
        gdf = gpd.read_file(tile['label_gpkg'], layer=tile['label_layer'])
    except Exception as e:
        ax.text(0.5, 0.5, f'no data\n{e}', transform=ax.transAxes,
                ha='center', va='center', color='#999', fontsize=8)
        return

    for _, row in gdf.iterrows():
        cat = row.get('Category', '')
        if not cat:
            continue
        minerals, tier = parse_category(str(cat))
        color = blend_mineral_color(minerals)
        alpha = LABEL_TIER_ALPHA.get(tier, 0.40)
        gpd.GeoDataFrame(geometry=[row.geometry], crs=gdf.crs).plot(
            ax=ax, color=[color], edgecolor='none', alpha=alpha,
        )

    ax.text(0.02, 0.03, f'{len(gdf):,}', transform=ax.transAxes,
            fontsize=7, color='#333', va='bottom',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))


def plot_predicted_panel(ax, tile: dict) -> None:
    """Plot predicted mineral polygons from vectroscopy GeoPackage."""
    setup_ax(ax, tile['img'])
    total = 0
    for mineral in MINERALS:   # render bottom-to-top: olivine first, other last
        try:
            gdf = gpd.read_file(tile['pred_gpkg'], layer=mineral)
        except Exception:
            continue
        if gdf.empty:
            continue
        for tier in [1, 2, 3]:
            subset = gdf[gdf['confidence'] == tier]
            if subset.empty:
                continue
            subset.plot(
                ax=ax,
                color=MINERAL_COLORS[mineral],
                edgecolor='none',
                alpha=TIER_ALPHA[tier],
            )
        total += len(gdf)

    ax.text(0.02, 0.03, f'{total:,}', transform=ax.transAxes,
            fontsize=7, color='#333', va='bottom',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))


# ── figure assembly ───────────────────────────────────────────────────────────

def main():
    n_rows = len(TILES)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, 4.5 * n_rows),
                             constrained_layout=True)

    for row, tile in enumerate(TILES):
        axes[row, 0].set_title(f"{tile['label']} — Labels", fontsize=11,
                               fontweight='bold', pad=4)
        axes[row, 1].set_title(f"{tile['label']} — Predicted", fontsize=11,
                               fontweight='bold', pad=4)
        plot_label_panel(axes[row, 0], tile)
        plot_predicted_panel(axes[row, 1], tile)

    # Mineral legend
    mineral_handles = [
        mpatches.Patch(facecolor=MINERAL_COLORS[m], label=m.capitalize())
        for m in MINERALS
    ]
    # Confidence tier legend
    tier_handles = [
        mpatches.Patch(facecolor='#888888', alpha=TIER_ALPHA[t], label=f'Tier {t}')
        for t in sorted(TIER_ALPHA)
    ]
    fig.legend(handles=mineral_handles + tier_handles,
               loc='lower center', ncol=8, fontsize=9,
               framealpha=0.85, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle('Label polygons vs. supervised vectroscopy — T0435 & T0434',
                 fontsize=12, y=1.01)

    out = os.path.join(REPORTS, 'fig_labels_vs_predicted.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()

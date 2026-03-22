"""
Figure: Predicted vectroscopy mineral maps for a 2×2 tile area over Nili Fossae.

Layout (north = top, east = right):
    [ 25°N 73°E | 25°N 78°E ]
    [ 20°N 73°E | 20°N 78°E ]

Each panel shows vectorized mineral polygons coloured by class with
confidence-tier opacity, exactly as in plot_labels_vs_predicted.py.

Output: reports/fig_nili_map.png

Usage:
    conda run -n crism python scripts/plot_nili_map.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fig_style import MINERAL_COLORS, DPI
from scripts.plot_labels_vs_predicted import setup_ax, plot_predicted_panel

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DIR = os.path.join(PROJ, 'data', 'vector')

# 2×2 grid arranged spatially: row 0 = north (25°N), row 1 = south (20°N)
#                               col 0 = west (73°E), col 1 = east (78°E)
GRID = [
    [
        {
            'label': '25°N 73°E (T1321)',
            'pred_gpkg': os.path.join(VECTOR_DIR, 't1321_mrral_25n073_0327_4_mineral_map.gpkg'),
            'img': '/mnt/mrdr/mc13/t1321_mrral_25n073_0327_4.img',
        },
        {
            'label': '25°N 78°E (T1322)',
            'pred_gpkg': os.path.join(VECTOR_DIR, 't1322_mrral_25n078_0327_4_mineral_map.gpkg'),
            'img': '/mnt/mrdr/mc13/t1322_mrral_25n078_0327_4.img',
        },
    ],
    [
        {
            'label': '20°N 73°E (T1249)',
            'pred_gpkg': os.path.join(VECTOR_DIR, 't1249_mrral_20n073_0327_4_mineral_map.gpkg'),
            'img': '/mnt/mrdr/mc13/t1249_mrral_20n073_0327_4.img',
        },
        {
            'label': '20°N 78°E (T1250)',
            'pred_gpkg': os.path.join(VECTOR_DIR, 't1250_mrral_20n078_0327_4_mineral_map.gpkg'),
            'img': '/mnt/mrdr/mc13/t1250_mrral_20n078_0327_4.img',
        },
    ],
]

MINERALS = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']
TIER_ALPHA = {1: 0.25, 2: 0.42, 3: 0.58, 4: 0.75, 5: 0.90}


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)
    fig.suptitle('Predicted mineral maps — Nili Fossae 2×2 (T1249/T1250/T1321/T1322)',
                 fontsize=11)

    for ri, row in enumerate(GRID):
        for ci, tile in enumerate(row):
            ax = axes[ri, ci]
            ax.set_title(tile['label'], fontsize=9, fontweight='bold', pad=3)
            plot_predicted_panel(ax, tile)

    # Mineral + tier legend
    mineral_handles = [
        mpatches.Patch(facecolor=MINERAL_COLORS[m], label=m.capitalize())
        for m in MINERALS
    ]
    tier_handles = [
        mpatches.Patch(facecolor='#888888', alpha=TIER_ALPHA[t], label=f'Tier {t}')
        for t in sorted(TIER_ALPHA)
    ]
    fig.legend(handles=mineral_handles + tier_handles,
               loc='lower center', ncol=8, fontsize=9,
               framealpha=0.85, bbox_to_anchor=(0.5, -0.03))

    out = os.path.join(PROJ, 'reports', 'fig_nili_map.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()

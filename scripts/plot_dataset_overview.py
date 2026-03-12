"""
Generate fig_dataset_overview.png — dataset composition overview.

Shows pixel counts per class per basin (Argyre / Hellas) and
the train/val/test split composition.

Usage:
    python scripts/plot_dataset_overview.py
    python scripts/plot_dataset_overview.py --out reports/fig_dataset_overview.png
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=os.path.join(PROJ, 'reports', 'fig_dataset_overview.png'))
    args = parser.parse_args()

    from config_loader import load_config
    cfg = load_config()
    parquet = os.path.join(cfg['output_dir'], 'mrral_pixels.parquet')
    print(f'Loading {parquet}...')
    df = pd.read_parquet(parquet)
    print(f'  {len(df):,} total pixels')

    # Identify source basin by tile_id:
    #   Argyre = tiles with a matching T####.gpkg in gpkg_dir
    #   Hellas = tiles extracted from the north_hellas GPKG (no individual T####.gpkg)
    # This correctly separates the two regions; soft labels (0.5) appear in BOTH
    # basins (Argyre 'hcp + olivine' polygons + Hellas 'Olivine' polygons).
    import glob as _glob
    gpkg_dir = cfg['gpkg_dir']
    argyre_tile_ids = {
        os.path.basename(g).replace('.gpkg', '').lower()
        for g in _glob.glob(os.path.join(gpkg_dir, 'T*.gpkg'))
    }
    argyre = df[df['tile_id'].isin(argyre_tile_ids)]
    hellas = df[~df['tile_id'].isin(argyre_tile_ids)]
    label_cols = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']

    print(f'  Argyre basin: {len(argyre):,} pixels ({argyre["tile_id"].nunique()} tiles)')
    print(f'  Hellas basin: {len(hellas):,} pixels ({hellas["tile_id"].nunique()} tiles)')

    # ─── Class breakdown per basin ────────────────────────────────────────────
    # For display, we use 5-class schema (olivine = max(t1,t2))
    def class_counts(sub):
        olivine = ((sub['olivine_t1'] > 0.4) | (sub['olivine_t2'] > 0.4)).sum()
        lcp     = (sub['lcp'] > 0.4).sum()
        hcp     = (sub['hcp'] > 0.4).sum()
        plag    = (sub['plagioclase'] > 0.4).sum()
        other   = (sub['other'] > 0.4).sum()
        return olivine, lcp, hcp, plag, other

    arg_ol, arg_lcp, arg_hcp, arg_plag, arg_other = class_counts(argyre)
    hel_ol, hel_lcp, hel_hcp, hel_plag, hel_other = class_counts(hellas)

    # Argyre HCP: polygons labelled "hcp + olivine" produce olivine_t1=0.5, hcp=1.0.
    # Split Argyre HCP into mixed (co-labelled with soft olivine) vs pure.
    arg_hcp_mixed = ((argyre['hcp'] > 0.4) &
                     ((argyre['olivine_t1'] > 0.0) | (argyre['olivine_t2'] > 0.0))).sum()
    arg_hcp_pure  = ((argyre['hcp'] > 0.4) &
                     (argyre['olivine_t1'] == 0.0) & (argyre['olivine_t2'] == 0.0)).sum()

    # ─── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 6), facecolor='#0d1b2a')
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.13, wspace=0.38)

    # Colours
    C_ARGYRE   = '#4e9af1'    # blue
    C_HELLAS   = '#f4a742'    # orange
    C_TRAIN    = '#56b4e9'
    C_VAL      = '#e69f00'
    C_TEST     = '#cc79a7'
    C_MIXED    = '#9b59b6'    # purple for Olivine+HCP

    TEXT_COLOR = 'white'
    GRID_COLOR = '#2a3f5f'

    # ── Panel 1: grouped bar chart (class × basin) ────────────────────────────
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_facecolor('#0d1b2a')

    # Class display names (in display order)
    class_labels = ['Olivine', 'LCP', 'HCP\n(pure)', 'Olivine\n+HCP', 'Plagioclase', 'Other']
    arg_vals = np.array([arg_ol,  arg_lcp,  arg_hcp_pure,  arg_hcp_mixed, arg_plag,  arg_other]) / 1e3
    hel_vals = np.array([hel_ol,  hel_lcp,  hel_hcp,       0,             hel_plag,  hel_other]) / 1e3

    x = np.arange(len(class_labels))
    w = 0.38
    b1 = ax1.bar(x - w/2, arg_vals, w, color=C_ARGYRE, alpha=0.9, label='Argyre basin')
    b2 = ax1.bar(x + w/2, hel_vals, w, color=C_HELLAS, alpha=0.9, label='Hellas basin')

    # Highlight the "Olivine+HCP" bar differently
    ax1.bar((x - w/2)[3], arg_vals[3], w, color=C_MIXED, alpha=0.95,
            label='Olivine+HCP (Argyre)')

    ax1.set_xticks(x)
    ax1.set_xticklabels(class_labels, color=TEXT_COLOR, fontsize=9)
    ax1.set_ylabel('Positive pixels (thousands)', color=TEXT_COLOR, fontsize=10)
    ax1.set_title('Positive pixels per class', color=TEXT_COLOR, fontsize=11, pad=8)
    ax1.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.spines[['left', 'bottom']].set_color(GRID_COLOR)
    ax1.yaxis.grid(True, color=GRID_COLOR, linestyle='--', linewidth=0.6, alpha=0.6)
    ax1.set_axisbelow(True)

    leg = ax1.legend(fontsize=8, facecolor='#162a40', edgecolor='#2a3f5f',
                     labelcolor=TEXT_COLOR, loc='upper right')

    # Value labels on bars
    for bar in [*b1, *b2]:
        h = bar.get_height()
        if h > 1:
            ax1.text(bar.get_x() + bar.get_width()/2, h + 2,
                     f'{h:.0f}k', ha='center', va='bottom', fontsize=7, color=TEXT_COLOR)

    # ── Panel 2: split + basin composition ────────────────────────────────────
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_facecolor('#0d1b2a')

    # Stacked horizontal bars: Train / Val / Test, coloured by basin
    splits = ['train', 'val', 'test']
    arg_splits = [len(argyre[argyre['split'] == s]) / 1e3 for s in splits]
    hel_splits = [len(hellas[hellas['split'] == s]) / 1e3 for s in splits]

    y    = np.arange(3)
    h    = 0.45
    ax2.barh(y, arg_splits, h, color=C_ARGYRE, alpha=0.9, label='Argyre basin')
    ax2.barh(y, hel_splits, h, left=arg_splits, color=C_HELLAS, alpha=0.9, label='Hellas basin')

    ax2.set_yticks(y)
    ax2.set_yticklabels(['Train', 'Val', 'Test'], color=TEXT_COLOR, fontsize=10)
    ax2.set_xlabel('Pixels (thousands)', color=TEXT_COLOR, fontsize=10)
    ax2.set_title('Train / Val / Test split by basin', color=TEXT_COLOR, fontsize=11, pad=8)
    ax2.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.spines[['left', 'bottom']].set_color(GRID_COLOR)
    ax2.xaxis.grid(True, color=GRID_COLOR, linestyle='--', linewidth=0.6, alpha=0.6)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=8, facecolor='#162a40', edgecolor='#2a3f5f',
               labelcolor=TEXT_COLOR, loc='lower right')

    # Total labels per bar
    for i, (a, h_) in enumerate(zip(arg_splits, hel_splits)):
        total = a + h_
        ax2.text(total + 5, i, f'{total:.0f}k', va='center', fontsize=8, color=TEXT_COLOR)

    # ── Title ─────────────────────────────────────────────────────────────────
    total_k = len(df) / 1e3
    n_tiles = df['tile_id'].nunique()
    fig.suptitle(
        f'CRISM MRDR Dataset Overview  ·  {total_k:.0f}k pixels  ·  {n_tiles} unique tiles  '
        f'·  Argyre basin + Hellas basin',
        color=TEXT_COLOR, fontsize=12, fontweight='bold', y=0.97,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved → {args.out}')


if __name__ == '__main__':
    main()

"""
Generate fig_v5_dataset_overview.png — three-panel summary of the v5 labeled
dataset (mrral_pixels.parquet after sup-batch + Hellas + stratified split +
label collapse fix).

Usage:
    conda run -n crism python scripts/figures/fig_dataset.py
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _utils import (
    CLASS_COLORS, CRISM_LABEL_COLS, SPLIT_COLORS, TIER_COLORS,
    get_wavelengths_59, load_mrral_parquet,
)

OUT_PATH = '/mnt/mrdr/crism_classification/reports/v5/fig_v5_dataset_overview.png'


def main():
    df = load_mrral_parquet()
    print(f'rows: {len(df):,}, tiles: {df.tile_id.nunique()}')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

    # Panel 1: per-split per-class positive counts (grouped bars)
    splits = ['train', 'val', 'test']
    pos_counts = {
        s: {c: int((df[df['split'] == s][c] > 0.4).sum()) for c in CRISM_LABEL_COLS}
        for s in splits
    }
    ax = axes[0]
    x = np.arange(len(CRISM_LABEL_COLS))
    width = 0.28
    for i, s in enumerate(splits):
        vals = [pos_counts[s][c] for c in CRISM_LABEL_COLS]
        ax.bar(x + (i - 1) * width, vals, width, label=s,
               color=SPLIT_COLORS[s], edgecolor='black', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(CRISM_LABEL_COLS, rotation=15)
    ax.set_ylabel('Positive pixels (label > 0.4)')
    ax.set_title('Per-class positives across splits\n(stratified by HCP/plagioclase/LCP)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3, which='both')

    # Panel 2: confidence tier composition by split (stacked bars)
    ax = axes[1]
    tier_order = ['High', 'Moderate', 'Low']
    counts = {s: df[df['split'] == s]['confidence_tier'].value_counts().reindex(tier_order, fill_value=0).values
              for s in splits}
    bottom = np.zeros(len(splits))
    for tier in tier_order:
        vals = np.array([counts[s][tier_order.index(tier)] for s in splits])
        ax.bar(splits, vals, bottom=bottom, label=tier,
               color=TIER_COLORS[tier], edgecolor='black', linewidth=0.5)
        bottom += vals
    ax.set_ylabel('Pixels (millions)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
    ax.set_title('Confidence tier composition\nby split')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Panel 3: median spectrum per class (train only).
    # Use median rather than mean — CRISM I/F at low wavelengths is noisy
    # and occasional negative outliers wreck the mean.
    ax = axes[2]
    wls = get_wavelengths_59()
    band_cols = [f'm{i}' for i in range(59)]
    train = df[df['split'] == 'train']
    for cls in CRISM_LABEL_COLS:
        pos = train[train[cls] > 0.4]
        if len(pos) == 0:
            continue
        sub = pos.sample(min(5000, len(pos)), random_state=0)
        spec = sub[band_cols].values.astype(np.float32)
        # Mask spurious negatives and saturation before computing summary
        spec = np.where((spec < 0) | (spec > 1), np.nan, spec)
        median = np.nanmedian(spec, axis=0)
        ax.plot(wls, median, label=f'{cls} (n={len(pos):,})',
                color=CLASS_COLORS[cls], linewidth=1.8)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Median reflectance (I/F)')
    ax.set_title('Median reflectance spectrum per class\n(train split, NaN/negatives masked)')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim(wls.min() - 20, wls.max() + 20)
    ax.set_ylim(0, 0.35)

    fig.suptitle(
        f'CRISM mineral-classification dataset (v5): {len(df):,} pixels across '
        f'{df.tile_id.nunique()} tiles\n'
        f'10 sup-batch tiles + full Hellas region + stratified tile split + hard olivine labels',
        fontsize=10, y=1.02,
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {OUT_PATH}')


if __name__ == '__main__':
    main()

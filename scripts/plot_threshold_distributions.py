"""
Diagnostic figure: per-class similarity distributions with Otsu + tier thresholds.

For each of the 5 mineral classes, plots a histogram of valid-pixel probabilities
pooled from the calibration tiles, and overlays:
  - Otsu threshold (noise/signal split)
  - p50 / p67 / p90 of above-Otsu signal pixels (tier 1 / 2 / 3 boundaries)

Helps sanity-check that distributions are bimodal and that Otsu is splitting them
at the right point.

Output: reports/fig_threshold_distributions.png

Usage:
    conda run -n crism python scripts/plot_threshold_distributions.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.compute_global_thresholds import pool_valid_probs
from scripts.fig_style import DPI

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBS = [
    '/tmp/t0435_proto_pca95_probs.npz',
    '/tmp/t0434_proto_pca95_probs.npz',
]
CLASS_NAMES = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']
SIGNAL_PERCENTILES = [50, 67, 90]
TIER_COLORS = ['#43a047', '#fb8c00', '#e53935']   # tier 1 / 2 / 3


def main():
    print('Pooling probs...')
    pooled = pool_valid_probs(PROBS)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), constrained_layout=True)
    fig.suptitle('Similarity distributions — Otsu split + tier thresholds', fontsize=12)

    for ci, name in enumerate(CLASS_NAMES):
        ax = axes[ci]
        vals = pooled[ci]

        # Otsu threshold
        otsu = float(threshold_otsu(vals))
        signal = vals[vals > otsu]
        n_sig = len(signal)
        n_tot = len(vals)
        tiers = [float(np.percentile(signal, p)) for p in SIGNAL_PERCENTILES]

        # Histogram — separate bins for noise and signal to show both modes clearly
        ax.hist(vals, bins=80, color='#90a4ae', alpha=0.7, density=True, label='all pixels')

        # Otsu line
        ax.axvline(otsu, color='#212121', lw=1.5, linestyle='--',
                   label=f'Otsu {otsu:.3f}')

        # Tier threshold lines
        tier_labels = ['tier 1', 'tier 2', 'tier 3']
        for t, color, lbl in zip(tiers, TIER_COLORS, tier_labels):
            ax.axvline(t, color=color, lw=1.2, linestyle='-', label=f'{lbl} {t:.3f}')

        ax.set_title(name, fontsize=10, fontweight='bold')
        ax.set_xlabel('similarity', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6.5, loc='upper right')

        # Annotate signal fraction
        ax.text(0.03, 0.97, f'signal: {100*n_sig/n_tot:.1f}%',
                transform=ax.transAxes, fontsize=7, va='top', color='#333')

    out = os.path.join(PROJ, 'reports', 'fig_threshold_distributions.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()

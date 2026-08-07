"""
Generate fig_v5_architecture.png — block diagram of the SpatialSpectralClassifier
data flow. Designed for the methods section of the paper.

Usage:
    conda run -n crism python scripts/figures/fig_architecture.py
"""
from __future__ import annotations

import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt

OUT_PATH = '/Volumes/Mars_GIS/CRISM/MRDR/crism_classification/reports/v5/fig_v5_architecture.png'


def block(ax, x, y, w, h, label, color='#cdeaf7', alpha=1.0, fontsize=10,
          edgecolor='#1f77b4', lw=1.6, zorder=2):
    """Draw a rounded box with centered text."""
    rect = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=lw, edgecolor=edgecolor, facecolor=color, alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
            fontsize=fontsize, zorder=zorder + 1)


def arrow(ax, x0, y0, x1, y1, color='black', lw=1.5):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 7)
    ax.axis('off')

    # Input
    block(ax, 0.2, 4.5, 1.8, 1.5,
          'Input patch\n(7×7×59)\nmrral reflectance',
          color='#f0f0f0', edgecolor='gray', fontsize=10)

    arrow(ax, 2.0, 5.25, 2.9, 5.25)

    # Tokenize: each pixel → 128-d token
    block(ax, 2.9, 4.5, 1.8, 1.5,
          'Per-pixel\nlinear projection\n(59 → 128)',
          color='#fff2cc', edgecolor='#bf8b00', fontsize=10)
    ax.text(3.8, 4.2, '49 tokens + 1 CLS token, +pos. embed.',
            ha='center', va='top', fontsize=8, style='italic', color='#555')

    arrow(ax, 4.7, 5.25, 5.6, 5.25)

    # Transformer blocks
    for i in range(6):
        block(ax, 5.6 + i * 0.65, 4.65, 0.6, 1.2,
              f'Block\n{i+1}', color='#cdeaf7',
              edgecolor='#1f77b4', fontsize=8)
    ax.text(5.6 + 3 * 0.65, 4.45, 'Transformer encoder\n(6 layers, 4 heads, 128-d, pre-LN)',
            ha='center', va='top', fontsize=9, style='italic', color='#555')

    arrow(ax, 5.6 + 6 * 0.65 + 0.6, 5.25, 5.6 + 6 * 0.65 + 1.4, 5.25)

    # Center-pixel pool
    block(ax, 5.6 + 6 * 0.65 + 1.4, 4.5, 1.7, 1.5,
          'Take center\npixel token\n(position 3,3)',
          color='#e8f6e8', edgecolor='#2ca02c', fontsize=10)

    arrow(ax, 5.6 + 6 * 0.65 + 3.1, 5.25, 5.6 + 6 * 0.65 + 4.0, 5.25)

    # Head
    block(ax, 5.6 + 6 * 0.65 + 4.0, 4.5, 1.7, 1.5,
          'Classification head\n(Linear 128 → 5)\n+sigmoid',
          color='#fde2e1', edgecolor='#d62728', fontsize=10)

    # Output labels under head
    ax.text(5.6 + 6 * 0.65 + 4.85, 3.95,
            'P(olivine), P(LCP), P(HCP),\nP(plagioclase), P(other)',
            ha='center', va='top', fontsize=8.5, color='#666')

    # ── Pre-training pathway (below) ──────────────────────────────────────────
    ax.text(7, 3.1, 'Encoder is pre-trained as a masked autoencoder (SpatialSpectralMAE) on '
                    'the full unlabeled global MRDR dataset',
            ha='center', va='center', fontsize=10, color='#444', style='italic')

    # Schematic of MAE training
    block(ax, 0.6, 1.4, 2.0, 1.1, 'Random tile\npatches', color='#f0f0f0',
          edgecolor='gray', fontsize=9)
    arrow(ax, 2.6, 1.95, 3.4, 1.95)
    block(ax, 3.4, 1.4, 2.0, 1.1, 'Mask 75% of\nspatial tokens',
          color='#fff2cc', edgecolor='#bf8b00', fontsize=9)
    arrow(ax, 5.4, 1.95, 6.2, 1.95)
    block(ax, 6.2, 1.4, 2.0, 1.1, 'Encoder\n(shared with\nclassifier)',
          color='#cdeaf7', edgecolor='#1f77b4', fontsize=9)
    arrow(ax, 8.2, 1.95, 9.0, 1.95)
    block(ax, 9.0, 1.4, 2.0, 1.1, 'Light decoder\n(2 layers, 64-d)\n→ reconstruct',
          color='#e8f6e8', edgecolor='#2ca02c', fontsize=9)
    arrow(ax, 11.0, 1.95, 11.8, 1.95)
    block(ax, 11.8, 1.4, 1.9, 1.1, 'MSE loss on\nmasked pixels',
          color='#fde2e1', edgecolor='#d62728', fontsize=9)

    ax.text(7, 0.6, 'After pre-training, encoder weights are loaded into the classifier '
                    'and fine-tuned with a small learning rate (encoder_lr_scale ∈ {0.001, 0.01, 0.1})',
            ha='center', va='center', fontsize=9, color='#444', style='italic')

    # Top label
    ax.text(0.2, 6.6, 'SpatialSpectralClassifier — inference pathway',
            fontsize=13, fontweight='bold')
    ax.text(0.6, 2.7, 'SpatialSpectralMAE — pre-training pathway',
            fontsize=12, fontweight='bold')

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {OUT_PATH}')


if __name__ == '__main__':
    main()

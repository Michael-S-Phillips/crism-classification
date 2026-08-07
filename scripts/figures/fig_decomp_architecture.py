"""
Generate fig_v5_decomp_architecture.png — block diagram of the DecompSpVit
signal/noise decomposition encoder.

Usage:
    conda run -n crism python scripts/figures/fig_decomp_architecture.py
"""
from __future__ import annotations

import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt

OUT_PATH = '/Volumes/Mars_GIS/CRISM/MRDR/crism_classification/reports/v5/fig_v5_decomp_architecture.png'


def block(ax, x, y, w, h, label, color='#cdeaf7', edgecolor='#1f77b4',
          fontsize=10, lw=1.6):
    rect = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=lw, edgecolor=edgecolor, facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha='center', va='center', fontsize=fontsize)


def arrow(ax, x0, y0, x1, y1, color='black', lw=1.5, ls='-'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                linestyle=ls, shrinkA=2, shrinkB=2))


def main():
    fig, ax = plt.subplots(figsize=(13.5, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    # Input
    block(ax, 0.2, 6.0, 1.7, 1.2, 'Patch x\n(7×7×59)',
          color='#f0f0f0', edgecolor='gray')
    arrow(ax, 1.9, 6.6, 2.9, 6.6)

    # Encoder
    block(ax, 2.9, 5.5, 2.6, 2.0,
          'Shared encoder\n(6L ViT, 128-d,\nMAE-pretrained)',
          color='#cdeaf7', edgecolor='#1f77b4', fontsize=10)

    # Tokens output
    arrow(ax, 5.5, 7.0, 6.8, 7.5)
    arrow(ax, 5.5, 6.6, 6.8, 6.4)
    arrow(ax, 5.5, 6.2, 6.8, 5.3)
    arrow(ax, 5.5, 6.0, 6.8, 3.6)

    # CLS path
    block(ax, 6.8, 7.1, 1.4, 0.6, 'CLS token', color='#fff2cc', edgecolor='#bf8b00',
          fontsize=9)
    # Spatial tokens path
    block(ax, 6.8, 6.0, 1.4, 0.6, '49 spatial\ntokens', color='#fff2cc',
          edgecolor='#bf8b00', fontsize=9)
    block(ax, 6.8, 5.0, 1.4, 0.6, 'center-pixel\ntoken (3,3)',
          color='#e8f6e8', edgecolor='#2ca02c', fontsize=9)
    block(ax, 6.8, 3.3, 1.4, 0.6, '49 spatial\ntokens', color='#fff2cc',
          edgecolor='#bf8b00', fontsize=9)

    # Heads
    # Atmosphere head (from CLS)
    arrow(ax, 8.2, 7.4, 9.4, 7.4)
    block(ax, 9.4, 6.9, 2.2, 1.0, 'Atmosphere head\n(MLP 128 → 2·59)',
          color='#ffe0e0', edgecolor='#c44', fontsize=9)
    arrow(ax, 11.6, 7.4, 12.6, 7.7)
    arrow(ax, 11.6, 7.4, 12.6, 7.1)
    ax.text(12.7, 7.85, 'T_hat (B,59)', fontsize=9, va='center')
    ax.text(12.7, 7.05, 'b_hat (B,59)', fontsize=9, va='center')

    # Signal decoder
    arrow(ax, 8.2, 6.3, 9.4, 6.3)
    block(ax, 9.4, 5.8, 2.2, 1.0, 'Signal decoder\n(MLP per token)',
          color='#e8f6e8', edgecolor='#2ca02c', fontsize=9)
    arrow(ax, 11.6, 6.3, 12.6, 6.3)
    ax.text(12.7, 6.3, 's_hat (B,49,59)', fontsize=9, va='center')

    # Classifier head
    arrow(ax, 8.2, 5.3, 9.4, 5.0)
    block(ax, 9.4, 4.5, 2.2, 1.0, 'Classification head\n(Linear 128 → 5)',
          color='#fde2e1', edgecolor='#d62728', fontsize=9)
    arrow(ax, 11.6, 5.0, 12.6, 5.0)
    ax.text(12.7, 5.0, 'logits (B,5)', fontsize=9, va='center')

    # Residual decoder
    arrow(ax, 8.2, 3.6, 9.4, 3.6)
    block(ax, 9.4, 3.1, 2.2, 1.0, 'Residual decoder\n(MLP per token)',
          color='#f5e0ff', edgecolor='#a050a0', fontsize=9)
    arrow(ax, 11.6, 3.6, 12.6, 3.6)
    ax.text(12.7, 3.6, 'eps_hat (B,49,59)', fontsize=9, va='center')

    # Reconstruction box at bottom
    block(ax, 4.5, 0.6, 5.5, 1.2,
          'x_hat = T_hat · s_hat + b_hat + eps_hat\n(reconstruction loss vs x)',
          color='#fffae0', edgecolor='#998800', fontsize=11)

    # Dashed arrows from each output to the reconstruction
    arrow(ax, 13.5, 7.7, 9.7, 1.8, color='#888', lw=1.0, ls='--')
    arrow(ax, 13.5, 7.1, 9.5, 1.8, color='#888', lw=1.0, ls='--')
    arrow(ax, 13.5, 6.3, 8.5, 1.8, color='#888', lw=1.0, ls='--')
    arrow(ax, 13.5, 3.6, 7.0, 1.8, color='#888', lw=1.0, ls='--')

    # Title
    ax.text(0.2, 8.1, 'DecompSpVit — physics-informed decomposition encoder',
            fontsize=13, fontweight='bold')
    ax.text(0.2, 0.2,
            'The classifier reads the encoder embedding (not s_hat). The shared encoder is pressured by '
            'both the classification loss\nand the reconstruction loss to represent surface mineralogy, '
            'with atmospheric attenuation T, additive path radiance b, and stochastic residual ε '
            'factored out.',
            fontsize=9.5, color='#444', style='italic')

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {OUT_PATH}')


if __name__ == '__main__':
    main()

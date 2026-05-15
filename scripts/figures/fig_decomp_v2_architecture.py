"""
Generate fig_v5_decomp_v2_architecture.png — block diagram of the
DecompSpVitAdv (adversarial signal/noise decomposition) encoder.

Design notes (vs v1 figure):
- Small, consistent arrow heads (no oversized triangles)
- No regularizer side-panel; the only inline equation is the additive
  reconstruction x = s + n
- Two parallel "lanes" (signal on top, noise on bottom) make the GRL
  asymmetry visually obvious
- The gradient reversal is drawn as a distinct red marker on the
  noise → discriminator path
- Generous whitespace; no text overlaps

Usage:
    conda run -n crism python scripts/figures/fig_decomp_v2_architecture.py
"""
from __future__ import annotations

import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT_PATH = '/mnt/mrdr/crism_classification/reports/v5/fig_v5_decomp_v2_architecture.png'

# Palette — same role coding as v1 but with a dedicated red for the
# adversarial path so it pops.
COLOR_INPUT     = '#e9ecef'
COLOR_ENCODER   = '#cdeaf7'
COLOR_SIGNAL    = '#d6f0c4'
COLOR_NOISE     = '#ead9f5'
COLOR_CLS_HEAD  = '#fcd5cf'
COLOR_DISC_HEAD = '#ffd6cc'
COLOR_GRL       = '#ff6a4d'   # high-contrast red for the adversarial marker
COLOR_OUTPUT    = '#fafafa'
COLOR_RECON     = '#fff5d6'
TEXT_MUTED      = '#555'
EDGE            = '#333'

# Standardised arrow style — small, consistent, never huge.
ARROW_STYLE = '-|>,head_length=0.32,head_width=0.22'


def block(ax, x, y, w, h, label, color, fontsize=10, fontweight='normal',
          edgecolor=EDGE, lw=1.2, text_color='#111'):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.05",
        linewidth=lw, edgecolor=edgecolor, facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight, color=text_color)


def arr(ax, x0, y0, x1, y1, color='#222', lw=1.4, ls='-', alpha=1.0):
    """Thin, consistent arrow."""
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=ARROW_STYLE,
                                color=color, lw=lw,
                                linestyle=ls, alpha=alpha,
                                shrinkA=2, shrinkB=2))


def dim(ax, x, y, text, fontsize=8, color=TEXT_MUTED, ha='center'):
    ax.text(x, y, text, ha=ha, va='center', fontsize=fontsize,
            style='italic', color=color)


def main():
    fig, ax = plt.subplots(figsize=(15.5, 8.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # ─── Title ──────────────────────────────────────────────────────────────
    ax.text(0.4, 9.5,
            'DecompSpVitAdv (v2) — adversarial signal/noise decomposition',
            fontsize=14, fontweight='bold', color='#1a1a1a')
    ax.text(0.4, 9.10,
            'Two parallel embeddings off a shared encoder; gradient reversal '
            'on the noise side forces it to be class-uninformative.',
            fontsize=9.5, color=TEXT_MUTED, style='italic')

    # ─── Input ──────────────────────────────────────────────────────────────
    block(ax, 0.3, 5.3, 1.4, 1.0, 'Patch\n$x$',
          color=COLOR_INPUT, fontsize=11, fontweight='bold')
    dim(ax, 1.0, 5.05, '(B, 7, 7, 59)', fontsize=8.5)

    arr(ax, 1.7, 5.8, 2.7, 5.8)

    # ─── Shared encoder ─────────────────────────────────────────────────────
    block(ax, 2.7, 5.0, 2.4, 1.6,
          'Shared encoder\nSpatialSpectralTransformer\n6L · 4H · 128-d\n'
          '(MAE-pretrained)',
          color=COLOR_ENCODER, fontsize=9.5)
    dim(ax, 3.9, 4.75, '(B, 50, 128)', fontsize=8.5)

    # Two branches out of the encoder (signal up, noise down)
    arr(ax, 5.1, 6.2, 6.0, 7.2)   # to signal projection
    arr(ax, 5.1, 5.4, 6.0, 4.4)   # to noise projection

    # ─── Signal lane (top) ──────────────────────────────────────────────────
    # Signal projection
    block(ax, 6.0, 6.8, 1.7, 0.9, 'Signal proj.\nLinear 128→128',
          color=COLOR_SIGNAL, fontsize=9)
    dim(ax, 6.85, 6.5, 's_emb (B,49,128)', fontsize=7.5)

    arr(ax, 7.7, 7.25, 8.6, 7.25)

    # Signal decoder → s_hat
    block(ax, 8.6, 7.4, 1.9, 0.9, 'Signal decoder\nMLP 128→256→59',
          color=COLOR_SIGNAL, fontsize=8.8)
    arr(ax, 10.5, 7.85, 11.5, 7.85)
    block(ax, 11.5, 7.55, 1.4, 0.6, r'$\hat{s}$',
          color=COLOR_SIGNAL, fontsize=13, fontweight='bold')
    dim(ax, 13.05, 7.85, '(B, 49, 59)', fontsize=8, ha='left')
    dim(ax, 13.05, 7.55, 'surface reflectance', fontsize=7.5, ha='left')

    # Classifier (split off the s_emb path, downward to a head)
    arr(ax, 8.6, 7.0, 8.6, 6.45)
    block(ax, 8.6, 5.5, 1.9, 0.95,
          'Classifier\n(center-pixel s_emb)\nLinear 128→5',
          color=COLOR_CLS_HEAD, fontsize=8.8)
    arr(ax, 10.5, 5.97, 11.5, 5.97)
    block(ax, 11.5, 5.7, 1.4, 0.55, 'logits',
          color=COLOR_CLS_HEAD, fontsize=10, fontweight='bold')
    dim(ax, 13.05, 5.97, '(B, 5)', fontsize=8, ha='left')

    # ─── Noise lane (bottom) ────────────────────────────────────────────────
    # Noise projection
    block(ax, 6.0, 3.5, 1.7, 0.9, 'Noise proj.\nLinear 128→128',
          color=COLOR_NOISE, fontsize=9)
    dim(ax, 6.85, 3.2, 'n_emb (B,49,128)', fontsize=7.5)

    arr(ax, 7.7, 3.95, 8.6, 3.95)

    # Noise decoder → n_hat
    block(ax, 8.6, 3.4, 1.9, 0.9, 'Noise decoder\nMLP 128→256→59',
          color=COLOR_NOISE, fontsize=8.8)
    arr(ax, 10.5, 3.85, 11.5, 3.85)
    block(ax, 11.5, 3.55, 1.4, 0.6, r'$\hat{n}$',
          color=COLOR_NOISE, fontsize=13, fontweight='bold')
    dim(ax, 13.05, 3.85, '(B, 49, 59)', fontsize=8, ha='left')
    dim(ax, 13.05, 3.55, 'noise / residual', fontsize=7.5, ha='left')

    # GRL marker between n_emb and discriminator (drawn as a small lozenge in red)
    arr(ax, 8.6, 3.5, 8.6, 2.85)
    grl_x, grl_y = 8.6 + 1.9 / 2 - 0.5, 2.35
    grl = patches.FancyBboxPatch(
        (grl_x, grl_y), 1.0, 0.5,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.5, edgecolor=COLOR_GRL, facecolor='white',
    )
    ax.add_patch(grl)
    ax.text(grl_x + 0.5, grl_y + 0.25,
            r'$\mathrm{GRL}$ ($\times\!-\lambda_{adv}$)',
            ha='center', va='center', fontsize=8.5, color=COLOR_GRL,
            fontweight='bold')
    arr(ax, 9.1, 2.35, 9.1, 1.85, color=COLOR_GRL, lw=1.4)

    # Discriminator
    block(ax, 8.6, 0.9, 1.9, 0.95,
          'Discriminator\n(center-pixel n_emb)\nMLP 128→64→5',
          color=COLOR_DISC_HEAD, fontsize=8.8)
    arr(ax, 10.5, 1.37, 11.5, 1.37, color=COLOR_GRL, lw=1.4)
    block(ax, 11.5, 1.1, 1.4, 0.55, 'disc_logits',
          color=COLOR_DISC_HEAD, fontsize=9.5, fontweight='bold')
    dim(ax, 13.05, 1.37, '(B, 5)', fontsize=8, ha='left')

    # ─── Adversarial-loss annotation ────────────────────────────────────────
    ax.text(15.0, 1.37,
            r'$\mathcal{L}_{\mathrm{adv}}$',
            ha='left', va='center', fontsize=13, color=COLOR_GRL,
            fontweight='bold')
    # Tiny note explaining what GRL does
    ax.text(9.55, 2.6,
            'forward = identity\nbackward: grad ↦ $-\\lambda_{adv}\\cdot$grad',
            ha='left', va='center', fontsize=7.5, color=COLOR_GRL,
            style='italic')

    # ─── Classification-loss annotation ─────────────────────────────────────
    ax.text(15.0, 5.97,
            r'$\mathcal{L}_{\mathrm{cls}}$',
            ha='left', va='center', fontsize=13, color='#c44',
            fontweight='bold')

    # ─── Additive reconstruction (centered, between the lanes) ──────────────
    block(ax, 11.0, 5.05 - 0.4, 3.0, 0.85,
          r'$\hat{x} = \hat{s} + \hat{n}$',
          color=COLOR_RECON, fontsize=14, fontweight='bold')
    # Lines from s_hat and n_hat to the recon block
    arr(ax, 12.2, 7.55, 12.5, 5.5, color='#999', lw=1.0, ls='--')
    arr(ax, 12.2, 3.85, 12.5, 4.8, color='#999', lw=1.0, ls='--')
    # Recon loss arrow back toward x
    arr(ax, 11.0, 5.05, 1.7, 5.5, color='#c44', lw=1.4, ls='-')
    ax.text(5.6, 5.45, r'$\mathcal{L}_{\mathrm{recon}} = \|x - \hat{x}\|^2$',
            fontsize=10, color='#c44', fontweight='bold',
            ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='none', alpha=0.85))

    # ─── Legend at bottom-left ──────────────────────────────────────────────
    legend_handles = [
        patches.Patch(facecolor=COLOR_ENCODER, edgecolor=EDGE, label='shared encoder'),
        patches.Patch(facecolor=COLOR_SIGNAL,  edgecolor=EDGE, label='signal path'),
        patches.Patch(facecolor=COLOR_NOISE,   edgecolor=EDGE, label='noise path'),
        patches.Patch(facecolor=COLOR_CLS_HEAD,edgecolor=EDGE, label='classifier'),
        patches.Patch(facecolor=COLOR_DISC_HEAD,edgecolor=EDGE, label='discriminator'),
        patches.Patch(facecolor=COLOR_RECON,   edgecolor=EDGE, label='reconstruction'),
        Line2D([0], [0], color=COLOR_GRL, lw=1.5, label='gradient reversal'),
        Line2D([0], [0], color='#c44', lw=1.5, label='loss path'),
    ]
    ax.legend(handles=legend_handles, loc='lower left',
              bbox_to_anchor=(0.02, 0.02), fontsize=8.5,
              ncol=4, frameon=True, framealpha=0.95, edgecolor='#bbb')

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Wrote {OUT_PATH}')


if __name__ == '__main__':
    main()

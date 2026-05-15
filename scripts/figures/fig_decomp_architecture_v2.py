"""
Generate fig_v5_decomp_architecture_v2.png — a richer, more publication-quality
visualization of the DecompSpVit signal/noise decomposition encoder.

Improvements over v1:
- Larger canvas with cleaner spacing
- Color-coded by component role (encoder / heads / outputs / loss)
- Dimension annotations next to every tensor
- Inline rendering of the decomposition equation
- Sample-spectrum sketches next to s_hat and ε_hat to convey what each output looks like
- Explicit reconstruction-loop arrows that close T·s + b + ε → x_hat
- Gradient-flow arrows (dotted) from the loss back to each head

Usage:
    conda run -n crism python scripts/figures/fig_decomp_architecture_v2.py
"""
from __future__ import annotations

import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

OUT_PATH = '/mnt/mrdr/crism_classification/reports/v5/fig_v5_decomp_architecture_v2.png'

# Component color palette — distinguishes structural roles at a glance.
COLOR_INPUT     = '#e9ecef'
COLOR_ENCODER   = '#cdeaf7'
COLOR_TOKEN     = '#fff3c4'   # CLS / spatial-token annotations
COLOR_CENTER    = '#cfe9c4'   # center-pixel token gets its own color
COLOR_ATMOS     = '#ffd9d9'   # atmosphere head (T, b)
COLOR_SIGNAL    = '#d6f0c4'   # signal decoder
COLOR_RESIDUAL  = '#ead9f5'   # residual decoder
COLOR_CLS       = '#fcd5cf'   # classification head
COLOR_RECON     = '#fff5d6'   # reconstruction node
COLOR_LOSS      = '#fde5e5'   # loss components
EDGE_LIGHT      = '#7d7d7d'
TEXT_MUTED      = '#555'


def block(ax, x, y, w, h, label, color, edgecolor='#333', fontsize=10,
          lw=1.3, fontweight='normal', text_color='#222'):
    rect = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.06",
        linewidth=lw, edgecolor=edgecolor, facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight, color=text_color)


def arrow(ax, x0, y0, x1, y1, color='#222', lw=1.6, ls='-', alpha=1.0,
          headwidth=8, headlength=10):
    style = (f'-|>,head_length={headlength/3:.1f},'
             f'head_width={headwidth/3:.1f}')
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                linestyle=ls, alpha=alpha,
                                shrinkA=2, shrinkB=2))


def dim_label(ax, x, y, text, fontsize=8, color=TEXT_MUTED, ha='center'):
    """Italicised dimensional annotation."""
    ax.text(x, y, text, ha=ha, va='center', fontsize=fontsize,
            style='italic', color=color)


def draw_mini_spectrum(ax, x0, y0, w, h, kind='signal', color='#222'):
    """Draw a tiny inline spectrum sketch to indicate what s_hat or eps_hat
    look like. kind ∈ {'signal', 'residual', 'noisy_input'}."""
    rng = np.random.default_rng(7 if kind == 'signal' else 13)
    n = 30
    xs = np.linspace(x0 + w * 0.08, x0 + w * 0.92, n)
    if kind == 'signal':
        # Smooth, characteristic mineral spectrum with absorption near 1µm
        t = np.linspace(0, 1, n)
        ys = 0.55 + 0.32 * t - 0.18 * np.exp(-((t - 0.45) / 0.10) ** 2)
        ys = y0 + h * (0.15 + 0.65 * ys)
    elif kind == 'residual':
        # High-frequency noise
        ys = y0 + h * (0.5 + 0.25 * rng.standard_normal(n) * 0.4)
    else:  # noisy_input
        t = np.linspace(0, 1, n)
        signal = 0.55 + 0.32 * t - 0.18 * np.exp(-((t - 0.45) / 0.10) ** 2)
        ys = y0 + h * (0.15 + 0.65 * (signal + 0.04 * rng.standard_normal(n)))
    ax.plot(xs, ys, color=color, linewidth=1.0)


def main():
    fig, ax = plt.subplots(figsize=(16, 9.5))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis('off')

    # ───── Section title ──────────────────────────────────────────────────────
    ax.text(0.4, 10.5, 'DecompSpVit — physics-informed signal/noise decomposition encoder',
            fontsize=15, fontweight='bold', color='#1a1a1a')
    ax.text(0.4, 10.05,
            'I/F = T(λ) · s(λ, r, c) + b(λ) + ε(r, c, λ) — patch-level atmospheric '
            'terms (T, b) + per-pixel signal and residual',
            fontsize=10, color=TEXT_MUTED, style='italic')

    # ───── Input ──────────────────────────────────────────────────────────────
    block(ax, 0.3, 6.5, 1.7, 1.3, 'Input patch\n$x$',
          color=COLOR_INPUT, fontsize=11, fontweight='bold')
    dim_label(ax, 1.15, 6.2, '(B, 7, 7, 59)', fontsize=9)
    # Tiny noisy-input sketch above the input box
    draw_mini_spectrum(ax, 0.35, 7.85, 1.6, 0.7, kind='noisy_input', color='#444')
    ax.text(1.15, 8.6, 'noisy I/F', ha='center', fontsize=8, color=TEXT_MUTED, style='italic')

    arrow(ax, 2.0, 7.15, 3.0, 7.15)

    # ───── Encoder ────────────────────────────────────────────────────────────
    block(ax, 3.0, 6.0, 2.8, 2.3,
          'Shared encoder\n(SpatialSpectralTransformer)\n6 layers · 4 heads · 128-d\n'
          'MAE-pretrained, fine-tuned',
          color=COLOR_ENCODER, fontsize=10)
    dim_label(ax, 4.4, 5.7, '(B, 50, 128)\nCLS + 49 spatial tokens', fontsize=8)

    # Three branches out of the encoder
    arrow(ax, 5.8, 7.85, 7.0, 8.55, lw=1.3)
    arrow(ax, 5.8, 7.15, 7.0, 7.15, lw=1.3)
    arrow(ax, 5.8, 6.55, 7.0, 5.85, lw=1.3)

    # ───── Token compartments ────────────────────────────────────────────────
    block(ax, 7.0, 8.25, 1.7, 0.7, 'CLS token',
          color=COLOR_TOKEN, fontsize=10)
    dim_label(ax, 7.85, 7.97, '(B, 128)', fontsize=8)

    block(ax, 7.0, 6.85, 1.7, 0.7, '49 spatial tokens',
          color=COLOR_TOKEN, fontsize=10)
    dim_label(ax, 7.85, 6.55, '(B, 49, 128)', fontsize=8)

    block(ax, 7.0, 5.5, 1.7, 0.7, 'center-pixel\ntoken (3, 3)',
          color=COLOR_CENTER, fontsize=9, fontweight='bold')
    dim_label(ax, 7.85, 5.2, '(B, 128)', fontsize=8)

    # ───── Heads ─────────────────────────────────────────────────────────────
    # Atmosphere head (CLS → T, b)
    arrow(ax, 8.7, 8.6, 10.0, 8.6, lw=1.4)
    block(ax, 10.0, 8.05, 2.3, 1.1,
          'Atmosphere head\nMLP 128 → 2·59',
          color=COLOR_ATMOS, fontsize=9)
    arrow(ax, 12.3, 8.85, 13.4, 9.25, lw=1.3)
    arrow(ax, 12.3, 8.35, 13.4, 7.85, lw=1.3)
    block(ax, 13.4, 9.05, 1.7, 0.55, r'$\hat{T}$',
          color=COLOR_ATMOS, fontsize=12, fontweight='bold')
    dim_label(ax, 15.18, 9.32, '(B, 59)\n∈ [0.3, 1.0]', fontsize=7.5, ha='left')
    block(ax, 13.4, 7.6, 1.7, 0.55, r'$\hat{b}$',
          color=COLOR_ATMOS, fontsize=12, fontweight='bold')
    dim_label(ax, 15.18, 7.87, '(B, 59)\nunbounded', fontsize=7.5, ha='left')

    # Signal decoder (49 tokens → s_hat)
    arrow(ax, 8.7, 7.2, 10.0, 6.85, lw=1.4)
    block(ax, 10.0, 6.3, 2.3, 1.1,
          'Signal decoder\nMLP per token\n128 → 256 → 59',
          color=COLOR_SIGNAL, fontsize=9)
    arrow(ax, 12.3, 6.85, 13.4, 6.55, lw=1.3)
    block(ax, 13.4, 6.25, 1.7, 0.6, r'$\hat{s}$',
          color=COLOR_SIGNAL, fontsize=12, fontweight='bold')
    dim_label(ax, 15.18, 6.55, '(B, 49, 59)\nsurface reflectance', fontsize=7.5, ha='left')
    # tiny signal sketch next to the s_hat box
    draw_mini_spectrum(ax, 13.4, 5.5, 1.7, 0.55, kind='signal', color='#2c7a2c')

    # Classifier head (center token → logits)
    arrow(ax, 8.7, 5.85, 10.0, 4.9, lw=1.4)
    block(ax, 10.0, 4.45, 2.3, 0.9,
          'Classification head\nLinear 128 → 5',
          color=COLOR_CLS, fontsize=9)
    arrow(ax, 12.3, 4.9, 13.4, 4.9, lw=1.3)
    block(ax, 13.4, 4.6, 1.7, 0.6, 'logits',
          color=COLOR_CLS, fontsize=11, fontweight='bold')
    dim_label(ax, 15.18, 4.92, '(B, 5)\nP(olivine | LCP | HCP | plag | other)',
              fontsize=7.5, ha='left')

    # Residual decoder
    arrow(ax, 8.7, 6.9, 10.0, 3.4, lw=1.4, alpha=0.85)
    block(ax, 10.0, 2.85, 2.3, 1.1,
          'Residual decoder\nMLP per token\n128 → 256 → 59',
          color=COLOR_RESIDUAL, fontsize=9)
    arrow(ax, 12.3, 3.4, 13.4, 3.4, lw=1.3)
    block(ax, 13.4, 3.1, 1.7, 0.6, r'$\hat{\varepsilon}$',
          color=COLOR_RESIDUAL, fontsize=12, fontweight='bold')
    dim_label(ax, 15.18, 3.4, '(B, 49, 59)\nstochastic residual', fontsize=7.5, ha='left')

    # ───── Reconstruction node ───────────────────────────────────────────────
    block(ax, 3.5, 0.65, 7.0, 1.5,
          r'$\hat{x} = \hat{T}\cdot \hat{s} + \hat{b} + \hat{\varepsilon}$',
          color=COLOR_RECON, fontsize=15, fontweight='bold')
    ax.text(7.0, 0.35,
            'reconstruction = patch-level atmosphere · per-pixel signal + per-pixel residual',
            ha='center', va='center', fontsize=8.5,
            color=TEXT_MUTED, style='italic')

    # Dashed arrows from each output into the reconstruction node
    for src_x, src_y in [(13.5, 9.05),  # T
                          (13.5, 7.6),   # b
                          (13.5, 6.25),  # s
                          (13.5, 3.1)]:  # eps
        arrow(ax, src_x, src_y, 10.5, 1.55,
              color='#9a9a9a', lw=1.0, ls='--', alpha=0.85)

    # Reconstruction arrow back to the input for the MSE
    arrow(ax, 3.5, 1.4, 1.15, 6.5, color='#c44', lw=1.5, ls='-', alpha=0.85)
    ax.text(0.55, 3.85,
            r'$\mathcal{L}_{\mathrm{recon}} = \|x - \hat{x}\|^2$',
            ha='center', fontsize=10, color='#c44', fontweight='bold')

    # ───── Classification loss back to logits ────────────────────────────────
    block(ax, 13.0, 1.05, 2.5, 0.65,
          r'$\mathcal{L}_{\mathrm{cls}}$ = ASL(class weights)',
          color=COLOR_LOSS, fontsize=9)
    arrow(ax, 14.25, 1.7, 14.25, 4.6, color='#c44', lw=1.3, ls='-', alpha=0.7)

    # ───── Regularizer call-outs (small panel bottom-right) ──────────────────
    ax.text(13.0, 2.45, 'Regularizers (anti-trivial-solution):',
            fontsize=8.5, color=TEXT_MUTED, fontweight='bold')
    reg_lines = [
        r'  $\lambda_\varepsilon \cdot \|\hat{\varepsilon}\|^2$         '
        '— keep residual small',
        r'  $\lambda_T \cdot \|\hat{T} - 1\|^2$  — atm. prior',
        r'  $\lambda_b \cdot \|\hat{b}\|^2$        — path-radiance prior',
        r'  $\lambda_{\mathrm{sm}} \cdot \mathrm{TV}(\hat{s})$ — spatial signal smoothness',
    ]
    for i, line in enumerate(reg_lines):
        ax.text(13.0, 2.18 - i * 0.22, line, fontsize=7.5, color=TEXT_MUTED)

    # ───── Legend ─────────────────────────────────────────────────────────────
    legend_handles = [
        patches.Patch(facecolor=COLOR_ENCODER, edgecolor='#333', label='shared encoder'),
        patches.Patch(facecolor=COLOR_TOKEN, edgecolor='#333', label='token routing'),
        patches.Patch(facecolor=COLOR_ATMOS, edgecolor='#333', label='atmosphere'),
        patches.Patch(facecolor=COLOR_SIGNAL, edgecolor='#333', label='signal'),
        patches.Patch(facecolor=COLOR_RESIDUAL, edgecolor='#333', label='residual'),
        patches.Patch(facecolor=COLOR_CLS, edgecolor='#333', label='classifier'),
        patches.Patch(facecolor=COLOR_RECON, edgecolor='#333', label='reconstruction'),
        Line2D([0], [0], color='#9a9a9a', linestyle='--', label='into recon'),
        Line2D([0], [0], color='#c44', linestyle='-', label='loss arrow'),
    ]
    ax.legend(handles=legend_handles, loc='lower left',
              bbox_to_anchor=(0.012, 0.005), fontsize=8,
              ncol=3, frameon=True, framealpha=0.95, edgecolor='#bbb')

    # No footer note here — the legend now occupies that area, and the equation +
    # the regularizer panel already carry the conceptual content.

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Wrote {OUT_PATH}')


if __name__ == '__main__':
    main()

"""Publication-quality two-panel architecture figure for the "v3 denoise" model.

Panel (A): Denoising spatial-spectral MAE pre-training
Panel (B): SpatialSpectralClassifier fine-tuning (5-way mineral head)

All architecture details verified against:
  - models/spatial_spectral_transformer.py
  - models/spatial_mae.py
  - models/denoising_spatial_mae.py
  - models/noise_augmentation.py
  - scripts/hpc_pretrain_denoising.slurm
  - scripts/hpc_finetune_v3_v4.slurm

Run:
    conda run -n crism python scripts/figures/fig_v3_denoise_architecture.py

Outputs:
    reports/fig_v3_denoise_architecture.png  (300 DPI)
    reports/fig_v3_denoise_architecture.svg
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D

# ──────────────────────────────────────────────────────────────────────────────
# Palette — cohesive, colorblind-friendly. Cool blues for transformer blocks,
# warm orange for corruption/masking, deep green for the classifier output.
# ──────────────────────────────────────────────────────────────────────────────
C_BG_PANEL   = '#fbfbfd'
C_FRAME      = '#9aa3ad'
C_ENC_FILL   = '#dde7f3'     # encoder transformer block fill
C_ENC_EDGE   = '#3b5a7e'     # encoder edge
C_DEC_FILL   = '#e6dff0'     # decoder transformer block fill (subtler, lighter)
C_DEC_EDGE   = '#6a4f8a'
C_CORRUPT    = '#f2a25c'     # warm: corruption / noise
C_MASK       = '#d96b4b'     # darker warm for masked-out tokens
C_CLS        = '#f7c948'     # gold/yellow for CLS token (distinct)
C_CLS_EDGE   = '#a17e15'
C_HEAD_FILL  = '#c9e3c9'     # green: classifier head
C_HEAD_EDGE  = '#34754b'
C_LOSS_FILL  = '#f7dad6'     # soft pink for loss annotation
C_LOSS_EDGE  = '#b94c39'
C_TEXT       = '#1f2a36'
C_TEXT_MUTED = '#4d5763'
C_ARROW      = '#3a4654'

# Class colors (match the project's CLASS_COLORS in scripts/figures/_utils.py)
CLASS_COLORS = {
    'olivine':     '#2ca02c',
    'LCP':         '#1f77b4',
    'HCP':         '#d62728',
    'plagioclase': '#ff7f0e',
    'bland':       '#7f7f7f',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.linewidth': 0.0,
    'svg.fonttype': 'none',  # editable text in SVG
})


# ──────────────────────────────────────────────────────────────────────────────
# Drawing primitives
# ──────────────────────────────────────────────────────────────────────────────
def rbox(ax, x, y, w, h, label='', *, fc=C_ENC_FILL, ec=C_ENC_EDGE, lw=1.2,
         fontsize=9, color=C_TEXT, weight='normal', italic=False, zorder=2,
         rounding=0.06, pad=0.015, va='center', ha='center'):
    bb = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f'round,pad={pad},rounding_size={rounding}',
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder,
    )
    ax.add_patch(bb)
    if label:
        style = 'italic' if italic else 'normal'
        ax.text(x + w / 2, y + h / 2, label,
                ha=ha, va=va, fontsize=fontsize, color=color, weight=weight,
                style=style, zorder=zorder + 1)
    return bb


def arrow(ax, x0, y0, x1, y1, *, color=C_ARROW, lw=1.3, style='-|>',
          mutation=12, connectionstyle='arc3,rad=0', zorder=3, alpha=1.0):
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, color=color, lw=lw,
        mutation_scale=mutation, shrinkA=2, shrinkB=2,
        connectionstyle=connectionstyle, zorder=zorder, alpha=alpha,
    )
    ax.add_patch(arr)
    return arr


def text(ax, x, y, s, *, fontsize=9, color=C_TEXT, weight='normal',
         italic=False, ha='center', va='center', zorder=5, bbox=None):
    style = 'italic' if italic else 'normal'
    return ax.text(x, y, s, ha=ha, va=va, fontsize=fontsize, color=color,
                   weight=weight, style=style, zorder=zorder, bbox=bbox)


# White background bbox to use on labels that may sit over arrows/lines
WHITE_BG = dict(facecolor='white', edgecolor='none', pad=1.2)


# ──────────────────────────────────────────────────────────────────────────────
# Specialized sub-drawings
# ──────────────────────────────────────────────────────────────────────────────
def draw_patch_grid(ax, x0, y0, cell=0.18, *, masked_ids=None, corrupt_alpha=0.0,
                    seed=7, label_corner=False, edge_color='#3a4654'):
    """Draw a 7x7 grid of squares colored by a stylized hyperspectral pattern."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:7, 0:7].astype(float)
    base = 0.45 + 0.30 * np.sin(0.6 * xx + 0.2 * yy) + 0.20 * np.cos(0.45 * yy - 0.3 * xx)
    base += 0.06 * rng.standard_normal((7, 7))
    base = (base - base.min()) / (np.ptp(base) + 1e-9)

    masked_set = set(masked_ids or [])
    cmap = plt.get_cmap('viridis')
    for r in range(7):
        for c in range(7):
            flat = r * 7 + c
            px = x0 + c * cell
            py = y0 + (6 - r) * cell  # flip so row 0 is on top
            if flat in masked_set:
                ax.add_patch(Rectangle(
                    (px, py), cell * 0.92, cell * 0.92,
                    facecolor='#e8eaee', edgecolor='#9aa3ad', linewidth=0.5,
                    zorder=4,
                ))
                ax.plot(
                    [px + cell * 0.15, px + cell * 0.77],
                    [py + cell * 0.15, py + cell * 0.77],
                    color='#7a8390', lw=0.8, zorder=5,
                )
                ax.plot(
                    [px + cell * 0.15, px + cell * 0.77],
                    [py + cell * 0.77, py + cell * 0.15],
                    color='#7a8390', lw=0.8, zorder=5,
                )
            else:
                color = cmap(0.10 + 0.80 * base[r, c])
                ax.add_patch(Rectangle(
                    (px, py), cell * 0.92, cell * 0.92,
                    facecolor=color, edgecolor=edge_color, linewidth=0.4,
                    zorder=4,
                ))
                if corrupt_alpha > 0:
                    nspeck = 3
                    sx = px + cell * 0.92 * rng.random(nspeck)
                    sy = py + cell * 0.92 * rng.random(nspeck)
                    ax.scatter(
                        sx, sy, s=2.5, c='#ffffff', alpha=corrupt_alpha,
                        zorder=5, linewidths=0,
                    )

    if label_corner == 'center':
        r, c = 3, 3
        px = x0 + c * cell
        py = y0 + (6 - r) * cell
        ring = Circle((px + cell * 0.46, py + cell * 0.46),
                      radius=cell * 0.55,
                      facecolor='none', edgecolor='#d62728', lw=1.8, zorder=8)
        ax.add_patch(ring)


def draw_token_strip(ax, x0, y0, n=8, *, w=0.20, h=0.32, gap=0.06,
                     fill=C_ENC_FILL, edge=C_ENC_EDGE, cls_first=True,
                     label=None):
    """A horizontal strip of n token boxes. If cls_first, first one is CLS (gold)."""
    for i in range(n):
        x = x0 + i * (w + gap)
        if cls_first and i == 0:
            ax.add_patch(FancyBboxPatch(
                (x, y0), w, h,
                boxstyle='round,pad=0.005,rounding_size=0.04',
                facecolor=C_CLS, edgecolor=C_CLS_EDGE, linewidth=1.0, zorder=6,
            ))
            ax.text(x + w / 2, y0 + h / 2, 'CLS',
                    ha='center', va='center', fontsize=7, color='#5b4b0f',
                    weight='bold', zorder=7)
        else:
            ax.add_patch(FancyBboxPatch(
                (x, y0), w, h,
                boxstyle='round,pad=0.005,rounding_size=0.04',
                facecolor=fill, edgecolor=edge, linewidth=0.9, zorder=6,
            ))


def draw_transformer_stack(ax, x, y, w, h, *, n_layers=6, label='',
                           fc=C_ENC_FILL, ec=C_ENC_EDGE,
                           sub_label='', show_repeat=True,
                           sub_label_below=True, label_fontsize=10):
    """Stylized 'stack' of transformer layers — visible front block + shadows.

    sub_label is placed BELOW the block (centered, multiline okay).
    """
    # Back shadows
    for i, dx in enumerate([0.12, 0.06]):
        ax.add_patch(FancyBboxPatch(
            (x + dx, y + dx), w, h,
            boxstyle='round,pad=0.015,rounding_size=0.08',
            linewidth=0.9, edgecolor=ec, facecolor=fc, alpha=0.55 - 0.18 * i,
            zorder=2,
        ))

    # Front block (no internal label printing here so we can place precisely)
    bb = FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.015,rounding_size=0.08',
        linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=3,
    )
    ax.add_patch(bb)
    ax.text(x + w / 2, y + h / 2, label,
            ha='center', va='center', fontsize=label_fontsize,
            color=C_TEXT, weight='bold', zorder=4)

    # "x N" annotation top-right
    if show_repeat:
        ax.text(x + w + 0.08, y + h - 0.08,
                f'×{n_layers}', ha='left', va='top',
                fontsize=11, color=ec, weight='bold', zorder=4)

    # Sub-label below the block (allow multiline)
    if sub_label and sub_label_below:
        ax.text(x + w / 2, y - 0.12, sub_label,
                ha='center', va='top',
                fontsize=8, color=C_TEXT_MUTED, style='italic', zorder=4)


def draw_plus_circle(ax, x, y, r=0.10, label='+', color=C_ENC_EDGE):
    circ = Circle((x, y), radius=r, facecolor='white', edgecolor=color,
                  linewidth=1.3, zorder=6)
    ax.add_patch(circ)
    ax.text(x, y, label, ha='center', va='center', fontsize=10,
            color=color, weight='bold', zorder=7)


def draw_spectrum_pair(ax, x, y, *, w=1.8, h=0.55, seed=2):
    """Draw a tiny 'noisy → clean' spectrum comparison with wavelength axis.

    Drawn as a self-contained sub-panel: a white card with title above the
    curves, small inline color legend below the title, and wavelength axis
    labels inside the card's bottom margin. No external collisions.
    """
    rng = np.random.default_rng(seed)
    bands = np.linspace(0, 1, 59)
    clean = (
        0.55 + 0.15 * np.sin(2.0 * np.pi * bands * 0.9 + 0.5)
        - 0.12 * np.exp(-((bands - 0.27) / 0.05) ** 2)
        - 0.18 * np.exp(-((bands - 0.72) / 0.08) ** 2)
    )
    noisy = clean + 0.025 * rng.standard_normal(59)
    spike_profile = np.exp(-0.5 * ((np.arange(59) - 15) / (3.0 / 2.355)) ** 2)
    spike_profile[(np.arange(59) < 13) | (np.arange(59) > 17)] = 0
    noisy = noisy + 0.05 * spike_profile

    def _plot_curve(xs, ys, color, alpha=1.0, lw=1.0):
        ys_n = (ys - ys.min()) / (np.ptp(ys) + 1e-9)
        ax.plot(x + xs * w, y + ys_n * h, color=color, alpha=alpha, lw=lw, zorder=4)

    # Self-contained card. Reserve generous top space for title + legend,
    # and bottom space for the wavelength axis labels, so nothing outside
    # the card needs to make room for these.
    card_top_pad = 0.85
    card_bot_pad = 0.42
    ax.add_patch(FancyBboxPatch(
        (x - 0.12, y - card_bot_pad), w + 0.24, h + card_top_pad + card_bot_pad,
        boxstyle='round,pad=0.01,rounding_size=0.05',
        facecolor='white', edgecolor='#cfd5dd', linewidth=0.9, zorder=2,
    ))
    _plot_curve(bands, noisy, color=C_CORRUPT, alpha=0.95, lw=1.1)
    _plot_curve(bands, clean, color=C_ENC_EDGE, alpha=0.95, lw=1.2)

    # Title above curves
    ax.text(x + w / 2, y + h + 0.52,
            'reconstructed vs. clean target',
            ha='center', va='bottom', fontsize=8, color=C_TEXT, weight='bold',
            zorder=5)
    # Small inline color legend (above plot, below title)
    ax.text(x + w / 2, y + h + 0.22,
            'reconstructed (orange)   ·   clean (blue)',
            ha='center', va='bottom', fontsize=7, color=C_TEXT_MUTED,
            style='italic', zorder=5)
    # Wavelength axis labels INSIDE the card's bottom pad
    ax.text(x, y - 0.18, '0.4 µm', ha='center', va='top',
            fontsize=7, color=C_TEXT_MUTED, zorder=5)
    ax.text(x + w, y - 0.18, '2.6 µm', ha='center', va='top',
            fontsize=7, color=C_TEXT_MUTED, zorder=5)
    ax.text(x + w / 2, y - 0.18, 'wavelength', ha='center', va='top',
            fontsize=7, color=C_TEXT_MUTED, style='italic', zorder=5)


# ──────────────────────────────────────────────────────────────────────────────
# Panel A — Pretraining (Denoising MAE)
# ──────────────────────────────────────────────────────────────────────────────
def draw_panel_a(ax):
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.axis('off')

    # Soft background frame
    ax.add_patch(FancyBboxPatch(
        (0.15, 0.20), 17.7, 8.6,
        boxstyle='round,pad=0.0,rounding_size=0.15',
        facecolor=C_BG_PANEL, edgecolor=C_FRAME, linewidth=1.0, zorder=0,
    ))

    # Title
    text(ax, 0.45, 8.45,
         '(A)  Stage 1 — Denoising Spatial-Spectral MAE pretraining',
         fontsize=13, weight='bold', ha='left', color=C_TEXT)
    text(ax, 0.45, 8.10,
         'Self-supervised reconstruction of clean spectra from corrupted, '
         'partially masked patches',
         fontsize=9, ha='left', italic=True, color=C_TEXT_MUTED)

    # Pipeline y center
    py_center = 6.30
    grid_y0 = 5.10   # bottom of 7x7 grid (grid is 1.72 tall with cell=0.245)

    # ── (1) Clean input patch ────────────────────────────────────────────────
    text(ax, 1.40, 7.55, 'clean patch', fontsize=9, weight='bold',
         color=C_TEXT, ha='center')
    text(ax, 1.40, 7.28, '7 × 7 × 59 bands', fontsize=8, italic=True,
         color=C_TEXT_MUTED, ha='center')
    draw_patch_grid(ax, x0=0.55, y0=grid_y0, cell=0.245)

    # ── (2) Noise corruption box ─────────────────────────────────────────────
    arrow(ax, 2.40, py_center, 2.85, py_center, lw=1.4)
    rbox(ax, 2.95, 5.85, 1.85, 0.95,
         'CRISM noise\ncorruption',
         fc='#fce7d2', ec=C_CORRUPT, lw=1.3, fontsize=9, weight='bold')
    text(ax, 3.875, 5.65,
         r'$\sigma_{\mathrm{gauss}}{=}0.0087$' + '\n'
         + r'$\sigma_{\mathrm{spike}}{=}0.0058$  (1 µm seam)' + '\n'
         + r'$\sigma_{\mathrm{col}}{=}0.0049$',
         fontsize=7.5, color=C_TEXT_MUTED, italic=True, ha='center', va='top')

    # ── (3) Corrupted patch ──────────────────────────────────────────────────
    arrow(ax, 4.85, py_center, 5.30, py_center, lw=1.4)
    text(ax, 6.20, 7.55, 'corrupted', fontsize=9, weight='bold',
         color=C_TEXT, ha='center')
    text(ax, 6.20, 7.28, '+ Gaussian + spike + col bias', fontsize=8,
         italic=True, color=C_TEXT_MUTED, ha='center')
    draw_patch_grid(ax, x0=5.35, y0=grid_y0, cell=0.245,
                    corrupt_alpha=0.85, seed=11)

    # ── (4) Random masking ────────────────────────────────────────────────────
    arrow(ax, 7.05, py_center, 7.55, py_center, lw=1.4)
    rbox(ax, 7.60, 6.00, 1.55, 0.65,
         'random mask\n75 %', fc='#f9d6c6', ec=C_MASK, lw=1.3,
         fontsize=8.5, weight='bold')

    arrow(ax, 9.20, py_center, 9.65, py_center, lw=1.4)
    # Masked patch
    text(ax, 10.50, 7.55, 'visible 25 %', fontsize=9, weight='bold',
         color=C_TEXT, ha='center')
    text(ax, 10.50, 7.28, r'~ 12 of 49 tokens', fontsize=8,
         italic=True, color=C_TEXT_MUTED, ha='center')
    rng_local = np.random.default_rng(3)
    masked = sorted(rng_local.choice(49, size=37, replace=False).tolist())
    draw_patch_grid(ax, x0=9.65, y0=grid_y0, cell=0.245, masked_ids=masked, seed=11)

    # ── (5) Linear band embedding + CLS + positional ─────────────────────────
    arrow(ax, 11.40, py_center, 11.85, py_center, lw=1.4)
    # band_embed label ABOVE the strip with comfortable spacing
    text(ax, 12.95, 7.55, 'band_embed', fontsize=9, weight='bold',
         color=C_TEXT, ha='center')
    text(ax, 12.95, 7.28, r'Linear(59 → 128)', fontsize=8, italic=True,
         color=C_TEXT_MUTED, ha='center')
    # Token strip
    draw_token_strip(ax, x0=11.90, y0=6.10, n=9, w=0.20, h=0.46, gap=0.04,
                     fill=C_ENC_FILL, edge=C_ENC_EDGE, cls_first=True)
    # Label below strip (more breathing room)
    text(ax, 12.95, 5.92, 'CLS + visible tokens', fontsize=7.5,
         italic=True, color=C_TEXT_MUTED, ha='center', va='top')

    # + pos embed
    draw_plus_circle(ax, 12.95, 5.55, r=0.11, label='+', color=C_ENC_EDGE)
    text(ax, 12.95, 5.32,
         'learned positional embedding',
         fontsize=7.5, italic=True, color=C_TEXT_MUTED, ha='center', va='top')

    # ── (6) Encoder stack ─────────────────────────────────────────────────────
    arrow(ax, 14.10, py_center, 14.55, py_center, lw=1.4)
    draw_transformer_stack(
        ax, x=14.60, y=5.65, w=2.45, h=1.30, n_layers=6,
        label='Transformer\nEncoder',
        fc=C_ENC_FILL, ec=C_ENC_EDGE,
        sub_label='embed_dim 128 · 4 heads\nFF 512 · pre-LN',
        label_fontsize=10,
    )

    # ── (7) Lower-row pipeline (right-to-left flow) ──────────────────────────
    # Tier-y of lower-row boxes (centers)
    low_y = 3.40

    # Encoder bottom-center → drop straight down, then jog left to the
    # insert-mask block.
    enc_cx = 14.60 + 2.45 / 2

    # Insert-mask reinsertion node (rightmost on lower row) — give it
    # enough width to hold its longish sub-label.
    insert_x = 13.10
    insert_w = 2.70
    insert_y = 2.85
    insert_h = 1.20
    rbox(ax, insert_x, insert_y, insert_w, insert_h,
         '',
         fc='#f9d6c6', ec=C_MASK, lw=1.3)
    # Bold title (top), italic subtext (bottom) — both inside the box
    text(ax, insert_x + insert_w / 2, insert_y + insert_h - 0.30,
         'insert mask tokens',
         fontsize=9, weight='bold', color=C_TEXT, ha='center', va='center')
    text(ax, insert_x + insert_w / 2, insert_y + 0.42,
         'projected encoder out  +\nlearnable mask token  +\ndecoder pos. embed',
         fontsize=7.0, italic=True, color=C_TEXT_MUTED,
         ha='center', va='center')

    # Vertical drop from encoder to insert-mask box top
    arrow(ax, enc_cx, 5.65, enc_cx, insert_y + insert_h, lw=1.4)

    # Decoder stack — to the LEFT of insert-mask block
    dec_x = 9.40
    dec_w = 2.40
    draw_transformer_stack(
        ax, x=dec_x, y=2.85, w=dec_w, h=1.20, n_layers=2,
        label='Transformer\nDecoder',
        fc=C_DEC_FILL, ec=C_DEC_EDGE,
        sub_label='decoder_dim 64 · 2 layers',
        label_fontsize=10,
    )

    # Arrow from insert-mask LEFT edge → decoder RIGHT edge
    arrow(ax, insert_x, low_y + 0.05, dec_x + dec_w, low_y + 0.05, lw=1.4)

    # ── (8) Reconstruction head — LEFT of decoder ────────────────────────────
    head_x = 6.55
    head_w = 1.95
    arrow(ax, dec_x, low_y + 0.05, head_x + head_w, low_y + 0.05, lw=1.4)
    rbox(ax, head_x, 3.05, head_w, 0.90,
         'Linear(64 → 59)\nrecon head',
         fc=C_DEC_FILL, ec=C_DEC_EDGE, lw=1.2, fontsize=8.5)

    # ── (9) Spectrum comparison — LEFT of recon head ─────────────────────────
    # Sub-panel is fully self-contained (title, legend, wavelength axis labels
    # all live INSIDE its white card). Position it so its card does not
    # vertically overlap the Pretraining loss callout below it.
    spec_x = 3.55
    spec_w = 2.40
    arrow(ax, head_x, low_y + 0.05, spec_x + spec_w + 0.14, low_y + 0.05, lw=1.4)
    draw_spectrum_pair(ax, x=spec_x, y=3.55, w=spec_w, h=0.50, seed=4)

    # ── (10) Loss callout — far LEFT sidebar, BELOW the spectrum card ────────
    # Moved down so it no longer collides vertically with the spectrum card,
    # and given its own vertical leader up to the spectrum card.
    loss_x, loss_y, loss_w, loss_h = 0.55, 1.20, 3.10, 1.55
    rbox(ax, loss_x, loss_y, loss_w, loss_h,
         '',
         fc=C_LOSS_FILL, ec=C_LOSS_EDGE, lw=1.2)
    cx = loss_x + loss_w / 2
    text(ax, cx, loss_y + loss_h - 0.30,
         'Pretraining loss',
         fontsize=9.5, weight='bold', color=C_LOSS_EDGE, ha='center')
    text(ax, cx, loss_y + loss_h - 0.70,
         r'$\mathcal{L}_{\mathrm{denoise}} = '
         r'\frac{1}{49NB}\sum \|\hat{x} - x_{\mathrm{clean}}\|^{2}$',
         fontsize=8.5, color=C_TEXT, ha='center')
    text(ax, cx, loss_y + 0.35,
         'MSE over ALL 49 positions\ntarget = clean spectrum',
         fontsize=7.3, italic=True, color=C_TEXT_MUTED, ha='center')

    # Vertical leader from top of loss callout up to spectrum card bottom
    leader_x = spec_x + spec_w / 2
    arrow(ax, leader_x, loss_y + loss_h, leader_x, 3.13,
          lw=0.9, alpha=0.7, mutation=8, style='-|>',
          color=C_LOSS_EDGE)

    # Legend chip — masked vs visible squares (bottom right, comfortable margin)
    legend_x, legend_y = 13.40, 0.55
    ax.add_patch(Rectangle((legend_x, legend_y), 0.22, 0.22,
                           facecolor='#4575b4', edgecolor=C_FRAME, lw=0.6))
    text(ax, legend_x + 0.30, legend_y + 0.11, 'visible token',
         fontsize=7.5, ha='left', color=C_TEXT_MUTED)
    ax.add_patch(Rectangle((legend_x + 1.60, legend_y), 0.22, 0.22,
                           facecolor='#e8eaee', edgecolor='#9aa3ad', lw=0.6))
    ax.plot([legend_x + 1.63, legend_x + 1.79],
            [legend_y + 0.04, legend_y + 0.18], color='#7a8390', lw=0.8)
    ax.plot([legend_x + 1.63, legend_x + 1.79],
            [legend_y + 0.18, legend_y + 0.04], color='#7a8390', lw=0.8)
    text(ax, legend_x + 1.92, legend_y + 0.11, 'masked token',
         fontsize=7.5, ha='left', color=C_TEXT_MUTED)


# ──────────────────────────────────────────────────────────────────────────────
# Panel B — Fine-tuning (SpatialSpectralClassifier)
# ──────────────────────────────────────────────────────────────────────────────
def draw_panel_b(ax):
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.add_patch(FancyBboxPatch(
        (0.15, 0.20), 17.7, 8.6,
        boxstyle='round,pad=0.0,rounding_size=0.15',
        facecolor=C_BG_PANEL, edgecolor=C_FRAME, linewidth=1.0, zorder=0,
    ))

    text(ax, 0.45, 8.45,
         '(B)  Stage 2 — SpatialSpectralClassifier fine-tuning',
         fontsize=13, weight='bold', ha='left', color=C_TEXT)
    text(ax, 0.45, 8.10,
         'No corruption, no masking — full 7×7 patch encoded; '
         'center pixel decodes to class logits',
         fontsize=9, ha='left', italic=True, color=C_TEXT_MUTED)

    py_center = 6.30
    grid_y0 = 5.10

    # ── (1) Clean patch ───────────────────────────────────────────────────────
    text(ax, 1.40, 7.55, 'clean patch', fontsize=9, weight='bold',
         color=C_TEXT, ha='center')
    text(ax, 1.40, 7.28, '7 × 7 × 59 bands', fontsize=8,
         italic=True, color=C_TEXT_MUTED, ha='center')
    draw_patch_grid(ax, x0=0.55, y0=grid_y0, cell=0.245,
                    label_corner='center')
    text(ax, 1.40, 4.90, 'center pixel highlighted', fontsize=7.5,
         italic=True, color='#9c2b2b', ha='center')

    # ── (2) band_embed ───────────────────────────────────────────────────────
    arrow(ax, 2.40, py_center, 2.85, py_center, lw=1.4)
    rbox(ax, 2.95, 5.85, 1.95, 0.95,
         'band_embed\nLinear(59 → 128)',
         fc='#eef3fa', ec=C_ENC_EDGE, lw=1.2, fontsize=9)
    text(ax, 3.925, 5.65,
         'each pixel → 128-d token',
         fontsize=7.5, italic=True, color=C_TEXT_MUTED, ha='center', va='top')

    # ── (3) Full 49 tokens + CLS into encoder ────────────────────────────────
    arrow(ax, 4.95, py_center, 5.40, py_center, lw=1.4)
    draw_token_strip(ax, x0=5.45, y0=6.10, n=10, w=0.18, h=0.46, gap=0.04,
                     fill=C_ENC_FILL, edge=C_ENC_EDGE, cls_first=True)
    text(ax, 6.55, 5.92, 'CLS + all 49 spatial tokens',
         fontsize=7.5, italic=True, color=C_TEXT_MUTED,
         ha='center', va='top')

    # + pos embed
    draw_plus_circle(ax, 6.55, 5.55, r=0.11, label='+', color=C_ENC_EDGE)
    text(ax, 6.55, 5.32, 'learned positional embedding',
         fontsize=7.5, italic=True, color=C_TEXT_MUTED, ha='center', va='top')

    # ── (4) Encoder stack ─────────────────────────────────────────────────────
    arrow(ax, 7.75, py_center, 8.20, py_center, lw=1.4)
    draw_transformer_stack(
        ax, x=8.25, y=5.65, w=2.70, h=1.30, n_layers=6,
        label='Transformer\nEncoder',
        fc=C_ENC_FILL, ec=C_ENC_EDGE,
        sub_label='embed_dim 128 · 4 heads\n(weights from Stage 1)',
        label_fontsize=10,
    )

    # ── (5) Sequence out — highlight center-pixel slot ───────────────────────
    arrow(ax, 11.15, py_center, 11.60, py_center, lw=1.4)
    # Token strip with one box (center pixel = slot 25) highlighted in red
    strip_x = 11.65
    strip_y = 6.10
    w, h, gap = 0.16, 0.46, 0.03
    n_show = 12
    highlight_idx = 5
    for i in range(n_show):
        x = strip_x + i * (w + gap)
        if i == 0:
            ax.add_patch(FancyBboxPatch(
                (x, strip_y), w, h,
                boxstyle='round,pad=0.005,rounding_size=0.04',
                facecolor=C_CLS, edgecolor=C_CLS_EDGE, linewidth=1.0, zorder=6,
            ))
            ax.text(x + w / 2, strip_y + h / 2, 'CLS',
                    ha='center', va='center', fontsize=6.5,
                    color='#5b4b0f', weight='bold', zorder=7)
        elif i == highlight_idx:
            ax.add_patch(FancyBboxPatch(
                (x, strip_y - 0.05), w + 0.02, h + 0.10,
                boxstyle='round,pad=0.005,rounding_size=0.04',
                facecolor='#fde2e1', edgecolor='#d62728', linewidth=1.6,
                zorder=6,
            ))
        else:
            ax.add_patch(FancyBboxPatch(
                (x, strip_y), w, h,
                boxstyle='round,pad=0.005,rounding_size=0.04',
                facecolor=C_ENC_FILL, edgecolor=C_ENC_EDGE, linewidth=0.9,
                zorder=6,
            ))

    strip_total_w = n_show * (w + gap) - gap
    strip_center_x = strip_x + strip_total_w / 2

    # Caption placed ABOVE the highlighted center-pixel slot only (not above
    # the whole strip), with a downward leader pointing into the red slot.
    # Shortened to "center pixel (slot 25)" — full description "slot 25 =
    # CLS + 24" lives in the figure caption — so it does not collide with the
    # encoder's "(weights from Stage 1)" sub-label to its left.
    highlight_cx = strip_x + highlight_idx * (w + gap) + w / 2
    label_y = strip_y + h + 0.55
    text(ax, highlight_cx, label_y,
         'center pixel  (slot 25)',
         fontsize=7.8, italic=True, color='#9c2b2b', ha='center', va='center')
    # Downward leader from label to top of the highlighted token box
    arrow(ax, highlight_cx, label_y - 0.10,
          highlight_cx, strip_y + h + 0.07,
          lw=0.9, alpha=0.85, mutation=8, style='-|>', color='#9c2b2b')

    # ── (6) Linear head ───────────────────────────────────────────────────────
    head_center_x = strip_x + highlight_idx * (w + gap) + w / 2
    arrow(ax, head_center_x, strip_y - 0.10,
          head_center_x, 4.55,
          lw=1.4)
    rbox(ax, head_center_x - 1.10, 3.65, 2.20, 0.90,
         'head:  Linear(128 → 5)',
         fc=C_HEAD_FILL, ec=C_HEAD_EDGE, lw=1.3, fontsize=10, weight='bold')

    # ── (7) Class probability bars ───────────────────────────────────────────
    arrow(ax, head_center_x, 3.65, head_center_x, 3.20, lw=1.4)

    classes = [
        ('olivine',     0.78),
        ('LCP',         0.12),
        ('HCP',         0.62),
        ('plagioclase', 0.05),
        ('bland',       0.08),
    ]
    bar_w = 0.50
    bar_gap = 0.45
    n_bars = len(classes)
    total_w = n_bars * bar_w + (n_bars - 1) * bar_gap
    bar_x0 = head_center_x - total_w / 2
    bar_y0 = 1.85
    bar_h_max = 1.20
    for i, (name, p) in enumerate(classes):
        cx = bar_x0 + i * (bar_w + bar_gap)
        bh = bar_h_max * p
        color = CLASS_COLORS[name]
        ax.add_patch(Rectangle(
            (cx, bar_y0), bar_w, bar_h_max,
            facecolor='none', edgecolor='#cfd5dd', linewidth=0.6, zorder=3,
        ))
        ax.add_patch(Rectangle(
            (cx, bar_y0), bar_w, bh,
            facecolor=color, edgecolor='#3a4654', linewidth=0.7, zorder=4,
        ))
        ax.text(cx + bar_w / 2, bar_y0 - 0.08, name,
                ha='center', va='top', fontsize=7.8, color=C_TEXT)
        ax.text(cx + bar_w / 2, bar_y0 + bh + 0.06, f'{p:.2f}',
                ha='center', va='bottom', fontsize=7.5,
                color=C_TEXT_MUTED, style='italic')

    text(ax, head_center_x, 1.00,
         '5 class probabilities (per-pixel)',
         fontsize=8.5, italic=True, color=C_TEXT_MUTED, ha='center')

    # ── (8) Loss + LR sidebars on the LEFT, below the patch ──────────────────
    # Both callouts widened from w=4.20 to w=5.40 so the longer subtitle/
    # italic lines fit comfortably INSIDE their boxes with margin to spare.
    cbox_x = 0.50
    cbox_w = 5.40
    cbox_cx = cbox_x + cbox_w / 2

    # Fine-tuning loss
    rbox(ax, cbox_x, 2.55, cbox_w, 1.55,
         '',
         fc=C_LOSS_FILL, ec=C_LOSS_EDGE, lw=1.2)
    text(ax, cbox_cx, 3.85, 'Fine-tuning loss', fontsize=10, weight='bold',
         color=C_LOSS_EDGE, ha='center')
    text(ax, cbox_cx, 3.45,
         'Asymmetric Loss (ASL)',
         fontsize=9.5, color=C_TEXT, ha='center', weight='bold')
    text(ax, cbox_cx, 3.08,
         r'$\gamma^{-}{=}4,\;\;\gamma^{+}{=}0,\;\;\mathrm{clip}{=}0.05$',
         fontsize=10, color=C_TEXT, ha='center')
    text(ax, cbox_cx, 2.73,
         'multi-label BCE w/ asymmetric focal modulation',
         fontsize=7.5, italic=True, color=C_TEXT_MUTED, ha='center')

    # Differential learning rates — widened; italic explanation line broken
    # onto its own row so it cannot overflow horizontally.
    rbox(ax, cbox_x, 0.50, cbox_w, 1.85, '',
         fc='#e6efe0', ec='#5e8c4d', lw=1.2)
    text(ax, cbox_cx, 2.10, 'Differential learning rates',
         fontsize=10, weight='bold', color='#3e5e30', ha='center')
    text(ax, cbox_cx, 1.70,
         r'head LR  $=$  $5{\times}10^{-4}$',
         fontsize=9.5, color=C_TEXT, ha='center')
    text(ax, cbox_cx, 1.30,
         r'encoder LR  $=$  $\mathrm{lr\_scale}\cdot$ head LR',
         fontsize=9.5, color=C_TEXT, ha='center')
    text(ax, cbox_cx, 0.90,
         r'lr_scale $\in \{0.001,\,0.01,\,0.1\}$',
         fontsize=8.5, color=C_TEXT, ha='center')
    text(ax, cbox_cx, 0.65,
         'encoder updates slowly',
         fontsize=7.3, italic=True, color=C_TEXT_MUTED, ha='center')


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # Wider figure for more horizontal breathing room
    fig = plt.figure(figsize=(18, 11), dpi=300)
    gs = fig.add_gridspec(
        nrows=2, ncols=1,
        height_ratios=[1.0, 1.0],
        hspace=0.10,
        left=0.012, right=0.988, top=0.985, bottom=0.015,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])

    draw_panel_a(ax_a)
    draw_panel_b(ax_b)

    out_dir = '/mnt/mrdr/crism_classification/reports'
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, 'fig_v3_denoise_architecture.png')
    svg_path = os.path.join(out_dir, 'fig_v3_denoise_architecture.svg')

    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(svg_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    png_size = os.path.getsize(png_path) / (1024 * 1024)
    svg_size = os.path.getsize(svg_path) / (1024 * 1024)
    print(f'Wrote {png_path}  ({png_size:.2f} MB)')
    print(f'Wrote {svg_path}  ({svg_size:.2f} MB)')


if __name__ == '__main__':
    main()

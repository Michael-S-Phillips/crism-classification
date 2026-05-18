"""
Signal / noise decomposition comparison: v3 (denoising) vs v4 (SPEND) MAE encoders.

For three representative pixels (olivine, hcp, plagioclase), shows:
  - Col 1: clean center-pixel spectrum (reference, color-coded by class)
  - Col 2: v3 (denoising) decomposition — two stacked subplots sharing x-axis:
           Top: signal estimate (recon, dashed) overlaid on clean (solid)
           Bottom: noise estimate (clean - recon), ±0.01 I/F y-range
  - Col 3: v4 (SPEND) decomposition — same stacked-subplot pattern

Title: "What each MAE considers signal vs noise — pretrained encoders on real CRISM pixels"
Subtitle: "Center pixel; encoder sees only 25% of spatial positions;
          `noise` = clean − recon"

Usage (no args needed):
    conda run -n crism python scripts/figures/fig_v5_pretrain_signal_noise.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

# ── project root on sys.path ────────────────────────────────────────────────
PROJECT_ROOT = '/mnt/mrdr/crism_classification'
sys.path.insert(0, PROJECT_ROOT)

from models.denoising_spatial_mae import DenoisingSpatialSpectralMAE
from models.spend_spatial_mae import SpendSpatialSpectralMAE

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts', 'figures'))
from _utils import (
    CLASS_COLORS, build_mrral_map, find_representative_pixels,
    get_wavelengths_59, load_mrral_parquet, read_patch_from_tile,
)

# ── paths ────────────────────────────────────────────────────────────────────
CKPT_V3 = os.path.join(PROJECT_ROOT, 'checkpoints', 'spatial_mae_denoising_128d_6l_best.pt')
CKPT_V4 = os.path.join(PROJECT_ROOT, 'checkpoints', 'spatial_mae_spend_128d_6l_best.pt')
OUT_PATH = os.path.join(PROJECT_ROOT, 'reports', 'v5', 'fig_v5_pretrain_signal_noise.png')

CLASSES_TO_SHOW = ['olivine', 'hcp', 'plagioclase']

# Center pixel flat index for a 7×7 patch (row 3, col 3 → 3*7+3 = 24)
CENTER_IDX = 24

# Noise subplot y-range (±0.01 I/F)
NOISE_YLIM = 0.012


# ── model loaders (mirrored from fig_v5_pretrain_reconstructions.py) ─────────

def load_v3(path: str) -> DenoisingSpatialSpectralMAE:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    m = DenoisingSpatialSpectralMAE(
        n_bands=59, patch_size=7,
        embed_dim=cfg.get('embed_dim', 128),
        n_heads=cfg.get('n_heads', 4),
        n_layers=cfg.get('n_layers', 6),
        decoder_dim=cfg.get('decoder_dim', 64),
        decoder_layers=cfg.get('decoder_layers', 2),
        mask_ratio=cfg.get('mask_ratio', 0.75),
    )
    m.load_state_dict(ckpt['mae_state'])
    m.eval()
    return m


def load_v4(path: str) -> SpendSpatialSpectralMAE:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    m = SpendSpatialSpectralMAE(
        n_bands=59, patch_size=7,
        embed_dim=cfg.get('embed_dim', 128),
        n_heads=cfg.get('n_heads', 4),
        n_layers=cfg.get('n_layers', 6),
        decoder_dim=cfg.get('decoder_dim', 64),
        decoder_layers=cfg.get('decoder_layers', 2),
        mask_ratio=cfg.get('mask_ratio', 0.75),
        spectral_mask_ratio=0.0,  # eval mode: no band masking
    )
    m.load_state_dict(ckpt['mae_state'])
    m.eval()
    return m


# ── inference helper ─────────────────────────────────────────────────────────

def run_model(model, patch_raw: np.ndarray, seed: int = 42):
    """Normalize patch, run model, unnormalize reconstruction to I/F.

    Returns:
        recon_if   : (49, 59) float32 — all spatial positions, I/F units
        mask_bool  : (49,)  bool      — True = spatially masked at encoder
        mean, std  : scalars used for normalization (for debug)
    """
    mean = patch_raw.mean()
    std = float(patch_raw.std()) + 1e-8
    patch_norm = (patch_raw - mean) / std

    x = torch.from_numpy(patch_norm).unsqueeze(0).float()  # (1, 7, 7, 59)

    torch.manual_seed(seed)  # fix spatial mask for reproducibility
    with torch.no_grad():
        _loss, recon_norm, mask = model(x)

    # unnormalize back to I/F
    recon_if = recon_norm[0].numpy() * std + mean  # (49, 59)
    mask_bool = mask[0].numpy().astype(bool)       # (49,)
    return recon_if, mask_bool, mean, std


# ── plotting helper for stacked decomposition ─────────────────────────────────

def plot_decomposition(
    fig,
    outer_gs_cell,
    wls: np.ndarray,
    valid: np.ndarray,
    center_clean: np.ndarray,
    center_recon: np.ndarray,
    color: str,
    model_label: str,
    y_ref_lim: tuple,
):
    """Draw the stacked signal/noise subplots inside a gridspec cell.

    Top subplot:  signal estimate (recon dashed) + clean (solid gray)
    Bottom subplot: noise estimate (clean - recon), ±NOISE_YLIM

    Parameters
    ----------
    outer_gs_cell : GridSpecFromSubplotSpec or SubplotSpec — the cell to subdivide
    y_ref_lim     : (ymin, ymax) matching the clean-spectrum reference panel
    """
    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_gs_cell,
        height_ratios=[2.5, 1],
        hspace=0.08,
    )
    ax_sig = fig.add_subplot(inner[0])
    ax_noise = fig.add_subplot(inner[1], sharex=ax_sig)

    # Signal panel
    ax_sig.plot(wls[valid], center_clean[valid],
                color='#888888', linewidth=1.2, linestyle='-', alpha=0.7, label='clean')
    ax_sig.plot(wls[valid], center_recon[valid],
                color=color, linewidth=1.5, linestyle='--', label='recon (signal)')
    ax_sig.set_ylim(y_ref_lim)
    ax_sig.set_ylabel('I/F', fontsize=8)
    ax_sig.set_title(f'{model_label}\nsignal estimate', fontsize=9)
    ax_sig.legend(fontsize=7, loc='upper right')
    ax_sig.grid(alpha=0.3)
    plt.setp(ax_sig.get_xticklabels(), visible=False)

    # Noise panel
    noise = center_clean - center_recon
    noise_mae = float(np.abs(noise[valid]).mean())
    ax_noise.plot(wls[valid], noise[valid],
                  color=color, linewidth=1.1, linestyle='-')
    ax_noise.axhline(0, color='black', linewidth=0.6, linestyle=':')
    ax_noise.set_ylim(-NOISE_YLIM, NOISE_YLIM)
    ax_noise.set_ylabel('noise\n(I/F)', fontsize=7)
    ax_noise.set_xlabel('Wavelength (nm)', fontsize=8)
    ax_noise.set_title(f'noise = clean − recon  [MAE={noise_mae:.4f}]', fontsize=8)
    ax_noise.grid(alpha=0.3)

    return ax_sig, ax_noise


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print('Loading models ...')
    v3 = load_v3(CKPT_V3)
    v4 = load_v4(CKPT_V4)
    ckpt_v3 = torch.load(CKPT_V3, map_location='cpu', weights_only=False)
    ckpt_v4 = torch.load(CKPT_V4, map_location='cpu', weights_only=False)
    print(f'  v3 (denoising) — epoch {ckpt_v3["epoch"]}')
    print(f'  v4 (SPEND)     — epoch {ckpt_v4["epoch"]}')

    print('Loading pixel index ...')
    df = load_mrral_parquet()
    mrral_map = build_mrral_map()
    pixels = find_representative_pixels(df, n_per_class=1, seed=42)
    wls = get_wavelengths_59()

    n_rows = len(CLASSES_TO_SHOW)
    n_cols = 3  # clean | v3 decomp | v4 decomp

    # Use outer gridspec with 3 cols; col 0 is simple, cols 1-2 subdivided
    fig = plt.figure(figsize=(16.0, 4.5 * n_rows))
    outer_gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                                 hspace=0.50, wspace=0.35)

    for row_i, cls in enumerate(CLASSES_TO_SHOW):
        sel = pixels.get(cls, [])
        if not sel:
            print(f'  WARNING: no representative pixel found for {cls}')
            continue
        tid, pr, pc = sel[0]
        mrral_path = mrral_map.get(tid)
        if not mrral_path or not os.path.exists(mrral_path):
            print(f'  WARNING: mrral file not found for tile {tid}')
            continue

        print(f'  [{cls}] tile={tid} pixel=({pr},{pc})')

        patch_raw = read_patch_from_tile(mrral_path, pr, pc, patch_size=7, n_bands=59)
        center_clean = patch_raw[3, 3, :]  # (59,) I/F

        # Run both models with the SAME spatial mask seed
        recon_v3, mask_v3, _, _ = run_model(v3, patch_raw, seed=42)
        recon_v4, mask_v4, _, _ = run_model(v4, patch_raw, seed=42)

        center_recon_v3 = recon_v3[CENTER_IDX]  # (59,)
        center_recon_v4 = recon_v4[CENTER_IDX]

        color = CLASS_COLORS.get(cls, '#333333')
        valid = (center_clean > 0.001) & (center_clean < 0.499)

        # Compute shared y-range for signal panels (from clean spectrum)
        y_clean_min = float(center_clean[valid].min()) - 0.005
        y_clean_max = float(center_clean[valid].max()) + 0.005
        y_ref_lim = (y_clean_min, y_clean_max)

        # ── Col 1: clean center-pixel spectrum ───────────────────────────────
        ax_clean = fig.add_subplot(outer_gs[row_i, 0])
        ax_clean.plot(wls[valid], center_clean[valid],
                      color=color, linewidth=1.8, label='clean')
        ax_clean.set_xlabel('Wavelength (nm)', fontsize=8)
        ax_clean.set_ylabel('I/F', fontsize=8)
        ax_clean.set_title(f'{cls}\ntile {tid}  pixel ({pr},{pc})',
                           color=color, fontsize=9.5)
        ax_clean.grid(alpha=0.3)
        ax_clean.set_ylim(y_ref_lim)

        # ── Col 2: v3 denoising decomposition ────────────────────────────────
        plot_decomposition(
            fig, outer_gs[row_i, 1],
            wls, valid, center_clean, center_recon_v3,
            color=color, model_label='v3 (denoising MAE)',
            y_ref_lim=y_ref_lim,
        )

        # ── Col 3: v4 SPEND decomposition ────────────────────────────────────
        plot_decomposition(
            fig, outer_gs[row_i, 2],
            wls, valid, center_clean, center_recon_v4,
            color=color, model_label='v4 (SPEND MAE)',
            y_ref_lim=y_ref_lim,
        )

        noise_v3 = center_clean - center_recon_v3
        noise_v4 = center_clean - center_recon_v4
        print(f'    noise MAE: v3={np.abs(noise_v3[valid]).mean():.5f}  '
              f'v4={np.abs(noise_v4[valid]).mean():.5f}  '
              f'center masked: v3={bool(mask_v3[CENTER_IDX])}  '
              f'v4={bool(mask_v4[CENTER_IDX])}')

    fig.suptitle(
        'What each MAE considers signal vs noise — pretrained encoders on real CRISM pixels\n'
        'Center pixel; encoder sees only 25% of spatial positions; '
        '`noise` = clean − recon',
        fontsize=11,
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\nWrote {OUT_PATH}')
    sz = os.path.getsize(OUT_PATH)
    print(f'File size: {sz/1024:.1f} KB')


if __name__ == '__main__':
    main()

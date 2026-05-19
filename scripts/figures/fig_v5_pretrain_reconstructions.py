"""
Side-by-side MAE reconstruction quality comparison: denoising MAE vs SPEND MAE.

For three representative pixels (olivine, hcp, plagioclase), shows:
  - Col 1: clean center-pixel spectrum (reference, color-coded by class)
  - Col 2: denoising MAE reconstruction overlaid on clean
           solid=clean, dashed=recon, markers at spatially-masked positions
  - Col 3: SPEND MAE (at spectral_mask_ratio=0) reconstruction — same overlay
  - Col 4: residuals at center pixel for both models in one panel

Title: MAE reconstruction quality — denoising vs SPEND
Subtitle: encoder sees only ~25% of spatial positions; decoder fills in 75%
          from the spectral prior

Usage (no args needed):
    conda run -n crism python scripts/figures/fig_v5_pretrain_reconstructions.py
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
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
CKPT_DENOISING = os.path.join(PROJECT_ROOT, 'checkpoints', 'spatial_mae_denoising_128d_6l_best.pt')
CKPT_SPEND = os.path.join(PROJECT_ROOT, 'checkpoints', 'spatial_mae_spend_128d_6l_best.pt')
OUT_PATH = os.path.join(PROJECT_ROOT, 'reports', 'v5', 'fig_v5_pretrain_reconstructions.png')

CLASSES_TO_SHOW = ['olivine', 'hcp', 'plagioclase']

# Center pixel flat index for a 7×7 patch (row 3, col 3 → 3*7+3 = 24)
CENTER_IDX = 24


# ── model loaders ────────────────────────────────────────────────────────────

def load_denoising(path: str) -> DenoisingSpatialSpectralMAE:
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


def load_spend(path: str) -> SpendSpatialSpectralMAE:
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
        spectral_mask_ratio=0.0,  # eval mode: no band masking — apples-to-apples with denoising
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


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print('Loading models ...')
    mae_denoising = load_denoising(CKPT_DENOISING)
    mae_spend = load_spend(CKPT_SPEND)
    print(f'  denoising — epoch {torch.load(CKPT_DENOISING, map_location="cpu", weights_only=False)["epoch"]}')
    print(f'  SPEND     — epoch {torch.load(CKPT_SPEND, map_location="cpu", weights_only=False)["epoch"]}')

    print('Loading pixel index ...')
    df = load_mrral_parquet()
    mrral_map = build_mrral_map()
    pixels = find_representative_pixels(df, n_per_class=1, seed=42)
    wls = get_wavelengths_59()

    fig, axes = plt.subplots(
        len(CLASSES_TO_SHOW), 4,
        figsize=(16.0, 3.2 * len(CLASSES_TO_SHOW)),
    )
    if len(CLASSES_TO_SHOW) == 1:
        axes = axes[None, :]

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
        recon_denoising, mask_denoising, _, _ = run_model(mae_denoising, patch_raw, seed=42)
        recon_spend, mask_spend, _, _ = run_model(mae_spend, patch_raw, seed=42)

        center_recon_denoising = recon_denoising[CENTER_IDX]  # (59,)
        center_recon_spend = recon_spend[CENTER_IDX]

        color = CLASS_COLORS.get(cls, '#333333')
        # valid wavelength mask: skip nodata zeros and saturated values
        valid = (center_clean > 0.001) & (center_clean < 0.499)

        # ── Col 1: clean center-pixel spectrum ───────────────────────────────
        ax = axes[row_i, 0]
        ax.plot(wls[valid], center_clean[valid], color=color, linewidth=1.8, label='clean')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('I/F')
        ax.set_title(f'{cls}\ntile {tid}  ({pr},{pc})', color=color, fontsize=9.5)
        ax.grid(alpha=0.3)

        # ── Col 2: denoising MAE reconstruction ─────────────────────────────
        ax = axes[row_i, 1]
        ax.plot(wls[valid], center_clean[valid],
                color='#888', linewidth=1.2, linestyle='-', alpha=0.7, label='clean')
        ax.plot(wls[valid], center_recon_denoising[valid],
                color=color, linewidth=1.4, linestyle='--', label='denoising recon')
        # mark center pixel masked or not
        center_masked_denoising = bool(mask_denoising[CENTER_IDX])
        marker_label = 'masked pos' if center_masked_denoising else 'visible pos'
        marker_style = 'x' if center_masked_denoising else 'o'
        ax.scatter(wls[valid][::6], center_recon_denoising[valid][::6],
                   color=color, s=20, marker=marker_style,
                   label=marker_label, zorder=4, alpha=0.8)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('I/F')
        center_status_denoising = 'masked' if center_masked_denoising else 'visible'
        ax.set_title(f'denoising recon\ncenter = {center_status_denoising}', fontsize=9.5)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)

        # ── Col 3: SPEND MAE reconstruction ─────────────────────────────────
        ax = axes[row_i, 2]
        ax.plot(wls[valid], center_clean[valid],
                color='#888', linewidth=1.2, linestyle='-', alpha=0.7, label='clean')
        ax.plot(wls[valid], center_recon_spend[valid],
                color=color, linewidth=1.4, linestyle='--', label='SPEND recon')
        center_masked_spend = bool(mask_spend[CENTER_IDX])
        marker_label_spend = 'masked pos' if center_masked_spend else 'visible pos'
        marker_style_spend = 'x' if center_masked_spend else 'o'
        ax.scatter(wls[valid][::6], center_recon_spend[valid][::6],
                   color=color, s=20, marker=marker_style_spend,
                   label=marker_label_spend, zorder=4, alpha=0.8)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('I/F')
        center_status_spend = 'masked' if center_masked_spend else 'visible'
        ax.set_title(f'SPEND recon\ncenter = {center_status_spend}', fontsize=9.5)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)

        # ── Col 4: residuals ─────────────────────────────────────────────────
        ax = axes[row_i, 3]
        resid_denoising = center_recon_denoising - center_clean
        resid_spend = center_recon_spend - center_clean
        ax.plot(wls[valid], resid_denoising[valid],
                color=color, linewidth=1.3, linestyle='-',
                label=f'denoising  MAE={np.abs(resid_denoising[valid]).mean():.4f}')
        ax.plot(wls[valid], resid_spend[valid],
                color=color, linewidth=1.3, linestyle='--', alpha=0.7,
                label=f'SPEND  MAE={np.abs(resid_spend[valid]).mean():.4f}')
        ax.axhline(0, color='black', linewidth=0.6, linestyle=':')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('recon - clean (I/F)')
        ax.set_title('residuals at center pixel\nsolid=denoising  dashed=SPEND', fontsize=9.5)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)

        print(f'    denoising residual MAE: {np.abs(resid_denoising[valid]).mean():.5f}  '
              f'SPEND: {np.abs(resid_spend[valid]).mean():.5f}  '
              f'center masked: {center_masked_denoising}')

    fig.suptitle(
        'MAE reconstruction quality — denoising vs SPEND\n'
        'encoder sees only ~25% of spatial positions; decoder fills in 75% from the spectral prior',
        fontsize=11,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\nWrote {OUT_PATH}')


if __name__ == '__main__':
    main()

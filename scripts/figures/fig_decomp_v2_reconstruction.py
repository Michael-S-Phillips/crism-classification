"""
Generate v2 reconstruction diagnostic figures — for a representative pixel
per class, show how DecompSpVitAdv decomposed the input spectrum into
signal (s_hat) and noise (n_hat).

Produces two figures, one per checkpoint:
  reports/v5/fig_v5_decomp_v2_recon_lrscale001.png
  reports/v5/fig_v5_decomp_v2_recon_lrscale0001.png

Each figure is a 3×4 grid (3 mineral classes × 4 diagnostic panels):
  Col 1: input x + s_hat overlaid spectrally (center pixel)
  Col 2: n_hat spectrum (center pixel)
  Col 3: spatial heatmap of |n_hat| at the diagnostic 1 µm band
  Col 4: per-column-of-patch mean |n_hat| (column-artifact diagnostic)

Usage:
    conda run -n crism python scripts/figures/fig_decomp_v2_reconstruction.py
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, '/mnt/mrdr/crism_classification')
from models.decomp_spatial_vit_adv import DecompSpVitAdv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _utils import (
    CLASS_COLORS, build_mrral_map, find_representative_pixels,
    get_wavelengths_59, load_mrral_parquet, read_patch_from_tile,
)

CHECKPOINTS = [
    ('spvit_decomp_v2_lrscale001',
     '/mnt/mrdr/crism_classification/checkpoints/spvit_decomp_v2_lrscale001_best.pt'),
    ('spvit_decomp_v2_lrscale0001',
     '/mnt/mrdr/crism_classification/checkpoints/spvit_decomp_v2_lrscale0001_best.pt'),
]
CLASSES_TO_SHOW = ['olivine', 'hcp', 'plagioclase']
DIAGNOSTIC_BAND_NM = 1000   # detector seam region
OUT_DIR = '/mnt/mrdr/crism_classification/reports/v5'


def load_model(ckpt_path: str) -> DecompSpVitAdv:
    model = DecompSpVitAdv(
        n_bands=59, patch_size=7, n_classes=5,
        embed_dim=128, n_heads=4, n_layers=6, dropout=0.1,
        lambda_adv=0.0,   # frozen at inference
    )
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    df = load_mrral_parquet()
    mrral_map = build_mrral_map()
    pixels = find_representative_pixels(df, n_per_class=1, seed=42)
    wls = get_wavelengths_59()
    diag_band_idx = int(np.argmin(np.abs(wls - DIAGNOSTIC_BAND_NM)))
    diag_band_wl = float(wls[diag_band_idx])

    os.makedirs(OUT_DIR, exist_ok=True)

    for ckpt_name, ckpt_path in CHECKPOINTS:
        if not os.path.exists(ckpt_path):
            print(f'  SKIP — {ckpt_path} not found')
            continue
        print(f'\n=== {ckpt_name} ===')
        model = load_model(ckpt_path)

        fig, axes = plt.subplots(
            len(CLASSES_TO_SHOW), 4,
            figsize=(15, 3.0 * len(CLASSES_TO_SHOW)),
            gridspec_kw={'width_ratios': [1.5, 1.3, 1.0, 1.0]},
        )
        if len(CLASSES_TO_SHOW) == 1:
            axes = axes[None, :]

        for row_i, cls in enumerate(CLASSES_TO_SHOW):
            sel = pixels.get(cls, [])
            if not sel:
                print(f'  no pixel found for class {cls}')
                continue
            tid, pr, pc = sel[0]
            mrral = mrral_map.get(tid)
            if not (mrral and os.path.exists(mrral)):
                print(f'  tile {tid} not found locally for {cls}')
                continue

            patch = read_patch_from_tile(mrral, pr, pc, patch_size=7, n_bands=59)
            x_t = torch.from_numpy(patch).unsqueeze(0)
            with torch.no_grad():
                _logits, s_hat_t, n_hat_t, x_hat_t, _, _, _ = model(x_t)
            s_hat = s_hat_t[0].numpy().reshape(7, 7, 59)
            n_hat = n_hat_t[0].numpy().reshape(7, 7, 59)
            x_hat = x_hat_t[0].numpy().reshape(7, 7, 59)
            x_arr = patch              # (7, 7, 59)

            color = CLASS_COLORS[cls]

            # ─── Col 1: input + signal (center pixel) ──────────────────────
            ax = axes[row_i, 0]
            x_center = x_arr[3, 3]
            s_center = s_hat[3, 3]
            # Mask the short-wavelength noise zone for plotting
            valid = (x_center > 0) & (x_center < 0.5)
            ax.plot(wls[valid], x_center[valid], color='#222',
                    linewidth=1.5, label='input $x$')
            ax.plot(wls[valid], s_center[valid], color=color,
                    linewidth=2.0, label='signal $\\hat{s}$')
            ax.axvspan(950, 1050, alpha=0.08, color='red',
                       label='1 µm seam region')
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel('Reflectance (I/F)')
            ax.set_title(f'{cls}: tile {tid}, pixel ({pr},{pc})\n'
                         'input vs predicted signal',
                         color=color, fontsize=9.5)
            ax.legend(fontsize=8, loc='lower right')
            ax.grid(alpha=0.3)

            # ─── Col 2: noise spectrum (center pixel) ──────────────────────
            ax = axes[row_i, 1]
            n_center = n_hat[3, 3]
            ax.plot(wls, n_center, color='#a050a0', linewidth=1.5,
                    label='noise $\\hat{n}$')
            ax.axhline(0, color='black', linewidth=0.4, alpha=0.5)
            ax.axvspan(950, 1050, alpha=0.08, color='red')
            ax.set_xlabel('Wavelength (nm)')
            ax.set_ylabel(r'$\hat{n}$ (I/F units)')
            ax.set_title('noise spectrum (center pixel)', fontsize=9.5)
            ax.grid(alpha=0.3)
            # Annotate noise magnitude
            ax.text(0.02, 0.95, f'||$\\hat{{n}}$||={np.linalg.norm(n_center):.3f}',
                    transform=ax.transAxes, fontsize=8.5, va='top',
                    color=TEXT_GREY if False else '#666',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='none', alpha=0.85))

            # ─── Col 3: spatial |n_hat| heatmap at 1 µm ────────────────────
            ax = axes[row_i, 2]
            n_at_band = np.abs(n_hat[:, :, diag_band_idx])
            vmax = max(np.abs(n_hat).max() * 0.5, 1e-3)
            im = ax.imshow(n_at_band, cmap='magma', vmin=0, vmax=vmax,
                           interpolation='nearest')
            ax.set_xticks(range(7)); ax.set_yticks(range(7))
            ax.set_xticklabels(range(7), fontsize=7)
            ax.set_yticklabels(range(7), fontsize=7)
            ax.set_xlabel('patch col')
            ax.set_ylabel('patch row')
            ax.set_title(f'|$\\hat{{n}}$| at {diag_band_wl:.0f} nm\n(spatial)',
                         fontsize=9.5)
            plt.colorbar(im, ax=ax, shrink=0.85)

            # ─── Col 4: per-column-of-patch mean |n_hat| ───────────────────
            ax = axes[row_i, 3]
            # column = horizontal position in the patch.
            # If the model captured column-correlated detector artifacts,
            # the mean |n_hat| should vary across columns more than across rows.
            col_means = np.abs(n_hat).mean(axis=(0, 2))    # mean over row & band → (7,)
            row_means = np.abs(n_hat).mean(axis=(1, 2))    # mean over col & band → (7,)
            xs = np.arange(7)
            ax.plot(xs, col_means, 'o-', color='#1f77b4', linewidth=1.7,
                    markersize=6, label='per-col mean')
            ax.plot(xs, row_means, 's-', color='#7d7d7d', linewidth=1.2,
                    markersize=5, alpha=0.7, label='per-row mean (ref)')
            ax.set_xlabel('patch position (col or row index)')
            ax.set_ylabel(r'mean $|\hat{n}|$ over bands+other-axis')
            ax.set_title('column-vs-row structure\nof $|\\hat{n}|$',
                         fontsize=9.5)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        fig.suptitle(
            f'DecompSpVitAdv reconstruction — {ckpt_name}\n'
            'Does the noise branch capture instrumental structure '
            '(1 µm seam, column artifacts)?',
            fontsize=11, y=1.005,
        )
        fig.tight_layout()
        out_path = os.path.join(OUT_DIR, f'fig_v5_decomp_v2_recon_{ckpt_name.split("_")[-1]}.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'  Wrote {out_path}')


# Stub used only if the import-name shadow ever resolves to a defined symbol
TEXT_GREY = '#666'

if __name__ == '__main__':
    main()

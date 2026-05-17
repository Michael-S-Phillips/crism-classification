"""
Visualize the SPEND-style spectral-partition objective on real CRISM pixels.

For three representative pixels (olivine, hcp, plagioclase), show:
  - Col 1: clean center-pixel spectrum (reference)
  - Col 2: input-half bands (gray) vs target-half bands (colored) overlaid
           on the spectrum, for one sample partition
  - Col 3: model's predicted target-band values (line) overlaid on actual
           target-band values (markers) at the same partition
  - Col 4: residual = (prediction − target) per band; should look like
           centered i.i.d. noise (flat, no structure)

Usage:
    conda run -n crism python scripts/figures/fig_spend_partition.py \\
        --checkpoint checkpoints/spatial_mae_spend_128d_6l_best.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, '/mnt/mrdr/crism_classification')
from models.spend_spatial_mae import SpendSpatialSpectralMAE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _utils import (
    CLASS_COLORS, build_mrral_map, find_representative_pixels,
    get_wavelengths_59, load_mrral_parquet, read_patch_from_tile,
)

OUT_PATH = '/mnt/mrdr/crism_classification/reports/v5/fig_v5_spend_partition.png'
CLASSES_TO_SHOW = ['olivine', 'hcp', 'plagioclase']


def load_model(checkpoint_path: str) -> SpendSpatialSpectralMAE:
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    model = SpendSpatialSpectralMAE(
        n_bands=59, patch_size=7,
        embed_dim=cfg.get('embed_dim', 128),
        n_heads=cfg.get('n_heads', 4),
        n_layers=cfg.get('n_layers', 6),
        decoder_dim=cfg.get('decoder_dim', 64),
        decoder_layers=cfg.get('decoder_layers', 2),
        mask_ratio=cfg.get('mask_ratio', 0.75),
        spectral_mask_ratio=cfg.get('spectral_mask_ratio', 0.5),
    )
    model.load_state_dict(ckpt['mae_state'])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to a trained SPEND .pt checkpoint')
    parser.add_argument('--out', type=str, default=OUT_PATH)
    args = parser.parse_args()

    model = load_model(args.checkpoint)

    df = load_mrral_parquet()
    mrral_map = build_mrral_map()
    pixels = find_representative_pixels(df, n_per_class=1, seed=42)
    wls = get_wavelengths_59()

    fig, axes = plt.subplots(
        len(CLASSES_TO_SHOW), 4,
        figsize=(15.5, 3.0 * len(CLASSES_TO_SHOW)),
    )
    if len(CLASSES_TO_SHOW) == 1:
        axes = axes[None, :]

    for row_i, cls in enumerate(CLASSES_TO_SHOW):
        sel = pixels.get(cls, [])
        if not sel:
            continue
        tid, pr, pc = sel[0]
        mrral = mrral_map.get(tid)
        if not (mrral and os.path.exists(mrral)):
            continue

        patch = read_patch_from_tile(mrral, pr, pc, patch_size=7, n_bands=59)
        x_clean = torch.from_numpy(patch).unsqueeze(0).float()
        center_clean = patch[3, 3]

        # Use a fixed partition (evens = target) so the figure is reproducible.
        target_mask = torch.zeros(59, dtype=torch.bool)
        target_mask[torch.arange(0, 59, 2)] = True
        model._partition_bands = lambda device: target_mask.to(device)
        model.spectral_mask_ratio = 0.5

        with torch.no_grad():
            torch.manual_seed(7 + row_i)
            _, recon, _ = model(x_clean)
        recon_center = recon[0, 24].numpy()  # center spatial token

        color = CLASS_COLORS[cls]
        valid = (center_clean > 0) & (center_clean < 0.5)

        # Col 1: clean spectrum
        ax = axes[row_i, 0]
        ax.plot(wls[valid], center_clean[valid], color=color, linewidth=1.8)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('I/F')
        ax.set_title(f'{cls}: tile {tid}\nclean center pixel ({pr},{pc})',
                     color=color, fontsize=9.5)
        ax.grid(alpha=0.3)

        # Col 2: partition view — input vs target bands
        ax = axes[row_i, 1]
        ax.plot(wls[valid], center_clean[valid], color='#bbb',
                linewidth=1.0, alpha=0.6, label='spectrum')
        input_idx = (~target_mask).numpy()
        target_idx = target_mask.numpy()
        ax.scatter(wls[input_idx], center_clean[input_idx],
                   color='#888', s=18, label='input-half (seen)', zorder=3)
        ax.scatter(wls[target_idx], center_clean[target_idx],
                   color=color, s=22, marker='x',
                   label='target-half (predicted)', zorder=4)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('I/F')
        ax.set_title('partition: which bands are\nseen vs predicted',
                     fontsize=9.5)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.3)

        # Col 3: prediction overlay
        ax = axes[row_i, 2]
        target_wls = wls[target_idx]
        actual_targets = center_clean[target_idx]
        pred_targets = recon_center[target_idx]
        ax.scatter(target_wls, actual_targets, color='#666', s=22, marker='x',
                   label='observed', zorder=3)
        ax.plot(target_wls, pred_targets, color=color, linewidth=1.4,
                label='model prediction', zorder=2)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('I/F')
        ax.set_title('target-band reconstruction\n(model vs observation)',
                     fontsize=9.5)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.3)

        # Col 4: residual
        ax = axes[row_i, 3]
        residual = pred_targets - actual_targets
        ax.plot(target_wls, residual, color=color, linewidth=1.2)
        ax.axhline(0, color='black', linewidth=0.4)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('residual (I/F)')
        ax.set_title('prediction − observation\n(should look noise-like)',
                     fontsize=9.5)
        ax.grid(alpha=0.3)

    fig.suptitle(
        'SPEND-style spectral partition — what the model predicts\n'
        '(even bands target, odd bands visible to encoder)',
        fontsize=11, y=1.005,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()

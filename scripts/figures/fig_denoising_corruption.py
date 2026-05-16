"""
Visualize the denoising-MAE corruption pipeline on real CRISM pixels.

For three representative pixels (olivine, hcp, plagioclase), show:
  - the clean input spectrum
  - the corrupted spectrum (model's training view)
  - each corruption component (gauss, spike, column) plotted separately
  - a spatial view of the column-bias contribution at one band

Usage:
    conda run -n crism python scripts/figures/fig_denoising_corruption.py
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, '/mnt/mrdr/crism_classification')
from models.noise_augmentation import CrismNoiseAugmentation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _utils import (
    CLASS_COLORS, build_mrral_map, find_representative_pixels,
    get_wavelengths_59, load_mrral_parquet, read_patch_from_tile,
)

OUT_PATH = '/mnt/mrdr/crism_classification/reports/v5/fig_v5_denoising_corruption.png'
CLASSES_TO_SHOW = ['olivine', 'hcp', 'plagioclase']


def isolate_component(aug, x_clean, which, seed):
    """Forward with only `which` ∈ {'gauss', 'spike', 'column'} enabled. Returns delta."""
    saved = (aug.sigma_gauss, aug.sigma_spike, aug.sigma_column)
    aug.sigma_gauss, aug.sigma_spike, aug.sigma_column = 0.0, 0.0, 0.0
    if which == 'gauss':   aug.sigma_gauss = saved[0]
    if which == 'spike':   aug.sigma_spike = saved[1]
    if which == 'column':  aug.sigma_column = saved[2]
    aug.train()
    torch.manual_seed(seed)
    out = aug(x_clean)
    aug.sigma_gauss, aug.sigma_spike, aug.sigma_column = saved
    return (out - x_clean).numpy()


def main():
    df = load_mrral_parquet()
    mrral_map = build_mrral_map()
    pixels = find_representative_pixels(df, n_per_class=1, seed=42)
    wls = get_wavelengths_59()

    aug = CrismNoiseAugmentation()
    aug.train()
    torch.manual_seed(7)

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
        x_clean = torch.from_numpy(patch).unsqueeze(0)
        center_clean = patch[3, 3]

        torch.manual_seed(7 + row_i)
        x_corrupted = aug(x_clean).numpy()[0]
        center_corrupted = x_corrupted[3, 3]

        color = CLASS_COLORS[cls]

        # Col 1: clean spectrum
        ax = axes[row_i, 0]
        valid = (center_clean > 0) & (center_clean < 0.5)
        ax.plot(wls[valid], center_clean[valid], color=color, linewidth=1.8)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('I/F')
        ax.set_title(f'{cls}: tile {tid}\nclean center pixel ({pr},{pc})',
                     color=color, fontsize=9.5)
        ax.grid(alpha=0.3)

        # Col 2: corrupted spectrum (all components)
        ax = axes[row_i, 1]
        ax.plot(wls[valid], center_clean[valid], color='#aaa',
                linewidth=1.0, label='clean')
        ax.plot(wls[valid], center_corrupted[valid], color=color,
                linewidth=1.5, label='corrupted')
        ax.axvspan(wls[13], wls[17], alpha=0.08, color='red',
                   label='spike region')
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('I/F')
        ax.set_title('clean + all 3 corruptions\n(what the encoder sees)',
                     fontsize=9.5)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.3)

        # Col 3: each corruption component (delta from clean)
        ax = axes[row_i, 2]
        delta_gauss = isolate_component(aug, x_clean, 'gauss', seed=10 + row_i)[0, 3, 3]
        delta_spike = isolate_component(aug, x_clean, 'spike', seed=20 + row_i)[0, 3, 3]
        delta_column = isolate_component(aug, x_clean, 'column', seed=30 + row_i)[0, 3, 3]
        ax.plot(wls, delta_gauss, color='#2c7a2c', linewidth=1.2,
                label=f'Gaussian (σ={aug.sigma_gauss})')
        ax.plot(wls, delta_spike, color='#c44', linewidth=1.5,
                label=f'1 µm spike (σ={aug.sigma_spike})')
        ax.plot(wls, delta_column, color='#1f77b4', linewidth=1.2,
                label=f'column bias (σ={aug.sigma_column})')
        ax.axhline(0, color='black', linewidth=0.4)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('corruption (I/F)')
        ax.set_title('individual corruption components\n(center pixel)', fontsize=9.5)
        ax.legend(fontsize=7.5, loc='lower right')
        ax.grid(alpha=0.3)

        # Col 4: spatial pattern of column-bias at one band
        ax = axes[row_i, 3]
        col_delta_2d = isolate_component(aug, x_clean, 'column', seed=40 + row_i)[0, :, :, 30]
        vmax = max(np.abs(col_delta_2d).max() * 1.05, 1e-3)
        im = ax.imshow(col_delta_2d, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       interpolation='nearest')
        ax.set_xticks(range(7)); ax.set_yticks(range(7))
        ax.set_xlabel('patch col'); ax.set_ylabel('patch row')
        ax.set_title(f'column-bias at {wls[30]:.0f} nm\n(rows uniform within col)',
                     fontsize=9.5)
        plt.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(
        'Denoising-MAE corruption — what the model is asked to remove\n'
        'σ values: gauss=0.0087, 1 µm spike=0.0058, column=0.0049 (data-informed)',
        fontsize=11, y=1.005,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Wrote {OUT_PATH}')


if __name__ == '__main__':
    main()

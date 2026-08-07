"""
Generate fig_v5_mae_reconstruction.png — demonstrate the pre-trained
SpatialSpectralMAE by feeding a real patch through the encoder + decoder
and showing original vs masked vs reconstructed spectra on the masked
pixels.

Usage:
    conda run -n crism python scripts/figures/fig_mae.py
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, '/Volumes/Mars_GIS/CRISM/MRDR/crism_classification')
from models.spatial_mae import SpatialSpectralMAE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _utils import (
    CLASS_COLORS, build_mrral_map, find_representative_pixels,
    get_wavelengths_59, load_mrral_parquet, read_patch_from_tile,
)

OUT_PATH = '/Volumes/Mars_GIS/CRISM/MRDR/crism_classification/reports/v5/fig_v5_mae_reconstruction.png'
MAE_CKPT = '/Volumes/Mars_GIS/CRISM/MRDR/crism_classification/checkpoints/spatial_mae_128d_6l_best.pt'
CLASSES_TO_SHOW = ['olivine', 'hcp', 'plagioclase']
RGB_BANDS = (53, 25, 6)


def main():
    df = load_mrral_parquet()
    mrral_map = build_mrral_map()
    pixels = find_representative_pixels(df, n_per_class=1, seed=42)

    ckpt = torch.load(MAE_CKPT, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    mae = SpatialSpectralMAE(
        n_bands=cfg.get('n_bands', 59),
        patch_size=cfg.get('patch_size', 7),
        embed_dim=cfg.get('embed_dim', 128),
        n_heads=cfg.get('n_heads', 4),
        n_layers=cfg.get('n_layers', 6),
        decoder_dim=cfg.get('decoder_dim', 64),
        decoder_layers=cfg.get('decoder_layers', 2),
        mask_ratio=cfg.get('mask_ratio', 0.75),
        dropout=0.0,
    )
    mae.load_state_dict(ckpt['mae_state'])
    mae.eval()
    print(f'MAE loaded (epoch {ckpt.get("epoch", "?")}, '
          f'training loss {ckpt.get("mae_loss", float("nan")):.6f})')

    wls = get_wavelengths_59()

    n_classes = len(CLASSES_TO_SHOW)
    fig, axes = plt.subplots(
        n_classes, 3, figsize=(13, 3.0 * n_classes),
        gridspec_kw={'width_ratios': [1, 1, 2.2]},
    )
    if n_classes == 1:
        axes = axes[None, :]

    torch.manual_seed(0)  # deterministic mask placement

    for row_i, cls in enumerate(CLASSES_TO_SHOW):
        sel = pixels.get(cls, [])
        if not sel:
            continue
        tid, pr, pc = sel[0]
        mrral = mrral_map.get(tid)
        if not (mrral and os.path.exists(mrral)):
            continue

        patch = read_patch_from_tile(mrral, pr, pc, patch_size=7, n_bands=59)
        x = torch.from_numpy(patch).unsqueeze(0)  # (1, 7, 7, 59)
        # SpatialSpectralMAE expects (B, H, W, C) per the model's encoder.
        # Verify shape compatibility by trying the forward pass.
        with torch.no_grad():
            loss, recon, mask = mae(x)
        recon = recon[0].numpy()              # (49, 59)
        mask = mask[0].numpy().astype(bool)   # (49,)
        x_flat = x[0].reshape(49, 59).numpy() # (49, 59)

        # Panel 1: original patch (false-color, single)
        rgb_orig = patch[:, :, list(RGB_BANDS)].astype(np.float32)
        for i in range(3):
            lo, hi = np.percentile(rgb_orig[:, :, i], (5, 95))
            rgb_orig[:, :, i] = np.clip((rgb_orig[:, :, i] - lo) / max(hi - lo, 1e-6), 0, 1)
        ax = axes[row_i, 0]
        ax.imshow(rgb_orig, interpolation='nearest')
        ax.set_title(f'Original patch\n({cls}, tile {tid})',
                     color=CLASS_COLORS[cls], fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])

        # Panel 2: masked patch — masked pixels shown as gray
        masked_view = rgb_orig.copy()
        mask_2d = mask.reshape(7, 7)
        masked_view[mask_2d] = 0.45  # gray box for masked positions
        ax = axes[row_i, 1]
        ax.imshow(masked_view, interpolation='nearest')
        for r in range(7):
            for c in range(7):
                if mask_2d[r, c]:
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                               fill=False, edgecolor='red',
                                               linewidth=1.2))
        n_masked = int(mask.sum())
        ax.set_title(f'Masked input\n({n_masked} of 49 pixels masked)',
                     fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])

        # Panel 3: spectra — overlay original vs reconstructed for masked pixels
        ax = axes[row_i, 2]
        # Mask short-wavelength noise
        bad = (x_flat[:, :3] < 0).any(axis=1)
        masked_idxs = np.where(mask)[0]
        # Plot every masked pixel — original solid, reconstruction dashed
        for j, idx in enumerate(masked_idxs[:8]):  # cap at 8 for readability
            color = plt.cm.viridis(j / max(1, len(masked_idxs[:8]) - 1))
            o = x_flat[idx]; r = recon[idx]
            ok = (o >= 0) & (o <= 0.5)
            ax.plot(wls[ok], o[ok], color=color, linewidth=1.2, alpha=0.85,
                    label=f'pix {idx}' if j < 4 else None)
            ax.plot(wls[ok], r[ok], color=color, linewidth=1.2, linestyle='--',
                    alpha=0.85)
        # Solid handle for "original" and dashed handle for "recon"
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color='black', linewidth=1.4, label='original'),
            Line2D([0], [0], color='black', linewidth=1.4, linestyle='--',
                   label='reconstruction'),
        ]
        ax.legend(handles=handles, loc='lower right', fontsize=8.5)
        # Compute MSE on the masked pixels only for display
        mse = float(((recon[mask] - x_flat[mask]) ** 2).mean())
        ax.set_title(
            f'Masked-pixel spectra (sample of {min(len(masked_idxs), 8)}/{n_masked})\n'
            f'reconstruction MSE on masked pixels = {mse:.5f}',
            fontsize=9.5,
        )
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('I/F')
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 0.35)

    fig.suptitle(
        'SpatialSpectralMAE reconstruction on held-out patches\n'
        f'(encoder later loaded into the SpatialSpectralClassifier; pre-training '
        f'epoch {ckpt.get("epoch", "?")}, training MSE = {ckpt.get("mae_loss", float("nan")):.5f})',
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {OUT_PATH}')


if __name__ == '__main__':
    main()

"""
Generate fig_v5_example_patch.png — for several mineral classes, show
(left) a tile-context crop with the patch location marked, (middle) the
7×7 spatial patch as a false-color image, (right) the center-pixel and
patch-mean spectra.

Usage:
    conda run -n crism python scripts/figures/fig_patch.py
"""
from __future__ import annotations

import os
import sys

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _utils import (
    CLASS_COLORS, build_mrral_map, find_representative_pixels,
    get_wavelengths_59, load_mrral_parquet, read_patch_from_tile,
)

OUT_PATH = '/mnt/mrdr/crism_classification/reports/v5/fig_v5_example_patch.png'

# CRISM false-color band picks (R, G, B) in the 59-band index range, chosen to
# approximate a visible-IR false color. These correspond to ~2.4, ~1.5, ~0.7 µm.
RGB_BANDS = (53, 25, 6)
CLASSES_TO_SHOW = ['olivine', 'hcp', 'plagioclase']


def crop_tile_context(mrral_path, row, col, half=50):
    """Read a 2*half × 2*half false-color crop around (row, col)."""
    with rasterio.open(mrral_path) as src:
        h, w = src.height, src.width
        r0 = max(row - half, 0); r1 = min(row + half, h)
        c0 = max(col - half, 0); c1 = min(col + half, w)
        win = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
        # Read just the 3 RGB bands
        rgb = np.stack([src.read(b + 1, window=win) for b in RGB_BANDS], axis=-1)
    rgb = rgb.astype(np.float32)
    rgb[rgb >= 65535] = np.nan
    # Per-band percentile stretch (5/98 like the project's standard renderer)
    out = np.zeros_like(rgb)
    for i in range(3):
        b = rgb[:, :, i]
        valid = b[~np.isnan(b)]
        if valid.size == 0:
            continue
        lo, hi = np.percentile(valid, (5, 98))
        out[:, :, i] = np.clip((b - lo) / max(hi - lo, 1e-6), 0, 1)
    out = np.nan_to_num(out, nan=0.0)
    patch_local = (row - r0, col - c0)
    return out, patch_local, (r0, c0)


def main():
    df = load_mrral_parquet()
    mrral_map = build_mrral_map()
    pixels = find_representative_pixels(df, n_per_class=1, seed=42)

    n_rows = len(CLASSES_TO_SHOW)
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(13, 3.4 * n_rows),
        gridspec_kw={'width_ratios': [1.1, 1, 1.8]},
    )
    if n_rows == 1:
        axes = axes[None, :]
    wls = get_wavelengths_59()

    for row_i, cls in enumerate(CLASSES_TO_SHOW):
        sel = pixels.get(cls, [])
        if not sel:
            print(f'  no pixel found for class {cls}, skipping row')
            continue
        tid, pr, pc = sel[0]
        mrral = mrral_map.get(tid)
        if not mrral or not os.path.exists(mrral):
            print(f'  tile {tid} not found locally, skipping {cls}')
            continue

        # Tile context
        rgb, (lr, lc), (r0, c0) = crop_tile_context(mrral, pr, pc, half=60)
        ax = axes[row_i, 0]
        ax.imshow(rgb, origin='upper')
        rect = patches.Rectangle((lc - 3.5, lr - 3.5), 7, 7,
                                 fill=False, edgecolor=CLASS_COLORS[cls],
                                 linewidth=2.5)
        ax.add_patch(rect)
        ax.set_title(f'{cls}: tile {tid}\n7×7 patch @ ({pr}, {pc})',
                     color=CLASS_COLORS[cls], fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        # 7x7 patch as false-color
        patch = read_patch_from_tile(mrral, pr, pc, patch_size=7, n_bands=59)
        rgb_patch = patch[:, :, list(RGB_BANDS)].astype(np.float32)
        # Per-band stretch within the patch
        rgb_norm = np.zeros_like(rgb_patch)
        for i in range(3):
            b = rgb_patch[:, :, i]
            lo, hi = np.percentile(b, (5, 95))
            rgb_norm[:, :, i] = np.clip((b - lo) / max(hi - lo, 1e-6), 0, 1)
        ax = axes[row_i, 1]
        ax.imshow(rgb_norm, origin='upper', interpolation='nearest')
        # Mark center pixel
        ax.plot(3, 3, 'x', color=CLASS_COLORS[cls], markersize=14, markeredgewidth=2.5)
        ax.set_title('7×7 patch (false color)', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        # Spectra
        ax = axes[row_i, 2]
        center_spec = patch[3, 3, :]
        mean_spec = patch.mean(axis=(0, 1))
        # Mask the noise zone outside CRISM's useful range — set NaN to skip plotting
        bad_mask = (center_spec < 0) | (center_spec > 0.5)
        center_plot = np.where(bad_mask, np.nan, center_spec)
        mean_plot = np.where(bad_mask, np.nan, mean_spec)
        ax.plot(wls, center_plot, color=CLASS_COLORS[cls], linewidth=2.2,
                label='center pixel (3,3)')
        ax.plot(wls, mean_plot, color=CLASS_COLORS[cls], linewidth=1.5,
                linestyle='--', label='patch mean', alpha=0.75)
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Reflectance (I/F)')
        ax.set_title('Spectra')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(alpha=0.3)
        ax.set_ylim(0, max(0.3, np.nanmax(mean_plot) * 1.1) if np.any(~bad_mask) else 0.3)

    fig.suptitle(
        'Example 7×7 spatial patches and spectra by mineral class\n'
        f'patches selected from confidently-positive (label > 0.9, High-confidence) pixels',
        fontsize=11, y=1.005,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {OUT_PATH}')


if __name__ == '__main__':
    main()

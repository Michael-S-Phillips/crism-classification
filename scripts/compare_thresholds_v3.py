"""
Compare two confidence-thresholding strategies on the v3 (denoising) classifier
output for the Nili Fossae test tiles (t1249, t1250, t1321, t1322).

Strategy A — uniform: probability >= 0.5 for all classes.
Strategy B — inverse-proportional to val_AP:
    threshold = 0.99 − 0.49 × val_AP
    (val_AP from the ft_v3_denoising_lrscale001 best checkpoint:
     olivine 0.860, lcp 0.742, hcp 0.440, plagioclase 0.093, other 0.787.)

For each tile, loads the per-pixel probabilities saved by
`scripts/classify_tile_supervised.py --save_probs`, applies both threshold
sets, and produces:
  - Per-pixel best-class maps (argmax of probs over classes that passed
    threshold; "no detection" where no class passed)
  - A summary figure: 4 tiles × 4 cols (RGB, uniform map, inverse map,
    pixel-level "what changed" map)
  - A per-class detection-area summary printed to stdout

Outputs:
  reports/v5/fig_v5_nili_threshold_comparison.png

Usage (no args needed):
    conda run -n crism python scripts/compare_thresholds_v3.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap

PROJECT_ROOT = '/mnt/mrdr/crism_classification'

# Per-class val_AP for the best v3 denoising classifier (ft_v3_denoising_lrscale001).
# Pulled from wandb 2026-05-19.
VAL_AP = {
    'olivine':     0.860,
    'lcp':         0.742,
    'hcp':         0.440,
    'plagioclase': 0.093,
    'other':       0.787,
}
CLASS_NAMES = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']
CLASS_COLORS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#aaaaaa']  # 5 classes
NO_DETECT_COLOR = '#ffffff'  # white for "no class passed threshold"

# Threshold strategies (in the order classes appear in CLASS_NAMES)
UNIFORM_THRESH = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
INVERSE_THRESH = np.array([0.99 - 0.49 * VAL_AP[c] for c in CLASS_NAMES])

# Tiles to compare (Nili Fossae MC13)
TILES = ['t1249', 't1250', 't1321', 't1322']
PROBS_DIR = '/tmp/v3_nili'
TILE_DIR = '/mnt/mrdr/mc13'
TILE_MRRAL = {
    't1249': f'{TILE_DIR}/t1249_mrral_20n073_0327_4.img',
    't1250': f'{TILE_DIR}/t1250_mrral_20n078_0327_4.img',
    't1321': f'{TILE_DIR}/t1321_mrral_25n073_0327_4.img',
    't1322': f'{TILE_DIR}/t1322_mrral_25n078_0327_4.img',
}

OUT_PATH = os.path.join(PROJECT_ROOT, 'reports', 'v5',
                        'fig_v5_nili_threshold_comparison.png')


def best_class_under_threshold(probs_hwc: np.ndarray, valid_mask: np.ndarray,
                                threshold: np.ndarray) -> np.ndarray:
    """Per-pixel argmax over classes that meet their threshold.

    Returns (H, W) int8 array:
        -1 → invalid pixel
         5 → no class passed threshold (will render as 'no detection')
         0..4 → class index of the highest-prob class that passed
    """
    H, W, C = probs_hwc.shape
    above = probs_hwc >= threshold[None, None, :]      # (H, W, C) bool
    # Mask the probs so argmax only considers classes passing threshold
    masked = np.where(above, probs_hwc, -1.0)
    best = np.argmax(masked, axis=-1).astype(np.int8)  # (H, W)
    any_passed = above.any(axis=-1)                    # (H, W)
    out = np.full((H, W), -1, dtype=np.int8)
    out[valid_mask] = 5                                # default: no detection
    out[valid_mask & any_passed] = best[valid_mask & any_passed]
    return out


def load_rgb(mrral_path: str) -> np.ndarray:
    """False-color RGB from selected mrral bands. Returns (H, W, 3) uint8."""
    with rasterio.open(mrral_path) as src:
        # Use bands roughly at 2.2 µm / 1.5 µm / 1.1 µm — typical CRISM continuum
        r = src.read(46).astype(np.float32)
        g = src.read(33).astype(np.float32)
        b = src.read(20).astype(np.float32)
    nodata = (r >= 65535) | (g >= 65535) | (b >= 65535)
    img = np.stack([r, g, b], axis=-1)
    img[nodata] = 0
    p2, p98 = np.percentile(img[~nodata], [2, 98])
    img = np.clip((img - p2) / max(1e-6, p98 - p2), 0, 1)
    return (img * 255).astype(np.uint8)


def class_map_to_rgb(class_idx: np.ndarray) -> np.ndarray:
    """(H, W) class-index map → (H, W, 3) uint8 RGB.
    -1: black (invalid), 0..4: class color, 5: white (no detection)."""
    colors = CLASS_COLORS + [NO_DETECT_COLOR]
    rgb = np.zeros((*class_idx.shape, 3), dtype=np.uint8)
    for i, hex_c in enumerate(colors):
        r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
        m = class_idx == i
        rgb[m] = (r, g, b)
    # invalid stays black
    return rgb


def changed_map_rgb(uniform_idx: np.ndarray, inverse_idx: np.ndarray) -> np.ndarray:
    """RGB map highlighting pixels whose prediction CHANGED between strategies.

    - Same class (and not invalid): light gray
    - Dropped to 'no detection' under stricter inverse: darker gray
    - Changed class entirely: colored by the NEW (inverse) class
    - Invalid: black
    """
    out = np.zeros((*uniform_idx.shape, 3), dtype=np.uint8)
    invalid = uniform_idx == -1
    same = (uniform_idx == inverse_idx) & ~invalid
    dropped = (~invalid) & (uniform_idx != 5) & (inverse_idx == 5)
    changed = (~invalid) & ~same & ~dropped

    out[same] = (235, 235, 235)
    out[dropped] = (90, 90, 90)
    for ci, hex_c in enumerate(CLASS_COLORS):
        r, g, b = int(hex_c[1:3], 16), int(hex_c[3:5], 16), int(hex_c[5:7], 16)
        m = changed & (inverse_idx == ci)
        out[m] = (r, g, b)
    return out


def main():
    print('Inverse-proportional thresholds:')
    for c, t in zip(CLASS_NAMES, INVERSE_THRESH):
        print(f'  {c:<12}  val_AP={VAL_AP[c]:.3f}  threshold={t:.3f}')
    print()

    fig, axes = plt.subplots(len(TILES), 4, figsize=(16, 4 * len(TILES)))
    if len(TILES) == 1:
        axes = axes[None, :]

    summary_rows = []

    for row_i, tid in enumerate(TILES):
        probs_path = os.path.join(PROBS_DIR, f'{tid}_probs.npz')
        if not os.path.exists(probs_path):
            print(f'SKIP {tid}: missing {probs_path}')
            continue

        data = np.load(probs_path)
        probs = data['probs'].astype(np.float32)        # (H, W, 5)
        valid_mask = data['valid_mask']                  # (H, W)

        uniform_idx = best_class_under_threshold(probs, valid_mask, UNIFORM_THRESH)
        inverse_idx = best_class_under_threshold(probs, valid_mask, INVERSE_THRESH)

        # Per-class detection counts
        n_valid = int(valid_mask.sum())
        for ci, cname in enumerate(CLASS_NAMES):
            u_count = int((uniform_idx == ci).sum())
            i_count = int((inverse_idx == ci).sum())
            summary_rows.append({
                'tile': tid, 'class': cname,
                'uniform': u_count, 'inverse': i_count,
                'uniform_pct': 100.0 * u_count / n_valid,
                'inverse_pct': 100.0 * i_count / n_valid,
            })
        # also count "no detection"
        u_nodet = int((uniform_idx == 5).sum())
        i_nodet = int((inverse_idx == 5).sum())
        summary_rows.append({
            'tile': tid, 'class': '(no detection)',
            'uniform': u_nodet, 'inverse': i_nodet,
            'uniform_pct': 100.0 * u_nodet / n_valid,
            'inverse_pct': 100.0 * i_nodet / n_valid,
        })

        rgb = load_rgb(TILE_MRRAL[tid])
        unif_rgb = class_map_to_rgb(uniform_idx)
        inv_rgb = class_map_to_rgb(inverse_idx)
        diff_rgb = changed_map_rgb(uniform_idx, inverse_idx)

        ax = axes[row_i, 0]
        ax.imshow(rgb)
        ax.set_title(f'{tid}  false color (RGB ≈ 2.2/1.5/1.1 µm)', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[row_i, 1]
        ax.imshow(unif_rgb)
        ax.set_title('uniform threshold = 0.5', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[row_i, 2]
        ax.imshow(inv_rgb)
        ax.set_title('inverse-proportional thresholds', fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[row_i, 3]
        ax.imshow(diff_rgb)
        ax.set_title('changed → colored by NEW class\n(dropped → dark gray)',
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    # Legend
    legend_handles = [
        mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
        for i in range(len(CLASS_NAMES))
    ] + [
        mpatches.Patch(color=NO_DETECT_COLOR, label='no detection', edgecolor='black'),
        mpatches.Patch(color=(0.92, 0.92, 0.92), label='unchanged', edgecolor='black'),
        mpatches.Patch(color=(0.35, 0.35, 0.35), label='dropped (under stricter τ)'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=8, fontsize=8,
               bbox_to_anchor=(0.5, -0.01), frameon=False)

    fig.suptitle(
        'v3 denoising classifier — Nili Fossae test tiles\n'
        'uniform 0.5 vs val_AP-informed inverse-proportional thresholds',
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\nWrote {OUT_PATH}')

    # Pretty-print summary
    print('\nPer-tile, per-class detection summary (% of valid pixels):')
    print(f'  {"tile":<7} {"class":<16} {"uniform":>10} {"inverse":>10}')
    for r in summary_rows:
        print(f'  {r["tile"]:<7} {r["class"]:<16} '
              f'{r["uniform_pct"]:>9.2f}% {r["inverse_pct"]:>9.2f}%')


if __name__ == '__main__':
    main()

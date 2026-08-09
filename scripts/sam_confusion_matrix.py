"""Spectral-angle (SAM) confusion matrix across the hand-core dataset's classes.

Answers "which classes are actually distinguishable by their spectra?" — as
opposed to "which does the model get right", which confounds spectra with
architecture, loss and class balance.

Method
  1. Sample pixels per class from EXACTLY the sources the hand-core policy
     admits (scripts/sample_class_spectra.py writes that .npz).
  2. Continuum-remove, and drop the 4 detector-overlap bands, which CR sets to
     1.0 by construction — leaving them in adds a constant to every spectrum and
     biases every angle toward zero.
  3. Split each class 50/50. Build the endmember (median spectrum) from the fit
     half; classify the held-out half. Without this split each class matches its
     own endmember trivially and the diagonal is meaningless.
  4. Assign each held-out pixel to the minimum-angle endmember. Rows are the
     true class, columns the assignment, row-normalised -> recall on the
     diagonal.

SAM is scale-invariant, so it compares band SHAPE and ignores overall
brightness. That matters here because the sources differ in albedo (MTRDR plag
sits ~2x brighter than hand-labelled plag), which a Euclidean metric conflates
with a genuine spectral difference.

Usage
    python scripts/sample_class_spectra.py          # writes the .npz
    python scripts/sam_confusion_matrix.py --npz <path> [--out reports/sam_confusion.png]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.continuum_removal import continuum_removed, good_band_mask_59  # noqa: E402
from sam_analysis.sam import spectral_angle  # noqa: E402

NODATA = 65535.0
PHYS_MAX = 1.0
CLIP_MAX = 0.5

# Which sampled series to treat as a class, in display order. Both plagioclase
# sources are carried separately on purpose — whether they behave as one class
# is exactly the question.
SERIES = [
    ('olivine', 'olivine__hand'),
    ('lcp', 'lcp__hand'),
    ('hcp', 'hcp__hand'),
    ('plag(hand)', 'plagioclase__hand'),
    ('plag(MTRDR)', 'plagioclase__MTRDR'),
    ('alteration', 'alteration__hand'),
    ('bland', 'bland__v3 review'),
    ('junk', 'junk__v3 review'),
]


def prep(a: np.ndarray, good: np.ndarray) -> np.ndarray:
    """Mask nodata exactly as the training reader does, CR, keep good bands."""
    a = a.astype(np.float32).copy()
    a[(a > PHYS_MAX) | (a == NODATA) | (~np.isfinite(a))] = np.nan
    a = np.clip(a, 0.0, CLIP_MAX)
    a = a[np.isfinite(a).all(axis=1)]
    cr = continuum_removed(a[:, None, None, :].copy())[:, 0, 0, :]
    return cr[:, good]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--npz', required=True, help='output of sample_class_spectra.py')
    ap.add_argument('--out', default='reports/sam_confusion.png')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    d = np.load(args.npz)
    good = good_band_mask_59()
    rng = np.random.default_rng(args.seed)

    names, fit, test = [], [], []
    for label, key in SERIES:
        if key not in d.files:
            print(f'  skip {label}: {key} not in npz')
            continue
        X = prep(d[key], good)
        if len(X) < 20:
            print(f'  skip {label}: only {len(X)} usable spectra')
            continue
        idx = rng.permutation(len(X))
        h = len(X) // 2
        names.append(label)
        fit.append(X[idx[:h]])
        test.append(X[idx[h:]])

    ends = np.stack([np.nanmedian(f, axis=0) for f in fit])          # (C, B)
    C = len(names)

    conf = np.zeros((C, C), dtype=np.int64)
    mean_ang = np.zeros((C, C), dtype=np.float64)
    for i, X in enumerate(test):
        angs = np.stack([spectral_angle(X, ends[j]) for j in range(C)], axis=1)
        mean_ang[i] = np.nanmean(np.degrees(angs), axis=0)
        pred = np.nanargmin(angs, axis=1)
        for j in range(C):
            conf[i, j] = int((pred == j).sum())

    row = conf.sum(axis=1, keepdims=True).clip(min=1)
    recall = conf / row

    w = max(len(n) for n in names) + 2
    print('\nSAM confusion — rows = true class, cols = nearest endmember '
          '(row-normalised, so the diagonal is recall)\n')
    print(' ' * w + ''.join(f'{n:>13}' for n in names))
    for i, n in enumerate(names):
        cells = ''.join(
            (f'{recall[i, j]:>12.2f}*' if i == j else f'{recall[i, j]:>13.2f}')
            for j in range(C))
        print(f'{n:<{w}}{cells}')
    print('\n(* = diagonal / recall)')

    print('\nmean spectral angle to each endmember, degrees '
          '(lower = more similar):\n')
    print(' ' * w + ''.join(f'{n:>13}' for n in names))
    for i, n in enumerate(names):
        print(f'{n:<{w}}' + ''.join(f'{mean_ang[i, j]:>13.2f}' for j in range(C)))

    print('\nper-class recall, worst first:')
    order = np.argsort(np.diag(recall))
    for i in order:
        conf_with = int(np.argmax(np.where(np.arange(C) == i, -1, recall[i])))
        print(f'  {names[i]:<13}{np.diag(recall)[i]:.2f}   '
              f'most confused with {names[conf_with]} ({recall[i, conf_with]:.2f})')

    _plot(recall, names, args.out)
    print(f'\nwrote {args.out}')


def _plot(recall: np.ndarray, names: list[str], out: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # Sequential = ONE hue, light -> dark. Never a rainbow: recall is a
    # magnitude, and a multi-hue ramp invents category boundaries that the data
    # does not have.
    cmap = LinearSegmentedColormap.from_list(
        'seq', ['#f7fbff', '#c6dbef', '#6baed6', '#2171b5', '#08306b'])
    INK, MUTED, GRID = '#1a1a1a', '#5c5c5c', '#d8d8d8'

    n = len(names)
    fig, ax = plt.subplots(figsize=(1.15 * n + 3.2, 1.05 * n + 2.4))
    fig.patch.set_facecolor('white')
    ax.imshow(recall, cmap=cmap, vmin=0, vmax=1)

    for i in range(n):
        for j in range(n):
            v = recall[i, j]
            if v < 0.005:
                continue
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=10,
                    color='white' if v > 0.55 else INK,
                    fontweight='600' if i == j else 'normal')
    # 2px surface gap between cells, and a ring on the diagonal
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', length=0)
    # Diagonal ring drawn last, above the cell fills and the white gap grid.
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor='#e6550d', linewidth=2.5,
                                   zorder=5, clip_on=False))

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=10, color=INK)
    ax.set_yticklabels(names, fontsize=10, color=INK)
    ax.set_xlabel('assigned to (nearest endmember by spectral angle)',
                  fontsize=10.5, color=MUTED, labelpad=12)
    ax.set_ylabel('true class', fontsize=10.5, color=MUTED, labelpad=9)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_title('Spectral-angle confusion — hand-core classes',
                 fontsize=14.5, color=INK, fontweight='600', loc='left', pad=14)
    # Caption sits below the axis label, not on top of it.
    fig.text(0.01, -0.02,
             'Endmembers fit on one half of each class; the held-out half is '
             'classified. Orange ring marks the diagonal (recall).\n'
             'SAM compares band shape and ignores brightness. '
             'Detector-overlap bands excluded.',
             fontsize=9.5, color=MUTED, va='top', ha='left', linespacing=1.6)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor='white', bbox_inches='tight')


if __name__ == '__main__':
    main()

"""Do a dual-CR cache's two 59-band halves actually differ?

The failure this exists to catch: hull-CR written into BOTH halves with linear-CR
never computed. That produces finite floats, the correct (.., 118) shape, and a
``shard_index.json`` saying ``dual=True`` -- so every other check in the pipeline
passes. The pretrain then learns a duplicated hull-CR representation, the
fine-tune inherits it, and the experiment reads as the hypothesis being
FALSIFIED rather than as a build bug. It was a real possibility: the ``--dual``
global-cache tests originally asserted only ``shape[-1] == 118`` and
``isfinite``, and a mutation duplicating hull into both halves passed all of them.

Two on-disk formats, and they are NOT interchangeable:

  * global pretrain shards -- ``np.save`` (build_global_patch_cache.py:302), so
    each ``.npy`` carries a 128-byte header and ``np.load(mmap_mode='r')`` knows
    its own shape. This is the default mode here.
  * labeled fine-tune cache -- a headerless ``np.memmap``
    (build_cr_labeled_cache.py:86), deliberately, because
    ``CRISMSpectralPatchDataset`` memmaps it with an exact expected shape. There
    is no header to read, so ``--headerless`` requires the channel count and
    infers the row count from the file size. Note a byte count alone CANNOT
    distinguish 59 channels from 118, since ``59 * 2k == 118 * k`` -- so pass
    ``--parquet`` to cross-check the inferred row count against the labels.

Usage
    # global pretrain cache (a directory of shards, or one .npy)
    python scripts/check_dual_cache.py /xdisk/sbyrne/phillipsm/crism_patch_cache_dualcr

    # labeled fine-tune cache (headerless memmap)
    python scripts/check_dual_cache.py --headerless \
        <data>/patch_cache_handcore_dualcr/mrral_train_patches_p7.npy

Exit status is 1 if any file fails, so this is usable as a job gate.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.continuum_removal import CR_SCALES  # noqa: E402

N_BANDS = 59
SAMPLE_ROWS = 256


def _open(path: str, headerless: bool, n_ch: int, patch: int):
    """Return a read-only array view, honouring the two on-disk formats."""
    if not headerless:
        return np.load(path, mmap_mode='r')
    itemsize = patch * patch * n_ch * 4
    nbytes = os.path.getsize(path)
    if nbytes % itemsize:
        raise SystemExit(
            f'{path}: {nbytes:,} bytes is not a whole number of '
            f'{patch}x{patch}x{n_ch} float32 patches ({itemsize:,} each). '
            f'If this file has a 128-byte .npy header, drop --headerless.')
    return np.memmap(path, dtype='float32', mode='r',
                     shape=(nbytes // itemsize, patch, patch, n_ch))


def _sample(arr, n_rows: int) -> np.ndarray:
    """Evenly spread rows — deterministic, and only these rows are read."""
    idx = np.unique(np.linspace(0, len(arr) - 1, min(len(arr), n_rows)).astype(np.int64))
    return np.asarray(arr[idx], dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='+',
                    help='.npy file(s), or a directory of global_patches_*.npy')
    ap.add_argument('--headerless', action='store_true',
                    help='raw np.memmap with no .npy header (the labeled cache)')
    ap.add_argument('--n_channels', type=int, default=2 * N_BANDS,
                    help='channels; only used by --headerless (default 118)')
    ap.add_argument('--patch_size', type=int, default=7)
    ap.add_argument('--parquet', default=None,
                    help='cross-check the row count against this parquet')
    ap.add_argument('--sample_rows', type=int, default=SAMPLE_ROWS)
    args = ap.parse_args()

    files: list[str] = []
    for p in args.paths:
        if os.path.isdir(p):
            hits = sorted(glob.glob(os.path.join(p, 'global_patches_*.npy')))
            if not hits:
                raise SystemExit(f'no global_patches_*.npy under {p}')
            files += hits
        else:
            files.append(p)

    want_rows = None
    if args.parquet:
        import pandas as pd
        want_rows = len(pd.read_parquet(args.parquet, columns=['split']))

    bad = 0
    for path in files:
        name = os.path.basename(path)
        arr = _open(path, args.headerless, args.n_channels, args.patch_size)
        if arr.shape[-1] != 2 * N_BANDS:
            print(f'FAIL {name}: {arr.shape[-1]} channels, expected 118')
            bad += 1
            continue

        s = _sample(arr, args.sample_rows)
        hull, lin = s[..., :N_BANDS], s[..., N_BANDS:]

        problems = []
        # The load-bearing check. atol is loose on purpose: a duplicated
        # transform is bit-identical, so anything near-equal is the bug.
        if np.allclose(hull, lin, atol=1e-6):
            problems.append('halves identical (linear-CR never computed?)')
        if not np.isfinite(s).all():
            problems.append('non-finite values')
        # Channel ORDER, which sign cannot detect: linear-CR is clipped to
        # [0, 2], so both halves are non-negative after standardisation. Level
        # does detect it. Both transforms are ratios concentrated near 1.0, and
        # each is divided by its OWN std, so the halves land at distinct
        # predictable levels -- ~1/hull_std vs ~1/linear_std, a 2.4x gap. If a
        # half sits closer to the other's expected level, the two were written
        # in the wrong order (hull must occupy 0-58).
        h50, l50 = float(np.median(hull)), float(np.median(lin))
        exp_h, exp_l = 1.0 / CR_SCALES['hull_std'], 1.0 / CR_SCALES['linear_std']
        if abs(h50 - exp_l) < abs(h50 - exp_h) and abs(l50 - exp_h) < abs(l50 - exp_l):
            problems.append(
                f'halves look SWAPPED: hull p50 {h50:.2f} is nearer the linear '
                f'level {exp_l:.2f} than the hull level {exp_h:.2f}, and linear '
                f'p50 {l50:.2f} is nearer {exp_h:.2f}. Channels 0-58 must be hull.')

        corr = float(np.corrcoef(hull.ravel(), lin.ravel())[0, 1])
        status = 'FAIL' if problems else 'ok  '
        print(f'{status} {name}  n={len(arr):,}  '
              f'hull p50={h50:7.3f} std={hull.std():.3f}  '
              f'linear p50={l50:7.3f} std={lin.std():.3f}  '
              f'corr={corr:+.3f}')
        for m in problems:
            print(f'       -> {m}')
        if problems:
            bad += 1

        if want_rows is not None and len(arr) != want_rows:
            print(f'       -> row count {len(arr):,} != parquet {want_rows:,}')
            bad += 1

    total = sum(len(_open(f, args.headerless, args.n_channels, args.patch_size))
                for f in files)
    print(f'\n{len(files)} file(s), {total:,} patches, {bad} bad')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()

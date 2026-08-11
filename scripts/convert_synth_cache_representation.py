#!/usr/bin/env python
"""Convert a RAW 59-band synth patch cache to a continuum-removed representation.

The MTRDR plagioclase patch caches on disk are RAW reflectance:

    data/patch_cache/mtrdr_plag_patches_p7.npy      (8671, 7, 7, 59)  p50 0.239
    data/contrastive/extra_plag_roi/patches.npy     (1817, 7, 7, 59)  p50 0.225

SyntheticPatchDataset serves them VERBATIM (no transform), so
``train.py --continuum_removed --cache_is_cr --synth_train_cache <raw>`` trained
plagioclase on raw patches while every other class got hull-CR patches -- a ~4x
level offset on exactly one class, which is trivially separable in validation and
then over-fires at inference where the whole tile is CR. This script produces the
matching-representation copy:

    --mode hull  ->  continuum_removed(x)                  (N, P, P, 59)
    --mode dual  ->  dual_continuum(x, standardize=True)   (N, P, P, 118)

ROW ORDER AND ROW COUNT ARE PRESERVED EXACTLY. The caches are aligned
row-for-row with a parquet (``SyntheticPatchDataset`` asserts
``cache.shape[0] == len(parquet)``), so a reorder or a dropped row would
silently mislabel plagioclase patches rather than fail.

Usage:
    python scripts/convert_synth_cache_representation.py \
        --input  data/patch_cache/mtrdr_plag_patches_p7.npy \
        --output /somewhere/mtrdr_plag_patches_p7_dual.npy \
        --mode dual
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.continuum_removal import (                                # noqa: E402
    RAW_LEVEL_MAX, continuum_removed, detect_representation, dual_continuum,
    sample_level)

# The pipeline's raw-patch hazards, copied from CRISMSpectralPatchDataset
# (data/dataset.py:440-442) rather than re-invented: 65535 is the CRISM NODATA
# sentinel, reflectance (I/F) above 1.0 is corrupt (the 410 nm blue edge carries
# spikes to ~1180 I/F), and valid patches are clipped to [0, 0.5].
NODATA = 65535.0
PHYS_MAX = 1.0
CLIP_MAX = 0.5

MODE_CHANNELS = {'hull': 59, 'dual': 118}


def sanitize(chunk: np.ndarray) -> tuple[np.ndarray, int]:
    """Apply the pipeline's raw-patch nodata policy. Returns (clean, n_flagged).

    Identical to CRISMSpectralPatchDataset.__getitem__: NODATA / non-finite /
    physically-impossible (> 1.0 I/F) values are set to 0.0 FIRST -- not clipped,
    which would cap a 1180 I/F spike to a plausible-looking 0.5 -- then the
    remainder is clipped to [0, CLIP_MAX]. Rows are never dropped (that would
    break parquet alignment); a fully-zeroed spectrum is degenerate and both
    continuum removals map it to 1.0, the pipeline's documented behaviour.
    """
    out = np.array(chunk, dtype=np.float32, copy=True)
    bad = (out == NODATA) | ~np.isfinite(out) | (out > PHYS_MAX)
    n_flagged = int(bad.sum())
    out[bad] = 0.0
    np.clip(out, 0.0, CLIP_MAX, out=out)
    return out, n_flagged


def transform_chunk(chunk: np.ndarray, mode: str) -> tuple[np.ndarray, int]:
    """sanitize + the requested transform. chunk: (n, P, P, 59)."""
    clean, n_flagged = sanitize(chunk)
    if mode == 'hull':
        return continuum_removed(clean), n_flagged
    if mode == 'dual':
        return dual_continuum(clean, standardize=True), n_flagged
    raise ValueError(f'unknown mode {mode!r}; expected one of {sorted(MODE_CHANNELS)}')


def describe(arr, n_rows: int = 256) -> dict:
    """Sampled summary statistics of a patch cache (memmap-safe)."""
    from data.continuum_removal import _sample_row_idx
    idx = _sample_row_idx(len(arr), n_rows)
    v = np.asarray(arr[idx], dtype=np.float64)
    flat = v.reshape(-1, v.shape[-1])
    finite = flat[np.isfinite(flat)]
    out = {
        'shape': tuple(arr.shape),
        'n_sampled_rows': int(len(idx)),
        'p50': float(np.median(finite)),
        'mean': float(finite.mean()),
        'min': float(finite.min()),
        'max': float(finite.max()),
        'frac_ge_0.9': float((finite >= 0.9).mean()),
        'n_nonfinite': int(flat.size - finite.size),
    }
    if arr.shape[-1] == 118:
        hull, lin = flat[:, :59], flat[:, 59:]
        out['hull_block'] = {'p50': float(np.median(hull)),
                             'std': float(hull.std()),
                             'min': float(hull.min()), 'max': float(hull.max())}
        out['linear_block'] = {'p50': float(np.median(lin)),
                               'std': float(lin.std()),
                               'min': float(lin.min()), 'max': float(lin.max())}
    return out


def _fmt(stats: dict) -> str:
    parts = [f"shape={stats['shape']}",
             f"p50={stats['p50']:.4f}", f"mean={stats['mean']:.4f}",
             f"min={stats['min']:.4f}", f"max={stats['max']:.4f}",
             f"frac>=0.9={stats['frac_ge_0.9']:.4f}",
             f"nonfinite={stats['n_nonfinite']}"]
    lines = ['  ' + '  '.join(parts)]
    for blk in ('hull_block', 'linear_block'):
        if blk in stats:
            b = stats[blk]
            lines.append(f"  {blk:<13} p50={b['p50']:.4f} std={b['std']:.4f} "
                         f"min={b['min']:.4f} max={b['max']:.4f}")
    return '\n'.join(lines)


def convert(input_path: str, output_path: str, mode: str,
            chunk_rows: int = 512, force: bool = False,
            log=print) -> dict:
    """Transform a raw 59-band cache into `mode`'s representation, chunk by chunk.

    Returns {'before': stats, 'after': stats, 'n_flagged': int}. Refuses to run
    on input that is already transformed, and refuses to clobber an existing
    output unless `force`.
    """
    if mode not in MODE_CHANNELS:
        raise ValueError(f'unknown mode {mode!r}; expected one of {sorted(MODE_CHANNELS)}')
    if os.path.exists(output_path) and not force:
        raise FileExistsError(
            f'{output_path} exists; pass --force to overwrite. Refusing to '
            f'clobber a cache another job may be aligned against.')

    src = np.load(input_path, mmap_mode='r')
    if src.ndim != 4:
        raise ValueError(f'expected a (N, P, P, C) cache, got shape {src.shape}')
    n, ph, pw, n_ch = src.shape
    if ph != pw:
        raise ValueError(f'expected square patches, got {ph}x{pw}')

    # ── Refuse already-transformed input ────────────────────────────────────
    # 118 channels is dual by construction; at 59 the level decides. Both
    # transforms are idempotent-looking but NOT idempotent: hull-CR of a hull-CR
    # spectrum is not the identity, and dual of a dual is a shape error only if
    # the width happens to disagree. A robust median over a spread sample of
    # rows separates the states by ~6x the relevant spread (see RAW_LEVEL_MAX).
    if n_ch != 59:
        raise ValueError(
            f'{input_path} has {n_ch} channels, not 59: this converter takes a '
            f'RAW 59-band cache. A 118-channel cache is already dual-CR.')
    level = sample_level(src)
    if level > RAW_LEVEL_MAX:
        raise ValueError(
            f'{input_path} looks ALREADY CONTINUUM-REMOVED, not raw: median '
            f'level {level:.4f} over {min(n, 256)} sampled rows is above '
            f'RAW_LEVEL_MAX={RAW_LEVEL_MAX} (raw patches are clipped to '
            f'[0, {CLIP_MAX}] so they cannot exceed it; hull-CR centres at '
            f'0.934). Refusing to double-transform. Pass the RAW cache.')

    before = describe(src)
    log(f'input  {input_path}  ({detect_representation(src)})')
    log(_fmt(before))

    out_ch = MODE_CHANNELS[mode]
    tmp_path = output_path + '.partial'
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    dst = np.lib.format.open_memmap(
        tmp_path, mode='w+', dtype=np.float32, shape=(n, ph, pw, out_ch))
    n_flagged = 0
    try:
        for start in range(0, n, chunk_rows):
            stop = min(start + chunk_rows, n)
            block, flagged = transform_chunk(np.asarray(src[start:stop]), mode)
            assert block.shape == (stop - start, ph, pw, out_ch), block.shape
            # Row i of the chunk lands at row start+i: contiguous, in order,
            # every row written exactly once.
            dst[start:stop] = block
            n_flagged += flagged
            log(f'  rows {start}-{stop - 1} of {n}')
        dst.flush()
    finally:
        del dst
    os.replace(tmp_path, output_path)

    after = np.load(output_path, mmap_mode='r')
    assert after.shape == (n, ph, pw, out_ch), after.shape
    stats_after = describe(after)
    log(f'output {output_path}  ({mode})')
    log(_fmt(stats_after))
    log(f'nodata/implausible values zeroed before transform: {n_flagged}')
    return {'before': before, 'after': stats_after, 'n_flagged': n_flagged}


def build_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', required=True,
                    help='RAW 59-band synth patch cache (.npy)')
    ap.add_argument('--output', required=True, help='destination .npy')
    ap.add_argument('--mode', required=True, choices=sorted(MODE_CHANNELS),
                    help="'hull' -> 59-channel continuum_removed; "
                         "'dual' -> 118-channel hull ⊕ linear")
    ap.add_argument('--chunk_rows', type=int, default=512,
                    help='patches transformed per chunk (default 512)')
    ap.add_argument('--force', action='store_true',
                    help='overwrite an existing --output')
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = build_args(argv)
    try:
        convert(args.input, args.output, args.mode,
                chunk_rows=args.chunk_rows, force=args.force)
    except (ValueError, FileExistsError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

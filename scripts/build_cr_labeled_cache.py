"""Convert a raw labeled patch cache into a continuum-removed (CR) cache.

The fine-tune path (`CRISMSpectralPatchDataset`) can CR patches on read, but that
recomputes a per-pixel convex hull for every patch every epoch — prohibitive over
millions of labeled patches. This one-time converter reads an existing raw labeled
cache (`mrral_{split}_patches_p{P}.npy`, as built by scripts/cache_mrral_patches.py)
and writes a parallel CR cache plus the brightness sidecar the `cache_is_cr` +
`return_brightness` reader expects:

    <out_dir>/mrral_{split}_patches_p{P}.npy            (n, P, P, 59)  CR patches
    <out_dir>/mrral_{split}_patches_p{P}_brightness.npy (n, P, P)      pre-CR brightness

Fine-tune then reads it with `--continuum_removed --cache_is_cr --brightness_aux`.

Usage:
    python scripts/build_cr_labeled_cache.py \
        --raw_dir data/patch_cache_7cls --out_dir data/patch_cache_7cls_cr \
        --splits train val test --patch_size 7 --chunk 4096
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.continuum_removal import continuum_removed, brightness_scalar  # noqa: E402


# ── parallel worker state ─────────────────────────────────────────────────────
# continuum_removed is a single-threaded per-spectrum hull loop; over millions of
# labeled patches that pins one core for hours while the rest sit idle. Workers
# open the raw + output memmaps by path and each CR a disjoint [s:e] slice, so the
# written bytes are IDENTICAL to the serial path — just spread across cores.
_W: dict = {}


def _init_worker(raw_path, cr_path, br_path, n, P):
    _W['raw'] = np.memmap(raw_path, dtype='float32', mode='r', shape=(n, P, P, 59))
    _W['cr'] = np.memmap(cr_path, dtype='float32', mode='r+', shape=(n, P, P, 59))
    _W['br'] = np.load(br_path, mmap_mode='r+')


def _cr_range(se) -> int:
    s, e = se
    block = np.asarray(_W['raw'][s:e], dtype=np.float32)
    _W['cr'][s:e] = continuum_removed(block)
    _W['br'][s:e] = brightness_scalar(block)
    _W['cr'].flush()
    _W['br'].flush()
    return e - s


def convert_split(raw_dir: str, out_dir: str, split: str, patch_size: int,
                  chunk: int, jobs: int = 1) -> int:
    fname = f'mrral_{split}_patches_p{patch_size}.npy'
    raw_path = os.path.join(raw_dir, fname)
    if not os.path.exists(raw_path):
        print(f'  {split}: {raw_path} missing, skipping')
        return 0
    # The labeled cache is a RAW headerless memmap (written by
    # cache_mrral_patches.py via np.memmap and read by CRISMSpectralPatchDataset
    # the same way — the .npy extension carries no npy header). So we memmap it,
    # deriving the patch count from the file size, NOT np.load.
    itembytes = patch_size * patch_size * 59 * 4
    nbytes = os.path.getsize(raw_path)
    if nbytes % itembytes != 0:
        raise ValueError(
            f'{raw_path} size {nbytes:,} is not a multiple of the per-patch byte '
            f'count {itembytes:,} (P={patch_size}, 59 bands, float32) — not a raw '
            f'patch memmap?')
    n = nbytes // itembytes
    raw = np.memmap(raw_path, dtype='float32', mode='r',
                    shape=(n, patch_size, patch_size, 59))
    os.makedirs(out_dir, exist_ok=True)
    # CR patches: RAW headerless memmap (fine-tune np.memmaps them with an exact
    # byte-count guard — a .npy header would misalign and fail that guard).
    cr_out = np.memmap(os.path.join(out_dir, fname), dtype='float32', mode='w+',
                       shape=(n, patch_size, patch_size, 59))
    # Brightness sidecar: a real .npy (fine-tune reads it via np.load).
    br_path = os.path.join(out_dir, f'mrral_{split}_patches_p{patch_size}_brightness.npy')
    br_out = np.lib.format.open_memmap(
        br_path, mode='w+', dtype='float32', shape=(n, patch_size, patch_size))
    ranges = [(s, min(s + chunk, n)) for s in range(0, n, chunk)]

    if jobs > 1 and len(ranges) > 1:
        # Release the main-process handles so workers own the files, then fan the
        # disjoint chunk ranges across a pool. Output bytes are identical to serial.
        cr_path = os.path.join(out_dir, fname)
        cr_out.flush(); br_out.flush()
        del cr_out, br_out, raw
        done = 0
        with mp.Pool(jobs, initializer=_init_worker,
                     initargs=(raw_path, cr_path, br_path, n, patch_size)) as pool:
            for c in pool.imap_unordered(_cr_range, ranges):
                done += c
                print(f'  {split}: {done:,}/{n:,} ({jobs} workers)', flush=True)
    else:
        for i, (s, e) in enumerate(ranges):
            block = np.asarray(raw[s:e], dtype=np.float32)      # (b, P, P, 59)
            cr_out[s:e] = continuum_removed(block)              # (b, P, P, 59)
            br_out[s:e] = brightness_scalar(block)              # (b, P, P)
            if i % 20 == 0:
                print(f'  {split}: {e:,}/{n:,}', flush=True)
        cr_out.flush(); br_out.flush()
    print(f'  {split}: wrote {n:,} CR patches + brightness sidecar')
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    ap.add_argument('--patch_size', type=int, default=7)
    ap.add_argument('--chunk', type=int, default=4096)
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 1) - 1),
                    help='parallel worker processes for the CR hull loop '
                         '(default: cpu_count-1; 1 = serial).')
    args = ap.parse_args()
    print(f'CR cache build: jobs={args.jobs}, chunk={args.chunk}')
    total = 0
    skipped = []
    for split in args.splits:
        n = convert_split(args.raw_dir, args.out_dir, split,
                          args.patch_size, args.chunk, jobs=args.jobs)
        if n == 0:
            skipped.append(split)
        total += n
    print(f'done: {total:,} patches converted → {args.out_dir}')
    # A half-built CR cache is worse than none: the fine-tune reads the splits
    # that exist and used to fall through to a different representation for the
    # ones that don't. Exit non-zero so a launcher can't treat this as success.
    if skipped:
        print(f'ERROR: requested split(s) {skipped} had no raw cache in '
              f'{args.raw_dir} and were NOT written. {args.out_dir} is '
              f'INCOMPLETE — build the missing raw splits first, then re-run.',
              file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

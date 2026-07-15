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
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.continuum_removal import continuum_removed, brightness_scalar  # noqa: E402


def convert_split(raw_dir: str, out_dir: str, split: str, patch_size: int,
                  chunk: int) -> int:
    fname = f'mrral_{split}_patches_p{patch_size}.npy'
    raw_path = os.path.join(raw_dir, fname)
    if not os.path.exists(raw_path):
        print(f'  {split}: {raw_path} missing, skipping')
        return 0
    raw = np.load(raw_path, mmap_mode='r')  # (n, P, P, 59)
    n = raw.shape[0]
    assert raw.shape[1:] == (patch_size, patch_size, 59), raw.shape
    os.makedirs(out_dir, exist_ok=True)
    cr_out = np.lib.format.open_memmap(
        os.path.join(out_dir, fname), mode='w+', dtype='float32',
        shape=(n, patch_size, patch_size, 59))
    br_out = np.lib.format.open_memmap(
        os.path.join(out_dir, f'mrral_{split}_patches_p{patch_size}_brightness.npy'),
        mode='w+', dtype='float32', shape=(n, patch_size, patch_size))
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        block = np.asarray(raw[s:e], dtype=np.float32)          # (b, P, P, 59)
        cr_out[s:e] = continuum_removed(block)                  # (b, P, P, 59)
        br_out[s:e] = brightness_scalar(block)                  # (b, P, P)
        if s % (chunk * 20) == 0:
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
    args = ap.parse_args()
    total = 0
    for split in args.splits:
        total += convert_split(args.raw_dir, args.out_dir, split,
                               args.patch_size, args.chunk)
    print(f'done: {total:,} patches converted → {args.out_dir}')


if __name__ == '__main__':
    main()

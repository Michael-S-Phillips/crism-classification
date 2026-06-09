"""Build a row-aligned parquet of plagioclase-positive labels for the MTRDR
ROI patches at ``data/contrastive/extra_plag_roi/``.

The patches.npy already exists at the canonical (N, 7, 7, 59) float32 shape
that SyntheticPatchDataset (data/dataset.py) consumes; this script produces a
matching parquet whose ``i``-th row labels patches[i]. Output rows have
``plagioclase = 1.0`` and every other label column = 0, confidence_weight =
2.0 (same tier as Reviewed) and confidence_tier = 'ROI' for provenance.

Once produced, the supervised classifier (scripts/train.py) consumes the
pair via ``--synth_train_cache <patches.npy> --synth_train_parquet
<this_output>``. The trainer's ConcatDataset stitches the MTRDR plag rows
into the train pool alongside the regular labeled pixels.

Usage:
    conda run -n crism python scripts/build_mtrdr_plag_rows.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LABEL_COLS = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']
DEFAULT_PATCHES_NPY = 'data/contrastive/extra_plag_roi/patches.npy'
DEFAULT_META_PARQUET = 'data/contrastive/extra_plag_roi/meta.parquet'
DEFAULT_OUT_PARQUET = 'data/mtrdr_plag_rows.parquet'
DEFAULT_WEIGHT = 2.0
DEFAULT_TIER = 'ROI'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patches_npy', default=DEFAULT_PATCHES_NPY)
    ap.add_argument('--meta_parquet', default=DEFAULT_META_PARQUET)
    ap.add_argument('--out', default=DEFAULT_OUT_PARQUET)
    ap.add_argument('--weight', type=float, default=DEFAULT_WEIGHT,
                    help='confidence_weight assigned to each MTRDR plag row.')
    ap.add_argument('--tier', default=DEFAULT_TIER,
                    help="confidence_tier string for provenance "
                         "(default 'ROI' — distinguishes from gpkg-labeled "
                         "High/Moderate/Low and review-tool Reviewed rows).")
    args = ap.parse_args()

    patches = np.load(args.patches_npy, mmap_mode='r')
    meta = pd.read_parquet(args.meta_parquet)
    if patches.shape[0] != len(meta):
        raise ValueError(
            f'row mismatch: patches.npy has {patches.shape[0]} rows, '
            f'meta.parquet has {len(meta)}')
    if patches.shape[1:] != (7, 7, 59):
        raise ValueError(
            f'patches shape {patches.shape} is not (N, 7, 7, 59)')

    n = len(meta)
    print(f'building {n:,} MTRDR plagioclase training rows')
    print(f'  source patches: {args.patches_npy}')
    print(f'  source meta:    {args.meta_parquet}')

    # SyntheticPatchDataset only reads LABEL_COLS + confidence_weight from
    # the parquet (see data/dataset.py:407-410), so a minimal schema is
    # enough. Carry the provenance columns through anyway so the file is
    # self-describing for anyone inspecting it later.
    out = pd.DataFrame({
        'tile_id': meta['tile_id'].astype(str).values,
        'pixel_row': meta['pixel_row'].astype('int64').values,
        'pixel_col': meta['pixel_col'].astype('int64').values,
        'source_polygon': meta.get('source_polygon', pd.Series([''] * n)).astype(str).values,
        'source_gpkg': meta.get('source_gpkg', pd.Series([''] * n)).astype(str).values,
        'roi_confidence': meta.get('confidence', pd.Series([''] * n)).astype(str).values,
        'region': meta.get('region', pd.Series([''] * n)).astype(str).values,
    })
    # Label columns: plag = 1.0, everything else = 0.0
    for c in LABEL_COLS:
        out[c] = np.zeros(n, dtype=np.float64)
    out['plagioclase'] = np.ones(n, dtype=np.float64)
    out['confidence_weight'] = np.full(n, args.weight, dtype=np.float64)
    out['confidence_tier'] = args.tier
    out['split'] = 'train'

    print(f'  per-row label: plagioclase = 1.0; weight = {args.weight}; tier = {args.tier!r}')
    print(f'  unique source obsids: {meta["tile_id"].nunique()}')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    tmp = args.out + '.tmp'
    out.to_parquet(tmp, index=False)
    os.replace(tmp, args.out)
    print(f'\nwrote {args.out} ({len(out):,} rows)')
    print(f'\nuse with:')
    print(f'  python scripts/train.py --model spatial_vit \\')
    print(f'      --synth_train_cache {args.patches_npy} \\')
    print(f'      --synth_train_parquet {args.out} \\')
    print(f'      ...other args...')


if __name__ == '__main__':
    main()

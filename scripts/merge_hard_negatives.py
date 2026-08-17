"""Merge mined dust hard negatives into the 7-class training parquet.

Runs on HPC, where mrral_pixels_7cls_handcore.parquet lives. Reads that file's
schema rather than assuming it, labels every mined pixel bland, and delegates
split assignment to split_units.assign_unit_balanced_splits over the CONCATENATED
frame -- so a mined negative near a val unit is absorbed into that unit and
follows its split. Writes a NEW parquet; the input stays an input.

Spec: docs/superpowers/specs/2026-08-17-dust-hard-negatives-design.md
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.split_units import assign_unit_balanced_splits  # noqa: E402

BLAND_CANDIDATES = ('bland', 'other')
MINERAL_COLS = ('olivine', 'olivine_t1', 'olivine_t2', 'lcp', 'hcp',
                'plagioclase', 'alteration', 'junk')


def bland_column_of(columns) -> str:
    for c in BLAND_CANDIDATES:
        if c in columns:
            return c
    raise ValueError(
        f'no bland column: tried {BLAND_CANDIDATES}, parquet has {list(columns)}')


def build_negative_rows(neg_df, target_columns, bland_col, start_id: int):
    """Mined pixels as rows matching `target_columns` exactly, labelled bland."""
    n = len(neg_df)
    out = pd.DataFrame(index=range(n))
    for col in target_columns:
        if col in ('tile_id', 'pixel_row', 'pixel_col') or col.startswith('band_'):
            out[col] = neg_df[col].to_numpy() if col in neg_df.columns else 0.0
        elif col == 'polygon_id':
            out[col] = [f'dustneg_{start_id + i}' for i in range(n)]
        elif col == bland_col:
            out[col] = np.ones(n, dtype=np.float32)
        elif col in MINERAL_COLS:
            out[col] = np.zeros(n, dtype=np.float32)
        elif col == 'split':
            out[col] = pd.Series([pd.NA] * n, dtype='object')
        elif col == 'confidence_weight':
            out[col] = np.ones(n, dtype=np.float32)
        elif col == 'confidence_tier':
            out[col] = 'dust_hard_negative'
        else:
            out[col] = np.zeros(n, dtype=np.float32)
    return out[list(target_columns)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--labels', required=True, help='training parquet (input, untouched)')
    ap.add_argument('--negatives', required=True, help='parquet from mine_dust_hard_negatives')
    ap.add_argument('--out', required=True, help='NEW parquet to write')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    base = pd.read_parquet(args.labels)
    neg = pd.read_parquet(args.negatives)
    bland_col = bland_column_of(base.columns)
    print(f'base {len(base):,} rows; bland column is {bland_col!r}; '
          f'{len(neg):,} mined negatives')

    rows = build_negative_rows(neg, base.columns, bland_col, start_id=0)
    merged = pd.concat([base, rows], ignore_index=True)

    label_cols = [c for c in base.columns
                  if c in MINERAL_COLS or c == bland_col]
    merged['split'] = assign_unit_balanced_splits(merged, label_cols, seed=args.seed)
    print('split distribution after reassignment:')
    print(merged['split'].value_counts())
    print('mined-negative split distribution:')
    print(merged.iloc[len(base):]['split'].value_counts())

    merged.to_parquet(args.out, index=False)
    print(f'wrote {args.out}: {len(merged):,} rows')


if __name__ == '__main__':
    main()

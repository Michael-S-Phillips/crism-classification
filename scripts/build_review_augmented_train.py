"""Merge the MC13 polygon-review outputs into a new training parquet.

Concatenates:
  - Existing data/mrral_pixels.parquet (all splits unchanged)
  - data/mc13_review/confirmed_pixels.parquet (reviewed-positive examples)
  - data/mc13_review/hard_negatives.parquet (rows where corrected_class was
    set — i.e. negative_of is blank — count as positive examples for the
    corrected class; ambiguous rows are deferred until we wire up a loss
    that consumes them)

For pixels that appear in both existing and review (by tile_id +
pixel_row + pixel_col), the review version wins (manually verified).

The reviewed rows are tagged with confidence_tier='Reviewed' and a higher
confidence_weight so the loss sees them as the strongest signal.

Output: data/mrral_pixels_with_review.parquet (existing file is untouched).
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.review.persistence import (
    confirmed_schema_columns,
    hard_negatives_schema_columns,
)

REVIEW_WEIGHT = 2.0           # vs 1.0 default for existing labeled pixels
REVIEW_TIER = 'Reviewed'


def _read_parquet_path_or_dir(path: str) -> Optional[pd.DataFrame]:
    """Read a parquet at ``path`` — handles both single-file (legacy) and
    directory (per-polygon dataset) layouts. Returns None if the path is
    missing or the directory is empty."""
    if not os.path.exists(path):
        return None
    if os.path.isdir(path):
        # Directory dataset — pyarrow reads all .parquet files in the dir as
        # a unified table. Guard against an empty directory.
        files = [f for f in os.listdir(path) if f.endswith('.parquet')]
        if not files:
            return None
    return pd.read_parquet(path)


def _load_confirmed(path: str) -> pd.DataFrame:
    df = _read_parquet_path_or_dir(path)
    if df is None or df.empty:
        return pd.DataFrame(columns=confirmed_schema_columns())
    expected = confirmed_schema_columns()
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f'confirmed pixel parquet missing columns: {missing}')
    return df[expected]


def _load_corrected_hard_neg(path: str) -> pd.DataFrame:
    """From hard_negatives parquet(s), take only rows where corrected_class
    was set (i.e. negative_of is blank/null) — these are positive examples
    for the corrected class."""
    df = _read_parquet_path_or_dir(path)
    if df is None or df.empty:
        return pd.DataFrame(columns=confirmed_schema_columns())
    # negative_of '' or NaN → corrected positive row
    is_corrected = df['negative_of'].isna() | (df['negative_of'].astype(str) == '')
    df = df[is_corrected]
    return df[confirmed_schema_columns()]


def _ambiguous_row_count(path: str) -> int:
    df = _read_parquet_path_or_dir(path)
    if df is None or df.empty:
        return 0
    return int((df['negative_of'].astype(str) == 'ambiguous').sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--existing', default='data/mrral_pixels.parquet')
    # As of 2026-06-10 the review app writes per-polygon parquet files into
    # a directory; these defaults point at the new directory layout. If the
    # legacy single-file paths are still around, pd.read_parquet handles
    # either gracefully (see _read_parquet_path_or_dir below).
    ap.add_argument('--confirmed',
                    default='data/mc13_review/confirmed_pixels')
    ap.add_argument('--hard_negatives',
                    default='data/mc13_review/hard_negatives')
    ap.add_argument('--out', default='data/mrral_pixels_with_review.parquet')
    ap.add_argument('--review_weight', type=float, default=REVIEW_WEIGHT)
    ap.add_argument('--dry_run', action='store_true',
                    help='Print stats but don\'t write the parquet.')
    args = ap.parse_args()

    print(f'loading existing: {args.existing}')
    existing = pd.read_parquet(args.existing)
    print(f'  rows: {len(existing):,}')

    review_parts = []
    if os.path.exists(args.confirmed):
        conf = _load_confirmed(args.confirmed)
        print(f'loaded confirmed: {len(conf):,} rows ({conf["polygon_id"].nunique()} polygons)')
        review_parts.append(conf)
    if os.path.exists(args.hard_negatives):
        hn = _load_corrected_hard_neg(args.hard_negatives)
        print(f'loaded corrected hard-neg: {len(hn):,} rows '
              f'({hn["polygon_id"].nunique()} polygons)')
        review_parts.append(hn)
        ambig = _ambiguous_row_count(args.hard_negatives)
        if ambig:
            print(f'  (deferred: {ambig:,} ambiguous-tagged hard-neg rows — '
                  f'kept in hard_negatives.parquet, not merged into train)')
    if not review_parts:
        print('no review data — output would equal input. exiting.')
        return

    review = pd.concat(review_parts, ignore_index=True)
    review['confidence_weight'] = args.review_weight
    review['confidence_tier'] = REVIEW_TIER
    review['split'] = 'train'

    # Dedupe: if a pixel appears in both, the reviewed row wins.
    review_keys = set(zip(review['tile_id'], review['pixel_row'].astype(int),
                          review['pixel_col'].astype(int)))
    existing_keys = list(zip(existing['tile_id'],
                              existing['pixel_row'].astype(int),
                              existing['pixel_col'].astype(int)))
    keep_mask = [k not in review_keys for k in existing_keys]
    n_dropped = len(existing) - sum(keep_mask)
    existing_kept = existing[keep_mask]
    print(f'dedupe: dropped {n_dropped:,} existing rows that overlap with review')

    # Concat. Column order: existing's order (review already matches the
    # confirmed schema which is mrral_pixels schema).
    out = pd.concat([existing_kept, review], ignore_index=True)
    out = out[existing.columns.tolist()]  # enforce existing column order

    print()
    print('=== summary (positive rows = label column > 0.5) ===')
    for split in ['train', 'val', 'test']:
        sub = out[out.split == split]
        print(f'{split}: {len(sub):,} rows')
        for c in ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']:
            n_pos = (sub[c] > 0.5).sum()
            print(f'  {c:>12s}: {n_pos:>9,d}')
        if 'confidence_tier' in sub.columns:
            tiers = sub.confidence_tier.value_counts().to_dict()
            print(f'  tiers: {tiers}')
        print()

    if args.dry_run:
        print('--dry_run set; not writing.')
        return

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    tmp = args.out + '.tmp'
    out.to_parquet(tmp, index=False)
    os.replace(tmp, args.out)
    print(f'wrote {args.out} ({len(out):,} rows)')


if __name__ == '__main__':
    main()

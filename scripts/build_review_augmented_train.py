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

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.review.persistence import (
    confirmed_schema_columns,
    hard_negatives_schema_columns,
)

REVIEW_WEIGHT = 2.0           # vs 1.0 default for existing labeled pixels
REVIEW_TIER = 'Reviewed'
AMBIGUOUS_TIER = 'Ambiguous'
AMBIGUOUS_WEIGHT = 3.0


def _stratified_cap_per_group(df: pd.DataFrame,
                                group_cols: list,
                                max_per_group: int,
                                seed: int = 0) -> pd.DataFrame:
    """Sample at most ``max_per_group`` rows from each (tile, polygon) group.

    Used to balance the contribution of each polygon (and, for the legacy
    dust-harvest data where polygon_id is uniformly 0, each tile). Without
    this, a single 759k-pixel rejected LCP polygon would supply ~10x more
    bland training signal than every other bland polygon combined."""
    if not max_per_group or max_per_group <= 0:
        return df
    if df.empty:
        return df
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in df.groupby(group_cols, sort=False):
        if len(g) <= max_per_group:
            parts.append(g)
        else:
            idx = rng.choice(len(g), size=max_per_group, replace=False)
            parts.append(g.iloc[idx])
    return pd.concat(parts, ignore_index=True)


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


def _read_hn_subset_by_tag(path: str, tag: Optional[str]) -> pd.DataFrame:
    """Read one tag-subset of hard_negatives using pyarrow predicate pushdown.

    ``tag=None`` selects the "corrected to a mineral" rows (negative_of is
    null or empty string). ``tag='ambiguous'`` / ``tag='alteration'`` select
    the named tags. Predicate pushdown means pyarrow only materializes the
    matching rows — critical because the legacy hard_negatives file is
    2.6 GB and would otherwise expand to 5+ GB pandas just to be filtered.
    """
    empty = pd.DataFrame(columns=confirmed_schema_columns())
    if not os.path.exists(path):
        return empty
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if f.endswith('.parquet')]
        if not files:
            return empty

    import pyarrow.dataset as ds
    import pyarrow.compute as pc

    dataset = ds.dataset(path, format='parquet')
    if tag is None:
        expr = pc.field('negative_of').is_null() | (
            pc.field('negative_of') == '')
    else:
        expr = pc.field('negative_of') == tag

    table = dataset.to_table(filter=expr)
    if table.num_rows == 0:
        return empty
    df = table.to_pandas()
    del table
    return df[confirmed_schema_columns()]


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
    ap.add_argument(
        '--max_pixels_per_polygon', type=int, default=None,
        help='If set, sample at most N pixels per (tile_id, polygon_id) group '
             'across ALL classes after merging existing + review data. Used to '
             'balance the bland pool (one giant rejected polygon would '
             'otherwise contribute 10x more bland signal than every other '
             'bland polygon combined).')
    ap.add_argument(
        '--include_ambiguous', action='store_true',
        help='Fold negative_of=ambiguous rows into the training pool as '
             'universal hard negatives (all label columns 0). Tagged with '
             "confidence_tier='Ambiguous' and a higher confidence_weight "
             '(--ambiguous_weight, default 3.0) so the loss treats them as '
             'strong "this is not any of our minerals" examples.')
    ap.add_argument(
        '--ambiguous_weight', type=float, default=AMBIGUOUS_WEIGHT,
        help='confidence_weight for ambiguous rows (default 3.0).')
    ap.add_argument(
        '--with_alteration_column', action='store_true',
        help='Add an `alteration` column to the output parquet. Set to 1.0 '
             'for rows tagged negative_of=alteration in hard_negatives; '
             '0.0 for all other rows. The existing 5-class classifier '
             'ignores the extra column; the next 6-class classifier uses it.')
    ap.add_argument('--seed', type=int, default=0,
                    help='Random seed for per-polygon sub-sampling.')
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
    # Read hard_negatives subsets via pyarrow predicate pushdown so we only
    # materialize the matching rows. The legacy hard_negatives file is
    # 2.6 GB on disk → would expand to 5+ GB pandas if read whole.
    corrected_hn = _read_hn_subset_by_tag(args.hard_negatives, tag=None)
    ambig_hn = _read_hn_subset_by_tag(args.hard_negatives, tag='ambiguous')
    alt_hn = _read_hn_subset_by_tag(args.hard_negatives, tag='alteration')
    if not corrected_hn.empty:
        print(f'loaded corrected hard-neg: {len(corrected_hn):,} rows '
              f'({corrected_hn["polygon_id"].nunique()} polygons)')
        review_parts.append(corrected_hn)
    if len(ambig_hn) > 0 and not args.include_ambiguous:
        print(f'  (deferred: {len(ambig_hn):,} ambiguous-tagged hard-neg rows '
              f'— pass --include_ambiguous to fold them into the train pool)')
    if not review_parts:
        print('no review data — output would equal input. exiting.')
        return

    review = pd.concat(review_parts, ignore_index=True)
    del review_parts, corrected_hn  # free the corrected sub-pool ref
    review['confidence_weight'] = args.review_weight
    review['confidence_tier'] = REVIEW_TIER
    review['split'] = 'train'

    # Apply per-polygon cap to REVIEW pool immediately. This is the largest
    # data source (8.1M corrected rows from MC11 bland-rejects) and the cap
    # is what keeps subsequent operations from running OOM.
    if args.max_pixels_per_polygon:
        before = len(review)
        review = _stratified_cap_per_group(
            review, ['tile_id', 'polygon_id'],
            args.max_pixels_per_polygon, seed=args.seed)
        import gc
        gc.collect()
        print(f'per-polygon cap on review (N={args.max_pixels_per_polygon}): '
              f'{before:,} → {len(review):,} rows')

    # Optional: --include_ambiguous folds the all-zero-label rejection rows
    # into the train pool as universal hard negatives, with a higher weight
    # so the loss treats them as strong "not any of our minerals" signal.
    ambig_df = None
    if args.include_ambiguous and len(ambig_hn) > 0:
        ambig_df = ambig_hn.copy()
        ambig_df['confidence_weight'] = args.ambiguous_weight
        ambig_df['confidence_tier'] = AMBIGUOUS_TIER
        ambig_df['split'] = 'train'
        print(f'  + {len(ambig_df):,} ambiguous-tagged rows (weight='
              f'{args.ambiguous_weight}, tier={AMBIGUOUS_TIER})')
    del ambig_hn

    # Dedupe: if a pixel appears in both, the reviewed row wins.
    review_keys = set(zip(review['tile_id'], review['pixel_row'].astype(int),
                          review['pixel_col'].astype(int)))
    existing_keys = list(zip(existing['tile_id'],
                              existing['pixel_row'].astype(int),
                              existing['pixel_col'].astype(int)))
    keep_mask = [k not in review_keys for k in existing_keys]
    n_dropped = len(existing) - sum(keep_mask)
    existing_kept = existing[keep_mask]
    del review_keys, existing_keys, keep_mask  # free the sets/lists
    print(f'dedupe: dropped {n_dropped:,} existing rows that overlap with review')

    # Apply per-polygon cap to EXISTING pool's TRAIN rows only. The cap
    # balances the legacy dust harvest (113k pixels/tile across 8 tiles,
    # all polygon_id=0) but val/test must be left untouched for fair
    # downstream evaluation.
    if args.max_pixels_per_polygon:
        is_train = existing_kept['split'] == 'train'
        train_part = existing_kept[is_train]
        nontrain_part = existing_kept[~is_train]
        before = len(train_part)
        train_part = _stratified_cap_per_group(
            train_part, ['tile_id', 'polygon_id'],
            args.max_pixels_per_polygon, seed=args.seed)
        existing_kept = pd.concat([train_part, nontrain_part], ignore_index=True)
        del train_part, nontrain_part
        import gc; gc.collect()
        print(f'per-polygon cap on existing-train (N={args.max_pixels_per_polygon}): '
              f'{before:,} → {len(existing_kept) - (~is_train).sum():,} train rows '
              f'(val/test unchanged)')

    # Concat. Column order: existing's order (review already matches the
    # confirmed schema which is mrral_pixels schema).
    parts = [existing_kept, review]
    if ambig_df is not None and len(ambig_df) > 0:
        parts.append(ambig_df)
    out = pd.concat(parts, ignore_index=True)
    out = out[existing.columns.tolist()]  # enforce existing column order
    del parts, existing_kept, review, ambig_df

    # Optional: add an alteration column for the future 6-class model. The
    # existing 5-class classifier ignores the column (it's not in LABEL_COLS);
    # the next model reads it as a positive output.
    if args.with_alteration_column:
        # Only initialize alteration=0 if the column is missing from `out`.
        # If the existing parquet was patched via
        # scripts/patch_mrral_pixels_with_alteration.py the column is already
        # populated with the 111k legacy gpkg alteration positives — do not
        # clobber them with zero.
        if 'alteration' not in out.columns:
            out['alteration'] = 0.0
            print('alteration column added (initialized to 0; legacy data had no '
                  'alteration). Consider running '
                  'scripts/patch_mrral_pixels_with_alteration.py first to '
                  'recover the ~111k alteration positives from the source gpkgs.')
        legacy_alt = int((out['alteration'] > 0.5).sum())
        if legacy_alt:
            print(f'preserving {legacy_alt:,} legacy alteration positives '
                  f'from the existing parquet')
        # alt_hn was already loaded once above (no re-read of the 2.6 GB file)
        if len(alt_hn) > 0:
            alt_pos = alt_hn.copy()
            alt_pos['alteration'] = 1.0
            alt_pos['confidence_weight'] = args.review_weight
            alt_pos['confidence_tier'] = REVIEW_TIER
            alt_pos['split'] = 'train'
            # Ensure alt_pos has same column set as out
            for c in out.columns:
                if c not in alt_pos.columns:
                    alt_pos[c] = 0.0 if c not in (
                        'tile_id', 'confidence_tier', 'split') else ''
            alt_pos = alt_pos[out.columns.tolist()]
            out = pd.concat([out, alt_pos], ignore_index=True)
            print(f'+ {len(alt_pos):,} alteration-tagged review rows '
                  f'(positive label in `alteration` column)')

    print()
    print('=== summary (positive rows = label column > 0.5) ===')
    label_cols_summary = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp',
                          'plagioclase', 'other']
    if 'alteration' in out.columns:
        label_cols_summary.append('alteration')
    for split in ['train', 'val', 'test']:
        sub = out[out.split == split]
        print(f'{split}: {len(sub):,} rows')
        for c in label_cols_summary:
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

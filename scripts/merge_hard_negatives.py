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
from collections import Counter

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


def bland_confidence_of(base: pd.DataFrame, bland_col: str) -> tuple[str, float]:
    """The (confidence_tier, confidence_weight) this target parquet's own bland
    rows use, so mined dust negatives read at train time as an ordinary bland
    row instead of an invented tier.

    Design spec (2026-08-17-dust-hard-negatives-design.md, lines 114-115):
    "confidence_weight / confidence_tier: match whatever the existing bland
    rows use, read from the target parquet. Do not invent a tier." A tier
    that matches no data/dataset.py WEIGHT_SCHEMES key falls back to the
    stamped confidence_weight, which is harmless only by coincidence under
    the scheme active today ('level', where high == the fallback == 1.0) and
    silently under-weights mined rows relative to real bland rows under any
    scheme that treats 'high' differently (e.g. 'hand_up').

    Picked by majority vote over bland_col > 0 rows: the tier borne by the
    most bland rows already in the file, then -- restricted to that tier --
    the most common confidence_weight. Ties are broken by sorting candidates
    (alphabetically for the tier, numerically ascending for the weight) so
    the choice is deterministic rather than dependent on row order.

    Raises ValueError rather than guessing when the base parquet has no
    confidence_tier column, or no rows with bland_col > 0 -- there is nothing
    to "match" in that case, and a made-up default is exactly the bug this
    function exists to avoid.
    """
    if 'confidence_tier' not in base.columns:
        raise ValueError(
            "base parquet has no 'confidence_tier' column; cannot match mined "
            "negatives to existing bland rows' confidence tier/weight")
    bland_rows = base[base[bland_col] > 0]
    if bland_rows.empty:
        raise ValueError(
            f'base parquet has no rows with {bland_col!r} > 0; cannot infer '
            'the confidence tier/weight mined negatives should carry -- '
            'refusing to invent one')

    tier_counts = Counter(bland_rows['confidence_tier'])
    top = max(tier_counts.values())
    tier = sorted(t for t, c in tier_counts.items() if c == top)[0]

    weight = 1.0
    if 'confidence_weight' in base.columns:
        same_tier = bland_rows.loc[bland_rows['confidence_tier'] == tier,
                                    'confidence_weight']
        same_tier = pd.to_numeric(same_tier, errors='coerce').dropna()
        if not same_tier.empty:
            weight_counts = Counter(same_tier)
            wtop = max(weight_counts.values())
            weight = sorted(w for w, c in weight_counts.items() if c == wtop)[0]
    return tier, float(weight)


def build_negative_rows(neg_df, target_columns, bland_col, start_id: int,
                         confidence_tier: str, confidence_weight: float):
    """Mined pixels as rows matching `target_columns` exactly, labelled bland.

    `confidence_tier`/`confidence_weight` must come from `bland_confidence_of`
    (or an equivalent read of the target parquet's own bland rows) -- see that
    function's docstring for why hard-coding either is a bug.
    """
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
            out[col] = np.full(n, confidence_weight, dtype=np.float32)
        elif col == 'confidence_tier':
            out[col] = confidence_tier
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
    conf_tier, conf_weight = bland_confidence_of(base, bland_col)
    print(f'base {len(base):,} rows; bland column is {bland_col!r}; '
          f'confidence_tier={conf_tier!r} confidence_weight={conf_weight}; '
          f'{len(neg):,} mined negatives')

    rows = build_negative_rows(neg, base.columns, bland_col, start_id=0,
                                confidence_tier=conf_tier,
                                confidence_weight=conf_weight)
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

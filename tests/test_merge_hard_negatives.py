"""Merging mined negatives into the training parquet: schema and labels."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.merge_hard_negatives import (
    bland_column_of, bland_confidence_of, build_negative_rows)

SEVEN = ['tile_id', 'polygon_id', 'pixel_row', 'pixel_col', 'olivine', 'lcp',
         'hcp', 'plagioclase', 'bland', 'alteration', 'junk',
         'confidence_weight', 'confidence_tier', 'split'] + \
        [f'band_{i:02d}' for i in range(59)]
FIVE = [c.replace('bland', 'other') for c in SEVEN]


def _neg(n=3):
    d = {'tile_id': ['t9001'] * n,
         'pixel_row': np.arange(n), 'pixel_col': np.arange(n)}
    for b in range(59):
        d[f'band_{b:02d}'] = np.full(n, 0.1, dtype=np.float32)
    d['RBR'] = np.full(n, 6.0)
    return pd.DataFrame(d)


def _base(bland_tiers, bland_weights=None, bland_col='bland',
          columns=None, other_tier='High', other_weight=1.0):
    """A minimal base parquet: some bland rows with the given
    (tier[, weight]) values, plus one non-bland row so `bland_col > 0`
    filtering is exercised rather than trivially satisfied by every row."""
    if columns is None:
        columns = SEVEN
    n_bland = len(bland_tiers)
    if bland_weights is None:
        bland_weights = [1.0] * n_bland
    d = {}
    for col in columns:
        if col == bland_col:
            d[col] = [1.0] * n_bland + [0.0]
        elif col in ('olivine', 'lcp', 'hcp', 'plagioclase', 'alteration',
                     'junk'):
            d[col] = [0.0] * n_bland + [1.0]
        elif col == 'confidence_tier':
            d[col] = list(bland_tiers) + [other_tier]
        elif col == 'confidence_weight':
            d[col] = list(bland_weights) + [other_weight]
        elif col.startswith('band_'):
            d[col] = np.full(n_bland + 1, 0.1, dtype=np.float32)
        else:
            d[col] = list(range(n_bland + 1))
    return pd.DataFrame(d)


def test_bland_column_is_detected_in_both_vocabularies():
    """The 7-class build calls it 'bland'; older parquets call it 'other'.
    Hard-coding either silently mislabels every mined pixel."""
    assert bland_column_of(SEVEN) == 'bland'
    assert bland_column_of(FIVE) == 'other'


def test_missing_bland_column_raises():
    with pytest.raises(ValueError, match='bland'):
        bland_column_of(['tile_id', 'olivine', 'lcp'])


def test_rows_are_labelled_bland_and_nothing_else():
    out = build_negative_rows(_neg(), SEVEN, 'bland', start_id=0,
                               confidence_tier='High', confidence_weight=1.0)
    assert (out['bland'] == 1).all()
    for c in ('olivine', 'lcp', 'hcp', 'plagioclase', 'alteration', 'junk'):
        assert (out[c] == 0).all(), f'{c} must be 0 on a dust negative'


def test_output_columns_match_the_target_schema_exactly():
    """A column order or set mismatch makes the concat produce NaN columns that
    train silently as zeros."""
    out = build_negative_rows(_neg(), SEVEN, 'bland', start_id=0,
                               confidence_tier='High', confidence_weight=1.0)
    assert list(out.columns) == SEVEN


def test_polygon_ids_are_unique_and_offset():
    """Each mined pixel needs its own synthetic polygon so polygon_units can
    place it geographically; colliding with a real polygon_id would merge a dust
    pixel into a labelled unit."""
    out = build_negative_rows(_neg(4), SEVEN, 'bland', start_id=100,
                               confidence_tier='High', confidence_weight=1.0)
    assert out['polygon_id'].nunique() == 4
    assert all(str(v).startswith('dustneg_') for v in out['polygon_id'])
    assert 'dustneg_100' in set(out['polygon_id'])


def test_bands_are_carried_through_unchanged():
    neg = _neg(2)
    neg['band_07'] = [0.31, 0.42]
    out = build_negative_rows(neg, SEVEN, 'bland', start_id=0,
                               confidence_tier='High', confidence_weight=1.0)
    assert out['band_07'].tolist() == pytest.approx([0.31, 0.42])


def test_split_is_not_assigned_here():
    """Splits come from assign_unit_balanced_splits over the CONCATENATED frame.
    Writing 'train' here would put dust from val terrain into train."""
    out = build_negative_rows(_neg(), SEVEN, 'bland', start_id=0,
                               confidence_tier='High', confidence_weight=1.0)
    assert out['split'].isna().all()


# --- confidence_tier / confidence_weight must match the base parquet's own
# bland rows (design spec lines 114-115), not an invented value. ---


def test_bland_confidence_of_matches_the_only_tier_present():
    base = _base(bland_tiers=['High', 'High', 'High'], bland_weights=[1.0, 1.0, 1.0])
    tier, weight = bland_confidence_of(base, 'bland')
    assert tier == 'High'
    assert weight == pytest.approx(1.0)


def test_bland_confidence_of_picks_majority_tier_when_mixed():
    """Real bland rows in a target parquet are not guaranteed to share one
    tier. The majority tier is the sensible representative; picking the wrong
    one would still be an invented value, just a differently-invented one."""
    base = _base(bland_tiers=['Moderate', 'High', 'High', 'High'],
                 bland_weights=[0.85, 1.0, 1.0, 1.0])
    tier, weight = bland_confidence_of(base, 'bland')
    assert tier == 'High'
    assert weight == pytest.approx(1.0)


def test_bland_confidence_of_breaks_ties_deterministically():
    """2 vs 2 tie between 'High' and 'Low' -- alphabetically 'High' sorts
    first. The specific rule matters less than that it never depends on row
    order (i.e. is not "whichever tier happened to appear first")."""
    base_a = _base(bland_tiers=['High', 'High', 'Low', 'Low'],
                   bland_weights=[1.0, 1.0, 0.70, 0.70])
    base_b = _base(bland_tiers=['Low', 'Low', 'High', 'High'],
                   bland_weights=[0.70, 0.70, 1.0, 1.0])
    assert bland_confidence_of(base_a, 'bland') == bland_confidence_of(base_b, 'bland')
    tier, _ = bland_confidence_of(base_a, 'bland')
    assert tier == 'High'


def test_bland_confidence_of_raises_when_base_has_no_bland_rows():
    base = _base(bland_tiers=[])
    with pytest.raises(ValueError, match='bland'):
        bland_confidence_of(base, 'bland')


def test_bland_confidence_of_raises_when_no_confidence_tier_column():
    base = _base(bland_tiers=['High'])
    base = base.drop(columns=['confidence_tier'])
    with pytest.raises(ValueError, match='confidence_tier'):
        bland_confidence_of(base, 'bland')


def test_build_negative_rows_uses_the_supplied_confidence_not_a_fixed_value():
    """The core regression: mined rows must carry whatever
    (confidence_tier, confidence_weight) the caller determined from the base
    parquet's bland rows -- not a value invented inside build_negative_rows.
    Exercised at two different (tier, weight) pairs so a hard-coded constant
    of either kind cannot pass both."""
    out_a = build_negative_rows(_neg(3), SEVEN, 'bland', start_id=0,
                                 confidence_tier='Moderate', confidence_weight=0.85)
    assert (out_a['confidence_tier'] == 'Moderate').all()
    assert out_a['confidence_weight'].to_numpy() == pytest.approx([0.85, 0.85, 0.85])

    out_b = build_negative_rows(_neg(2), SEVEN, 'bland', start_id=0,
                                 confidence_tier='reviewed-legacy', confidence_weight=1.5)
    assert (out_b['confidence_tier'] == 'reviewed-legacy').all()
    assert out_b['confidence_weight'].to_numpy() == pytest.approx([1.5, 1.5])


def test_merged_negatives_end_to_end_adopt_base_bland_confidence():
    """Full base -> bland_confidence_of -> build_negative_rows chain, the way
    main() calls it: mined rows should be indistinguishable in confidence
    from the base parquet's real bland rows."""
    base = _base(bland_tiers=['Low', 'Low', 'Low'], bland_weights=[0.70, 0.70, 0.70],
                 other_tier='High', other_weight=1.0)
    bland_col = bland_column_of(base.columns)
    tier, weight = bland_confidence_of(base, bland_col)
    rows = build_negative_rows(_neg(3), base.columns, bland_col, start_id=0,
                                confidence_tier=tier, confidence_weight=weight)
    assert (rows['confidence_tier'] == 'Low').all()
    assert rows['confidence_weight'].to_numpy() == pytest.approx([0.70, 0.70, 0.70])

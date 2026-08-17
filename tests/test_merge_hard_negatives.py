"""Merging mined negatives into the training parquet: schema and labels."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.merge_hard_negatives import bland_column_of, build_negative_rows

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


def test_bland_column_is_detected_in_both_vocabularies():
    """The 7-class build calls it 'bland'; older parquets call it 'other'.
    Hard-coding either silently mislabels every mined pixel."""
    assert bland_column_of(SEVEN) == 'bland'
    assert bland_column_of(FIVE) == 'other'


def test_missing_bland_column_raises():
    with pytest.raises(ValueError, match='bland'):
        bland_column_of(['tile_id', 'olivine', 'lcp'])


def test_rows_are_labelled_bland_and_nothing_else():
    out = build_negative_rows(_neg(), SEVEN, 'bland', start_id=0)
    assert (out['bland'] == 1).all()
    for c in ('olivine', 'lcp', 'hcp', 'plagioclase', 'alteration', 'junk'):
        assert (out[c] == 0).all(), f'{c} must be 0 on a dust negative'


def test_output_columns_match_the_target_schema_exactly():
    """A column order or set mismatch makes the concat produce NaN columns that
    train silently as zeros."""
    out = build_negative_rows(_neg(), SEVEN, 'bland', start_id=0)
    assert list(out.columns) == SEVEN


def test_polygon_ids_are_unique_and_offset():
    """Each mined pixel needs its own synthetic polygon so polygon_units can
    place it geographically; colliding with a real polygon_id would merge a dust
    pixel into a labelled unit."""
    out = build_negative_rows(_neg(4), SEVEN, 'bland', start_id=100)
    assert out['polygon_id'].nunique() == 4
    assert all(str(v).startswith('dustneg_') for v in out['polygon_id'])
    assert 'dustneg_100' in set(out['polygon_id'])


def test_bands_are_carried_through_unchanged():
    neg = _neg(2)
    neg['band_07'] = [0.31, 0.42]
    out = build_negative_rows(neg, SEVEN, 'bland', start_id=0)
    assert out['band_07'].tolist() == pytest.approx([0.31, 0.42])


def test_split_is_not_assigned_here():
    """Splits come from assign_unit_balanced_splits over the CONCATENATED frame.
    Writing 'train' here would put dust from val terrain into train."""
    out = build_negative_rows(_neg(), SEVEN, 'bland', start_id=0)
    assert out['split'].isna().all()

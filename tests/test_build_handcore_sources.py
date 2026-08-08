import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scripts.build_7cls_dataset import _read_hn_tag, _session_of

_LABEL = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other',
          'alteration']


def _hn_frame(n, tier, negative_of, tile='t1250', poly_start=0):
    d = {'tile_id': [tile] * n,
         'polygon_id': [poly_start + i // 10 for i in range(n)],
         'pixel_row': list(range(n)), 'pixel_col': list(range(n)),
         'negative_of': [negative_of] * n,
         'confidence_weight': [1.0] * n, 'confidence_tier': [tier] * n,
         'split': ['train'] * n}
    for c in _LABEL:
        d[c] = [0.0] * n
    for i in range(59):
        d[f'm{i}'] = [0.1] * n
    return pd.DataFrame(d)


def _write_session(root, name, frame):
    d = root / name / 'hard_negatives'
    d.mkdir(parents=True)
    frame.to_parquet(d / 'p_0001.parquet')
    return str(d)


def test_session_of_classifies_dirs():
    assert _session_of('/x/data/mc13_review/hard_negatives') == 'legacy'
    assert _session_of('/x/data/mc13_review_7cls_v3/hard_negatives') == 'v3'


def test_read_hn_tag_stamps_session(tmp_path):
    legacy = _write_session(tmp_path, 'mc13_review',
                            _hn_frame(20, 'High', 'ambiguous'))
    v3 = _write_session(tmp_path, 'mc13_review_7cls_v3',
                        _hn_frame(30, 'Reviewed-High', 'ambiguous',
                                  poly_start=100))
    out = _read_hn_tag([legacy, v3], 'ambiguous')
    assert 'review_session' in out.columns
    assert set(out['review_session']) == {'legacy', 'v3'}
    assert (out['review_session'] == 'legacy').sum() == 20
    assert (out['review_session'] == 'v3').sum() == 30


def test_confirmed_loader_preserves_session_through_template_align(tmp_path):
    """Regression: the loader returns df[template.columns], which would drop
    review_session and make every legacy confirm look like a graded v3 row."""
    from scripts.build_7cls_dataset import load_confirmed_mineral_positives

    for name, tier, n in [('mc13_review', 'High', 10),
                          ('mc13_review_7cls_v3', 'Reviewed-High', 10)]:
        d = tmp_path / name / 'confirmed_pixels'
        d.mkdir(parents=True)
        f = _hn_frame(n, tier, '', poly_start=0)
        f['lcp'] = 1.0
        f.drop(columns=['negative_of']).to_parquet(d / 'p_0001.parquet')

    template = _hn_frame(1, 'High', '').drop(columns=['negative_of'])
    template['bland'] = 0.0
    template['junk'] = 0.0
    out = load_confirmed_mineral_positives(
        [str(tmp_path / 'mc13_review' / 'confirmed_pixels'),
         str(tmp_path / 'mc13_review_7cls_v3' / 'confirmed_pixels')],
        template)
    assert 'review_session' in out.columns
    assert set(out['review_session']) == {'legacy', 'v3'}

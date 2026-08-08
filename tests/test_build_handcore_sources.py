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


from scripts.build_7cls_dataset import _filter_review_grades


def _graded(session, tier, n, poly_start):
    f = _hn_frame(n, tier, 'ambiguous', poly_start=poly_start)
    f['review_session'] = session
    return f


def test_grade_filter_keeps_only_named_v3_grades():
    df = pd.concat([
        _graded('v3', 'Reviewed-High', 10, 0),
        _graded('v3', 'Reviewed-Moderate', 10, 10),
        _graded('v3', 'Reviewed-Low', 10, 20),
    ], ignore_index=True)
    out = _filter_review_grades(df, ['High', 'Moderate'])
    assert set(out['confidence_tier']) == {'Reviewed-High', 'Reviewed-Moderate'}
    assert len(out) == 20


def test_grade_filter_leaves_legacy_rows_untouched():
    # Legacy is stamped tier='High'; the v3 grade filter must not judge it.
    df = pd.concat([
        _graded('legacy', 'High', 15, 0),
        _graded('v3', 'Reviewed-Low', 10, 100),
    ], ignore_index=True)
    out = _filter_review_grades(df, ['High', 'Moderate'])
    assert (out['review_session'] == 'legacy').sum() == 15
    assert (out['review_session'] == 'v3').sum() == 0


def test_grade_filter_is_noop_on_empty():
    assert _filter_review_grades(pd.DataFrame(), ['High']).empty


from scripts.build_7cls_dataset import _apply_legacy_policy


def test_legacy_dropped_for_unlisted_class():
    df = pd.concat([
        _graded('legacy', 'High', 20, 0),
        _graded('v3', 'Reviewed-High', 10, 100),
    ], ignore_index=True)
    out = _apply_legacy_policy(df, 'bland', ['alteration', 'lcp', 'hcp'],
                               confirm_cap=5000, seed=42)
    assert (out['review_session'] == 'legacy').sum() == 0
    assert (out['review_session'] == 'v3').sum() == 10


def test_legacy_kept_for_listed_class():
    df = _graded('legacy', 'High', 20, 0)
    out = _apply_legacy_policy(df, 'alteration', ['alteration', 'lcp', 'hcp'],
                               confirm_cap=5000, seed=42)
    assert len(out) == 20


def test_legacy_confirm_cap_applies_per_polygon():
    # 3 polygons x 40 rows, cap 10 -> 30 rows kept, legacy confirms only.
    f = _hn_frame(120, 'High', '', poly_start=0)
    f['polygon_id'] = [i // 40 for i in range(120)]
    f['review_session'] = 'legacy'
    out = _apply_legacy_policy(f, 'lcp', ['lcp'], confirm_cap=10, seed=42,
                               is_confirm=True)
    assert len(out) == 30
    assert out.groupby('polygon_id').size().max() == 10


def test_legacy_hard_negatives_ignore_confirm_cap():
    f = _hn_frame(120, 'High', 'alteration', poly_start=0)
    f['polygon_id'] = [i // 40 for i in range(120)]
    f['review_session'] = 'legacy'
    out = _apply_legacy_policy(f, 'alteration', ['alteration'], confirm_cap=10,
                               seed=42, is_confirm=False)
    assert len(out) == 120


from scripts.build_7cls_dataset import _build_base


def _base_frame(tmp_path):
    n = 40
    d = {'tile_id': ['t1250'] * n,
         'polygon_id': [i // 10 for i in range(n)],
         'pixel_row': list(range(n)), 'pixel_col': list(range(n)),
         'confidence_weight': [1.0] * n, 'confidence_tier': ['High'] * n,
         'split': ['train'] * n}
    for c in _LABEL:
        d[c] = [0.0] * n
    # First 20 rows are minerals, last 20 are bland ('other').
    d['lcp'] = [1.0] * 20 + [0.0] * 20
    d['other'] = [0.0] * 20 + [1.0] * 20
    for i in range(59):
        d[f'm{i}'] = [0.1] * n
    p = tmp_path / 'base.parquet'
    pd.DataFrame(d).to_parquet(p)
    return str(p)


def test_bland_sources_review_drops_base_other_rows(tmp_path):
    out = _build_base(_base_frame(tmp_path), 300_000, bland_sources='review')
    assert len(out) == 20
    assert (out['bland'] > 0).sum() == 0


def test_bland_sources_all_keeps_base_other_rows(tmp_path):
    out = _build_base(_base_frame(tmp_path), 300_000, bland_sources='all')
    assert len(out) == 40
    assert (out['bland'] > 0).sum() == 20


# ── CLI policy-flag defaults must stay inert ─────────────────────────────────

import scripts.build_7cls_dataset as b


def test_policy_flag_defaults_are_inert():
    """A bare invocation must reproduce the pre-hand-core build.

    The hand-core source policy is opt-in: every flag defaults to the fully
    permissive value so no filtering happens unless explicitly asked for. If
    you change a default here, you silently change every unflagged rebuild of
    data/mrral_pixels_7cls.parquet -- and the champion's data lineage stops
    being reproducible. Narrow the policy on the command line instead.
    """
    args = b._build_parser().parse_args([])

    # Every v3 reviewer grade admitted -> _filter_review_grades drops nothing.
    assert set(args.review_grades) == {'High', 'Moderate', 'Low'}

    # Every class admitted -> _apply_legacy_policy never drops the legacy
    # session, whichever fragment it is keyed to.
    assert set(args.legacy_classes) == set(b._ALL_POLICY_CLASSES)

    # No cap tighter than the one the loaders already applied upstream.
    assert args.legacy_confirm_cap == b.MAX_PX_PER_POLYGON

    # Base parquet's bland rows retained.
    assert args.bland_sources == 'all'


def test_all_policy_classes_covers_every_policy_key():
    """_ALL_POLICY_CLASSES must cover every class main()'s policy table keys
    on, or the permissive default would still drop a fragment."""
    for cls in ('lcp', 'bland', 'junk', 'alteration'):
        assert cls in b._ALL_POLICY_CLASSES

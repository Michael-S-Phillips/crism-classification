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


def test_session_of_treats_any_versioned_session_as_graded():
    """A FUTURE review session must not be mistaken for ungraded legacy data.

    Matching only the literal '_7cls_v3' would classify a mc13_review_7cls_v4
    directory as 'legacy' -- it would then bypass the --review_grades bar
    entirely and be governed by the per-class --legacy_classes admission
    instead. Any _7cls_v<N> dir is a graded session.
    """
    assert _session_of('/x/data/mc13_review_7cls_v4/hard_negatives') == 'v3'
    assert _session_of('/x/data/mc13_review_7cls_v10/confirmed_pixels') == 'v3'
    # ...and the ungraded legacy dir still classifies as legacy.
    assert _session_of('/x/data/mc13_review/confirmed_pixels') == 'legacy'


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


def test_grade_filter_raises_when_provenance_is_missing():
    """Losing review_session must be LOUD, not a silent no-op.

    If provenance is stripped upstream the filter has nothing to key on, so
    returning the frame unchanged would admit exactly the ungraded rows the
    policy exists to exclude -- with no error and no warning.
    """
    df = _graded('v3', 'Reviewed-Low', 10, 0).drop(columns=['review_session'])
    with pytest.raises(ValueError, match='review_session'):
        _filter_review_grades(df, ['High'])


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


def test_legacy_policy_raises_when_provenance_is_missing():
    """Same silent-failure guard for the per-class legacy admission."""
    df = _graded('legacy', 'High', 10, 0).drop(columns=['review_session'])
    with pytest.raises(ValueError, match='review_session'):
        _apply_legacy_policy(df, 'bland', ['alteration'], confirm_cap=5000,
                             seed=42)


def test_legacy_policy_is_noop_on_empty():
    """An empty frame carries no rows to misclassify -> clean no-op, no raise."""
    assert _apply_legacy_policy(pd.DataFrame(), 'bland', ['alteration'],
                                confirm_cap=5000, seed=42).empty


from scripts.build_7cls_dataset import _stamp_legacy_tier


def test_stamp_legacy_tier_restamps_legacy_only():
    """The provenance fix: legacy (ungraded) rows get their own tier so they
    stop colliding with hand-labeled 'High' rows in the weight-scheme tables;
    v3-graded rows must keep their existing 'Reviewed-*' tiers unchanged."""
    df = pd.concat([
        _graded('legacy', 'High', 15, 0),
        _graded('v3', 'Reviewed-High', 10, 100),
        _graded('v3', 'Reviewed-Moderate', 5, 200),
    ], ignore_index=True)
    out = _stamp_legacy_tier(df)
    legacy_tiers = set(out.loc[out['review_session'] == 'legacy', 'confidence_tier'])
    v3_tiers = set(out.loc[out['review_session'] == 'v3', 'confidence_tier'])
    assert legacy_tiers == {'Reviewed-Legacy'}
    assert v3_tiers == {'Reviewed-High', 'Reviewed-Moderate'}
    assert (out['review_session'] == 'legacy').sum() == 15
    assert (out['review_session'] == 'v3').sum() == 15


def test_stamp_legacy_tier_noop_without_review_session():
    """Hand-labeled base rows never carry review_session and must pass
    through untouched (this helper is only ever called on review
    fragments, never on the base frame)."""
    df = _hn_frame(10, 'High', '').drop(columns=['negative_of'])
    assert 'review_session' not in df.columns
    out = _stamp_legacy_tier(df)
    assert set(out['confidence_tier']) == {'High'}


def test_stamp_legacy_tier_noop_on_empty():
    assert _stamp_legacy_tier(pd.DataFrame()).empty


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


# ── Order preservation when the confirm cap binds nothing ────────────────────

from pandas.testing import assert_frame_equal


def _mixed_session_confirms():
    """40 confirm rows over 4 polygons, legacy and v3 INTERLEAVED row by row."""
    f = _hn_frame(40, 'High', '', poly_start=0)
    f['polygon_id'] = [i // 10 for i in range(40)]
    f['review_session'] = ['legacy' if i % 2 == 0 else 'v3' for i in range(40)]
    return f


def test_legacy_confirm_noop_cap_preserves_row_order():
    """A non-binding confirm cap must return the frame UNCHANGED, in order.

    Failure mode this locks down: _apply_legacy_policy's is_confirm branch ends
    in pd.concat([df[~is_legacy], legacy]), which hoists all v3 rows to the
    front and pushes legacy rows to the back even when the cap drops nothing.
    _joint_resplit's greedy unit assignment is order-sensitive at ties, so that
    reordering alone silently moves ~250 rows between train and val on every
    bare rebuild -- breaking the guarantee that unflagged runs reproduce the
    pre-hand-core dataset. Row counts stay identical, so a length assertion
    cannot catch it; compare the whole frame including order.
    """
    f = _mixed_session_confirms()
    # cap 1000 >> 10 rows/polygon, so it binds nothing.
    out = _apply_legacy_policy(f, 'lcp', ['lcp'], confirm_cap=1000, seed=42,
                               is_confirm=True)
    assert_frame_equal(out, f.reset_index(drop=True))


def test_legacy_confirm_noop_cap_keeps_both_sessions_intact():
    """The admitted + is_confirm=True path on a MIXED frame: v3 rows must be
    neither dropped nor duplicated, and legacy rows must all survive."""
    f = _mixed_session_confirms()
    out = _apply_legacy_policy(f, 'lcp', ['lcp'], confirm_cap=1000, seed=42,
                               is_confirm=True)
    assert len(out) == 40
    assert (out['review_session'] == 'v3').sum() == 20
    assert (out['review_session'] == 'legacy').sum() == 20
    # no duplication: pixel_row was unique per input row
    assert out['pixel_row'].is_unique


def test_legacy_confirm_binding_cap_still_caps_on_mixed_frame():
    """The guard must not disarm a cap that genuinely binds: v3 rows pass
    through whole while legacy rows are capped per polygon."""
    f = _mixed_session_confirms()
    out = _apply_legacy_policy(f, 'lcp', ['lcp'], confirm_cap=2, seed=42,
                               is_confirm=True)
    # 20 v3 rows untouched + 4 polygons x 2 legacy rows kept
    assert (out['review_session'] == 'v3').sum() == 20
    assert (out['review_session'] == 'legacy').sum() == 8
    legacy = out[out['review_session'] == 'legacy']
    assert legacy.groupby('polygon_id').size().max() == 2


# ── Champion parquet clobber guard ───────────────────────────────────────────

def test_default_out_rejected_when_policy_is_narrowed():
    """The spec requires data/mrral_pixels_7cls.parquet not be clobbered. A
    narrowed policy writing to the default path would destroy the champion's
    data lineage, so it must be refused."""
    args = b._build_parser().parse_args(['--bland_sources', 'review'])
    err = b._out_clobber_error(args)
    assert err is not None
    assert '--out' in err


def test_default_out_allowed_when_policy_is_permissive():
    """A bare invocation reproduces the champion's dataset, so writing to the
    default path is exactly right."""
    assert b._out_clobber_error(b._build_parser().parse_args([])) is None


def test_narrowed_policy_allowed_with_explicit_out():
    """The hand-core recipe passes --out, so it must sail through."""
    args = b._build_parser().parse_args(
        ['--bland_sources', 'review', '--review_grades', 'High', 'Moderate',
         '--legacy_classes', 'alteration', 'lcp', 'hcp',
         '--legacy_confirm_cap', '5000',
         '--out', 'data/mrral_pixels_7cls_handcore.parquet'])
    assert b._out_clobber_error(args) is None


@pytest.mark.parametrize('argv', [
    ['--review_grades', 'High', 'Moderate'],
    ['--legacy_classes', 'alteration', 'lcp', 'hcp'],
    ['--legacy_confirm_cap', '5000'],
    ['--bland_sources', 'review'],
    ['--ndviz_dir', ''],
])
def test_each_policy_flag_alone_trips_the_guard(argv):
    """Every recipe flag changes the dataset, so any of them alone must be
    enough to refuse the default --out. --ndviz_dir '' disables a whole
    relabel session, so it counts too."""
    assert b._out_clobber_error(b._build_parser().parse_args(argv)) is not None


# ── Policy block must precede the all_cols projection ────────────────────────

def test_policy_block_runs_before_all_cols_projection():
    """Invariant C: the review policy block MUST come before `all_cols`.

    Failure mode: `all_cols = base.columns.tolist()` is taken from the BASE
    frame, which has no `review_session` column, and every fragment is then
    projected with `frag[all_cols]`. If the policy block were ever moved below
    that projection, the provenance column would already be gone -- so the
    grade bar and the legacy admission would BOTH become no-ops. The build
    would still succeed with no error and no warning; the dataset would just
    quietly contain the ungraded rows the policy exists to exclude. Nothing
    else in the suite detects this, because every helper-level test constructs
    its own frame.
    """
    import inspect

    src = inspect.getsource(b.main)
    policy_at = src.find('Applying review source policy')
    projection_at = src.find('all_cols = base.columns')
    assert policy_at != -1, 'policy block marker not found in main()'
    assert projection_at != -1, 'all_cols projection not found in main()'
    assert policy_at < projection_at, (
        'the review source policy block must run BEFORE '
        '`all_cols = base.columns` -- the projection strips review_session, '
        'which would turn every policy filter into a silent no-op')

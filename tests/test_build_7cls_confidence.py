import math
import os
import sys

import numpy as np
import pandas as pd

from scripts.build_7cls_dataset import load_confirmed_mineral_positives, load_bland_review, load_reassigned_minerals
from scripts.build_7cls_dataset import load_junk_ambiguous, load_alteration_mc11
from scripts.build_7cls_dataset import _build_base, BALANCE_COLS, MAX_PX_PER_POLYGON

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
import split_units as su

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _confirmed_row(polygon_id, label_col, weight, tier, n=3):
    d = {
        'tile_id': ['t1250'] * n,
        'polygon_id': np.full(n, polygon_id, dtype=np.int64),
        'pixel_row': np.arange(n), 'pixel_col': np.arange(n),
    }
    for i in range(59):
        d[f'm{i}'] = np.zeros(n)
    for c in ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase',
              'other', 'alteration']:
        d[c] = np.zeros(n)
    d[label_col] = np.ones(n)
    d['confidence_weight'] = np.full(n, weight)
    d['confidence_tier'] = [tier] * n
    d['split'] = ['train'] * n
    return pd.DataFrame(d)


def test_confirmed_preserves_per_polygon_weight(tmp_path):
    cdir = tmp_path / 'confirmed'
    cdir.mkdir()
    _confirmed_row(1, 'olivine_t1', 1.0, 'Reviewed-High').to_parquet(
        cdir / 'p_00000001.parquet', index=False)
    _confirmed_row(2, 'hcp', 0.5, 'Reviewed-Low').to_parquet(
        cdir / 'p_00000002.parquet', index=False)
    template = _confirmed_row(0, 'olivine_t1', 1.0, 'Reviewed-High').assign(
        bland=0.0, junk=0.0)
    out = load_confirmed_mineral_positives(str(cdir), template)
    # weights are NOT flattened to 2.0 — each polygon keeps its stamped value
    assert set(np.unique(out['confidence_weight'])) == {1.0, 0.5}
    assert set(out['confidence_tier']) == {'Reviewed-High', 'Reviewed-Low'}


def test_confirmed_fills_legacy_rows_in_mixed_dir(tmp_path):
    cdir = tmp_path / 'confirmed'
    cdir.mkdir()
    # new-style file: has confidence columns
    _confirmed_row(1, 'hcp', 0.5, 'Reviewed-Low').to_parquet(
        cdir / 'p_00000001.parquet', index=False)
    # legacy-style file: confidence columns absent
    legacy = _confirmed_row(2, 'olivine_t1', 1.0, 'Reviewed-High').drop(
        columns=['confidence_weight', 'confidence_tier'])
    legacy.to_parquet(cdir / 'p_00000002.parquet', index=False)
    template = _confirmed_row(0, 'olivine_t1', 1.0, 'Reviewed-High').assign(
        bland=0.0, junk=0.0)
    out = load_confirmed_mineral_positives(str(cdir), template)
    assert out['confidence_weight'].notna().all()  # no NaN leaks through
    # legacy rows (olivine) filled to 1.0/High; new rows (hcp) keep 0.5/Reviewed-Low
    oliv = out[out['olivine_t1'] > 0.5]
    hcp = out[out['hcp'] > 0.5]
    assert (oliv['confidence_weight'] == 1.0).all()
    assert (oliv['confidence_tier'] == 'High').all()
    assert (hcp['confidence_weight'] == 0.5).all()
    assert (hcp['confidence_tier'] == 'Reviewed-Low').all()


def _hardneg_row(polygon_id, label_col, n=3):
    d = {
        'tile_id': ['t1250'] * n,  # MC13 region
        'polygon_id': np.full(n, polygon_id, dtype=np.int64),
        'pixel_row': np.arange(n), 'pixel_col': np.arange(n),
    }
    for i in range(59):
        d[f'm{i}'] = np.zeros(n)
    for c in ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase',
              'other', 'alteration']:
        d[c] = np.zeros(n)
    d[label_col] = np.ones(n)
    d['confidence_weight'] = np.full(n, 0.5)
    d['confidence_tier'] = ['Reviewed-Low'] * n
    d['split'] = ['train'] * n
    d['negative_of'] = [''] * n
    return pd.DataFrame(d)


def _write_hardneg_dir(tmp_path):
    hdir = tmp_path / 'hardneg'
    hdir.mkdir()
    _hardneg_row(1, 'olivine_t1').to_parquet(hdir / 'p_01.parquet', index=False)
    _hardneg_row(2, 'other').to_parquet(hdir / 'p_02.parquet', index=False)
    return str(hdir)


def test_reassigned_minerals_routed_to_positives(tmp_path):
    hdir = _write_hardneg_dir(tmp_path)
    out = load_reassigned_minerals(hdir)
    assert (out['olivine_t1'] > 0).all()
    assert (out['bland'] == 0).all()
    assert set(out['confidence_weight'].unique()) == {0.5}


def test_reassigned_minerals_routes_olivine_t2_only(tmp_path):
    hdir = tmp_path / 'hardneg'
    hdir.mkdir()
    _hardneg_row(3, 'olivine_t2').to_parquet(hdir / 'p_03.parquet', index=False)
    out = load_reassigned_minerals(str(hdir))
    assert len(out) > 0
    assert (out['olivine_t2'] > 0).all()
    assert (out['bland'] == 0).all()


def test_bland_review_excludes_mineral_reassignments(tmp_path):
    hdir = _write_hardneg_dir(tmp_path)
    out = load_bland_review(hdir, 'mc13_blands', mc13=True, seed_offset=10,
                            n_bland=1000)
    # only the other=1.0 polygon survives in the bland pool
    assert (out['bland'] > 0).all()
    assert (out['olivine_t1'] == 0).all()


def _tagged_hardneg_row(polygon_id, tag, weight, tier, n=3):
    d = {
        'tile_id': ['t1250'] * n,
        'polygon_id': np.full(n, polygon_id, dtype=np.int64),
        'pixel_row': np.arange(n), 'pixel_col': np.arange(n),
    }
    for i in range(59):
        d[f'm{i}'] = np.zeros(n)
    for c in ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase',
              'other', 'alteration']:
        d[c] = np.zeros(n)
    d['confidence_weight'] = np.full(n, weight)
    d['confidence_tier'] = [tier] * n
    d['split'] = ['train'] * n
    d['negative_of'] = [tag] * n
    return pd.DataFrame(d)


def test_junk_preserves_per_polygon_weight(tmp_path):
    hdir = tmp_path / 'hardneg'
    hdir.mkdir()
    _tagged_hardneg_row(1, 'ambiguous', 0.5, 'Reviewed-Low').to_parquet(
        hdir / 'p_amb.parquet', index=False)
    out = load_junk_ambiguous(str(hdir))
    assert (out['junk'] > 0).all()
    assert (out['confidence_weight'] == 0.5).all()
    assert (out['confidence_tier'] == 'Reviewed-Low').all()


def test_junk_applies_per_polygon_cap(tmp_path):
    """Junk (ambiguous) is the only review loader without the per-polygon
    cap — a top-5-polygon-holds-56%-of-the-class concentration. A 25k-row
    polygon must be capped to MAX_PX_PER_POLYGON (20,000); a 1k-row polygon
    is left intact."""
    hdir = tmp_path / 'hardneg'
    hdir.mkdir()
    _tagged_hardneg_row(1, 'ambiguous', 1.0, 'High', n=25_000).to_parquet(
        hdir / 'p_big.parquet', index=False)
    _tagged_hardneg_row(2, 'ambiguous', 1.0, 'High', n=1_000).to_parquet(
        hdir / 'p_small.parquet', index=False)
    out = load_junk_ambiguous(str(hdir))
    big = out[out['polygon_id'] == 1]
    small = out[out['polygon_id'] == 2]
    assert len(big) == MAX_PX_PER_POLYGON
    assert len(small) == 1_000


def test_alteration_preserves_per_polygon_weight(tmp_path):
    hdir = tmp_path / 'hardneg'
    hdir.mkdir()
    _tagged_hardneg_row(2, 'alteration', 0.75, 'Reviewed-Moderate').to_parquet(
        hdir / 'p_alt.parquet', index=False)
    out = load_alteration_mc11(str(hdir))
    assert (out['alteration'] > 0).all()
    assert (out['confidence_weight'] == 0.75).all()
    assert (out['confidence_tier'] == 'Reviewed-Moderate').all()


def test_bland_review_preserves_per_polygon_weight(tmp_path):
    hdir = tmp_path / 'hardneg'
    hdir.mkdir()
    # _hardneg_row sets confidence_weight=0.5, tier 'Reviewed-Low', negative_of=''
    _hardneg_row(3, 'other').to_parquet(hdir / 'p_bland.parquet', index=False)
    out = load_bland_review(str(hdir), 'mc13_blands', mc13=True, seed_offset=10,
                            n_bland=1000)
    assert (out['bland'] > 0).all()
    assert (out['confidence_weight'] == 0.5).all()
    assert (out['confidence_tier'] == 'Reviewed-Low').all()


def test_confirmed_preserves_alteration_label(tmp_path):
    # An alteration confirm (alteration=1.0 in the parquet) must survive the
    # build — the loader used to stamp alteration=0.0, wiping it to an
    # all-zero-label row.
    cdir = tmp_path / 'confirmed'
    cdir.mkdir()
    _confirmed_row(1, 'alteration', 1.0, 'Reviewed-High').to_parquet(
        cdir / 'p_alt.parquet', index=False)
    # legacy file without an alteration column -> NaN after concat -> fill 0
    legacy = _confirmed_row(2, 'hcp', 1.0, 'Reviewed-High').drop(
        columns=['alteration'])
    legacy.to_parquet(cdir / 'p_leg.parquet', index=False)
    template = _confirmed_row(0, 'olivine_t1', 1.0, 'Reviewed-High').assign(
        bland=0.0, junk=0.0)
    out = load_confirmed_mineral_positives(str(cdir), template)
    alt_rows = out[out['polygon_id'] == 1]
    leg_rows = out[out['polygon_id'] == 2]
    assert (alt_rows['alteration'] == 1.0).all()
    assert (leg_rows['alteration'] == 0.0).all()  # NaN filled, not propagated


def test_reassigned_preserves_cooccurring_alteration(tmp_path):
    hdir = tmp_path / 'hardneg'
    hdir.mkdir()
    row = _hardneg_row(1, 'olivine_t1')
    row['alteration'] = 1.0  # co-occurring alteration on a mineral reassignment
    row.to_parquet(hdir / 'p_coalt.parquet', index=False)
    out = load_reassigned_minerals(str(hdir))
    assert (out['olivine_t1'] > 0).all()
    assert (out['alteration'] == 1.0).all()


def test_confirmed_loads_from_multiple_dirs(tmp_path):
    d1 = tmp_path / 'old_confirmed'; d1.mkdir()
    d2 = tmp_path / 'new_confirmed'; d2.mkdir()
    # old-style file: no alteration column, High/1.0
    old = _confirmed_row(1, 'hcp', 1.0, 'High').drop(columns=['alteration'])
    old.to_parquet(d1 / 'p_old.parquet', index=False)
    _confirmed_row(2, 'olivine_t1', 0.5, 'Reviewed-Low').to_parquet(
        d2 / 'p_new.parquet', index=False)
    template = _confirmed_row(0, 'olivine_t1', 1.0, 'Reviewed-High').assign(
        bland=0.0, junk=0.0)
    out = load_confirmed_mineral_positives([str(d1), str(d2)], template)
    assert set(out['polygon_id']) == {1, 2}
    assert set(np.unique(out['confidence_weight'])) == {1.0, 0.5}
    assert out['alteration'].notna().all()


def test_hn_loaders_read_multiple_dirs(tmp_path):
    d1 = tmp_path / 'old_hn'; d1.mkdir()
    d2 = tmp_path / 'new_hn'; d2.mkdir()
    _hardneg_row(1, 'other').to_parquet(d1 / 'p_b1.parquet', index=False)
    _hardneg_row(2, 'other').to_parquet(d2 / 'p_b2.parquet', index=False)
    out = load_bland_review([str(d1), str(d2)], 'mc13_blands', mc13=True,
                            seed_offset=10, n_bland=1000)
    assert set(out['polygon_id']) == {1, 2}
    assert (out['bland'] > 0).all()


# ── Task B: unit-balanced splits ─────────────────────────────────────────────

def _spread_tiles(n):
    """n real tile_ids whose centers are pairwise >0.25deg apart (csv order)."""
    cc = pd.read_csv(os.path.join(PROJ, 'data', 'tile_centers.csv'))
    return cc['tile_id'].tolist()[:n]


def _confirmed_poly(tile_id, polygon_id, label_col, n, weight=1.0,
                    tier='Reviewed-High'):
    """A confirmed-pixels parquet frame: n pixels of one class on one tile."""
    d = {
        'tile_id': [tile_id] * n,
        'polygon_id': np.full(n, polygon_id, dtype=np.int64),
        'pixel_row': np.full(n, 750, dtype=np.int64),
        'pixel_col': np.full(n, 750, dtype=np.int64),
    }
    for i in range(59):
        d[f'm{i}'] = np.zeros(n)
    for c in ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase',
              'other', 'alteration']:
        d[c] = np.zeros(n)
    d[label_col] = np.ones(n)
    d['confidence_weight'] = np.full(n, weight)
    d['confidence_tier'] = [tier] * n
    d['split'] = ['train'] * n
    return pd.DataFrame(d)


def _centroid(tile_id, mean_row, mean_col):
    lat, lon = su.tile_center_deg(tile_id)
    lon_c = lon + 5.0 * ((mean_col / 1500.0) - 0.5)
    lat_c = lat - 5.0 * ((mean_row / 1500.0) - 0.5)
    return lat_c, lon_c


def _geo_dist(a, b):
    dlat = a[0] - b[0]
    dlon = (a[1] - b[1] + 180.0) % 360.0 - 180.0
    mlat = math.radians((a[0] + b[0]) / 2.0)
    return math.hypot(dlat, dlon * math.cos(mlat))


def test_confirmed_multi_unit_holds_out_every_class_in_val(tmp_path):
    """Multi-unit confirmed dir -> every mineral class keeps >=5% pixels in val.

    Under the old tile-level split a whole class can land entirely in train
    (esp. a class backed by one tile), leaving 0% in val. The unit-balanced
    splitter's min-holdout guard forces every class to keep >=MIN_HOLDOUT_FRAC.
    """
    cdir = tmp_path / 'confirmed'
    cdir.mkdir()
    tiles = _spread_tiles(40)
    # Classes the confirmed loader actually preserves (plagioclase is zeroed by
    # the loader — it comes from gpkg/reassigned/synth, not confirmed).
    # olivine_t2 gets a single unit (worst case for the min-holdout guard);
    # the rest get 3 spread units each.
    layout = {
        'olivine_t1': tiles[0:3],
        'olivine_t2': tiles[3:4],
        'lcp':        tiles[4:7],
        'hcp':        tiles[7:10],
        'alteration': tiles[10:13],
    }
    pid = 1
    for cls, cls_tiles in layout.items():
        for t in cls_tiles:
            _confirmed_poly(t, pid, cls, 1000).to_parquet(
                cdir / f'p_{pid:08d}.parquet', index=False)
            pid += 1
    template = _confirmed_poly(tiles[0], 0, 'olivine_t1', 3).assign(
        bland=0.0, junk=0.0)
    out = load_confirmed_mineral_positives(str(cdir), template)
    frac = su.achieved_fractions(out, out['split'], list(layout))
    for c in layout:
        assert frac.loc[c, 'val'] >= su.MIN_HOLDOUT_FRAC, (c, frac.loc[c].to_dict())


def test_build_base_overrides_inherited_split_no_leakage(tmp_path):
    """_build_base overrides gpkg rows' inherited splits with the unit splitter.

    Two same-class (plag) polygon pairs straddle adjacent tile edges so each
    pair is <0.25deg apart (one geologic unit); they arrive pre-assigned to
    ALTERNATING train/val. After _build_base, each unit lands in ONE split
    (override happened) and no val polygon sits within LINK_DEG of a same-class
    train polygon.
    """
    # t0433 right edge (col1500)->lon318 ; t0434 left edge (col0)->lon318 : merge
    # t0434 right edge (col1500)->lon323; t0435 left edge (col0)->lon323 : merge
    # the two units are ~5deg apart -> separate.
    def _poly(tile_id, polygon_id, mean_col, split, n=500):
        d = {
            'tile_id': [tile_id] * n,
            'polygon_id': np.full(n, polygon_id, dtype=np.int64),
            'pixel_row': np.full(n, 750, dtype=np.int64),
            'pixel_col': np.full(n, mean_col, dtype=np.int64),
            'other': np.zeros(n),
            'olivine_t1': np.zeros(n), 'olivine_t2': np.zeros(n),
            'lcp': np.zeros(n), 'hcp': np.zeros(n),
            'plagioclase': np.ones(n), 'alteration': np.zeros(n),
            'split': [split] * n,
        }
        return pd.DataFrame(d)

    def _bland(tile_id, polygon_id, n=500):
        d = {
            'tile_id': [tile_id] * n,
            'polygon_id': np.full(n, polygon_id, dtype=np.int64),
            'pixel_row': np.full(n, 750, dtype=np.int64),
            'pixel_col': np.full(n, 750, dtype=np.int64),
            'other': np.ones(n),
            'olivine_t1': np.zeros(n), 'olivine_t2': np.zeros(n),
            'lcp': np.zeros(n), 'hcp': np.zeros(n),
            'plagioclase': np.zeros(n), 'alteration': np.zeros(n),
            'split': ['train'] * n,
        }
        return pd.DataFrame(d)

    meta = [(1, 't0433', 1500), (2, 't0434', 0),
            (3, 't0434', 1500), (4, 't0435', 0)]
    frames = [
        _poly('t0433', 1, 1500, 'train'),
        _poly('t0434', 2, 0,    'val'),
        _poly('t0434', 3, 1500, 'train'),
        _poly('t0435', 4, 0,    'val'),
        _bland('t0101', 10),
        _bland('t0102', 11),
    ]
    base_df = pd.concat(frames, ignore_index=True)
    path = tmp_path / 'base.parquet'
    base_df.to_parquet(path, index=False)

    out = _build_base(str(path), n_bland_target=10_000)
    plag = out[out['plagioclase'] > 0.5]

    # override: each unit (pair) collapses to a single split (was alternating)
    poly_split = {p: plag[plag['polygon_id'] == p]['split'].iloc[0]
                  for p, *_ in meta}
    assert poly_split[1] == poly_split[2], 'unit (t0433/t0434 edge) must share a split'
    assert poly_split[3] == poly_split[4], 'unit (t0434/t0435 edge) must share a split'

    # independent leakage scan: no val plag polygon within LINK_DEG of a
    # same-class train plag polygon.
    cents = {p: _centroid(t, 750, mc) for p, t, mc in meta}
    for p, t, mc in meta:
        if poly_split[p] != 'val':
            continue
        for q, tq, mcq in meta:
            if poly_split[q] != 'train':
                continue
            d = _geo_dist(cents[p], cents[q])
            assert d > su.LINK_DEG, (
                f'val plag poly {p} within {d:.3f}deg of train plag poly {q}')


# ── Task F: joint re-split across sources at concat time ────────────────────

_JOINT_CLASSES = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase',
                  'bland', 'alteration', 'junk']


def _union_poly(tile_id, polygon_id, label_col, split, n=1000,
                mean_row=750, mean_col=750):
    """A concat-time union-frame fragment: n pixels of one class on one tile,
    with a pre-assigned (per-source, provisional) split."""
    d = {
        'tile_id': [tile_id] * n,
        'polygon_id': np.full(n, polygon_id, dtype=np.int64),
        'pixel_row': np.full(n, mean_row, dtype=np.int64),
        'pixel_col': np.full(n, mean_col, dtype=np.int64),
    }
    for c in _JOINT_CLASSES + ['other']:
        d[c] = np.zeros(n)
    d[label_col] = np.ones(n)
    if label_col == 'bland':
        d['other'] = np.ones(n)  # 'other' mirrors bland in the union frame
    d['split'] = [split] * n
    return pd.DataFrame(d)


def _count_leak_pairs(df, cls):
    """Independent centroid math: same-class val/train polygon pairs within
    LINK_DEG (cos-lat scaled, 360-wrap) over the union frame."""
    pos = df[df[cls] > 0.5]
    g = (pos.groupby(['tile_id', 'polygon_id'])
            .agg(mr=('pixel_row', 'mean'), mc=('pixel_col', 'mean'),
                 split=('split', 'first'))
            .reset_index())
    cents = [(_centroid(r.tile_id, r.mr, r.mc), r.split)
             for r in g.itertuples()]
    n = 0
    for ca, sa in cents:
        if sa != 'val':
            continue
        for cb, sb in cents:
            if sb == 'train' and _geo_dist(ca, cb) <= su.LINK_DEG:
                n += 1
    return n


def test_joint_resplit_kills_cross_source_leakage_and_keeps_holdout():
    """Sources split independently leak: a same-class (alteration) polygon
    pair straddling adjacent tiles lands train in one source, val in the
    other. main's joint re-split over the concatenated frame must (a) zero
    out same-class val/train pairs within LINK_DEG across the union and
    (b) keep >=5% val AND test pixels for every class present.
    """
    from scripts.build_7cls_dataset import _joint_resplit

    # Straddling pair: t0433 right edge and t0434 left edge share lon 318.0
    # (<0.25 deg apart => one geologic unit), but come from two different
    # sources whose independent splitters disagreed: train vs val.
    src_a = [_union_poly('t0433', 101, 'alteration', 'train', mean_col=1500)]
    src_b = [_union_poly('t0434', 201, 'alteration', 'val', mean_col=0)]

    # 3 spread units per class (pairwise >0.25 deg apart at tile centers) so
    # the min-holdout guard has donor units for every class.
    tiles = _spread_tiles(3 * len(_JOINT_CLASSES))
    pid, ti = 1000, 0
    for cls in _JOINT_CLASSES:
        for k in range(3):
            dst = src_a if k % 2 == 0 else src_b
            dst.append(_union_poly(tiles[ti], pid, cls, 'train'))
            pid += 1
            ti += 1

    union = pd.concat(src_a + src_b, ignore_index=True)

    # sanity: the constructed union DOES straddle before the joint re-split
    assert _count_leak_pairs(union, 'alteration') >= 1

    labels_before = union[_JOINT_CLASSES + ['other']].copy()
    out = _joint_resplit(union)

    # (a) zero same-class val/train pairs within LINK_DEG across the union
    for cls in _JOINT_CLASSES:
        assert _count_leak_pairs(out, cls) == 0, f'{cls} still leaks'

    # (b) every class present holds >=5% of its pixels in val AND test
    frac = su.achieved_fractions(out, out['split'], _JOINT_CLASSES)
    for cls in _JOINT_CLASSES:
        assert frac.loc[cls, 'val'] >= su.MIN_HOLDOUT_FRAC, (
            cls, frac.loc[cls].to_dict())
        assert frac.loc[cls, 'test'] >= su.MIN_HOLDOUT_FRAC, (
            cls, frac.loc[cls].to_dict())

    # only 'split' is overridden — labels untouched, 'other' still mirrors bland
    assert labels_before.equals(out[_JOINT_CLASSES + ['other']])
    assert (out['other'] == out['bland']).all()


def test_joint_resplit_balance_cols_cover_all_8_classes():
    from scripts.build_7cls_dataset import JOINT_BALANCE_COLS
    assert JOINT_BALANCE_COLS == ['olivine_t1', 'olivine_t2', 'lcp', 'hcp',
                                  'plagioclase', 'bland', 'alteration', 'junk']


# ── Task N: ndviz relabel session ingestion (pixel-level supersede) ──────────

_NDVIZ_OUT_LABEL_COLS = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp',
                         'plagioclase', 'other', 'alteration', 'bland', 'junk']


def _out_frame_row(tile_id, polygon_id, label_col, pixel_rows, pixel_cols,
                   split='train', weight=1.0, tier='High'):
    """A combined-frame ('out') fragment: pixels of one class on one tile,
    carrying the full 7-class label schema + confidence + split."""
    n = len(pixel_rows)
    d = {
        'tile_id': [tile_id] * n,
        'polygon_id': np.full(n, polygon_id, dtype=np.int64),
        'pixel_row': np.array(pixel_rows, dtype=np.int64),
        'pixel_col': np.array(pixel_cols, dtype=np.int64),
    }
    for i in range(59):
        d[f'm{i}'] = np.zeros(n)
    for c in _NDVIZ_OUT_LABEL_COLS:
        d[c] = np.zeros(n)
    d[label_col] = np.ones(n)
    if label_col == 'bland':
        d['other'] = np.ones(n)
    d['confidence_weight'] = np.full(n, weight, dtype=np.float32)
    d['confidence_tier'] = [tier] * n
    d['split'] = [split] * n
    return pd.DataFrame(d)


def _ndviz_relabel_row(tile_id, polygon_id, negative_of, label_col,
                       pixel_rows, pixel_cols, weight=1.0,
                       tier='Reviewed-Moderate'):
    """A review-format ndviz relabel row (full 59-band spectra, real coords,
    confidence_weight/tier, negative_of stamp). label_col=None for pure
    discards / ambiguous where no positive mineral label is written."""
    n = len(pixel_rows)
    d = {
        'tile_id': [tile_id] * n,
        'polygon_id': np.full(n, polygon_id, dtype=np.int64),
        'pixel_row': np.array(pixel_rows, dtype=np.int64),
        'pixel_col': np.array(pixel_cols, dtype=np.int64),
    }
    for i in range(59):
        d[f'm{i}'] = np.zeros(n)
    for c in ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase',
              'other', 'alteration']:
        d[c] = np.zeros(n)
    if label_col is not None:
        d[label_col] = np.ones(n)
    d['confidence_weight'] = np.full(n, weight, dtype=np.float32)
    d['confidence_tier'] = [tier] * n
    d['negative_of'] = [negative_of] * n
    return pd.DataFrame(d)


def test_apply_ndviz_relabels_lcp_to_hcp(tmp_path):
    """Relabel hand-lcp pixels to hcp: those pixels become hcp with the ndviz
    weight/tier, no lcp rows remain at them, other lcp pixels stay intact, and
    the joint re-split runs after (every row gets a split)."""
    from scripts.build_7cls_dataset import _apply_ndviz, _joint_resplit
    t = _spread_tiles(1)[0]
    # 10 base lcp pixels on tile t, rows 0..9 at col 5
    out = _out_frame_row(t, 1, 'lcp', list(range(10)), [5] * 10, split='train')
    ndviz = tmp_path / 'ndviz'
    ndviz.mkdir()
    # relabel the first 5 pixels -> hcp (mineral reassignment, negative_of='')
    _ndviz_relabel_row(t, 1, '', 'hcp', list(range(5)), [5] * 5,
                       weight=0.6, tier='Reviewed-Moderate').to_parquet(
        ndviz / 'r.parquet', index=False)

    out2 = _apply_ndviz(out, str(ndviz))

    relabeled = out2[out2['pixel_row'] < 5]
    assert len(relabeled) == 5
    assert (relabeled['hcp'] > 0.5).all()
    assert (relabeled['lcp'] < 0.5).all()
    assert (relabeled['confidence_weight'] == np.float32(0.6)).all()
    assert (relabeled['confidence_tier'] == 'Reviewed-Moderate').all()
    # no lcp rows survive at the relabeled pixels
    assert len(out2[(out2['pixel_row'] < 5) & (out2['lcp'] > 0.5)]) == 0
    # untouched lcp pixels (rows 5..9) intact
    intact = out2[out2['pixel_row'] >= 5]
    assert len(intact) == 5
    assert (intact['lcp'] > 0.5).all()
    # joint re-split runs after -> every row assigned a valid split
    out3 = _joint_resplit(out2)
    assert out3['split'].isin(['train', 'val', 'test']).all()


def test_apply_ndviz_discard_removes_pixels(tmp_path):
    """A discard (negative_of='<orig class>') suppresses the pixel and adds no
    positive row."""
    from scripts.build_7cls_dataset import _apply_ndviz
    t = _spread_tiles(1)[0]
    out = _out_frame_row(t, 1, 'lcp', list(range(10)), [5] * 10)
    ndviz = tmp_path / 'ndviz'
    ndviz.mkdir()
    # discard rows 0..2: negative_of is the original class name
    _ndviz_relabel_row(t, 1, 'lcp', 'lcp', [0, 1, 2], [5, 5, 5]).to_parquet(
        ndviz / 'd.parquet', index=False)
    out2 = _apply_ndviz(out, str(ndviz))
    assert len(out2) == 7
    assert len(out2[out2['pixel_row'] < 3]) == 0


def test_apply_ndviz_ambiguous_becomes_junk(tmp_path):
    """An ambiguous relabel supersedes the base pixel and adds a junk=1 row."""
    from scripts.build_7cls_dataset import _apply_ndviz
    t = _spread_tiles(1)[0]
    out = _out_frame_row(t, 1, 'lcp', list(range(10)), [5] * 10)
    ndviz = tmp_path / 'ndviz'
    ndviz.mkdir()
    _ndviz_relabel_row(t, 2, 'ambiguous', None, [0, 1, 2], [5, 5, 5],
                       weight=0.75, tier='Reviewed-Moderate').to_parquet(
        ndviz / 'a.parquet', index=False)
    out2 = _apply_ndviz(out, str(ndviz))
    junk = out2[out2['junk'] > 0.5]
    assert len(junk) == 3
    assert (junk['confidence_weight'] == np.float32(0.75)).all()
    assert (junk['confidence_tier'] == 'Reviewed-Moderate').all()
    # base lcp pixels at 0..2 superseded (removed)
    assert len(out2[(out2['pixel_row'] < 3) & (out2['lcp'] > 0.5)]) == 0


def test_apply_ndviz_absent_dir_is_noop(tmp_path):
    """Absent ndviz dir -> output identical to a run without the feature."""
    from scripts.build_7cls_dataset import _apply_ndviz
    t = _spread_tiles(1)[0]
    out = _out_frame_row(t, 1, 'lcp', list(range(10)), [5] * 10)
    out_ref = out.copy()
    out2 = _apply_ndviz(out, str(tmp_path / 'does_not_exist'))
    assert len(out2) == len(out_ref)
    pd.testing.assert_frame_equal(
        out2.reset_index(drop=True), out_ref.reset_index(drop=True))


def test_load_ndviz_relabels_absent_is_empty(tmp_path):
    from scripts.build_7cls_dataset import load_ndviz_relabels
    pos, keys = load_ndviz_relabels(str(tmp_path / 'nope'))
    assert pos.empty
    assert len(keys) == 0


def test_load_ndviz_relabels_suppression_covers_all_decisions(tmp_path):
    """suppression_keys covers EVERY decision type (reassign, alteration,
    ambiguous, AND discard) — every ndviz pixel supersedes lower sources."""
    from scripts.build_7cls_dataset import load_ndviz_relabels
    t = _spread_tiles(1)[0]
    ndviz = tmp_path / 'ndviz'
    ndviz.mkdir()
    _ndviz_relabel_row(t, 1, '', 'hcp', [0], [0]).to_parquet(
        ndviz / 'p1.parquet', index=False)          # reassign
    _ndviz_relabel_row(t, 2, 'alteration', None, [1], [1]).to_parquet(
        ndviz / 'p2.parquet', index=False)           # alteration
    _ndviz_relabel_row(t, 3, 'ambiguous', None, [2], [2]).to_parquet(
        ndviz / 'p3.parquet', index=False)           # junk
    _ndviz_relabel_row(t, 4, 'lcp', 'lcp', [3], [3]).to_parquet(
        ndviz / 'p4.parquet', index=False)           # discard
    pos, keys = load_ndviz_relabels(str(ndviz))
    # 4 pixels suppressed regardless of decision type
    assert len(keys) == 4
    # positives: hcp (reassign) + alteration + junk = 3 rows; discard adds none
    assert len(pos) == 3
    assert (pos['hcp'] > 0.5).sum() == 1
    assert (pos['alteration'] > 0.5).sum() == 1
    assert (pos['junk'] > 0.5).sum() == 1


def test_apply_ndviz_bland_reassignment(tmp_path):
    """negative_of='' with other>0.5 and no mineral -> bland stamp."""
    from scripts.build_7cls_dataset import _apply_ndviz
    t = _spread_tiles(1)[0]
    out = _out_frame_row(t, 1, 'lcp', list(range(4)), [5] * 4)
    ndviz = tmp_path / 'ndviz'
    ndviz.mkdir()
    row = _ndviz_relabel_row(t, 1, '', 'other', [0, 1], [5, 5])
    row.to_parquet(ndviz / 'b.parquet', index=False)
    out2 = _apply_ndviz(out, str(ndviz))
    bland = out2[out2['bland'] > 0.5]
    assert len(bland) == 2
    assert (bland['other'] == bland['bland']).all()
    assert len(out2[(out2['pixel_row'] < 2) & (out2['lcp'] > 0.5)]) == 0


# ── hand-label policy flag (--hand_minerals) ─────────────────────────────────

_MAFIC_COLS = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp']


def _base_poly(tile_id, polygon_id, labels, n=200, mean_row=750, mean_col=750):
    """A base-parquet gpkg/bland fragment: n pixels on one tile with the given
    label columns set to 1.0 (dict {col: value}). Carries the base-parquet
    schema (no bland/junk columns — those are stamped by _build_base)."""
    d = {
        'tile_id': [tile_id] * n,
        'polygon_id': np.full(n, polygon_id, dtype=np.int64),
        'pixel_row': np.full(n, mean_row, dtype=np.int64),
        'pixel_col': np.full(n, mean_col, dtype=np.int64),
        'other': np.zeros(n),
        'olivine_t1': np.zeros(n), 'olivine_t2': np.zeros(n),
        'lcp': np.zeros(n), 'hcp': np.zeros(n),
        'plagioclase': np.zeros(n), 'alteration': np.zeros(n),
        'split': ['train'] * n,
    }
    for c, v in labels.items():
        d[c] = np.full(n, float(v))
    return pd.DataFrame(d)


def _write_policy_base(tmp_path):
    """Synthetic base parquet with spread units: a pure-plag polygon, a
    multi-label plag+olivine polygon, one each of olivine/lcp/hcp mafic
    polygons, and two bland tiles. Returns (path, ids) where ids maps a
    descriptive name to polygon_id."""
    tiles = _spread_tiles(7)
    frames = [
        _base_poly(tiles[0], 1, {'plagioclase': 1.0}),                 # pure plag
        _base_poly(tiles[1], 2, {'plagioclase': 1.0, 'olivine_t1': 1.0}),  # multi
        _base_poly(tiles[2], 3, {'olivine_t1': 1.0}),                  # mafic
        _base_poly(tiles[3], 4, {'lcp': 1.0}),                         # mafic
        _base_poly(tiles[4], 5, {'hcp': 1.0}),                         # mafic
        _base_poly(tiles[5], 10, {'other': 1.0}),                      # bland
        _base_poly(tiles[6], 11, {'other': 1.0}),                      # bland
    ]
    base_df = pd.concat(frames, ignore_index=True)
    path = tmp_path / 'base.parquet'
    base_df.to_parquet(path, index=False)
    ids = {'plag': 1, 'multi': 2, 'olivine': 3, 'lcp': 4, 'hcp': 5,
           'bland_a': 10, 'bland_b': 11}
    return str(path), ids


def test_hand_minerals_all_is_byte_identical_default(tmp_path):
    """Default (no arg) == explicit hand_minerals='all', frame-for-frame."""
    path, _ = _write_policy_base(tmp_path)
    default = _build_base(path, n_bland_target=10_000)
    explicit = _build_base(path, n_bland_target=10_000, hand_minerals='all')
    pd.testing.assert_frame_equal(
        default.reset_index(drop=True), explicit.reset_index(drop=True))
    # every source polygon survives with its labels intact
    for pid in (1, 2, 3, 4, 5, 10, 11):
        assert (default['polygon_id'] == pid).any(), f'polygon {pid} dropped'
    assert (default['plagioclase'] > 0.5).any()
    assert ((default[_MAFIC_COLS] > 0.5).any(axis=1)).any()


def test_hand_minerals_plag_only_keeps_plag_zeroes_mafic(tmp_path):
    """plag_only: keep plag>0.5 non-bland rows (mafic cols zeroed), drop
    olivine/lcp/hcp hand rows, bland tiles untouched."""
    path, ids = _write_policy_base(tmp_path)
    out = _build_base(path, n_bland_target=10_000, hand_minerals='plag_only')

    non_bland = out[out['bland'] < 0.5]
    # plag polygons kept (pure + multi-label)
    assert set(non_bland['polygon_id']) == {ids['plag'], ids['multi']}
    # kept rows all carry plag
    assert (non_bland['plagioclase'] > 0.5).all()
    # NO mafic label leaks through any non-bland row (multi-label row zeroed)
    assert (non_bland[_MAFIC_COLS] <= 0.5).all().all()
    # the pure-mafic polygons are entirely gone
    for pid in (ids['olivine'], ids['lcp'], ids['hcp']):
        assert not (out['polygon_id'] == pid).any()
    # bland tiles intact
    bland = out[out['bland'] > 0.5]
    assert set(bland['polygon_id']) == {ids['bland_a'], ids['bland_b']}


def test_hand_minerals_none_drops_all_nonbland(tmp_path):
    """none: every non-bland gpkg row dropped; only bland tiles remain."""
    path, ids = _write_policy_base(tmp_path)
    out = _build_base(path, n_bland_target=10_000, hand_minerals='none')
    assert (out['bland'] > 0.5).all()
    assert (out['plagioclase'] <= 0.5).all()
    assert (out[_MAFIC_COLS] <= 0.5).all().all()
    assert set(out['polygon_id']) == {ids['bland_a'], ids['bland_b']}

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

"""Tests for the rank-stratified review-set builder.

The builder's product is a calibration curve, so the properties that matter are
(a) a polygon lands in the stratum its score actually belongs to, (b) the
budget is redistributed rather than dropped when cells are empty, (c) already
reviewed polygons never reappear, (d) a seed reproduces the set exactly, and
(e) the review app can read what was emitted.
"""
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from scripts.build_review_set_stratified import (
    BAND_COLS, MINERALS, STRATA, VECTORIZER_COLS,
    allocate_cells, bland_pool_mask, build, equal_fill, load_candidates,
    mean_band_depth, sample_cell, source_uid, stratum_index,
    stratum_layer_name,
)
from scripts.review.polygon_queue import PolygonQueue, _canonical_layer

MARS_WKT = (
    'GEOGCS["GCS_Mars_2000",DATUM["D_Mars_2000",'
    'SPHEROID["Mars_2000_IAU_IAG",3396190,169.8944472]],'
    'PRIMEM["Reference_Meridian",0],UNIT["Degree",0.0174532925199433]]'
)


# --------------------------------------------------------------------------
# synthetic source tree
# --------------------------------------------------------------------------

def _spectrum(depth: float) -> np.ndarray:
    """59-band mean spectrum with a controllable absorption depth."""
    spec = np.full(59, 0.20, dtype=np.float32)
    if depth > 0:
        spec[28:34] = 0.20 * (1.0 - depth)
    return spec


# Source layer names are LITERAL, not derived from stratum_layer_name(): a
# fixture that reuses the code under test would follow it into a bug. These are
# the names the real mc_deploy_pyx vectorization wrote (rank prefix + shortest
# round-tripping threshold), renumbered for the 6 rungs used here.
SOURCE_LAYERS = {
    0: 'thresh_06_0.50', 1: 'thresh_05_0.85', 2: 'thresh_04_0.97',
    3: 'thresh_03_0.99', 4: 'thresh_02_0.999', 5: 'thresh_01_0.9999',
}


def _make_source_gpkg(path, mineral, tile_id, rows):
    """rows: list of (stratum_index, mean_prob, depth). One layer per stratum."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    by_stratum = {}
    for si, mp, depth in rows:
        by_stratum.setdefault(si, []).append((mp, depth))
    for si, items in by_stratum.items():
        lo = STRATA[si][0]
        recs, geoms = [], []
        for k, (mp, depth) in enumerate(items):
            rec = {'tile_id': tile_id, 'mineral': mineral, 'threshold': lo,
                   'count_px': 20 + k, 'mean_prob': mp}
            spec = _spectrum(depth)
            rec.update({c: float(spec[i]) for i, c in enumerate(BAND_COLS)})
            recs.append(rec)
            x = 10.0 + 0.01 * k
            geoms.append(Polygon([(x, 0), (x + 0.005, 0),
                                  (x + 0.005, 0.005), (x, 0.005)]))
        gdf = gpd.GeoDataFrame(pd.DataFrame(recs), geometry=geoms, crs=MARS_WKT)
        gdf = gdf[VECTORIZER_COLS + ['geometry']]
        gdf.to_file(path, layer=SOURCE_LAYERS[si], driver='GPKG')


def _build_tree(root, charts=('mc11', 'mc13', 'mc26'), per_cell=12, seed=0):
    """A miniature mc_deploy_pyx tree: every mineral x chart x stratum filled,
    except plagioclase which (as in the real deployment) has nothing in the
    top two strata."""
    rng = np.random.default_rng(seed)
    for mineral in MINERALS:
        for ci, mc in enumerate(charts):
            rows = []
            for si, (lo, hi) in enumerate(STRATA):
                if mineral == 'plagioclase' and si >= 4:
                    continue
                span = (hi - lo) * 0.98
                for k in range(per_cell):
                    mp = lo + span * (k + 0.5) / per_cell
                    rows.append((si, float(mp), float(rng.uniform(0.0, 0.4))))
            _make_source_gpkg(os.path.join(root, mc, f'{mineral}.gpkg'),
                              mineral, f't{100 + ci}', rows)
    return root


# --------------------------------------------------------------------------
# strata
# --------------------------------------------------------------------------

def test_stratum_boundaries_are_half_open_lower_inclusive():
    # The headline case: 0.99 is the LOWER edge of [0.99, 0.999), not the
    # upper edge of [0.97, 0.99).
    assert stratum_index(0.99) == 3
    assert STRATA[stratum_index(0.99)] == (0.99, 0.999)
    assert stratum_index(0.98999) == 2

    assert stratum_index(0.5) == 0
    assert stratum_index(0.8499999) == 0
    assert stratum_index(0.85) == 1
    assert stratum_index(0.97) == 2
    assert stratum_index(0.999) == 4
    assert stratum_index(0.9999) == 5
    assert stratum_index(1.0) == 5          # last stratum is closed at 1.0


def test_scores_below_the_ladder_have_no_stratum():
    assert stratum_index(0.4999) is None
    assert stratum_index(0.0) is None
    assert stratum_index(float('nan')) is None


def test_stratum_layer_names_are_distinct_and_canonicalise_to_the_stratum():
    physical = [stratum_layer_name(i) for i in range(len(STRATA))]
    assert len(set(physical)) == len(STRATA)
    # The trailing float is what PolygonQueue parses back out, so it must
    # round-trip to the stratum's lower bound (2 dp would send 0.999 and
    # 0.9999 both to 1.00).
    for i, name in enumerate(physical):
        assert float(name.split('_')[-1]) == pytest.approx(STRATA[i][0], abs=1e-12)
    bare = [stratum_layer_name(i, rank_prefixed=False) for i in range(len(STRATA))]
    assert len(set(bare)) == len(STRATA), f'unprefixed names collide: {bare}'
    canon = [_canonical_layer(STRATA[i][0]) for i in range(len(STRATA))]
    assert len(set(canon)) == len(STRATA), f'uid tokens collide: {canon}'
    assert canon[4] == 'thresh_0.999' and canon[5] == 'thresh_0.9999'


# --------------------------------------------------------------------------
# band depth
# --------------------------------------------------------------------------

def test_mean_band_depth_separates_flat_from_absorbing():
    flat = _spectrum(0.0)[None, :]
    deep = _spectrum(0.5)[None, :]
    d_flat = mean_band_depth(flat)[0]
    d_deep = mean_band_depth(deep)[0]
    assert d_flat == pytest.approx(0.0, abs=1e-6)
    assert d_deep > 0.01
    assert d_deep > d_flat


def test_mean_band_depth_is_monotone_in_absorption():
    specs = np.stack([_spectrum(d) for d in (0.0, 0.1, 0.3, 0.6)])
    depths = mean_band_depth(specs)
    assert np.all(np.diff(depths) > 0)


def test_bland_pool_is_the_flattest_fraction():
    cell = pd.DataFrame({
        'cand_key': [f'k{i}' for i in range(10)],
        'band_depth': [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.05],
    })
    mask = bland_pool_mask(cell, 0.30)
    assert mask.sum() == 3
    assert set(cell.loc[mask, 'band_depth']) == {0.05, 0.1, 0.2}


# --------------------------------------------------------------------------
# allocation
# --------------------------------------------------------------------------

def test_equal_fill_spends_the_whole_budget_when_capacity_allows():
    cap = {'a': 100, 'b': 100, 'c': 100}
    alloc = equal_fill(cap, 60, [['a', 'b', 'c']])
    assert sum(alloc.values()) == 60
    assert alloc == {'a': 20, 'b': 20, 'c': 20}


def test_equal_fill_redistributes_from_empty_keys():
    cap = {'a': 100, 'b': 0, 'c': 100}
    alloc = equal_fill(cap, 60, [['a', 'c']])
    assert alloc['b'] == 0
    assert sum(alloc.values()) == 60, 'budget was dropped, not redistributed'


def test_equal_fill_caps_at_total_capacity():
    cap = {'a': 5, 'b': 3}
    alloc = equal_fill(cap, 100, [['a', 'b']])
    assert alloc == {'a': 5, 'b': 3}
    assert sum(alloc.values()) == 8


def test_equal_fill_exhausts_a_tier_before_spilling_to_the_next():
    cap = {'hi': 3, 'lo': 100}
    alloc = equal_fill(cap, 20, [['hi'], ['lo']])
    assert sum(alloc.values()) == 20
    assert alloc['hi'] == 3, 'high-priority tier was not filled to capacity'
    assert alloc['lo'] == 17


def test_allocate_cells_redistributes_into_the_top_strata():
    """Empty plagioclase cells at the top must buy MORE top-stratum polygons
    elsewhere, not shrink the review set."""
    cap = {(m, s): 1000 for m in MINERALS for s in range(len(STRATA))}
    cap[('plagioclase', 5)] = 0
    cap[('plagioclase', 4)] = 0
    alloc = allocate_cells(cap, 600)

    assert sum(alloc.values()) == 600, 'unused budget was dropped'
    assert alloc[('plagioclase', 5)] == 0 and alloc[('plagioclase', 4)] == 0

    base = 600 // 24
    top = [alloc[(m, 5)] for m in MINERALS if m != 'plagioclase']
    bottom = [alloc[(m, 0)] for m in MINERALS]
    assert min(top) > base, f'top stratum did not absorb the freed budget: {top}'
    assert all(b == base for b in bottom), \
        f'freed budget leaked into the LOWEST stratum: {bottom}'


def test_allocate_cells_matches_the_real_deployment_shape():
    """With only plagioclase short at the top, the set still totals `budget`."""
    cap = {(m, s): 500 for m in MINERALS for s in range(len(STRATA))}
    cap[('plagioclase', 5)] = 0
    cap[('plagioclase', 4)] = 7
    cap[('alteration', 5)] = 22
    alloc = allocate_cells(cap, 600)
    assert sum(alloc.values()) == 600
    assert alloc[('plagioclase', 4)] == 7
    assert alloc[('alteration', 5)] == 22


# --------------------------------------------------------------------------
# candidate loading / already-reviewed exclusion
# --------------------------------------------------------------------------

def test_source_uid_matches_what_polygon_queue_would_assign(tmp_path):
    """The exclusion key must be the SAME string the review app writes."""
    path = str(tmp_path / 'src' / 'olivine.gpkg')
    _make_source_gpkg(path, 'olivine', 't0777',
                      [(3, 0.991, 0.1), (3, 0.995, 0.2), (5, 0.99995, 0.3)])
    q_uids = {i.polygon_uid for i in PolygonQueue(path, 'olivine')}
    mine = {source_uid('t0777', 3, 0), source_uid('t0777', 3, 1),
            source_uid('t0777', 5, 0)}
    assert mine <= q_uids, f'{mine - q_uids} not produced by PolygonQueue'


def test_already_reviewed_polygons_are_excluded(tmp_path):
    src = _build_tree(str(tmp_path / 'src'), charts=('mc11',), per_cell=6)
    full = load_candidates(src, ['mc11'], MINERALS, set(), verbose=False)
    assert not full.empty

    victims = sorted(full[full['mineral'] == 'pyx']['source_uid'])[:5]
    dec = tmp_path / 'rev' / 'decisions.csv'
    dec.parent.mkdir(parents=True)
    pd.DataFrame({'polygon_uid': victims, 'decision': ['confirm'] * 5}).to_csv(
        dec, index=False)

    from scripts.build_review_set_stratified import reviewed_uids
    skip = reviewed_uids([str(dec)])
    assert skip == set(victims)

    trimmed = load_candidates(src, ['mc11'], MINERALS, skip, verbose=False)
    kept = set(trimmed[trimmed['mineral'] == 'pyx']['source_uid'])
    assert kept.isdisjoint(victims), 'reviewed polygons came back'
    # Exclusion is by uid, which is only unique WITHIN a mineral, so other
    # minerals may legitimately lose a same-named row; count only pyx.
    n_pyx_before = int((full['mineral'] == 'pyx').sum())
    assert int((trimmed['mineral'] == 'pyx').sum()) == n_pyx_before - 5


def test_candidates_land_in_the_stratum_their_score_implies(tmp_path):
    src = _build_tree(str(tmp_path / 'src'), charts=('mc11',), per_cell=6)
    cands = load_candidates(src, ['mc11'], MINERALS, set(), verbose=False)
    for _, row in cands.iterrows():
        assert stratum_index(row['mean_prob']) == row['stratum']
        lo, hi = STRATA[int(row['stratum'])]
        assert lo <= row['mean_prob'] <= hi
        # candidates for a stratum are drawn from the layer at its lower bound
        assert row['threshold'] == pytest.approx(lo)


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------

def _cell(n=40, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'cand_key': [f'olivine::t100::thresh_0.99::{i}' for i in range(n)],
        'mc': [['mc11', 'mc13', 'mc26'][i % 3] for i in range(n)],
        'band_depth': rng.uniform(0, 1, n),
    })


def test_sampling_is_reproducible_under_a_fixed_seed():
    cell = _cell()
    a = sample_cell(cell, 12, np.random.default_rng(7))
    b = sample_cell(cell, 12, np.random.default_rng(7))
    assert a == b
    # and it is actually random, not a fixed prefix
    c = sample_cell(cell, 12, np.random.default_rng(8))
    assert set(a) != set(c)


def test_sampling_is_independent_of_row_order():
    cell = _cell()
    shuffled = cell.sample(frac=1.0, random_state=3).reset_index(drop=True)
    a = sample_cell(cell, 12, np.random.default_rng(7))
    b = sample_cell(shuffled, 12, np.random.default_rng(7))
    assert set(a) == set(b)


def test_sampling_includes_the_required_bland_share():
    cell = _cell(n=60)
    picks = sample_cell(cell, 20, np.random.default_rng(11),
                        bland_share=0.35, bland_pool_frac=0.30)
    assert len(picks) == 20
    mask = bland_pool_mask(cell, 0.30)
    bland_keys = set(cell.loc[mask, 'cand_key'])
    n_bland = len(set(picks) & bland_keys)
    assert n_bland == 7, f'expected round(0.35*20)=7 bland picks, got {n_bland}'


def test_sampling_spreads_across_charts():
    cell = _cell(n=60)
    picks = sample_cell(cell, 12, np.random.default_rng(5))
    charts = cell.set_index('cand_key').loc[picks, 'mc']
    assert set(charts) == {'mc11', 'mc13', 'mc26'}
    assert charts.value_counts().min() >= 3


def test_sampling_never_exceeds_the_cell():
    cell = _cell(n=5)
    picks = sample_cell(cell, 50, np.random.default_rng(2))
    assert len(picks) == 5
    assert len(set(picks)) == 5


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------

def _run(tmp_path, budget=120, seed=42):
    src = _build_tree(str(tmp_path / 'src'), per_cell=12)
    out = str(tmp_path / 'out')
    return build(src_dir=src, out_dir=out, charts=['mc11', 'mc13', 'mc26'],
                 budget=budget, seed=seed, decisions_globs=[],
                 bland_share=0.35, bland_pool_frac=0.30, verbose=False), out


def test_end_to_end_emits_the_full_budget(tmp_path):
    res, out = _run(tmp_path)
    man = res['manifest']
    assert len(man) == 120
    assert set(man['mineral']) == set(MINERALS)
    assert set(man['mc']) == {'mc11', 'mc13', 'mc26'}
    assert os.path.exists(os.path.join(out, 'allocation.csv'))
    assert os.path.exists(os.path.join(out, 'manifest.csv'))


def test_emitted_gpkg_is_readable_by_polygon_queue(tmp_path):
    res, out = _run(tmp_path)
    man = res['manifest']
    total = 0
    for mineral in MINERALS:
        path = os.path.join(out, f'{mineral}.gpkg')
        assert os.path.exists(path), f'{mineral}.gpkg not written'
        items = list(PolygonQueue(path, mineral))
        expected = int((man['mineral'] == mineral).sum())
        assert len(items) == expected, \
            f'{mineral}: queue yielded {len(items)}, manifest says {expected}'
        total += len(items)
        # Every uid must be unique — a threshold token that collides across two
        # strata would make decisions.csv ambiguous for exactly the rungs the
        # calibration depends on.
        uids = [it.polygon_uid for it in items]
        assert len(set(uids)) == len(uids), \
            f'{mineral}: duplicate polygon_uids emitted'
        # the layer token PolygonQueue reports IS the stratum lower bound
        valid = {_canonical_layer(lo) for lo, _ in STRATA}
        for it in items:
            assert it.layer in valid, \
                f'{mineral}: layer token {it.layer!r} is not a stratum bound'
        # one distinct token per stratum actually present
        n_strata = man.loc[man['mineral'] == mineral, 'stratum'].nunique()
        assert len({it.layer for it in items}) == n_strata
        # queue walks strata strictest-first
        probs = [it.pred_prob for it in items]
        assert probs == sorted(probs, reverse=True)
    assert total == len(man) == 120


def test_manifest_review_uids_match_polygon_queue_uids(tmp_path):
    """decisions.csv will key on PolygonQueue's uid; the manifest must predict
    it exactly, otherwise the completed review cannot be joined to strata."""
    res, out = _run(tmp_path)
    man = res['manifest']
    # Uniqueness is required PER MINERAL — polygon_uid carries no gpkg or class
    # token, so the same string legitimately names a polygon in every mineral's
    # gpkg (see test_exclusion_is_uid_only_and_therefore_conservative).
    for mineral in MINERALS:
        sub = man[man['mineral'] == mineral]
        assert sub['review_uid'].is_unique, \
            f'{mineral}: review_uid collides — two strata share a token'
    for mineral in MINERALS:
        got = {i.polygon_uid for i in
               PolygonQueue(os.path.join(out, f'{mineral}.gpkg'), mineral)}
        want = set(man.loc[man['mineral'] == mineral, 'review_uid'])
        assert got == want
        assert len(want) == int((man['mineral'] == mineral).sum())


def test_emitted_gpkg_carries_the_vectorizer_schema(tmp_path):
    _res, out = _run(tmp_path)
    gdf = gpd.read_file(os.path.join(out, 'pyx.gpkg'),
                        layer=stratum_layer_name(3))
    for col in VECTORIZER_COLS:
        assert col in gdf.columns, f'missing vectorizer column {col}'
    assert list(gdf.columns[:5]) == VECTORIZER_COLS[:5]
    assert gdf.crs is not None
    assert (gdf['mineral'] == 'pyx').all()
    assert np.allclose(gdf['threshold'].to_numpy(), STRATA[3][0])


def test_end_to_end_is_reproducible(tmp_path):
    a, _ = _run(tmp_path / 'a', seed=99)
    b, _ = _run(tmp_path / 'b', seed=99)
    c, _ = _run(tmp_path / 'c', seed=100)
    ka = sorted(a['selected']['cand_key'])
    kb = sorted(b['selected']['cand_key'])
    kc = sorted(c['selected']['cand_key'])
    assert ka == kb
    assert ka != kc


def test_end_to_end_excludes_reviewed_uids(tmp_path):
    src = _build_tree(str(tmp_path / 'src'), per_cell=12)
    first = build(src_dir=src, out_dir=str(tmp_path / 'o1'),
                  charts=['mc11', 'mc13', 'mc26'], budget=120, seed=3,
                  decisions_globs=[], bland_share=0.35, bland_pool_frac=0.30,
                  verbose=False)
    picked = first['manifest']
    dec = tmp_path / 'rev' / 'decisions.csv'
    dec.parent.mkdir(parents=True)
    olv = picked[picked['mineral'] == 'olivine']
    pd.DataFrame({'polygon_uid': olv['source_uid']}).to_csv(dec, index=False)

    second = build(src_dir=src, out_dir=str(tmp_path / 'o2'),
                   charts=['mc11', 'mc13', 'mc26'], budget=120, seed=3,
                   decisions_globs=[str(dec)], bland_share=0.35,
                   bland_pool_frac=0.30, verbose=False)
    got = set(second['manifest'].loc[
        second['manifest']['mineral'] == 'olivine', 'source_uid'])
    assert got.isdisjoint(set(olv['source_uid'])), 'reviewed polygons reappeared'
    assert second['n_excluded_reviewed'] >= len(olv)
    assert len(second['manifest']) == 120, 'budget shrank after exclusion'


def test_exclusion_is_uid_only_and_therefore_conservative(tmp_path):
    """KNOWN LIMITATION, asserted so it cannot change silently.

    polygon_uid is `{tile}::{layer}::{index}` with no gpkg or class in it, so
    the same string names a different polygon in every mineral's gpkg. The
    builder excludes on the bare uid, which can drop a pyx polygon because an
    olivine polygon with the same uid was reviewed. That is conservative — it
    only ever removes candidates — but it is over-exclusion, and it is the
    reason the run reports the excluded count.
    """
    src = _build_tree(str(tmp_path / 'src'), charts=('mc11',), per_cell=6)
    full = load_candidates(src, ['mc11'], MINERALS, set(), verbose=False)
    victims = sorted(full[full['mineral'] == 'olivine']['source_uid'])[:3]
    trimmed = load_candidates(src, ['mc11'], MINERALS, set(victims), verbose=False)
    dropped = len(full) - len(trimmed)
    assert dropped > len(victims), (
        'uid-only exclusion no longer spills across minerals — if the matching '
        'was scoped by class, update this test and the docstring')
    assert set(trimmed['source_uid']).isdisjoint(victims)


def test_end_to_end_keeps_a_bland_share_in_every_stratum(tmp_path):
    res, _out = _run(tmp_path, budget=240)
    man = res['manifest']
    assert man['bland_pool'].mean() >= 0.25
    per_stratum = man.groupby('stratum')['bland_pool'].mean()
    assert (per_stratum > 0).all(), \
        f'a stratum has zero bland-adjacent polygons: {per_stratum.to_dict()}'


def test_no_duplicate_geometry_across_strata(tmp_path):
    """A polygon selected for stratum k cannot reappear as the same shape in
    k+1: identical geometry there would require every pixel >= hi_k, forcing
    mean_prob >= hi_k and excluding it from stratum k."""
    res, _out = _run(tmp_path)
    sel = res['selected']
    assert sel['cand_key'].is_unique
    assert sel[['mineral', 'source_uid']].drop_duplicates().shape[0] == len(sel)

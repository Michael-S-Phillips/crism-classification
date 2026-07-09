"""Tests for scripts/split_units.py — unit-aware pixel-balanced splitter.

Written test-first (TDD). Six groups per the plan:
  1. tile_center_deg from committed csv (+ full coverage of base-parquet tiles)
  2. cross-tile merge / distant separation
  3. ±5% balance on 40 synthetic units
  4. min-holdout guard
  5. determinism + no polygon spans splits
  6. leakage regression (no val polygon within link_deg of same-class train)
"""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))

import split_units as su

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_PARQUET = os.path.join(PROJ, 'data', 'mrral_pixels.parquet')


# ── helpers ────────────────────────────────────────────────────────────────

def make_poly(tile_id, polygon_id, mean_row, mean_col, n, labels, label_cols):
    """n rows for one polygon at (mean_row, mean_col) with given positive labels."""
    rec = {
        'tile_id': [tile_id] * n,
        'polygon_id': [polygon_id] * n,
        'pixel_row': [mean_row] * n,
        'pixel_col': [mean_col] * n,
    }
    for c in label_cols:
        rec[c] = [1.0 if c in labels else 0.0] * n
    return pd.DataFrame(rec)


# ── Group 1: tile_center_deg ─────────────────────────────────────────────────

def test_tile_center_known_values():
    assert su.tile_center_deg('t1444') == (32.5, 330.5)
    assert su.tile_center_deg('t1249') == (22.5, 75.5)
    assert su.tile_center_deg('t0434') == (-37.5, 320.5)


def test_tile_center_unknown_raises():
    with pytest.raises(KeyError):
        su.tile_center_deg('t9999')


@pytest.mark.skipif(not os.path.exists(BASE_PARQUET), reason='base parquet not available')
def test_csv_covers_all_base_parquet_tiles():
    tiles = pd.read_parquet(BASE_PARQUET, columns=['tile_id'])['tile_id'].unique()
    for t in tiles:
        lat, lon = su.tile_center_deg(t)  # must not raise
        assert -90 <= lat <= 90
        assert 0 <= lon <= 360


# ── Group 2: cross-tile merge / separation ──────────────────────────────────

def test_cross_tile_merge_and_separation():
    lc = ['a']
    # t1379 center (32.5, 5.5): east edge (col 1500) -> lon 8.0
    # t1380 center (32.5, 10.5): west edge (col 0)   -> lon 8.0  => coincide
    # t1380 center pixel (col 750) -> lon 10.5, ~2.5 deg away -> separate
    p1 = make_poly('t1379', 1, 750, 1500, 3, ['a'], lc)
    p2 = make_poly('t1380', 2, 750, 0, 3, ['a'], lc)
    p3 = make_poly('t1380', 3, 750, 750, 3, ['a'], lc)
    df = pd.concat([p1, p2, p3], ignore_index=True)
    units = su.polygon_units(df)
    u = {pid: units[df.polygon_id == pid].iloc[0] for pid in (1, 2, 3)}
    assert u[1] == u[2], 'adjacent-tile polygons <0.25deg apart should merge'
    assert u[3] != u[1], 'polygon ~2.5deg away should be separate'


# ── Group 3: balance ─────────────────────────────────────────────────────────

def _spread_tiles(n):
    """n real tile_ids whose centers are pairwise >0.25deg apart."""
    cc = pd.read_csv(os.path.join(PROJ, 'data', 'tile_centers.csv'))
    return cc['tile_id'].tolist()[:n]


def test_balance_40_units_within_5pct():
    lc = ['a', 'b', 'c']
    tiles = _spread_tiles(40)
    rng = np.random.default_rng(0)
    parts = []
    for i, t in enumerate(tiles):
        size = int(rng.integers(500, 5000))
        # each unit carries 1-3 classes
        labels = [c for c in lc if rng.random() < 0.6] or ['a']
        parts.append(make_poly(t, i, 750, 750, size, labels, lc))
    df = pd.concat(parts, ignore_index=True)
    splits = su.assign_unit_balanced_splits(df, lc, seed=42)
    frac = su.achieved_fractions(df, splits, lc)
    for c in lc:
        assert abs(frac.loc[c, 'train'] - 0.70) <= 0.05, (c, frac.loc[c])
        assert abs(frac.loc[c, 'val'] - 0.15) <= 0.05, (c, frac.loc[c])
        assert abs(frac.loc[c, 'test'] - 0.15) <= 0.05, (c, frac.loc[c])


# ── Group 4: min-holdout guard ──────────────────────────────────────────────

def test_min_holdout_guard_fires():
    lc = ['x', 'y']
    tiles = _spread_tiles(20)
    parts = []
    # class x only in 2 units, no other class -> greedy tends to skip holdout
    parts.append(make_poly(tiles[0], 0, 750, 750, 100_000, ['x'], lc))
    parts.append(make_poly(tiles[1], 1, 750, 750, 90_000, ['x'], lc))
    # class y spread over many units so greedy is busy elsewhere
    for i, t in enumerate(tiles[2:18], start=2):
        parts.append(make_poly(t, i, 750, 750, 3000, ['y'], lc))
    df = pd.concat(parts, ignore_index=True)
    splits = su.assign_unit_balanced_splits(df, lc, seed=42)
    frac = su.achieved_fractions(df, splits, lc)
    assert frac.loc['x', 'val'] >= su.MIN_HOLDOUT_FRAC
    assert frac.loc['x', 'test'] >= su.MIN_HOLDOUT_FRAC


# ── Group 5: determinism + no polygon spans splits ──────────────────────────

def test_determinism_and_no_polygon_split():
    lc = ['a', 'b']
    tiles = _spread_tiles(30)
    rng = np.random.default_rng(1)
    parts = []
    for i, t in enumerate(tiles):
        parts.append(make_poly(t, i, 750, 750, int(rng.integers(500, 4000)),
                               [c for c in lc if rng.random() < 0.7] or ['a'], lc))
    df = pd.concat(parts, ignore_index=True)
    s1 = su.assign_unit_balanced_splits(df, lc, seed=7)
    s2 = su.assign_unit_balanced_splits(df, lc, seed=7)
    assert (s1.values == s2.values).all(), 'same seed must give identical assignment'
    # no polygon spans splits
    tmp = df.copy()
    tmp['split'] = s1.values
    per_poly = tmp.groupby(['tile_id', 'polygon_id'])['split'].nunique()
    assert (per_poly == 1).all(), 'a polygon must not span multiple splits'


# ── Group 6: leakage regression ─────────────────────────────────────────────

def _centroid(tile_id, mean_row, mean_col):
    lat, lon = su.tile_center_deg(tile_id)
    lon_c = lon + 5.0 * ((mean_col / 1500.0) - 0.5)
    lat_c = lat - 5.0 * ((mean_row / 1500.0) - 0.5)
    return lat_c, lon_c


def _geo_dist(a, b, link=None):
    dlat = a[0] - b[0]
    dlon = (a[1] - b[1] + 180.0) % 360.0 - 180.0
    mlat = math.radians((a[0] + b[0]) / 2.0)
    return math.hypot(dlat, dlon * math.cos(mlat))


def test_no_val_polygon_near_same_class_train():
    lc = ['a', 'b', 'c']
    tiles = _spread_tiles(40)
    rng = np.random.default_rng(3)
    parts = []
    meta = []
    pid = 0
    for i, t in enumerate(tiles):
        # place a couple of adjacent polygons per tile to create real proximity
        for k in range(2):
            mr, mc = 500 + 400 * k, 500 + 400 * k
            labels = [c for c in lc if rng.random() < 0.6] or ['a']
            parts.append(make_poly(t, pid, mr, mc, int(rng.integers(500, 4000)), labels, lc))
            meta.append((pid, t, mr, mc, labels))
            pid += 1
    df = pd.concat(parts, ignore_index=True)
    splits = su.assign_unit_balanced_splits(df, lc, seed=42)
    poly_split = {p: splits[df.polygon_id == p].iloc[0] for p, *_ in meta}
    cents = {p: _centroid(t, mr, mc) for p, t, mr, mc, _ in meta}
    plabels = {p: labs for p, t, mr, mc, labs in meta}
    for p, t, mr, mc, labs in meta:
        if poly_split[p] != 'val':
            continue
        for q, *_ in meta:
            if poly_split[q] != 'train':
                continue
            shared = set(plabels[p]) & set(plabels[q])
            if not shared:
                continue
            d = _geo_dist(cents[p], cents[q])
            assert d > su.LINK_DEG, (
                f'val poly {p} within {d:.3f}deg of train poly {q}, shared {shared}')

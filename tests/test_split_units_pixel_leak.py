"""
FAILING regression test reproducing the review-only (--hand_minerals plag_only)
train/val pixel leak (2026-07-29 diagnosis, see
.superpowers/scratch/reviewonly_leak_diagnosis.md).

Root cause: split_units.polygon_units() clusters polygons into geographic
"units" using single-linkage on POLYGON CENTROID distance (LINK_DEG=0.25deg).
The MC13 review's confirmed_pixels threshold-ladder re-review process can
produce TWO distinct polygon_id entries for the same underlying detection at
different confidence thresholds / re-review passes; these nested/overlapping
polygons share literal (tile_id, pixel_row, pixel_col) pixels (empirically:
226,581 of 1,405,775 raw confirmed pixels -- ~16% -- are hit by >1
polygon_id; see repro_confirmed_leak.py / repro_joint_leak.py in the scratch
dir, which reproduce the real HPC build
(logs/7cls_data_rvw_23174610.log) and find 30,631 physical pixels (1.454%)
spanning >1 split in the shipped parquet after the joint re-split).

The mechanism: when two polygons that share literal pixels have CENTROIDS
more than LINK_DEG apart (observed in the real data: ~0.4-1.0deg -- a broad,
loose-threshold detection blob's mean centroid is pulled toward the whole
blob's mass, while a tight sub-region re-review polygon nested inside it can
sit far from that mean), single-linkage clustering in polygon_units() places
them in DIFFERENT units. assign_unit_balanced_splits() then assigns whole
units independently to train/val/test, so the SAME physical pixel can end up
as one row under a train-assigned polygon and another row under a
val/test-assigned polygon -- a literal duplicate-pixel leak.

This is the precise, deterministic root cause: polygon_units() should put any
two polygons that share >=1 literal pixel in the same unit REGARDLESS of
centroid distance, but it currently does not.

This test is intentionally NOT committed (scratch/local-only); it is expected
to FAIL against the current split_units.py.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))

import split_units as su


def _mk(tile_id, polygon_id, pixel_rows, pixel_cols, label_cols, positive_label):
    n = len(pixel_rows)
    rec = {
        'tile_id': [tile_id] * n,
        'polygon_id': [polygon_id] * n,
        'pixel_row': list(pixel_rows),
        'pixel_col': list(pixel_cols),
    }
    for c in label_cols:
        rec[c] = [1.0 if c == positive_label else 0.0] * n
    return pd.DataFrame(rec)


def _threshold_ladder_pair(tile='t0638'):
    """Build a (broad, tight) polygon pair mirroring the real repro
    (t0638, poly 949 = broad / poly 852 = tight): a loose-threshold blob
    (mostly pixels far away, near row 200) that ALSO includes a tail of
    pixels at rows 1170-1178, plus a strict-threshold re-review polygon that
    covers ONLY that same row-1170-1178 tail -- i.e. the two polygons
    literally share 9 physical pixels, but the broad polygon's mean centroid
    is dragged far away by its other 4000 pixels near row 200.
    """
    far_rows = np.full(4000, 200)
    far_cols = np.arange(4000) % 1500
    shared_rows = np.array([1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178])
    shared_cols = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900])
    broad_rows = np.concatenate([far_rows, shared_rows])
    broad_cols = np.concatenate([far_cols, shared_cols])
    broad = _mk(tile, 949, broad_rows, broad_cols, ['a'], 'a')
    tight = _mk(tile, 852, shared_rows, shared_cols, ['a'], 'a')
    return broad, tight, shared_rows, shared_cols


def test_overlapping_pixel_polygons_share_a_unit():
    """Root-cause assertion: two polygons that share literal pixels MUST be
    clustered into the same geographic unit by polygon_units(), regardless
    of how far apart their CENTROIDS land. Currently FALSE: the broad
    polygon's centroid is dragged ~3.2deg away by its other pixels, so
    single-linkage centroid clustering (LINK_DEG=0.25deg) puts them in
    different units even though they share 9 literal pixels.
    """
    broad, tight, shared_rows, shared_cols = _threshold_ladder_pair()
    df = pd.concat([broad, tight], ignore_index=True)

    # sanity: the two polygons really do share literal pixels
    key_cols = ['tile_id', 'pixel_row', 'pixel_col']
    dup = df.groupby(key_cols)['polygon_id'].nunique()
    n_shared = int((dup > 1).sum())
    assert n_shared == len(shared_rows), (
        f'test setup: expected {len(shared_rows)} literal shared pixels, got {n_shared}')

    units = su.polygon_units(df)
    u_broad = units[df.polygon_id == 949].iloc[0]
    u_tight = units[df.polygon_id == 852].iloc[0]
    assert u_broad == u_tight, (
        f'polygons 949 (broad) and 852 (tight) share {n_shared} literal pixels '
        f'but were placed in DIFFERENT units ({u_broad} vs {u_tight}) -- '
        'polygon_units() clusters by centroid distance only and misses '
        'literal pixel overlap between polygons whose centroids are '
        '>LINK_DEG apart (the nested threshold-ladder review pattern).')


def test_split_assignment_does_not_duplicate_a_pixel_across_splits():
    """End-to-end illustration: fold the two overlapping-pixel polygons above
    into a small corpus and run assign_unit_balanced_splits (the actual
    function build_7cls_dataset.py calls) -- no physical pixel should ever
    appear in more than one split.

    NOTE: whether this particular filler composition/seed actually drives the
    two units into different splits depends on the greedy balancer's
    corpus-wide holdout accounting (it is not deterministic w.r.t. filler
    choice the way test_overlapping_pixel_polygons_share_a_unit is), so this
    test may pass even though the underlying unit-assignment bug is present.
    It is kept as a secondary, closer-to-real-pipeline illustration; the
    authoritative regression is test_overlapping_pixel_polygons_share_a_unit
    above, and the real-data confirmation is
    repro_confirmed_leak.py / repro_joint_leak.py in the scratch dir (30,631
    leaked pixels in the actual reviewonly build).
    """
    lc = ['a']
    broad, tight, shared_rows, shared_cols = _threshold_ladder_pair()
    df = pd.concat([broad, tight], ignore_index=True)

    tiles_csv = pd.read_csv(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'tile_centers.csv'))
    filler_tiles = [t for t in tiles_csv['tile_id'].tolist() if t != 't0638'][:40]
    rng = np.random.default_rng(0)
    filler_parts = []
    for i, t in enumerate(filler_tiles):
        n = int(rng.integers(200, 1200))
        filler_parts.append(_mk(t, 2000 + i, [750] * n,
                                 list(rng.integers(0, 1500, size=n)), lc, 'a'))
    df = pd.concat([df] + filler_parts, ignore_index=True)

    splits = su.assign_unit_balanced_splits(df, lc, seed=42)
    tmp = df.copy()
    tmp['split'] = splits.values

    key_cols = ['tile_id', 'pixel_row', 'pixel_col']
    g = tmp.groupby(key_cols)['split'].nunique()
    leaked = g[g > 1]
    assert len(leaked) == 0, (
        f'{len(leaked)} physical pixels span >1 split -- literal pixel-level '
        f'leak from the broad/tight threshold-ladder polygon pair '
        f'(poly 949 split={tmp[tmp.polygon_id == 949].split.iloc[0]!r}, '
        f'poly 852 split={tmp[tmp.polygon_id == 852].split.iloc[0]!r})')

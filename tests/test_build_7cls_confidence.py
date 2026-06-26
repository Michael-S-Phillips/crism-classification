import os
import numpy as np
import pandas as pd

from scripts.build_7cls_dataset import load_confirmed_mineral_positives, load_bland_review, load_reassigned_minerals


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

import os
import numpy as np
import pandas as pd

from scripts.build_7cls_dataset import load_confirmed_mineral_positives


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

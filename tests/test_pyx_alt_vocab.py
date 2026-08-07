"""Hand-labeled 5-class pyx vocab (olivine/pyx/plagioclase/other/alteration)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import data.dataset as D
import scripts.classify_tile_supervised as C


def test_label_cols_pyx_alt_definition():
    assert D.LABEL_COLS_PYX_ALT == [
        'olivine', 'pyx', 'plagioclase', 'other', 'alteration']


def test_collapse_labels_builds_pyx_alt_columns():
    df = pd.DataFrame({
        'olivine_t1': [1.0, 0.0, 0.0, 0.0],
        'olivine_t2': [0.0, 0.0, 0.0, 0.0],
        'lcp':        [0.0, 1.0, 0.0, 0.0],
        'hcp':        [0.0, 0.0, 1.0, 0.0],
        'plagioclase':[0.0, 0.0, 0.0, 1.0],
        'other':      [0.0, 0.0, 0.0, 0.0],
        'alteration': [0.0, 0.0, 1.0, 0.0],
    })
    out = D._collapse_labels(df)
    for c in D.LABEL_COLS_PYX_ALT:
        assert c in out.columns, f'missing {c}'
    # pyx is the LCP/HCP merge
    np.testing.assert_allclose(
        out['pyx'].values, df[['lcp', 'hcp']].max(axis=1).values)
    assert out['pyx'].tolist() == [0.0, 1.0, 1.0, 0.0]


def test_classify_pyx_alt_overrides_5class_default():
    C.PYX_ALT_MODE = True
    try:
        C._set_n_classes({'head.weight': np.zeros((5, 272))})
        assert C.CLASS_NAMES == [
            'olivine', 'pyx', 'plagioclase', 'other', 'alteration']
        assert C.N_CLASSES == 5
    finally:
        C.PYX_ALT_MODE = False

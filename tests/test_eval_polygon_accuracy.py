"""Unit tests for scripts.eval_polygon_accuracy polygon→class logic.

Covers the helpers most likely to silently regress:

  * parse_category — gpkg ``Category`` string → (mineral list, tier)
  * minerals_to_label_indices — drop OOV (alteration/spinel) cleanly
  * is_pure_in_label_space — confusion-matrix eligibility
  * tile_region — region heuristic with the defaults and a custom override
  * confusion_matrix_single_mineral — multi-mineral polygons MUST be excluded
  * aggregate_results.correct — multi-mineral polygons are correct on ANY hit

Inference / rasterization is exercised in the smoke test, not here.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'scripts'))
from scripts.eval_polygon_accuracy import (
    DEFAULT_REGIONS,
    LABEL_COLS,
    aggregate_results,
    confusion_matrix_single_mineral,
    is_pure_in_label_space,
    minerals_to_label_indices,
    parse_category,
    per_class_summary,
    tile_region,
)


def test_parse_category_simple():
    mins, tier = parse_category('plagioclase (High)')
    assert mins == ['plagioclase']
    assert tier == 'High'


def test_parse_category_type1_olivine_collapses_to_olivine():
    """Type 1 / Type 2 olivine both parse as 'olivine' — that's the whole
    point of collapsing subtypes; the model output space is 5-class."""
    mins, tier = parse_category('Type 1 olivine (Moderate)')
    assert mins == ['olivine']
    assert tier == 'Moderate'

    mins2, tier2 = parse_category('Type 2 olivine (High)')
    assert mins2 == ['olivine']
    assert tier2 == 'High'


def test_parse_category_mixed():
    mins, tier = parse_category('hcp + olivine (High)')
    assert set(mins) == {'olivine', 'hcp'}
    assert tier == 'High'


def test_parse_category_typo_plagiolcase():
    """At least one gpkg has 'plagiolcase' typo. Spec says we still parse it."""
    mins, _ = parse_category('plagiolcase (Low)')
    assert mins == ['plagioclase']


def test_parse_category_other():
    mins, tier = parse_category('Other (High)')
    assert mins == ['other']
    assert tier == 'High'


def test_parse_category_empty():
    assert parse_category('') == ([], '')
    assert parse_category(None) == ([], '')


def test_minerals_to_label_indices_drops_oov():
    """alteration / spinel are out of the 5-class output space."""
    idx = minerals_to_label_indices(['alteration', 'plagioclase'])
    assert idx == {LABEL_COLS.index('plagioclase')}
    idx2 = minerals_to_label_indices(['spinel'])
    assert idx2 == set()


def test_is_pure_in_label_space():
    """alteration + plagioclase has exactly one in-vocab mineral, so it counts
    as pure for confusion-matrix purposes."""
    assert is_pure_in_label_space(['plagioclase'])
    assert is_pure_in_label_space(['alteration', 'plagioclase'])  # OOV strip
    assert not is_pure_in_label_space(['hcp', 'olivine'])
    assert not is_pure_in_label_space(['alteration', 'plagioclase', 'olivine'])


def test_tile_region_defaults():
    assert tile_region('t0433', DEFAULT_REGIONS) == 'argyre'
    assert tile_region('t1249', DEFAULT_REGIONS) == 'nili'
    assert tile_region('t0183', DEFAULT_REGIONS) == 'hellas'
    # Anything else falls through to 'cmu'
    assert tile_region('t9999', DEFAULT_REGIONS) == 'cmu'


def test_tile_region_custom_override():
    custom = {'argyre': {'t9999'}}
    assert tile_region('t9999', custom) == 'argyre'
    assert tile_region('t0433', custom) == 'cmu'


# -------------- aggregate_results / confusion matrix / per-class
def _make_polys():
    """Three polygons:
      A: pure plagioclase (High), correctly predicted
      B: pure plagioclase (High), incorrectly predicted as 'olivine'
      C: mixed 'hcp + olivine' (Moderate), predicted 'olivine' → correct
         (any-hit rule)
    """
    return [
        {'polygon_uid': 'TX/A', 'tile_id': 't0433',
         'category': 'plagioclase (High)', 'minerals': ['plagioclase'],
         'confidence_tier': 'High',
         'true_indices': [LABEL_COLS.index('plagioclase')],
         'is_pure': True,
         'geometry': None, 'polygon_id': 'A'},
        {'polygon_uid': 'TX/B', 'tile_id': 't0433',
         'category': 'plagioclase (High)', 'minerals': ['plagioclase'],
         'confidence_tier': 'High',
         'true_indices': [LABEL_COLS.index('plagioclase')],
         'is_pure': True,
         'geometry': None, 'polygon_id': 'B'},
        {'polygon_uid': 'TX/C', 'tile_id': 't1249',
         'category': 'hcp + olivine (Moderate)',
         'minerals': ['olivine', 'hcp'],
         'confidence_tier': 'Moderate',
         'true_indices': sorted({LABEL_COLS.index('olivine'),
                                  LABEL_COLS.index('hcp')}),
         'is_pure': False,
         'geometry': None, 'polygon_id': 'C'},
    ]


def _make_inference(pred_classes, n_pixels=(100, 100, 100)):
    """Construct the inference-style dict aggregate_results expects."""
    n = len(pred_classes)
    probs = np.zeros((n, len(LABEL_COLS)), dtype=np.float32)
    for i, c in enumerate(pred_classes):
        probs[i, c] = 0.9
    return {
        'polygon_mean_probs': probs,
        'n_pixels': np.array(n_pixels, dtype=np.int32),
        'pred_class': np.array(pred_classes, dtype=np.int32),
        'per_tile_pixel_preds': {},
        'n_no_tile': 0,
        'n_no_pixel': 0,
    }


def test_aggregate_results_correctness_multi_truth_any_hit():
    polys = _make_polys()
    pred = [LABEL_COLS.index('plagioclase'),     # A correct
            LABEL_COLS.index('olivine'),         # B wrong
            LABEL_COLS.index('olivine')]         # C correct (any-hit)
    inf = _make_inference(pred)
    df = aggregate_results(polys, inf, DEFAULT_REGIONS)
    # Pandas wraps the bools in numpy scalars; compare by truthiness, not identity.
    assert bool(df.loc[0, 'correct']) is True
    assert bool(df.loc[1, 'correct']) is False
    assert bool(df.loc[2, 'correct']) is True


def test_confusion_matrix_excludes_mixed_polygons():
    polys = _make_polys()
    pred = [LABEL_COLS.index('plagioclase'),
            LABEL_COLS.index('olivine'),
            LABEL_COLS.index('olivine')]    # C is mixed; must not enter CM
    inf = _make_inference(pred)
    df = aggregate_results(polys, inf, DEFAULT_REGIONS)
    cm = confusion_matrix_single_mineral(df)
    # Total CM count should equal #single-mineral polys with a prediction (= 2)
    assert cm.values.sum() == 2
    plag = LABEL_COLS.index('plagioclase')
    oliv = LABEL_COLS.index('olivine')
    assert cm.iloc[plag, plag] == 1     # A: plag → plag
    assert cm.iloc[plag, oliv] == 1     # B: plag → olivine
    # Sanity: no row for olivine truth (we have none in single-mineral set)
    assert cm.iloc[oliv].sum() == 0


def test_per_class_summary_only_pure():
    polys = _make_polys()
    pred = [LABEL_COLS.index('plagioclase'),
            LABEL_COLS.index('olivine'),
            LABEL_COLS.index('olivine')]
    inf = _make_inference(pred)
    df = aggregate_results(polys, inf, DEFAULT_REGIONS)
    cls_df = per_class_summary(df)
    # Plag has 2 (single-mineral) polys, 1 correct → recall 0.5
    plag_row = cls_df[cls_df['true_class'] == 'plagioclase']
    assert len(plag_row) == 1
    assert plag_row.iloc[0]['n_polygons'] == 2
    assert plag_row.iloc[0]['recall'] == 0.5
    # Olivine should NOT appear — C is mixed and is_pure=False
    assert 'olivine' not in cls_df['true_class'].values


def test_aggregate_results_nan_prediction_marks_correct_none():
    """If a polygon has no in-tile pixels, pred_class == -1 → correct == None."""
    polys = _make_polys()[:1]
    inf = {
        'polygon_mean_probs': np.full((1, len(LABEL_COLS)), np.nan, dtype=np.float32),
        'n_pixels': np.array([0], dtype=np.int32),
        'pred_class': np.array([-1], dtype=np.int32),
        'per_tile_pixel_preds': {},
        'n_no_tile': 0,
        'n_no_pixel': 1,
    }
    df = aggregate_results(polys, inf, DEFAULT_REGIONS)
    # ``correct=None`` is stored as a python None (not a numpy nan), so identity
    # check is fine.
    assert df.loc[0, 'correct'] is None
    assert df.loc[0, 'predicted_class'] == 'NONE'

import numpy as np
import pandas as pd
import pytest

import data.dataset as ds
from data.dataset import _collapse_labels


@pytest.fixture(autouse=True)
def _reset_scheme():
    ds.set_weight_scheme('level')
    yield
    ds.set_weight_scheme('level')


def _row(tier, weight):
    return pd.DataFrame({
        'olivine_t1': [0.0], 'olivine_t2': [0.0], 'lcp': [0.0], 'hcp': [1.0],
        'plagioclase': [0.0], 'other': [0.0],
        'confidence_tier': [tier], 'confidence_weight': [weight],
    })


def _w(tier, weight):
    return float(_collapse_labels(_row(tier, weight))['confidence_weight'].iloc[0])


def test_level_matches_todays_behaviour():
    # Hand tiers resolve through the table.
    assert _w('High', 0.1) == pytest.approx(1.0)
    assert _w('Moderate', 0.1) == pytest.approx(0.85)
    assert _w('Low', 0.1) == pytest.approx(0.70)
    # Reviewed-* deliberately pass the stamped weight through untouched.
    assert _w('Reviewed-High', 1.0) == pytest.approx(1.0)
    assert _w('Reviewed-Moderate', 0.75) == pytest.approx(0.75)
    assert _w('Reviewed-Low', 0.5) == pytest.approx(0.5)


def test_review_up_scales_reviewed_tiers():
    ds.set_weight_scheme('review_up')
    assert _w('Reviewed-High', 1.0) == pytest.approx(2.0)
    assert _w('Reviewed-Moderate', 0.75) == pytest.approx(1.7)
    assert _w('High', 0.1) == pytest.approx(1.0)   # hand untouched


def test_hand_up_scales_hand_tiers():
    ds.set_weight_scheme('hand_up')
    assert _w('High', 0.1) == pytest.approx(1.5)
    assert _w('Moderate', 0.1) == pytest.approx(1.3)
    assert _w('Reviewed-High', 1.0) == pytest.approx(1.0)


def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match='nonesuch'):
        ds.set_weight_scheme('nonesuch')


def test_active_scheme_roundtrip():
    ds.set_weight_scheme('review_up')
    assert ds.active_weight_scheme() == 'review_up'

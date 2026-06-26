import numpy as np
import pandas as pd
import pytest

from data.dataset import _collapse_labels


def _row(tier, weight):
    d = {'olivine_t1': [0.0], 'olivine_t2': [0.0], 'lcp': [0.0], 'hcp': [1.0],
         'plagioclase': [0.0], 'other': [0.0],
         'confidence_tier': [tier], 'confidence_weight': [weight]}
    return pd.DataFrame(d)


def test_reviewed_tiers_use_stamped_weight():
    for tier, w in [('Reviewed-High', 1.0), ('Reviewed-Moderate', 0.75),
                    ('Reviewed-Low', 0.5)]:
        out = _collapse_labels(_row(tier, w))
        assert float(out['confidence_weight'].iloc[0]) == w, tier


def test_base_high_moderate_low_unchanged():
    # Global tiers still map through _TIER_WEIGHTS, NOT the stamped weight.
    # Use approx because the result is np.float32 (limited precision).
    assert float(_collapse_labels(_row('Moderate', 0.50))['confidence_weight'].iloc[0]) == pytest.approx(0.85)
    assert float(_collapse_labels(_row('Low', 0.25))['confidence_weight'].iloc[0]) == pytest.approx(0.70)

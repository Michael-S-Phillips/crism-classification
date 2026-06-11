import numpy as np
import pytest
from data.label_parser import parse_category, get_confidence_tier, CLASSES

def test_classes_order():
    assert CLASSES == ['olivine_t1', 'olivine_t2', 'lcp', 'hcp',
                       'plagioclase', 'other', 'alteration']

def test_type1_olivine_high():
    label, weight = parse_category("Type 1 olivine (High)")
    np.testing.assert_array_almost_equal(label, [1, 0, 0, 0, 0, 0, 0])
    assert weight == 1.0

def test_type2_olivine_moderate():
    label, weight = parse_category("Type 2 olivine (Moderate)")
    np.testing.assert_array_almost_equal(label, [0, 1, 0, 0, 0, 0, 0])
    assert weight == 0.5

def test_lcp_high():
    label, weight = parse_category("lcp (High)")
    np.testing.assert_array_almost_equal(label, [0, 0, 1, 0, 0, 0, 0])
    assert weight == 1.0

def test_hcp_low():
    label, weight = parse_category("hcp (Low)")
    np.testing.assert_array_almost_equal(label, [0, 0, 0, 1, 0, 0, 0])
    assert weight == 0.25

def test_plagioclase_moderate():
    label, weight = parse_category("plagioclase (Moderate)")
    np.testing.assert_array_almost_equal(label, [0, 0, 0, 0, 1, 0, 0])
    assert weight == 0.5

def test_other_high():
    label, weight = parse_category("Other (High)")
    np.testing.assert_array_almost_equal(label, [0, 0, 0, 0, 0, 1, 0])
    assert weight == 1.0

def test_hcp_plus_olivine():
    label, weight = parse_category("hcp + olivine (High)")
    np.testing.assert_array_almost_equal(label, [0.5, 0.5, 0, 1, 0, 0, 0])
    assert weight == 1.0

def test_olivine_plus_plagioclase():
    label, weight = parse_category("olivine + plagioclase (Low)")
    np.testing.assert_array_almost_equal(label, [0.5, 0.5, 0, 0, 1, 0, 0])
    assert weight == 0.25

def test_hcp_plus_lcp():
    label, weight = parse_category("hcp + lcp (Moderate)")
    np.testing.assert_array_almost_equal(label, [0, 0, 1, 1, 0, 0, 0])
    assert weight == 0.5

def test_alteration_high():
    """Since 2026-06-10 'alteration' is a positive class, not a dropped token."""
    label, weight = parse_category("alteration (High)")
    np.testing.assert_array_almost_equal(label, [0, 0, 0, 0, 0, 0, 1])
    assert weight == 1.0

def test_alteration_plus_olivine():
    label, weight = parse_category("alteration + olivine (Low)")
    np.testing.assert_array_almost_equal(label, [0.5, 0.5, 0, 0, 0, 0, 1])
    assert weight == 0.25

def test_alteration_plus_plagioclase():
    label, weight = parse_category("alteration + plagioclase (Low)")
    np.testing.assert_array_almost_equal(label, [0, 0, 0, 0, 1, 0, 1])
    assert weight == 0.25

def test_lcp_plus_hcp_plus_olivine():
    label, weight = parse_category("hcp + lcp + olivine (Moderate)")
    assert label[2] == 1.0   # lcp
    assert label[3] == 1.0   # hcp
    assert label[0] == pytest.approx(0.5)  # olivine_t1
    assert label[1] == pytest.approx(0.5)  # olivine_t2
    assert label[6] == 0.0   # alteration

def test_unknown_category_returns_zeros():
    label, weight = parse_category("spinel (Low)")
    np.testing.assert_array_almost_equal(label, [0, 0, 0, 0, 0, 0, 0])
    assert weight == 0.25

def test_returns_numpy_array():
    label, weight = parse_category("lcp (High)")
    assert isinstance(label, np.ndarray)
    assert label.dtype == np.float32

def test_none_input_raises():
    with pytest.raises(ValueError, match="None"):
        parse_category(None)

def test_empty_string_raises():
    with pytest.raises(ValueError):
        parse_category("")

def test_whitespace_only_raises():
    with pytest.raises(ValueError):
        parse_category("   ")

def test_confidence_case_insensitive_high():
    """'(high)' lowercase should still give weight 1.0"""
    label, weight = parse_category("lcp (high)")
    assert weight == 1.0

def test_confidence_case_insensitive_moderate():
    label, weight = parse_category("lcp (MODERATE)")
    assert weight == 0.5

def test_missing_confidence_defaults_to_low():
    label, weight = parse_category("lcp")
    assert weight == 0.25

def test_get_confidence_tier_preserves_case():
    assert get_confidence_tier("lcp (High)") == "High"
    assert get_confidence_tier("lcp (Moderate)") == "Moderate"
    assert get_confidence_tier("lcp (Low)") == "Low"

def test_red_slope_ignored():
    label, weight = parse_category("red slope (Low)")
    np.testing.assert_array_almost_equal(label, [0, 0, 0, 0, 0, 0, 0])

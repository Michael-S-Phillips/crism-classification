# tests/test_plot_labels_vs_predicted.py
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_parse_category_single():
    """`"lcp (High)"` → (['lcp'], 'High')."""
    from scripts.plot_labels_vs_predicted import parse_category
    minerals, tier = parse_category('lcp (High)')
    assert minerals == ['lcp']
    assert tier == 'High'


def test_parse_category_mixed():
    """`"hcp + olivine (Moderate)"` → (['hcp', 'olivine'], 'Moderate')."""
    from scripts.plot_labels_vs_predicted import parse_category
    minerals, tier = parse_category('hcp + olivine (Moderate)')
    assert minerals == ['hcp', 'olivine']
    assert tier == 'Moderate'


def test_parse_category_three_way():
    """`"alteration + hcp + olivine (Low)"` → (['other', 'hcp', 'olivine'], 'Low')."""
    from scripts.plot_labels_vs_predicted import parse_category
    minerals, tier = parse_category('alteration + hcp + olivine (Low)')
    assert minerals == ['other', 'hcp', 'olivine']
    assert tier == 'Low'


def test_parse_category_type_olivine():
    """`"Type 2 olivine (High)"` → (['olivine'], 'High')."""
    from scripts.plot_labels_vs_predicted import parse_category
    minerals, tier = parse_category('Type 2 olivine (High)')
    assert minerals == ['olivine']
    assert tier == 'High'


def test_parse_category_other_uppercase():
    """`"Other (High)"` (capital O) → (['other'], 'High')."""
    from scripts.plot_labels_vs_predicted import parse_category
    minerals, tier = parse_category('Other (High)')
    assert minerals == ['other']
    assert tier == 'High'


def test_parse_category_red_slope():
    """`"red slope (Low)"` is treated as a single other token, not split further."""
    from scripts.plot_labels_vs_predicted import parse_category
    minerals, tier = parse_category('red slope (Low)')
    assert minerals == ['other']
    assert tier == 'Low'


def test_blend_single():
    """Single mineral returns its exact MINERAL_COLORS RGB."""
    import matplotlib.colors as mc
    from scripts.plot_labels_vs_predicted import blend_mineral_color
    from scripts.fig_style import MINERAL_COLORS
    result = blend_mineral_color(['olivine'])
    expected = mc.to_rgb(MINERAL_COLORS['olivine'])
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_blend_two():
    """Two minerals return the component-wise average of their RGB values."""
    import matplotlib.colors as mc
    from scripts.plot_labels_vs_predicted import blend_mineral_color
    from scripts.fig_style import MINERAL_COLORS
    r1, g1, b1 = mc.to_rgb(MINERAL_COLORS['hcp'])
    r2, g2, b2 = mc.to_rgb(MINERAL_COLORS['olivine'])
    result = blend_mineral_color(['hcp', 'olivine'])
    assert result[0] == pytest.approx((r1 + r2) / 2, abs=1e-6)
    assert result[1] == pytest.approx((g1 + g2) / 2, abs=1e-6)
    assert result[2] == pytest.approx((b1 + b2) / 2, abs=1e-6)

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_apply_median_filter_no_nan():
    """Median filter runs on finite float array without NaN."""
    from scripts.vectorize_tile_minerals import apply_median_filter

    arr = np.random.rand(20, 20).astype(np.float32)
    result = apply_median_filter(arr, size=3, iterations=2)
    assert result.shape == arr.shape
    assert np.isfinite(result).all(), "median filter should not introduce NaN"


def test_apply_median_filter_preserves_uniform():
    """Median filter leaves a uniform array unchanged."""
    from scripts.vectorize_tile_minerals import apply_median_filter

    arr = np.full((10, 10), 0.5, dtype=np.float32)
    result = apply_median_filter(arr, size=3, iterations=1)
    np.testing.assert_array_almost_equal(result, arr)


def test_assign_confidence_tiers():
    """assign_confidence_tiers maps Threshold float values to int tiers by rank."""
    import geopandas as gpd
    from shapely.geometry import Point
    from scripts.vectorize_tile_minerals import assign_confidence_tiers

    gdf = gpd.GeoDataFrame({
        'geometry': [Point(0, 0), Point(1, 0), Point(2, 0)],
        'Threshold': [0.28, 0.41, 0.57],
    })
    result = assign_confidence_tiers(gdf)
    assert list(result['confidence']) == [1, 2, 3]


def test_assign_confidence_tiers_missing_levels():
    """Works when some tiers have no polygons (e.g. only tiers 1 and 3)."""
    import geopandas as gpd
    from shapely.geometry import Point
    from scripts.vectorize_tile_minerals import assign_confidence_tiers

    gdf = gpd.GeoDataFrame({
        'geometry': [Point(0, 0), Point(1, 0)],
        'Threshold': [0.28, 0.57],
    })
    result = assign_confidence_tiers(gdf)
    # Only two distinct levels; mapped to 1 and 2
    assert set(result['confidence']) == {1, 2}


def test_load_thresholds_json(tmp_path):
    """load_thresholds_json parses JSON and returns thresholds dict."""
    import json
    from scripts.vectorize_tile_minerals import load_thresholds_json

    payload = {
        'thresholds': {
            'olivine': [0.28, 0.41, 0.57],
            'lcp': [0.82, 0.91, 0.96],
            'hcp': [0.04, 0.09, 0.18],
            'plagioclase': [0.03, 0.08, 0.15],
        }
    }
    f = tmp_path / 'thresh.json'
    f.write_text(json.dumps(payload))
    result = load_thresholds_json(str(f))
    assert result['olivine'] == [0.28, 0.41, 0.57]
    assert len(result) == 4

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


def test_assign_confidence_tiers_filters_zero_threshold():
    """Vectroscopy background sentinel polygons with Threshold=0 are dropped."""
    import geopandas as gpd
    from shapely.geometry import Point
    from scripts.vectorize_tile_minerals import assign_confidence_tiers

    gdf = gpd.GeoDataFrame({
        'geometry': [Point(0, 0), Point(1, 0), Point(2, 0), Point(3, 0)],
        'Threshold': [0.0, 0.28, 0.41, 0.57],
    })
    result = assign_confidence_tiers(gdf)
    assert len(result) == 3, "Threshold=0 sentinel row should be removed"
    assert list(result['confidence']) == [1, 2, 3]
    assert 0.0 not in result['Threshold'].values


def test_load_probs_npz_round_trip(tmp_path):
    """load_probs_npz correctly reconstructs probs, valid_mask, crs, and Affine from save_probs output."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from scripts.classify_tile_supervised import save_probs
    from scripts.vectorize_tile_minerals import load_probs_npz
    from rasterio.transform import Affine

    H, W = 5, 6
    probs = np.random.rand(H, W, 5).astype(np.float32)
    mask = np.ones((H, W), dtype=bool)
    mask[0, 0] = False
    # Realistic rasterio Affine: 200m pixels, top-left at (100000, -200000)
    transform_arr = np.array([200.0, 0.0, 100000.0, 0.0, -200.0, -200000.0], dtype=np.float64)
    crs_wkt = 'PROJCS["Mars_2000_Equidistant_Cylindrical",GEOGCS["GCS_Mars_2000",DATUM["D_Mars_2000",SPHEROID["Mars_2000_IAU_IAG",3396190.0,169.8944472]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Equidistant_Cylindrical"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",0.0],PARAMETER["Standard_Parallel_1",0.0],UNIT["Meter",1.0]]'

    out = tmp_path / 'rt_probs.npz'
    save_probs(str(out), probs, mask, transform_arr, crs_wkt)

    probs_out, mask_out, crs_out, transform_out = load_probs_npz(str(out))

    np.testing.assert_array_almost_equal(probs_out, probs)
    np.testing.assert_array_equal(mask_out, mask)
    assert isinstance(transform_out, Affine)
    assert transform_out.a == pytest.approx(200.0)
    assert transform_out.c == pytest.approx(100000.0)
    assert transform_out.e == pytest.approx(-200.0)
    assert crs_out.is_projected

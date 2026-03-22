import numpy as np
import json
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_npz(tmp_path, name, probs, valid_mask):
    """Helper: write a synthetic probs .npz."""
    path = tmp_path / name
    np.savez_compressed(
        str(path),
        probs=probs.astype(np.float32),
        valid_mask=valid_mask,
        transform=np.zeros(6),
        crs_wkt='',
    )
    return str(path)


def test_pool_valid_probs_single_tile(tmp_path):
    """pool_valid_probs returns correct valid-pixel probs for one tile."""
    from scripts.compute_global_thresholds import pool_valid_probs

    H, W = 4, 4
    probs = np.random.rand(H, W, 5).astype(np.float32)
    mask = np.ones((H, W), dtype=bool)
    mask[0, 0] = False  # one invalid pixel

    path = make_npz(tmp_path, 'tile.npz', probs, mask)
    result = pool_valid_probs([path])  # {0: array, ..., 4: array}

    for ci in range(5):
        expected = probs[:, :, ci][mask]
        np.testing.assert_array_almost_equal(result[ci], expected)


def test_pool_valid_probs_two_tiles(tmp_path):
    """pool_valid_probs concatenates across tiles."""
    from scripts.compute_global_thresholds import pool_valid_probs

    probs1 = np.ones((3, 3, 5), dtype=np.float32) * 0.3
    probs2 = np.ones((3, 3, 5), dtype=np.float32) * 0.7
    mask = np.ones((3, 3), dtype=bool)

    p1 = make_npz(tmp_path, 't1.npz', probs1, mask)
    p2 = make_npz(tmp_path, 't2.npz', probs2, mask)
    result = pool_valid_probs([p1, p2])

    # 9 valid pixels × 2 tiles = 18 per class
    assert len(result[0]) == 18
    np.testing.assert_almost_equal(result[0].mean(), 0.5)


def test_compute_thresholds_values(tmp_path):
    """compute_thresholds returns correct percentiles per class."""
    from scripts.compute_global_thresholds import compute_thresholds

    # Class 0: uniform [0,1], class 1: all 0.9
    pooled = {
        0: np.linspace(0, 1, 100, dtype=np.float32),
        1: np.full(100, 0.9, dtype=np.float32),
        2: np.zeros(100, dtype=np.float32),
        3: np.zeros(100, dtype=np.float32),
        4: np.zeros(100, dtype=np.float32),
    }
    CLASS_NAMES = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']
    result = compute_thresholds(pooled, CLASS_NAMES, percentiles=[33, 67, 90])

    # For uniform [0,1], 33rd pctile ≈ 0.33
    assert abs(result['olivine'][0] - 0.33) < 0.02
    assert abs(result['olivine'][1] - 0.67) < 0.02
    assert abs(result['olivine'][2] - 0.90) < 0.02
    # For all-0.9, all percentiles = 0.9
    assert result['lcp'][0] == pytest.approx(0.9, abs=0.01)
    assert result['lcp'][2] == pytest.approx(0.9, abs=0.01)


def test_write_thresholds_json(tmp_path):
    """write_thresholds_json produces valid JSON matching expected schema."""
    from scripts.compute_global_thresholds import write_thresholds_json

    thresholds = {
        'olivine': [0.28, 0.41, 0.57],
        'lcp': [0.82, 0.91, 0.96],
        'hcp': [0.04, 0.09, 0.18],
        'plagioclase': [0.03, 0.08, 0.15],
        'other': [0.35, 0.55, 0.75],
    }
    out = tmp_path / 'thresh.json'
    write_thresholds_json(
        str(out),
        thresholds=thresholds,
        tiles_used=['T0434', 'T0435'],
        percentiles=[33, 67, 90],
        morphology={'median_filter_size': 3, 'median_filter_iterations': 1,
                    'sieve_min_pixels': 9, 'majority_filter_iterations': 3,
                    'simplify_tolerance_meters': 200},
    )
    data = json.loads(out.read_text())
    assert 'generated' in data
    assert data['tiles_used'] == ['T0434', 'T0435']
    assert data['percentiles'] == [33, 67, 90]
    assert list(data['thresholds'].keys()) == ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']
    assert len(data['thresholds']['olivine']) == 3
    assert 'morphology' in data


def test_compute_otsu_thresholds_bimodal():
    """Otsu splits a bimodal distribution; tier thresholds come from the signal cluster."""
    from scripts.compute_global_thresholds import compute_otsu_thresholds

    rng = np.random.default_rng(42)
    # Class 0: 1000 noise pixels near 0.05, 200 signal pixels near 0.70
    noise = rng.normal(0.05, 0.01, 1000).astype(np.float32)
    signal = rng.normal(0.70, 0.05, 200).astype(np.float32)
    vals = np.concatenate([noise, signal])
    pooled = {0: vals, 1: vals, 2: vals, 3: vals, 4: vals}
    CLASS_NAMES = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']

    thresholds, otsu_vals = compute_otsu_thresholds(pooled, CLASS_NAMES)

    # Otsu should land somewhere between the two modes (noise ~0.05, signal ~0.70)
    assert 0.06 < otsu_vals['olivine'] < 0.60
    # All 5 tier thresholds must be above the Otsu split
    assert len(thresholds['olivine']) == 5
    for t in thresholds['olivine']:
        assert t > otsu_vals['olivine']
    # Tiers must be non-decreasing
    assert thresholds['olivine'] == sorted(thresholds['olivine'])


def test_compute_otsu_thresholds_signal_percentiles():
    """Tier thresholds are the 50th, 67th, 90th percentiles of above-Otsu pixels."""
    from scripts.compute_global_thresholds import compute_otsu_thresholds
    from skimage.filters import threshold_otsu

    rng = np.random.default_rng(7)
    noise = rng.normal(0.05, 0.01, 800).astype(np.float32)
    signal = rng.normal(0.60, 0.08, 200).astype(np.float32)
    vals = np.concatenate([noise, signal])
    pooled = {0: vals, 1: vals, 2: vals, 3: vals, 4: vals}
    CLASS_NAMES = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']

    thresholds, otsu_vals = compute_otsu_thresholds(pooled, CLASS_NAMES)

    otsu = otsu_vals['olivine']
    signal_pixels = vals[vals > otsu]
    expected = [float(np.percentile(signal_pixels, p)) for p in [50, 67, 90, 95, 99]]
    np.testing.assert_allclose(thresholds['olivine'], expected, rtol=1e-5)

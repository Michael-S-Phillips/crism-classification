import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_save_probs_output_shape(tmp_path):
    """save_probs writes (H,W,4) probs + valid_mask + transform + crs_wkt to npz."""
    from scripts.classify_tile_supervised import save_probs

    H, W = 10, 12
    probs_hw4 = np.random.rand(H, W, 4).astype(np.float32)
    valid_mask = np.ones((H, W), dtype=bool)
    valid_mask[0, 0] = False
    # transform_arr: rasterio Affine order (a,b,c,d,e,f) = (col_scale, col_shear, col_off,
    #                                                         row_shear, row_scale, row_off)
    transform_arr = np.array([200.0, 0.0, 100000.0, 0.0, -200.0, -200000.0])
    crs_wkt = 'PROJCS["Mars_2000_Equidistant_Cylindrical"]'

    out = tmp_path / 'test_probs.npz'
    save_probs(str(out), probs_hw4, valid_mask, transform_arr, crs_wkt)

    data = np.load(str(out), allow_pickle=True)
    assert data['probs'].shape == (H, W, 4)
    assert data['probs'].dtype == np.float32
    assert data['valid_mask'].shape == (H, W)
    assert data['valid_mask'].dtype == bool
    assert data['transform'].shape == (6,)
    # crs_wkt must be a non-empty string matching the input
    assert isinstance(str(data['crs_wkt']), str)
    assert len(str(data['crs_wkt'])) > 0
    assert str(data['crs_wkt']) == crs_wkt


def test_save_probs_values_preserved(tmp_path):
    """Saved probs match input values exactly."""
    from scripts.classify_tile_supervised import save_probs

    probs = np.array([[[0.1, 0.9, 0.2, 0.05]]], dtype=np.float32)  # (1,1,4)
    mask = np.array([[True]])
    t = np.zeros(6)
    out = tmp_path / 'p.npz'
    save_probs(str(out), probs, mask, t, '')
    data = np.load(str(out), allow_pickle=True)
    assert data['probs'].shape == (1, 1, 4)
    np.testing.assert_array_almost_equal(data['probs'], probs)

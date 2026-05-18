"""Tests for CRISMCachedPatchDataset + helpers."""
from __future__ import annotations

import numpy as np
import pytest

from data.cached_patch_dataset import compute_valid_centers


class TestComputeValidCenters:
    """A 2D nodata mask -> a (H, W) bool array of valid center positions."""

    def test_all_valid_returns_inner_grid_true(self):
        # 10x10 all-valid (no nodata anywhere) → every center where the
        # full 7x7 patch fits should be True.
        nodata = np.zeros((10, 10), dtype=bool)
        valid = compute_valid_centers(nodata, patch_size=7, min_valid_frac=0.8)
        assert valid.shape == (10, 10)
        # Centers in [3, 6] (inclusive) along each axis have a valid 7x7 patch.
        assert valid[3:7, 3:7].all()
        # Edge centers can't fit a 7x7 patch → must be False.
        assert not valid[:3, :].any()
        assert not valid[7:, :].any()
        assert not valid[:, :3].any()
        assert not valid[:, 7:].any()

    def test_all_nodata_returns_all_false(self):
        nodata = np.ones((10, 10), dtype=bool)
        valid = compute_valid_centers(nodata, patch_size=7, min_valid_frac=0.8)
        assert valid.shape == (10, 10)
        assert not valid.any()

    def test_threshold_boundary(self):
        # 7x7 = 49 pixels. min_valid_frac=0.8 → need ≥ 40 valid (39.2 → ≥40).
        # Build a tile where center (3, 3) has exactly 10 nodata pixels in its
        # 7x7 patch — valid frac = 39/49 ≈ 0.796 < 0.8 → must be False.
        nodata = np.zeros((10, 10), dtype=bool)
        # Set 10 nodata in the top-left of the 7x7 patch around center (3, 3)
        nodata[0, 0:10] = True            # row 0: 10 nodata
        valid = compute_valid_centers(nodata, patch_size=7, min_valid_frac=0.8)
        # Center (3, 3) sees rows 0-6: 7 nodata in row 0 → 42 valid ≥ 40 → True
        assert valid[3, 3]
        # Now set more nodata until we're below threshold
        nodata[1, 0:10] = True            # rows 0-1: 14 nodata
        valid = compute_valid_centers(nodata, patch_size=7, min_valid_frac=0.8)
        # Center (3, 3) sees rows 0-6: 14 nodata in rows 0-1 → 35 valid < 40 → False
        assert not valid[3, 3]

    def test_patch_size_must_be_odd(self):
        nodata = np.zeros((10, 10), dtype=bool)
        with pytest.raises(AssertionError):
            compute_valid_centers(nodata, patch_size=6, min_valid_frac=0.8)

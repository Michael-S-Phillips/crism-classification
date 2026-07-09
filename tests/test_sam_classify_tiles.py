"""Tests for tiered SAM classification of tiles (Component 4).

Strict TDD for the core math: synthetic 57-band cube with planted regions
matching synthetic endmembers plus a background endmember. Exercises argmin
assignment, threshold gating, valid-mask exclusion, and polygon counting.
"""
import numpy as np
import pytest
from affine import Affine
from pyproj import CRS

from scripts.label_quant.sam_classify_tiles import (
    NODATA,
    angles_deg,
    compute_valid_mask,
    assign_pixels,
    mineral_mask,
    polygonize_mask,
)

NB = 57


def _unit(band):
    v = np.zeros(NB)
    v[band] = 1.0
    return v


# --------------------------------------------------------------------------- #
# angles_deg
# --------------------------------------------------------------------------- #
def test_angles_deg_basic():
    X = np.array([_unit(0), _unit(1), 2.0 * _unit(0)])
    E = np.array([_unit(0), _unit(1)])
    A = angles_deg(X, E)  # (3, 2) degrees
    assert A.shape == (3, 2)
    assert A[0, 0] == pytest.approx(0.0, abs=1e-6)      # identical
    assert A[0, 1] == pytest.approx(90.0, abs=1e-6)     # orthogonal
    assert A[2, 0] == pytest.approx(0.0, abs=1e-6)      # scaled copy


# --------------------------------------------------------------------------- #
# valid mask: 65535 in-window and all-zero excluded
# --------------------------------------------------------------------------- #
def test_valid_mask_excludes_nodata_and_allzero():
    cube = np.ones((NB, 2, 2), dtype=float)
    cube[:, 0, 1] = 0.0            # all-zero pixel -> invalid
    cube[5, 1, 0] = NODATA         # one nodata band -> invalid
    valid = compute_valid_mask(cube)
    assert valid[0, 0]            # normal
    assert not valid[0, 1]        # all-zero
    assert not valid[1, 0]        # has 65535
    assert valid[1, 1]


# --------------------------------------------------------------------------- #
# argmin assignment + threshold gating
# --------------------------------------------------------------------------- #
def _planted_cube(rng):
    """4x4 cube: rows are 4 regions each aligned to endmember 0..3.
    Region 1 (lcp) is rotated ~2.5deg off its endmember to test gating."""
    E = np.array([_unit(0), _unit(1), _unit(2), _unit(3)])  # e0..e3
    H = W = 4
    cube = np.zeros((NB, H, W), dtype=float)
    for r in range(4):
        base = E[r].copy()
        if r == 1:
            # rotate 2.5 deg toward an orthogonal axis (band 40)
            t = np.radians(2.5)
            base = np.cos(t) * E[1] + np.sin(t) * _unit(40)
        for c in range(W):
            cube[:, r, c] = base * (1.0 + 0.0 * c)  # constant across the row
    return cube, E


def test_argmin_assignment_and_gating():
    rng = np.random.default_rng(0)
    cube, E = _planted_cube(rng)
    argmin, minang, valid = assign_pixels(cube, E)

    # every pixel in row r assigned to endmember r
    for r in range(4):
        assert np.all(argmin[r, :] == r)

    # row 1 pixels sit ~2.5 deg from their endmember
    assert np.allclose(minang[1, :], 2.5, atol=0.05)
    # rows 0,2,3 essentially 0 deg
    for r in (0, 2, 3):
        assert np.all(minang[r, :] < 1e-4)

    # gating: loose angle (3.0) admits row 1, tight (1.0) rejects it
    loose = mineral_mask(argmin, minang, valid, class_idx=1, angle=3.0)
    tight = mineral_mask(argmin, minang, valid, class_idx=1, angle=1.0)
    assert loose[1, :].all()
    assert not tight[1, :].any()
    # row-0 endmember still admitted at the tight threshold
    tight0 = mineral_mask(argmin, minang, valid, class_idx=0, angle=1.0)
    assert tight0[0, :].all()


def test_conservative_background_competition():
    # a pixel closer to a background endmember gets no mineral label even if
    # within the mineral threshold in absolute terms
    E = np.array([_unit(0), _unit(1)])  # idx0 = mineral, idx1 = background
    cube = np.zeros((NB, 1, 1), dtype=float)
    # 3 deg from background (idx1), hence 87 deg from mineral (idx0)
    t_bg = np.radians(3.0)
    cube[:, 0, 0] = np.cos(t_bg) * E[1] + np.sin(t_bg) * E[0]
    argmin, minang, valid = assign_pixels(cube, E)
    # argmin is background (idx1), so mineral(idx0) mask is empty at any angle
    m = mineral_mask(argmin, minang, valid, class_idx=0, angle=15.0)
    assert not m.any()


def test_invalid_pixels_get_no_label():
    E = np.array([_unit(0), _unit(1)])
    cube = np.zeros((NB, 1, 2), dtype=float)
    cube[:, 0, 0] = E[0]           # valid, mineral 0
    cube[:, 0, 1] = 0.0            # all-zero -> invalid
    argmin, minang, valid = assign_pixels(cube, E)
    assert argmin[0, 0] == 0
    assert argmin[0, 1] == -1
    m = mineral_mask(argmin, minang, valid, class_idx=0, angle=5.0)
    assert m[0, 0]
    assert not m[0, 1]


# --------------------------------------------------------------------------- #
# polygonization count sanity
# --------------------------------------------------------------------------- #
def test_polygonize_counts_and_min_px():
    transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)  # 1x1 px
    crs = CRS.from_epsg(4326)
    mask = np.zeros((20, 20), dtype=bool)
    valid = np.ones((20, 20), dtype=bool)
    mask[2:5, 2:5] = True     # 3x3 = 9 px block
    mask[10, 10] = True       # 1 px speckle -> dropped at min_px=4
    mask[15:17, 15:17] = True  # 2x2 = 4 px block -> kept at min_px=4

    gdf = polygonize_mask(mask, valid, transform, crs, min_px=4)
    assert len(gdf) == 2                       # 9px and 4px blocks, speckle gone
    assert set(gdf["count_px"]) == {9, 4}

    gdf_strict = polygonize_mask(mask, valid, transform, crs, min_px=9)
    assert len(gdf_strict) == 1
    assert gdf_strict.iloc[0]["count_px"] == 9

    empty = polygonize_mask(np.zeros((20, 20), dtype=bool), valid,
                            transform, crs, min_px=4)
    assert len(empty) == 0

"""Mining dust hard negatives: the criteria must be tile-relative and exclusive."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.mine_dust_hard_negatives import (
    FLOOR_TEST_TILES, MRRSU_IDX, select_dust_negatives, thin_mask,
)

CLASSES = ['olivine', 'pyx', 'plagioclase', 'bland', 'alteration', 'junk']


def _mrrsu(h, w, **overrides):
    """60-band mrrsu cube; every parameter mid-range unless overridden."""
    cube = np.full((60, h, w), 0.02, dtype=np.float32)
    for name, val in overrides.items():
        cube[MRRSU_IDX[name]] = val
    return cube


def _probs(h, w, mineral_p=0.99):
    p = np.zeros((h, w, len(CLASSES)), dtype=np.float32)
    p[:, :, CLASSES.index('pyx')] = mineral_p
    return p


def test_floor_test_tiles_are_the_eight_from_floor_test_sh():
    assert FLOOR_TEST_TILES == frozenset(
        {'t1249', 't1250', 't1321', 't1322', 't0434', 't0435', 't1086', 't1087'})


def test_a_dusty_indexless_confident_pixel_is_selected():
    h = w = 20
    # Half the tile dark+mafic, half bright+indexless, so tile percentiles split.
    cube = _mrrsu(h, w)
    cube[MRRSU_IDX['LCPINDEX2']][:, :10] = 0.06   # mafic half
    cube[MRRSU_IDX['OLINDEX3']][:, :10] = 0.06
    cube[MRRSU_IDX['HCPINDEX2']][:, :10] = 0.06
    cube[MRRSU_IDX['LCPINDEX2']][:, 10:] = 0.000  # dust half: no mafic
    cube[MRRSU_IDX['OLINDEX3']][:, 10:] = 0.000
    cube[MRRSU_IDX['HCPINDEX2']][:, 10:] = 0.000
    cube[MRRSU_IDX['RBR']][:, :10] = 3.0
    cube[MRRSU_IDX['RBR']][:, 10:] = 6.0          # dust half: red
    cube[MRRSU_IDX['R770']][:, :10] = 0.13
    cube[MRRSU_IDX['R770']][:, 10:] = 0.27        # dust half: bright
    valid = np.ones((h, w), bool)
    mask = select_dust_negatives(cube, _probs(h, w), CLASSES, valid)
    assert mask[:, 10:].mean() > 0.9, 'dusty indexless half should be mined'
    assert mask[:, :10].sum() == 0, 'mafic half must never be mined'


def test_alteration_signature_blocks_selection():
    """Without this the miner harvests real alteration and teaches the model to
    miss it -- alteration is a mineral in the 7-class vocab."""
    h = w = 20
    cube = _mrrsu(h, w)
    cube[MRRSU_IDX['LCPINDEX2']][:, 10:] = 0.0
    cube[MRRSU_IDX['OLINDEX3']][:, 10:] = 0.0
    cube[MRRSU_IDX['HCPINDEX2']][:, 10:] = 0.0
    cube[MRRSU_IDX['RBR']][:, 10:] = 6.0
    cube[MRRSU_IDX['R770']][:, 10:] = 0.27
    cube[MRRSU_IDX['D2300']][:, 10:] = 0.09       # strong alteration
    cube[MRRSU_IDX['BD1900_2']][:, 10:] = 0.09
    cube[MRRSU_IDX['BD2210_2']][:, 10:] = 0.09
    valid = np.ones((h, w), bool)
    mask = select_dust_negatives(cube, _probs(h, w), CLASSES, valid)
    assert mask.sum() == 0


def test_pixels_no_model_is_confident_about_are_not_hard_negatives():
    """Easy negatives carry no gradient; only pixels that fool a model qualify."""
    h = w = 20
    cube = _mrrsu(h, w)
    cube[MRRSU_IDX['LCPINDEX2']][:, 10:] = 0.0
    cube[MRRSU_IDX['OLINDEX3']][:, 10:] = 0.0
    cube[MRRSU_IDX['HCPINDEX2']][:, 10:] = 0.0
    cube[MRRSU_IDX['RBR']][:, 10:] = 6.0
    cube[MRRSU_IDX['R770']][:, 10:] = 0.27
    valid = np.ones((h, w), bool)
    mask = select_dust_negatives(cube, _probs(h, w, mineral_p=0.10), CLASSES, valid)
    assert mask.sum() == 0


def test_invalid_pixels_are_never_selected():
    h = w = 20
    cube = _mrrsu(h, w)
    cube[MRRSU_IDX['LCPINDEX2']][:] = 0.0
    cube[MRRSU_IDX['OLINDEX3']][:] = 0.0
    cube[MRRSU_IDX['HCPINDEX2']][:] = 0.0
    cube[MRRSU_IDX['RBR']][:] = 6.0
    cube[MRRSU_IDX['R770']][:] = 0.27
    valid = np.ones((h, w), bool)
    valid[5, 5] = False
    mask = select_dust_negatives(cube, _probs(h, w), CLASSES, valid)
    assert not mask[5, 5]


def test_thinning_enforces_separation_and_cap():
    """One dust mantle must not supply the whole negative set."""
    mask = np.ones((40, 40), bool)
    out = thin_mask(mask, min_sep=4, max_per_tile=1000, seed=0)
    ys, xs = np.nonzero(out)
    assert out.sum() < mask.sum()
    d2 = (ys[:, None] - ys[None, :]) ** 2 + (xs[:, None] - xs[None, :]) ** 2
    np.fill_diagonal(d2, 10 ** 9)
    assert d2.min() >= 16, 'two kept pixels closer than min_sep'


def test_thinning_respects_max_per_tile():
    mask = np.ones((60, 60), bool)
    out = thin_mask(mask, min_sep=1, max_per_tile=25, seed=0)
    assert out.sum() == 25


def test_thinning_is_deterministic():
    mask = np.random.default_rng(1).random((30, 30)) > 0.5
    a = thin_mask(mask, min_sep=2, max_per_tile=50, seed=7)
    b = thin_mask(mask, min_sep=2, max_per_tile=50, seed=7)
    assert np.array_equal(a, b)


def test_uniformly_mafic_tile_is_never_mined():
    """Regression: a flat (zero-variance) mafic/dusty band used to tie its own
    percentile under inclusive <=/>=, so an unambiguously mafic tile selected
    every pixel instead of none. A degenerate band carries no percentile
    signal at all and must fall back to an absolute check, not an
    unconditional pass."""
    h = w = 20
    cube = _mrrsu(h, w)
    cube[MRRSU_IDX['OLINDEX3']][:] = 0.15     # flat, unambiguously mafic
    cube[MRRSU_IDX['LCPINDEX2']][:] = 0.15
    cube[MRRSU_IDX['HCPINDEX2']][:] = 0.15
    cube[MRRSU_IDX['RBR']][:] = 6.0           # flat, dusty
    cube[MRRSU_IDX['R770']][:] = 0.27
    valid = np.ones((h, w), bool)
    mask = select_dust_negatives(cube, _probs(h, w), CLASSES, valid)
    assert mask.sum() == 0, 'a uniformly mafic tile must never be mined'


def test_uniformly_altered_tile_is_never_mined():
    """Regression: the zero-variance pass-through in the alteration veto
    tested variance only, never magnitude, so a tile flat at a strongly
    altered value slipped through as a "dust" hard negative."""
    h = w = 20
    cube = _mrrsu(h, w)
    cube[MRRSU_IDX['OLINDEX3']][:] = 0.0      # no mafic signature
    cube[MRRSU_IDX['LCPINDEX2']][:] = 0.0
    cube[MRRSU_IDX['HCPINDEX2']][:] = 0.0
    cube[MRRSU_IDX['RBR']][:] = 6.0           # dusty
    cube[MRRSU_IDX['R770']][:] = 0.27
    cube[MRRSU_IDX['D2300']][:] = 0.5         # flat, uniformly strong alteration
    cube[MRRSU_IDX['BD1900_2']][:] = 0.5
    cube[MRRSU_IDX['BD2210_2']][:] = 0.5
    valid = np.ones((h, w), bool)
    mask = select_dust_negatives(cube, _probs(h, w), CLASSES, valid)
    assert mask.sum() == 0, 'a uniformly altered tile must never be mined'

"""Task 8: inference CR path parity.

scripts/classify_tile_supervised.py --continuum_removed continuum-removes each
pixel's patch identically to training (data.continuum_removal.cr_patch) and, with
--brightness_aux, feeds the center-pixel brightness scalar to the aux model.

Parity: the inference CR transform of a small synthetic tile must equal cr_patch
applied to the same extracted patches (same clip → same patches → same CR).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.continuum_removal import cr_patch


def _synthetic_tile(H=16, W=18, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 0.5, size=(H, W, 59)).astype(np.float32)


def test_cr_transform_batch_matches_cr_patch():
    """Inference CR (cr_transform_batch) equals cr_patch per extracted patch, and
    the aux brightness is the center-pixel brightness scalar."""
    from scripts.classify_tile_supervised import (
        extract_patches_batched, cr_transform_batch, PATCH_SIZE, PAD)

    tile = _synthetic_tile(H=16, W=18, seed=3)
    for patches, idx in extract_patches_batched(tile, batch_size=64):
        cr_batch, bright = cr_transform_batch(patches)
        assert cr_batch.shape == patches.shape
        assert bright.shape == (len(patches), 1)
        assert np.all(np.isfinite(cr_batch))
        assert cr_batch.max() <= 1.0001
        for j in range(len(patches)):
            cr_ref, b_ref = cr_patch(patches[j])
            np.testing.assert_allclose(cr_batch[j], cr_ref, rtol=0, atol=1e-6)
            np.testing.assert_allclose(bright[j, 0], b_ref[PAD, PAD], rtol=0, atol=1e-6)


def test_run_supervised_cr_brightness_aux_shape():
    """run_supervised with continuum_removed + brightness_aux drives the aux model
    (aux_dim=1) and returns (H*W, N_CLASSES) probabilities in [0,1]."""
    import scripts.classify_tile_supervised as cls
    from models.spatial_spectral_classifier_aux import SpatialSpectralClassifierAux

    tile = _synthetic_tile(H=12, W=14, seed=1)
    H, W, _ = tile.shape
    model = SpatialSpectralClassifierAux(
        n_bands=59, patch_size=cls.PATCH_SIZE, n_classes=cls.N_CLASSES,
        embed_dim=32, n_heads=2, n_layers=2, aux_dim=1)
    model.eval()

    probs = cls.run_supervised(tile, model, torch.device('cpu'), batch_size=64,
                               continuum_removed=True, brightness_aux=True)
    assert probs.shape == (H * W, cls.N_CLASSES)
    assert np.all(np.isfinite(probs))
    assert probs.min() >= 0.0 and probs.max() <= 1.0


def test_run_supervised_cr_plain_shape():
    """CR without brightness aux drives the plain classifier on CR patches."""
    import scripts.classify_tile_supervised as cls
    from models.spatial_spectral_transformer import SpatialSpectralClassifier

    tile = _synthetic_tile(H=10, W=10, seed=2)
    H, W, _ = tile.shape
    model = SpatialSpectralClassifier(
        n_bands=59, patch_size=cls.PATCH_SIZE, n_classes=cls.N_CLASSES,
        embed_dim=32, n_heads=2, n_layers=2)
    model.eval()
    probs = cls.run_supervised(tile, model, torch.device('cpu'), batch_size=64,
                               continuum_removed=True)
    assert probs.shape == (H * W, cls.N_CLASSES)
    assert np.all(np.isfinite(probs))

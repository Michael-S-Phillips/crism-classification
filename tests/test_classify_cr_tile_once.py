"""CR classify optimization: continuum-remove the whole tile once instead of
re-CR'ing every overlapping patch. This must be bit-for-bit identical to the
old per-patch path (cr_transform_batch), just ~49x cheaper.
"""
import numpy as np

from scripts.classify_tile_supervised import (
    PAD, extract_patches_batched, cr_transform_batch)
from data.continuum_removal import continuum_removed as cr_cube, brightness_scalar


def test_tile_once_cr_matches_per_patch(seed=0):
    rng = np.random.default_rng(seed)
    H, W = 11, 9                       # small tile; edges exercise the 0-padding
    tile = (0.05 + 0.3 * rng.random((H, W, 59))).astype(np.float32)

    # New path: CR the padded tile once, slice CR patches from it.
    padded = np.pad(tile, ((PAD, PAD), (PAD, PAD), (0, 0)), mode='constant')
    cr_padded = cr_cube(padded).astype(np.float32)
    bright_flat = brightness_scalar(tile).reshape(-1).astype(np.float32)
    new_patches, new_idx = next(extract_patches_batched(
        cr_padded, batch_size=H * W, already_padded=True))
    new_bright = bright_flat[new_idx].reshape(-1, 1)

    # Old path: extract raw patches, CR each patch, take center-pixel brightness.
    old_patches_raw, old_idx = next(extract_patches_batched(tile, batch_size=H * W))
    old_patches, old_bright = cr_transform_batch(old_patches_raw)

    # Same pixel ordering, same CR patch values (incl. zero-padded borders),
    # same center-pixel brightness.
    assert np.array_equal(new_idx, old_idx)
    assert new_patches.shape == old_patches.shape == (H * W, 2 * PAD + 1, 2 * PAD + 1, 59)
    assert np.allclose(new_patches, old_patches, atol=1e-6, equal_nan=True)
    assert np.allclose(new_bright, old_bright, atol=1e-6)


def test_border_pixels_cr_identically():
    """The top-left pixel's patch is mostly zero-pad — the path that pads-then-CRs
    must reproduce exactly how cr_transform_batch CRs those pad pixels."""
    rng = np.random.default_rng(1)
    tile = (0.05 + 0.3 * rng.random((7, 7, 59))).astype(np.float32)
    padded = np.pad(tile, ((PAD, PAD), (PAD, PAD), (0, 0)), mode='constant')
    cr_padded = cr_cube(padded).astype(np.float32)
    new_patches, _ = next(extract_patches_batched(cr_padded, batch_size=49, already_padded=True))
    old_raw, _ = next(extract_patches_batched(tile, batch_size=49))
    old_patches, _ = cr_transform_batch(old_raw)
    # pixel 0 == top-left corner, whose 7x7 patch has PAD rows/cols of zero pad
    assert np.allclose(new_patches[0], old_patches[0], atol=1e-6, equal_nan=True)

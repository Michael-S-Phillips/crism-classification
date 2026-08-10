"""Tests for scripts/build_global_patch_cache.py."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest
import rasterio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


def _make_fake_tile(path_prefix: str, H: int = 50, W: int = 50, n_bands: int = 59,
                    nodata_value: float = 65535, nodata_frac: float = 0.0,
                    seed: int = 0):
    """Write a minimal ENVI/.img+.hdr pair we can read with rasterio."""
    rng = np.random.default_rng(seed)
    data = rng.uniform(0.0, 0.3, size=(n_bands, H, W)).astype(np.float32)
    if nodata_frac > 0:
        nodata_mask = rng.uniform(0, 1, size=(H, W)) < nodata_frac
        data[:, nodata_mask] = nodata_value

    img_path = path_prefix + '.img'
    hdr_path = path_prefix + '.hdr'

    profile = {
        'driver': 'ENVI',
        'dtype': 'float32',
        'count': n_bands,
        'height': H,
        'width': W,
        'nodata': nodata_value,
        'interleave': 'bsq',
    }
    with rasterio.open(img_path, 'w', **profile) as dst:
        for b in range(n_bands):
            dst.write(data[b], b + 1)
    return hdr_path, img_path


def test_extract_patches_from_tile_returns_correct_shape(tmp_path):
    from build_global_patch_cache import extract_patches_from_tile

    hdr, _img = _make_fake_tile(str(tmp_path / 'tile_a'), H=50, W=50)
    patches, n_skipped_short = extract_patches_from_tile(
        hdr_path=hdr,
        n_target=30,
        patch_size=7,
        min_valid_frac=0.8,
        clip_max=0.5,
        nodata_value=65535,
        seed=42,
    )
    assert patches.shape == (30, 7, 7, 59)
    assert patches.dtype == np.float32
    assert n_skipped_short == 0


def test_extract_patches_clipped_to_clip_max(tmp_path):
    from build_global_patch_cache import extract_patches_from_tile

    hdr, _img = _make_fake_tile(str(tmp_path / 'tile_a'), H=50, W=50)
    patches, _ = extract_patches_from_tile(
        hdr_path=hdr, n_target=10, patch_size=7,
        min_valid_frac=0.8, clip_max=0.5, nodata_value=65535, seed=0,
    )
    assert (patches >= 0).all()
    assert (patches <= 0.5).all()


def test_extract_skips_short_when_fewer_valid_centers(tmp_path):
    """A tile with nearly all nodata returns fewer than n_target patches."""
    from build_global_patch_cache import extract_patches_from_tile

    # 80% nodata → most centers fail min_valid_frac=0.8
    hdr, _img = _make_fake_tile(str(tmp_path / 'tile_b'), H=30, W=30, nodata_frac=0.8)
    patches, n_skipped_short = extract_patches_from_tile(
        hdr_path=hdr, n_target=100, patch_size=7,
        min_valid_frac=0.8, clip_max=0.5, nodata_value=65535, seed=0,
    )
    # Should return fewer patches than requested
    assert patches.shape[0] < 100
    assert n_skipped_short == 100 - patches.shape[0]


import subprocess


def _make_synthetic_data_root(tmp_path, n_tiles: int = 3, H: int = 50, W: int = 50):
    """Create n_tiles fake mrral tiles in mc<NN>/ subdirectories."""
    root = tmp_path / 'fake_mrdr'
    root.mkdir()
    mc = root / 'mc02'
    mc.mkdir()
    hdrs = []
    for i in range(n_tiles):
        prefix = mc / f't{i:04d}_mrral_00n000_0327_4'
        hdr, _img = _make_fake_tile(str(prefix), H=H, W=W, seed=i)
        hdrs.append(hdr)
    return str(root), hdrs


def test_main_builds_a_complete_cache_end_to_end(tmp_path):
    """Run the builder against a 3-tile synthetic data root, confirm shards
    and shard_index.json are written correctly."""
    import glob as _glob
    data_root, _hdrs = _make_synthetic_data_root(tmp_path, n_tiles=3, H=60, W=60)
    output = tmp_path / 'patch_cache'

    # Invoke the builder as a subprocess so we exercise the CLI entry point.
    # Small targets: 30 patches/tile × 3 tiles = 90 total, shards of 40 → 3 shards.
    result = subprocess.run(
        [
            sys.executable, '-u',
            os.path.join(os.path.dirname(__file__), '..', 'scripts', 'build_global_patch_cache.py'),
            '--output', str(output),
            '--data_root', data_root,
            '--workers', '2',
            '--seed', '42',
            '--patches_per_tile_target', '30',
            '--patches_per_shard', '40',
        ],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"builder failed: {result.stderr}"

    # Expect 3 shards: 40 + 40 + 10
    shard_files = sorted(_glob.glob(str(output / 'global_patches_*.npy')))
    assert len(shard_files) == 3, f"expected 3 shards, got {shard_files}"

    sizes = [np.load(f, mmap_mode='r').shape[0] for f in shard_files]
    assert sum(sizes) == 90
    assert sizes[0] == 40
    assert sizes[1] == 40
    assert sizes[2] == 10

    # Shard index well-formed
    index_path = output / 'shard_index.json'
    assert index_path.exists()
    with open(index_path) as f:
        idx = json.load(f)
    assert idx['n_shards'] == 3
    assert idx['patches_per_shard'] == 40
    assert idx['patch_size'] == 7
    assert idx['n_bands'] == 59
    assert idx['min_valid_frac'] == 0.8
    assert idx['clip_max'] == 0.5
    assert idx['seed'] == 42
    assert idx['tiles_used'] == 3
    assert isinstance(idx['shards'], list) and len(idx['shards']) == 3


def test_dual_mode_emits_118_channels(tmp_path):
    """extract_patches_from_tile(dual=True) must return (n,7,7,118).

    This calls the BUILDER, not dual_continuum -- Task 2 already covers the
    transform. Writes a small ENVI tile with rasterio, the same pattern
    tests/test_dataset_cr.py uses, so the real read path is exercised.

    NOTE on the brief's version of this test: it passed the .img path as the
    first positional arg (which the function calls `hdr_path` and treats
    with `hdr_path.replace('.hdr', '.img')`). That happens to work only by
    accident -- a string with no '.hdr' substring passes through
    .replace('.hdr', '.img') unchanged, so the .img path survives as-is. It
    would silently break for any tile whose directory or filename contains
    the literal substring '.hdr'. Passing the real .hdr path (as documented
    in the function's own docstring/signature) is what's actually correct,
    so that's what this version does. Also unpacks explicitly per the
    arity documented in extract_patches_from_tile's docstring, rather than
    the brief's `out[0] if isinstance(out, tuple) else out`, which is always
    a tuple here and would swallow a wrong arity silently.
    """
    import numpy as np
    import rasterio
    from build_global_patch_cache import extract_patches_from_tile

    H = W = 40
    n_bands = 59
    rng = np.random.default_rng(0)
    data = rng.uniform(0.05, 0.35, size=(n_bands, H, W)).astype(np.float32)
    prefix = str(tmp_path / 't9999_mrral_00n000_0327_4')
    img_path = prefix + '.img'
    hdr_path = prefix + '.hdr'
    with rasterio.open(img_path, 'w', driver='ENVI', dtype='float32',
                       count=n_bands, height=H, width=W, interleave='bsq') as dst:
        for b in range(n_bands):
            dst.write(data[b], b + 1)
    assert os.path.exists(hdr_path), 'rasterio should have written a sibling .hdr'

    patches, brightness, n_skipped_short = extract_patches_from_tile(
        hdr_path, n_target=12, patch_size=7, seed=0,
        continuum_removed=True, dual=True)
    assert patches.ndim == 4 and patches.shape[1:3] == (7, 7)
    assert patches.shape[-1] == 118, f'expected 118 channels, got {patches.shape}'
    assert np.isfinite(patches).all()
    assert brightness.shape == (patches.shape[0], 7, 7)


def test_non_dual_still_emits_59(tmp_path):
    """The existing hull-only path must be untouched."""
    import numpy as np
    import rasterio
    from build_global_patch_cache import extract_patches_from_tile

    H = W = 40
    rng = np.random.default_rng(1)
    data = rng.uniform(0.05, 0.35, size=(59, H, W)).astype(np.float32)
    prefix = str(tmp_path / 't9998_mrral_00n000_0327_4')
    img_path = prefix + '.img'
    hdr_path = prefix + '.hdr'
    with rasterio.open(img_path, 'w', driver='ENVI', dtype='float32', count=59,
                       height=H, width=W, interleave='bsq') as dst:
        for b in range(59):
            dst.write(data[b], b + 1)
    patches, brightness, n_skipped_short = extract_patches_from_tile(
        hdr_path, n_target=8, patch_size=7, seed=0,
        continuum_removed=True)
    assert patches.shape[-1] == 59
    assert brightness.shape == (patches.shape[0], 7, 7)


def test_dual_requires_continuum_removed_at_cli(tmp_path):
    """--dual without --continuum_removed must be rejected before any work
    happens, not fail deep inside a worker on some tile.

    NOTE: a first draft of this test asserted `'--dual' in stderr and
    'continuum_removed' in stderr`, which passes *today*, before --dual even
    exists as a recognized flag -- argparse's own "unrecognized arguments:
    --dual" error contains '--dual', and its auto-generated usage line lists
    every other flag including the pre-existing '--continuum_removed', so
    both substrings are present for reasons that have nothing to do with the
    guard this test is supposed to exercise. It could not fail for the
    claimed reason. Asserting the exact `parser.error()` message the brief
    specifies avoids that: this string cannot appear unless --dual is a
    recognized argument AND the guard fired.
    """
    import subprocess
    result = subprocess.run(
        [
            sys.executable, '-u',
            os.path.join(os.path.dirname(__file__), '..', 'scripts',
                         'build_global_patch_cache.py'),
            '--output', str(tmp_path / 'out'),
            '--data_root', str(tmp_path),
            '--dual',
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert '--dual requires --continuum_removed' in result.stderr, result.stderr


def test_non_dual_cr_output_is_byte_identical_to_pre_dual_baseline(tmp_path):
    """Regression guard for the Task 4 constraint that adding --dual must not
    change a single byte of the existing hull-only CR path.

    The expected hash below was captured by running
    extract_patches_from_tile() with these exact arguments against
    tools/build_global_patch_cache.py BEFORE the --dual branch was added
    (i.e. against the code as committed at the start of Task 4). Comparing a
    sha256 of the raw output bytes (rather than just shape/dtype, which the
    other tests already cover) means any change to the non-dual numeric
    path -- CR formula, clipping order, RNG draw sequence, dtype -- flips
    this test, not just a shape change.
    """
    import hashlib
    import numpy as np
    import rasterio
    from build_global_patch_cache import extract_patches_from_tile

    H = W = 40
    rng = np.random.default_rng(7)
    data = rng.uniform(0.05, 0.35, size=(59, H, W)).astype(np.float32)
    prefix = str(tmp_path / 'tile_golden')
    img_path = prefix + '.img'
    hdr_path = prefix + '.hdr'
    with rasterio.open(img_path, 'w', driver='ENVI', dtype='float32', count=59,
                       height=H, width=W, interleave='bsq') as dst:
        for b in range(59):
            dst.write(data[b], b + 1)

    patches, brightness, n_skipped_short = extract_patches_from_tile(
        hdr_path=hdr_path, n_target=15, patch_size=7, min_valid_frac=0.8,
        clip_max=0.5, nodata_value=65535, seed=123, continuum_removed=True)

    assert patches.shape == (15, 7, 7, 59)
    assert n_skipped_short == 0
    EXPECTED_PATCHES_SHA256 = (
        '8dd7d27e1e5cee9af3b8b5daaed07cc5007194ad68f3ea505f71640d388dd075')
    EXPECTED_BRIGHTNESS_SHA256 = (
        'b861bdbb0d913db35619bbe28146eeb570a0c320a1f020ae336604cc83513226')
    assert hashlib.sha256(patches.tobytes()).hexdigest() == EXPECTED_PATCHES_SHA256
    assert hashlib.sha256(brightness.tobytes()).hexdigest() == EXPECTED_BRIGHTNESS_SHA256

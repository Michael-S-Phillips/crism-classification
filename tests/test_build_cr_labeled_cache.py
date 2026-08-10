"""Test the raw→CR labeled-cache converter matches data.continuum_removal."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.continuum_removal import continuum_removed, brightness_scalar  # noqa: E402
from scripts.build_cr_labeled_cache import convert_split  # noqa: E402


def test_convert_split_matches_module(tmp_path):
    raw_dir = tmp_path / 'raw'
    out_dir = tmp_path / 'cr'
    raw_dir.mkdir()
    rng = np.random.default_rng(0)
    n, P = 37, 7
    patches = (0.05 + 0.3 * rng.random((n, P, P, 59))).astype(np.float32)
    # Write the RAW cache the way cache_mrral_patches.py does: a headerless
    # memmap (NOT np.save — that would add a .npy header the real reader/converter
    # never expect). This is what makes the test reflect production.
    raw_path = raw_dir / f'mrral_train_patches_p{P}.npy'
    mm = np.memmap(raw_path, dtype='float32', mode='w+', shape=(n, P, P, 59))
    mm[:] = patches
    mm.flush(); del mm

    count = convert_split(str(raw_dir), str(out_dir), 'train', P, chunk=8)
    assert count == n

    cr_path = out_dir / f'mrral_train_patches_p{P}.npy'
    # CR patches must be a RAW headerless memmap of exactly n*P*P*59*4 bytes
    # (the fine-tune reads them via np.memmap with a byte-count guard).
    assert os.path.getsize(cr_path) == n * P * P * 59 * 4
    cr = np.memmap(cr_path, dtype='float32', mode='r', shape=(n, P, P, 59))
    # Brightness sidecar is a real .npy (fine-tune reads it via np.load).
    br = np.load(out_dir / f'mrral_train_patches_p{P}_brightness.npy')
    assert br.shape == (n, P, P)
    for i in (0, 5, 19, n - 1):
        assert np.allclose(cr[i], continuum_removed(patches[i]), atol=1e-5)
        assert np.allclose(br[i], brightness_scalar(patches[i]), atol=1e-5)


def test_missing_split_is_skipped(tmp_path):
    raw_dir = tmp_path / 'raw'
    raw_dir.mkdir()
    count = convert_split(str(raw_dir), str(tmp_path / 'cr'), 'val', 7, chunk=8)
    assert count == 0


def test_dual_writes_118_channel_cache(tmp_path):
    """The byte-exact size guard in CRISMSpectralPatchDataset keys off channel
    count, so a dual cache must be written at 118 or it will be rejected."""
    import numpy as np
    from scripts.build_cr_labeled_cache import convert_split

    n, P = 24, 7
    raw_dir = tmp_path / 'raw'; raw_dir.mkdir()
    fp = np.memmap(str(raw_dir / f'mrral_train_patches_p{P}.npy'),
                   dtype='float32', mode='w+', shape=(n, P, P, 59))
    fp[:] = np.random.default_rng(0).uniform(0.05, 0.4, (n, P, P, 59))
    fp.flush(); del fp

    out_dir = tmp_path / 'cr'
    got = convert_split(str(raw_dir), str(out_dir), 'train', P, chunk=8,
                        jobs=1, dual=True)
    assert got == n

    cr = np.memmap(str(out_dir / f'mrral_train_patches_p{P}.npy'),
                   dtype='float32', mode='r', shape=(n, P, P, 118))
    assert np.isfinite(cr).all()
    br = np.load(out_dir / f'mrral_train_patches_p{P}_brightness.npy')
    assert br.shape == (n, P, P), 'brightness sidecar must still be written'


def test_dual_matches_dual_continuum_module(tmp_path):
    """The brief's shape-only test above would pass even if the dual writer
    called continuum_removed() twice, wrote garbage into the linear half, or
    got the hull/linear channel order backwards -- as long as it wrote 118
    finite floats. Assert the actual values match data.continuum_removal's
    dual_continuum() on a per-patch basis, including the 0-58 hull / 59-117
    linear channel-order contract.
    """
    from data.continuum_removal import dual_continuum
    from scripts.build_cr_labeled_cache import convert_split

    n, P = 13, 7
    raw_dir = tmp_path / 'raw'; raw_dir.mkdir()
    rng = np.random.default_rng(1)
    patches = (0.05 + 0.3 * rng.random((n, P, P, 59))).astype(np.float32)
    mm = np.memmap(str(raw_dir / f'mrral_train_patches_p{P}.npy'),
                   dtype='float32', mode='w+', shape=(n, P, P, 59))
    mm[:] = patches
    mm.flush(); del mm

    out_dir = tmp_path / 'cr'
    got = convert_split(str(raw_dir), str(out_dir), 'train', P, chunk=5,
                        jobs=1, dual=True)
    assert got == n

    cr = np.memmap(str(out_dir / f'mrral_train_patches_p{P}.npy'),
                   dtype='float32', mode='r', shape=(n, P, P, 118))
    for i in (0, 5, n - 1):
        expected = dual_continuum(patches[i])
        assert np.allclose(cr[i], expected, atol=1e-5)
        # channel-order contract: hull half must differ from the linear half
        # for real, non-degenerate spectra (catches a swapped concatenation).
        assert not np.allclose(cr[i, ..., :59], cr[i, ..., 59:], atol=1e-3)


def test_dual_parallel_matches_serial_at_118_channels(tmp_path):
    """The 59-band parallel path is documented to be byte-identical to serial
    (see test_build_cr_cache_parallel.py); that guarantee must independently
    hold for the 118-channel --dual path, where the worker init function has
    to open the output memmap at the right channel count or it silently
    writes at the wrong strides instead of raising.
    """
    n, P = 130, 7
    raw_dir = tmp_path / 'raw'; raw_dir.mkdir()
    rng = np.random.default_rng(2)
    raw = (0.05 + 0.3 * rng.random((n, P, P, 59))).astype(np.float32)
    mm = np.memmap(str(raw_dir / f'mrral_train_patches_p{P}.npy'),
                   dtype='float32', mode='w+', shape=(n, P, P, 59))
    mm[:] = raw
    mm.flush(); del mm

    ser_dir, par_dir = tmp_path / 'ser', tmp_path / 'par'
    n_ser = convert_split(str(raw_dir), str(ser_dir), 'train', P, chunk=32,
                          jobs=1, dual=True)
    n_par = convert_split(str(raw_dir), str(par_dir), 'train', P, chunk=32,
                          jobs=3, dual=True)
    assert n_ser == n_par == n

    cr_s = np.asarray(np.memmap(ser_dir / f'mrral_train_patches_p{P}.npy',
                                dtype='float32', mode='r', shape=(n, P, P, 118)))
    cr_p = np.asarray(np.memmap(par_dir / f'mrral_train_patches_p{P}.npy',
                                dtype='float32', mode='r', shape=(n, P, P, 118)))
    assert cr_s.shape == cr_p.shape == (n, P, P, 118)
    assert np.array_equal(cr_s, cr_p), '118-channel parallel output differs from serial'

    br_s = np.load(ser_dir / f'mrral_train_patches_p{P}_brightness.npy')
    br_p = np.load(par_dir / f'mrral_train_patches_p{P}_brightness.npy')
    assert br_s.shape == br_p.shape == (n, P, P)
    assert np.array_equal(br_s, br_p), '118-channel brightness differs from serial'

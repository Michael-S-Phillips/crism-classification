"""The parallel CR-cache build must be byte-identical to the serial path —
it only spreads the same per-chunk continuum_removed work across workers.
"""
import numpy as np

from scripts.build_cr_labeled_cache import convert_split

P, N = 7, 130  # 130 patches @ chunk=32 -> 5 chunks (exercises multi-worker disjoint writes)


def _write_raw(raw_dir):
    rng = np.random.default_rng(0)
    raw = (0.05 + 0.3 * rng.random((N, P, P, 59))).astype(np.float32)
    mm = np.memmap(raw_dir / 'mrral_train_patches_p7.npy', dtype='float32',
                   mode='w+', shape=(N, P, P, 59))
    mm[:] = raw
    mm.flush()
    del mm


def test_parallel_matches_serial(tmp_path):
    raw_dir = tmp_path / 'raw'; raw_dir.mkdir()
    _write_raw(raw_dir)
    ser, par = tmp_path / 'ser', tmp_path / 'par'
    convert_split(str(raw_dir), str(ser), 'train', P, chunk=32, jobs=1)
    convert_split(str(raw_dir), str(par), 'train', P, chunk=32, jobs=3)

    cr_s = np.asarray(np.memmap(ser / 'mrral_train_patches_p7.npy', dtype='float32',
                                mode='r', shape=(N, P, P, 59)))
    cr_p = np.asarray(np.memmap(par / 'mrral_train_patches_p7.npy', dtype='float32',
                                mode='r', shape=(N, P, P, 59)))
    assert np.array_equal(cr_s, cr_p), 'parallel CR patches differ from serial'
    br_s = np.load(ser / 'mrral_train_patches_p7_brightness.npy')
    br_p = np.load(par / 'mrral_train_patches_p7_brightness.npy')
    assert np.array_equal(br_s, br_p), 'parallel brightness differs from serial'
    # sanity: it really is a CR cache (bounded by 1.0 on good bands)
    assert np.nanmax(cr_s) <= 1.0001

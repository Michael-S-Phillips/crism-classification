"""Synth (MTRDR plagioclase) patch caches must match the run's representation.

THE BUG. SyntheticPatchDataset serves its .npy VERBATIM -- it applies no
transform, only a shape assertion. Both MTRDR plag caches on disk are RAW
reflectance (p50 0.194 and 0.189 over the full arrays), while hull-CR data is
bounded [0, 1] and centres at 0.934. scripts/hpc_finetune_handcore.slurm runs
`train.py --continuum_removed --cache_is_cr` AND passes those raw caches via
--synth_train_cache / --synth_val_cache, so ft_7cls_handcore_level trained
plagioclase -- and only plagioclase -- at ~4x the level of every other class.
That is trivially separable in validation and over-fires at inference, where the
whole tile is CR.

Three parts are tested here:
  1. scripts/convert_synth_cache_representation.py  (raw -> hull | dual)
  2. SyntheticPatchDataset's expected-representation guard, wired from
     training/train_torch.py, so a mismatched cache FAILS instead of training
  3. scripts/hpc_finetune_dualcr.slurm's dual plag cache build + preflight

Every test in this file was mutation-verified: the source was broken in the exact
way the test claims to catch, the test was observed to fail, and the source was
restored from a cp backup. See the task report.
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.continuum_removal import (CR_SCALES, RAW_LEVEL_MAX,  # noqa: E402
                                    WAVELENGTHS_59, continuum_removed,
                                    detect_representation,
                                    linear_continuum_removed, sample_level)
from scripts.convert_synth_cache_representation import (  # noqa: E402
    CLIP_MAX, NODATA, convert, sanitize)

CONVERTER = os.path.join(ROOT, 'scripts', 'convert_synth_cache_representation.py')
DUALCR_SLURM = os.path.join(ROOT, 'scripts', 'hpc_finetune_dualcr.slurm')

# 12 distinct GOOD bands (outside the 1 um detector-overlap exclusion window),
# used as per-row fingerprints: row i gets an absorption centred on bands[i], and
# hull-CR's deepest band recovers i. A reorder or a dropped row is then visible.
FINGERPRINT_BANDS = list(range(22, 56, 3))


# ── fixtures ────────────────────────────────────────────────────────────────

def _raw_patches(bands=None) -> np.ndarray:
    """(N, 7, 7, 59) raw-reflectance patches; row i carries absorption bands[i].

    Levels (0.16-0.26) and slope mirror the real MTRDR plag caches. Noise-free on
    purpose: the fingerprint must be exact so a row-order test cannot pass by
    luck.
    """
    bands = FINGERPRINT_BANDS if bands is None else bands
    wl = WAVELENGTHS_59
    base = 0.20 + 0.06 * (wl - wl.min()) / (wl.max() - wl.min())
    specs = [base - 0.04 * np.exp(-0.5 * ((wl - wl[b]) / 60.0) ** 2) for b in bands]
    arr = np.stack(specs).astype(np.float32)[:, None, None, :]
    return np.ascontiguousarray(np.repeat(np.repeat(arr, 7, axis=1), 7, axis=2))


def _save(path, arr) -> str:
    np.save(str(path), np.asarray(arr, dtype=np.float32))
    p = str(path)
    return p if p.endswith('.npy') else p + '.npy'


def _rows_parquet(path, n, splits=None) -> str:
    """Parquet row-aligned with an n-row synth cache (SyntheticPatchDataset schema)."""
    splits = ['train'] * n if splits is None else splits
    d = {'tile_id': [f't{i:04d}' for i in range(n)],
         'polygon_id': list(range(n)),
         'pixel_row': list(range(n)), 'pixel_col': list(range(n)),
         'olivine_t1': [0.0] * n, 'olivine_t2': [0.0] * n, 'lcp': [0.0] * n,
         'hcp': [0.0] * n, 'plagioclase': [1.0] * n, 'other': [0.0] * n,
         'confidence_weight': [1.0] * n, 'confidence_tier': ['High'] * n,
         'split': splits}
    pd.DataFrame(d).to_parquet(str(path), index=False)
    return str(path)


def _fingerprints(arr) -> list:
    """Deepest hull-CR band per row -> the row's identity (hull block only)."""
    a = np.asarray(arr)[:, 0, 0, :59]
    return a.argmin(axis=1).tolist()


# ── 0. the threshold is not a magic number ──────────────────────────────────

def test_raw_level_max_is_pinned_to_clip_max():
    """RAW_LEVEL_MAX must stay equal to CRISMSpectralPatchDataset.CLIP_MAX.

    The whole raw-vs-hull argument is "a raw patch cannot exceed CLIP_MAX by
    construction". data.dataset imports data.continuum_removal, so the constant
    cannot be imported the other way without a cycle; this test is what keeps the
    restated literal honest if CLIP_MAX ever changes.
    """
    from data.dataset import CRISMSpectralPatchDataset
    assert RAW_LEVEL_MAX == CRISMSpectralPatchDataset.CLIP_MAX == CLIP_MAX


# ── 1. the converter ────────────────────────────────────────────────────────

def test_hull_mode_is_exactly_continuum_removed(tmp_path):
    """--mode hull output must be bit-identical to continuum_removed(clean input),
    and must LOOK like hull-CR: 59 channels, bounded <= 1, median well above the
    raw ceiling. A converter that copied its input would satisfy the shape
    assertion alone."""
    raw = _raw_patches()
    src = _save(tmp_path / 'raw.npy', raw)
    out = str(tmp_path / 'hull.npy')
    convert(src, out, 'hull', log=lambda *a: None)

    got = np.load(out)
    want = continuum_removed(sanitize(raw)[0])
    assert got.shape == raw.shape
    np.testing.assert_array_equal(got, want)
    assert got.max() <= 1.0
    assert float(np.median(got)) > RAW_LEVEL_MAX
    assert detect_representation(got) == 'hull'


def test_dual_mode_layout_is_hull_then_linear_and_standardized(tmp_path):
    """--mode dual must write hull/hull_std in 0-58 and linear/linear_std in
    59-117. That channel order is load-bearing for the encoder, and dropping the
    standardisation would hand the pretrain's loss to the linear block."""
    raw = _raw_patches()
    src = _save(tmp_path / 'raw.npy', raw)
    out = str(tmp_path / 'dual.npy')
    convert(src, out, 'dual', log=lambda *a: None)

    got = np.load(out)
    assert got.shape == raw.shape[:3] + (118,)
    clean = sanitize(raw)[0]
    np.testing.assert_allclose(got[..., :59] * CR_SCALES['hull_std'],
                               continuum_removed(clean), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(got[..., 59:] * CR_SCALES['linear_std'],
                               linear_continuum_removed(clean), rtol=1e-5, atol=1e-6)
    # Standardisation's purpose: comparable per-block spread. Unstandardized the
    # linear block carries 2.45x the variance.
    s_hull, s_lin = got[..., :59].std(), got[..., 59:].std()
    assert 0.4 < s_hull / s_lin < 2.5, (s_hull, s_lin)
    assert detect_representation(got) == 'dual'


@pytest.mark.parametrize('mode', ['hull', 'dual'])
def test_row_order_and_row_count_are_preserved_exactly(tmp_path, mode):
    """Row i of the output must be the transform of row i of the input.

    These caches are aligned row-for-row with a parquet (data/dataset.py asserts
    cache.shape[0] == len(parquet)), so a permutation or a dropped row does not
    raise -- it silently mislabels every plagioclase patch. Each input row carries
    a unique absorption band, and hull-CR's deepest band recovers it.
    """
    raw = _raw_patches()
    src = _save(tmp_path / 'raw.npy', raw)
    out = str(tmp_path / f'{mode}.npy')
    convert(src, out, mode, chunk_rows=5, log=lambda *a: None)

    got = np.load(out)
    assert got.shape[0] == raw.shape[0] == len(FINGERPRINT_BANDS)
    assert _fingerprints(got) == FINGERPRINT_BANDS, (
        f'row order changed: {_fingerprints(got)} != {FINGERPRINT_BANDS}')


@pytest.mark.parametrize('mode', ['hull', 'dual'])
def test_chunking_changes_not_a_single_byte(tmp_path, mode):
    """chunk_rows must be a memory knob only. An off-by-one in the write window
    would shift or duplicate rows, which the fingerprint test catches only for
    the specific chunk size it uses."""
    src = _save(tmp_path / 'raw.npy', _raw_patches())
    a, b = str(tmp_path / 'a.npy'), str(tmp_path / 'b.npy')
    convert(src, a, mode, chunk_rows=1, log=lambda *x: None)
    convert(src, b, mode, chunk_rows=10_000, log=lambda *x: None)
    assert open(a, 'rb').read() == open(b, 'rb').read()


def test_refuses_input_that_is_already_hull_cr(tmp_path):
    """Double-transforming is not a no-op: hull-CR of hull-CR is not the identity.
    The refusal must fire on the statistics, and must not leave a partial file."""
    hull = continuum_removed(sanitize(_raw_patches())[0])
    src = _save(tmp_path / 'hull_in.npy', hull)
    out = str(tmp_path / 'out.npy')
    with pytest.raises(ValueError, match='ALREADY CONTINUUM-REMOVED'):
        convert(src, out, 'hull', log=lambda *a: None)
    assert not os.path.exists(out)
    assert not os.path.exists(out + '.partial')


def test_refuses_a_118_channel_input(tmp_path):
    src = _save(tmp_path / 'dual_in.npy',
                np.concatenate([_raw_patches()] * 2, axis=-1))
    with pytest.raises(ValueError, match='118 channels'):
        convert(src, str(tmp_path / 'o.npy'), 'dual', log=lambda *a: None)


def test_detection_uses_many_rows_not_one(tmp_path):
    """A single-pixel or first-row check gets both cases backwards.

    RAW cache whose first row is a uniform 0.95 (CR-looking but legal raw, below
    PHYS_MAX) must still convert; HULL cache whose first row is all zeros
    (raw-looking, and a real degenerate NODATA window) must still be refused.
    """
    raw = _raw_patches().copy()
    raw[0] = 0.95
    src = _save(tmp_path / 'raw_liar.npy', raw)
    convert(src, str(tmp_path / 'ok.npy'), 'hull', log=lambda *a: None)  # no raise

    hull = continuum_removed(sanitize(_raw_patches())[0]).copy()
    hull[0] = 0.0
    src2 = _save(tmp_path / 'hull_liar.npy', hull)
    with pytest.raises(ValueError, match='ALREADY CONTINUUM-REMOVED'):
        convert(src2, str(tmp_path / 'no.npy'), 'hull', log=lambda *a: None)


def test_nodata_and_implausible_values_are_zeroed_before_clipping(tmp_path):
    """The pipeline's policy, not a new one (CRISMSpectralPatchDataset.__getitem__):
    NODATA / non-finite / >1.0 I/F are set to 0.0 FIRST, then the rest is clipped
    to [0, CLIP_MAX]. Clipping a 1180 I/F blue-edge spike instead would cap it to
    a plausible-looking 0.5 and hand it to the hull as a real reflectance. Rows are
    never dropped: alignment outranks purity."""
    raw = _raw_patches().copy()
    raw[0, 0, 0, 0] = NODATA
    raw[0, 0, 1, 0] = 1180.0          # real 410 nm blue-edge spike magnitude
    raw[0, 0, 2, 0] = np.nan
    raw[0, 0, 3, 0] = 0.8             # plausible but above CLIP_MAX -> clipped

    clean, n_flagged = sanitize(raw)
    assert n_flagged == 3
    assert clean[0, 0, 0, 0] == 0.0 and clean[0, 0, 1, 0] == 0.0
    assert clean[0, 0, 2, 0] == 0.0
    assert clean[0, 0, 3, 0] == CLIP_MAX
    assert np.isfinite(clean).all()

    src = _save(tmp_path / 'raw.npy', raw)
    out = str(tmp_path / 'hull.npy')
    stats = convert(src, out, 'hull', log=lambda *a: None)
    got = np.load(out)
    assert np.isfinite(got).all()
    assert got.max() <= 1.0
    assert stats['n_flagged'] == 3


def test_refuses_to_clobber_an_existing_output_without_force(tmp_path):
    src = _save(tmp_path / 'raw.npy', _raw_patches())
    out = str(tmp_path / 'out.npy')
    convert(src, out, 'hull', log=lambda *a: None)
    first = open(out, 'rb').read()
    with pytest.raises(FileExistsError):
        convert(src, out, 'dual', log=lambda *a: None)
    assert open(out, 'rb').read() == first, 'output was modified despite refusing'
    convert(src, out, 'dual', force=True, log=lambda *a: None)
    assert np.load(out).shape[-1] == 118


def test_cli_exits_nonzero_on_a_transformed_input(tmp_path):
    """The SLURM job checks the exit status, so the refusal must be an exit code,
    not just a traceback-free message."""
    hull = continuum_removed(sanitize(_raw_patches())[0])
    src = _save(tmp_path / 'hull_in.npy', hull)
    p = subprocess.run([sys.executable, CONVERTER, '--input', src,
                        '--output', str(tmp_path / 'o.npy'), '--mode', 'dual'],
                       cwd=ROOT, capture_output=True, text=True)
    assert p.returncode != 0, p.stdout
    assert 'ALREADY CONTINUUM-REMOVED' in p.stderr, p.stderr


def test_cli_converts_and_reports(tmp_path):
    src = _save(tmp_path / 'raw.npy', _raw_patches())
    out = str(tmp_path / 'dual.npy')
    p = subprocess.run([sys.executable, CONVERTER, '--input', src,
                        '--output', out, '--mode', 'dual'],
                       cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert np.load(out, mmap_mode='r').shape[-1] == 118
    assert 'hull_block' in p.stdout and 'linear_block' in p.stdout


# ── 2. the dataset guard ────────────────────────────────────────────────────

def _synth(tmp_path, arr, name='synth'):
    npy = _save(tmp_path / f'{name}.npy', arr)
    pq = _rows_parquet(tmp_path / f'{name}.parquet', len(arr))
    return npy, pq


def _hull(arr=None):
    return continuum_removed(sanitize(_raw_patches() if arr is None else arr)[0])


def _dual(tmp_path):
    src = _save(tmp_path / '_raw_for_dual.npy', _raw_patches())
    out = str(tmp_path / '_dual.npy')
    convert(src, out, 'dual', log=lambda *a: None)
    return np.load(out)


def test_hull_run_rejects_a_raw_synth_cache(tmp_path):
    """The ft_7cls_handcore_level bug, as a unit test."""
    from data.dataset import SyntheticPatchDataset
    npy, pq = _synth(tmp_path, _raw_patches())
    with pytest.raises(ValueError, match='looks RAW'):
        SyntheticPatchDataset(npy, pq, expect_repr='hull')


def test_raw_run_rejects_a_hull_synth_cache(tmp_path):
    """The mirror image: a CR cache injected into a raw-reflectance run."""
    from data.dataset import SyntheticPatchDataset
    npy, pq = _synth(tmp_path, _hull())
    with pytest.raises(ValueError, match='looks HULL'):
        SyntheticPatchDataset(npy, pq, expect_repr='raw')


def test_dual_expectation_rejects_a_59_channel_cache(tmp_path):
    from data.dataset import SyntheticPatchDataset
    npy, pq = _synth(tmp_path, _hull())
    with pytest.raises(AssertionError):
        SyntheticPatchDataset(npy, pq, expect_repr='dual')


def test_dual_expectation_rejects_a_raw_lookalike_at_118_channels(tmp_path):
    """118 channels is necessary but not sufficient: raw duplicated into two
    blocks has the right width and the wrong levels."""
    from data.dataset import SyntheticPatchDataset
    raw = _raw_patches()
    npy, pq = _synth(tmp_path, np.concatenate([raw, raw], axis=-1))
    with pytest.raises(ValueError, match='does not look'):
        SyntheticPatchDataset(npy, pq, expect_repr='dual')


@pytest.mark.parametrize('repr_name', ['raw', 'hull', 'dual'])
def test_matching_representations_are_accepted(tmp_path, repr_name):
    """The guard must not be a blanket refusal."""
    from data.dataset import SyntheticPatchDataset
    arr = {'raw': _raw_patches, 'hull': _hull,
           'dual': lambda: _dual(tmp_path)}[repr_name]()
    npy, pq = _synth(tmp_path, arr, name=repr_name)
    ds = SyntheticPatchDataset(npy, pq, expect_repr=repr_name)
    assert len(ds) == len(arr)
    patch, label, weight = ds[0]
    assert tuple(patch.shape) == (7, 7, 118 if repr_name == 'dual' else 59)


def test_default_expect_repr_is_inert_and_serves_verbatim(tmp_path):
    """BASE behaviour with no expect_repr: a RAW cache loads without complaint and
    every row is served byte-for-byte. This is what keeps every pre-existing
    caller working."""
    import inspect
    from data.dataset import SyntheticPatchDataset
    assert inspect.signature(
        SyntheticPatchDataset.__init__).parameters['expect_repr'].default == 'any'

    raw = _raw_patches()
    npy, pq = _synth(tmp_path, raw)
    ds = SyntheticPatchDataset(npy, pq)          # no expect_repr at all
    assert len(ds) == len(raw)
    for i in range(len(raw)):
        np.testing.assert_array_equal(ds[i][0].numpy(), raw[i])


def test_bad_expect_repr_value_is_rejected(tmp_path):
    """A typo'd representation name must not silently become a no-op check.

    NOTE the match string: `match='expect_repr'` passed even with the validity
    check deleted, because pytest's tmp_path is named after the test and the
    mismatch error quotes the cache PATH. Mutation M16 caught it. Match on the
    enumeration message instead.
    """
    from data.dataset import SyntheticPatchDataset
    npy, pq = _synth(tmp_path, _raw_patches())
    with pytest.raises(ValueError, match=r'expected one of'):
        SyntheticPatchDataset(npy, pq, expect_repr='cr')


# ── 3. train_torch wiring ───────────────────────────────────────────────────

@pytest.mark.parametrize('cr, dual, want', [
    (True, False, 'hull'),    # --continuum_removed [--cache_is_cr]: the handcore case
    (False, False, 'raw'),    # raw run: pre-existing behaviour
    (True, True, 'dual'),     # --dual_cr
])
def test_expected_synth_repr_mapping(cr, dual, want):
    from training.train_torch import _expected_synth_repr
    assert _expected_synth_repr(cr, dual) == want


def _labeled_cache(tmp_path, df, n_channels=59):
    """Headerless memmap labeled patch cache, as build_cr_labeled_cache.py writes."""
    rng = np.random.default_rng(0)
    for split in ('train', 'val'):
        sub = df[df['split'] == split]
        fp = np.memmap(str(tmp_path / f'mrral_{split}_patches_p7.npy'),
                       dtype='float32', mode='w+',
                       shape=(len(sub), 7, 7, n_channels))
        fp[:] = rng.uniform(0.05, 0.35,
                            size=(len(sub), 7, 7, n_channels)).astype(np.float32)
        fp.flush()
        del fp


def _tiny_labeled_df(n=24):
    d = {'tile_id': ['t0001'] * n, 'polygon_id': list(range(n)),
         'pixel_row': [3] * n, 'pixel_col': [3] * n,
         'confidence_weight': np.ones(n, dtype=np.float32),
         'confidence_tier': ['High'] * n,
         'split': ['train'] * (n // 2) + ['val'] * (n - n // 2)}
    for c in ('olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other'):
        d[c] = np.zeros(n, dtype=np.float32)
    d['lcp'] = np.ones(n, dtype=np.float32)
    return pd.DataFrame(d)


def _train(tmp_path, synth_arr, **extra):
    """One epoch of train_torch_model with a synth cache concatenated in."""
    from data.dataset import LABEL_COLS
    from models.spatial_spectral_transformer import SpatialSpectralClassifier
    from training.train_torch import train_torch_model

    df = _tiny_labeled_df()
    cache = tmp_path / 'cache'
    cache.mkdir(exist_ok=True)
    _labeled_cache(cache, df)
    npy, pq = _synth(tmp_path, synth_arr, name='synth_for_train')
    model = SpatialSpectralClassifier(
        n_bands=59, patch_size=7, n_classes=len(LABEL_COLS),
        embed_dim=16, n_heads=2, n_layers=1)
    return train_torch_model(
        model=model, df=df, model_name='synth_repr_probe', max_epochs=1,
        batch_size=8, lr=1e-3, use_wandb=False, checkpoint_dir=None,
        mrral_map={'t0001': '/nonexistent.img'}, patch_size=7,
        cache_dir=str(cache), device='cpu',
        synth_train_cache=npy, synth_train_parquet=pq, **extra)


def test_handcore_style_run_refuses_a_raw_synth_cache(tmp_path):
    """`--continuum_removed --cache_is_cr` + a RAW synth cache must FAIL.

    This is what makes scripts/hpc_finetune_handcore.slurm fail as currently
    written, which is the entire point: it was training plagioclase on raw patches
    in a CR run and reporting success.
    """
    with pytest.raises(ValueError, match='looks RAW'):
        _train(tmp_path, _raw_patches(), continuum_removed=True, cache_is_cr=True)


def test_cr_run_accepts_a_hull_synth_cache(tmp_path):
    """...and the corrected invocation trains. The guard is about agreement, not
    about refusing synth caches."""
    hist = _train(tmp_path, _hull(), continuum_removed=True, cache_is_cr=True)
    assert hist is not None


def test_raw_run_still_accepts_a_raw_synth_cache(tmp_path):
    """The pre-existing 59-band raw path is untouched: no flags, raw cache, trains.

    Every caller that does not pass --continuum_removed keeps working exactly as
    before -- the guard resolves to 'raw' for them.
    """
    hist = _train(tmp_path, _raw_patches())
    assert hist is not None


# ── 4. scripts/train.py parse-time width check ──────────────────────────────

def _parse_args(argv):
    code = ("import sys; sys.argv=['train.py']+%r; "
            "import scripts.train as t; a=t.build_args(); "
            "print('PARSED_OK', a.dual_cr)" % (argv,))
    p = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                       capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def _dual_cr_argv(tmp_path, cache, pq):
    return ['--dual_cr', '--continuum_removed', '--model', 'spatial_vit',
            '--mrral_parquets', '_', '--synth_train_cache', cache,
            '--synth_train_parquet', pq]


def test_dual_cr_accepts_a_118_channel_synth_cache(tmp_path):
    """A genuine dual plag cache must now be ACCEPTED -- otherwise
    hpc_finetune_dualcr.slurm cannot inject plagioclase and the arm carries a
    second variable."""
    dual = _dual(tmp_path)
    npy, pq = _synth(tmp_path, dual, name='dual_ok')
    rc, out, err = _parse_args(_dual_cr_argv(tmp_path, npy, pq))
    assert rc == 0, f'a 118-channel synth cache was refused:\n{out}\n{err}'
    assert 'PARSED_OK True' in out, out


def test_dual_cr_rejects_a_59_channel_synth_cache(tmp_path):
    """The load-bearing half of Task 7's refusal survives: mixing 59 into 118 is
    still fatal, now diagnosed from the real array width."""
    npy, pq = _synth(tmp_path, _hull(), name='hull_59')
    rc, out, err = _parse_args(_dual_cr_argv(tmp_path, npy, pq))
    assert rc != 0, f'a 59-channel synth cache was accepted under --dual_cr:\n{out}'
    assert '118-channel dual-CR' in err, err


def test_dual_cr_rejects_an_unreadable_synth_cache(tmp_path):
    rc, out, err = _parse_args(
        _dual_cr_argv(tmp_path, str(tmp_path / 'nope.npy'), 'x'))
    assert rc != 0, out
    assert 'cannot be verified' in err, err


# ── 5. hpc_finetune_dualcr.slurm ────────────────────────────────────────────

def _slurm_text():
    return open(DUALCR_SLURM).read()


def _plag_preflight():
    """The job's OWN plag preflight heredoc, verbatim."""
    from tests.test_dual_cr_wiring import _extract_heredocs
    blocks = [b for b in _extract_heredocs(DUALCR_SLURM) if 'plag cache OK' in b]
    assert len(blocks) == 1, f'expected exactly one plag preflight, got {len(blocks)}'
    return blocks[0]


def _run_preflight(code, argv):
    p = subprocess.run([sys.executable, '-c', code] + list(argv), cwd=ROOT,
                       capture_output=True, text=True, timeout=300)
    return p.returncode, p.stdout + p.stderr


def _code_lines():
    """Non-comment, non-blank lines only. A `grep` over the whole file is not a
    test: this job's HEADER mentions --synth_train_cache and --mode hull in prose,
    so a whole-file substring check passes even after the flag is deleted from the
    command. That exact false pass was caught by mutation M23."""
    return [l for l in _slurm_text().splitlines()
            if l.strip() and not l.strip().startswith('#')]


def _train_invocation():
    """The `$PYTHON -u scripts/train.py ... \\` continuation block, verbatim."""
    lines = _code_lines()
    i = next(i for i, l in enumerate(lines) if 'scripts/train.py' in l)
    block = []
    while True:
        block.append(lines[i])
        if not lines[i].rstrip().endswith('\\'):
            break
        i += 1
    return '\n'.join(block)


def test_slurm_builds_and_passes_the_dual_plag_cache():
    """The job must (a) call the converter in --mode dual and (b) actually pass all
    four --synth_* flags, with their variables, ON THE train.py COMMAND LINE.
    Task 7's version omitted them entirely."""
    code = '\n'.join(_code_lines())
    assert 'convert_synth_cache_representation.py' in code
    assert '--mode dual' in code

    inv = _train_invocation()
    for flag, var in (('--synth_train_cache', '"$SYNTH_TRAIN"'),
                      ('--synth_train_parquet', '"$SYNTH_TRAIN_ROWS"'),
                      ('--synth_val_cache', '"$SYNTH_VAL"'),
                      ('--synth_val_parquet', '"$SYNTH_VAL_ROWS"')):
        assert f'{flag} {var}' in inv, (
            f'{flag} {var} is not on the train.py command line:\n{inv}')


def test_slurm_keeps_every_data_path_on_xdisk():
    """/groups filled up and killed two cache builds with Errno 28. Only the code
    checkout and the interpreter may live there."""
    for line in _slurm_text().splitlines():
        s = line.strip()
        if not s or s.startswith('#') or '/groups' not in s:
            continue
        assert s.startswith('WORK_DIR=') or s.startswith('PYTHON='), (
            f'data path on /groups: {s}')
    synth = [l for l in _slurm_text().splitlines() if l.startswith('SYNTH_')]
    assert len(synth) >= 6, synth
    for line in synth:
        val = line.split('=', 1)[1].split()[0]
        assert val.startswith('${DATA_DIR}') or val.startswith('${SYNTH_DIR}'), line


def test_slurm_header_still_warns_about_the_plag_comparator():
    """Task 7's warning must be UPDATED, not deleted: the dual side is fixed, the
    hull-CR baseline is still built on a raw plag cache."""
    src = _slurm_text()
    assert 'ft_7cls_handcore_level is therefore NOT a valid plagioclase comparator' \
        in src.replace('\n# ', ' ').replace('#', ''), src[:2000]
    assert 'RAW-REFLECTANCE' in src


def test_plag_preflight_accepts_a_real_converted_dual_cache(tmp_path):
    """Run the job's heredoc against a cache from the REAL converter."""
    dual = _dual(tmp_path)
    npy = _save(tmp_path / 'plag_dual.npy', dual)
    pq = _rows_parquet(tmp_path / 'plag.parquet', len(dual))
    rc, log = _run_preflight(_plag_preflight(), [npy, pq, npy, pq])
    assert rc == 0, f'preflight REJECTED a correct dual plag cache:\n{log}'
    assert '118' in log, log


def test_plag_preflight_rejects_a_59channel_cache_with_a_MATCHING_BYTE_COUNT(tmp_path):
    """The trap that has already shipped twice in this plan.

    A 59-channel cache with 2k rows occupies exactly as many data bytes as a
    118-channel cache with k rows, because 59 * 2k == 118 * k. A byte-arithmetic
    preflight passes it. Only the true array shape catches it.
    """
    k = len(FINGERPRINT_BANDS)
    raw = np.concatenate([_raw_patches(), _raw_patches()], axis=0)   # 2k rows, 59 ch
    assert raw.shape == (2 * k, 7, 7, 59)
    assert raw.size * 4 == k * 7 * 7 * 118 * 4, 'the byte-count trap is not set up'
    npy = _save(tmp_path / 'liar.npy', raw)
    pq = _rows_parquet(tmp_path / 'liar.parquet', k)
    rc, log = _run_preflight(_plag_preflight(), [npy, pq, npy, pq])
    assert rc != 0, f'preflight accepted a 59-channel cache:\n{log}'
    assert 'not 118' in log, log


def test_plag_preflight_rejects_a_row_count_mismatch(tmp_path):
    """SyntheticPatchDataset asserts cache rows == parquet rows; catch it before
    the GPU is allocated."""
    dual = _dual(tmp_path)
    npy = _save(tmp_path / 'plag_dual.npy', dual)
    pq = _rows_parquet(tmp_path / 'short.parquet', len(dual) - 1)
    rc, log = _run_preflight(_plag_preflight(), [npy, pq, npy, pq])
    assert rc != 0, f'preflight accepted a row-count mismatch:\n{log}'
    assert 'parquet' in log.lower(), log

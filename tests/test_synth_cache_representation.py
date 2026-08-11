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
    """Run the job's heredoc against a cache from the REAL converter.

    Uses the converter's own OUTPUT PATH, not a re-save of its array: as of
    2026-08-11 the preflight also demands the brightness sidecar the converter
    writes beside it, and a re-saved copy would have none.
    """
    npy = _converted(tmp_path, 'dual', 'plag')
    dual = np.load(npy, mmap_mode='r')
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


# ── 6. the brightness sidecar ───────────────────────────────────────────────
#
# THE BUG (2026-08-11, hpc_finetune_handcore --array=0-1, died in seconds):
#
#   INFO Concatenating 8671 synthetic plag patches into train set
#   RuntimeError: each element in list of batch should be of equal size
#
# Under --brightness_aux, CRISMSpectralPatchDataset._finish returns the 4-tuple
# (patch, brightness (1,), label, weight) while SyntheticPatchDataset returned a
# 3-tuple. ConcatDataset mixes them and default_collate rejects the first batch
# that spans the boundary. Latent since the beginning: until 07547e8 the
# --synth_* flags were silently dropped for spatial_vit_aux, so the two dataset
# types had never actually been concatenated in a brightness run.
#
# The fix mirrors the labeled cache: build_cr_labeled_cache.py writes an
# (N, P, P) float32 .npy brightness sidecar beside its CR patches, and the
# converter now does the same beside a converted synth cache.

import shutil                                                       # noqa: E402
import subprocess as _sp                                            # noqa: E402

import torch                                                        # noqa: E402
from torch.utils.data import ConcatDataset, DataLoader              # noqa: E402

from data.continuum_removal import brightness_scalar, cr_patch      # noqa: E402
from data.dataset import (LABEL_COLS, CRISMSpectralPatchDataset,    # noqa: E402
                          SyntheticPatchDataset, synth_brightness_path)
from scripts.convert_synth_cache_representation import (  # noqa: E402
    describe_brightness, transform_chunk)

HANDCORE_SLURM = os.path.join(ROOT, 'scripts', 'hpc_finetune_handcore.slurm')


def _converted(tmp_path, mode, name='conv', arr=None):
    """Run the REAL converter; return the output path (sidecar sits beside it)."""
    src = _save(tmp_path / f'_{name}_raw.npy',
                _raw_patches() if arr is None else arr)
    out = str(tmp_path / f'{name}_{mode}.npy')
    convert(src, out, mode, log=lambda *a: None)
    return out


def _level_ramp(n=10) -> np.ndarray:
    """(n,7,7,59) raw patches whose brightness is strictly increasing in row i.

    Row-identity by LEVEL, not by absorption band: brightness_scalar is a mean
    over good bands, so a level ramp makes the sidecar's row order directly
    readable and any reorder/misindex visible.

    A within-patch spatial gradient rides on top, so no two pixels of a patch
    share a brightness. Without it every pixel is equal and a test that claims to
    read the CENTRE pixel passes when handed a corner (mutation M5 survived
    exactly that way).
    """
    lv = (0.05 + 0.03 * np.arange(n, dtype=np.float32))[:, None, None, None]
    r = np.arange(7, dtype=np.float32)[None, :, None, None]
    c = np.arange(7, dtype=np.float32)[None, None, :, None]
    return np.ascontiguousarray(
        np.broadcast_to(lv * (1.0 + 0.02 * r + 0.01 * c),
                        (n, 7, 7, 59)).astype(np.float32))


# ── 6a. the converter writes it ─────────────────────────────────────────────

@pytest.mark.parametrize('mode', ['hull', 'dual'])
def test_converter_writes_a_sidecar_in_the_labeled_cache_layout(tmp_path, mode):
    """<stem>_brightness.npy, (N, P, P) float32, np.load-able.

    The layout is not a free choice: CRISMSpectralPatchDataset indexes the
    labeled sidecar as bright[row, half, half] (data/dataset.py:585), so an (N,)
    per-row scalar array would raise IndexError, and an (N, P, P, 59) map would
    feed a whole spectrum where a scalar belongs. build_cr_labeled_cache.py:90
    opens exactly (n, P, P) via np.lib.format.open_memmap.
    """
    out = _converted(tmp_path, mode)
    n = len(np.load(out, mmap_mode='r'))
    side = synth_brightness_path(out)
    assert side == os.path.splitext(out)[0] + '_brightness.npy'
    assert os.path.exists(side), f'converter wrote no sidecar at {side}'
    b = np.load(side)                    # np.load, not np.memmap: a real .npy
    assert b.shape == (n, 7, 7), b.shape
    assert b.dtype == np.float32, b.dtype
    assert np.isfinite(b).all()


@pytest.mark.parametrize('mode', ['hull', 'dual'])
def test_sidecar_equals_finish_brightness_of_the_sanitized_raw(tmp_path, mode):
    """Exact agreement with what CRISMSpectralPatchDataset._finish would compute.

    _finish's two paths reach brightness differently -- the hull path unpacks
    cr_patch(patch) -> (cr, brightness), the dual path calls brightness_scalar
    separately -- but cr_patch IS (continuum_removed, brightness_scalar) of the
    same array (data/continuum_removal.py:243-250), so both reduce to
    brightness_scalar of the pre-transform patch. This asserts against the hull
    path's cr_patch for BOTH modes, which is only true if they really agree.
    """
    raw = _raw_patches()
    out = _converted(tmp_path, mode, arr=raw)
    b = np.load(synth_brightness_path(out))
    clean, _ = sanitize(raw)
    np.testing.assert_allclose(b, brightness_scalar(clean), rtol=0, atol=0)
    for i in range(len(raw)):
        _, expect = cr_patch(clean[i])           # the _finish hull path, verbatim
        np.testing.assert_allclose(b[i], expect, rtol=0, atol=0)


def test_sidecar_comes_from_the_raw_patch_not_the_transformed_one(tmp_path):
    """hull and dual sidecars of the same input must be BYTE-IDENTICAL.

    They can only agree if brightness is taken BEFORE the transform. Computing it
    after would give hull-CR levels (~0.93) in one file and standardized dual
    levels (~13) in the other -- and would silently feed the aux head a
    continuum-removed quantity instead of the albedo the head exists to see.
    """
    raw = _raw_patches()
    bh = np.load(synth_brightness_path(_converted(tmp_path, 'hull', 'h', raw)))
    bd = np.load(synth_brightness_path(_converted(tmp_path, 'dual', 'd', raw)))
    np.testing.assert_array_equal(bh, bd)
    # ...and it is the RAW level (raw patches sit in [0, CLIP_MAX]), not a CR one.
    assert bh.max() <= CLIP_MAX, bh.max()
    assert abs(float(np.median(bh)) - float(np.median(brightness_scalar(raw)))) < 1e-6


def test_sidecar_rows_stay_aligned_with_patch_rows(tmp_path):
    """Row i of the sidecar must describe row i of the cache.

    A misaligned sidecar mislabels plagioclase brightness rather than failing:
    the shapes still match and training proceeds. The level ramp makes the row
    order readable -- brightness must be strictly increasing in row index.
    """
    raw = _level_ramp(10)
    out = _converted(tmp_path, 'hull', 'ramp', raw)
    b = np.load(synth_brightness_path(out))
    centre = b[:, 3, 3]
    assert np.all(np.diff(centre) > 0), centre
    np.testing.assert_allclose(centre, brightness_scalar(raw)[:, 3, 3],
                               rtol=0, atol=1e-6)


def test_chunking_does_not_change_the_sidecar(tmp_path):
    """chunk_rows must not affect a single byte of the sidecar, exactly as it must
    not affect the patches (test_chunking_changes_not_a_single_byte)."""
    raw = _level_ramp(9)
    a = _converted(tmp_path, 'hull', 'chunkA', raw)
    src = _save(tmp_path / '_chunkB_raw.npy', raw)
    b_out = str(tmp_path / 'chunkB_hull.npy')
    convert(src, b_out, 'hull', chunk_rows=2, log=lambda *a: None)
    assert open(synth_brightness_path(a), 'rb').read() == \
        open(synth_brightness_path(b_out), 'rb').read()


def test_converter_refuses_to_clobber_an_existing_sidecar(tmp_path):
    """A stale sidecar with the patches deleted must not survive a rebuild.

    The pre-existing guard covered --output only, so `rm cache.npy && convert`
    would leave the OLD sidecar in place beside NEW patches -- silently
    misaligned if the source ever changed row count.
    """
    src = _save(tmp_path / 'raw.npy', _raw_patches())
    out = str(tmp_path / 'out.npy')
    np.save(synth_brightness_path(out), np.zeros((3, 7, 7), dtype=np.float32))
    with pytest.raises(FileExistsError, match='_brightness.npy'):
        convert(src, out, 'hull', log=lambda *a: None)
    convert(src, out, 'hull', force=True, log=lambda *a: None)   # --force still works
    assert np.load(synth_brightness_path(out)).shape[0] == len(_raw_patches())


def test_transform_chunk_returns_brightness_for_both_modes():
    """The chunk-level contract, independent of file IO."""
    raw = _raw_patches()[:4]
    for mode, n_ch in (('hull', 59), ('dual', 118)):
        block, bright, _ = transform_chunk(raw, mode)
        assert block.shape == (4, 7, 7, n_ch)
        assert bright.shape == (4, 7, 7) and bright.dtype == np.float32
        np.testing.assert_allclose(bright, brightness_scalar(sanitize(raw)[0]),
                                   rtol=0, atol=0)


def test_describe_brightness_reports_the_centre_pixel_range(tmp_path):
    """The reported range must cover the CENTRE pixel specifically -- that, not
    the whole map, is the scalar the datasets serve as the aux feature."""
    raw = _level_ramp(6)
    out = _converted(tmp_path, 'hull', 'desc', raw)
    st = describe_brightness(np.load(synth_brightness_path(out)))
    centre = brightness_scalar(raw)[:, 3, 3]
    assert st['shape'] == (6, 7, 7)
    np.testing.assert_allclose(st['centre_min'], centre.min(), rtol=0, atol=1e-6)
    np.testing.assert_allclose(st['centre_max'], centre.max(), rtol=0, atol=1e-6)


# ── 6b. SyntheticPatchDataset.return_brightness ─────────────────────────────

def test_synth_dataset_returns_the_finish_4tuple(tmp_path):
    """Same length, ORDER and dtypes as CRISMSpectralPatchDataset._finish.

    A tuple-length check alone is not enough: (patch, label, bright, weight)
    collates perfectly and trains the aux head on one-hot labels.
    """
    out = _converted(tmp_path, 'hull')
    n = len(np.load(out, mmap_mode='r'))
    pq = _rows_parquet(tmp_path / 'rows.parquet', n)
    ds = SyntheticPatchDataset(out, pq, patch_size=7, expect_repr='hull',
                               return_brightness=True)
    b = np.load(synth_brightness_path(out))
    patch, bright, label, weight = ds[3]
    assert patch.shape == (7, 7, 59) and patch.dtype == torch.float32
    assert bright.shape == (1,) and bright.dtype == torch.float32
    assert label.shape == (len(LABEL_COLS),) and label.dtype == torch.float32
    assert weight.shape == () and weight.dtype == torch.float32
    np.testing.assert_allclose(bright.item(), b[3, 3, 3], rtol=0, atol=1e-6)
    # element 1 is brightness, not the label: they must not be interchangeable
    assert bright.numel() == 1 and label.numel() == len(LABEL_COLS)


def test_synth_dataset_default_is_the_unchanged_3tuple(tmp_path):
    """Inertness. With the sidecar SITTING RIGHT THERE, the default caller still
    gets exactly the pre-2026-08-11 3-tuple and identical patch bytes."""
    out = _converted(tmp_path, 'hull')
    n = len(np.load(out, mmap_mode='r'))
    pq = _rows_parquet(tmp_path / 'rows.parquet', n)
    assert os.path.exists(synth_brightness_path(out))
    ds = SyntheticPatchDataset(out, pq, patch_size=7, expect_repr='hull')
    item = ds[2]
    assert len(item) == 3, 'default return_brightness must stay False'
    ds_b = SyntheticPatchDataset(out, pq, patch_size=7, expect_repr='hull',
                                 return_brightness=True)
    np.testing.assert_array_equal(item[0].numpy(), ds_b[2][0].numpy())
    assert torch.equal(item[1], ds_b[2][2]) and torch.equal(item[2], ds_b[2][3])


def test_synth_dataset_without_sidecar_raises_naming_file_and_builder(tmp_path):
    """Absent sidecar must be LOUD at construction, never a zero fill.

    A zero (or synthesised) brightness trains plagioclase against a constant aux
    feature: the run completes, val_mAP looks normal, and the aux head has simply
    learned "plag == aux 0.0" -- undetectable except at inference.
    """
    out = _converted(tmp_path, 'hull')
    n = len(np.load(out, mmap_mode='r'))
    pq = _rows_parquet(tmp_path / 'rows.parquet', n)
    side = synth_brightness_path(out)
    os.remove(side)
    with pytest.raises(FileNotFoundError) as e:
        SyntheticPatchDataset(out, pq, patch_size=7, expect_repr='hull',
                              return_brightness=True)
    msg = str(e.value)
    assert side in msg, msg
    assert 'convert_synth_cache_representation.py' in msg, msg


def test_synth_brightness_is_indexed_by_the_npy_row_not_the_split_row(tmp_path):
    """Under a split filter the sidecar must be indexed by the MAPPED row.

    self._indices[idx] maps a filtered position to its row in the full npy; the
    sidecar is aligned with the npy, not with the filtered frame. Indexing it
    with `idx` pairs patch _indices[idx] with brightness idx -- a silent
    mislabel, and the real caches ARE split-filtered (split='train'/'val').
    """
    raw = _level_ramp(10)
    out = _converted(tmp_path, 'hull', 'split', raw)
    splits = ['val'] * 6 + ['train'] * 4          # train rows are npy rows 6..9
    pq = _rows_parquet(tmp_path / 'rows.parquet', 10, splits=splits)
    ds = SyntheticPatchDataset(out, pq, patch_size=7, split='train',
                               expect_repr='hull', return_brightness=True)
    assert len(ds) == 4
    b = np.load(synth_brightness_path(out))
    for i in range(4):
        _, bright, _, _ = ds[i]
        np.testing.assert_allclose(bright.item(), b[6 + i, 3, 3], rtol=0, atol=1e-6)
        assert abs(bright.item() - b[i, 3, 3]) > 1e-3, (
            'brightness was read at the FILTERED index, not the npy row')


def test_synth_dataset_rejects_a_sidecar_of_the_wrong_shape(tmp_path):
    """A sidecar from a different cache (or an (N,) scalar layout) must fail at
    construction rather than IndexError mid-epoch."""
    out = _converted(tmp_path, 'hull')
    n = len(np.load(out, mmap_mode='r'))
    pq = _rows_parquet(tmp_path / 'rows.parquet', n)
    np.save(synth_brightness_path(out), np.zeros(n, dtype=np.float32))
    with pytest.raises(ValueError, match='brightness sidecar'):
        SyntheticPatchDataset(out, pq, patch_size=7, expect_repr='hull',
                              return_brightness=True)


# ── 6c. THE LOAD-BEARING TEST: the two datasets must collate together ───────

def _labeled_cr_dataset(tmp_path, n=6, P=7):
    """A brightness-returning CRISMSpectralPatchDataset over a cache_is_cr cache."""
    rng = np.random.default_rng(7)
    cr = rng.uniform(0.6, 1.0, (n, P, P, 59)).astype(np.float32)
    fp = np.memmap(str(tmp_path / f'mrral_train_patches_p{P}.npy'),
                   dtype='float32', mode='w+', shape=(n, P, P, 59))
    fp[:] = cr; fp.flush(); del fp
    np.save(str(tmp_path / f'mrral_train_patches_p{P}_brightness.npy'),
            rng.uniform(0.05, 0.4, (n, P, P)).astype(np.float32))
    d = {'tile_id': ['t0001'] * n, 'pixel_row': [3] * n, 'pixel_col': [3] * n,
         'confidence_weight': [1.0] * n, 'confidence_tier': ['High'] * n,
         'split': ['train'] * n}
    for c in ('olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other'):
        d[c] = [0.0] * n
    d['lcp'] = [1.0] * n
    df = pd.DataFrame(d)
    return CRISMSpectralPatchDataset(
        df, {}, patch_size=P, cache_dir=str(tmp_path), split='train',
        continuum_removed=True, return_brightness=True, cache_is_cr=True), n


def test_concat_of_labeled_and_synth_collates_under_brightness_aux(tmp_path):
    """THE crash, as a test. A DataLoader over ConcatDataset([labeled, synth])
    with brightness on BOTH halves must collate every batch, including the one
    that straddles the boundary -- into 4 tensors with the aux in position 1.
    """
    labeled_dir = tmp_path / 'labeled'; labeled_dir.mkdir()
    labeled, n_lab = _labeled_cr_dataset(labeled_dir)
    out = _converted(tmp_path, 'hull')
    n_syn = len(np.load(out, mmap_mode='r'))
    pq = _rows_parquet(tmp_path / 'rows.parquet', n_syn)
    synth = SyntheticPatchDataset(out, pq, patch_size=7, expect_repr='hull',
                                  return_brightness=True)
    loader = DataLoader(ConcatDataset([labeled, synth]), batch_size=4,
                        shuffle=False)
    seen = 0
    straddled = False
    for patch, bright, label, weight in loader:      # 4-unpack: arity + order
        bs = patch.shape[0]
        if seen < n_lab < seen + bs:
            straddled = True
        assert patch.shape == (bs, 7, 7, 59) and patch.dtype == torch.float32
        assert bright.shape == (bs, 1) and bright.dtype == torch.float32
        assert label.shape == (bs, len(LABEL_COLS))
        assert weight.shape == (bs,)
        assert torch.isfinite(bright).all()
        seen += bs
    assert seen == n_lab + n_syn
    assert straddled, 'no batch mixed the two dataset types; test proves nothing'


def test_concat_without_synth_brightness_is_the_original_runtimeerror(tmp_path):
    """Regression documentation: the un-fixed pairing still fails the same way.

    This is the exact message from the killed job. It pins WHY return_brightness
    must track brightness_aux -- if some future change made a 3-tuple collate
    silently against a 4-tuple, the fix above would have become unnecessary and
    this test would tell us.
    """
    labeled_dir = tmp_path / 'labeled'; labeled_dir.mkdir()
    labeled, _ = _labeled_cr_dataset(labeled_dir)
    out = _converted(tmp_path, 'hull')
    n_syn = len(np.load(out, mmap_mode='r'))
    pq = _rows_parquet(tmp_path / 'rows.parquet', n_syn)
    synth = SyntheticPatchDataset(out, pq, patch_size=7, expect_repr='hull')
    loader = DataLoader(ConcatDataset([labeled, synth]), batch_size=4,
                        shuffle=False)
    with pytest.raises(RuntimeError, match='equal size'):
        for _ in loader:
            pass


# ── 6d. train_torch wiring ──────────────────────────────────────────────────

def _train_aux(tmp_path, mode='hull'):
    """One epoch of the real handcore configuration: --continuum_removed
    --cache_is_cr --brightness_aux with a converted synth cache concatenated in."""
    from models.spatial_spectral_classifier_aux import SpatialSpectralClassifierAux
    from training.train_torch import train_torch_model

    df = _tiny_labeled_df()
    cache = tmp_path / 'cache'; cache.mkdir(exist_ok=True)
    _labeled_cache(cache, df)
    for split in ('train', 'val'):
        k = int((df['split'] == split).sum())
        np.save(str(cache / f'mrral_{split}_patches_p7_brightness.npy'),
                np.full((k, 7, 7), 0.2, dtype=np.float32))
    out = _converted(tmp_path, mode, 'train_synth')
    n = len(np.load(out, mmap_mode='r'))
    # Mixed splits, so BOTH SyntheticPatchDataset constructions get a non-empty
    # dataset. With an all-'train' parquet the val synth set filters down to zero
    # rows, is never collated, and dropping return_brightness from the val
    # construction goes undetected -- mutation M12 survived exactly that way.
    pq = _rows_parquet(tmp_path / 'train_synth_rows.parquet', n,
                       splits=['train'] * (n // 2) + ['val'] * (n - n // 2))
    model = SpatialSpectralClassifierAux(
        n_bands=59, patch_size=7, n_classes=len(LABEL_COLS),
        embed_dim=16, n_heads=2, n_layers=1, aux_dim=1)
    return train_torch_model(
        model=model, df=df, model_name='synth_bright_probe', max_epochs=1,
        batch_size=8, lr=1e-3, use_wandb=False, checkpoint_dir=None,
        mrral_map={'t0001': '/nonexistent.img'}, patch_size=7,
        cache_dir=str(cache), device='cpu',
        continuum_removed=True, cache_is_cr=True,
        brightness_aux=True, is_aux_model=True,
        synth_train_cache=out, synth_train_parquet=pq,
        synth_val_cache=out, synth_val_parquet=pq)


def test_brightness_aux_run_concatenates_the_synth_cache_and_trains(tmp_path):
    """The job that died, end to end through train_torch_model.

    Both SyntheticPatchDataset constructions (train AND val) must receive
    return_brightness=brightness_aux; leaving either one out reproduces the
    RuntimeError in the corresponding loader.
    """
    hist = _train_aux(tmp_path)
    assert hist is not None


def test_brightness_aux_run_fails_loudly_when_the_sidecar_is_missing(tmp_path):
    """No silent degradation: a synth cache without its sidecar must stop the run
    at dataset construction, before a GPU-day is spent."""
    from models.spatial_spectral_classifier_aux import SpatialSpectralClassifierAux
    from training.train_torch import train_torch_model
    df = _tiny_labeled_df()
    cache = tmp_path / 'cache'; cache.mkdir(exist_ok=True)
    _labeled_cache(cache, df)
    for split in ('train', 'val'):
        k = int((df['split'] == split).sum())
        np.save(str(cache / f'mrral_{split}_patches_p7_brightness.npy'),
                np.full((k, 7, 7), 0.2, dtype=np.float32))
    out = _converted(tmp_path, 'hull', 'nosidecar')
    os.remove(synth_brightness_path(out))
    n = len(np.load(out, mmap_mode='r'))
    pq = _rows_parquet(tmp_path / 'rows.parquet', n)
    model = SpatialSpectralClassifierAux(
        n_bands=59, patch_size=7, n_classes=len(LABEL_COLS),
        embed_dim=16, n_heads=2, n_layers=1, aux_dim=1)
    with pytest.raises(FileNotFoundError, match='_brightness.npy'):
        train_torch_model(
            model=model, df=df, model_name='probe', max_epochs=1, batch_size=8,
            lr=1e-3, use_wandb=False, checkpoint_dir=None,
            mrral_map={'t0001': '/nonexistent.img'}, patch_size=7,
            cache_dir=str(cache), device='cpu', continuum_removed=True,
            cache_is_cr=True, brightness_aux=True, is_aux_model=True,
            synth_train_cache=out, synth_train_parquet=pq)


def test_a_non_brightness_run_still_gets_the_3tuple_path(tmp_path):
    """Inertness at the wiring level: without brightness_aux, train_torch must not
    ask the synth dataset for brightness (there may be no sidecar at all)."""
    out = _converted(tmp_path, 'hull', 'plain')
    os.remove(synth_brightness_path(out))
    hist = _train(tmp_path, np.load(out), continuum_removed=True, cache_is_cr=True)
    assert hist is not None


# ── 6e. the SLURM jobs must build the sidecars ──────────────────────────────

def _synth_build_block(slurm_path):
    """The `mkdir -p "$SYNTH_DIR"` ... `done` cache-build loop, verbatim."""
    lines = open(slurm_path).read().splitlines()
    i = next(i for i, l in enumerate(lines) if l.strip() == 'mkdir -p "$SYNTH_DIR"')
    j = next(j for j in range(i, len(lines)) if lines[j].strip() == 'done')
    return '\n'.join(lines[i:j + 1])


def _stub_converter(tmp_path):
    """A $PYTHON stub that logs its argv and writes patches + sidecar at --output.

    Running the REAL build loop against a stub is the point: a grep for
    '_brightness' in the file would pass on a loop that never reaches the
    converter, and the whole bug was a guard that skipped the build.
    """
    log = tmp_path / 'invocations.log'
    stub = tmp_path / 'python_stub.sh'
    stub.write_text(
        '#!/bin/bash\n'
        f'echo "$@" >> "{log}"\n'
        'out=""; prev=""\n'
        'for a in "$@"; do\n'
        '  if [ "$prev" = "--output" ]; then out="$a"; fi\n'
        '  prev="$a"\n'
        'done\n'
        '[ -n "$out" ] || exit 3\n'
        'mkdir -p "$(dirname "$out")"\n'
        'printf patches > "$out"\n'
        'printf bright > "${out%.npy}_brightness.npy"\n')
    stub.chmod(0o755)
    return str(stub), log


def _run_build_block(tmp_path, slurm_path, raw_var, out_var, preexisting):
    """Execute a job's real build loop with a stub converter.

    `preexisting` seeds ${SYNTH_DIR} before the loop runs: 'none', 'patches'
    (the state actually on the HPC today) or 'both'. Returns (rc, invocations,
    synth_dir).
    """
    synth = tmp_path / 'synth'
    raws = tmp_path / 'raws'; raws.mkdir(parents=True, exist_ok=True)
    for nm in ('train_raw.npy', 'val_raw.npy'):
        (raws / nm).write_text('raw')
    finals = {'train': 'plag_train_conv.npy', 'val': 'plag_val_conv.npy'}
    if preexisting != 'none':
        synth.mkdir(parents=True, exist_ok=True)
        for f in finals.values():
            (synth / f).write_text('stale-patches')
            if preexisting == 'both':
                (synth / f.replace('.npy', '_brightness.npy')).write_text('stale')
    stub, log = _stub_converter(tmp_path)
    preamble = '\n'.join([
        'set -o pipefail',
        f'PYTHON={stub}',
        f'SYNTH_DIR={synth}',
        f'{raw_var[0]}={raws}/train_raw.npy',
        f'{raw_var[1]}={raws}/val_raw.npy',
        f'{out_var[0]}={synth}/{finals["train"]}',
        f'{out_var[1]}={synth}/{finals["val"]}',
    ])
    p = _sp.run(['bash', '-c', preamble + '\n' + _synth_build_block(slurm_path)],
                capture_output=True, text=True, timeout=120)
    inv = log.read_text().splitlines() if log.exists() else []
    return p.returncode, inv, synth, p.stdout + p.stderr


JOBS = [
    (HANDCORE_SLURM, ('MTRDR_TRAIN_RAW', 'MTRDR_VAL_RAW'),
     ('MTRDR_TRAIN_PATCHES', 'MTRDR_VAL_PATCHES'), '--mode hull'),
    (DUALCR_SLURM, ('SYNTH_TRAIN_RAW', 'SYNTH_VAL_RAW'),
     ('SYNTH_TRAIN', 'SYNTH_VAL'), '--mode dual'),
]
JOB_IDS = ['handcore', 'dualcr']


@pytest.mark.parametrize('slurm, raw_var, out_var, mode_flag', JOBS, ids=JOB_IDS)
def test_build_loop_converts_when_nothing_is_present(tmp_path, slurm, raw_var,
                                                     out_var, mode_flag):
    rc, inv, synth, log = _run_build_block(tmp_path, slurm, raw_var, out_var, 'none')
    assert rc == 0, log
    assert len(inv) == 2, inv
    for line in inv:
        assert mode_flag in line, line
    for f in ('plag_train_conv.npy', 'plag_val_conv.npy'):
        assert (synth / f).exists(), f'{f} not installed: {log}'
        assert (synth / f.replace('.npy', '_brightness.npy')).exists(), \
            f'sidecar for {f} not installed: {log}'


@pytest.mark.parametrize('slurm, raw_var, out_var, mode_flag', JOBS, ids=JOB_IDS)
def test_build_loop_rebuilds_when_only_the_sidecar_is_missing(tmp_path, slurm,
                                                              raw_var, out_var,
                                                              mode_flag):
    """THE HPC STATE TODAY. Both jobs already have converted caches on xdisk with
    no sidecar. A patches-only existence check reports "present", skips the
    build, and the sidecar never appears -- so the job dies exactly as it did.
    """
    rc, inv, synth, log = _run_build_block(tmp_path, slurm, raw_var, out_var,
                                           'patches')
    assert rc == 0, log
    assert len(inv) == 2, (
        f'the guard skipped the rebuild although the sidecar was absent: {inv}')
    for f in ('plag_train_conv.npy', 'plag_val_conv.npy'):
        assert (synth / f.replace('.npy', '_brightness.npy')).exists(), log
        assert (synth / f).read_text() == 'patches', 'patches were not replaced'


@pytest.mark.parametrize('slurm, raw_var, out_var, mode_flag', JOBS, ids=JOB_IDS)
def test_build_loop_skips_when_both_files_are_present(tmp_path, slurm, raw_var,
                                                      out_var, mode_flag):
    """Built once and reused: a complete pair must not be rebuilt (an 8,671-patch
    hull-CR conversion is minutes of a GPU allocation, per array task)."""
    rc, inv, synth, log = _run_build_block(tmp_path, slurm, raw_var, out_var, 'both')
    assert rc == 0, log
    assert inv == [], f'converter re-ran on a complete pair: {inv}'
    assert (synth / 'plag_train_conv.npy').read_text() == 'stale-patches'


@pytest.mark.parametrize('slurm, raw_var, out_var, mode_flag', JOBS, ids=JOB_IDS)
def test_build_is_atomic_via_a_temp_path_inside_synth_dir(tmp_path, slurm, raw_var,
                                                          out_var, mode_flag):
    """hpc_finetune_handcore is --array=0-1: two tasks reach this loop at once and
    both can miss an "if absent" check. The converter must therefore write to a
    UNIQUE path and the result be renamed in, never written to the final path
    directly. The temp path must also be under ${SYNTH_DIR} -- a rename across
    filesystems is a copy, and a half-copied cache is what this prevents.
    """
    rc, inv, synth, log = _run_build_block(tmp_path, slurm, raw_var, out_var, 'none')
    assert rc == 0, log
    finals = {str(synth / 'plag_train_conv.npy'), str(synth / 'plag_val_conv.npy')}
    for line in inv:
        out = line.split('--output ')[1].split()[0]
        assert out not in finals, (
            f'converter wrote straight to the final path {out}: two concurrent '
            f'array tasks would interleave into one file')
        assert out.startswith(str(synth) + '/'), (
            f'temp output {out} is not under SYNTH_DIR; the rename would cross '
            f'filesystems and stop being atomic')
    # nothing left behind
    leftovers = [p for p in os.listdir(synth) if p.startswith('.build.')]
    assert leftovers == [], f'temp build dirs not cleaned up: {leftovers}'


def _hull_plag_preflight():
    from tests.test_dual_cr_wiring import _extract_heredocs
    blocks = [b for b in _extract_heredocs(HANDCORE_SLURM) if 'plag cache OK' in b]
    assert len(blocks) == 1, f'expected one plag preflight, got {len(blocks)}'
    return blocks[0]


@pytest.mark.parametrize('mode, preflight', [('hull', _hull_plag_preflight),
                                             ('dual', _plag_preflight)],
                         ids=['handcore', 'dualcr'])
def test_plag_preflight_accepts_a_converted_cache_with_its_sidecar(tmp_path, mode,
                                                                   preflight):
    out = _converted(tmp_path, mode, 'pf')
    n = len(np.load(out, mmap_mode='r'))
    pq = _rows_parquet(tmp_path / 'pf.parquet', n)
    rc, log = _run_preflight(preflight(), [out, pq, out, pq])
    assert rc == 0, f'preflight rejected a correct cache + sidecar:\n{log}'
    assert 'brightness sidecar' in log, log


@pytest.mark.parametrize('mode, preflight', [('hull', _hull_plag_preflight),
                                             ('dual', _plag_preflight)],
                         ids=['handcore', 'dualcr'])
def test_plag_preflight_rejects_a_cache_with_no_sidecar(tmp_path, mode, preflight):
    """The pre-2026-08-11 caches on the HPC. Both jobs run --brightness_aux, so
    this must stop the job before SLURM hands over a GPU."""
    out = _converted(tmp_path, mode, 'pf')
    os.remove(synth_brightness_path(out))
    n = len(np.load(out, mmap_mode='r'))
    pq = _rows_parquet(tmp_path / 'pf.parquet', n)
    rc, log = _run_preflight(preflight(), [out, pq, out, pq])
    assert rc != 0, f'preflight accepted a cache with no sidecar:\n{log}'
    assert 'brightness sidecar' in log, log


@pytest.mark.parametrize('mode, preflight', [('hull', _hull_plag_preflight),
                                             ('dual', _plag_preflight)],
                         ids=['handcore', 'dualcr'])
def test_plag_preflight_rejects_a_sidecar_with_the_wrong_row_count(tmp_path, mode,
                                                                   preflight):
    """A sidecar left over from a different cache has the right NAME and the wrong
    rows; SyntheticPatchDataset would then serve mismatched brightness."""
    out = _converted(tmp_path, mode, 'pf')
    n = len(np.load(out, mmap_mode='r'))
    np.save(synth_brightness_path(out),
            np.zeros((n - 1, 7, 7), dtype=np.float32))
    pq = _rows_parquet(tmp_path / 'pf.parquet', n)
    rc, log = _run_preflight(preflight(), [out, pq, out, pq])
    assert rc != 0, f'preflight accepted a short sidecar:\n{log}'


def test_handcore_slurm_keeps_every_data_path_on_xdisk():
    """/groups filled up and killed two cache builds with Errno 28; the new temp
    build dir must not reintroduce one."""
    for line in open(HANDCORE_SLURM).read().splitlines():
        s = line.strip()
        if not s or s.startswith('#') or '/groups' not in s:
            continue
        assert s.startswith('WORK_DIR=') or s.startswith('PYTHON='), \
            f'data path on /groups: {s}'
    assert '/tmp' not in _synth_build_block(HANDCORE_SLURM), \
        'the atomic build must stage inside ${SYNTH_DIR} on xdisk, not /tmp'
    assert '/tmp' not in _synth_build_block(DUALCR_SLURM)

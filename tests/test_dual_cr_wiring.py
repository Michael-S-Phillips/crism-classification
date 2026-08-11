"""Task 7: --dual_cr wiring through scripts/train.py and training/train_torch.py.

The dual-CR experiment tests whether upper-hull CR's flattening of alteration's
1-2 um arch is what makes alteration a false attractor. A wiring bug that quietly
feeds the wrong representation produces a plausible null result and the hypothesis
reads as falsified. So every guard here exists to make a mis-wired run FAIL rather
than run.

Precedent: commit 07547e8. argparse accepted --synth_*, one model branch didn't
forward them, and a run completed "successfully" having injected zero of the MTRDR
plagioclase it was configured to inject. An accepted-but-ignored flag is the worst
outcome available.
"""
from __future__ import annotations

import ast
import inspect
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

TRAIN_PY = os.path.join(ROOT, 'scripts', 'train.py')

# The commit this task branched from. Used only by the BASE-comparison test,
# which skips (rather than fails) if the object is not reachable.
BASE_SHA = '23d2296'


def _parse_args_in_subprocess(argv):
    """Run scripts/train.build_args() with argv; return (rc, stdout, stderr).

    build_args() is where every parser.error guard lives, and it does no I/O, so
    this exercises the guards without a config, a parquet or a GPU.
    """
    code = ("import sys; sys.argv=['train.py']+%r; "
            "import scripts.train as t; a=t.build_args(); "
            "print('PARSED_OK', a.dual_cr)" % (argv,))
    p = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                       capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


# ── Guards ───────────────────────────────────────────────────────────────────

def test_dual_cr_requires_continuum_removed():
    """--dual_cr without --continuum_removed must be a parse error.

    dual_continuum() is only reachable through the continuum-removal branch of
    CRISMSpectralPatchDataset; accepting --dual_cr alone would serve raw patches
    to a 118-channel encoder.
    """
    rc, out, err = _parse_args_in_subprocess(
        ['--dual_cr', '--model', 'spatial_vit', '--mrral_parquets', '_'])
    assert rc != 0, f'--dual_cr alone was accepted:\n{out}\n{err}'
    assert '--dual_cr requires --continuum_removed' in err, err


def test_dual_cr_accepted_for_the_two_supported_models():
    """The wired branches must actually accept the flag (guard not over-broad)."""
    for model in ('spatial_vit',):
        rc, out, err = _parse_args_in_subprocess(
            ['--dual_cr', '--continuum_removed', '--model', model,
             '--mrral_parquets', '_'])
        assert rc == 0, f'{model} rejected a valid --dual_cr:\n{out}\n{err}'
        assert 'PARSED_OK True' in out, out
    # spatial_vit_aux additionally needs an aux source, but that check lives in
    # main(), not build_args(), so parsing alone must still succeed.
    rc, out, err = _parse_args_in_subprocess(
        ['--dual_cr', '--continuum_removed', '--model', 'spatial_vit_aux',
         '--brightness_aux', '--mrral_parquets', '_'])
    assert rc == 0, err
    assert 'PARSED_OK True' in out, out


@pytest.mark.parametrize('model', [
    'decomp_spatial_vit',    # DecompSpVit: reconstructs a physical raw-space
    'decomp_spatial_vit_adv',  # decomposition; neither forwards continuum_removed
    'spectral_vit',          # 59-band pixel models, no patch dataset
    'spectral_cnn',
    'spectral_hybrid',
    'cnn',                   # n_bands=60 mrrsu params
    'vit',
    'mlp',
    'rf',                    # sklearn: no encoder at all
])
def test_dual_cr_rejected_for_unsupported_models(model):
    """Every branch that cannot honour --dual_cr must refuse it at parse time.

    These branches either build a non-SpatialSpectral encoder or never forward
    continuum_removed to train_torch_model, so accepting the flag would silently
    train the 59-band representation while the command line claimed otherwise.
    """
    rc, out, err = _parse_args_in_subprocess(
        ['--dual_cr', '--continuum_removed', '--model', model,
         '--mrral_parquets', '_'])
    assert rc != 0, f'--model {model} accepted --dual_cr and would ignore it:\n{out}'
    assert '--dual_cr is not supported by --model' in err, err


def test_dual_cr_rejects_synthetic_patch_mixing():
    """--dual_cr + any --synth_* must refuse: those caches are 59-channel.

    SyntheticPatchDataset (data/dataset.py) reads a fixed-width .npy written by
    build_mtrdr_plag_patches.py at 59 channels. ConcatDataset would happily mix
    it with 118-channel patches and the collate would blow up mid-epoch at best,
    or silently under-represent plagioclase at worst.
    """
    for flag in ('--synth_train_cache', '--synth_train_parquet',
                 '--synth_val_cache', '--synth_val_parquet'):
        rc, out, err = _parse_args_in_subprocess(
            ['--dual_cr', '--continuum_removed', '--model', 'spatial_vit',
             '--mrral_parquets', '_', flag, 'x'])
        assert rc != 0, f'{flag} was accepted alongside --dual_cr:\n{out}'
        assert '--dual_cr is incompatible with --synth_*' in err, err


# ── Source-level wiring ──────────────────────────────────────────────────────

def _call_kwargs(src, func_name):
    """[{kwarg: source_text}] for every call to func_name in src, via AST."""
    tree = ast.parse(src)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = getattr(f, 'id', None) or getattr(f, 'attr', None)
        if name != func_name:
            continue
        out.append({kw.arg: ast.unparse(kw.value)
                    for kw in node.keywords if kw.arg})
    return out


def test_both_spatial_branches_wire_n_bands_and_dual_cr():
    """The two SpatialSpectral* constructions must take n_bands from --dual_cr,
    and both their train_torch_model call sites must forward dual_cr.

    Same failure shape as the synth_* audit: a hardcoded n_bands=59 next to an
    accepted --dual_cr builds a 59-channel encoder for 118-channel patches.
    """
    src = open(TRAIN_PY).read()

    for cls in ('SpatialSpectralClassifier', 'SpatialSpectralClassifierAux'):
        calls = _call_kwargs(src, cls)
        assert calls, f'no {cls}(...) construction found'
        for kw in calls:
            assert 'n_bands' in kw, f'{cls} built without an explicit n_bands'
            assert kw['n_bands'] != '59', (
                f'{cls} still hardcodes n_bands=59; --dual_cr would be accepted '
                f'and then ignored')
            assert 'dual_cr' in kw['n_bands'], (
                f'{cls} n_bands={kw["n_bands"]!r} does not depend on dual_cr')

    # Both patch-based call sites forward dual_cr.
    forwarding = [kw for kw in _call_kwargs(src, 'train_torch_model')
                  if 'continuum_removed' in kw]
    assert len(forwarding) >= 2, (
        'expected the spatial_vit and spatial_vit_aux call sites to forward '
        'continuum_removed')
    for kw in forwarding:
        assert kw.get('dual_cr') == 'args.dual_cr', (
            'a call site forwards continuum_removed but not dual_cr=args.dual_cr')


def test_args_dual_cr_is_only_read_in_sanctioned_places():
    """`args.dual_cr` may appear ONLY as a guard, as `118 if ... else 59`, or as
    `dual_cr=args.dual_cr`.

    This is what makes the 59-band parity argument exact rather than statistical:
    if every read of args.dual_cr is one of these three forms, then dual_cr=False
    reduces each of them to the pre-existing literal (59) or to the dataset's own
    default (False), so nothing downstream can differ from BASE.
    """
    src = open(TRAIN_PY).read()
    tree = ast.parse(src)

    # Collect the enclosing statement source for every args.dual_cr read.
    reads = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == 'dual_cr' \
                and isinstance(node.value, ast.Name) and node.value.id == 'args':
            reads.append(node)
    assert reads, 'args.dual_cr is never read — the flag does nothing'

    # Sanctioned forms, checked structurally over the whole file.
    guards = [n for n in ast.walk(tree)
              if isinstance(n, ast.If)
              and 'args.dual_cr' in ast.unparse(n.test)]
    ternaries = [n for n in ast.walk(tree)
                 if isinstance(n, ast.IfExp)
                 and ast.unparse(n.test) == 'args.dual_cr'
                 and ast.unparse(n.body) == '118'
                 and ast.unparse(n.orelse) == '59']
    forwards = [n for n in ast.walk(tree)
                if isinstance(n, ast.keyword) and n.arg == 'dual_cr'
                and ast.unparse(n.value) == 'args.dual_cr']

    accounted = set()
    for group in (guards, ternaries, forwards):
        for n in group:
            for sub in ast.walk(n):
                if sub in reads:
                    accounted.add(id(sub))
    stray = [ast.unparse(n) for n in reads if id(n) not in accounted]
    assert not stray, (
        f'args.dual_cr read outside a guard / "118 if args.dual_cr else 59" / '
        f'dual_cr=args.dual_cr: {stray}')
    assert ternaries, 'no "118 if args.dual_cr else 59" n_bands expression found'
    assert len(forwards) >= 2, 'fewer than two dual_cr=args.dual_cr forwards'


# ── train_torch.py → dataset ─────────────────────────────────────────────────

def _tiny_patch_df(n=24):
    from data.dataset import LABEL_COLS
    d = {'tile_id': ['t0001'] * n,
         'polygon_id': list(range(n)),
         'pixel_row': [3] * n, 'pixel_col': [3] * n,
         'confidence_weight': np.ones(n, dtype=np.float32),
         'confidence_tier': ['High'] * n,
         'split': ['train'] * (n // 2) + ['val'] * (n - n // 2)}
    for c in ('olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other'):
        d[c] = np.zeros(n, dtype=np.float32)
    d['lcp'] = np.ones(n, dtype=np.float32)
    return pd.DataFrame(d)


def _run_capture(tmp_path, n_channels, **extra):
    """Run one epoch of train_torch_model over a memmap cache, capturing every
    kwarg that reaches CRISMSpectralPatchDataset."""
    import torch
    import data.dataset as dsmod
    from training.train_torch import train_torch_model
    from models.spatial_spectral_transformer import SpatialSpectralClassifier

    df = _tiny_patch_df()
    captured = []
    real = dsmod.CRISMSpectralPatchDataset

    class Recorder(real):
        def __init__(self, *a, **kw):
            captured.append(dict(kw))
            super().__init__(*a, **kw)

    for split in ('train', 'val'):
        sub = df[df['split'] == split]
        fp = np.memmap(str(tmp_path / f'mrral_{split}_patches_p7.npy'),
                       dtype='float32', mode='w+',
                       shape=(len(sub), 7, 7, n_channels))
        fp[:] = np.random.default_rng(0).uniform(
            0.05, 0.35, size=(len(sub), 7, 7, n_channels)).astype(np.float32)
        fp.flush()
        del fp

    dsmod.CRISMSpectralPatchDataset = Recorder
    try:
        model = SpatialSpectralClassifier(
            n_bands=n_channels, patch_size=7, n_classes=len(dsmod.LABEL_COLS),
            embed_dim=16, n_heads=2, n_layers=1)
        train_torch_model(
            model=model, df=df, model_name='dual_cr_capture',
            max_epochs=1, batch_size=8, lr=1e-3, use_wandb=False,
            checkpoint_dir=None, mrral_map={'t0001': '/nonexistent.img'},
            patch_size=7, cache_dir=str(tmp_path), device='cpu', **extra)
    finally:
        dsmod.CRISMSpectralPatchDataset = real
    return captured


def test_train_torch_forwards_dual_cr_to_the_dataset(tmp_path):
    """dual_cr=True must reach CRISMSpectralPatchDataset, which then serves 118
    channels. Without the forward the dataset defaults to 59 and the byte-size
    guard rejects the dual cache — or worse, a 59-channel cache is accepted and
    the dual run silently trains hull-only."""
    cap = _run_capture(tmp_path, 118, continuum_removed=True,
                       cache_is_cr=True, dual_cr=True)
    assert cap, 'CRISMSpectralPatchDataset was never constructed'
    for kw in cap:
        assert kw.get('dual_cr') is True, (
            f'dual_cr did not reach the dataset: {kw}')


def test_59band_argument_parity_when_dual_cr_absent(tmp_path):
    """With dual_cr left at its default, the kwargs reaching
    CRISMSpectralPatchDataset are exactly BASE's.

    BASE (23d2296) passed:
        patch_size, cache_dir, split, continuum_removed, return_brightness,
        cache_is_cr
    and nothing else. The only addition is dual_cr, whose value here must be the
    class's own default, so the call is indistinguishable from BASE's.

    This replaces the brief's one-epoch loss comparison, which needs HPC data.
    It is stronger: exact rather than statistical.
    """
    BASE_KWARGS = {'patch_size', 'cache_dir', 'split', 'continuum_removed',
                   'return_brightness', 'cache_is_cr'}
    from data.dataset import CRISMSpectralPatchDataset
    default_dual = inspect.signature(
        CRISMSpectralPatchDataset.__init__).parameters['dual_cr'].default
    assert default_dual is False, 'dual_cr default is not inert'

    cap = _run_capture(tmp_path, 59)
    assert cap, 'CRISMSpectralPatchDataset was never constructed'
    for kw in cap:
        added = set(kw) - BASE_KWARGS
        assert added == {'dual_cr'}, (
            f'kwargs reaching the dataset changed beyond dual_cr: added={added}')
        assert kw['dual_cr'] == default_dual, (
            f'dual_cr={kw["dual_cr"]!r} is not the inert default')
        assert kw['continuum_removed'] is False
        assert kw['return_brightness'] is False
        assert kw['cache_is_cr'] is False
        assert kw['patch_size'] == 7


def test_train_py_kwargs_are_identical_to_base_except_dual_cr():
    """Byte-compare every model-construction and train_torch_model kwarg in
    scripts/train.py against BASE, allowing only the sanctioned dual-CR edits.

    Catches an accidental change to a shared 59-band value that would invalidate
    any comparison against ft_7cls_handcore_level.
    """
    probe = subprocess.run(['git', 'cat-file', '-e', f'{BASE_SHA}:scripts/train.py'],
                           cwd=ROOT, capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip(f'BASE {BASE_SHA} not reachable from this checkout')
    base = subprocess.run(['git', 'show', f'{BASE_SHA}:scripts/train.py'],
                          cwd=ROOT, capture_output=True, text=True).stdout
    head = open(TRAIN_PY).read()

    CONSTRUCTORS = ('train_torch_model', 'SpatialSpectralClassifier',
                    'SpatialSpectralClassifierAux', 'DecompSpVit',
                    'DecompSpVitAdv', 'SpectralTransformer', 'SpectralCNN1D',
                    'SpectralViT', 'SpectralSpatialCNN', 'MLP',
                    'SpectralHybridClassifier')
    ALLOWED_NEW = {'dual_cr': 'args.dual_cr'}
    ALLOWED_CHANGED = {'n_bands': ('59', '118 if args.dual_cr else 59')}

    for fn in CONSTRUCTORS:
        b_calls, h_calls = _call_kwargs(base, fn), _call_kwargs(head, fn)
        assert len(b_calls) == len(h_calls), (
            f'{fn}: BASE has {len(b_calls)} call sites, HEAD has {len(h_calls)}')
        for i, (bkw, hkw) in enumerate(zip(b_calls, h_calls)):
            for k in set(hkw) - set(bkw):
                assert k in ALLOWED_NEW and hkw[k] == ALLOWED_NEW[k], (
                    f'{fn} call {i}: unsanctioned new kwarg {k}={hkw[k]!r}')
            assert not (set(bkw) - set(hkw)), (
                f'{fn} call {i}: kwargs dropped vs BASE: {set(bkw) - set(hkw)}')
            for k in set(bkw) & set(hkw):
                if bkw[k] == hkw[k]:
                    continue
                allowed = ALLOWED_CHANGED.get(k)
                assert allowed and (bkw[k], hkw[k]) == allowed, (
                    f'{fn} call {i}: {k} changed {bkw[k]!r} -> {hkw[k]!r} '
                    f'outside the sanctioned dual-CR edit')


# ── pretrain guard ───────────────────────────────────────────────────────────

def test_pretrain_denoising_118_requires_continuum_removed():
    """--n_bands 118 without --continuum_removed leaves normalize=True, which
    per-patch z-scores the dual patch.

    The two blocks sit ~7.5x apart in mean; that offset dominates the patch std,
    so a per-patch z-score shrinks the real within-block signal roughly 4x. The
    run would look fine and learn almost nothing.
    """
    p = subprocess.run(
        [sys.executable, 'scripts/pretrain_spatial_mae_denoising.py',
         '--n_bands', '118', '--n_channel_blocks', '2'],
        cwd=ROOT, capture_output=True, text=True)
    assert p.returncode != 0, f'accepted 118 bands without CR:\n{p.stdout}'
    assert '--n_bands 118 requires --continuum_removed' in p.stderr, p.stderr


def test_pretrain_denoising_59band_defaults_still_parse():
    """The existing 59-band pretrain invocation must be untouched by the guard."""
    p = subprocess.run(
        [sys.executable, 'scripts/pretrain_spatial_mae_denoising.py', '--help'],
        cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert '--n_bands' in p.stdout

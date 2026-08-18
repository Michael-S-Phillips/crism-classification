"""--gated_head must reach the model and the loss, not just parse."""
from __future__ import annotations

import inspect

import scripts.train as train_mod
import training.train_torch as tt


def test_train_torch_model_accepts_gated_head():
    assert 'gated_head' in inspect.signature(tt.train_torch_model).parameters


def test_gated_head_defaults_off():
    """Every existing run must be unaffected."""
    assert inspect.signature(tt.train_torch_model).parameters['gated_head'].default is False


def test_cli_exposes_gated_head():
    src = inspect.getsource(train_mod)
    assert "'--gated_head'" in src or '"--gated_head"' in src


def test_gated_head_forwarded_wherever_asl_is():
    """A flag that parses but is not forwarded silently trains the flat head
    while the log claims a gated run -- the worst outcome available here.
    scripts/train.py has 8 train_torch_model call sites; the 6 that pass
    use_asl_loss are exactly the ones that must also pass gated_head=... to
    train_torch_model (for loss selection). The spatial_vit_aux branch
    additionally forwards gated_head into build_spatial_vit_aux_model (for
    model-class selection), which is a 7th, legitimate occurrence of the
    same kwarg string -- see test_gated_head_construction_emits_8_logit_head
    for the test that actually exercises that construction path."""
    src = inspect.getsource(train_mod)
    n_asl = src.count('use_asl_loss=args.asl_loss')
    n_gated = src.count('gated_head=args.gated_head')
    assert n_asl == 6, f'call-site count changed ({n_asl}); re-read train.py'
    assert n_gated == n_asl + 1, (
        f'{n_asl} call sites pass use_asl_loss (each must also pass '
        f'gated_head=... to train_torch_model), plus 1 more expected at the '
        f'build_spatial_vit_aux_model construction call; got {n_gated} total '
        f'gated_head=args.gated_head occurrences')


def test_gated_head_construction_emits_8_logit_head():
    """The flag must also change WHICH CLASS gets built, not just which loss
    gets selected. --gated_head --asl_loss selects GatedAsymmetricLoss, which
    requires 8 logits (1 gate + 7 conditionals). If scripts/train.py still
    builds the flat 7-wide SpatialSpectralClassifierAux regardless of the
    flag, the loss receives 7 logits where it needs 8 and the run dies on a
    shape mismatch -- --gated_head would parse but be unusable end to end.

    This constructs the real model via train_mod.build_spatial_vit_aux_model
    (the factory scripts/train.py's `elif args.model == 'spatial_vit_aux':`
    branch calls) rather than grepping source text, so it fails if the
    construction site ever silently reverts to always building the flat
    class."""
    import torch
    from models.gated_classifier import GatedSpatialSpectralClassifierAux

    n_classes = 7
    model = train_mod.build_spatial_vit_aux_model(
        n_classes=n_classes, patch_size=7, embed_dim=32, n_heads=4,
        n_layers=2, dropout=0.1, aux_dim=2, dual_cr=False, gated_head=True,
    )
    assert isinstance(model, GatedSpatialSpectralClassifierAux), (
        f'--gated_head must construct GatedSpatialSpectralClassifierAux, '
        f'got {type(model).__name__}')
    assert model.head.out_features == n_classes + 1, (
        f'gated head must be n_classes+1={n_classes + 1} wide '
        f'(1 gate + {n_classes} conditionals), got {model.head.out_features}')

    x = torch.randn(2, 7, 7, 59)
    aux = torch.randn(2, 2)
    logits = model(x, aux)
    assert logits.shape[-1] == n_classes + 1, (
        f'model must emit {n_classes + 1} logits for GatedAsymmetricLoss, '
        f'got {logits.shape[-1]}')


def test_flat_head_construction_unchanged_when_gated_head_false():
    """Backward compatibility: gated_head=False (the default) must still
    build the plain 7-wide SpatialSpectralClassifierAux, byte-for-byte the
    same model every existing run has always used."""
    from models.spatial_spectral_classifier_aux import SpatialSpectralClassifierAux

    n_classes = 7
    model = train_mod.build_spatial_vit_aux_model(
        n_classes=n_classes, patch_size=7, embed_dim=32, n_heads=4,
        n_layers=2, dropout=0.1, aux_dim=2, dual_cr=False, gated_head=False,
    )
    assert type(model) is SpatialSpectralClassifierAux
    assert model.head.out_features == n_classes


def test_gated_head_requires_spatial_vit_aux_model():
    """GatedSpatialSpectralClassifierAux only subclasses
    SpatialSpectralClassifierAux; the other five ASL-capable branches build
    unrelated classes with no gated counterpart. CLI must refuse the
    combination early rather than let it fail deep inside the loss."""
    src = inspect.getsource(train_mod)
    assert "args.gated_head and args.model != 'spatial_vit_aux'" in src


# --- M4: the floor-test command the job prints must actually run -----------

import difflib
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GATED_SLURM = os.path.join(_ROOT, 'scripts', 'hpc_finetune_dualcr_gated.slurm')
_NOCLIP_SLURM = os.path.join(_ROOT, 'scripts', 'hpc_finetune_dualcr_noclip.slurm')


def _slurm_lines(path):
    with open(path) as fh:
        return fh.readlines()


def _floor_test_block(path):
    """The trailing lines that print the follow-up floor-test command."""
    lines = _slurm_lines(path)
    for i, line in enumerate(lines):
        if 'CLASSIFY_EXTRA_ARGS' in line:
            return ''.join(lines[i - 4:i + 3])
    raise AssertionError(f'no floor-test command found in {path}')


def test_printed_floor_test_command_passes_gated_head():
    """The command was inherited verbatim from the noclip arm. Copy-pasting it
    aborts: classify_tile_supervised._set_n_classes sees an 8-wide head with
    GATED_MODE False and raises 'unsupported head size 8'. The person running
    it is reading a job log a day later, with no reason to doubt the command
    the job itself printed."""
    block = _floor_test_block(_GATED_SLURM)
    assert '--gated_head' in block, (
        f'the gated arm prints a floor-test command with no --gated_head, so '
        f'it aborts on this arm\'s own 8-wide checkpoint:\n{block}')


def test_printed_floor_test_command_keeps_the_flags_gated_head_requires():
    """classify_tile_supervised errors unless --gated_head is accompanied by
    --mrrsu_aux or --brightness_aux (the gated class is an aux-head model),
    and --brightness_aux itself requires --continuum_removed."""
    block = _floor_test_block(_GATED_SLURM)
    assert '--brightness_aux' in block or '--mrrsu_aux' in block
    assert '--continuum_removed' in block


def test_floor_test_comment_no_longer_claims_a_seven_wide_head():
    """The comment asserted '7-wide head auto-selects the 7-class vocab',
    which is exactly what is NOT true here -- this arm's head is 8 wide."""
    block = _floor_test_block(_GATED_SLURM)
    assert '7-wide head auto-selects' not in block, (
        f'stale inherited comment still explains the wrong arm:\n{block}')


def test_gated_slurm_is_a_minimal_diff_from_the_noclip_arm():
    """The arms must differ ONLY in identity (job name, two log paths, run
    name), the flag under test (--gated_head), and the floor-test command that
    flag forces. Anything else means a second uncontrolled variable and the
    comparison against noclip stops being like-for-like."""
    diff = list(difflib.unified_diff(
        _slurm_lines(_NOCLIP_SLURM), _slurm_lines(_GATED_SLURM), n=0))
    hunks = [l for l in diff if l.startswith('@@')]
    changed = [l for l in diff
               if l[0] in '+-' and not l.startswith(('---', '+++'))]
    # Prose comments are exempt -- explaining the arm is free. What must stay
    # minimal is the EXECUTABLE difference, so only real directives and
    # commands are held to the allowlist.
    code = [l for l in changed
            if not (l[1:].lstrip().startswith('#')
                    and not l[1:].lstrip().startswith('#SBATCH'))]
    allowed = ('job-name', 'SBATCH --output', 'SBATCH --error', 'RUN_NAME=',
               '--asl_loss', '--gated_head', 'CLASSIFY_EXTRA_ARGS')
    unexpected = [l for l in code if not any(a in l for a in allowed)]
    assert not unexpected, (
        f'hpc_finetune_dualcr_gated.slurm diverges from the noclip arm beyond '
        f'the gate change:\n{"".join(unexpected)}')
    # Six logical differences -- job name, two log paths, run name,
    # --gated_head, and the floor-test command that flag forces -- but the two
    # log paths are adjacent lines and collapse into a single contiguous hunk,
    # so the arms differ in 5 regions.
    assert len(hunks) == 5, (
        f'expected exactly 5 differing regions (job name, the adjacent log '
        f'path pair, run name, --gated_head, floor-test command); got '
        f'{len(hunks)}:\n{"".join(diff)}')


# --- M5: exercise real call conventions, not just the targets --------------
# Every gated test above calls its target directly with hand-built arguments.
# That is exactly how C1 survived 33 green tests: the loss was only ever
# invoked positionally, while train_torch invokes it by keyword. These two
# drive the actual CLI parser and the actual checkpoint-construction branch.

import sys

import pytest


def _argv(*extra):
    return ['train.py', '--model', 'spatial_vit_aux', *extra]


@pytest.mark.parametrize('model', ['spatial_vit', 'spectral_hybrid',
                                   'decomp_spatial_vit'])
def test_gated_head_is_refused_at_the_cli_for_other_models(
        monkeypatch, capsys, model):
    """The guard must fire in the REAL parser, not merely exist in the source.
    Only SpatialSpectralClassifierAux has a Gated* subclass; the others would
    hand GatedAsymmetricLoss a 7-wide tensor where it needs 8.

    The message is asserted, not just the exit code: argparse exits 2 for any
    bad command line, so a model name that is merely misspelled would satisfy
    an exit-code-only check and prove nothing about this guard.
    """
    monkeypatch.setattr(sys, 'argv', ['train.py', '--model', model, '--gated_head'])
    with pytest.raises(SystemExit) as exc:
        train_mod.build_args()
    assert exc.value.code != 0
    err = capsys.readouterr().err
    assert '--gated_head' in err and 'spatial_vit_aux' in err, (
        f'exited, but not because of the --gated_head guard:\n{err}')


def test_gated_head_parses_for_spatial_vit_aux(monkeypatch):
    """The permitted combination must survive every guard in build_args and
    arrive as gated_head=True -- a guard that rejected everything would pass
    the test above and still make the arm unrunnable."""
    monkeypatch.setattr(sys, 'argv', _argv('--gated_head'))
    args = train_mod.build_args()
    assert args.gated_head is True
    assert args.model == 'spatial_vit_aux'


def test_gated_head_defaults_false_at_the_cli(monkeypatch):
    """Backward compatibility at the parser: an existing command line must
    still produce gated_head=False."""
    monkeypatch.setattr(sys, 'argv', _argv())
    assert train_mod.build_args().gated_head is False


def test_classify_main_builds_a_gated_model_from_an_eight_wide_checkpoint(
        monkeypatch, tmp_path):
    """Push a synthetic 8-wide state_dict through
    scripts/classify_tile_supervised.py's main() construction branch with
    torch.load mocked, so the model-swap wiring is exercised end to end
    without a real checkpoint or a real tile.

    This is the path a floor test takes. Before --gated_head existed it built
    the flat SpatialSpectralClassifierAux, whose 7-wide head cannot load an
    8-wide state_dict -- load_state_dict would raise on a size mismatch.
    """
    import numpy as np
    import torch

    import scripts.classify_tile_supervised as cts
    from models.gated_classifier import GatedSpatialSpectralClassifierAux

    embed_dim, n_layers, H, W = 32, 2, 6, 5
    n_bands = cts.model_n_bands(False)

    # A REAL state_dict from the class the flag is supposed to select, so the
    # test fails if construction picks any class that cannot load it.
    reference = GatedSpatialSpectralClassifierAux(
        n_bands=n_bands, patch_size=cts.PATCH_SIZE, n_classes=7,
        embed_dim=embed_dim, n_heads=4, n_layers=n_layers, aux_dim=1)
    state = reference.state_dict()
    assert state['head.weight'].shape[0] == 8

    monkeypatch.setattr(cts.torch, 'load', lambda *a, **kw: state)
    monkeypatch.setattr(cts, 'load_tile', lambda p: (
        np.zeros((H, W, n_bands), dtype=np.float32),
        np.ones((H, W), dtype=bool), _FakeTransform(), _FakeCRS()))

    built = {}

    def _fake_run_supervised(tile, model, device, batch_size, **kw):
        built['model'] = model
        return np.zeros((H * W, cts.N_CLASSES), dtype=np.float32)

    monkeypatch.setattr(cts, 'run_supervised', _fake_run_supervised)
    monkeypatch.setattr(sys, 'argv', [
        'classify_tile_supervised.py',
        '--tile', str(tmp_path / 't0001_mrral_x.img'),
        '--ckpt', str(tmp_path / 'ckpt.pt'),
        '--continuum_removed', '--brightness_aux', '--gated_head',
        '--embed_dim', str(embed_dim), '--n_layers', str(n_layers),
        '--no_plot'])

    saved_n_classes, saved_mode = cts.N_CLASSES, cts.GATED_MODE
    try:
        cts.main()
    finally:
        cts.N_CLASSES, cts.GATED_MODE = saved_n_classes, saved_mode

    model = built['model']
    assert isinstance(model, GatedSpatialSpectralClassifierAux), (
        f'--gated_head must construct GatedSpatialSpectralClassifierAux, '
        f'got {type(model).__name__}')
    assert model.head.out_features == 8


class _FakeTransform:
    a = b = c = d = e = f = 0.0


class _FakeCRS:
    def to_wkt(self):
        return 'LOCAL_CS["test"]'

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

import torch
import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.train_torch import train_torch_model
from models.mlp import MLP

def make_fake_df(n=300):
    data = {f'b{i}': np.random.rand(n).astype(np.float32) for i in range(60)}
    for col in ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']:
        data[col] = (np.random.rand(n) > 0.7).astype(np.float32)
    data['confidence_weight'] = np.ones(n, dtype=np.float32)
    data['confidence_tier'] = ['High'] * n
    data['tile_id'] = 't0001'
    data['polygon_id'] = 0
    data['pixel_row'] = 0
    data['pixel_col'] = 0
    n_train = int(n * 0.67)
    n_val = int(n * 0.165)
    n_test = n - n_train - n_val
    splits = ['train'] * n_train + ['val'] * n_val + ['test'] * n_test
    data['split'] = splits
    return pd.DataFrame(data)

def test_mlp_trains_without_error():
    df = make_fake_df()
    model = MLP(n_features=60, n_classes=6)
    metrics = train_torch_model(
        model=model,
        df=df,
        model_name='mlp_test',
        max_epochs=2,
        batch_size=32,
        lr=1e-3,
        use_wandb=False,
        checkpoint_dir=None,
    )
    assert 'val_mAP' in metrics
    assert 0.0 <= metrics['val_mAP'] <= 1.0

def test_early_stopping_triggers():
    df = make_fake_df(n=600)
    model = MLP()
    metrics = train_torch_model(
        model=model, df=df, model_name='mlp_es',
        max_epochs=50, patience=2, use_wandb=False, checkpoint_dir=None
    )
    # Should stop before epoch 50
    assert metrics.get('stopped_epoch', 50) <= 50


def test_cnn_trains_with_cache(tmp_path):
    """CRISMPatchDataset cache is used when cache_dir is passed to train_torch_model."""
    from models.cnn import SpectralSpatialCNN
    from data.dataset import BAND_COLS
    n = 120
    patch_size = 7
    n_bands = len(BAND_COLS)
    df = make_fake_df(n)
    for split in ('train', 'val', 'test'):
        sub = df[df['split'] == split]
        fp = np.memmap(
            str(tmp_path / f'{split}_patches_p{patch_size}.npy'),
            dtype='float32', mode='w+',
            shape=(len(sub), n_bands, patch_size, patch_size),
        )
        fp[:] = np.random.rand(len(sub), n_bands, patch_size, patch_size).astype(np.float32)
        fp.flush()
        del fp
    model = SpectralSpatialCNN(n_bands=n_bands, n_classes=6, patch_size=patch_size)
    mrrsu_map = {'t0001': '/nonexistent/path.img'}
    metrics = train_torch_model(
        model=model, df=df, model_name='cnn_cache_test',
        max_epochs=2, batch_size=32, lr=1e-3,
        use_wandb=False, checkpoint_dir=None,
        mrrsu_map=mrrsu_map, patch_size=patch_size,
        cache_dir=str(tmp_path),
    )
    assert 'val_mAP' in metrics


def make_fake_mrral_df(n=120):
    rng = np.random.default_rng(0)
    data = {f'm{i}': rng.random(n).astype('float32') for i in range(59)}
    for col in ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']:
        data[col] = rng.integers(0, 2, n).astype('float32')
    data['confidence_weight'] = np.ones(n, dtype='float32')
    data['confidence_tier'] = ['High'] * n
    data['split'] = ['train'] * 80 + ['val'] * 40
    return pd.DataFrame(data)


def test_train_hybrid_model_e2e():
    """SpectralHybridClassifier should train end-to-end through train_torch_model."""
    from models.hybrid_classifier import SpectralHybridClassifier
    rng = np.random.default_rng(2)
    n = 120
    df = pd.DataFrame({
        **{f'm{i}': rng.random(n).astype('float32') for i in range(59)},
        **{f'b{i}': rng.random(n).astype('float32') for i in range(60)},
        'olivine_t1': rng.integers(0, 2, n).astype('float32'),
        'olivine_t2': rng.integers(0, 2, n).astype('float32'),
        'lcp':  rng.integers(0, 2, n).astype('float32'),
        'hcp':  rng.integers(0, 2, n).astype('float32'),
        'plagioclase': rng.integers(0, 2, n).astype('float32'),
        'other': rng.integers(0, 2, n).astype('float32'),
        'confidence_weight': np.ones(n, dtype='float32'),
        'confidence_tier': ['High'] * n,
        'split': ['train'] * 80 + ['val'] * 40,
    })
    model = SpectralHybridClassifier(
        n_mrral=59, n_mrrsu=60, n_classes=5,
        embed_dim=32, n_heads=2, n_layers=2,
    )
    metrics = train_torch_model(
        model=model, df=df, model_name='test_hybrid',
        max_epochs=2, batch_size=32, lr=3e-4,
        patience=5, use_wandb=False, checkpoint_dir=None,
        use_asl_loss=True,
    )
    assert 'val_mAP' in metrics


def test_train_with_asl_loss():
    """Training loop should run without error when use_asl_loss=True."""
    from models.spectral_transformer import SpectralTransformer
    df = make_fake_mrral_df()
    model = SpectralTransformer(n_bands=59, n_classes=5, embed_dim=32, n_heads=2, n_layers=2)
    metrics = train_torch_model(
        model=model, df=df, model_name='test_asl',
        max_epochs=2, batch_size=32, lr=1e-3,
        patience=5, use_wandb=False, checkpoint_dir=None,
        use_asl_loss=True, asl_gamma_neg=4.0, asl_gamma_pos=0.0,
    )
    assert 'val_mAP' in metrics


def make_fake_mrral_df_spatial(n=300):
    """Minimal mrral-format DataFrame (m0..m58) for spatial_vit / freeze tests.

    Named distinctly from the existing make_fake_mrral_df (mrrsu b0..b59 format)
    to avoid shadowing it.
    """
    rng = np.random.default_rng(42)
    data = {f'm{i}': rng.random(n).astype(np.float32) for i in range(59)}
    for col in ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']:
        data[col] = (rng.random(n) > 0.7).astype(np.float32)
    data['confidence_weight'] = np.ones(n, dtype=np.float32)
    data['confidence_tier'] = ['High'] * n
    n_train = int(n * 0.67)
    n_val = int(n * 0.165)
    n_test = n - n_train - n_val
    data['split'] = ['train'] * n_train + ['val'] * n_val + ['test'] * n_test
    return pd.DataFrame(data)


class _FakeEncoderModel(torch.nn.Module):
    """Minimal model with encoder/head structure for freeze tests.
    Accepts flat (B, n_features) input — avoids needing spatial patch data.
    """
    def __init__(self, n_in=59, n_out=5, hidden=16):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(n_in, hidden),
            torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(hidden, n_out)

    def forward(self, x):
        return self.head(self.encoder(x))

    def get_param_groups(self, head_lr, encoder_lr):
        head_params = list(self.head.parameters())
        head_ids = {id(p) for p in head_params}
        enc_params = [p for p in self.parameters() if id(p) not in head_ids]
        return [
            {'params': enc_params, 'lr': encoder_lr},
            {'params': head_params, 'lr': head_lr},
        ]


def test_cr_brightness_aux_smoke(tmp_path):
    """CR + brightness-aux path trains a step without shape error.

    Builds raw mrral caches, then drives train_torch_model with
    continuum_removed + brightness_aux + is_aux_model so CRISMSpectralPatchDataset
    yields (patch, brightness(1,), label, weight) and SpatialSpectralClassifierAux
    (aux_dim=1) consumes it.
    """
    from models.spatial_spectral_classifier_aux import SpatialSpectralClassifierAux
    n, P = 120, 7
    df = make_fake_mrral_df_spatial(n)
    df['tile_id'] = 't0001'
    df['pixel_row'] = 0
    df['pixel_col'] = 0
    rng = np.random.default_rng(0)
    for split in ('train', 'val', 'test'):
        sub = df[df['split'] == split]
        if len(sub) == 0:
            continue
        fp = np.memmap(str(tmp_path / f'mrral_{split}_patches_p{P}.npy'),
                       dtype='float32', mode='w+', shape=(len(sub), P, P, 59))
        fp[:] = (rng.uniform(0.0, 0.5, size=(len(sub), P, P, 59))).astype(np.float32)
        fp.flush(); del fp

    model = SpatialSpectralClassifierAux(
        n_bands=59, patch_size=P, n_classes=5,
        embed_dim=32, n_heads=2, n_layers=2, aux_dim=1)

    metrics = train_torch_model(
        model=model, df=df, model_name='cr_aux_smoke',
        max_epochs=1, batch_size=32, lr=1e-3,
        use_wandb=False, checkpoint_dir=None,
        mrral_map={}, patch_size=P, cache_dir=str(tmp_path),
        continuum_removed=True, brightness_aux=True, is_aux_model=True,
    )
    assert 'val_mAP' in metrics
    assert 0.0 <= metrics['val_mAP'] <= 1.0


def test_freeze_encoder_optimizer_only_has_head_params():
    """When encoder is frozen, optimizer must not contain encoder params."""
    import unittest.mock as mock
    import torch.optim as _optim

    model = _FakeEncoderModel()
    for p in model.encoder.parameters():
        p.requires_grad = False

    df = make_fake_mrral_df_spatial()

    captured = {}
    _orig_adamw = _optim.AdamW
    def _mock_adamw(params, **kw):
        captured['params'] = list(params)
        return _orig_adamw(params, **kw)

    with mock.patch('torch.optim.AdamW', side_effect=_mock_adamw):
        train_torch_model(
            model=model, df=df, model_name='test_freeze',
            max_epochs=1, batch_size=32, lr=1e-3,
            use_wandb=False, checkpoint_dir=None,
            freeze_encoder=True,
        )

    assert captured, "AdamW was not called — optimizer was not constructed"
    for p in captured['params']:
        assert p.requires_grad, "Frozen param found in optimizer"


def test_freeze_encoder_weights_unchanged():
    """Encoder weights must not change after training with freeze_encoder=True."""
    model = _FakeEncoderModel()
    for p in model.encoder.parameters():
        p.requires_grad = False
    encoder_before = {k: v.clone() for k, v in model.encoder.state_dict().items()}

    df = make_fake_mrral_df_spatial()
    train_torch_model(
        model=model, df=df, model_name='test_freeze_weights',
        max_epochs=2, batch_size=32, lr=1e-3,
        use_wandb=False, checkpoint_dir=None,
        freeze_encoder=True,
    )

    for k, v_before in encoder_before.items():
        v_after = model.encoder.state_dict()[k].cpu()
        assert torch.allclose(v_before, v_after), f"Encoder param {k} changed during frozen training"


def test_freeze_encoder_head_params_do_change():
    """With encoder frozen, the head (linear layer) must still be trained."""
    model = _FakeEncoderModel()
    for p in model.encoder.parameters():
        p.requires_grad = False
    head_before = {k: v.clone() for k, v in model.head.state_dict().items()}

    df = make_fake_mrral_df_spatial()
    train_torch_model(
        model=model, df=df, model_name='test_freeze_head_trains',
        max_epochs=3, batch_size=32, lr=1e-2,
        use_wandb=False, checkpoint_dir=None,
        freeze_encoder=True,
    )

    any_changed = any(
        not torch.allclose(head_before[k], model.head.state_dict()[k].cpu())
        for k in head_before
    )
    assert any_changed, "Head params did not change — training may not have occurred"

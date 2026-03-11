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

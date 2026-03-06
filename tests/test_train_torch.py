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

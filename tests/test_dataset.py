import numpy as np
import pytest
import pandas as pd
from data.dataset import CRISMPixelDataset, load_sklearn_arrays

PARQUET = '/mnt/crism/MRDR/crism_classification/data/pixels.parquet'

@pytest.fixture
def small_df():
    df = pd.read_parquet(PARQUET)
    return df[df['split'] == 'train'].head(200)

def test_dataset_len(small_df):
    ds = CRISMPixelDataset(small_df)
    assert len(ds) == 200

def test_dataset_item_shapes(small_df):
    ds = CRISMPixelDataset(small_df)
    features, labels, weight = ds[0]
    assert features.shape == (60,)
    assert labels.shape == (6,)
    assert weight.shape == ()

def test_dataset_item_types(small_df):
    import torch
    ds = CRISMPixelDataset(small_df)
    features, labels, weight = ds[0]
    assert features.dtype == torch.float32
    assert labels.dtype == torch.float32
    assert weight.dtype == torch.float32

def test_load_sklearn_arrays_shapes():
    X_tr, y_tr, w_tr, X_v, y_v, w_v, X_te, y_te, w_te = load_sklearn_arrays(PARQUET)
    assert X_tr.shape[1] == 60
    assert y_tr.shape[1] == 6
    assert w_tr.shape[0] == X_tr.shape[0]
    assert X_v.shape[1] == 60
    assert X_te.shape[1] == 60

def test_load_sklearn_no_nan():
    X_tr, y_tr, w_tr, *_ = load_sklearn_arrays(PARQUET)
    assert not np.isnan(X_tr).any()
    assert not np.isnan(y_tr).any()

def test_patch_dataset_shape(small_df):
    from data.dataset import CRISMPatchDataset
    import yaml, os
    cfg_path = '/mnt/crism/MRDR/crism_classification/config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    from data.extract_pixels import find_tile_pairs
    pairs = find_tile_pairs(cfg['gpkg_dir'], cfg['data_root'])
    mrrsu_map = {tid: p for tid, _, p in pairs}
    ds = CRISMPatchDataset(small_df, mrrsu_map, patch_size=7)
    patch, labels, weight = ds[0]
    assert patch.shape == (60, 7, 7)

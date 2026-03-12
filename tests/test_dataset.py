import os
import numpy as np
import pytest
import pandas as pd
import torch
from data.dataset import CRISMPixelDataset, load_sklearn_arrays

PARQUET = '/mnt/gigas/CRISM/MRDR/crism_classification/data/pixels.parquet'

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

def test_patch_dataset_uses_cache(tmp_path):
    """CRISMPatchDataset loads from memmap cache instead of rasterio when available."""
    import torch
    from data.dataset import CRISMPatchDataset, BAND_COLS

    n = 4
    patch_size = 7
    n_bands = len(BAND_COLS)

    data = {f'b{i}': np.zeros(n, dtype=np.float32) for i in range(n_bands)}
    for col in ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']:
        data[col] = np.zeros(n, dtype=np.float32)
    data['confidence_weight'] = np.ones(n, dtype=np.float32)
    data['confidence_tier'] = ['High'] * n
    data['tile_id'] = ['t0001'] * n
    data['pixel_row'] = [0] * n
    data['pixel_col'] = [0] * n
    data['split'] = ['train'] * n
    df = pd.DataFrame(data)

    sentinel = np.arange(n * n_bands * patch_size * patch_size,
                         dtype=np.float32).reshape(n, n_bands, patch_size, patch_size)
    cache_file = tmp_path / 'train_patches_p7.npy'
    fp = np.memmap(str(cache_file), dtype='float32', mode='w+',
                   shape=(n, n_bands, patch_size, patch_size))
    fp[:] = sentinel
    fp.flush()
    del fp

    ds = CRISMPatchDataset(df, mrrsu_map={}, patch_size=7,
                           cache_dir=str(tmp_path), split='train')
    patch, labels, weight = ds[0]

    assert patch.shape == (n_bands, patch_size, patch_size)
    assert patch.dtype == torch.float32
    np.testing.assert_allclose(patch.numpy(), sentinel[0])


def test_patch_dataset_shape(small_df):
    from data.dataset import CRISMPatchDataset
    import yaml, os
    cfg_path = '/mnt/gigas/CRISM/MRDR/crism_classification/config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    from data.extract_pixels import find_tile_pairs
    pairs = find_tile_pairs(cfg['gpkg_dir'], cfg['data_root'])
    mrrsu_map = {tid: p for tid, _, p in pairs}
    ds = CRISMPatchDataset(small_df, mrrsu_map, patch_size=7)
    patch, labels, weight = ds[0]
    assert patch.shape == (60, 7, 7)


# --- CRISMSpectralDataset tests ---

MRRAL_PARQUET = '/mnt/gigas/CRISM/MRDR/crism_classification/data/mrral_pixels.parquet'


@pytest.fixture
def small_mrral_df():
    if not os.path.exists(MRRAL_PARQUET):
        pytest.skip("mrral_pixels.parquet not yet built — run scripts/build_mrral_dataset.py")
    df = pd.read_parquet(MRRAL_PARQUET)
    return df[df['split'] == 'train'].head(200)


def test_crism_spectral_dataset_shape(small_mrral_df):
    import torch
    from data.dataset import CRISMSpectralDataset
    ds = CRISMSpectralDataset(small_mrral_df)
    feat, label, weight = ds[0]
    assert feat.shape == (59,), f"Expected (59,), got {feat.shape}"
    assert label.shape == (6,)
    assert weight.shape == ()
    assert feat.dtype == torch.float32


def test_crism_spectral_dataset_len(small_mrral_df):
    from data.dataset import CRISMSpectralDataset
    ds = CRISMSpectralDataset(small_mrral_df)
    assert len(ds) == 200


def test_crism_spectral_dataset_raises_on_missing_columns():
    from data.dataset import CRISMSpectralDataset
    df = pd.DataFrame({'olivine_t1': [0.0], 'confidence_weight': [1.0]})
    with pytest.raises(ValueError, match="missing columns"):
        CRISMSpectralDataset(df)


@pytest.fixture
def combined_df():
    """Synthetic dataframe with both mrral and mrrsu columns for unit tests."""
    import numpy as np
    n = 40
    rng = np.random.default_rng(99)
    data = {
        'tile_id': ['t0001'] * n,
        'polygon_id': list(range(n)),
        'pixel_row': list(range(n)),
        'pixel_col': [0] * n,
        **{f'm{i}': rng.random(n).astype('float32') for i in range(59)},
        **{f'b{i}': rng.random(n).astype('float32') for i in range(60)},
        'olivine_t1': rng.integers(0, 2, n).astype('float32'),
        'olivine_t2': rng.integers(0, 2, n).astype('float32'),
        'lcp':         rng.integers(0, 2, n).astype('float32'),
        'hcp':         rng.integers(0, 2, n).astype('float32'),
        'plagioclase': rng.integers(0, 2, n).astype('float32'),
        'other':       rng.integers(0, 2, n).astype('float32'),
        'confidence_weight': np.ones(n, dtype='float32'),
        'confidence_tier': ['High'] * n,
        'split': ['train'] * 30 + ['val'] * 10,
    }
    return pd.DataFrame(data)


def test_combined_dataset_feature_shape(combined_df):
    import torch
    from data.dataset import CRISMCombinedDataset
    ds = CRISMCombinedDataset(combined_df)
    feat, label, weight = ds[0]
    assert feat.shape == (119,), f"Expected (119,), got {feat.shape}"
    assert label.shape == (5,), f"Expected (5,) classes, got {label.shape}"
    assert feat.dtype == torch.float32


def test_combined_dataset_splits_correctly(combined_df):
    import numpy as np
    from data.dataset import CRISMCombinedDataset, MRRAL_BAND_COLS, BAND_COLS
    ds = CRISMCombinedDataset(combined_df)
    feat, _, _ = ds[0]
    expected_mrral = combined_df[MRRAL_BAND_COLS].iloc[0].values
    expected_mrrsu = combined_df[BAND_COLS].iloc[0].values
    np.testing.assert_allclose(feat[:59].numpy(), expected_mrral, rtol=1e-5)
    np.testing.assert_allclose(feat[59:].numpy(), expected_mrrsu, rtol=1e-5)


def test_combined_dataset_raises_on_missing_mrral(combined_df):
    from data.dataset import CRISMCombinedDataset
    df_no_mrral = combined_df.drop(columns=[f'm{i}' for i in range(59)])
    with pytest.raises(ValueError, match="mrral"):
        CRISMCombinedDataset(df_no_mrral)


def test_combined_dataset_raises_on_missing_mrrsu(combined_df):
    from data.dataset import CRISMCombinedDataset
    df_no_mrrsu = combined_df.drop(columns=[f'b{i}' for i in range(60)])
    with pytest.raises(ValueError, match="mrrsu"):
        CRISMCombinedDataset(df_no_mrrsu)


def test_crism_spectral_dataset_high_conf_filtering(small_mrral_df):
    from data.dataset import CRISMSpectralDataset
    high_df = small_mrral_df[small_mrral_df['confidence_tier'] == 'High']
    if len(high_df) == 0:
        pytest.skip("No High-confidence pixels in sample")
    ds_all = CRISMSpectralDataset(small_mrral_df)
    ds_high = CRISMSpectralDataset(high_df)
    assert len(ds_high) <= len(ds_all)


def test_spectral_patch_dataset_shape():
    """CRISMSpectralPatchDataset should yield (7, 7, 59) float32 patches."""
    import glob
    from data.dataset import CRISMSpectralPatchDataset
    import rasterio

    mrral_files = sorted(glob.glob('/mnt/crism/MRDR/mc*/t*mrral*.hdr'))[:5]
    mrral_map = {}
    for hdr in mrral_files:
        basename = os.path.basename(hdr)
        tile_id = basename.split('_mrral_')[0]
        mrral_map[tile_id] = hdr.replace('.hdr', '.img')

    if not mrral_map:
        pytest.skip("No mrral tiles found")

    tile_id = list(mrral_map.keys())[0]
    with rasterio.open(mrral_map[tile_id]) as src:
        H, W = src.height, src.width

    df = pd.DataFrame({
        'tile_id': [tile_id] * 4,
        'pixel_row': [H // 2] * 4,
        'pixel_col': [W // 2] * 4,
        'olivine_t1': [0.0] * 4, 'olivine_t2': [0.0] * 4,
        'lcp': [1.0] * 4, 'hcp': [0.0] * 4,
        'plagioclase': [0.0] * 4, 'other': [0.0] * 4,
        'confidence_weight': [1.0] * 4,
        'confidence_tier': ['High'] * 4,
        'split': ['train'] * 4,
    })

    ds = CRISMSpectralPatchDataset(df, mrral_map, patch_size=7)
    patch, labels, weights = ds[0]
    assert patch.shape == (7, 7, 59), f"Got {patch.shape}"
    assert patch.dtype == torch.float32
    assert labels.shape == (5,)
    assert weights.shape == ()
    assert patch.min().item() >= 0.0
    assert patch.max().item() <= 0.5

import pandas as pd
import pytest
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import load_config

PARQUET_PATH = os.path.join(load_config()['output_dir'], 'pixels.parquet')


def test_parquet_exists():
    assert os.path.exists(PARQUET_PATH), "Run: python scripts/build_dataset.py first"


def test_parquet_schema():
    df = pd.read_parquet(PARQUET_PATH)
    required = ['tile_id', 'polygon_id', 'pixel_row', 'pixel_col',
                'b0', 'b59', 'olivine_t1', 'olivine_t2', 'lcp', 'hcp',
                'plagioclase', 'other', 'confidence_weight',
                'confidence_tier', 'split']
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_no_nodata_in_features():
    df = pd.read_parquet(PARQUET_PATH)
    band_cols = [f'b{i}' for i in range(60)]
    assert not df[band_cols].isnull().any().any()
    assert (df[band_cols] != 65535).all().all()


def test_split_values():
    df = pd.read_parquet(PARQUET_PATH)
    assert set(df['split'].unique()).issubset({'train', 'val', 'test'})


def test_split_covers_all_classes():
    df = pd.read_parquet(PARQUET_PATH)
    label_cols = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']
    for split in ['train', 'val', 'test']:
        sub = df[df['split'] == split]
        for col in label_cols:
            assert sub[col].sum() > 0, f"No {col} in split={split}"


def test_other_class_not_overrepresented():
    df = pd.read_parquet(PARQUET_PATH)
    n_other = (df['other'] == 1.0).sum()
    n_total = len(df)
    assert n_other / n_total < 0.30, f"Other is {n_other/n_total:.1%} of dataset"


def test_confidence_weights_valid():
    df = pd.read_parquet(PARQUET_PATH)
    assert df['confidence_weight'].isin([0.25, 0.5, 1.0]).all()

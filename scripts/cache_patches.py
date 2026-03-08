"""
Pre-extract spatial patches from mrrsu rasters and save as numpy memmaps.

Creates {cache_dir}/{split}_patches_p{patch_size}.npy for each split.
Shape: (n_split_samples, n_bands, patch_size, patch_size) float32.
Row order matches pixels.parquet row order within each split.

Usage:
    conda run -n crism python scripts/cache_patches.py
    conda run -n crism python scripts/cache_patches.py --patch_size 7 --config config.yaml
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import CRISMPatchDataset, BAND_COLS
from data.extract_pixels import find_tile_pairs


def cache_split(split: str, df: pd.DataFrame, mrrsu_map: dict,
                patch_size: int, cache_dir: str) -> None:
    out_path = os.path.join(cache_dir, f'{split}_patches_p{patch_size}.npy')
    if os.path.exists(out_path):
        print(f'[cache] {split}: already exists at {out_path}, skipping.')
        return

    sub = df[df['split'] == split].reset_index(drop=True)
    n = len(sub)
    n_bands = len(BAND_COLS)
    print(f'[cache] {split}: {n:,} samples → {out_path}')

    fp = np.memmap(out_path, dtype='float32', mode='w+',
                   shape=(n, n_bands, patch_size, patch_size))

    # Use live CRISMPatchDataset to extract patches (no cache_dir → rasterio path)
    ds = CRISMPatchDataset(sub, mrrsu_map, patch_size=patch_size)

    for idx in tqdm(range(n), desc=split, unit='patch'):
        patch, _labels, _weight = ds[idx]
        fp[idx] = patch.numpy()

    fp.flush()
    del fp
    ds.close()
    print(f'[cache] {split}: done.')


def main():
    parser = argparse.ArgumentParser(description='Pre-cache CRISM spatial patches.')
    parser.add_argument('--patch_size', type=int, default=7)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    args = parser.parse_args()

    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.config
    )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    parquet_path = os.path.join(cfg['output_dir'], 'pixels.parquet')
    cache_dir = cfg['patch_cache_dir']
    os.makedirs(cache_dir, exist_ok=True)

    print(f'Loading {parquet_path} ...')
    df = pd.read_parquet(parquet_path)

    print(f'Finding tile pairs in {cfg["data_root"]} ...')
    pairs = find_tile_pairs(cfg['gpkg_dir'], cfg['data_root'])
    mrrsu_map = {tid: p for tid, _, p in pairs}
    print(f'Found {len(mrrsu_map)} tiles.')

    for split in args.splits:
        cache_split(split, df, mrrsu_map, args.patch_size, cache_dir)

    print('All splits cached.')


if __name__ == '__main__':
    main()

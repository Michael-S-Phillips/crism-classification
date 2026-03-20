"""
Pre-extract spatial mrral patches and save as numpy memmaps.

Creates {cache_dir}/mrral_{split}_patches_p{patch_size}.npy for each split.
Shape: (n_split_samples, patch_size, patch_size, 59) float32.
Row order matches mrral_pixels.parquet row order within each split.

Usage:
    conda run -n crism python scripts/cache_mrral_patches.py
    conda run -n crism python scripts/cache_mrral_patches.py --patch_size 7
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import CRISMSpectralPatchDataset


def cache_split(split, df, mrral_map, patch_size, cache_dir):
    out_path = os.path.join(cache_dir, f'mrral_{split}_patches_p{patch_size}.npy')
    if os.path.exists(out_path):
        print(f'[cache] {split}: already exists at {out_path}, skipping.')
        return

    sub = df[df['split'] == split].reset_index(drop=True)
    n = len(sub)
    print(f'[cache] {split}: {n:,} samples → {out_path}')

    fp = np.memmap(out_path, dtype='float32', mode='w+',
                   shape=(n, patch_size, patch_size, 59))

    # Use live dataset (no cache) to read from tiles
    ds = CRISMSpectralPatchDataset(sub, mrral_map, patch_size=patch_size)

    for idx in tqdm(range(n), desc=split, unit='patch'):
        patch, _labels, _weight = ds[idx]
        fp[idx] = patch.numpy()

    fp.flush()
    del fp
    ds.close()
    print(f'[cache] {split}: done.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', type=int, default=7)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    args = parser.parse_args()

    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.config
    )
    from config_loader import load_config
    cfg = load_config(cfg_path)

    mrral_parquet = os.path.join(cfg['output_dir'], 'mrral_pixels.parquet')
    cache_dir = cfg['patch_cache_dir']
    os.makedirs(cache_dir, exist_ok=True)

    print(f'Loading {mrral_parquet} ...')
    df = pd.read_parquet(mrral_parquet)

    print('Building mrral tile map ...')
    data_root = cfg.get('data_root', '/mnt/crism/MRDR')
    mrral_hdrs = sorted(glob.glob(os.path.join(data_root, 'mc*', 't*mrral*.hdr')))
    if not mrral_hdrs:
        mrral_hdrs = sorted(glob.glob('/mnt/crism/MRDR/mc*/t*mrral*.hdr'))
    mrral_map = {}
    for hdr in mrral_hdrs:
        tid = os.path.basename(hdr).split('_mrral_')[0]
        mrral_map[tid] = hdr.replace('.hdr', '.img')
    print(f'Found {len(mrral_map)} tiles.')

    for split in args.splits:
        cache_split(split, df, mrral_map, args.patch_size, cache_dir)

    print('All splits cached.')


if __name__ == '__main__':
    main()

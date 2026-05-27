# scripts/build_mrrsu_aux.py
"""Build the aligned mrrsu auxiliary cache: per labeled pixel, the 7x7-mean of
RPEAK1 (band 8) and BD1300 (band 17) from the paired mrrsu tile.

Writes, into <output_dir> (default data/patch_cache):
  mrrsu_aux_{train,val,test}.npy   (n_split, 2) float32, parquet-row order,
                                   column 0 = RPEAK1 mean, column 1 = BD1300 mean
  mrrsu_aux_stats.json             {"mean": [r,b], "std": [r,b]} computed on the
                                   TRAIN split's finite rows (pre-z-score)

Row order matches mrral_pixels.parquet within each split (same as the patch cache).

Usage:
  conda run -n crism python scripts/build_mrrsu_aux.py
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import load_config
from data.mrrsu_aux import mean_pool_nodata, RPEAK1_BAND, BD1300_BAND, NODATA


def build_mrrsu_map(cfg) -> dict:
    data_root = cfg.get('data_root', '/mnt/mrdr')
    hdrs = sorted(set(glob.glob(os.path.join(data_root, 'mc*', 't*mrrsu*.hdr'))
                      + glob.glob(os.path.join(data_root, 't*mrrsu*.hdr'))))
    return {os.path.basename(h).split('_mrrsu_')[0]: h.replace('.hdr', '.img')
            for h in hdrs}


def build_split(df_split, mrrsu_map, patch_size):
    """Return (n,2) float32 array of [RPEAK1_mean, BD1300_mean] per row."""
    out = np.full((len(df_split), 2), np.nan, dtype=np.float32)
    # group by tile so each mrrsu raster is read + pooled once
    for tid, grp in df_split.groupby('tile_id', sort=False):
        if str(tid).startswith('SYNTH_') or tid not in mrrsu_map:
            continue  # synthetic rows / tiles without a paired mrrsu stay NaN
        with rasterio.open(mrrsu_map[tid]) as src:
            rpeak = src.read(RPEAK1_BAND + 1).astype(np.float32)   # rasterio is 1-indexed
            bd = src.read(BD1300_BAND + 1).astype(np.float32)
        rpeak_m = mean_pool_nodata(rpeak, patch_size=patch_size, nodata=NODATA)
        bd_m = mean_pool_nodata(bd, patch_size=patch_size, nodata=NODATA)
        rows = grp.index.to_numpy()
        rr = grp['pixel_row'].to_numpy().astype(int)
        cc = grp['pixel_col'].to_numpy().astype(int)
        H, W = rpeak_m.shape
        inb = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        out[rows[inb], 0] = rpeak_m[rr[inb], cc[inb]]
        out[rows[inb], 1] = bd_m[rr[inb], cc[inb]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patch_size', type=int, default=7)
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--output_dir', default=None)
    ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    args = ap.parse_args()

    cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            args.config)
    cfg = load_config(cfg_path)
    out_dir = args.output_dir or cfg['patch_cache_dir']
    os.makedirs(out_dir, exist_ok=True)

    parquet = os.path.join(cfg['output_dir'], 'mrral_pixels.parquet')
    df = pd.read_parquet(parquet, columns=['tile_id', 'pixel_row', 'pixel_col', 'split'])
    mrrsu_map = build_mrrsu_map(cfg)
    print(f'mrrsu tiles found: {len(mrrsu_map)}')

    arrays = {}
    for split in args.splits:
        sub = df[df['split'] == split].reset_index(drop=True)
        arr = build_split(sub, mrrsu_map, args.patch_size)
        path = os.path.join(out_dir, f'mrrsu_aux_{split}.npy')
        np.save(path, arr)
        n_nan = int(np.isnan(arr).any(axis=1).sum())
        print(f'  {split}: {len(arr):,} rows -> {path}  ({n_nan:,} NaN rows)')
        arrays[split] = arr

    # Train-split stats over finite rows (per feature), pre-z-score
    tr = arrays['train']
    finite = np.isfinite(tr).all(axis=1)
    mean = tr[finite].mean(axis=0).tolist()
    std = (tr[finite].std(axis=0) + 1e-8).tolist()
    stats_path = os.path.join(out_dir, 'mrrsu_aux_stats.json')
    with open(stats_path, 'w') as f:
        json.dump({'mean': mean, 'std': std}, f, indent=2)
    print(f'wrote {stats_path}  mean={mean} std={std}')


if __name__ == '__main__':
    main()

"""Re-extract spectra for specific tiles in a labeled-pixels parquet from the
(repaired) tile .img files on disk.

Why: data/mrral_pixels.parquet was extracted while some PDS downloads were
silently truncated (GDAL zero-fills reads past EOF). The tiles were re-fetched
and verified (reports/img_integrity_audit.csv: all OK), but the parquet still
carries the fossilized zero tails for rows extracted from the bad copies —
t0360 is the only tile whose truncation fell inside the m0..m58 band window
AND which carries labeled rows.

Usage:
    python scripts/patch_parquet_from_repaired_tiles.py \
        --parquet data/mrral_pixels.parquet --tiles t0360
Writes atomically (tmp + rename); prints before/after zero-fraction audits.
NOTE: run the same patch on the HPC copy (or rsync the patched parquet) before
the next dataset build.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import rasterio

BAND_COLS = [f'm{i}' for i in range(59)]
TILE_GLOBS = ['/mnt/mrdr/mc*/{tid}_mrral_*_0327_4.img']


def find_tile_img(tid: str) -> str:
    for pat in TILE_GLOBS:
        hits = glob.glob(pat.format(tid=tid))
        if hits:
            return hits[0]
    raise FileNotFoundError(f'no mrral img for {tid}')


def patch(parquet: str, tiles: list[str]) -> None:
    df = pd.read_parquet(parquet)
    for tid in tiles:
        mask = df['tile_id'] == tid
        n = int(mask.sum())
        if n == 0:
            print(f'{tid}: no rows, skipping')
            continue
        rows = df.loc[mask, 'pixel_row'].to_numpy(dtype=int)
        cols = df.loc[mask, 'pixel_col'].to_numpy(dtype=int)
        before_zero = float((df.loc[mask, BAND_COLS].to_numpy() == 0).mean())
        img = find_tile_img(tid)
        with rasterio.open(img) as src:
            cube = src.read(indexes=list(range(1, 60)))  # bands 1..59 -> m0..m58
        spectra = cube[:, rows, cols].T.astype(np.float64)  # (n, 59)
        df.loc[mask, BAND_COLS] = spectra
        after_zero = float((spectra == 0).mean())
        n_nodata = int((spectra == 65535).any(axis=1).sum())
        print(f'{tid}: {n:,} rows re-extracted from {os.path.basename(img)} | '
              f'zero-frac {before_zero:.3f} -> {after_zero:.3f} | '
              f'rows w/ NODATA bands: {n_nodata}')
    tmp = parquet + '.tmp'
    df.to_parquet(tmp, index=False)
    os.replace(tmp, parquet)
    print(f'patched parquet written atomically: {parquet}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parquet', default='data/mrral_pixels.parquet')
    ap.add_argument('--tiles', nargs='+', default=['t0360'])
    args = ap.parse_args()
    patch(args.parquet, args.tiles)

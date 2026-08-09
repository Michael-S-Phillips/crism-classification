"""Re-extract one tile's spectra inside an existing review session's parquets.

Why this exists (2026-08-08): the v3 review session extracted t1444's pixels
while that tile was still downloading. 537,525 rows were frozen with
reflectance 0.0 across 2251-2457 nm — self-consistent, so nothing crashed and
nothing warned. The tile on disk finished two hours later and has been correct
ever since; only the parquet is stale. Re-downloading from PDS fixes nothing.

This repairs the parquets in place from the tile already on disk, replicating
scripts/review/loader.py's extraction semantics exactly:
  * read bands 1..N_BANDS (1-indexed rasterio) as float32
  * store as float64 (persistence.py does `.astype(np.float64)`)
  * NODATA (65535) is NOT rewritten — the original extraction dropped any pixel
    with a NODATA band, so a repaired pixel that now reads NODATA would mean the
    tile changed shape. That is reported and left untouched rather than guessed.

Only rows whose tile_id matches are touched; every other row, column, dtype and
the row order are preserved. Re-runnable: a repaired file simply reports 0
changes on a second pass.

Usage
    python scripts/reextract_stale_review_tile.py --tile t1444 \
        --session data/mc13_review_7cls_v3 [--dry_run]

Verify afterwards with:
    python scripts/audit_spectra_quality.py <session>/hard_negatives \
        --verify_against_tiles
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_BANDS = 59
NODATA = 65535.0
BANDS = [f'm{i}' for i in range(N_BANDS)]


def load_tile_cube(tile: str, data_root: str) -> np.ndarray:
    import rasterio
    hits = sorted(glob.glob(os.path.join(data_root, 'mc*', f'{tile}_mrral*.img')))
    if not hits:
        raise SystemExit(f'ERROR: no mrral .img for {tile} under {data_root}')
    if len(hits) > 1:
        print(f'  NOTE: {len(hits)} candidates, using {hits[0]}')
    with rasterio.open(hits[0]) as src:
        if src.count < N_BANDS:
            raise SystemExit(
                f'ERROR: {hits[0]} has {src.count} bands, need >= {N_BANDS} — '
                f'the tile itself is incomplete; re-download it first.')
        cube = src.read(list(range(1, N_BANDS + 1))).astype(np.float32)
    print(f'  loaded {hits[0]}  shape={cube.shape}')
    return cube


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tile', required=True, help="e.g. t1444")
    ap.add_argument('--session', required=True,
                    help='review session dir containing hard_negatives/ and/or '
                         'confirmed_pixels/')
    ap.add_argument('--data_root', default=None,
                    help='tile root (default: config data_root)')
    ap.add_argument('--dry_run', action='store_true',
                    help='report what would change; write nothing')
    args = ap.parse_args()

    root = args.data_root
    if root is None:
        from config_loader import load_config
        root = load_config()['data_root']

    print(f'Re-extracting {args.tile} from {root} into {args.session}')
    cube = load_tile_cube(args.tile, root)
    _, H, W = cube.shape

    files = []
    for sub in ('hard_negatives', 'confirmed_pixels'):
        files += sorted(glob.glob(os.path.join(args.session, sub, '*.parquet')))
    if not files:
        raise SystemExit(f'ERROR: no parquet fragments under {args.session}')

    n_files = n_rows = n_changed = n_oob = n_nodata = 0
    for f in files:
        df = pd.read_parquet(f)
        if 'tile_id' not in df.columns:
            continue
        m = (df['tile_id'] == args.tile).to_numpy()
        if not m.any():
            continue
        idx = np.flatnonzero(m)
        r = df['pixel_row'].to_numpy()[idx].astype(np.int64)
        c = df['pixel_col'].to_numpy()[idx].astype(np.int64)

        inb = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        if not inb.all():
            n_oob += int((~inb).sum())
            idx, r, c = idx[inb], r[inb], c[inb]
        if len(idx) == 0:
            continue

        new = cube[:, r, c].T.astype(np.float64)          # (n, 59)
        bad = (new == NODATA).any(axis=1)
        if bad.any():
            # The original extraction dropped any pixel with a NODATA band, so
            # this should be impossible. Leave those rows alone and report.
            n_nodata += int(bad.sum())
            keep = ~bad
            idx, new = idx[keep], new[keep]
        if len(idx) == 0:
            continue

        old = df.iloc[idx][BANDS].to_numpy(np.float64)
        changed = int((~np.isclose(old, new, atol=1e-9, equal_nan=True)).any(axis=1).sum())

        for j, col in enumerate(BANDS):
            vals = df[col].to_numpy(np.float64).copy()
            vals[idx] = new[:, j]
            df[col] = vals.astype(df[col].dtype)

        if not args.dry_run:
            df.to_parquet(f, index=False)
        n_files += 1
        n_rows += len(idx)
        n_changed += changed

    verb = 'would repair' if args.dry_run else 'repaired'
    print(f'\n{verb}: {n_rows:,} {args.tile} rows across {n_files} file(s); '
          f'{n_changed:,} rows actually differed')
    if n_oob:
        print(f'  WARNING: {n_oob:,} rows had pixel coords outside the tile — '
              f'left untouched')
    if n_nodata:
        print(f'  WARNING: {n_nodata:,} rows now read NODATA on the tile — '
              f'left untouched (the tile may have changed shape)')
    if args.dry_run:
        print('  [dry_run] nothing written')


if __name__ == '__main__':
    main()

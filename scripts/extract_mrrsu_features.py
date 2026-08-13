"""Extract the 60 mrrsu summary parameters at each labeled pixel.

Row order is preserved EXACTLY: the output is aligned row-for-row with the input
parquet, because downstream code (Tasks 4 and 5) joins them positionally. A
reorder, a dropped row, or a row/col transposition attaches every label to the
wrong pixel's parameters and produces a plausible but meaningless baseline --
with no error anywhere. This exact bug class hit the MTRDR plagioclase caches
earlier in this project.

Feature columns are named by the REAL parameter name (OLINDEX3, BD1300, ...)
taken from ``read_mrrsu_cube``'s returned names, not positional p0..p59, so
downstream stages never need a header to decode them.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mrrsu_bands import N_MRRSU_BANDS, read_mrrsu_cube  # noqa: E402


def _smooth_nanmean(cube: np.ndarray, size: int = 7) -> np.ndarray:
    """7x7 mean ignoring NaN. RPEAK1 is documented as a REGIONAL discriminant
    in data/mrrsu_aux.py, so a per-pixel read understates it. NaN pixels must
    be excluded from each window's mean rather than propagated or zero-filled
    without correcting the denominator -- a single nodata pixel must not blank
    (or skew) its whole 7x7 neighbourhood."""
    filled = np.nan_to_num(cube, nan=0.0)
    valid = np.isfinite(cube).astype(np.float32)
    num = uniform_filter(filled, size=(size, size, 1), mode='nearest')
    den = uniform_filter(valid, size=(size, size, 1), mode='nearest')
    with np.errstate(invalid='ignore', divide='ignore'):
        out = num / den
    out[den == 0] = np.nan
    return out.astype(np.float32)


def extract_features(df: pd.DataFrame, mrrsu_map: dict[str, str],
                     smooth: bool = False, reader=None) -> pd.DataFrame:
    """Extract the 60 mrrsu parameters at each row's (tile_id, pixel_row,
    pixel_col), returning a DataFrame row-aligned with ``df`` (same length,
    same index, same row order).

    Parameters
    ----------
    df : DataFrame with at least tile_id, pixel_row, pixel_col columns.
    mrrsu_map : tile_id -> path to that tile's mrrsu .img (used by the default
        reader; ignored when ``reader`` is supplied).
    smooth : apply the 7x7 NaN-aware mean before sampling.
    reader : callable(tile_id) -> (cube, names), overriding the default
        ``read_mrrsu_cube(mrrsu_map[tile_id])``. Used by tests to avoid real
        I/O.
    """
    reader = reader or (lambda tid: read_mrrsu_cube(mrrsu_map[tid]))
    out = np.full((len(df), N_MRRSU_BANDS), np.nan, dtype=np.float32)
    col_names: list[str] | None = None
    rows = df['pixel_row'].to_numpy(np.int64)
    cols = df['pixel_col'].to_numpy(np.int64)
    # groupby(...).indices maps each tile_id to the INTEGER POSITIONS (not
    # labels) of its rows within df, in their original order -- scattering
    # results back via `out[idx[ok]] = ...` therefore preserves input row
    # order exactly, regardless of how many tiles are interleaved or what
    # order groupby visits them in.
    for tid, idx in df.groupby('tile_id', sort=False).indices.items():
        if tid not in mrrsu_map and reader is None:
            continue
        cube, names = reader(tid)
        if col_names is None:
            col_names = list(names)
        elif list(names) != col_names:
            raise ValueError(
                f'{tid}: band order differs from earlier tiles — a threshold '
                f'would be applied to the wrong parameter')
        if smooth:
            cube = _smooth_nanmean(cube)
        h, w = cube.shape[:2]
        r, c = rows[idx], cols[idx]
        ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        # Positional assignment into `idx` preserves input row order exactly.
        out[idx[ok]] = cube[r[ok], c[ok], :]
    if col_names is None:
        raise ValueError('no tile was read; cannot name the feature columns')
    return pd.DataFrame(out, columns=col_names, index=df.index)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--smooth', action='store_true',
                    help='7x7 nan-aware mean; matches the deep model receptive '
                         'field and the RPEAK1 regional note in mrrsu_aux.py')
    ap.add_argument('--data_root', default=None)
    args = ap.parse_args()

    from config_loader import load_config
    root = args.data_root or load_config()['data_root']
    hdrs = sorted(set(glob.glob(os.path.join(root, 'mc*', 't*mrrsu*.hdr'))
                      + glob.glob(os.path.join(root, 't*mrrsu*.hdr'))))
    mrrsu_map = {os.path.basename(h).split('_mrrsu_')[0]: h.replace('.hdr', '.img')
                 for h in hdrs}

    df = pd.read_parquet(args.parquet,
                         columns=['tile_id', 'pixel_row', 'pixel_col', 'split'])
    print(f'{len(df):,} rows, {df.tile_id.nunique()} tiles, '
          f'{sum(t in mrrsu_map for t in df.tile_id.unique())} with an mrrsu tile')
    feats = extract_features(df, mrrsu_map, smooth=args.smooth)
    result = pd.concat([df.reset_index(drop=True),
                        feats.reset_index(drop=True)], axis=1)
    assert len(result) == len(df), 'row count changed — alignment broken'
    result.to_parquet(args.out, index=False)
    n_all_nan = int(np.isnan(feats.to_numpy()).all(axis=1).sum())
    print(f'wrote {args.out}  ({n_all_nan:,} rows all-NaN — no mrrsu coverage)')


if __name__ == '__main__':
    main()

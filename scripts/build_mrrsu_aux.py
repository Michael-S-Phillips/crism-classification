# scripts/build_mrrsu_aux.py
"""Build the aligned mrrsu auxiliary cache: per labeled pixel, the 7x7-mean of
RPEAK1 (band 8) and BD1300 (band 17) from the paired mrrsu tile.

Writes, into <output_dir> (default data/patch_cache):
  mrrsu_aux_{train,val,test}.npy   (n_split, 2) float32, parquet-row order,
                                   column 0 = RPEAK1 mean, column 1 = BD1300 mean
  mrrsu_aux_stats.json             stats JSON whose shape depends on --norm_mode:
    zscore        : {"mode": "zscore", "mean": [r,b], "std": [r,b], ...}
    minmax        : {"mode": "minmax", "min":  [r,b], "max": [r,b], ...}
    pertile_zscore: {"mode": "pertile_zscore",
                     "fallback_mean": [r,b], "fallback_std": [r,b],
                     "min_valid_per_tile": N, ...}
    Every JSON also carries: version=2, physical_ranges (per-band [lo, hi]).

Stats are computed from rows whose entries pass ``physically_valid_mask`` on both
bands AFTER the 7x7 NODATA-masking pool. NaN aux entries in the saved .npy files
are propagated downstream — the dataset / inference paths replace them with 0.0
post-transform (== "no information").

Row order matches mrral_pixels.parquet within each split (same as the patch cache).

Usage:
  conda run -n crism python scripts/build_mrrsu_aux.py
  conda run -n crism python scripts/build_mrrsu_aux.py --norm_mode minmax
  conda run -n crism python scripts/build_mrrsu_aux.py --norm_mode pertile_zscore \\
      --min_valid_per_tile 2000

Tile-limit modes for local smoke tests:
  --limit_tiles N            keep only the first N tiles in mrral_pixels.parquet
                             before building any split. Useful for CPU-only
                             sanity checks of stats JSON / output shape.
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
from data.mrrsu_aux import (
    AUX_BAND_ORDER,
    BAND_VALID_RANGES,
    BD1300_BAND,
    NODATA,
    RPEAK1_BAND,
    apply_invalid_to_nan,
    mean_pool_nodata,
    physically_valid_mask,
)


STATS_VERSION = 2
NORM_MODES = ("zscore", "minmax", "pertile_zscore")


def build_mrrsu_map(cfg) -> dict:
    data_root = cfg.get('data_root', '/mnt/mrdr')
    hdrs = sorted(set(glob.glob(os.path.join(data_root, 'mc*', 't*mrrsu*.hdr'))
                      + glob.glob(os.path.join(data_root, 't*mrrsu*.hdr'))))
    return {os.path.basename(h).split('_mrrsu_')[0]: h.replace('.hdr', '.img')
            for h in hdrs}


def build_split(df_split, mrrsu_map, patch_size):
    """Return (n,2) float32 array of [RPEAK1_mean, BD1300_mean] per row.

    Out-of-physical-range and sentinel pixels are converted to NaN *before* the
    7x7 mean (via ``apply_invalid_to_nan`` -> ``mean_pool_nodata``'s isfinite
    filter), so the resulting mean only averages physically-plausible
    neighbors. Windows with zero valid neighbors are NaN. After pooling, any
    pooled mean that itself falls outside the physical range (unlikely but
    possible at strong gradients) is also NaN-ed out.
    """
    out = np.full((len(df_split), 2), np.nan, dtype=np.float32)
    # group by tile so each mrrsu raster is read + pooled once
    for tid, grp in df_split.groupby('tile_id', sort=False):
        if str(tid).startswith('SYNTH_') or tid not in mrrsu_map:
            continue  # synthetic rows / tiles without a paired mrrsu stay NaN
        with rasterio.open(mrrsu_map[tid]) as src:
            rpeak = src.read(RPEAK1_BAND + 1).astype(np.float32)   # rasterio is 1-indexed
            bd = src.read(BD1300_BAND + 1).astype(np.float32)
        # Convert physically-implausible pixels to NaN *before* the windowed mean
        # so they don't pollute neighbour averages.
        rpeak_clean = apply_invalid_to_nan(rpeak, "RPEAK1")
        bd_clean = apply_invalid_to_nan(bd, "BD1300")
        rpeak_m = mean_pool_nodata(rpeak_clean, patch_size=patch_size, nodata=NODATA)
        bd_m = mean_pool_nodata(bd_clean, patch_size=patch_size, nodata=NODATA)
        # Belt-and-braces: pooled means that fall outside the physical range
        # (because pooling can shift values slightly) are also NaN.
        rpeak_m = np.where(physically_valid_mask(rpeak_m, "RPEAK1"), rpeak_m, np.nan)
        bd_m = np.where(physically_valid_mask(bd_m, "BD1300"), bd_m, np.nan)
        rows = grp.index.to_numpy()
        rr = grp['pixel_row'].to_numpy().astype(int)
        cc = grp['pixel_col'].to_numpy().astype(int)
        H, W = rpeak_m.shape
        inb = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
        out[rows[inb], 0] = rpeak_m[rr[inb], cc[inb]]
        out[rows[inb], 1] = bd_m[rr[inb], cc[inb]]
    return out


def _compute_valid_mask(arr: np.ndarray) -> np.ndarray:
    """Boolean (N,) mask: row is valid iff both columns are physically valid."""
    m_rp = physically_valid_mask(arr[:, 0], "RPEAK1")
    m_bd = physically_valid_mask(arr[:, 1], "BD1300")
    return m_rp & m_bd


def compute_stats(train_arr: np.ndarray, mode: str, min_valid_per_tile: int) -> dict:
    """Compute and return a stats JSON dict for ``mode``.

    Always includes the global zscore mean/std as the ``pertile_zscore``
    fallback so the same code path can be reused at inference. ``min`` /
    ``max`` are only present in ``minmax`` mode.
    """
    valid = _compute_valid_mask(train_arr)
    tr = train_arr[valid]
    if len(tr) == 0:
        raise ValueError("no physically-valid rows in train split — cannot compute stats")
    mean = tr.mean(axis=0).astype(np.float64)
    std = (tr.std(axis=0) + 1e-8).astype(np.float64)

    stats: dict = {
        "version": STATS_VERSION,
        "mode": mode,
        "physical_ranges": {b: list(BAND_VALID_RANGES[b]) for b in AUX_BAND_ORDER},
        "band_order": list(AUX_BAND_ORDER),
        "n_valid_train_rows": int(len(tr)),
        "n_train_rows": int(len(train_arr)),
    }

    if mode == "zscore":
        stats["mean"] = mean.tolist()
        stats["std"] = std.tolist()
    elif mode == "minmax":
        mn = tr.min(axis=0).astype(np.float64)
        mx = tr.max(axis=0).astype(np.float64)
        # Guard against degenerate identical-min-max.
        denom = mx - mn
        denom = np.where(denom < 1e-8, 1.0, denom)
        stats["min"] = mn.tolist()
        stats["max"] = (mn + denom).tolist()
    elif mode == "pertile_zscore":
        stats["fallback_mean"] = mean.tolist()
        stats["fallback_std"] = std.tolist()
        stats["min_valid_per_tile"] = int(min_valid_per_tile)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected one of {NORM_MODES}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patch_size', type=int, default=7)
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--output_dir', default=None)
    ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    ap.add_argument('--norm_mode', choices=NORM_MODES, default='zscore',
                    help='normalization variant; controls what is written into '
                         'mrrsu_aux_stats.json (default: zscore).')
    ap.add_argument('--min_valid_per_tile', type=int, default=1000,
                    help='only relevant for --norm_mode pertile_zscore: at '
                         'inference time, tiles with fewer valid pixels than '
                         'this fall back to the global fallback_mean/std.')
    ap.add_argument('--limit_tiles', type=int, default=None,
                    help='OPTIONAL DEBUG: keep only the first N tiles in the '
                         'parquet before building. Useful for CPU smoke tests; '
                         'do not use for production caches.')
    args = ap.parse_args()

    cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            args.config)
    cfg = load_config(cfg_path)
    out_dir = args.output_dir or cfg['patch_cache_dir']
    os.makedirs(out_dir, exist_ok=True)

    parquet = os.path.join(cfg['output_dir'], 'mrral_pixels.parquet')
    df = pd.read_parquet(parquet, columns=['tile_id', 'pixel_row', 'pixel_col', 'split'])
    if args.limit_tiles is not None:
        keep = list(df['tile_id'].drop_duplicates().head(args.limit_tiles))
        df = df[df['tile_id'].isin(keep)].reset_index(drop=True)
        print(f'--limit_tiles {args.limit_tiles}: kept {len(keep)} tiles, '
              f'{len(df):,} rows')
    mrrsu_map = build_mrrsu_map(cfg)
    print(f'mrrsu tiles found: {len(mrrsu_map)}')
    print(f'norm_mode: {args.norm_mode}')

    arrays = {}
    for split in args.splits:
        sub = df[df['split'] == split].reset_index(drop=True)
        arr = build_split(sub, mrrsu_map, args.patch_size)
        path = os.path.join(out_dir, f'mrrsu_aux_{split}.npy')
        np.save(path, arr)
        n_nan = int(np.isnan(arr).any(axis=1).sum())
        print(f'  {split}: {len(arr):,} rows -> {path}  ({n_nan:,} NaN rows)')
        arrays[split] = arr

    if 'train' not in arrays:
        raise RuntimeError("--splits must include 'train' to compute stats")

    stats = compute_stats(arrays['train'], args.norm_mode, args.min_valid_per_tile)
    stats_path = os.path.join(out_dir, 'mrrsu_aux_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'wrote {stats_path}')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()

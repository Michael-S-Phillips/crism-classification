"""Mine bright-dusty pixels that current models confidently call mafic.

Spec: docs/superpowers/specs/2026-08-17-dust-hard-negatives-design.md

A pixel qualifies when all five hold:
  1. no mafic signature   OLINDEX3/LCPINDEX2/HCPINDEX2 below tile p40
  2. no alteration        BD1900_2/D2300/BD2210_2 below tile p60
  3. dusty                RBR AND R770 above tile p60
  4. hard                 some model fires >= 0.90 for a mineral there
  5. physically valid     passes the PHYS_MAX/nodata test

Every threshold is a TILE-RELATIVE percentile. Absolute cuts do not transfer:
t1249's whole-tile LCPINDEX2 median (0.0299) exceeds t1321's 90th percentile.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_loader import load_config  # noqa: E402

NODATA = 65535
PHYS_MAX = 1.0
CLIP_MAX = 0.5
N_BANDS = 59
PATCH = 7

# 0-based mrrsu band indices; rasterio band number is index + 1.
MRRSU_IDX = {
    'R770': 0, 'RBR': 1, 'RPEAK1': 8, 'OLINDEX3': 15, 'BD1300': 17,
    'LCPINDEX2': 18, 'HCPINDEX2': 19, 'BD1900_2': 27, 'BD2210_2': 34,
    'D2300': 41,
}
MAFIC = ('OLINDEX3', 'LCPINDEX2', 'HCPINDEX2')
ALTERATION = ('BD1900_2', 'D2300', 'BD2210_2')
DUSTY = ('RBR', 'R770')

# scripts/floor_test.sh — training on these would make the floor test
# partly train-on-test, destroying the one comparator MODELS.md relies on.
FLOOR_TEST_TILES = frozenset(
    {'t1249', 't1250', 't1321', 't1322', 't0434', 't0435', 't1086', 't1087'})

MAFIC_PCTL = 40.0
ALTERATION_PCTL = 60.0
DUSTY_PCTL = 60.0
HARD_P = 0.90
NON_MINERAL = frozenset({'bland', 'other', 'junk'})


def _pctl(band: np.ndarray, valid: np.ndarray, q: float) -> float:
    v = band[valid]
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if v.size else np.inf


def _below_pctl(b: np.ndarray, valid: np.ndarray, q: float) -> np.ndarray:
    """True where b is at-or-below its tile p-th percentile.

    Inclusive on purpose: a clean tile-relative split (half the tile pinned to
    one value, half to another -- the exact shape a real dust mantle abutting
    mafic terrain produces) lands the percentile exactly on one group's own
    value. MAFIC/DUSTY define the *wanted* profile, so a tie must still count
    as qualifying, or a 50/50 split silently excludes the population the
    percentile was meant to admit.
    """
    return np.isfinite(b) & (b <= _pctl(b, valid, q))


def _above_pctl(b: np.ndarray, valid: np.ndarray, q: float) -> np.ndarray:
    """True where b is at-or-above its tile p-th percentile (see _below_pctl)."""
    return np.isfinite(b) & (b >= _pctl(b, valid, q))


def _no_alteration_signature(b: np.ndarray, valid: np.ndarray, q: float) -> np.ndarray:
    """True where b is strictly below its tile p-th percentile.

    Alteration is a veto, not a target profile: a tied boundary must resolve
    toward exclusion, or a large altered block sitting exactly at its own
    percentile slips through as a "dust" hard negative -- teaching the model
    to miss real alteration, which is the one failure this filter exists to
    prevent. The single exception is a perfectly flat band (zero variance
    across valid pixels): every pixel then equals the percentile by
    construction, carrying no signal at all, and a strict '<' would veto the
    whole tile for nothing. That happens on real bland/featureless ground and
    is exactly what the default-filled synthetic test tiles exercise.
    """
    v = b[valid]
    v = v[np.isfinite(v)]
    if v.size == 0 or np.ptp(v) == 0:
        return np.isfinite(b) & valid
    return np.isfinite(b) & (b < _pctl(b, valid, q))


def select_dust_negatives(mrrsu, probs, class_names, valid) -> np.ndarray:
    """Bool (H, W) mask of dust hard negatives. Tile-relative throughout."""
    keep = valid.copy()
    for name in MAFIC:
        keep &= _below_pctl(mrrsu[MRRSU_IDX[name]], valid, MAFIC_PCTL)
    for name in ALTERATION:
        keep &= _no_alteration_signature(mrrsu[MRRSU_IDX[name]], valid, ALTERATION_PCTL)
    for name in DUSTY:
        keep &= _above_pctl(mrrsu[MRRSU_IDX[name]], valid, DUSTY_PCTL)
    mineral_cols = [i for i, c in enumerate(class_names) if c not in NON_MINERAL]
    if not mineral_cols:
        raise ValueError(f'no mineral classes among {class_names}')
    keep &= probs[:, :, mineral_cols].max(axis=2) >= HARD_P
    return keep


def thin_mask(mask, min_sep: int, max_per_tile: int, seed: int) -> np.ndarray:
    """Greedy spatial thinning: no two kept pixels within min_sep, capped.

    Without this one large dust mantle supplies the whole negative set and the
    model learns a location rather than a spectral class.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.zeros_like(mask)
    order = np.random.default_rng(seed).permutation(len(ys))
    out = np.zeros_like(mask)
    kept_y: list[int] = []
    kept_x: list[int] = []
    sep2 = min_sep * min_sep
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        if kept_y:
            dy = np.asarray(kept_y) - y
            dx = np.asarray(kept_x) - x
            if (dy * dy + dx * dx).min() < sep2:
                continue
        out[y, x] = True
        kept_y.append(y)
        kept_x.append(x)
        if len(kept_y) >= max_per_tile:
            break
    return out


def _load_tile(mrral_path):
    with rasterio.open(mrral_path) as src:
        data = src.read(list(range(1, N_BANDS + 1))).astype(np.float32)
    nodata = (data == NODATA) | ~np.isfinite(data) | (data > PHYS_MAX)
    data = np.clip(data, 0.0, CLIP_MAX)
    data[nodata] = 0.0
    return data, ~nodata.any(axis=0)


def _patch_valid(valid, pad=PATCH // 2):
    """True where the whole 7x7 neighbourhood is valid. The classifier reads a
    patch, so a mined centre with a nodata neighbour teaches the padding."""
    from scipy.ndimage import minimum_filter
    ok = minimum_filter(valid.astype(np.uint8), size=PATCH).astype(bool)
    ok[:pad, :] = False; ok[-pad:, :] = False
    ok[:, :pad] = False; ok[:, -pad:] = False
    return ok


def mine_tile(tid, mrral_path, mrrsu_path, probs_path, min_sep, max_per_tile, seed):
    cube, valid = _load_tile(mrral_path)
    with rasterio.open(mrrsu_path) as s:
        mrrsu = s.read().astype(np.float32)
    mrrsu[(mrrsu == NODATA) | ~np.isfinite(mrrsu)] = np.nan
    d = np.load(probs_path, allow_pickle=True)
    probs = d['probs'].astype(np.float32)
    names = [str(x) for x in d['class_names']]
    valid = valid & d['valid_mask'].astype(bool) & _patch_valid(valid)
    mask = select_dust_negatives(mrrsu, probs, names, valid)
    mask = thin_mask(mask, min_sep, max_per_tile, seed)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    out = {'tile_id': tid, 'pixel_row': ys, 'pixel_col': xs}
    for b in range(N_BANDS):
        out[f'band_{b:02d}'] = cube[b][ys, xs]
    for name, idx in MRRSU_IDX.items():
        out[name] = mrrsu[idx][ys, xs]
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--probs_dir', default='data/mc_deploy_pyx_physmax/probs')
    ap.add_argument('--out', default='data/hard_negatives_dust.parquet')
    ap.add_argument('--min_sep', type=int, default=5)
    ap.add_argument('--max_per_tile', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    root = load_config()['data_root']
    frames, skipped = [], []
    for p in sorted(glob.glob(os.path.join(args.probs_dir, '*', '*_probs.npz'))):
        tid = os.path.basename(p).replace('_probs.npz', '')
        if tid in FLOOR_TEST_TILES:
            skipped.append(tid)
            continue
        mr = sorted(glob.glob(os.path.join(root, 'mc*', f'{tid}_mrral_*.img')))
        su = sorted(glob.glob(os.path.join(root, 'mc*', f'{tid}_mrrsu_*.img')))
        if not mr or not su:
            print(f'  WARNING: missing mrral/mrrsu for {tid}', file=sys.stderr)
            continue
        df = mine_tile(tid, mr[0], su[0], p, args.min_sep, args.max_per_tile,
                       args.seed)
        n = 0 if df is None else len(df)
        print(f'  {tid}: {n:,} mined', flush=True)
        if df is not None:
            frames.append(df)
    print(f'excluded {len(skipped)} floor-test tiles: {sorted(skipped)}')
    if not frames:
        raise SystemExit('nothing mined')
    out = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f'wrote {args.out}: {len(out):,} rows from {out.tile_id.nunique()} tiles')
    print('\nmined-population mrrsu medians (audit the worst tiles before merging):')
    for name in MRRSU_IDX:
        print(f'  {name:<11} {out[name].median():.4f}')


if __name__ == '__main__':
    main()

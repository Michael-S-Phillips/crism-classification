"""Retroactively stamp the alteration column onto data/mrral_pixels.parquet
based on the original gpkg Category strings.

Until 2026-06-10 the label parser silently dropped "alteration" tokens
(``_TOKEN_MAP['alteration'] = {}``), so the existing mrral_pixels.parquet
has no information about the ~157 alteration-bearing polygons across
Argyre / Hellas / Nili gpkgs. We need that signal in the training pool to
train a 6-class classifier with alteration as a primary output.

This script:
  1. Loads data/mrral_pixels.parquet
  2. Iterates every gpkg in /mnt/mrdr/categorized_mineral_units/
  3. For each polygon whose Category contains "alteration", maps
     (tile_id, polygon_id) → 1.0  (multi-label compatible with the mafic
     labels that may also be present)
  4. Writes back to data/mrral_pixels.parquet with an alteration column.

The mafic labels (olivine, lcp, hcp, plag) on those polygons stay as they
were — the parser already set them correctly. Only the missing alteration
signal is recovered.

Run:
    conda run -n crism python scripts/patch_mrral_pixels_with_alteration.py
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_GPKG_DIR_CANDIDATES = [
    '/mnt/mrdr/categorized_mineral_units',                   # local workstation
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 'data', 'categorized_mineral_units'),       # HPC checkout
]
GPKG_DIR_DEFAULT = next((d for d in _GPKG_DIR_CANDIDATES if os.path.isdir(d)),
                       _GPKG_DIR_CANDIDATES[0])


_CONF_RE = re.compile(r'\((\w+)\)')


def _tile_id_from_filename(fname: str) -> str | None:
    """T0183.gpkg → 't0183'.  north_hellas_*.gpkg → None (multi-tile)."""
    stem = os.path.splitext(os.path.basename(fname))[0]
    if not re.match(r'^[Tt]\d+', stem):
        return None
    return stem.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet',
                    default='data/mrral_pixels.parquet',
                    help='Parquet to patch in place (atomic rename).')
    ap.add_argument('--gpkg_dir', default=GPKG_DIR_DEFAULT)
    ap.add_argument('--dry_run', action='store_true',
                    help='Print stats but do not write the parquet.')
    args = ap.parse_args()

    print(f'loading {args.parquet}')
    df = pd.read_parquet(args.parquet)
    print(f'  {len(df):,} rows  ·  {df["tile_id"].nunique()} tiles  ·  '
          f'columns include alteration: {"alteration" in df.columns}')

    # Build (tile_id, polygon_id) → 1 lookup for every alteration polygon
    alt_keys: set[tuple[str, int]] = set()
    gpkgs = sorted(glob.glob(os.path.join(args.gpkg_dir, '*.gpkg')))
    print(f'\nscanning {len(gpkgs)} gpkgs for alteration polygons:')
    n_alt_polys = 0
    for path in gpkgs:
        tid = _tile_id_from_filename(path)
        if tid is None:
            continue  # skip multi-tile gpkgs (north_hellas_*)
        try:
            gdf = gpd.read_file(path)
        except Exception as e:
            print(f'  skip {path}: {e}')
            continue
        if 'Category' not in gdf.columns:
            continue
        cats = gdf['Category'].astype(str).str.lower()
        alt_mask = cats.str.contains('alteration')
        if not alt_mask.any():
            continue
        n_in_tile = int(alt_mask.sum())
        n_alt_polys += n_in_tile
        # polygon_id == row position in the gdf (matches extract_pixels.py)
        for idx in gdf.index[alt_mask]:
            alt_keys.add((tid, int(idx)))
        print(f'  {os.path.basename(path):28s} → {n_in_tile} alteration polygon(s)')
    print(f'\ntotal alteration polygons: {n_alt_polys}')
    print(f'unique (tile, polygon) keys: {len(alt_keys)}')

    if 'alteration' in df.columns:
        existing_alt = int((df['alteration'] > 0.5).sum())
        print(f'\nalteration column already present in parquet '
              f'({existing_alt:,} positive pixels); will overwrite from gpkg scan')
    df['alteration'] = np.float32(0.0)

    # Vectorize: build the lookup as a multi-index, then locate matching rows
    if alt_keys:
        key_series = pd.MultiIndex.from_tuples(list(alt_keys),
                                                 names=['tile_id', 'polygon_id'])
        df_idx = pd.MultiIndex.from_arrays(
            [df['tile_id'].astype(str), df['polygon_id'].astype('int64')],
            names=['tile_id', 'polygon_id'])
        mask = df_idx.isin(key_series)
        n_pixels = int(mask.sum())
        df.loc[mask, 'alteration'] = np.float32(1.0)
        print(f'\nstamped alteration = 1.0 on {n_pixels:,} pixels '
              f'({n_pixels / max(len(df), 1) * 100:.2f}% of the parquet)')

    print()
    print('per-split alteration positive counts:')
    if 'split' in df.columns:
        for split in ('train', 'val', 'test'):
            sub = df[df['split'] == split]
            n_pos = int((sub['alteration'] > 0.5).sum())
            print(f'  {split:>5s}: {n_pos:>7,d} positive of {len(sub):>9,d} rows')
    else:
        print(f'  (no split column)  total alt: '
              f'{int((df["alteration"] > 0.5).sum()):,}')

    if args.dry_run:
        print('\n--dry_run set; not writing.')
        return

    tmp = args.parquet + '.tmp'
    df.to_parquet(tmp, index=False)
    os.replace(tmp, args.parquet)
    print(f'\nwrote {args.parquet}')


if __name__ == '__main__':
    main()

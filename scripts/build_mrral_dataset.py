"""
Extract mrral (59-band, 410-2457 nm) spectra for all labeled polygons.
Writes data/mrral_pixels.parquet with columns m0..m58 plus standard metadata.

Per-tile gating:
  - Non-bland tiles: existing "Other"-categorized polygons are SKIPPED via
    the extract_mrral_pixels_from_pair `other_polygon_ids` parameter (set
    to an empty set). Their mineral-category polygons still contribute.
  - Bland tiles (BLAND_TILES_ORDERED): all polygons (which is just the
    single "Other (High)" polygon authored by build_bland_other_gpkgs.py)
    contribute via other_polygon_ids=None (no filter).
  - Bland-tile rows are subsampled to 113K per tile post-extraction.
  - Bland-tile split assignment is explicit (70/15/15) BEFORE the merge
    against the mrrsu parquet, since these tiles have no rows there.

Spec: docs/superpowers/specs/2026-05-20-relabel-other-bland-tiles-design.md

Usage:
    conda run -n crism python scripts/build_mrral_dataset.py
"""
import os
import sys
import logging

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


BLAND_TILES_ORDERED = [
    't1241', 't1242', 't1243', 't1280',
    't1313', 't1314', 't1315', 't1336',
]
BLAND_TILES = set(BLAND_TILES_ORDERED)

SAMPLE_PER_TILE = 113_000
SPLIT_FRACS = {'train': 0.70, 'val': 0.15, 'test': 0.15}
SEED = 42


def other_polygon_ids_for_tile(tile_id: str):
    """Return the `other_polygon_ids` argument for extract_mrral_pixels_from_pair.

    Bland tiles: None (no filter — all polygons contribute).
    Non-bland tiles: empty set (filter out "Other"-categorized polygons).
    """
    return None if tile_id in BLAND_TILES else set()


def subsample_bland_other_rows(df: pd.DataFrame, sample_per_tile: int = SAMPLE_PER_TILE,
                                seed: int = SEED) -> pd.DataFrame:
    """Randomly subsample bland-tile rows to ~sample_per_tile each.

    Reproducible: per-tile RNG seeded with `seed + i` where `i` is the
    tile's index in BLAND_TILES_ORDERED. Rows from non-bland tiles are
    untouched.
    """
    bland_mask = df['tile_id'].isin(BLAND_TILES)
    keep_idx = list(df.index[~bland_mask])

    for i, tid in enumerate(BLAND_TILES_ORDERED):
        tile_idx = df.index[bland_mask & (df['tile_id'] == tid)].to_numpy()
        if len(tile_idx) == 0:
            continue
        rng = np.random.default_rng(seed + i)
        n_keep = min(sample_per_tile, len(tile_idx))
        chosen = rng.choice(tile_idx, size=n_keep, replace=False)
        keep_idx.extend(chosen.tolist())

    return df.loc[keep_idx].reset_index(drop=True)


def assign_bland_tile_splits(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Overwrite the `split` column for bland-tile rows with a 70/15/15 random
    train/val/test assignment. Non-bland rows are left untouched.

    Per-tile RNG (seed + 100 + i) so the assignment is independent across
    tiles AND independent of the subsampling RNG.
    """
    splits_array = np.array(list(SPLIT_FRACS.keys()))
    probs = np.array(list(SPLIT_FRACS.values()))

    out = df.copy()
    for i, tid in enumerate(BLAND_TILES_ORDERED):
        mask = (out['tile_id'] == tid)
        n = int(mask.sum())
        if n == 0:
            continue
        rng = np.random.default_rng(seed + 100 + i)
        chosen = rng.choice(splits_array, size=n, p=probs)
        out.loc[mask, 'split'] = chosen

    return out


def main():
    from config_loader import load_config
    cfg = load_config()
    from data.extract_pixels import find_mrral_pairs, extract_mrral_pixels_from_pair

    pairs = find_mrral_pairs(cfg['gpkg_dir'], cfg['data_root'])
    logging.info(f"Found {len(pairs)} mrral tile pairs")

    # Re-use same train/val/test split as mrrsu parquet — join on tile+polygon+row+col.
    mrrsu_parquet = os.path.join(cfg['output_dir'], 'pixels.parquet')
    mrrsu_splits = pd.read_parquet(
        mrrsu_parquet,
        columns=['tile_id', 'polygon_id', 'pixel_row', 'pixel_col', 'split'],
    )
    logging.info(f"Loaded {len(mrrsu_splits)} split entries from {mrrsu_parquet}")

    all_records = []
    for i, (tile_id, gpkg_path, mrral_path) in enumerate(pairs):
        gate = other_polygon_ids_for_tile(tile_id)
        gate_desc = 'BLAND (allow all)' if gate is None else 'block Other'
        logging.info(f"[{i+1}/{len(pairs)}] Processing {tile_id}  ({gate_desc})")
        records = extract_mrral_pixels_from_pair(
            tile_id, mrral_path, gpkg_path,
            other_polygon_ids=gate,
        )
        logging.info(f"  {len(records)} pixels extracted")
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    del all_records
    logging.info(f"Total pixels before subsample/merge: {len(df)}")

    df = subsample_bland_other_rows(df)
    logging.info(f"After bland-tile subsample: {len(df)}")

    df = df.merge(
        mrrsu_splits,
        on=['tile_id', 'polygon_id', 'pixel_row', 'pixel_col'],
        how='left',
    )
    df['split'] = df['split'].fillna('train')

    df = assign_bland_tile_splits(df)

    out = os.path.join(cfg['output_dir'], 'mrral_pixels.parquet')
    df.to_parquet(out, index=False)
    logging.info(f"Wrote {len(df)} pixels to {out}")
    logging.info(f"Splits: {df['split'].value_counts().to_dict()}")
    logging.info(f"'other' label total: {int(df['other'].sum())}")
    logging.info(f"'other' by split: {df[df['other']==1]['split'].value_counts().to_dict()}")
    logging.info(f"Columns (first 10): {list(df.columns[:10])}")


if __name__ == '__main__':
    main()

"""
Build the pixel-level dataset from all (gpkg, mrrsu) pairs.

Usage:
    conda run -n crism python scripts/build_dataset.py
    conda run -n crism python scripts/build_dataset.py --config config.yaml
"""
import argparse
import logging
import os
import random
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.extract_pixels import find_tile_pairs, extract_pixels_from_pair

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def assign_tile_splits(tile_ids, train_frac=0.70, val_frac=0.15, seed=42):
    """Assign each tile_id to 'train', 'val', or 'test'."""
    rng = random.Random(seed)
    shuffled = list(tile_ids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    split_map = {}
    for i, tid in enumerate(shuffled):
        if i < n_train:
            split_map[tid] = 'train'
        elif i < n_train + n_val:
            split_map[tid] = 'val'
        else:
            split_map[tid] = 'test'
    return split_map


def sample_other_polygon_ids(pairs, max_polygons, seed=42):
    """
    Randomly select up to max_polygons 'Other' polygon indices across all tiles.

    Returns dict: tile_id -> set of polygon indices to include.
    Tiles not in the returned dict have no sampled Other polygons.
    """
    all_other = []  # list of (tile_id, poly_idx)
    for tile_id, gpkg_path, _ in pairs:
        gdf = gpd.read_file(gpkg_path)
        for idx, row in gdf.iterrows():
            cat = row.get('Category', '')
            if cat and 'other' in str(cat).lower():
                all_other.append((tile_id, idx))

    logger.info(f"Found {len(all_other)} total 'Other' polygons; sampling {min(max_polygons, len(all_other))}")
    rng = random.Random(seed)
    sampled = rng.sample(all_other, min(max_polygons, len(all_other)))

    result = {}
    for tile_id, poly_idx in sampled:
        result.setdefault(tile_id, set()).add(poly_idx)
    return result


def main():
    parser = argparse.ArgumentParser(description="Build pixels.parquet from CRISM tile pairs")
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()

    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.config
    )
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    gpkg_dir = cfg['gpkg_dir']
    data_root = cfg['data_root']
    output_dir = cfg['output_dir']
    other_max = cfg.get('other_max_polygons', 400)
    seed = cfg['split']['random_seed']

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Finding tile pairs...")
    pairs = find_tile_pairs(gpkg_dir, data_root)
    logger.info(f"Found {len(pairs)} tile pairs")

    tile_ids = [p[0] for p in pairs]
    split_map = assign_tile_splits(
        tile_ids,
        train_frac=cfg['split']['train'],
        val_frac=cfg['split']['val'],
        seed=seed,
    )
    n_train = sum(v == 'train' for v in split_map.values())
    n_val = sum(v == 'val' for v in split_map.values())
    n_test = sum(v == 'test' for v in split_map.values())
    logger.info(f"Tile split: {n_train} train / {n_val} val / {n_test} test")

    logger.info("Sampling 'Other' polygons...")
    other_ids = sample_other_polygon_ids(pairs, other_max, seed=seed)

    all_records = []
    for tile_id, gpkg_path, mrrsu_path in tqdm(pairs, desc="Extracting pixels"):
        # Pass sampled Other set; empty set for tiles with no sampled Others
        tile_other_ids = other_ids.get(tile_id, set())
        records = extract_pixels_from_pair(
            tile_id=tile_id,
            mrrsu_path=mrrsu_path,
            gpkg_path=gpkg_path,
            n_bands=60,
            other_polygon_ids=tile_other_ids,
        )
        split = split_map[tile_id]
        for r in records:
            r['split'] = split
        all_records.extend(records)
        logger.info(f"  {tile_id}: {len(records)} pixels -> {split}")

    df = pd.DataFrame(all_records)
    out_path = os.path.join(output_dir, 'pixels.parquet')
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(df)} total pixels to {out_path}")

    # Summary stats
    for split in ['train', 'val', 'test']:
        sub = df[df['split'] == split]
        logger.info(f"  {split}: {len(sub)} pixels from {sub['tile_id'].nunique()} tiles")

    label_cols = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']
    logger.info("Label coverage (pixels with value > 0):")
    for col in label_cols:
        logger.info(f"  {col}: {(df[col] > 0).sum()}")


if __name__ == '__main__':
    main()

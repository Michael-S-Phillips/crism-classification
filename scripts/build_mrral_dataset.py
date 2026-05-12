"""
Extract mrral (59-band, 410-2457 nm) spectra for all labeled polygons.
Writes data/mrral_pixels.parquet with columns m0..m58 plus standard metadata.

Usage:
    conda run -n crism python scripts/build_mrral_dataset.py
"""
import os
import sys
import logging
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    from config_loader import load_config
    cfg = load_config()
    from data.extract_pixels import find_mrral_pairs, extract_mrral_pixels_from_pair

    pairs = find_mrral_pairs(cfg['gpkg_dir'], cfg['data_root'])
    logging.info(f"Found {len(pairs)} mrral tile pairs")

    # Re-use same train/val/test split as mrrsu parquet — join on tile+polygon+row+col.
    # Load only the join keys + split column to keep peak memory low.
    mrrsu_parquet = os.path.join(cfg['output_dir'], 'pixels.parquet')
    mrrsu_splits = pd.read_parquet(
        mrrsu_parquet,
        columns=['tile_id', 'polygon_id', 'pixel_row', 'pixel_col', 'split'],
    )
    logging.info(f"Loaded {len(mrrsu_splits)} split entries from {mrrsu_parquet}")

    all_records = []
    for i, (tile_id, gpkg_path, mrral_path) in enumerate(pairs):
        logging.info(f"[{i+1}/{len(pairs)}] Processing {tile_id}")
        records = extract_mrral_pixels_from_pair(tile_id, mrral_path, gpkg_path)
        logging.info(f"  {len(records)} pixels extracted")
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    del all_records
    logging.info(f"Total pixels before split assignment: {len(df)}")

    # Vectorised left-join to assign split. Pixels not present in the mrrsu
    # parquet default to 'train'. Using merge instead of df.apply(axis=1) avoids
    # per-row Python lambdas — required for the 1.7M-row scale here (df.apply
    # was triggering the OOM killer).
    df = df.merge(
        mrrsu_splits,
        on=['tile_id', 'polygon_id', 'pixel_row', 'pixel_col'],
        how='left',
    )
    df['split'] = df['split'].fillna('train')

    out = os.path.join(cfg['output_dir'], 'mrral_pixels.parquet')
    df.to_parquet(out, index=False)
    logging.info(f"Wrote {len(df)} pixels to {out}")
    logging.info(f"Splits: {df['split'].value_counts().to_dict()}")
    logging.info(f"Columns (first 10): {list(df.columns[:10])}")


if __name__ == '__main__':
    main()

"""
Extract mrral (59-band, 410-2457 nm) spectra for all labeled polygons.
Writes data/mrral_pixels.parquet with columns m0..m58 plus standard metadata.

Usage:
    conda run -n crism python scripts/build_mrral_dataset.py
"""
import os
import sys
import logging
import yaml
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    cfg = yaml.safe_load(open(os.path.join(PROJ, 'config.yaml')))
    from data.extract_pixels import find_mrral_pairs, extract_mrral_pixels_from_pair

    pairs = find_mrral_pairs(cfg['gpkg_dir'], cfg['data_root'])
    logging.info(f"Found {len(pairs)} mrral tile pairs")

    # Re-use same train/val/test split as mrrsu parquet — join on tile+polygon+row+col
    mrrsu_parquet = os.path.join(cfg['output_dir'], 'pixels.parquet')
    mrrsu_df = pd.read_parquet(mrrsu_parquet)
    split_map = mrrsu_df.set_index(
        ['tile_id', 'polygon_id', 'pixel_row', 'pixel_col']
    )['split'].to_dict()
    logging.info(f"Loaded split map from {mrrsu_parquet} ({len(split_map)} entries)")

    all_records = []
    for i, (tile_id, gpkg_path, mrral_path) in enumerate(pairs):
        logging.info(f"[{i+1}/{len(pairs)}] Processing {tile_id}")
        records = extract_mrral_pixels_from_pair(tile_id, mrral_path, gpkg_path)
        logging.info(f"  {len(records)} pixels extracted")
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    logging.info(f"Total pixels before split assignment: {len(df)}")

    # Assign split from mrrsu parquet; default 'train' for any new pixels
    df['split'] = df.apply(
        lambda r: split_map.get(
            (r['tile_id'], int(r['polygon_id']), int(r['pixel_row']), int(r['pixel_col'])),
            'train'
        ),
        axis=1
    )

    out = os.path.join(cfg['output_dir'], 'mrral_pixels.parquet')
    df.to_parquet(out, index=False)
    logging.info(f"Wrote {len(df)} pixels to {out}")
    logging.info(f"Splits: {df['split'].value_counts().to_dict()}")
    logging.info(f"Columns (first 10): {list(df.columns[:10])}")


if __name__ == '__main__':
    main()

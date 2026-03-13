"""
Extract mrral spectra for Hellas basin mineral polygons and append to mrral_pixels.parquet.

The Hellas GPKG (north_hellas_mafics_geometries_fixed.gpkg) differs from the
per-tile T#### GPKGs in three ways:
  1. Single file covering many tiles — we spatially clip polygons per tile.
  2. Uses 'Interpreta' column (Olivine/LCP/HCP/Plagioclase) not 'Category'.
  3. No confidence annotation — all detections default to Moderate.

CRS handling:
  The GPKG is stored with an "Undefined geographic SRS" (WGS84 parameters) but
  the coordinates are actually Mars geographic lon/lat. We override the CRS to
  the Mars geographic base CRS derived from each tile before reprojecting.

Olivine handling:
  'Olivine' without type 1/2 distinction → olivine_t1=0.5, olivine_t2=0.5
  (soft label, consistent with label_parser._TOKEN_MAP['olivine']).

Split assignment:
  All Hellas pixels are assigned to 'train'. The existing val/test splits come
  from geographically distinct tiles and are unchanged.

Usage:
    conda run -n crism python scripts/build_hellas_dataset.py
    conda run -n crism python scripts/build_hellas_dataset.py --dry_run
"""
import argparse
import glob
import logging
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HELLAS_GPKG_NAME = 'north_hellas_mafics_geometries_fixed.gpkg'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', action='store_true',
                        help='Count intersecting tiles without extracting pixels')
    args = parser.parse_args()

    from config_loader import load_config
    cfg = load_config()
    gpkg_dir = cfg['gpkg_dir']
    data_root = cfg['data_root']
    out_parquet = os.path.join(cfg['output_dir'], 'mrral_pixels.parquet')

    hellas_gpkg = os.path.join(gpkg_dir, HELLAS_GPKG_NAME)
    if not os.path.exists(hellas_gpkg):
        logging.error(f"Hellas GPKG not found: {hellas_gpkg}")
        sys.exit(1)

    from data.extract_pixels import load_hellas_gdf, extract_hellas_pixels_from_tile

    logging.info(f"Loading Hellas GPKG: {hellas_gpkg}")
    hellas_gdf = load_hellas_gdf(hellas_gpkg)
    cat_counts = hellas_gdf['Category'].str.split(' (', regex=False).str[0].value_counts().to_dict()
    logging.info(f"  {len(hellas_gdf)} polygons, categories: {cat_counts}")

    # Find all mrral tiles
    mrral_files = sorted(glob.glob(
        os.path.join(data_root, '**', '*mrral*.img'), recursive=True
    ))
    logging.info(f"Found {len(mrral_files)} mrral tiles to check")

    all_records = []
    n_tiles_with_data = 0

    for i, mrral_path in enumerate(mrral_files):
        tile_id = os.path.basename(mrral_path).split('_mrral')[0].lower()

        if args.dry_run:
            # Quick spatial check without reading raster data
            import rasterio
            from shapely.geometry import box
            try:
                with rasterio.open(mrral_path) as src:
                    if src.crs is None:
                        continue
                    from pyproj import CRS as ProjCRS
                    mars_geog_crs = ProjCRS.from_wkt(src.crs.to_wkt()).geodetic_crs
                    trial_gdf = hellas_gdf.set_crs(
                        mars_geog_crs, allow_override=True
                    ).to_crs(src.crs)
                    tile_box = box(*src.bounds)
                    n_intersect = trial_gdf.intersects(tile_box).sum()
                    if n_intersect > 0:
                        logging.info(f"  DRY: {tile_id} — {n_intersect} polygons would be extracted")
                        n_tiles_with_data += 1
            except Exception as e:
                logging.warning(f"  DRY: {tile_id} error: {e}")
            continue

        records = extract_hellas_pixels_from_tile(tile_id, mrral_path, hellas_gdf)
        if records:
            all_records.extend(records)
            n_tiles_with_data += 1
            logging.info(f"[{i+1}/{len(mrral_files)}] {tile_id}: {len(records)} pixels extracted")

    if args.dry_run:
        logging.info(f"\nDRY RUN complete: {n_tiles_with_data} tiles would have data extracted")
        return

    if not all_records:
        logging.warning("No pixels extracted — check CRS and spatial overlap.")
        return

    hellas_df = pd.DataFrame(all_records)
    hellas_df['split'] = 'train'
    logging.info(f"\nExtracted {len(hellas_df)} Hellas pixels from {n_tiles_with_data} tiles")
    logging.info(f"Label distribution:")
    from data.dataset import LABEL_COLS
    for col in LABEL_COLS:
        if col in hellas_df.columns:
            n_pos = (hellas_df[col] > 0.4).sum()
            logging.info(f"  {col}: {n_pos} positive pixels ({100*n_pos/len(hellas_df):.1f}%)")

    # Append to existing mrral_pixels.parquet
    if os.path.exists(out_parquet):
        existing_df = pd.read_parquet(out_parquet)
        logging.info(f"Existing parquet: {len(existing_df)} pixels")

        # Deduplicate: drop any existing pixels from Hellas tiles (in case of re-run)
        hellas_tile_ids = set(hellas_df['tile_id'].unique())
        existing_df = existing_df[~existing_df['tile_id'].isin(hellas_tile_ids)]
        logging.info(f"After removing stale Hellas entries: {len(existing_df)} pixels")

        combined_df = pd.concat([existing_df, hellas_df], ignore_index=True)
    else:
        logging.warning(f"No existing parquet found at {out_parquet} — writing Hellas-only")
        combined_df = hellas_df

    combined_df.to_parquet(out_parquet, index=False)
    logging.info(f"Wrote {len(combined_df)} total pixels to {out_parquet}")
    logging.info(f"Splits: {combined_df['split'].value_counts().to_dict()}")
    logging.info(f"Confidence tiers: {combined_df['confidence_tier'].value_counts().to_dict()}")


if __name__ == '__main__':
    main()

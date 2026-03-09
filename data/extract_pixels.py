# crism_classification/data/extract_pixels.py
"""
Extract per-pixel spectral parameter values from CRISM mrrsu rasters
using geopackage polygon ROIs as masks.
"""
import os
import glob
import logging
from typing import List, Tuple, Dict, Any, Optional, Set

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize

from data.label_parser import parse_category, get_confidence_tier, CLASSES

logger = logging.getLogger(__name__)

NODATA_VALUE = 65535


# GPKGs that cover multiple tiles (not tile-specific) — skip in find_*_pairs
_REGION_GPKG_PREFIXES = ('north_hellas',)


def _is_region_gpkg(fname: str) -> bool:
    """Return True for region-wide GPKGs that don't map 1:1 to a tile."""
    return any(fname.lower().startswith(p) for p in _REGION_GPKG_PREFIXES)


def find_tile_pairs(
    gpkg_dir: str,
    data_root: str
) -> List[Tuple[str, str, str]]:
    """
    Find (tile_id, gpkg_path, mrrsu_path) triples by matching gpkg filenames
    to mrrsu image files under data_root.

    Returns list sorted by tile_id.
    """
    pairs = []
    for fname in sorted(os.listdir(gpkg_dir)):
        if not fname.endswith('.gpkg'):
            continue
        if _is_region_gpkg(fname):
            continue
        tile_id = fname.replace('.gpkg', '').lower()
        gpkg_path = os.path.join(gpkg_dir, fname)
        matches = glob.glob(
            os.path.join(data_root, '**', f'{tile_id}_mrrsu*.img'),
            recursive=True
        )
        if not matches:
            logger.warning(f"No mrrsu file found for {tile_id}, skipping.")
            continue
        matches = sorted(matches)
        if len(matches) > 1:
            logger.warning(f"Multiple mrrsu files for {tile_id}, using: {matches[0]}")
        pairs.append((tile_id, gpkg_path, matches[0]))
    return pairs


MRRAL_N_BANDS = 59  # bands 0-58 (0-indexed), wavelengths 410-2457 nm
                    # bands 59-71 (2530-3923 nm) excluded — too noisy


def find_mrral_pairs(
    gpkg_dir: str,
    data_root: str
) -> List[Tuple[str, str, str]]:
    """
    Find (tile_id, gpkg_path, mrral_path) triples.
    Mirrors find_tile_pairs but matches mrral instead of mrrsu files.
    """
    pairs = []
    for fname in sorted(os.listdir(gpkg_dir)):
        if not fname.endswith('.gpkg'):
            continue
        if _is_region_gpkg(fname):
            continue
        tile_id = fname.replace('.gpkg', '').lower()
        gpkg_path = os.path.join(gpkg_dir, fname)
        matches = glob.glob(
            os.path.join(data_root, '**', f'{tile_id}_mrral*.img'),
            recursive=True
        )
        if not matches:
            logger.warning(f"No mrral file found for {tile_id}, skipping.")
            continue
        if len(matches) > 1:
            logger.warning(f"Multiple mrral files for {tile_id}, using: {sorted(matches)[0]}")
        pairs.append((tile_id, gpkg_path, sorted(matches)[0]))
    return pairs


def extract_mrral_pixels_from_pair(
    tile_id: str,
    mrral_path: str,
    gpkg_path: str,
    other_polygon_ids: Optional[Set] = None,
    gdf=None,
) -> List[Dict[str, Any]]:
    """
    Extract per-pixel mrral spectra (59 bands, 410-2457 nm) from one tile.

    Identical logic to extract_pixels_from_pair but:
    - reads mrral file instead of mrrsu
    - stores bands as m0..m58 (not b0..b59)
    - reads exactly MRRAL_N_BANDS=59 bands regardless of file band count
    """
    records = []

    with rasterio.open(mrral_path) as src:
        raster_crs = src.crs
        transform = src.transform
        height, width = src.height, src.width
        actual_bands = min(MRRAL_N_BANDS, src.count)

        if gdf is None:
            gdf = gpd.read_file(gpkg_path)
        if gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)

        for poly_idx, row in gdf.iterrows():
            category = row.get('Category', '')
            if not category or isinstance(category, float):
                continue
            if other_polygon_ids is not None and 'other' in category.lower():
                if poly_idx not in other_polygon_ids:
                    continue
            try:
                label, conf_weight = parse_category(str(category))
            except ValueError:
                logger.warning(f"Could not parse {category!r} in {tile_id}, skipping.")
                continue
            conf_tier = get_confidence_tier(str(category))
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            try:
                mask = rasterize(
                    [(geom, 1)], out_shape=(height, width),
                    transform=transform, fill=0, dtype=np.uint8
                ).astype(bool)
            except Exception as e:
                logger.warning(f"Rasterize failed polygon {poly_idx} in {tile_id}: {e}")
                continue

            pixel_rows, pixel_cols = np.where(mask)
            if len(pixel_rows) == 0:
                continue

            row_min, row_max = int(pixel_rows.min()), int(pixel_rows.max()) + 1
            col_min, col_max = int(pixel_cols.min()), int(pixel_cols.max()) + 1
            window = rasterio.windows.Window(
                col_min, row_min, col_max - col_min, row_max - row_min
            )
            chunk = src.read(list(range(1, actual_bands + 1)), window=window)

            for r, c in zip(pixel_rows, pixel_cols):
                pixel_vals = chunk[:, r - row_min, c - col_min]
                if np.any(pixel_vals >= NODATA_VALUE):
                    continue
                if np.any(np.isnan(pixel_vals)):
                    pixel_vals = np.nan_to_num(pixel_vals, nan=0.0)
                record: Dict[str, Any] = {
                    'tile_id': tile_id,
                    'polygon_id': int(poly_idx),
                    'pixel_row': int(r),
                    'pixel_col': int(c),
                }
                for b_idx in range(actual_bands):
                    record[f'm{b_idx}'] = float(pixel_vals[b_idx])
                for cls_idx, cls_name in enumerate(CLASSES):
                    record[cls_name] = float(label[cls_idx])
                record['confidence_weight'] = float(conf_weight)
                record['confidence_tier'] = conf_tier
                records.append(record)

    return records


def extract_pixels_from_pair(
    tile_id: str,
    mrrsu_path: str,
    gpkg_path: str,
    n_bands: int = 60,
    other_polygon_ids: Optional[Set] = None,
    gdf=None,
) -> List[Dict[str, Any]]:
    """
    Extract per-pixel records from one (gpkg, mrrsu) pair.

    For each polygon:
      - Rasterizes the polygon to a pixel mask aligned to the mrrsu grid
      - Reads all n_bands values at each masked pixel
      - Drops pixels where any band >= NODATA_VALUE; imputes NaN bands to 0.0
      - Parses the Category string into multi-hot labels + confidence weight

    Parameters
    ----------
    tile_id : str
    mrrsu_path : str
    gpkg_path : str
    n_bands : int
    other_polygon_ids : set, optional
        If provided, only 'Other' polygons whose index is in this set are included.
        Pass an empty set to exclude all 'Other' polygons.
        Pass None to include all 'Other' polygons (no filtering).
    gdf : GeoDataFrame, optional
        Pre-loaded GeoDataFrame for gpkg_path. If None, the file is read from disk.

    Returns
    -------
    List of dicts, one per valid pixel.
    """
    records = []

    with rasterio.open(mrrsu_path) as src:
        raster_crs = src.crs
        transform = src.transform
        height, width = src.height, src.width
        actual_bands = min(n_bands, src.count)

        if gdf is None:
            gdf = gpd.read_file(gpkg_path)
        if gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)

        for poly_idx, row in gdf.iterrows():
            category = row.get('Category', '')
            if not category or (isinstance(category, float)):
                continue

            # Filter Other polygons if a sampling set is specified
            if other_polygon_ids is not None and 'other' in category.lower():
                if poly_idx not in other_polygon_ids:
                    continue

            try:
                label, conf_weight = parse_category(str(category))
            except ValueError:
                logger.warning(f"Could not parse category {category!r} in {tile_id}, skipping.")
                continue

            conf_tier = get_confidence_tier(str(category))

            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            try:
                mask = rasterize(
                    [(geom, 1)],
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8
                ).astype(bool)
            except Exception as e:
                logger.warning(f"Rasterize failed for polygon {poly_idx} in {tile_id}: {e}")
                continue

            pixel_rows, pixel_cols = np.where(mask)
            if len(pixel_rows) == 0:
                continue

            # Windowed read for efficiency
            row_min, row_max = int(pixel_rows.min()), int(pixel_rows.max()) + 1
            col_min, col_max = int(pixel_cols.min()), int(pixel_cols.max()) + 1
            window = rasterio.windows.Window(
                col_min, row_min,
                col_max - col_min, row_max - row_min
            )
            chunk = src.read(
                list(range(1, actual_bands + 1)),
                window=window
            )  # (bands, h, w)

            for r, c in zip(pixel_rows, pixel_cols):
                local_r = r - row_min
                local_c = c - col_min
                pixel_vals = chunk[:, local_r, local_c]

                if np.any(pixel_vals >= NODATA_VALUE):
                    continue
                # Impute NaN bands (e.g. BD640_2 / band 5 is undefined in CRISM MRDR)
                if np.any(np.isnan(pixel_vals)):
                    pixel_vals = np.nan_to_num(pixel_vals, nan=0.0)

                record: Dict[str, Any] = {
                    'tile_id': tile_id,
                    'polygon_id': int(poly_idx),
                    'pixel_row': int(r),
                    'pixel_col': int(c),
                }
                for b_idx in range(actual_bands):
                    record[f'b{b_idx}'] = float(pixel_vals[b_idx])
                for cls_idx, cls_name in enumerate(CLASSES):
                    record[cls_name] = float(label[cls_idx])
                record['confidence_weight'] = float(conf_weight)
                record['confidence_tier'] = conf_tier
                records.append(record)

    return records


def load_hellas_gdf(hellas_gpkg: str, category_col: str = 'Interpreta') -> gpd.GeoDataFrame:
    """
    Load and validate the Hellas region GPKG.

    The file uses an "Undefined geographic SRS" with WGS84 parameters, but the
    coordinates are actually Mars geographic (lon/lat). The CRS metadata is
    deliberately left unset here — callers must assign the Mars geographic CRS
    for each target tile via set_crs(tile_crs.geodetic_crs, allow_override=True).

    Returns the GeoDataFrame with a 'Category' column synthesised from
    the Interpreta field with default '(Moderate)' confidence appended.
    """
    gdf = gpd.read_file(hellas_gpkg)
    if category_col not in gdf.columns:
        raise ValueError(
            f"Expected column '{category_col}' in {hellas_gpkg}. "
            f"Available: {list(gdf.columns)}"
        )
    # Synthesise a Category string matching the existing label_parser format.
    # - No type distinction for olivine → parse_category maps 'Olivine' to
    #   olivine_t1=0.5, olivine_t2=0.5 (soft label for both types).
    # - No confidence metadata → default Moderate (weight=0.5, tier='Moderate').
    gdf = gdf[gdf[category_col].notna() & (gdf[category_col] != '')].copy()
    gdf['Category'] = gdf[category_col].astype(str) + ' (Moderate)'
    return gdf


def extract_hellas_pixels_from_tile(
    tile_id: str,
    mrral_path: str,
    hellas_gdf: gpd.GeoDataFrame,
) -> List[Dict[str, Any]]:
    """
    Extract mrral pixels from Hellas region polygons that intersect one tile.

    The Hellas GPKG coordinates are Mars geographic (lon/lat) stored with an
    incorrect "Undefined geographic SRS" WKT. We fix this by overriding the
    CRS to the Mars geographic base CRS derived from the tile's projected CRS,
    then reprojecting to the tile's coordinate system.

    Parameters
    ----------
    tile_id : str
        Tile identifier string (used for record metadata).
    mrral_path : str
        Path to the mrral .img file for this tile.
    hellas_gdf : GeoDataFrame
        Loaded from load_hellas_gdf(). Must have a 'Category' column.
    """
    from shapely.geometry import box

    records = []

    with rasterio.open(mrral_path) as src:
        raster_crs = src.crs
        if raster_crs is None:
            logger.warning(f"Tile {tile_id} has no CRS — skipping Hellas extraction.")
            return records

        transform = src.transform
        height, width = src.height, src.width
        actual_bands = min(MRRAL_N_BANDS, src.count)

        # The Hellas GPKG has Mars geographic coordinates stored with a wrong
        # "Undefined geographic SRS" CRS definition. Override with the actual
        # Mars geographic CRS (the base geographic CRS of this tile).
        # rasterio.CRS doesn't have geodetic_crs; use pyproj.CRS for that.
        from pyproj import CRS as ProjCRS
        mars_geog_crs = ProjCRS.from_wkt(raster_crs.to_wkt()).geodetic_crs
        try:
            tile_gdf = hellas_gdf.set_crs(mars_geog_crs, allow_override=True).to_crs(raster_crs)
        except Exception as e:
            logger.warning(f"CRS reprojection failed for tile {tile_id}: {e}")
            return records

        # Spatial filter: keep only polygons that intersect this tile's extent
        tile_box = box(*src.bounds)
        tile_gdf = tile_gdf[tile_gdf.intersects(tile_box)]
        if len(tile_gdf) == 0:
            return records

        logger.info(f"  {tile_id}: {len(tile_gdf)} Hellas polygons intersect tile")

        for poly_idx, row in tile_gdf.iterrows():
            category = row.get('Category', '')
            if not category or isinstance(category, float):
                continue
            try:
                label, conf_weight = parse_category(str(category))
            except ValueError:
                logger.warning(f"Could not parse {category!r} in {tile_id}, skipping.")
                continue
            conf_tier = get_confidence_tier(str(category))
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            try:
                mask = rasterize(
                    [(geom, 1)], out_shape=(height, width),
                    transform=transform, fill=0, dtype=np.uint8
                ).astype(bool)
            except Exception as e:
                logger.warning(f"Rasterize failed polygon {poly_idx} in {tile_id}: {e}")
                continue

            pixel_rows, pixel_cols = np.where(mask)
            if len(pixel_rows) == 0:
                continue

            row_min, row_max = int(pixel_rows.min()), int(pixel_rows.max()) + 1
            col_min, col_max = int(pixel_cols.min()), int(pixel_cols.max()) + 1
            window = rasterio.windows.Window(
                col_min, row_min, col_max - col_min, row_max - row_min
            )
            chunk = src.read(list(range(1, actual_bands + 1)), window=window)

            for r, c in zip(pixel_rows, pixel_cols):
                pixel_vals = chunk[:, r - row_min, c - col_min]
                if np.any(pixel_vals >= NODATA_VALUE):
                    continue
                if np.any(np.isnan(pixel_vals)):
                    pixel_vals = np.nan_to_num(pixel_vals, nan=0.0)
                record: Dict[str, Any] = {
                    'tile_id': tile_id,
                    'polygon_id': int(poly_idx),
                    'pixel_row': int(r),
                    'pixel_col': int(c),
                }
                for b_idx in range(actual_bands):
                    record[f'm{b_idx}'] = float(pixel_vals[b_idx])
                for cls_idx, cls_name in enumerate(CLASSES):
                    record[cls_name] = float(label[cls_idx])
                record['confidence_weight'] = float(conf_weight)
                record['confidence_tier'] = conf_tier
                records.append(record)

    return records

"""Reads polygon-interior pixel spectra from an mrral tile."""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pyproj
import rasterio
import rasterio.features
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

NODATA = 65535
N_BANDS = 59


@dataclass(frozen=True)
class PixelBundle:
    rows: np.ndarray       # (n_pixels,) int64 — tile row index
    cols: np.ndarray       # (n_pixels,) int64 — tile col index
    spectra: np.ndarray    # (n_pixels, 59) float32 — reflectance
    mean: np.ndarray       # (59,) float32
    std: np.ndarray        # (59,) float32


def _find_mrral_img(tile_id: str, mrral_dir: str) -> str:
    pattern = os.path.join(mrral_dir, f'{tile_id}_mrral_*.img')
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f'no mrral .img for tile_id={tile_id} in {mrral_dir}')
    return matches[0]


def load_polygon_pixels(
    geometry: BaseGeometry,
    tile_id: str,
    mrral_dir: str,
    source_crs: Optional[Union[str, pyproj.CRS]] = None,
) -> PixelBundle:
    """Pull all interior pixel spectra for a polygon from its mrral tile.

    ``source_crs`` is the geometry's CRS (gpkg CRS). If provided and it differs
    from the tile CRS, the geometry is reprojected before rasterizing. mc13
    vector outputs are in geographic degrees while mrral tiles are per-tile
    equirectangular meters — passing ``source_crs`` is required for those.
    """
    img_path = _find_mrral_img(tile_id, mrral_dir)
    empty = PixelBundle(
        rows=np.zeros(0, dtype=np.int64),
        cols=np.zeros(0, dtype=np.int64),
        spectra=np.zeros((0, N_BANDS), dtype=np.float32),
        mean=np.zeros(N_BANDS, dtype=np.float32),
        std=np.zeros(N_BANDS, dtype=np.float32),
    )

    with rasterio.open(img_path) as src:
        if source_crs is not None and src.crs is not None:
            src_crs = pyproj.CRS.from_user_input(source_crs)
            dst_crs = pyproj.CRS.from_user_input(src.crs.to_wkt())
            if src_crs != dst_crs:
                transformer = pyproj.Transformer.from_crs(
                    src_crs, dst_crs, always_xy=True)
                geometry = shapely_transform(transformer.transform, geometry)
        # Rasterize the polygon onto the tile grid → boolean mask
        mask = rasterio.features.rasterize(
            [(geometry, 1)],
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            dtype='uint8',
        ).astype(bool)
        if not mask.any():
            return empty

        # Read all 59 bands once → (n_bands, h, w)
        cube = src.read(list(range(1, N_BANDS + 1))).astype(np.float32)

    # Build NODATA mask (any band == NODATA → drop that pixel)
    nodata_mask = (cube == NODATA).any(axis=0)
    keep = mask & ~nodata_mask
    if not keep.any():
        return empty

    rows, cols = np.where(keep)
    spectra = cube[:, rows, cols].T.copy()           # (n_pixels, 59)
    return PixelBundle(
        rows=rows.astype(np.int64),
        cols=cols.astype(np.int64),
        spectra=spectra,
        mean=spectra.mean(axis=0),
        std=spectra.std(axis=0),
    )

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
import rasterio.windows
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

NODATA = 65535
N_BANDS = 59

# False-color RGB band indices for the thumbnail. Chosen to spread across the
# 0.41-2.46 µm range so different minerals show contrast. 0-indexed against
# the 59-band mrral cube.
_RGB_BAND_INDICES = (50, 25, 10)  # ~2.2 µm, ~1.5 µm, ~0.8 µm


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


def _percentile_stretch(arr: np.ndarray, lo: float = 2.0,
                         hi: float = 98.0) -> np.ndarray:
    """Map array to [0, 1] using percentile clip; NaN-safe."""
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr)
    lo_v, hi_v = np.percentile(valid, [lo, hi])
    if hi_v <= lo_v:
        return np.zeros_like(arr)
    out = (arr - lo_v) / (hi_v - lo_v)
    return np.clip(out, 0.0, 1.0)


@dataclass(frozen=True)
class Thumbnail:
    rgb: np.ndarray        # (h, w, 3) uint8 — false-color crop with outline
    polygon_xy: np.ndarray # (n_vertices, 2) float — polygon outline in crop pixel coords


def load_thumbnail(
    geometry: BaseGeometry,
    tile_id: str,
    mrral_dir: str,
    source_crs: Optional[Union[str, pyproj.CRS]] = None,
    pad_factor: float = 4.0,
    min_pad_pixels: int = 30,
) -> Thumbnail:
    """Cropped false-color RGB of the polygon's neighborhood + outline coords.

    Reads only 3 bands within a windowed crop, so this is cheap even on the
    network-mounted tiles. ``pad_factor`` expands the polygon bbox; ``min_pad``
    ensures small polygons still get geological context.
    """
    img_path = _find_mrral_img(tile_id, mrral_dir)
    with rasterio.open(img_path) as src:
        if source_crs is not None and src.crs is not None:
            src_crs = pyproj.CRS.from_user_input(source_crs)
            dst_crs = pyproj.CRS.from_user_input(src.crs.to_wkt())
            if src_crs != dst_crs:
                transformer = pyproj.Transformer.from_crs(
                    src_crs, dst_crs, always_xy=True)
                geometry = shapely_transform(transformer.transform, geometry)

        # Polygon bbox → pixel bbox (rows/cols may be reversed if tile is
        # north-up vs south-up; normalize at the end).
        minx, miny, maxx, maxy = geometry.bounds
        inv = ~src.transform
        c0a, r0a = inv * (minx, maxy)
        c1a, r1a = inv * (maxx, miny)
        c0, c1 = sorted((c0a, c1a))
        r0, r1 = sorted((r0a, r1a))

        # Pad
        bbox_w = max(c1 - c0, 1.0)
        bbox_h = max(r1 - r0, 1.0)
        pad = max(bbox_w, bbox_h) * (pad_factor - 1) / 2
        pad = max(pad, min_pad_pixels)
        c0 = max(0, int(np.floor(c0 - pad)))
        r0 = max(0, int(np.floor(r0 - pad)))
        c1 = min(src.width, int(np.ceil(c1 + pad)))
        r1 = min(src.height, int(np.ceil(r1 + pad)))
        w = max(c1 - c0, 1)
        h = max(r1 - r0, 1)

        window = rasterio.windows.Window(c0, r0, w, h)
        chans = []
        for b_idx in _RGB_BAND_INDICES:
            arr = src.read(b_idx + 1, window=window).astype(np.float32)
            arr[(arr == NODATA) | ~np.isfinite(arr)] = np.nan
            chans.append(arr)

        # Rasterize the polygon edge onto the crop for overlay
        crop_transform = rasterio.windows.transform(window, src.transform)
        outline_mask = rasterio.features.rasterize(
            [(geometry.boundary, 1)],
            out_shape=(h, w),
            transform=crop_transform,
            fill=0, dtype='uint8',
        ).astype(bool)

    # Stack RGB with per-band percentile stretch
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for i, c in enumerate(chans):
        rgb[..., i] = _percentile_stretch(c)
    # NaN regions → black
    rgb[np.isnan(rgb)] = 0.0
    rgb_u8 = (rgb * 255).astype(np.uint8)

    # Burn the outline in saturated red (2-pixel dilation for visibility)
    if outline_mask.any():
        # Cheap dilation: shift in each cardinal direction
        thick = outline_mask.copy()
        thick[1:, :] |= outline_mask[:-1, :]
        thick[:-1, :] |= outline_mask[1:, :]
        thick[:, 1:] |= outline_mask[:, :-1]
        thick[:, :-1] |= outline_mask[:, 1:]
        rgb_u8[thick] = (255, 30, 30)

    # Polygon vertices in crop-pixel coordinates (for callers who want to
    # draw the outline themselves, e.g., with plotly).
    if hasattr(geometry, 'exterior') and geometry.exterior is not None:
        coords = np.asarray(geometry.exterior.coords)
        inv_crop = ~crop_transform
        xy = np.array([inv_crop * (x, y) for x, y in coords])
    else:
        xy = np.zeros((0, 2))

    return Thumbnail(rgb=rgb_u8, polygon_xy=xy)

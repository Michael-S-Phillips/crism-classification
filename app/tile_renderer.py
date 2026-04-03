"""Render CRISM mrral tile as false-color PNG for browser display."""
import io
from typing import Any

import numpy as np
import rasterio
import rasterio.enums
from PIL import Image

from app.config import CRISM_NODATA, FC_BANDS, TILE_DISPLAY_MAX_PX


def _percentile_stretch(band: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    """Stretch finite values to [0,255] uint8; NaN pixels → 0."""
    finite = band[np.isfinite(band)]
    if len(finite) == 0:
        return np.zeros_like(band, dtype=np.uint8)
    vmin = float(np.percentile(finite, lo))
    vmax = float(np.percentile(finite, hi))
    if vmax == vmin:
        return np.zeros_like(band, dtype=np.uint8)
    # Only compute stretched for finite values
    mask = np.isfinite(band)
    out = np.zeros_like(band, dtype=np.uint8)
    if np.any(mask):
        stretched = np.clip((band[mask] - vmin) / (vmax - vmin), 0, 1)
        out[mask] = (stretched * 255).astype(np.uint8)
    return out


def render_false_color(img_path: str) -> tuple[bytes, dict[str, Any]]:
    """Return (PNG bytes, metadata dict) for the tile.

    metadata keys: width, height, src_width, src_height, scale_x, scale_y
    """
    with rasterio.open(img_path) as src:
        src_h, src_w = src.height, src.width
        scale = min(1.0, TILE_DISPLAY_MAX_PX / max(src_h, src_w))
        out_h = max(1, int(src_h * scale))
        out_w = max(1, int(src_w * scale))

        bands_u8 = []
        alpha = None
        for band_idx in FC_BANDS:
            raw = src.read(
                band_idx,
                out_shape=(out_h, out_w),
                resampling=rasterio.enums.Resampling.bilinear,
            ).astype(np.float32)
            raw[raw == CRISM_NODATA] = np.nan
            raw[np.abs(raw) > 1] = np.nan
            if alpha is None:
                alpha = np.where(np.isfinite(raw), 255, 0).astype(np.uint8)
            bands_u8.append(_percentile_stretch(raw))

    rgba = np.stack(bands_u8 + [alpha], axis=-1)  # (H, W, 4)
    img = Image.fromarray(rgba, mode='RGBA')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    png_bytes = buf.getvalue()

    meta: dict[str, Any] = {
        'width':      out_w,
        'height':     out_h,
        'src_width':  src_w,
        'src_height': src_h,
        'scale_x':    out_w / src_w,
        'scale_y':    out_h / src_h,
    }
    return png_bytes, meta


def px_to_rowcol(img_x: float, img_y: float, meta: dict[str, Any]) -> tuple[int, int]:
    """Convert display-image pixel (x=col, y=row) to source raster (row, col)."""
    src_col = img_x / meta['scale_x']
    src_row = img_y / meta['scale_y']
    return int(src_row), int(src_col)

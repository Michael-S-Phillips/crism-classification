"""Band-name registry and cube reader for mrrsu summary-parameter tiles.

Indices are resolved from the tile's OWN header rather than hardcoded: a tile
written with a different band order would otherwise silently shift every
parameter, and a band-depth threshold applied to the wrong band produces a
plausible map with no error.
"""
from __future__ import annotations

import re

import numpy as np
import rasterio

from data.mrrsu_aux import NODATA

N_MRRSU_BANDS = 60

# Documented in CLAUDE.md; asserted against a real header in the tests so a
# reordered product is caught rather than absorbed.
CORE_INDICES = {'OLINDEX3': 15, 'BD1300': 17, 'LCPINDEX2': 18, 'HCPINDEX2': 19}


def read_band_names(hdr_path: str) -> list[str]:
    txt = open(hdr_path).read()
    m = re.search(r'band names\s*=\s*\{(.*?)\}', txt, re.S)
    if not m:
        raise ValueError(f'{hdr_path}: no "band names" block')
    return [n.strip() for n in m.group(1).replace('\n', ' ').split(',') if n.strip()]


def band_index(names: list[str], param: str) -> int:
    try:
        return names.index(param)
    except ValueError:
        raise KeyError(
            f'{param} not among the {len(names)} mrrsu band names') from None


def read_mrrsu_cube(img_path: str) -> tuple[np.ndarray, list[str]]:
    """(H, W, 60) float32 with nodata as NaN, plus the band-name list."""
    names = read_band_names(img_path.replace('.img', '.hdr'))
    with rasterio.open(img_path) as src:
        data = src.read(list(range(1, N_MRRSU_BANDS + 1))).astype(np.float32)
    data[(data == NODATA) | ~np.isfinite(data)] = np.nan
    return data.transpose(1, 2, 0), names

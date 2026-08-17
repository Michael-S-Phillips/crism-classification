"""Turn a probs .npz into a georeferenced multi-band GeoTIFF for QGIS.

The classifier writes probabilities as .npz (probs, valid_mask, transform,
crs_wkt, class_names), which nothing but Python can open. This converts one to a
float32 GeoTIFF with one band per class, each band NAMED after its class, so the
QGIS layer panel shows `olivine`/`lcp`/... instead of `Band 1`/`Band 2`.

Invalid pixels become NaN with nodata=NaN, so QGIS renders them transparent and
the min/max stretch is computed over real data only. Writing 0.0 there instead
would both paint a hard black frame and drag every histogram toward zero.

Usage
    python scripts/probs_to_geotiff.py <probs.npz> [-o out.tif] [--split]
    python scripts/probs_to_geotiff.py <dir_of_npz> -o outdir/

    --split  also write one single-band file per class, for cases where a
             multi-band raster is awkward (e.g. per-class styling in a project
             file). The multi-band file is always written.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling

# Overviews let QGIS draw a 1636x1483x7 float raster without reading it all.
OVERVIEWS = (2, 4, 8, 16)
MIN_OVERVIEW_PX = 32   # stop halving before a level would be smaller than this


def _overview_levels(h: int, w: int) -> tuple[int, ...]:
    """Overview factors that leave a level at least MIN_OVERVIEW_PX on a side.

    GDAL raises OverviewCreationError for levels that would reduce to 1x1, so a
    fixed ladder crashes on small rasters — real for a cropped or single-ROI
    tile, not just for tests.
    """
    return tuple(f for f in OVERVIEWS if min(h, w) // f >= MIN_OVERVIEW_PX)


def load_probs(npz_path: str):
    d = np.load(npz_path, allow_pickle=True)
    probs = d['probs'].astype(np.float32)
    valid = d['valid_mask'].astype(bool)
    transform = Affine(*[float(v) for v in d['transform']])
    crs = CRS.from_wkt(str(d['crs_wkt']))
    names = [str(x) for x in d['class_names']]
    if probs.shape[2] != len(names):
        raise ValueError(f'{npz_path}: {probs.shape[2]} channels but '
                         f'{len(names)} class names {names}')
    if probs.shape[:2] != valid.shape:
        raise ValueError(f'{npz_path}: probs {probs.shape[:2]} != '
                         f'valid_mask {valid.shape}')
    return probs, valid, transform, crs, names


def write_geotiff(path, cube, transform, crs, names):
    """cube: (H, W, C) float32 with NaN outside valid. Bands named by `names`."""
    h, w, c = cube.shape
    prof = dict(driver='GTiff', height=h, width=w, count=c, dtype='float32',
                crs=crs, transform=transform, nodata=float('nan'),
                tiled=True, blockxsize=512, blockysize=512,
                compress='deflate', predictor=3, BIGTIFF='IF_SAFER')
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with rasterio.open(path, 'w', **prof) as dst:
        for i, nm in enumerate(names):
            dst.write(cube[:, :, i], i + 1)
            dst.set_band_description(i + 1, nm)
        levels = _overview_levels(h, w)
        if levels:
            dst.build_overviews(levels, Resampling.average)
    return path


def convert(npz_path: str, out_path: str, split: bool = False) -> list[str]:
    probs, valid, transform, crs, names = load_probs(npz_path)
    cube = probs.copy()
    cube[~valid] = np.nan
    written = [write_geotiff(out_path, cube, transform, crs, names)]
    if split:
        stem = os.path.splitext(out_path)[0]
        for i, nm in enumerate(names):
            p = f'{stem}_{nm}.tif'
            written.append(write_geotiff(p, cube[:, :, i:i + 1], transform, crs,
                                        [nm]))
    return written


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', help='a *_probs.npz file, or a directory of them')
    ap.add_argument('-o', '--out', default=None,
                    help='output .tif (single input) or directory (directory input)')
    ap.add_argument('--split', action='store_true',
                    help='also write one single-band .tif per class')
    args = ap.parse_args()

    if os.path.isdir(args.src):
        srcs = sorted(glob.glob(os.path.join(args.src, '*_probs.npz')))
        if not srcs:
            raise SystemExit(f'no *_probs.npz in {args.src}')
        outdir = args.out or os.path.join(args.src, 'geotiff')
        os.makedirs(outdir, exist_ok=True)
        for s in srcs:
            stem = os.path.basename(s).replace('_probs.npz', '')
            for p in convert(s, os.path.join(outdir, f'{stem}_probs.tif'),
                             args.split):
                print(f'  wrote {p}')
    else:
        out = args.out or args.src.replace('_probs.npz', '_probs.tif')
        probs, valid, _t, _c, names = load_probs(args.src)
        print(f'{os.path.basename(args.src)}: {probs.shape[0]}x{probs.shape[1]}, '
              f'{len(names)} classes {names}, {valid.sum():,} valid px')
        for p in convert(args.src, out, args.split):
            print(f'  wrote {p}')


if __name__ == '__main__':
    main()

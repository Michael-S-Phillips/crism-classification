"""Extract MTRDR plagioclase training patches → synth-style cache.

For each plagioclase polygon in data/feldsreview_categorized/, locate the paired
MTRDR mtr3 I/F cube, rasterize the polygon into MTRDR pixel coords, extract a
7x7 patch around each in-polygon pixel, linear-interp the 489 MTRDR bands to the
59 MRDR mrral wavelengths, clip to [0, 0.5], and emit:

  data/patch_cache/mtrdr_plag_patches_p7.npy   (N, 7, 7, 59) float32
  data/patch_cache/mtrdr_plag_rows.parquet     N rows, train-split only,
                                                synth_plag_rows schema

Loadable as `--synth_train_cache` / `--synth_train_parquet` to supplement the
existing plag training data without touching mrral_pixels.parquet.

Usage:
  conda run -n crism python scripts/build_mtrdr_plag_patches.py
"""
import argparse
import glob
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import spectral.io.envi as envi
from rasterio.features import rasterize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.synthetic_plag import interp_to_mrral_wavelengths

NODATA = 65535.0
CLIP_MAX = 0.5
PATCH_SIZE = 7
PAD = PATCH_SIZE // 2
N_BANDS = 59
FELDSREVIEW_ROOT = '/Volumes/Mars_GIS/CRISM/MRDR/categorized_mineral_units/FeldsReview'
TIER_WEIGHT = {'High': 1.0, 'Moderate': 0.85, 'Low': 0.7}


def parse_tier(category: str) -> str:
    s = str(category)
    for tier in ('High', 'Moderate', 'Low'):
        if f'({tier})' in s:
            return tier
    return 'Moderate'


def find_cube(obsid: str) -> str | None:
    # exact match first
    cubes = glob.glob(os.path.join(FELDSREVIEW_ROOT, '**',
                                   f'{obsid}_07_if*j_mtr3.img'), recursive=True)
    if cubes:
        return cubes[0]
    # gpkg names sometimes use the obsid without leading zeros
    # (e.g. 'frt18ff2' here <-> 'frt00018ff2' on disk).
    if obsid.lower().startswith('frt'):
        short = obsid[3:]
        cubes = glob.glob(os.path.join(FELDSREVIEW_ROOT, '**',
                                       f'frt*{short}_07_if*j_mtr3.img'), recursive=True)
        if cubes:
            return cubes[0]
    return None


def load_target_wavelengths(mrral_hdr: str) -> np.ndarray:
    img = envi.open(mrral_hdr)
    return np.asarray(img.bands.centers, dtype=np.float64)[:N_BANDS]


def load_cube_wavelengths(img_path: str) -> np.ndarray:
    hdr = img_path.replace('.img', '.hdr')
    img = envi.open(hdr)
    return np.asarray(img.bands.centers, dtype=np.float64)


def obsid_from_gpkg(gpkg_basename: str) -> str:
    """Extract the CRISM obsid from a gpkg filename.

    Handles all observed patterns:
      'new__frt000088a2.gpkg'                          -> 'frt000088a2'
      'CartPoulet2013__frt000092b4_07_if164j_..._.gpkg' -> 'frt000092b4'
      'quantin2012_scat_files__HoldenCrater_frt580c.gpkg' -> 'frt580c'
    """
    import re
    name = gpkg_basename.split('__', 1)[-1].replace('.gpkg', '').lower()
    m = re.search(r'(frt[0-9a-f]+)', name)
    return m.group(1) if m else name


def extract_patches_for_polygon(
    cube: np.ndarray, cube_wl: np.ndarray, target_wl: np.ndarray,
    geom, transform, H: int, W: int,
) -> list[tuple[np.ndarray, int, int]]:
    """Return list of (patch_7x7x59, pixel_row, pixel_col) for pixels in geom."""
    mask = rasterize([(geom, 1)], out_shape=(H, W), transform=transform,
                     fill=0, dtype='uint8').astype(bool)
    rr, cc = np.where(mask)
    out = []
    for r, c in zip(rr, cc):
        patch = np.zeros((PATCH_SIZE, PATCH_SIZE, N_BANDS), dtype=np.float32)
        r0, r1 = max(0, r - PAD), min(H, r + PAD + 1)
        c0, c1 = max(0, c - PAD), min(W, c + PAD + 1)
        dr0 = max(0, PAD - r); dc0 = max(0, PAD - c)
        chunk = cube[:, r0:r1, c0:c1].transpose(1, 2, 0)   # (h, w, 489)
        chunk = np.where(chunk == NODATA, np.nan, chunk)
        for pi in range(chunk.shape[0]):
            for pj in range(chunk.shape[1]):
                spec = chunk[pi, pj]
                valid = np.isfinite(spec)
                if valid.sum() < 5:
                    continue   # patch entry stays zero (edge / all-NODATA)
                resampled = interp_to_mrral_wavelengths(
                    cube_wl, spec, target_wl).astype(np.float32)
                patch[dr0 + pi, dc0 + pj] = resampled
        patch = np.clip(patch, 0.0, CLIP_MAX)
        out.append((patch, int(r), int(c)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--categorized_dir', default='data/feldsreview_categorized')
    ap.add_argument('--mrral_hdr', default=None,
                    help='Any mrral .hdr to read the 59 MRDR target wavelengths from.')
    ap.add_argument('--output_dir', default='data/patch_cache')
    args = ap.parse_args()

    mrral_hdr = args.mrral_hdr or sorted(glob.glob('/Volumes/Mars_GIS/CRISM/MRDR/mc*/t*_mrral_*.hdr'))[0]
    target_wl = load_target_wavelengths(mrral_hdr)
    print(f'target MRDR wavelengths: {len(target_wl)} bands, '
          f'{target_wl[0]:.1f}-{target_wl[-1]:.1f} nm (from {os.path.basename(mrral_hdr)})')

    band_cols = [f'm{i}' for i in range(N_BANDS)]
    all_patches = []; all_rows = []

    gpkgs = sorted(glob.glob(os.path.join(args.categorized_dir, '*.gpkg')))
    for gp in gpkgs:
        g = gpd.read_file(gp)
        plag = g[g['Category'].astype(str).str.lower().str.contains('plagioclase')]
        if len(plag) == 0:
            continue
        obsid = obsid_from_gpkg(os.path.basename(gp))
        cube_path = find_cube(obsid)
        if not cube_path:
            print(f'  [{obsid}] no paired MTRDR cube, SKIP')
            continue
        cube_wl = load_cube_wavelengths(cube_path)
        with rasterio.open(cube_path) as src:
            H, W = src.height, src.width
            cube_crs = src.crs
            transform = src.transform
            cube = src.read().astype(np.float32)
        if plag.crs != cube_crs:
            plag = plag.to_crs(cube_crs)
        n_polys = 0; n_pix = 0
        for _, row in plag.iterrows():
            pid = int(row['Polygon Number'])
            tier = parse_tier(row['Category'])
            patches = extract_patches_for_polygon(
                cube, cube_wl, target_wl, row.geometry, transform, H, W)
            for patch, rr, cc in patches:
                all_patches.append(patch)
                center_spec = patch[PAD, PAD]
                rec = {
                    'tile_id': f'MTRDR_{obsid}',
                    'polygon_id': pid,
                    'pixel_row': rr, 'pixel_col': cc,
                    'olivine_t1': 0.0, 'olivine_t2': 0.0,
                    'lcp': 0.0, 'hcp': 0.0, 'plagioclase': 1.0, 'other': 0.0,
                    'confidence_tier': tier,
                    'split': 'train',
                }
                rec.update({band_cols[b]: float(center_spec[b]) for b in range(N_BANDS)})
                all_rows.append(rec)
            n_polys += 1; n_pix += len(patches)
        del cube
        print(f'  [{obsid}] {n_polys} plag polys, {n_pix} pixels')

    if not all_patches:
        print('no plag patches extracted'); return
    patches_arr = np.stack(all_patches, axis=0).astype(np.float32)
    df = pd.DataFrame.from_records(all_rows)
    ordered = (['tile_id', 'polygon_id', 'pixel_row', 'pixel_col']
               + band_cols
               + ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase',
                  'other', 'confidence_tier', 'split'])
    df = df[ordered]
    os.makedirs(args.output_dir, exist_ok=True)
    npy = os.path.join(args.output_dir, 'mtrdr_plag_patches_p7.npy')
    pq = os.path.join(args.output_dir, 'mtrdr_plag_rows.parquet')
    np.save(npy, patches_arr)
    df.to_parquet(pq, index=False)
    print(f'\nwrote {npy}  shape={patches_arr.shape}')
    print(f'wrote {pq}  rows={len(df)}')
    print(f'  tier distribution: {df.confidence_tier.value_counts().to_dict()}')
    print(f'  unique MTRDR scenes: {df.tile_id.nunique()}')


if __name__ == '__main__':
    main()

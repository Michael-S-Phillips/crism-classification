"""
Vectorize per-mineral classifier probability rasters using Vectroscopy.

Reads per-class probability rasters produced by classify_tile_supervised.py --save_probs,
applies global percentile thresholds from compute_global_thresholds.py, and writes
a GeoPackage with one layer per mineral class (olivine, lcp, hcp, plagioclase).

Each polygon carries:
  confidence (int 1-3): model-driven tier (1=low/33rd pctile, 2=medium/67th, 3=high/90th)
  mineral (str): class name
  threshold (float): lower probability bound for this polygon's tier
  mean_prob, std_prob, min_prob, max_prob, median_prob: zonal statistics
  count_px (int): pixel count within polygon

Usage:
    python scripts/vectorize_tile_minerals.py \\
        --tile /mnt/mrdr/mc26/t0435_mrral_40s323_0327_4.img \\
        --probs /tmp/t0435_mrral_40s323_0327_4_probs.npz \\
        --thresholds config/vectroscopy_thresholds.json \\
        --out data/vector/t0435_mineral_map.gpkg
"""
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import rasterstats
import rasterio
import scipy.ndimage
import geopandas as gpd

# Vectroscopy: no pip install available — loaded from git clone
# A .pth file in the crism conda env makes 'import core.vectroscopy' work directly.
# The VECTROSCOPY_SRC env var allows overriding the path if needed.
_VECTROSCOPY_SRC = os.environ.get('VECTROSCOPY_SRC', '')
if _VECTROSCOPY_SRC:
    sys.path.insert(0, _VECTROSCOPY_SRC)
try:
    import core.vectroscopy as _vp_module
except ImportError as e:
    raise ImportError(
        "Cannot import Vectroscopy. "
        "Clone the repo: git clone https://github.com/Tahn04/Vectroscopy.git /opt/Vectroscopy\n"
        "Then set VECTROSCOPY_SRC=/opt/Vectroscopy/src or add its src/ dir to sys.path."
    ) from e

CLASS_NAMES = ['olivine', 'lcp', 'hcp', 'plagioclase']


# ---------------------------------------------------------------------------
# Helpers (importable for testing)
# ---------------------------------------------------------------------------

def load_thresholds_json(path: str) -> Dict[str, List[float]]:
    """Load and return the thresholds dict from a vectroscopy_thresholds.json file.

    Returns:
        dict mapping mineral name → [t1, t2, t3] float list
    """
    with open(path) as f:
        data = json.load(f)
    return data['thresholds']


def apply_median_filter(arr: np.ndarray, size: int, iterations: int) -> np.ndarray:
    """Apply scipy median filter N times on a finite float array.

    Must be called BEFORE applying NaN mask (scipy median_filter does not handle NaN).

    Args:
        arr: (H, W) float32 array with no NaN values
        size: filter kernel size (scalar, applied to both axes)
        iterations: number of times to apply the filter

    Returns:
        filtered (H, W) float32 array
    """
    result = arr.copy()
    for _ in range(iterations):
        result = scipy.ndimage.median_filter(result, size=size)
    return result


def assign_confidence_tiers(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Map Vectroscopy 'Threshold' float column to integer confidence tiers by rank.

    Vectroscopy stores the float threshold value in the 'Threshold' column.
    It also returns a Threshold=0.0 sentinel for background/catch-all polygons;
    those are filtered out before ranking.

    We rank unique non-zero threshold values ascending: lowest → tier 1, next → tier 2,
    highest → tier 3.  This avoids floating-point equality comparisons.

    Args:
        gdf: GeoDataFrame with 'Threshold' column (float values matching t1/t2/t3,
             plus possibly 0.0 background sentinel)

    Returns:
        GeoDataFrame with background rows removed and new 'confidence' column (int 1–3).
    """
    # Drop background/sentinel polygons with Threshold == 0
    result = gdf[gdf['Threshold'] > 0].copy()
    unique_t = sorted(result['Threshold'].unique())
    tier_map = {v: i + 1 for i, v in enumerate(unique_t)}
    result['confidence'] = result['Threshold'].map(tier_map)
    return result


def vectorize_mineral(
    prob_2d: np.ndarray,
    valid_mask: np.ndarray,
    thresholds: List[float],
    mineral: str,
    input_crs,
    input_transform,
    median_size: int = 3,
    median_iter: int = 1,
) -> gpd.GeoDataFrame:
    """Run full per-mineral vectorization pipeline.

    Processing order:
      1. Median filter on finite prob_2d (before NaN masking)
      2. Apply NaN to invalid pixels
      3. Vectroscopy vectorize → GeoDataFrame in geographic CRS
      4. Reproject back to tile projected CRS
      5. Assign confidence tiers
      6. Compute zonal statistics
      7. Simplify geometry (200m tolerance, after zonal stats)

    Args:
        prob_2d: (H, W) float32 probability raster, finite (no NaN) on entry
        valid_mask: (H, W) bool, True = valid pixel
        thresholds: [t1, t2, t3] float values (33rd/67th/90th percentiles)
        mineral: class name string
        input_crs: rasterio.crs.CRS of the tile
        input_transform: rasterio.transform.Affine of the tile
        median_size: median filter kernel size
        median_iter: number of median filter iterations

    Returns:
        GeoDataFrame with columns: geometry, confidence, mineral, threshold,
        mean_prob, std_prob, min_prob, max_prob, median_prob, count_px.
        Empty GeoDataFrame if no pixels exceed thresholds[0].
    """
    # Step 1: median filter on finite array
    filtered = apply_median_filter(prob_2d, size=median_size, iterations=median_iter)

    # Step 2: mask nodata pixels
    # filtered is a copy (apply_median_filter returns arr.copy()); in-place NaN is safe here
    filtered[~valid_mask] = np.nan

    # Step 3: vectorize
    gdf = _vp_module.Vectroscopy.from_array(
        array=filtered,
        thresholds=thresholds,
        crs=input_crs,
        transform=input_transform,
        name=mineral,
    ).vectorize()

    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame()

    # Step 4: reproject back to tile projected CRS
    # (Vectroscopy reprojects to geographic CRS by default)
    gdf = gdf.to_crs(input_crs)

    # Step 5: confidence tiers
    gdf = assign_confidence_tiers(gdf)
    gdf['mineral'] = mineral

    # Step 6: zonal statistics from median-filtered array (with NaN nodata)
    stats = rasterstats.zonal_stats(
        vectors=gdf.geometry,
        raster=filtered,
        affine=input_transform,
        stats=['mean', 'std', 'min', 'max', 'median', 'count'],
        nodata=np.nan,
        all_touched=False,
    )
    stats_df = pd.DataFrame(stats).rename(columns={
        'mean': 'mean_prob', 'std': 'std_prob', 'min': 'min_prob',
        'max': 'max_prob', 'median': 'median_prob', 'count': 'count_px',
    })
    gdf = pd.concat([gdf.reset_index(drop=True), stats_df], axis=1)

    # Step 7: simplify geometry AFTER zonal stats (stored geometry matches stats)
    gdf['geometry'] = gdf['geometry'].simplify(tolerance=200, preserve_topology=True)

    # Finalise column selection and rename Threshold → threshold
    keep = ['geometry', 'confidence', 'mineral', 'Threshold',
            'mean_prob', 'std_prob', 'min_prob', 'max_prob', 'median_prob', 'count_px']
    gdf = gdf[[c for c in keep if c in gdf.columns]].rename(
        columns={'Threshold': 'threshold'})

    return gdf


def load_probs_npz(path: str) -> Tuple[np.ndarray, np.ndarray, object, object]:
    """Load probs .npz; return (probs, valid_mask, crs, transform).

    Returns:
        probs: (H, W, 4) float32
        valid_mask: (H, W) bool
        crs: rasterio.crs.CRS
        transform: rasterio.transform.Affine
    """
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    data = np.load(path, allow_pickle=True)
    probs = data['probs']
    valid_mask = data['valid_mask']
    crs = CRS.from_wkt(str(data['crs_wkt']))
    a, b, c, d, e, f = data['transform']
    transform = Affine(a, b, c, d, e, f)
    return probs, valid_mask, crs, transform


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Vectorize per-mineral probability rasters using Vectroscopy.')
    parser.add_argument('--tile', required=True,
                        help='Path to mrral .img tile (always required)')
    parser.add_argument('--probs', default=None, metavar='PATH',
                        help='.npz from classify_tile_supervised --save_probs (optional)')
    parser.add_argument('--ckpt', default=None, metavar='PATH',
                        help='Classifier checkpoint (required iff --probs absent)')
    parser.add_argument('--thresholds', required=True, metavar='JSON',
                        help='Path to vectroscopy_thresholds.json')
    parser.add_argument('--out', required=True, metavar='GPKG',
                        help='Output GeoPackage path')
    parser.add_argument('--median_size', type=int, default=3)
    parser.add_argument('--median_iter', type=int, default=1)
    # NOTE: The "morphology" section in vectroscopy_thresholds.json documents the same
    # sieve_px/majority_iter defaults for reproducibility, but does NOT control runtime
    # behaviour — Stage 3 reads only from these CLI args (not from the JSON).
    parser.add_argument('--sieve_px', type=int, default=9,
                        help='Min pixels for sieve filter (accepted for CLI compat; not yet wired to Vectroscopy from_array API)')
    parser.add_argument('--majority_iter', type=int, default=3,
                        help='Majority filter iterations (accepted for CLI compat; not yet wired to Vectroscopy from_array API)')
    args = parser.parse_args()

    # Warn if non-default sieve/majority values are passed (Issue #1)
    if args.sieve_px != 9 or args.majority_iter != 3:
        import warnings
        warnings.warn(
            f"--sieve_px and --majority_iter are accepted for CLI compatibility but are not "
            f"currently wired to the Vectroscopy from_array API. "
            f"Values sieve_px={args.sieve_px}, majority_iter={args.majority_iter} will be ignored.",
            UserWarning, stacklevel=2,
        )

    # Validate probs/ckpt logic
    if args.probs is None and args.ckpt is None:
        parser.error('--ckpt is required when --probs is not supplied')

    # Load CRS and transform from tile (authoritative source)
    with rasterio.open(args.tile) as src:
        input_crs = src.crs
        input_transform = src.transform

    # Load or compute probs
    if args.probs:
        print(f'Loading probs from {args.probs}')
        probs, valid_mask, _, _ = load_probs_npz(args.probs)
    else:
        print('Running inference inline...')
        # scripts/__init__.py does not exist, so insert scripts/ dir directly
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from classify_tile_supervised import (
            load_tile, load_classifier, run_supervised
        )
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tile, valid_mask, _, _ = load_tile(args.tile)
        H, W = valid_mask.shape
        model = load_classifier(args.ckpt, device)
        probs_flat = run_supervised(tile, model, device)  # (H*W, 5)
        probs = probs_flat.reshape(H, W, 5)[:, :, :4]    # (H, W, 4) drop "other"

    H, W = valid_mask.shape
    print(f'Tile: {H}×{W}, {valid_mask.sum():,} valid pixels')

    # Load thresholds
    thresholds_cfg = load_thresholds_json(args.thresholds)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    for ci, mineral in enumerate(CLASS_NAMES):
        t1, t2, t3 = thresholds_cfg[mineral]
        print(f'Vectorizing {mineral} (thresholds: {t1:.4f}/{t2:.4f}/{t3:.4f})...')

        prob_2d = probs[:, :, ci].copy().astype(np.float32)

        gdf = vectorize_mineral(
            prob_2d=prob_2d,
            valid_mask=valid_mask,
            thresholds=[t1, t2, t3],
            mineral=mineral,
            input_crs=input_crs,
            input_transform=input_transform,
            median_size=args.median_size,
            median_iter=args.median_iter,
        )

        if gdf.empty:
            print(f'  {mineral}: no polygons detected above threshold {t1:.4f}')
            continue

        print(f'  {mineral}: {len(gdf)} polygons '
              f'(tier 1: {(gdf["confidence"]==1).sum()}, '
              f'tier 2: {(gdf["confidence"]==2).sum()}, '
              f'tier 3: {(gdf["confidence"]==3).sum()})')

        gdf.to_file(args.out, layer=mineral, driver='GPKG')

    print(f'Saved → {args.out}')


if __name__ == '__main__':
    main()

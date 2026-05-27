"""Apply the regional plagioclase scorer to tiles -> candidate plag-region GeoPackage.

For each tile: SLIC superpixels over the valid mrral cube -> region-mean 59-band
spectrum -> regional plag scorer (data/regional_plag_scorer.json) -> plag
probability per superpixel -> threshold -> vectorize flagged regions to polygons
with the mean plag-probability as confidence.

This is a REGIONAL plag-likelihood indicator (per-pixel plag is near the MRDR
detection floor; regional spectral averaging recovers it, AUC ~0.92 tile-disjoint).

Usage:
  conda run -n crism python scripts/apply_regional_plag.py \\
    --tiles /mnt/mrdr/mc13/t1249_mrral_*.img /mnt/mrdr/mc26/t0505_mrral_*.img \\
    --out data/regional_plag/regional_plag.gpkg
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from scipy import ndimage
from skimage.segmentation import slic
from shapely.geometry import shape
import geopandas as gpd

NODATA = 65535.0
CLIP_MAX = 0.5
N_BANDS = 59
# Common Mars 2000 geographic CRS — tiles are per-region equirectangular, so we
# reproject all flagged regions here before merging into one GeoPackage.
COMMON_CRS = '+proj=longlat +R=3396190 +no_defs +type=crs'


def load_cube(mrral_path):
    with rasterio.open(mrral_path) as src:
        data = src.read(list(range(1, N_BANDS + 1))).astype(np.float32)  # (59,H,W)
        transform = src.transform
        crs = src.crs
    nodata = (data == NODATA) | ~np.isfinite(data)
    data = np.clip(data, 0.0, CLIP_MAX)
    data[nodata] = 0.0
    valid = ~nodata.any(axis=0)
    return data.transpose(1, 2, 0), valid, transform, crs   # (H,W,59)


def region_mean_spectra(cube, labels, label_ids):
    """(K, 59) mean spectrum per superpixel label."""
    out = np.zeros((len(label_ids), N_BANDS), dtype=np.float32)
    for b in range(N_BANDS):
        out[:, b] = ndimage.mean(cube[..., b], labels=labels, index=label_ids)
    return out


def score(region_means, scorer):
    mean = np.asarray(scorer['scaler_mean'], dtype=np.float32)
    std = np.asarray(scorer['scaler_std'], dtype=np.float32)
    coef = np.asarray(scorer['coef'], dtype=np.float32)
    z = (region_means - mean) / std
    logit = z @ coef + scorer['intercept']
    return 1.0 / (1.0 + np.exp(-logit))


def process_tile(mrral_path, scorer, target_superpixel_px):
    tid = os.path.basename(mrral_path).split('_mrral_')[0]
    cube, valid, transform, crs = load_cube(mrral_path)
    n_valid = int(valid.sum())
    if n_valid < target_superpixel_px:
        print(f'  [{tid}] too few valid pixels ({n_valid}), skipping')
        return None
    n_seg = max(50, n_valid // target_superpixel_px)
    labels = slic(cube, n_segments=n_seg, compactness=0.1, mask=valid,
                  channel_axis=-1, start_label=1)
    ids = np.unique(labels[labels > 0])
    means = region_mean_spectra(cube, labels, ids)
    probs = score(means, scorer)
    thr = scorer['prob_threshold']
    flagged = ids[probs >= thr]
    prob_by_id = dict(zip(ids.tolist(), probs.tolist()))
    print(f'  [{tid}] {len(ids)} superpixels, {len(flagged)} flagged plag (>= {thr:.3f})')
    if len(flagged) == 0:
        return None
    flag_mask = np.isin(labels, flagged)
    id_raster = np.where(flag_mask, labels, 0).astype(np.int32)
    recs = []
    for geom, val in features.shapes(id_raster, mask=flag_mask, transform=transform):
        recs.append({'geometry': shape(geom), 'tile_id': tid,
                     'plag_prob': float(prob_by_id[int(val)])})
    if not recs:
        return None
    gdf = gpd.GeoDataFrame(recs, crs=crs)
    return gdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tiles', nargs='+', required=True,
                    help='mrral .img paths or globs')
    ap.add_argument('--scorer', default='data/regional_plag_scorer.json')
    ap.add_argument('--out', default='data/regional_plag/regional_plag.gpkg')
    ap.add_argument('--target_superpixel_px', type=int, default=600,
                    help='approx pixels per superpixel (~region scale)')
    args = ap.parse_args()

    with open(args.scorer) as f:
        scorer = json.load(f)
    print(f'scorer: AUC={scorer.get("val_auc"):.3f} AP={scorer.get("val_ap"):.3f} '
          f'thr={scorer["prob_threshold"]:.3f} (P={scorer.get("threshold_precision",0):.0%})')

    paths = []
    for t in args.tiles:
        paths.extend(sorted(glob.glob(t)) if any(c in t for c in '*?[') else [t])
    print(f'tiles: {len(paths)}')

    gdfs = []
    for p in paths:
        g = process_tile(p, scorer, args.target_superpixel_px)
        if g is not None and len(g):
            gdfs.append(g)

    if not gdfs:
        print('No plag regions flagged on any tile.'); return
    # Reproject each tile's regions to the common Mars geographic CRS before merge.
    gdfs = [g.to_crs(COMMON_CRS) for g in gdfs]
    out_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=COMMON_CRS)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_gdf.to_file(args.out, driver='GPKG')
    print(f'wrote {args.out}: {len(out_gdf)} plag regions across {len(gdfs)} tiles '
          f'(mean conf {out_gdf.plag_prob.mean():.2f})')


if __name__ == '__main__':
    main()

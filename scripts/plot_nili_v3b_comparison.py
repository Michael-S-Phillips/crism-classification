"""
3-panel comparison on the Nili Fossae 4-tile area (t1249, t1250, t1321, t1322):
  Panel 1 — MAF band-parameter composite (R=OLINDEX3, G=LCPINDEX2, B=HCPINDEX2)
  Panel 2 — ft_6cls_v3b_lrscale0001 (v3 denoising enc, 6-class)
  Panel 3 — ft_6cls_mc11val_denoise (current champion, 6-class)

Probs sources:
  /tmp/v3b_nili_lrscale0001/t*_probs.npz
  /tmp/6cls_mc13/t*_probs.npz  (already exists from MC13 run)

Usage:
    conda run -n crism python scripts/plot_nili_v3b_comparison.py
"""
from __future__ import annotations

import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.warp
from affine import Affine
from pyproj import CRS
from rasterio.enums import Resampling

PROJECT_ROOT = '/Volumes/Mars_GIS/CRISM/MRDR/crism_classification'
MC_DIR       = '/Volumes/Mars_GIS/CRISM/MRDR/mc13'
NILI_TILES   = ['t1249', 't1250', 't1321', 't1322']

V3B_PROBS_DIR      = '/tmp/v3b_nili_lrscale0001'
CHAMPION_PROBS_DIR = '/tmp/6cls_mc13'

OUT_PATH = os.path.join(PROJECT_ROOT, 'reports', 'v5',
                        'fig_nili_v3b_vs_champion.png')

MARS_GEO_WKT = (
    'GEOGCS["GCS_Mars_2000",'
    'DATUM["D_Mars_2000",SPHEROID["Mars_2000_IAU_IAG",3396190,169.8944472]],'
    'PRIMEM["Reference_Meridian",0],'
    'UNIT["Degree",0.0174532925199433]]'
)
COMMON_CRS = CRS.from_wkt(MARS_GEO_WKT)

BAND_OLINDEX3  = 16
BAND_LCPINDEX2 = 19
BAND_HCPINDEX2 = 20

CLASS_NAMES = ['olivine', 'lcp', 'hcp', 'plagioclase', 'alteration']
CLASS_COLORS_01 = {
    'olivine':     np.array([ 0.85,  0.00,  0.00], dtype=np.float32),
    'lcp':         np.array([ 0.00,  0.00,  0.85], dtype=np.float32),
    'hcp':         np.array([ 0.00,  0.70,  0.00], dtype=np.float32),
    'plagioclase': np.array([-0.50, -0.50, -0.50], dtype=np.float32),
    'alteration':  np.array([ 0.70,  0.70,  0.00], dtype=np.float32),
}

# Same thresholds for both models (fair visual comparison)
VAL_AP = {
    'olivine': 0.996, 'lcp': 0.895, 'hcp': 0.643,
    'plagioclase': 0.325, 'other': 1.000, 'alteration': 0.978,
}
GLOBAL_MIN_THRESH = 0.8
THRESHOLDS = {c: max(0.99 - 0.49 * v, GLOBAL_MIN_THRESH) for c, v in VAL_AP.items()}
PROB_CHANNELS = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other', 'alteration']

DOWNSAMPLE = 1  # native resolution — only 4 tiles, small area


def discover_tiles(probs_dir: str) -> list[dict]:
    tiles = []
    for tid in NILI_TILES:
        npz = os.path.join(probs_dir, f'{tid}_probs.npz')
        if not os.path.exists(npz):
            print(f'  WARNING: missing {npz}')
            continue
        mrral = next(
            (p for p in [
                os.path.join(MC_DIR, f'{tid}_mrral_20n073_0327_4.img'),
                os.path.join(MC_DIR, f'{tid}_mrral_20n078_0327_4.img'),
                os.path.join(MC_DIR, f'{tid}_mrral_25n073_0327_4.img'),
                os.path.join(MC_DIR, f'{tid}_mrral_25n078_0327_4.img'),
            ] if os.path.exists(p)),
            None,
        )
        if mrral is None:
            import glob as _glob
            hits = _glob.glob(os.path.join(MC_DIR, f'{tid}_mrral_*_0327_4.img'))
            mrral = hits[0] if hits else None
        mrrsu = mrral.replace('_mrral_', '_mrrsu_') if mrral else None
        if mrral and mrrsu and os.path.exists(mrrsu):
            tiles.append({'tid': tid, 'mrral': mrral, 'mrrsu': mrrsu})
    print(f'  {len(tiles)} tiles found')
    return tiles


def compute_target_grid(tiles):
    bounds_list, res_list = [], []
    for t in tiles:
        with rasterio.open(t['mrral']) as src:
            b = rasterio.warp.transform_bounds(src.crs, COMMON_CRS, *src.bounds)
            bounds_list.append(b)
            res_list.append(abs(src.transform.a))
    xmin = min(b[0] for b in bounds_list)
    ymin = min(b[1] for b in bounds_list)
    xmax = max(b[2] for b in bounds_list)
    ymax = max(b[3] for b in bounds_list)
    native = max(res_list)
    target = native * DOWNSAMPLE
    deg = 1.0 / (3396190 * math.pi / 180)
    res_deg = target * deg
    n_x = int(math.ceil((xmax - xmin) / res_deg))
    n_y = int(math.ceil((ymax - ymin) / res_deg))
    transform = rasterio.transform.from_origin(xmin, ymax, res_deg, res_deg)
    return transform, (n_y, n_x), (xmin, ymin, xmax, ymax)


def reproject_band(path, band_idx, tgt_transform, tgt_shape):
    dst = np.full(tgt_shape, np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        arr = src.read(band_idx).astype(np.float32)
        arr[(arr == 65535) | ~np.isfinite(arr)] = np.nan
        rasterio.warp.reproject(
            source=arr, destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=tgt_transform, dst_crs=COMMON_CRS,
            resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan,
        )
    return dst


def reproject_rgb(rgb_hwc, src_t, src_crs, tgt_transform, tgt_shape):
    dst = np.full((tgt_shape[0], tgt_shape[1], 3), np.nan, dtype=np.float32)
    for c in range(3):
        rasterio.warp.reproject(
            source=rgb_hwc[:, :, c], destination=dst[:, :, c],
            src_transform=src_t, src_crs=src_crs,
            dst_transform=tgt_transform, dst_crs=COMMON_CRS,
            resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan,
        )
    return dst


def build_band_param_mosaic(tiles, tgt_transform, tgt_shape):
    out = np.full((tgt_shape[0], tgt_shape[1], 3), np.nan, dtype=np.float32)
    for t in tiles:
        for ci, band in enumerate([BAND_OLINDEX3, BAND_LCPINDEX2, BAND_HCPINDEX2]):
            arr = reproject_band(t['mrrsu'], band, tgt_transform, tgt_shape)
            mask = np.isnan(out[:, :, ci]) & np.isfinite(arr)
            out[mask, ci] = arr[mask]
    rgb = np.zeros_like(out)
    for c in range(3):
        ch = out[:, :, c]
        v = np.isfinite(ch)
        if v.any():
            p2, p98 = np.percentile(ch[v], [2, 98])
            rgb[:, :, c] = np.clip((ch - p2) / max(1e-6, p98 - p2), 0, 1)
    invalid = np.isnan(out).any(axis=-1)
    rgb[invalid] = 0.85
    return rgb, invalid


def build_classifier_mosaic(tiles, probs_dir, tgt_transform, tgt_shape):
    out = np.full((tgt_shape[0], tgt_shape[1], 3), np.nan, dtype=np.float32)
    stats = []
    for t in tiles:
        npz_path = os.path.join(probs_dir, f"{t['tid']}_probs.npz")
        npz = np.load(npz_path)
        probs = npz['probs'].astype(np.float32)
        valid = npz['valid_mask'].astype(bool)
        src_t = Affine(*[float(v) for v in npz['transform']])
        src_crs = CRS.from_wkt(str(npz['crs_wkt']))

        H, W, _ = probs.shape
        tile_rgb = np.zeros((H, W, 3), dtype=np.float32)
        any_passed = np.zeros((H, W), dtype=bool)
        s = {'tid': t['tid'], 'valid': int(valid.sum())}
        for cname in CLASS_NAMES:
            ci = PROB_CHANNELS.index(cname)
            passing = (probs[:, :, ci] >= THRESHOLDS[cname]) & valid
            tile_rgb[passing] += CLASS_COLORS_01[cname]
            any_passed |= passing
            s[cname] = int(passing.sum())
        np.clip(tile_rgb, 0.0, 1.0, out=tile_rgb)
        tile_rgb[valid & ~any_passed] = 1.0
        tile_rgb[~valid] = np.nan
        s['unclassified'] = int((valid & ~any_passed).sum())
        stats.append(s)

        reproj = reproject_rgb(tile_rgb, src_t, src_crs, tgt_transform, tgt_shape)
        mask = np.isfinite(reproj).all(axis=-1) & np.isnan(out[:, :, 0])
        out[mask] = reproj[mask]
    out[np.isnan(out).any(axis=-1)] = 0.85
    return out, stats


def print_stats(label, stats):
    print(f'\n{label}:')
    total_v = sum(s['valid'] for s in stats)
    total_u = sum(s['unclassified'] for s in stats)
    for s in stats:
        line = f"  {s['tid']}  valid={s['valid']:,}"
        for c in CLASS_NAMES:
            line += f"  {c[:3]}={s[c]:,}"
        line += f"  unclass={s['unclassified']:,} ({100*s['unclassified']/max(s['valid'],1):.0f}%)"
        print(line)
    print(f'  TOTAL  valid={total_v:,}  unclass={total_u:,} ({100*total_u/max(total_v,1):.0f}%)')


def main():
    print('Discovering tiles (v3b) …')
    tiles = discover_tiles(V3B_PROBS_DIR)
    if not tiles:
        print('No v3b probs found — run scripts/run_nili_inference_v3b.sh first.')
        sys.exit(1)

    print('Discovering tiles (champion, reusing /tmp/6cls_mc13) …')
    tiles_champ = discover_tiles(CHAMPION_PROBS_DIR)

    print(f'Computing target grid (downsample ×{DOWNSAMPLE}) …')
    tgt_t, tgt_shape, (xmin, ymin, xmax, ymax) = compute_target_grid(tiles)
    print(f'  grid: {tgt_shape[1]} × {tgt_shape[0]} px')

    print('Building band-parameter mosaic …')
    bparam, _ = build_band_param_mosaic(tiles, tgt_t, tgt_shape)

    print('Building v3b classifier mosaic …')
    v3b_rgb, v3b_stats = build_classifier_mosaic(tiles, V3B_PROBS_DIR, tgt_t, tgt_shape)

    print('Building champion classifier mosaic …')
    champ_rgb, champ_stats = build_classifier_mosaic(tiles_champ, CHAMPION_PROBS_DIR, tgt_t, tgt_shape)

    print_stats('v3b lrscale0001', v3b_stats)
    print_stats('champion (mc11val_denoise)', champ_stats)

    # ── figure: 1 row × 3 columns ─────────────────────────────────────────────
    lat_mid = (ymin + ymax) / 2
    cos_mid = math.cos(math.radians(lat_mid))
    panel_aspect = (ymax - ymin) / ((xmax - xmin) * cos_mid)
    panel_w = 6.0
    panel_h = panel_w * panel_aspect
    fig, axes = plt.subplots(1, 3, figsize=(panel_w * 3 + 0.5, panel_h + 1.5))

    extent = (xmin, xmax, ymin, ymax)
    for ax in axes:
        ax.set_xlabel('Longitude (°E)', fontsize=8)
        ax.tick_params(labelsize=7)

    axes[0].imshow(bparam, extent=extent, origin='upper', aspect='auto')
    axes[0].set_title('Band params\nR=OLINDEX3  G=LCPINDEX2  B=HCPINDEX2', fontsize=9)
    axes[0].set_ylabel('Latitude (°N)', fontsize=8)

    axes[1].imshow(v3b_rgb, extent=extent, origin='upper', aspect='auto')
    axes[1].set_title(
        'ft_6cls_v3b_lrscale0001\n(v3 denoising enc → 6-class)', fontsize=9)

    axes[2].imshow(champ_rgb, extent=extent, origin='upper', aspect='auto')
    axes[2].set_title(
        'ft_6cls_mc11val_denoise\n(plag-aware enc → 6-class, champion)', fontsize=9)

    # Legend on last panel
    def _mix(*names):
        c = np.zeros(3, dtype=np.float32)
        for n in names:
            c = c + CLASS_COLORS_01[n]
        return np.clip(c, 0, 1)

    handles = []
    for cname in ['olivine', 'lcp', 'hcp', 'alteration']:
        handles.append(mpatches.Patch(
            facecolor=CLASS_COLORS_01[cname],
            label=f'{cname} (τ={THRESHOLDS[cname]:.2f})'))
    handles.append(mpatches.Patch(
        facecolor=_mix('plagioclase'),
        label=f'plagioclase (τ={THRESHOLDS["plagioclase"]:.2f}; darkens)'))
    handles.append(mpatches.Patch(
        facecolor=_mix('olivine', 'hcp'), label='olivine + hcp'))
    handles.append(mpatches.Patch(
        facecolor=_mix('olivine', 'lcp'), label='olivine + lcp'))
    handles.append(mpatches.Patch(facecolor='white', edgecolor='black', label='unclassified'))
    axes[2].legend(handles=handles, loc='lower right', fontsize=6, ncol=1,
                   framealpha=0.9)

    fig.suptitle(
        'Nili Fossae (t1249, t1250, t1321, t1322) — v3b vs champion',
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\nSaved → {OUT_PATH}')


if __name__ == '__main__':
    main()

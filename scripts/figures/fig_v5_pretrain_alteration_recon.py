"""
MAE reconstruction quality comparison on alteration-mineral pixels.

Alteration polygons are sourced from /mnt/mrdr/categorized_mineral_units/T*.gpkg.
For three alteration pixels from DIFFERENT tiles, shows:
  - Col 1: clean center-pixel spectrum (reference, orange color)
  - Col 2: v3 (denoising MAE) recon overlay
  - Col 3: v4 (SPEND MAE) recon overlay
  - Col 4: residual comparison (v3 dashed vs v4 solid)

Title: "MAE reconstruction quality — alteration pixels"
Subtitle: "v3 denoising vs v4 SPEND on hydrated mineral polygons"

Usage (no args needed):
    conda run -n crism python scripts/figures/fig_v5_pretrain_alteration_recon.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── project root on sys.path ────────────────────────────────────────────────
PROJECT_ROOT = '/mnt/mrdr/crism_classification'
sys.path.insert(0, PROJECT_ROOT)

from models.denoising_spatial_mae import DenoisingSpatialSpectralMAE
from models.spend_spatial_mae import SpendSpatialSpectralMAE

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts', 'figures'))
from _utils import (
    get_wavelengths_59, read_patch_from_tile,
)

# ── paths ────────────────────────────────────────────────────────────────────
CKPT_V3 = os.path.join(PROJECT_ROOT, 'checkpoints', 'spatial_mae_denoising_128d_6l_best.pt')
CKPT_V4 = os.path.join(PROJECT_ROOT, 'checkpoints', 'spatial_mae_spend_128d_6l_best.pt')
OUT_PATH = os.path.join(PROJECT_ROOT, 'reports', 'v5', 'fig_v5_pretrain_alteration_recon.png')
GPKG_DIR = '/mnt/mrdr/categorized_mineral_units'

# Center pixel flat index for a 7×7 patch (row 3, col 3 → 3*7+3 = 24)
CENTER_IDX = 24

# Color for alteration pixels (warm amber)
ALTERATION_COLOR = '#9467bd'  # purple — distinguishable from mineral class palette


# ── model loaders (mirrored from fig_v5_pretrain_reconstructions.py) ─────────

def load_v3(path: str) -> DenoisingSpatialSpectralMAE:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    m = DenoisingSpatialSpectralMAE(
        n_bands=59, patch_size=7,
        embed_dim=cfg.get('embed_dim', 128),
        n_heads=cfg.get('n_heads', 4),
        n_layers=cfg.get('n_layers', 6),
        decoder_dim=cfg.get('decoder_dim', 64),
        decoder_layers=cfg.get('decoder_layers', 2),
        mask_ratio=cfg.get('mask_ratio', 0.75),
    )
    m.load_state_dict(ckpt['mae_state'])
    m.eval()
    return m


def load_v4(path: str) -> SpendSpatialSpectralMAE:
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    cfg = ckpt.get('config', {})
    m = SpendSpatialSpectralMAE(
        n_bands=59, patch_size=7,
        embed_dim=cfg.get('embed_dim', 128),
        n_heads=cfg.get('n_heads', 4),
        n_layers=cfg.get('n_layers', 6),
        decoder_dim=cfg.get('decoder_dim', 64),
        decoder_layers=cfg.get('decoder_layers', 2),
        mask_ratio=cfg.get('mask_ratio', 0.75),
        spectral_mask_ratio=0.0,  # eval mode: no band masking
    )
    m.load_state_dict(ckpt['mae_state'])
    m.eval()
    return m


# ── inference helper ─────────────────────────────────────────────────────────

def run_model(model, patch_raw: np.ndarray, seed: int = 42):
    """Normalize patch, run model, unnormalize reconstruction to I/F.

    Returns:
        recon_if   : (49, 59) float32 — all spatial positions, I/F units
        mask_bool  : (49,)  bool      — True = spatially masked at encoder
        mean, std  : scalars
    """
    mean = patch_raw.mean()
    std = float(patch_raw.std()) + 1e-8
    patch_norm = (patch_raw - mean) / std

    x = torch.from_numpy(patch_norm).unsqueeze(0).float()  # (1, 7, 7, 59)

    torch.manual_seed(seed)  # fix spatial mask for reproducibility
    with torch.no_grad():
        _loss, recon_norm, mask = model(x)

    recon_if = recon_norm[0].numpy() * std + mean  # (49, 59)
    mask_bool = mask[0].numpy().astype(bool)       # (49,)
    return recon_if, mask_bool, mean, std


# ── alteration pixel finder ───────────────────────────────────────────────────

def find_alteration_pixels(n: int = 3):
    """Return list of dicts with tile, pixel info for n alteration polygons.

    Walks T*.gpkg files sorted by name, filters for 'alteration' category,
    reprojects to tile CRS, rasterizes, picks valid centroid pixel.
    Takes one polygon per tile to ensure diversity.
    """
    import glob
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from shapely.geometry import mapping

    # Build mrral tile map
    hdrs = sorted(glob.glob('/mnt/mrdr/mc*/t*mrral*.hdr'))
    tile_id_map = {}
    for h in hdrs:
        tid = os.path.basename(h).split('_mrral_')[0]
        tile_id_map[tid] = h.replace('.hdr', '.img')

    found = []
    seen_tiles = set()

    for fname in sorted(os.listdir(GPKG_DIR)):
        if not fname.endswith('.gpkg') or not fname.startswith('T'):
            continue
        tile_num = fname[1:5]
        tile_id = f't{tile_num}'
        if tile_id in seen_tiles:
            continue
        mrral_path = tile_id_map.get(tile_id)
        if not mrral_path or not os.path.exists(mrral_path):
            continue

        try:
            gdf = gpd.read_file(os.path.join(GPKG_DIR, fname))
            gdf_alt = gdf[gdf['Category'].str.contains('alteration', case=False, na=False)].copy()
            if len(gdf_alt) == 0:
                continue

            gdf_alt['area'] = gdf_alt.geometry.area
            gdf_alt = gdf_alt[gdf_alt['area'] > 10000].sort_values('area', ascending=False)
            if len(gdf_alt) == 0:
                continue

            with rasterio.open(mrral_path) as src:
                raster_crs = src.crs
                transform = src.transform
                height, width = src.height, src.width

            for _, row in gdf_alt.iterrows():
                poly_gdf = gpd.GeoDataFrame(geometry=[row.geometry], crs=gdf.crs)
                if gdf.crs != raster_crs:
                    poly_gdf = poly_gdf.to_crs(raster_crs)
                poly_reprojected = poly_gdf.geometry.iloc[0]

                out_mask = np.zeros((height, width), dtype=np.uint8)
                try:
                    rasterize(
                        [(mapping(poly_reprojected), 1)],
                        out=out_mask,
                        transform=transform,
                        fill=0,
                        all_touched=False,
                    )
                except Exception:
                    continue

                valid_pixels = np.argwhere(out_mask == 1)
                if len(valid_pixels) == 0:
                    continue

                # Try centroid, then fallback to first valid pixels
                cent_row = int(valid_pixels[:, 0].mean())
                cent_col = int(valid_pixels[:, 1].mean())

                def check_pixel(pr, pc):
                    with rasterio.open(mrral_path) as src:
                        window = rasterio.windows.Window(int(pc), int(pr), 1, 1)
                        band1 = src.read(1, window=window)[0, 0]
                    return 0 < band1 < 65535

                pixel_ok = check_pixel(cent_row, cent_col)
                if not pixel_ok:
                    for pr2, pc2 in valid_pixels[:30]:
                        if check_pixel(int(pr2), int(pc2)):
                            cent_row, cent_col = int(pr2), int(pc2)
                            pixel_ok = True
                            break

                if not pixel_ok:
                    continue

                found.append({
                    'fname': fname,
                    'tile_id': tile_id,
                    'mrral_path': mrral_path,
                    'poly_num': row['Polygon Number'],
                    'category': row['Category'],
                    'area': float(row['area']),
                    'pixel_row': cent_row,
                    'pixel_col': cent_col,
                })
                seen_tiles.add(tile_id)
                break  # one per tile

        except Exception as e:
            print(f'  WARNING: {fname} — {e}')

        if len(found) >= n:
            break

    if len(found) < n:
        raise RuntimeError(
            f'Only found {len(found)} valid alteration pixels (need {n}). '
            'Check that mrral tiles and gpkg files are accessible.'
        )

    return found[:n]


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print('Loading models ...')
    v3 = load_v3(CKPT_V3)
    v4 = load_v4(CKPT_V4)
    ckpt_v3 = torch.load(CKPT_V3, map_location='cpu', weights_only=False)
    ckpt_v4 = torch.load(CKPT_V4, map_location='cpu', weights_only=False)
    print(f'  v3 (denoising) — epoch {ckpt_v3["epoch"]}')
    print(f'  v4 (SPEND)     — epoch {ckpt_v4["epoch"]}')

    print('Finding alteration pixels ...')
    alteration_pixels = find_alteration_pixels(n=3)
    wls = get_wavelengths_59()

    print('\nChosen alteration pixels (for reproducibility):')
    for p in alteration_pixels:
        print(f"  tile={p['tile_id']}, gpkg={p['fname']}, "
              f"polygon_num={p['poly_num']}, "
              f"row={p['pixel_row']}, col={p['pixel_col']}, "
              f"area={p['area']:.0f} m², category='{p['category']}'")

    n_rows = len(alteration_pixels)
    fig, axes = plt.subplots(
        n_rows, 4,
        figsize=(16.0, 3.4 * n_rows),
    )
    if n_rows == 1:
        axes = axes[None, :]

    for row_i, pinfo in enumerate(alteration_pixels):
        tid = pinfo['tile_id']
        pr = pinfo['pixel_row']
        pc = pinfo['pixel_col']
        mrral_path = pinfo['mrral_path']
        category = pinfo['category']

        print(f'\n  [{tid}] poly {pinfo["poly_num"]} pixel=({pr},{pc})')

        patch_raw = read_patch_from_tile(mrral_path, pr, pc, patch_size=7, n_bands=59)
        center_clean = patch_raw[3, 3, :]  # (59,) I/F

        # Run both models with the SAME spatial mask seed
        recon_v3, mask_v3, _, _ = run_model(v3, patch_raw, seed=42)
        recon_v4, mask_v4, _, _ = run_model(v4, patch_raw, seed=42)

        center_recon_v3 = recon_v3[CENTER_IDX]  # (59,)
        center_recon_v4 = recon_v4[CENTER_IDX]

        color = ALTERATION_COLOR
        valid = (center_clean > 0.001) & (center_clean < 0.499)

        # ── Col 1: clean center-pixel spectrum ───────────────────────────────
        ax = axes[row_i, 0]
        ax.plot(wls[valid], center_clean[valid], color=color, linewidth=1.8, label='clean')
        ax.set_xlabel('Wavelength (nm)', fontsize=8)
        ax.set_ylabel('I/F', fontsize=8)
        short_cat = category[:35] + '…' if len(category) > 35 else category
        ax.set_title(f'tile {tid}\n{short_cat}', color=color, fontsize=9)
        ax.grid(alpha=0.3)

        # ── Col 2: v3 denoising reconstruction ──────────────────────────────
        ax = axes[row_i, 1]
        ax.plot(wls[valid], center_clean[valid],
                color='#888', linewidth=1.2, linestyle='-', alpha=0.7, label='clean')
        ax.plot(wls[valid], center_recon_v3[valid],
                color=color, linewidth=1.4, linestyle='--', label='v3 recon')
        center_masked_v3 = bool(mask_v3[CENTER_IDX])
        marker_style = 'x' if center_masked_v3 else 'o'
        marker_label = 'masked' if center_masked_v3 else 'visible'
        ax.scatter(wls[valid][::6], center_recon_v3[valid][::6],
                   color=color, s=20, marker=marker_style,
                   label=marker_label, zorder=4, alpha=0.8)
        ax.set_xlabel('Wavelength (nm)', fontsize=8)
        ax.set_ylabel('I/F', fontsize=8)
        ax.set_title(f'v3 denoising recon\ncenter = {"masked" if center_masked_v3 else "visible"}',
                     fontsize=9)
        ax.legend(fontsize=7, loc='upper left', framealpha=0.85)
        ax.grid(alpha=0.3)

        # ── Col 3: v4 SPEND reconstruction ──────────────────────────────────
        ax = axes[row_i, 2]
        ax.plot(wls[valid], center_clean[valid],
                color='#888', linewidth=1.2, linestyle='-', alpha=0.7, label='clean')
        ax.plot(wls[valid], center_recon_v4[valid],
                color=color, linewidth=1.4, linestyle='--', label='v4 recon')
        center_masked_v4 = bool(mask_v4[CENTER_IDX])
        marker_style_v4 = 'x' if center_masked_v4 else 'o'
        marker_label_v4 = 'masked' if center_masked_v4 else 'visible'
        ax.scatter(wls[valid][::6], center_recon_v4[valid][::6],
                   color=color, s=20, marker=marker_style_v4,
                   label=marker_label_v4, zorder=4, alpha=0.8)
        ax.set_xlabel('Wavelength (nm)', fontsize=8)
        ax.set_ylabel('I/F', fontsize=8)
        ax.set_title(f'v4 SPEND recon\ncenter = {"masked" if center_masked_v4 else "visible"}',
                     fontsize=9)
        ax.legend(fontsize=7, loc='upper left', framealpha=0.85)
        ax.grid(alpha=0.3)

        # ── Col 4: residuals ─────────────────────────────────────────────────
        ax = axes[row_i, 3]
        resid_v3 = center_recon_v3 - center_clean
        resid_v4 = center_recon_v4 - center_clean
        mae_v3 = float(np.abs(resid_v3[valid]).mean())
        mae_v4 = float(np.abs(resid_v4[valid]).mean())
        ax.plot(wls[valid], resid_v3[valid],
                color='#d62728', linewidth=1.3, linestyle='--',
                label=f'v3  MAE={mae_v3:.4f}')
        ax.plot(wls[valid], resid_v4[valid],
                color='#1f77b4', linewidth=1.3, linestyle='-',
                label=f'v4  MAE={mae_v4:.4f}')
        ax.axhline(0, color='black', linewidth=0.6, linestyle=':')
        ax.set_xlabel('Wavelength (nm)', fontsize=8)
        ax.set_ylabel('recon − clean (I/F)', fontsize=8)
        ax.set_title('residuals\ndashed=v3  solid=v4', fontsize=9)
        ax.legend(fontsize=7, loc='upper left', framealpha=0.85)
        ax.grid(alpha=0.3)

        print(f'    residual MAE: v3={mae_v3:.5f}  v4={mae_v4:.5f}  '
              f'center masked: v3={center_masked_v3}  v4={center_masked_v4}')

    fig.suptitle(
        'MAE reconstruction quality — alteration pixels\n'
        'v3 denoising vs v4 SPEND on hydrated mineral polygons',
        fontsize=11,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\nWrote {OUT_PATH}')
    sz = os.path.getsize(OUT_PATH)
    print(f'File size: {sz/1024:.1f} KB')


if __name__ == '__main__':
    main()

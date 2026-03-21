"""
Spectral diagnostic plots per mineral class and confidence tier.

For each of the 5 mineral classes, shows:
  Left panel:  raw mean ± std reflectance spectra for tier 1 / 2 / 3 pixels
  Right panel: ratio spectra = mean(class tier) / mean(all other classified pixels)

Pixels are sampled from the vectorized mineral maps (GeoPackage) by burning polygon
footprints to a raster mask, then reading spectra from the mrral cube.  Spectra are
pooled from both calibration tiles (T0435 and T0434).

Output: reports/fig_mineral_spectra.png

Usage:
    conda run -n crism python scripts/plot_mineral_spectra.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.features
import geopandas as gpd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fig_style import DPI, MINERAL_COLORS

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TILES = [
    {
        'img': '/mnt/mrdr/mc26/t0435_mrral_40s323_0327_4.img',
        'gpkg': os.path.join(PROJ, 'data/vector/t0435_mrral_40s323_0327_4_mineral_map.gpkg'),
    },
    {
        'img': '/mnt/mrdr/mc26/t0434_mrral_40s318_0327_4.img',
        'gpkg': os.path.join(PROJ, 'data/vector/t0434_mrral_40s318_0327_4_mineral_map.gpkg'),
    },
]

CLASS_NAMES = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']
TIERS = [1, 2, 3]
TIER_COLORS = ['#43a047', '#fb8c00', '#e53935']   # tier 1 / 2 / 3
TIER_LABELS = ['Tier 1', 'Tier 2', 'Tier 3']
CRISM_NODATA = 65535.0
MAX_PIXELS_PER_CLASS_TIER = 10_000   # cap to keep memory bounded


def read_tile_cube(img_path):
    """Read full reflectance cube and wavelengths.

    Returns:
        cube: (H, W, B) float32 — NaN for nodata pixels
        wavelengths: (B,) float, nm
        valid_mask: (H, W) bool
        src_profile: rasterio profile dict (height, width, transform, crs)
    """
    with rasterio.open(img_path) as src:
        wavelengths = np.array([float(d) for d in src.descriptions])
        data = src.read().astype(np.float32)   # (B, H, W)
        profile = {
            'height': src.height,
            'width': src.width,
            'transform': src.transform,
            'crs': src.crs,
        }
        nodata = src.nodata

    if nodata is not None:
        data[data == nodata] = np.nan
    data[data == CRISM_NODATA] = np.nan

    cube = np.moveaxis(data, 0, -1)           # (H, W, B)
    valid_mask = np.isfinite(cube).all(axis=-1)
    return cube, wavelengths, valid_mask, profile


def rasterize_tier_mask(gpkg_path, mineral, tier, height, width, transform, crs):
    """Burn polygons for a given mineral + tier from a GeoPackage to a bool raster."""
    try:
        gdf = gpd.read_file(gpkg_path, layer=mineral)
    except Exception:
        return np.zeros((height, width), dtype=bool)

    gdf_tier = gdf[gdf['confidence'] == tier]
    if gdf_tier.empty:
        return np.zeros((height, width), dtype=bool)

    if gdf_tier.crs != crs:
        gdf_tier = gdf_tier.to_crs(crs)

    shapes = [(geom, 1) for geom in gdf_tier.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        return np.zeros((height, width), dtype=bool)

    burned = rasterio.features.rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    return burned.astype(bool)


def collect_spectra(tiles_info, rng=None):
    """Collect pixel spectra per mineral per tier across all tiles.

    Returns:
        spectra: dict[mineral][tier] = (N, B) float32 array
        wavelengths: (B,) float array
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Accumulators
    acc = {m: {t: [] for t in TIERS} for m in CLASS_NAMES}
    wavelengths = None

    for tile in tiles_info:
        img_name = os.path.basename(tile['img'])
        print(f'  Loading {img_name} ...')
        cube, wavs, valid_mask, prof = read_tile_cube(tile['img'])
        if wavelengths is None:
            wavelengths = wavs

        for mineral in CLASS_NAMES:
            for tier in TIERS:
                mask = rasterize_tier_mask(
                    tile['gpkg'], mineral, tier,
                    prof['height'], prof['width'],
                    prof['transform'], prof['crs'],
                )
                mask = mask & valid_mask
                pixels = cube[mask]   # (N, B)
                n = pixels.shape[0]
                if n == 0:
                    continue
                # Subsample to keep memory bounded
                if n > MAX_PIXELS_PER_CLASS_TIER:
                    idx = rng.choice(n, MAX_PIXELS_PER_CLASS_TIER, replace=False)
                    pixels = pixels[idx]
                acc[mineral][tier].append(pixels)

    # Concatenate across tiles
    spectra = {}
    for mineral in CLASS_NAMES:
        spectra[mineral] = {}
        for tier in TIERS:
            arrays = acc[mineral][tier]
            if arrays:
                combined = np.concatenate(arrays, axis=0)
                n = combined.shape[0]
                if n > MAX_PIXELS_PER_CLASS_TIER:
                    idx = rng.choice(n, MAX_PIXELS_PER_CLASS_TIER, replace=False)
                    combined = combined[idx]
                spectra[mineral][tier] = combined
            else:
                spectra[mineral][tier] = np.zeros((0, len(wavelengths)), dtype=np.float32)

    return spectra, wavelengths


def compute_other_mean(spectra, focal_mineral):
    """Mean spectrum of all classified pixels NOT belonging to focal_mineral."""
    arrays = []
    for other in CLASS_NAMES:
        if other == focal_mineral:
            continue
        for tier in TIERS:
            arr = spectra[other][tier]
            if arr.shape[0] > 0:
                arrays.append(arr)
    if not arrays:
        return None
    return np.nanmean(np.concatenate(arrays, axis=0), axis=0)


def main():
    print('Collecting pixel spectra ...')
    spectra, wav = collect_spectra(TILES)

    # Restrict to VNIR–SWIR range (400–2600 nm)
    band_mask = (wav >= 500) & (wav <= 2600)
    wav = wav[band_mask]
    for mineral in CLASS_NAMES:
        for tier in TIERS:
            arr = spectra[mineral][tier]
            if arr.shape[0] > 0:
                spectra[mineral][tier] = arr[:, band_mask]

    print('Pixel counts per mineral / tier:')
    for mineral in CLASS_NAMES:
        counts = [spectra[mineral][t].shape[0] for t in TIERS]
        print(f'  {mineral}: tier1={counts[0]:,}  tier2={counts[1]:,}  tier3={counts[2]:,}')

    fig, axes = plt.subplots(5, 2, figsize=(13, 17), constrained_layout=True)
    fig.suptitle(
        'CRISM Mineral Spectra by Confidence Tier — T0435 + T0434\n'
        'Left: raw reflectance   |   Right: ratio vs. all other classified pixels',
        fontsize=10,
    )

    for ri, mineral in enumerate(CLASS_NAMES):
        ax_raw = axes[ri, 0]
        ax_rat = axes[ri, 1]
        mcolor = MINERAL_COLORS[mineral]
        other_mean = compute_other_mean(spectra, mineral)

        for ti, tier in enumerate(TIERS):
            arr = spectra[mineral][tier]
            if arr.shape[0] == 0:
                continue

            mean_spec = np.nanmean(arr, axis=0)
            std_spec = np.nanstd(arr, axis=0)
            tcolor = TIER_COLORS[ti]
            tlabel = f'{TIER_LABELS[ti]} (n={arr.shape[0]:,})'

            # --- Raw ---
            ax_raw.plot(wav, mean_spec, color=tcolor, lw=1.2, label=tlabel, zorder=3)
            ax_raw.fill_between(
                wav,
                mean_spec - std_spec,
                mean_spec + std_spec,
                color=tcolor, alpha=0.12, zorder=2,
            )

            # --- Ratio ---
            if other_mean is not None:
                with np.errstate(invalid='ignore', divide='ignore'):
                    ratio = np.where(other_mean > 0, mean_spec / other_mean, np.nan)
                ax_rat.plot(wav, ratio, color=tcolor, lw=1.2, label=tlabel, zorder=3)

        # Reference line at ratio = 1
        ax_rat.axhline(1.0, color='#9e9e9e', lw=0.8, linestyle='--', zorder=1)

        # Style
        for ax, title in [(ax_raw, f'{mineral} — raw reflectance'),
                          (ax_rat, f'{mineral} / other classes')]:
            ax.set_title(title, fontsize=9, fontweight='bold', color=mcolor)
            ax.set_xlabel('Wavelength (nm)', fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_xlim(wav[0], wav[-1])
            ax.legend(fontsize=6, loc='upper right')

        ax_raw.set_ylabel('Reflectance', fontsize=7)
        ax_rat.set_ylabel('Ratio', fontsize=7)

    out = os.path.join(PROJ, 'reports', 'fig_mineral_spectra.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()

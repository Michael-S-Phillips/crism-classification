"""
Spectral diagnostic plots for the Nili Fossae 2×2 tile area.

Identical format to plot_mineral_spectra.py (raw mean±std + ratio vs.
'other'-class denominator), but sourced from the four Nili Fossae tiles
(T1249/T1250/T1321/T1322) rather than the calibration tiles.

Output: reports/fig_nili_spectra.png

Usage:
    conda run -n crism python scripts/plot_nili_spectra.py
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fig_style import DPI, MINERAL_COLORS
from scripts.plot_mineral_spectra import (
    collect_spectra,
    compute_other_mean,
    interp_bad_bands,
    BAD_BAND_RANGES,
    CLASS_NAMES,
    TIERS,
    TIER_COLORS,
    TIER_LABELS,
)

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DIR = os.path.join(PROJ, 'data', 'vector')

TILES = [
    {
        'img': '/mnt/mrdr/mc13/t1249_mrral_20n073_0327_4.img',
        'gpkg': os.path.join(VECTOR_DIR, 't1249_mrral_20n073_0327_4_mineral_map.gpkg'),
    },
    {
        'img': '/mnt/mrdr/mc13/t1250_mrral_20n078_0327_4.img',
        'gpkg': os.path.join(VECTOR_DIR, 't1250_mrral_20n078_0327_4_mineral_map.gpkg'),
    },
    {
        'img': '/mnt/mrdr/mc13/t1321_mrral_25n073_0327_4.img',
        'gpkg': os.path.join(VECTOR_DIR, 't1321_mrral_25n073_0327_4_mineral_map.gpkg'),
    },
    {
        'img': '/mnt/mrdr/mc13/t1322_mrral_25n078_0327_4.img',
        'gpkg': os.path.join(VECTOR_DIR, 't1322_mrral_25n078_0327_4_mineral_map.gpkg'),
    },
]


def main():
    print('Collecting pixel spectra ...')
    spectra, wav = collect_spectra(TILES)

    # Restrict to 500–2600 nm
    band_mask = (wav >= 500) & (wav <= 2600)
    wav = wav[band_mask]
    for mineral in CLASS_NAMES:
        for tier in TIERS:
            arr = spectra[mineral][tier]
            if arr.shape[0] > 0:
                spectra[mineral][tier] = arr[:, band_mask]

    print('Pixel counts per mineral / tier:')
    for mineral in CLASS_NAMES:
        parts = '  '.join(f'tier{t}={spectra[mineral][t].shape[0]:,}' for t in TIERS)
        print(f'  {mineral}: {parts}')

    fig, axes = plt.subplots(5, 2, figsize=(13, 17), constrained_layout=True)
    fig.suptitle(
        'CRISM Mineral Spectra by Confidence Tier — Nili Fossae (T1249/T1250/T1321/T1322)\n'
        'Left: raw reflectance   |   Right: ratio vs. \'other\' (neutral baseline)',
        fontsize=10,
    )

    for ri, mineral in enumerate(CLASS_NAMES):
        ax_raw = axes[ri, 0]
        ax_rat = axes[ri, 1]
        mcolor = MINERAL_COLORS[mineral]
        _om = compute_other_mean(spectra)
        other_mean = interp_bad_bands(_om, wav) if _om is not None else None

        for ti, tier in enumerate(TIERS):
            arr = spectra[mineral][tier]
            if arr.shape[0] == 0:
                continue

            mean_spec = interp_bad_bands(np.nanmean(arr, axis=0), wav)
            std_spec = interp_bad_bands(np.nanstd(arr, axis=0), wav)
            tcolor = TIER_COLORS[ti]
            tlabel = f'{TIER_LABELS[ti]} (n={arr.shape[0]:,})'

            ax_raw.plot(wav, mean_spec, color=tcolor, lw=1.2, label=tlabel, zorder=3)
            ax_raw.fill_between(wav, mean_spec - std_spec, mean_spec + std_spec,
                                color=tcolor, alpha=0.12, zorder=2)

            if other_mean is not None:
                with np.errstate(invalid='ignore', divide='ignore'):
                    ratio = np.where(other_mean > 0, mean_spec / other_mean, np.nan)
                ratio = interp_bad_bands(ratio, wav)
                ax_rat.plot(wav, ratio, color=tcolor, lw=1.2, label=tlabel, zorder=3)

        ax_rat.axhline(1.0, color='#9e9e9e', lw=0.8, linestyle='--', zorder=1)

        for ax, title in [(ax_raw, f'{mineral} — raw reflectance'),
                          (ax_rat, f'{mineral} / other (neutral)')]:
            ax.set_title(title, fontsize=9, fontweight='bold', color=mcolor)
            ax.set_xlabel('Wavelength (nm)', fontsize=7)
            ax.tick_params(labelsize=6)
            ax.set_xlim(wav[0], wav[-1])
            ax.legend(fontsize=6, loc='upper right')

        ax_raw.set_ylabel('Reflectance', fontsize=7)
        ax_rat.set_ylabel('Ratio', fontsize=7)

    out = os.path.join(PROJ, 'reports', 'fig_nili_spectra.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()

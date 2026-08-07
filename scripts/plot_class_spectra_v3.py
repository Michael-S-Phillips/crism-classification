"""Class-mean reflectance with project color scheme + new MTRDR plag.

Olivine is a single class (union of t1 + t2 labels). Plagioclase pool
augments the parquet labels with the new MTRDR plag center pixels.

Output: reports/fig_class_spectra_v3.png
"""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fig_style import MINERAL_COLORS, apply_style, despine

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJ, 'reports')
PARQUET = os.path.join(PROJ, 'data', 'mrral_pixels.parquet')
MTRDR_NPY = os.path.join(PROJ, 'data', 'patch_cache', 'mtrdr_plag_patches_p7.npy')

BAND_COLS = [f'm{i}' for i in range(59)]


def get_wavelengths():
    hdrs = glob.glob('/Volumes/Mars_GIS/CRISM/MRDR/mc*/t*_mrral_*.hdr')
    if hdrs:
        try:
            import spectral.io.envi as envi
            hdr = envi.read_envi_header(hdrs[0])
            return np.array(hdr['wavelength'], dtype=float)[:59]
        except Exception:
            pass
    return np.linspace(410, 2457, 59)


def class_mean(df, mask):
    if not mask.any():
        return None, 0
    arr = df.loc[mask, BAND_COLS].values.astype('float32')
    return arr.mean(axis=0), int(mask.sum())


def main():
    apply_style()
    wl = get_wavelengths()
    cols = BAND_COLS + ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase']
    df = pd.read_parquet(PARQUET, columns=cols)

    olivine_mask = (df['olivine_t1'] > 0.4) | (df['olivine_t2'] > 0.4)
    lcp_mask     = df['lcp'] > 0.4
    hcp_mask     = df['hcp'] > 0.4
    plag_mask    = df['plagioclase'] > 0.4

    olivine_mean, n_oli = class_mean(df, olivine_mask)
    lcp_mean,     n_lcp = class_mean(df, lcp_mask)
    hcp_mean,     n_hcp = class_mean(df, hcp_mask)
    plag_parquet, n_plag_p = class_mean(df, plag_mask)

    # New MTRDR plag patches — center pixel of each 7x7
    mtrdr = np.load(MTRDR_NPY)
    mtrdr_center = mtrdr[:, 3, 3, :]
    n_plag_m = mtrdr_center.shape[0]

    # Augmented plagioclase: union mean weighted by sample count
    if plag_parquet is not None:
        plag_all = (plag_parquet * n_plag_p + mtrdr_center.mean(axis=0) * n_plag_m) / (n_plag_p + n_plag_m)
        n_plag = n_plag_p + n_plag_m
    else:
        plag_all = mtrdr_center.mean(axis=0)
        n_plag = n_plag_m

    fig, ax = plt.subplots(figsize=(9, 5.2))

    series = [
        ('olivine',     olivine_mean, n_oli, MINERAL_COLORS['olivine']),
        ('LCP',         lcp_mean,     n_lcp, MINERAL_COLORS['lcp']),
        ('HCP',         hcp_mean,     n_hcp, MINERAL_COLORS['hcp']),
        ('plagioclase', plag_all,     n_plag, MINERAL_COLORS['plagioclase']),
    ]
    for label, mean, n, color in series:
        if mean is None:
            continue
        ax.plot(wl, mean, color=color, linewidth=2.2,
                label=f'{label}  (n={n:,})')

    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Reflectance', fontsize=12)
    ax.set_xlim(450, wl[-1])
    ax.set_ylim(0, 0.22)
    ax.legend(loc='lower right', frameon=False, fontsize=11)
    despine(ax)
    ax.grid(alpha=0.25)
    ax.set_title('Mean reflectance per mineral class', fontsize=13, color='#333')

    plt.tight_layout()
    out = os.path.join(REPORTS_DIR, 'fig_class_spectra_v3.png')
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'Saved {out}')
    print(f'  olivine n={n_oli:,}, LCP n={n_lcp:,}, HCP n={n_hcp:,}, '
          f'plag n={n_plag:,} (parquet {n_plag_p:,} + MTRDR {n_plag_m:,})')


if __name__ == '__main__':
    main()

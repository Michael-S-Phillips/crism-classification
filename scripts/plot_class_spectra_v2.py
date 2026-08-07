"""
Figure 3: Class spectral profiles — mean ± 1σ reflectance per mineral class.

Output: reports/fig_class_spectra_v2.png
"""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Insert project root so package imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fig_style import FIGSIZE_GRID, DPI, MINERAL_COLORS, LABEL_COLS, apply_style, despine
from config_loader import load_config

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJ, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

BAND_COLS = [f'm{i}' for i in range(59)]


def get_wavelengths(data_root: str) -> np.ndarray:
    """Read wavelength array from first mrral .hdr found under data_root.

    Falls back to np.linspace(410, 2457, 59) if no .hdr found or parse fails.
    """
    hdrs = glob.glob(os.path.join(data_root, '**', '*mrral*.hdr'), recursive=True)
    if hdrs:
        try:
            import spectral.io.envi as envi
            hdr = envi.read_envi_header(hdrs[0])
            return np.array(hdr['wavelength'], dtype=float)[:59]
        except (KeyError, ValueError, OSError) as exc:
            import warnings
            warnings.warn(
                f"Could not parse wavelengths from {hdrs[0]!r}: {exc}; using fallback."
            )
    return np.linspace(410, 2457, 59)


def main():
    apply_style()
    cfg = load_config()
    parquet_path = os.path.join(cfg['output_dir'], 'mrral_pixels.parquet')
    df = pd.read_parquet(parquet_path)
    wavelengths = get_wavelengths(cfg['data_root'])

    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_GRID, sharey=True)
    axes_flat = axes.flatten()  # indices 0–5, row-major

    # Panels 0–4: one class each
    for idx, cls in enumerate(LABEL_COLS):
        ax = axes_flat[idx]
        mask   = df[f'label_{cls}'] > 0.4
        subset = df.loc[mask, BAND_COLS].values.astype('float32')
        n      = len(subset)
        ax.set_title(f'{cls} (n={n:,})', fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Wavelength (nm)', fontsize=9)
        if idx % 3 == 0:
            ax.set_ylabel('Reflectance', fontsize=9)
        despine(ax)
        if n == 0:
            continue
        color  = MINERAL_COLORS[cls]
        mean   = subset.mean(axis=0)
        # ddof=1: sample std; for n=1 this yields NaN → fill_between draws nothing (safe)
        std    = subset.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
        ax.plot(wavelengths, mean, color=color, linewidth=1.5)
        lo = np.clip(mean - std, 0, 1)
        hi = np.clip(mean + std, 0, 1)
        ax.fill_between(wavelengths, lo, hi, color=color, alpha=0.25)

    # Panel 5: all-class overlay
    ax5 = axes_flat[5]
    for cls in LABEL_COLS:
        mask   = df[f'label_{cls}'] > 0.4
        subset = df.loc[mask, BAND_COLS].values.astype('float32')
        if len(subset) == 0:
            continue
        mean   = subset.mean(axis=0)
        ax5.plot(wavelengths, mean, color=MINERAL_COLORS[cls], label=cls, linewidth=1.5)
    ax5.set_ylim(0, 1)
    ax5.set_title('All classes', fontsize=10)
    ax5.legend(loc='upper right', fontsize=8)
    ax5.set_xlabel('Wavelength (nm)', fontsize=9)
    despine(ax5)

    plt.tight_layout()
    out = os.path.join(REPORTS_DIR, 'fig_class_spectra_v2.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()

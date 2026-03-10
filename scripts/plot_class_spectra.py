"""
Plot representative spectra for each mineral class from mrral_pixels.parquet.

For each class, we select:
  - median spectrum of all high-confidence positive pixels (the "representative" line)
  - 10th–90th percentile envelope (shaded)

Usage:
    python scripts/plot_class_spectra.py
    python scripts/plot_class_spectra.py --out reports/class_spectra.png
    python scripts/plot_class_spectra.py --split train --conf_tier High
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Wavelengths for bands m0..m58 (nm), derived from mrral .hdr files
WAVELENGTHS = [
    410.12, 442.63, 533.74, 598.86, 650.99, 683.59, 709.68, 742.30, 774.92,
    801.04, 833.68, 859.81, 892.48, 925.16, 951.31, 984.01, 1021.00, 1023.27,
    1047.20, 1055.99, 1079.96, 1152.06, 1211.09, 1250.45, 1257.01, 1263.57,
    1276.70, 1329.21, 1368.61, 1394.89, 1427.73, 1467.16, 1500.03, 1506.61,
    1559.21, 1625.00, 1657.91, 1690.82, 1750.09, 1809.39, 1875.30, 1928.06,
    1974.24, 1980.84, 2007.23, 2066.64, 2119.48, 2139.30, 2165.72, 2205.38,
    2231.82, 2251.65, 2291.33, 2317.79, 2331.02, 2350.87, 2390.58, 2430.30,
    2456.79,
]

BAND_COLS = [f'm{i}' for i in range(59)]

# Display names, colours, and diagnostic absorption wavelengths (nm)
CLASS_META = {
    'olivine_t1': dict(
        label='Olivine type 1',
        color='#2ca02c',
        absorptions=[1050, 1250],
    ),
    'olivine_t2': dict(
        label='Olivine type 2',
        color='#98df8a',
        absorptions=[820, 1050],
    ),
    'lcp': dict(
        label='Low-Ca pyroxene (LCP)',
        color='#1f77b4',
        absorptions=[920, 1820],
    ),
    'hcp': dict(
        label='High-Ca pyroxene (HCP)',
        color='#aec7e8',
        absorptions=[1000, 2300],
    ),
    'plagioclase': dict(
        label='Plagioclase feldspar',
        color='#d62728',
        absorptions=[1250],
    ),
    'other': dict(
        label='Other / unclassified',
        color='#7f7f7f',
        absorptions=[],
    ),
}


def load_spectra(df: pd.DataFrame, label_col: str, conf_tiers=('High', 'Moderate')) -> np.ndarray:
    """Return array of spectra for pixels positively labelled for label_col."""
    mask = df[label_col] > 0.4
    if conf_tiers:
        mask &= df['confidence_tier'].isin(conf_tiers)
    sub = df.loc[mask, BAND_COLS].values.astype('float32')
    return sub


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=os.path.join(PROJ, 'reports', 'class_spectra.png'))
    parser.add_argument('--split', default=None, help='Filter to train/val/test split')
    parser.add_argument('--conf_tier', nargs='+', default=['High', 'Moderate'],
                        help='Confidence tiers to include')
    parser.add_argument('--max_pixels', type=int, default=50_000,
                        help='Cap per-class sample size for speed')
    parser.add_argument('--ratio', action='store_true',
                        help='Divide each class spectrum by the grand mean of all labeled pixels')
    args = parser.parse_args()

    cfg = yaml.safe_load(open(os.path.join(PROJ, 'config.yaml')))
    parquet = os.path.join(cfg['output_dir'], 'mrral_pixels.parquet')
    print(f'Loading {parquet}...')
    df = pd.read_parquet(parquet)
    if args.split:
        df = df[df['split'] == args.split]
    print(f'  {len(df):,} pixels loaded')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    wl = np.array(WAVELENGTHS)
    classes = list(CLASS_META.keys())

    # Compute grand mean over all labeled pixels (any class > 0.4) for ratio mode
    any_label = df[[c for c in classes if c in df.columns]].gt(0.4).any(axis=1)
    all_labeled = np.clip(df.loc[any_label, BAND_COLS].values.astype('float32'), 0.0, 0.5)
    rng_grand = np.random.default_rng(0)
    if len(all_labeled) > 200_000:
        idx = rng_grand.choice(len(all_labeled), 200_000, replace=False)
        all_labeled = all_labeled[idx]
    grand_mean = all_labeled.mean(axis=0)
    grand_mean = np.where(grand_mean < 1e-6, 1e-6, grand_mean)  # avoid divide-by-zero

    fig, axes = plt.subplots(
        3, 2, figsize=(12, 10),
        sharex=True, sharey=False,
        constrained_layout=True,
    )
    axes_flat = axes.flatten()

    for ax, cls in zip(axes_flat, classes):
        meta = CLASS_META[cls]
        spectra = load_spectra(df, cls, conf_tiers=args.conf_tier)

        if len(spectra) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title(meta['label'])
            continue

        # Subsample if needed
        rng = np.random.default_rng(42)
        if len(spectra) > args.max_pixels:
            idx = rng.choice(len(spectra), args.max_pixels, replace=False)
            spectra = spectra[idx]

        median = np.median(spectra, axis=0)
        p10 = np.percentile(spectra, 10, axis=0)
        p90 = np.percentile(spectra, 90, axis=0)

        # Clip noise: reflectance outside [0, 0.5] is physically implausible
        spectra = np.clip(spectra, 0.0, 0.5)
        median = np.median(spectra, axis=0)
        p10 = np.percentile(spectra, 10, axis=0)
        p90 = np.percentile(spectra, 90, axis=0)

        if args.ratio:
            median = median / grand_mean
            p10 = p10 / grand_mean
            p90 = p90 / grand_mean

        color = meta['color']
        ax.fill_between(wl, p10, p90, alpha=0.25, color=color, linewidth=0)
        ax.plot(wl, median, color=color, linewidth=1.8)
        ax.plot(wl, p10, color=color, linewidth=0.6, linestyle='--', alpha=0.6)
        ax.plot(wl, p90, color=color, linewidth=0.6, linestyle='--', alpha=0.6)

        if args.ratio:
            ax.axhline(1.0, color='k', linewidth=0.8, linestyle='-', alpha=0.3)

        # Add some breathing room above p90 for absorption labels
        ymax = float(p90.max())
        ymin = float(p10.min()) - 0.01 * (float(p90.max()) - float(p10.min()))
        if not args.ratio:
            ymin = max(0.0, ymin)
        yrange = ymax - ymin
        ax.set_ylim(ymin, ymax + 0.12 * yrange)

        # Mark diagnostic absorptions with vertical dashed lines
        for ab_nm in meta['absorptions']:
            ax.axvline(ab_nm, color='k', linewidth=0.8, linestyle=':', alpha=0.5)
            ax.text(ab_nm + 12, ymax + 0.08 * yrange,
                    f'{ab_nm} nm', fontsize=7, va='top', color='#333333')

        n_label = f'{len(spectra):,}' if len(spectra) < args.max_pixels else f'~{args.max_pixels:,}'
        ax.set_title(f'{meta["label"]}\n(n = {n_label} pixels)', fontsize=10)
        ylabel = 'Class / mean reflectance' if args.ratio else 'Reflectance'
        ax.set_ylabel(ylabel, fontsize=9)
        fmt = '%.2f' if args.ratio else '%.3f'
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter(fmt))
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
        ax.set_xlim(wl[0], wl[-1])

    for ax in axes[-1]:
        ax.set_xlabel('Wavelength (nm)', fontsize=9)

    if args.ratio:
        title = ('Representative CRISM MRDR spectra — ratio to grand mean\n'
                 f'(conf: {", ".join(args.conf_tier)}; median ± 10th–90th percentile / grand mean)')
    else:
        title = ('Representative CRISM MRDR spectra by mineral class\n'
                 f'(conf: {", ".join(args.conf_tier)}; median ± 10th–90th percentile)')
    fig.suptitle(title, fontsize=11)

    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f'Saved → {args.out}')


if __name__ == '__main__':
    main()

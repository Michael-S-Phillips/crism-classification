"""Are the model's CONFIDENT predictions spectrally consistent with the class?

Motivating question (2026-08-10 floor-test review): "olivine and lcp are
overfiring on bland, with some highly confident positives being spectrally bland
in reality, especially at Nili." That is a checkable claim, and answering it
decides whether a threshold-ladder win is real detection or probability
saturation.

Method
  1. Read a floor-test probs .npz (probs (H, W, C) + valid_mask + class_names).
  2. For each confidence band, sample pixels the model called POSITIVE for the
     target class and pull their real 59-band spectra from the source tile.
  3. Continuum-remove, drop the detector-overlap bands, and assign each pixel to
     its nearest class endmember by SPECTRAL ANGLE. Endmembers come from the
     training data via scripts/sample_class_spectra.py, so "what the model said"
     is being judged against "what the training labels look like".
  4. Report, per confidence band, the fraction of the model's own positives that
     are spectrally nearest to some OTHER class -- especially bland.

What a result means
  * confident positives nearest their own class      -> real detections
  * confident positives nearest bland                -> the class is firing on
                                                        spectrally bland ground
  * agreement NOT improving as confidence rises      -> probability saturation:
                                                        the score carries no
                                                        information about
                                                        spectral support

This is a diagnostic, not ground truth. Nearest-endmember SAM is a weak
classifier (olivine scores 0.23 recall per scripts/sam_confusion_matrix.py) and
the model has 7x7 spatial context it cannot see. Read the TREND across
confidence bands, not the absolute agreement rate.

Usage
    python scripts/sample_class_spectra.py            # writes the endmember npz
    python scripts/audit_confident_predictions.py \
        --probs /tmp/floor_test_handcore_level/nili/t1250_probs.npz \
        --npz <spectra.npz> --classes olivine lcp
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.continuum_removal import continuum_removed, good_band_mask_59  # noqa: E402
from sam_analysis.sam import spectral_angle  # noqa: E402

N_BANDS = 59
NODATA, PHYS_MAX, CLIP_MAX = 65535.0, 1.0, 0.5
BANDS = [f'm{i}' for i in range(N_BANDS)]
DEFAULT_BANDS = [0.50, 0.75, 0.90, 0.99]

# Endmember series from sample_class_spectra.py -> display label.
SERIES = [('olivine', 'olivine__hand'), ('lcp', 'lcp__hand'),
          ('hcp', 'hcp__hand'), ('plagioclase', 'plagioclase__hand'),
          ('plag(MTRDR)', 'plagioclase__MTRDR'),
          ('alteration', 'alteration__hand'), ('bland', 'bland__v3 review'),
          ('junk', 'junk__v3 review')]


def _prep(a: np.ndarray, good: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32).copy()
    a[(a > PHYS_MAX) | (a == NODATA) | (~np.isfinite(a))] = np.nan
    a = np.clip(a, 0.0, CLIP_MAX)
    a = a[np.isfinite(a).all(axis=1)]
    if not len(a):
        return a.reshape(0, int(good.sum()))
    return continuum_removed(a[:, None, None, :].copy())[:, 0, 0, :][:, good]


def _find_tile_img(tile: str, data_root: str) -> list[str]:
    for pattern in (os.path.join(data_root, 'mc*', f'{tile}_mrral*.img'),
                    os.path.join(data_root, f'{tile}_mrral*.img')):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--probs', required=True, help='floor-test *_probs.npz')
    ap.add_argument('--npz', required=True, help='sample_class_spectra.py output')
    ap.add_argument('--classes', nargs='+', default=['olivine', 'lcp'])
    ap.add_argument('--thresholds', nargs='+', type=float, default=DEFAULT_BANDS)
    ap.add_argument('--sample', type=int, default=1500,
                    help='pixels sampled per confidence band')
    ap.add_argument('--data_root', default=None)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    root = args.data_root
    if root is None:
        from config_loader import load_config
        root = load_config()['data_root']

    good = good_band_mask_59()
    rng = np.random.default_rng(args.seed)

    d = np.load(args.npz)
    names, ends = [], []
    for label, key in SERIES:
        if key not in d.files:
            continue
        X = _prep(d[key], good)
        if len(X) >= 20:
            names.append(label)
            ends.append(np.nanmedian(X, axis=0))
    ends = np.stack(ends)

    p = np.load(args.probs, allow_pickle=True)
    probs, valid = p['probs'], p['valid_mask']
    cls_names = [str(x) for x in p['class_names']]
    tile = os.path.basename(args.probs).split('_probs')[0]

    hits = _find_tile_img(tile, root)
    if not hits:
        raise SystemExit(f'ERROR: no mrral .img for {tile} under {root}')

    import rasterio
    print(f'tile {tile}   endmembers: {", ".join(names)}')
    print(f'source: {hits[0]}\n')

    with rasterio.open(hits[0]) as src:
        bands = list(range(1, N_BANDS + 1))
        for cls in args.classes:
            if cls not in cls_names:
                print(f'  {cls}: not in {cls_names}')
                continue
            ci = cls_names.index(cls)
            print(f'=== model class: {cls} ===')
            print(f'  {"band":<12}{"n":>7}{"→own":>8}{"→bland":>9}'
                  f'{"top other":>16}{"mean ang":>10}')
            print('  ' + '-' * 64)
            for t in args.thresholds:
                m = valid & (probs[:, :, ci] >= t)
                rr, cc = np.nonzero(m)
                if len(rr) == 0:
                    print(f'  >={t:<10.2f}{0:>7}')
                    continue
                if len(rr) > args.sample:
                    k = rng.choice(len(rr), args.sample, replace=False)
                    rr, cc = rr[k], cc[k]
                spec = np.empty((len(rr), N_BANDS), dtype=np.float32)
                for i, (r, c) in enumerate(zip(rr, cc)):
                    win = rasterio.windows.Window(int(c), int(r), 1, 1)
                    spec[i] = src.read(bands, window=win).astype(np.float32)[:, 0, 0]
                X = _prep(spec, good)
                if not len(X):
                    print(f'  >={t:<10.2f}{0:>7}   (all nodata)')
                    continue
                angs = np.stack([spectral_angle(X, e) for e in ends], axis=1)
                pred = np.nanargmin(angs, axis=1)
                own = names.index(cls) if cls in names else -1
                frac_own = float((pred == own).mean()) if own >= 0 else float('nan')
                bi = names.index('bland') if 'bland' in names else -1
                frac_bland = float((pred == bi).mean()) if bi >= 0 else float('nan')
                counts = np.bincount(pred, minlength=len(names)).astype(float)
                counts[own] = -1
                other = names[int(np.argmax(counts))]
                other_f = float((pred == np.argmax(counts)).mean())
                print(f'  >={t:<10.2f}{len(X):>7}{frac_own:>8.2f}{frac_bland:>9.2f}'
                      f'{other + " " + format(other_f, ".2f"):>16}'
                      f'{np.degrees(np.nanmin(angs, axis=1)).mean():>10.2f}')
            print()


if __name__ == '__main__':
    main()

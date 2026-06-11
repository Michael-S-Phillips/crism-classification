"""
Compute global percentile-based probability thresholds for Vectroscopy vectorization.

Pools valid-pixel probabilities from multiple tile .npz files (produced by
classify_tile_supervised.py --save_probs) and computes percentile thresholds
per mineral class. Output JSON is consumed by vectorize_tile_minerals.py.

Usage:
    python scripts/compute_global_thresholds.py \
        --probs /tmp/t0434_mrral_40s318_0327_4_probs.npz /tmp/t0435_mrral_40s323_0327_4_probs.npz \
        --out config/vectroscopy_thresholds.json \
        --percentiles 33 67 90
"""
import argparse
import json
import os
from datetime import date
from typing import Dict, List

import numpy as np
from skimage.filters import threshold_otsu

CLASS_NAMES = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']

def _check_npz_channels(data, expected_channels):
    """Fail loudly if the npz was produced by a checkpoint whose class list
    doesn't match this script's channel constants (e.g. a 6-class alteration
    model feeding a 5-class vectorizer would silently drop/mislabel layers)."""
    if 'class_names' not in getattr(data, 'files', []):
        return  # legacy 5-class npz, no metadata — assume constants are right
    names = [str(x) for x in data['class_names']]
    if names != list(expected_channels):
        raise SystemExit(
            f'npz class_names {names} != this script\'s channel order '
            f'{list(expected_channels)}. This script needs updating for that '
            f'checkpoint\'s class list (6-class alteration support pending).')

DEFAULT_MORPHOLOGY = {
    'median_filter_size': 3,
    'median_filter_iterations': 1,
    'sieve_min_pixels': 9,
    'majority_filter_iterations': 3,
    'simplify_tolerance_meters': 200,
}


def pool_valid_probs(npz_paths: List[str]) -> Dict[int, np.ndarray]:
    """Pool valid-pixel probabilities per class across all input tiles.

    Args:
        npz_paths: list of .npz paths produced by classify_tile_supervised --save_probs

    Returns:
        dict mapping class index (0-4) → 1-D float32 array of valid-pixel probs
    """
    pooled = {ci: [] for ci in range(5)}
    for path in npz_paths:
        data = np.load(path)
        _check_npz_channels(data, CLASS_NAMES)
        probs = data['probs']          # (H, W, 5)
        valid_mask = data['valid_mask']  # (H, W) bool
        for ci in range(5):
            pooled[ci].append(probs[:, :, ci][valid_mask])
    return {ci: np.concatenate(pooled[ci]) for ci in range(5)}


def compute_otsu_thresholds(pooled: Dict[int, np.ndarray],
                            class_names: List[str],
                            ) -> tuple:
    """Compute per-class thresholds using Otsu's method + signal-distribution percentiles.

    Steps:
      1. Otsu's threshold splits noise vs. signal pixels.
      2. p50 / p67 / p90 / p95 / p99 of above-Otsu pixels become tier 1–5 thresholds.

    Args:
        pooled: dict from pool_valid_probs()
        class_names: list of class name strings in class-index order

    Returns:
        thresholds: dict name → [p50, p67, p90, p95, p99] signal percentile values
        otsu_values: dict name → float (Otsu split, stored for diagnostics)
    """
    thresholds = {}
    otsu_values = {}
    for ci, name in enumerate(class_names):
        vals = pooled[ci]
        otsu = float(threshold_otsu(vals))
        signal = vals[vals > otsu]
        if len(signal) == 0:
            signal = vals          # degenerate fallback
        t = [float(np.percentile(signal, p)) for p in [50, 67, 90, 95, 99]]
        thresholds[name] = t
        otsu_values[name] = otsu
        n_sig = len(signal)
        print(f'  {name:12s}: otsu={otsu:.4f}  signal={n_sig:,}/{len(vals):,} '
              f'({100*n_sig/len(vals):.1f}%)  tiers={[f"{v:.4f}" for v in t]}')
    return thresholds, otsu_values


def compute_thresholds(pooled: Dict[int, np.ndarray],
                       class_names: List[str],
                       percentiles: List[int]) -> Dict[str, List[float]]:
    """Compute percentile thresholds per mineral class.

    Args:
        pooled: dict from pool_valid_probs()
        class_names: list of class name strings in class-index order
        percentiles: list of 3 percentile values, e.g. [33, 67, 90]

    Returns:
        dict mapping mineral name → list of 3 float threshold values
    """
    thresholds = {}
    for ci, name in enumerate(class_names):
        vals = pooled[ci]
        t = [float(np.percentile(vals, p)) for p in percentiles]
        thresholds[name] = t
        print(f'  {name:12s}: {[f"{v:.4f}" for v in t]}  '
              f'(n={len(vals):,}, mean={vals.mean():.3f})')
    return thresholds


def write_thresholds_json(out_path: str, thresholds: Dict[str, List[float]],
                          tiles_used: List[str], percentiles: List[int],
                          morphology: dict,
                          otsu_thresholds: Dict[str, float] = None) -> None:
    """Write calibrated thresholds to JSON.

    Args:
        out_path: output file path
        thresholds: dict from compute_thresholds()
        tiles_used: list of tile identifiers used for calibration
        percentiles: percentile values used
        morphology: morphological parameter defaults (documentation only)
    """
    payload = {
        'generated': str(date.today()),
        'tiles_used': tiles_used,
        'percentiles': percentiles,
        'thresholds': thresholds,
        'morphology': morphology,
    }
    if otsu_thresholds is not None:
        payload['otsu_thresholds'] = otsu_thresholds
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Compute global percentile thresholds from tile prob rasters.')
    parser.add_argument('--probs', nargs='+', required=True, metavar='PATH',
                        help='.npz files from classify_tile_supervised --save_probs')
    parser.add_argument('--out', default='config/vectroscopy_thresholds.json',
                        help='Output JSON path (default: config/vectroscopy_thresholds.json)')
    parser.add_argument('--percentiles', type=int, nargs=3, default=[33, 67, 90],
                        metavar=('P1', 'P2', 'P3'),
                        help='Three percentile values for tiers 1/2/3 (default: 33 67 90)')
    parser.add_argument('--otsu', action='store_true',
                        help='Use Otsu noise/signal split + p50/p67/p90 of signal distribution '
                             'instead of global percentiles. Ignores --percentiles.')
    args = parser.parse_args()

    print(f'Pooling probs from {len(args.probs)} tile(s)...')
    pooled = pool_valid_probs(args.probs)

    def _npz_to_tile_id(p):
        stem = os.path.splitext(os.path.basename(p))[0]
        return stem[:-6] if stem.endswith('_probs') else stem

    tiles_used = [_npz_to_tile_id(p) for p in args.probs]

    if args.otsu:
        print('Computing Otsu + signal-distribution (p50/p67/p90/p95/p99) thresholds per class:')
        thresholds, otsu_vals = compute_otsu_thresholds(pooled, CLASS_NAMES)
        write_thresholds_json(args.out, thresholds, tiles_used,
                              [50, 67, 90, 95, 99], DEFAULT_MORPHOLOGY,
                              otsu_thresholds=otsu_vals)
    else:
        print(f'Computing {args.percentiles} percentile thresholds per class:')
        thresholds = compute_thresholds(pooled, CLASS_NAMES, args.percentiles)
        write_thresholds_json(args.out, thresholds, tiles_used,
                              args.percentiles, DEFAULT_MORPHOLOGY)
    print(f'Saved → {args.out}')


if __name__ == '__main__':
    main()

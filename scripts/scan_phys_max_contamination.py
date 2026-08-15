"""Find deployed tiles whose classification was affected by the load_tile PHYS_MAX bug.

Before commit 82afe80, `classify_tile_supervised.load_tile` masked only the
==NODATA and non-finite tests, then clipped to CLIP_MAX = 0.5. A blue-edge value
of ~3900 I/F therefore became 0.5 -- a plausible-looking reflectance -- and
survived as valid data. The training path (CRISMSpectralPatchDataset) has always
applied `patch > PHYS_MAX -> 0.0`, so the model was trained on 0.0 and deployed
on 0.5 for exactly those pixels.

A tile needs re-running if it has at least one pixel that the OLD code called
valid and the NEW code calls invalid:

    old_valid  = ~((band == NODATA) | ~isfinite(band)).any(axis=0)
    new_valid  = old_valid & ~(band > PHYS_MAX).any(axis=0)
    affected   = old_valid & ~new_valid

Note this undercounts the true blast radius. The classifier reads a 7x7 patch, so
a pixel that stays valid still changes if any neighbour was contaminated -- on
t1389, 0.79% of the pixels valid under BOTH versions moved by more than 0.01,
with a maximum probability shift of 0.96. The list this script writes is
therefore a list of tiles to re-run, not a count of pixels that changed.

Usage
    python scripts/scan_phys_max_contamination.py \
        --probs_dir data/mc_deploy_pyx/probs \
        --out reports/phys_max_contamination.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config_loader import load_config  # noqa: E402
from data.dataset import CRISMSpectralPatchDataset as _DS  # noqa: E402
from scripts.classify_tile_supervised import N_SRC_BANDS  # noqa: E402

NODATA = _DS.NODATA
PHYS_MAX = _DS.PHYS_MAX


def scan_tile(img_path: str) -> dict:
    """Count pixels the old load_tile kept and the new one drops.

    Only the first N_SRC_BANDS bands are examined, because that is exactly what
    load_tile reads. An mrral cube carries ~72 bands; the trailing ones are not
    reflectance on the model's grid and hold values up to 1e10. A first version
    of this script scanned all of them and reported 157 of 183 tiles and 91.6%
    of pixels "affected" -- it was measuring bands the classifier never sees.

    Accumulates band by band so peak memory is one band, not the whole cube --
    59 bands of an mrral tile is ~500 MB as float32 and these run in parallel.
    """
    with rasterio.open(img_path) as src:
        h, w = src.height, src.width
        n_bands = min(N_SRC_BANDS, src.count)
        if src.count < N_SRC_BANDS:
            print(f'  WARNING: {os.path.basename(img_path)} has {src.count} '
                  f'bands, fewer than N_SRC_BANDS={N_SRC_BANDS}', file=sys.stderr)
        bad = np.zeros((h, w), dtype=bool)     # any band > PHYS_MAX
        nod = np.zeros((h, w), dtype=bool)     # any band NODATA or non-finite
        raw_max = -np.inf
        for b in range(1, n_bands + 1):
            band = src.read(b).astype(np.float32)
            finite = np.isfinite(band)
            nod |= (band == NODATA) | ~finite
            over = finite & (band > PHYS_MAX) & (band != NODATA)
            bad |= over
            if over.any():
                raw_max = max(raw_max, float(band[over].max()))

    old_valid = ~nod
    affected = old_valid & bad
    n_aff = int(affected.sum())
    n_old = int(old_valid.sum())
    return {
        'tile': re.sub(r'_mrral.*', '', os.path.basename(img_path)),
        'path': img_path,
        'old_valid_px': n_old,
        'affected_px': n_aff,
        'affected_pct': 100.0 * n_aff / n_old if n_old else 0.0,
        'max_raw_value': raw_max if np.isfinite(raw_max) else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--probs_dir', default='data/mc_deploy_pyx/probs',
                    help='deployment output; its tile names define the scan set')
    ap.add_argument('--out', default='reports/phys_max_contamination.csv')
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()

    root = load_config()['data_root']
    wanted = sorted(
        os.path.basename(p).replace('_probs.npz', '')
        for p in glob.glob(os.path.join(args.probs_dir, '*', '*_probs.npz')))
    if not wanted:
        raise SystemExit(f'no *_probs.npz under {args.probs_dir}')

    paths = []
    for t in wanted:
        hits = sorted(glob.glob(os.path.join(root, 'mc*', f'{t}_mrral_*.img')))
        if not hits:
            print(f'  WARNING: no mrral source found for {t}', file=sys.stderr)
            continue
        paths.append(hits[0])
    print(f'scanning {len(paths)} of {len(wanted)} deployed tiles '
          f'(PHYS_MAX={PHYS_MAX}, NODATA={NODATA})', flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(scan_tile, p): p for p in paths}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            rows.append(r)
            flag = 'AFFECTED' if r['affected_px'] else 'clean'
            print(f"  [{i}/{len(paths)}] {r['tile']:<8} {flag:<9} "
                  f"{r['affected_px']:>9,} px ({r['affected_pct']:.3f}%)  "
                  f"raw max {r['max_raw_value']:.1f}", flush=True)

    rows.sort(key=lambda r: -r['affected_pct'])
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    aff = [r for r in rows if r['affected_px']]
    tot_aff = sum(r['affected_px'] for r in aff)
    tot_val = sum(r['old_valid_px'] for r in rows)
    print(f'\n{len(aff)} of {len(rows)} tiles affected; '
          f'{tot_aff:,} px of {tot_val:,} ({100.0 * tot_aff / tot_val:.3f}%)')
    print(f'wrote {args.out}')

    lst = os.path.splitext(args.out)[0] + '_tiles.txt'
    with open(lst, 'w') as fh:
        fh.writelines(r['path'] + '\n' for r in aff)
    print(f'wrote {lst}  ({len(aff)} paths, re-run these)')


if __name__ == '__main__':
    main()

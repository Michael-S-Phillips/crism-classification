"""Compute per-channel std constants for the dual continuum representation.

Written to data/mrral_cr_scales.json WITH provenance, so the numbers and the
thing that produced them never drift apart. Re-run only if the representation
definition changes.

    python scripts/sample_class_spectra.py         # writes the sample npz
    python scripts/compute_cr_scales.py --npz <spectra.npz>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.continuum_removal import (  # noqa: E402
    continuum_removed, linear_continuum_removed, good_band_mask_59)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'data', 'mrral_cr_scales.json')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--npz', required=True)
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    d = np.load(args.npz)
    G = good_band_mask_59()
    hull, lin, n = [], [], 0
    for k in d.files:
        if k in ('wav', 'good'):
            continue
        a = d[k].astype(np.float32).copy()
        a[(a > 1.0) | (a == 65535) | (~np.isfinite(a))] = np.nan
        a = np.clip(a, 0.0, 0.5)
        a = a[np.isfinite(a).all(axis=1)]
        if not len(a):
            continue
        n += len(a)
        hull.append(continuum_removed(a)[:, G].ravel())
        lin.append(linear_continuum_removed(a)[:, G].ravel())

    h = np.concatenate(hull)
    l = np.concatenate(lin)
    meta = {
        'hull_std': float(h.std()),
        'linear_std': float(l.std()),
        'hull_mean': float(h.mean()),
        'linear_mean': float(l.mean()),
        'n_spectra': int(n),
        'source': f'scripts/compute_cr_scales.py --npz {os.path.basename(args.npz)}',
        'computed': dt.datetime.now().strftime('%Y-%m-%d'),
        'note': ('Good bands only. hull-CR is bounded [0,1]; linear-CR is a '
                 'clipped ratio. Dividing each block by its own std here is '
                 'what makes the two blocks\' reconstruction targets '
                 'comparable; once they are, a pooled MAE loss already '
                 'weights them equally and the per-block loss is retained '
                 'only as a diagnostic.'),
    }
    with open(args.out, 'w') as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f'\nvariance ratio linear/hull = {l.std() / h.std():.2f}x')


if __name__ == '__main__':
    main()

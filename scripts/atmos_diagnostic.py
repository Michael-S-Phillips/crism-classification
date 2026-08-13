"""Is an hcp detection clinopyroxene, or residual atmospheric CO2?

No CRISM summary parameter tracks GASEOUS atmospheric CO2 -- the CO2-named ones
(BD1435, BD3200, ICER1_2, ICER2_2) are all CO2 ICE (frost/clouds), which is why
they feed the junk/ice veto elsewhere and are useless here. The physically
correct proxy is atmospheric path length, and mrrde carries it: elevation
(pressure falls roughly exponentially with elevation, so CO2 column scales
with it) and the incidence/emission angles that give the air-mass factor
1/cos(i) + 1/cos(e).

HCPINDEX2 sits in the 2 um region where volcano-scan residual leaks, so this
matters for hcp specifically. REPORTED, NOT GATED: real HCP occurs at low
elevation too, and a hard veto would suppress true detections. This module
only makes the confound visible.

Usage:
    conda run -n crism python scripts/atmos_diagnostic.py \\
        --probs /tmp/t1250_probs.npz \\
        --tile /mnt/.../t1250_mrral_20n078_0327_4.img \\
        --klass hcp --threshold 0.5
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NODATA = 65535.0

# mrrde band indices (0-based), verified against a real 19-band header:
#   6  INA at areoid - deg          11  INA at surface from MOLA - deg
#   7  EMA at areoid - deg          12  EMA at surface from MOLA - deg
#   8  Phase angle - deg            15  Elevation - meters relative to MOLA
#                                   17  Bolometic albedo
MRRDE_INA = 6
MRRDE_EMA = 7
MRRDE_ELEVATION = 15


def air_mass(ina_deg: np.ndarray, ema_deg: np.ndarray) -> np.ndarray:
    """1/cos(i) + 1/cos(e) -- the standard two-way atmospheric path-length
    factor. NaN in, NaN out (nodata angles must be masked to NaN by the
    caller before this is called, or a nodata sentinel would clip to a
    bogus-but-finite 89-degree air mass instead of dropping out)."""
    i = np.clip(np.deg2rad(ina_deg), 0, np.deg2rad(89.0))
    e = np.clip(np.deg2rad(ema_deg), 0, np.deg2rad(89.0))
    return (1.0 / np.cos(i) + 1.0 / np.cos(e)).astype(np.float32)


def detection_rate_by_decile(prob: np.ndarray, valid: np.ndarray,
                              covariate: np.ndarray, threshold: float = 0.5,
                              n: int = 10) -> list[dict]:
    """Bin pixels into `n` equal-POPULATION deciles of `covariate` (valid,
    finite pixels only) and report the class-detection rate in each.

    Equal-population (quantile) bins, not equal-width: a covariate like
    elevation is not remotely uniform across a tile, so equal-width bins
    would put almost all pixels in one or two deciles and starve the rest,
    making the reported rates noise-dominated exactly where the confound
    would need to be visible.
    """
    m = valid & np.isfinite(covariate)
    if not m.any():
        return []
    cov = covariate[m]
    det = (prob[m] >= threshold)
    edges = np.quantile(cov, np.linspace(0, 1, n + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    rows = []
    for k in range(n):
        sel = (cov >= edges[k]) & (cov < edges[k + 1])
        rows.append({'decile': k + 1,
                     'lo': float(edges[k]), 'hi': float(edges[k + 1]),
                     'n': int(sel.sum()),
                     'rate': float(det[sel].mean()) if sel.any() else 0.0})
    return rows


def derive_mrrde_path(mrral_path: str) -> str:
    """t..._mrral_..._.img -> t..._mrrde_..._.img in the same directory."""
    base = os.path.basename(mrral_path)
    return os.path.join(os.path.dirname(mrral_path),
                        base.replace('_mrral_', '_mrrde_'))


def load_mrrde_covariates(mrrde_path: str):
    """Return (elevation (H, W), air_mass (H, W)), both float32 with the
    65535 mrrde sentinel mapped to NaN in every input band BEFORE air_mass
    clips/cosines them -- clipping a raw 65535-degree sentinel to 89 degrees
    would otherwise produce a finite, bogus air mass instead of NaN."""
    import rasterio
    with rasterio.open(mrrde_path) as src:
        elev = src.read(MRRDE_ELEVATION + 1).astype(np.float32)
        ina = src.read(MRRDE_INA + 1).astype(np.float32)
        ema = src.read(MRRDE_EMA + 1).astype(np.float32)
    elev[elev == NODATA] = np.nan
    ina[ina == NODATA] = np.nan
    ema[ema == NODATA] = np.nan
    return elev, air_mass(ina, ema)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--probs', required=True, help='probs npz (any producer)')
    ap.add_argument('--tile', required=True, help='mrral .img (mrrde inferred)')
    ap.add_argument('--klass', default='hcp')
    ap.add_argument('--threshold', type=float, default=0.5)
    args = ap.parse_args()

    d = np.load(args.probs, allow_pickle=True)
    names = [str(x) for x in d['class_names']]
    if args.klass not in names:
        raise SystemExit(f'{args.klass} not in {names}')
    prob = d['probs'][:, :, names.index(args.klass)]
    valid = d['valid_mask'].astype(bool)

    mrrde_path = derive_mrrde_path(args.tile)
    if not os.path.exists(mrrde_path):
        raise SystemExit(f'no mrrde tile at {mrrde_path}')
    elev, am = load_mrrde_covariates(mrrde_path)

    # mrrde is co-registered with mrral by convention, never by guarantee --
    # assert it rather than silently broadcasting or truncating.
    if elev.shape != valid.shape:
        raise SystemExit(
            f'mrrde {elev.shape} is not co-registered with the probs npz '
            f'valid_mask {valid.shape}; refusing to guess an alignment')

    for label, cov in (('elevation (m)', elev), ('air mass', am)):
        print(f'\n{args.klass} detection rate by {label} decile '
              f'(threshold {args.threshold}):')
        print(f'  {"dec":>4}{"lo":>12}{"hi":>12}{"n":>10}{"rate":>8}')
        for r in detection_rate_by_decile(prob, valid, cov, args.threshold):
            print(f'  {r["decile"]:>4}{r["lo"]:>12.1f}{r["hi"]:>12.1f}'
                  f'{r["n"]:>10,}{r["rate"]:>8.3f}')
    print('\nReported, not gated: this does not correct or veto anything. '
          'Detections concentrated in the LOW-elevation / HIGH-air-mass '
          'deciles indicate residual atmospheric CO2 rather than '
          'clinopyroxene; real HCP occurs at low elevation too.')


if __name__ == '__main__':
    main()

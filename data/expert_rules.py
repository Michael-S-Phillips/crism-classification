"""Expert band-parameter rules over the CRISM summary parameters.

Structure is fixed by mineralogy and never fitted; only the cut points are
calibrated (scripts/fit_expert_rules.py). Two kinds of gate:

  VETO (hard)      artifacts and genuinely incompatible conditions only --
                   ice, saturation, non-physical values, dust for plagioclase.
  DOMINANCE (soft) cross-responding index pairs raise the TIER of the dominant
                   label without zeroing the other.

The vocabulary is MULTI-LABEL: a pixel can be olivine AND hcp. Exclusive gates
would fight the label structure and silently suppress real assemblages, so there
are none between co-occurring minerals.

Parameters are resolved through the band-name list that came from the tile's own
header (data.mrrsu_bands), never by hardcoded position.
"""
from __future__ import annotations

import numpy as np

from data.mrrsu_bands import band_index

CLASSES_7 = ['olivine', 'lcp', 'hcp', 'plagioclase', 'bland', 'alteration', 'junk']
CLASSES_PYX = ['olivine', 'pyx', 'plagioclase', 'bland', 'alteration', 'junk']

# Residual / derived classes: not scored from their own rule block.
DERIVED = ('bland', 'junk')

DEFAULT_RULES = {
    'vocab': CLASSES_7,
    'junk': {'icer_high': None, 'var_high': None, 'r770_max': None},
    'classes': {
        'olivine':     {'primary': {'param': 'OLINDEX3',  'threshold': None},
                        'ladder': None},
        'lcp':         {'primary': {'param': 'LCPINDEX2', 'threshold': None},
                        'dominance_over': 'HCPINDEX2', 'ladder': None},
        'hcp':         {'primary': {'param': 'HCPINDEX2', 'threshold': None},
                        'dominance_over': 'LCPINDEX2', 'ladder': None},
        'plagioclase': {'primary': {'param': 'BD1300', 'threshold': None},
                        'rpeak1_window': None, 'hydration_veto': None,
                        'ladder': None},
        'alteration':  {'groups': [
                            {'name': 'femg_phyllosilicate',
                             'requires': ['D2300', 'BD2290', 'BD1900R2'],
                             'thresholds': None},
                            {'name': 'al_phyllosilicate',
                             'requires': ['BD2210_2', 'BD1900R2'],
                             'thresholds': None},
                            {'name': 'hydrated_silica',
                             'requires': ['MIN2200', 'BD1900R2'],
                             'thresholds': None},
                            {'name': 'sulfate',
                             'requires': ['SINDEX2', 'BD1900R2'],
                             'thresholds': None},
                            # Anhydrous: NO BD1900R2. Requiring hydration here
                            # would silently reject Mg-carbonate.
                            {'name': 'carbonate',
                             'requires': ['BD2500_2', 'D2300'],
                             'thresholds': None},
                        ], 'ladder': None},
    },
}


def _p(cube: np.ndarray, names: list[str], param: str) -> np.ndarray:
    return cube[..., band_index(names, param)]


def _tier_score(value: np.ndarray, fires: np.ndarray, ladder) -> np.ndarray:
    """Map a firing pixel to the precision of the highest rung it clears."""
    out = np.zeros(value.shape, dtype=np.float32)
    for thresh, precision in sorted(ladder, key=lambda r: r[0]):
        out = np.where(fires & (value >= thresh), np.float32(precision), out)
    return out


def evaluate_rules(cube: np.ndarray, names: list[str], config: dict
                   ) -> dict[str, np.ndarray]:
    """Score every class of `config['vocab']` on an (H, W, 60) mrrsu cube.

    Returns {class: (H, W) float32 in [0, 1]}, keyed in vocab order. Pure: the
    config is read, never modified.
    """
    cfg = config['classes']
    shape = cube.shape[:2]
    finite = np.isfinite(cube).any(axis=-1)

    # ── junk: artifacts and ice ──────────────────────────────────────────────
    jc = config['junk']
    icer = np.fmax(np.nan_to_num(_p(cube, names, 'ICER1_2'), nan=0.0),
                   np.nan_to_num(_p(cube, names, 'ICER2_2'), nan=0.0))
    co2_ice = np.fmax(np.nan_to_num(_p(cube, names, 'BD1435'), nan=0.0),
                      np.nan_to_num(_p(cube, names, 'BD3200'), nan=0.0))
    r770 = _p(cube, names, 'R770')
    var = np.nan_to_num(_p(cube, names, 'VAR'), nan=0.0)
    is_junk = ((icer >= jc['icer_high']) | (co2_ice >= jc['icer_high'])
               | (np.nan_to_num(r770, nan=0.0) > jc['r770_max'])
               | (var >= jc['var_high'])) & finite
    ok = finite & ~is_junk

    out: dict[str, np.ndarray] = {}

    for cls, c in cfg.items():
        if cls in DERIVED:
            continue

        # ── alteration: disjunction of specific groups, ice vetoed ───────────
        if 'groups' in c:
            any_group = np.zeros(shape, dtype=bool)
            strength = np.zeros(shape, dtype=np.float32)
            for g in c['groups']:
                hit = ok.copy()
                gmin = np.full(shape, np.inf, dtype=np.float32)
                for param in g['requires']:
                    v = np.nan_to_num(_p(cube, names, param), nan=-np.inf)
                    hit &= v >= g['thresholds'][param]
                    gmin = np.minimum(gmin, v)
                any_group |= hit
                strength = np.where(hit, np.maximum(strength, gmin), strength)
            out[cls] = _tier_score(strength, any_group,
                                   c['ladder']).astype(np.float32)
            continue

        v = np.nan_to_num(_p(cube, names, c['primary']['param']), nan=-np.inf)
        fires = ok & (v >= c['primary']['threshold'])

        # ── plagioclase: RPEAK1 window + BD1300, dust vetoed ─────────────────
        if 'rpeak1_window' in c:
            # RPEAK1 is a WAVELENGTH in um (~0.7-0.8 for plagioclase), so the
            # test is a two-sided window: "high" would admit everything above.
            lo, hi = c['rpeak1_window']
            rp = _p(cube, names, 'RPEAK1')
            hyd = np.nan_to_num(_p(cube, names, 'BD1900R2'), nan=0.0)
            in_window = np.isfinite(rp) & (rp >= lo) & (rp <= hi)
            fires = fires & in_window & (hyd < c['hydration_veto'])

        # ── mafic minerals: own evidence only, no exclusivity ────────────────
        score = _tier_score(v, fires, c['ladder'])
        # Dominance raises the tier; it never gates. A basalt with both LCP and
        # HCP must receive both labels.
        dom = c.get('dominance_over')
        if dom is not None:
            other = np.nan_to_num(_p(cube, names, dom), nan=-np.inf)
            score = np.where(fires & (v > other), score,
                             np.where(fires, score * np.float32(0.5), score))
        out[cls] = score.astype(np.float32)

    out['junk'] = is_junk.astype(np.float32)

    mineral_keys = [k for k in out if k not in DERIVED]
    any_mineral = np.zeros(shape, dtype=bool)
    for k in mineral_keys:
        any_mineral |= out[k] > 0
    out['bland'] = (ok & ~any_mineral).astype(np.float32)

    missing = [c for c in config['vocab'] if c not in out]
    if missing:
        raise KeyError(
            f'{missing} in vocab but has no rule block in config["classes"]')
    return {k: out[k] for k in config['vocab']}

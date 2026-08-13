"""Expert band-parameter rules over the CRISM summary parameters.

Structure is fixed by mineralogy and never fitted; only the cut points are
calibrated (scripts/fit_expert_rules.py). Two kinds of gate:

  VETO (hard)      artifacts and genuinely incompatible conditions only --
                   ice, saturation, non-physical values, dust for plagioclase.
  DOMINANCE (soft) cross-responding index pairs raise the TIER of the dominant
                   label without zeroing the other. The demotion is one RUNG
                   STEP down the loser's own ladder, floored at the lowest
                   rung -- never a multiplicative factor. A factor is a
                   de-facto exclusive gate at the contract boundary: the
                   vectorizer selects with `>= 0.50`, so halving a ladder
                   precision (<= 1.0) drops every non-dominant pixel out of
                   EVERY polygon layer, which is the exclusivity this design
                   exists to avoid, merely relocated downstream.

The vocabulary is MULTI-LABEL: a pixel can be olivine AND hcp. Exclusive gates
would fight the label structure and silently suppress real assemblages, so there
are none between co-occurring minerals.

Parameters are resolved through the band-name list that came from the tile's own
header (data.mrrsu_bands), never by hardcoded position.

An uncalibrated cut point (`None`) is read as +inf, i.e. a comparison that can
never trigger: a `None` veto is inactive and a `None` detection threshold never
fires. Task 4 writes `float('inf')` for a veto it cannot calibrate and leaves a
class with no training positives at `None`, so both mean the same thing here.
"""
from __future__ import annotations

import functools

import numpy as np

from data.mrrsu_bands import band_index

CLASSES_7 = ['olivine', 'lcp', 'hcp', 'plagioclase', 'bland', 'alteration', 'junk']
CLASSES_PYX = ['olivine', 'pyx', 'plagioclase', 'bland', 'alteration', 'junk']

# Residual / derived classes: not scored from their own rule block.
DERIVED = ('bland', 'junk')

DEFAULT_RULES = {
    'vocab': CLASSES_7,
    # ICER1_2/ICER2_2 are ice-abundance RATIOS; BD1435/BD3200 are BAND DEPTHS.
    # Different scales, so they get separate cut points -- one threshold shared
    # between them would either never fire on CO2 ice or fire everywhere.
    'junk': {'icer_high': None, 'co2_ice_high': None, 'var_high': None,
             'r770_max': None},
    'classes': {
        'olivine':     {'primary': {'param': 'OLINDEX3',  'threshold': None},
                        'ladder': None},
        'lcp':         {'primary': {'param': 'LCPINDEX2', 'threshold': None},
                        'dominance_over': 'HCPINDEX2', 'ladder': None},
        'hcp':         {'primary': {'param': 'HCPINDEX2', 'threshold': None},
                        'dominance_over': 'LCPINDEX2', 'ladder': None},
        # Merged pyroxene (CLASSES_PYX). The merge exists BECAUSE LCPINDEX2 and
        # HCPINDEX2 cross-respond and the 2 um band-centre discrimination is
        # unreliable, so there is deliberately NO dominance term here: nothing
        # is left to be dominant over. Evidence is the elementwise max of the
        # two indices, i.e. the stronger pyroxene response either way.
        'pyx':         {'primary': {'param': ['LCPINDEX2', 'HCPINDEX2'],
                                    'reduce': 'max', 'threshold': None},
                        'ladder': None},
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


_REDUCERS = {'max': np.maximum, 'min': np.minimum}


def _p(cube: np.ndarray, names: list[str], param: str) -> np.ndarray:
    return cube[..., band_index(names, param)]


def _cut(threshold) -> float:
    """An uncalibrated (`None`) cut point is +inf: the test never triggers."""
    return np.inf if threshold is None else threshold


def _primary_value(cube: np.ndarray, names: list[str], primary: dict
                   ) -> np.ndarray:
    """Evidence array for a rule's primary term, NaN mapped to -inf.

    `param` is a parameter NAME, or a list of names plus a `reduce` ('max' or
    'min') combining them elementwise. The scalar-string form is unchanged.
    """
    param = primary['param']
    if isinstance(param, str):
        return np.nan_to_num(_p(cube, names, param), nan=-np.inf)
    how = primary.get('reduce', 'max')
    if how not in _REDUCERS:
        raise ValueError(f'unknown reduce {how!r}; expected one of '
                         f'{sorted(_REDUCERS)}')
    vals = [np.nan_to_num(_p(cube, names, p), nan=-np.inf) for p in param]
    return functools.reduce(_REDUCERS[how], vals)


def _tier_score(value: np.ndarray, fires: np.ndarray, ladder) -> np.ndarray:
    """Map a firing pixel to the precision of the highest rung it clears."""
    out = np.zeros(value.shape, dtype=np.float32)
    for thresh, precision in sorted(ladder or [], key=lambda r: r[0]):
        out = np.where(fires & (value >= thresh), np.float32(precision), out)
    return out


def _demote_one_rung(score: np.ndarray, ladder) -> np.ndarray:
    """Step a tier score down to the next-LOWER rung of its own ladder.

    Floored at the lowest rung: a firing label must keep a real ladder
    precision, because the vectorizer selects polygons with `>= 0.50` and a
    score pushed below the lowest rung would vanish from every threshold layer
    instead of merely ranking second.

    Distinct precisions only: adjacent rungs can share a precision, and
    stepping by rung INDEX would then be a no-op that leaves the demoted label
    tied with the dominant one.
    """
    precisions = sorted({float(p) for _t, p in (ladder or [])})
    if not precisions:
        return score
    out = score.copy()
    for i, p in enumerate(precisions):
        # Compare against the ORIGINAL score, never the partially rewritten
        # `out`, so a demotion cannot cascade down several rungs at once.
        out = np.where(score == np.float32(p),
                       np.float32(precisions[max(i - 1, 0)]), out)
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
    is_junk = ((icer >= _cut(jc['icer_high']))
               | (co2_ice >= _cut(jc['co2_ice_high']))
               | (np.nan_to_num(r770, nan=0.0) > _cut(jc['r770_max']))
               | (var >= _cut(jc['var_high']))) & finite
    ok = finite & ~is_junk

    out: dict[str, np.ndarray] = {}
    vocab = config['vocab']

    for cls, c in cfg.items():
        # Rule blocks outside this config's vocabulary are not scored: with the
        # pyx vocabulary the lcp/hcp blocks are inert, and vice versa.
        if cls in DERIVED or cls not in vocab:
            continue

        # ── alteration: disjunction of specific groups, ice vetoed ───────────
        if 'groups' in c:
            any_group = np.zeros(shape, dtype=bool)
            strength = np.zeros(shape, dtype=np.float32)
            for g in c['groups']:
                th = g['thresholds']
                hit = ok.copy()
                gmin = np.full(shape, np.inf, dtype=np.float32)
                for param in g['requires']:
                    v = np.nan_to_num(_p(cube, names, param), nan=-np.inf)
                    hit &= v >= _cut(th[param] if th is not None else None)
                    gmin = np.minimum(gmin, v)
                any_group |= hit
                strength = np.where(hit, np.maximum(strength, gmin), strength)
            out[cls] = _tier_score(strength, any_group,
                                   c['ladder']).astype(np.float32)
            continue

        v = _primary_value(cube, names, c['primary'])
        fires = ok & (v >= _cut(c['primary']['threshold']))

        # ── plagioclase: RPEAK1 window + BD1300, dust vetoed ─────────────────
        if 'rpeak1_window' in c:
            # RPEAK1 is a WAVELENGTH in um (~0.7-0.8 for plagioclase), so the
            # test is a two-sided window: "high" would admit everything above.
            lo, hi = c['rpeak1_window'] if c['rpeak1_window'] else (-np.inf,
                                                                   np.inf)
            rp = _p(cube, names, 'RPEAK1')
            hyd = np.nan_to_num(_p(cube, names, 'BD1900R2'), nan=0.0)
            in_window = np.isfinite(rp) & (rp >= lo) & (rp <= hi)
            fires = fires & in_window & (hyd < _cut(c['hydration_veto']))

        # ── mafic minerals: own evidence only, no exclusivity ────────────────
        score = _tier_score(v, fires, c['ladder'])
        # Dominance raises the tier; it never gates. A basalt with both LCP and
        # HCP must receive both labels -- and both must survive the vectorizer's
        # lowest threshold, so the loser steps ONE rung down its own ladder and
        # no further (see _demote_one_rung).
        dom = c.get('dominance_over')
        if dom is not None:
            other = np.nan_to_num(_p(cube, names, dom), nan=-np.inf)
            demoted = _demote_one_rung(score, c['ladder'])
            score = np.where(fires & (v <= other), demoted, score)
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

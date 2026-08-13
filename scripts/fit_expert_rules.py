"""Calibrate the expert ruleset on the TRAIN split.

Expert structure, data-fitted cut points. The logical form is fixed by
mineralogy in ``data/expert_rules.py`` and is NEVER searched over, so this
cannot overfit into an uninterpretable rule; only the thresholds move.

Two ideas carry the whole file:

RETENTION. Every cut point is placed so that a stated fraction of that class's
own training positives survives it (default 90%). A veto that rejects pixels
more artifact-like than 90% of real olivine is self-calibrating, interpretable,
and structurally unable to silently annihilate the class. The retention floor
is GUARANTEED, not approximated: `_veto_threshold` and `_detect_threshold`
return order statistics chosen so the surviving fraction is at least
`retention` exactly, ties included.

PRECISION LADDERS. Each rung carries the empirical precision of the rule at
that strictness, measured on the training positives with the calibrated vetoes
already applied -- so a ladder position means "this rule at this strictness is
right p% of the time", which is the same axis as the deep model's activation
and is what makes the two comparable at 0.5 / 0.9 / 0.99. Precision should be
non-decreasing along the ladder; a violation means the index is badly behaved
there and is REPORTED (printed and recorded in `config['calibration']
['warnings']`), never smoothed over.

The emitted JSON is meant to be audited by a domain reader: parameters are
named (OLINDEX3, BD1300, ...) because the Task-2 sidecar carries real parameter
names, and a `calibration` block records the row counts, the positives per
class and the retention each cut point actually achieved.

Usage
-----
    python scripts/fit_expert_rules.py \
        --features data/mrrsu_features.parquet \
        --labels   data/mrral_pixels_with_review_v2.parquet \
        --vocab 7cls --out config/expert_rules_7cls.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.expert_rules import (CLASSES_7, CLASSES_PYX, DEFAULT_RULES,  # noqa: E402
                               DERIVED, _cut)

# Strictness levels of the precision ladder, as quantiles of the evidence
# among the pixels the rule already fires on.
LADDER_QUANTILES = [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]

# Vetoes: (config key, parameter names combined the way evaluate_rules combines
# them). ICER1_2/ICER2_2 are ice-abundance RATIOS and BD1435/BD3200 are BAND
# DEPTHS, so they are calibrated separately -- one cut point cannot serve both
# scales.
VETOES = {
    'icer_high':    ['ICER1_2', 'ICER2_2'],
    'co2_ice_high': ['BD1435', 'BD3200'],
    'var_high':     ['VAR'],
    'r770_max':     ['R770'],
}

# Above this the junk vetoes are eating the training set rather than trimming
# artifacts from it, which is the failure mode that produces an empty map from
# a config that reads fine.
JUNK_REJECT_ALARM = 0.40


# ─────────────────────────────────────────────────────────────────────────────
# column access, mirroring the engine's NaN handling
# ─────────────────────────────────────────────────────────────────────────────

def _evidence(feat: pd.DataFrame, param: str) -> np.ndarray:
    """A parameter column with NaN mapped to -inf, as ``evaluate_rules`` does."""
    if param not in feat.columns:
        return np.full(len(feat), -np.inf)
    return np.nan_to_num(feat[param].to_numpy(dtype=float), nan=-np.inf)


def _veto_value(feat: pd.DataFrame, params: list[str]) -> np.ndarray | None:
    """The elementwise max of a veto's parameters, NaN as 0 (engine semantics).

    None when the sidecar carries none of them, which leaves the veto inactive
    rather than calibrated against nothing.
    """
    present = [p for p in params if p in feat.columns]
    if not present:
        return None
    out = np.zeros(len(feat))
    for p in present:
        out = np.fmax(out, np.nan_to_num(feat[p].to_numpy(dtype=float), nan=0.0))
    return out


def _primary_evidence(feat: pd.DataFrame, primary: dict) -> np.ndarray:
    """Evidence for a rule's primary term.

    `param` is a parameter NAME, or a LIST of names plus a `reduce` key ('max'
    or 'min') combining them elementwise -- the shape the merged `pyx` class
    uses. Reducing with the same rule the engine uses is the point: calibrating
    a pooled two-column quantile instead would place the threshold on a
    distribution the engine never evaluates.
    """
    param = primary['param']
    if isinstance(param, str):
        return _evidence(feat, param)
    how = primary.get('reduce', 'max')
    reducer = {'max': np.maximum, 'min': np.minimum}.get(how)
    if reducer is None:
        raise ValueError(f'unknown reduce {how!r}; expected "max" or "min"')
    out = _evidence(feat, param[0])
    for p in param[1:]:
        out = reducer(out, _evidence(feat, p))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# threshold placement
# ─────────────────────────────────────────────────────────────────────────────

def _order_stat_index(n: int, retention: float) -> int:
    """How many of `n` positives must survive to meet the retention floor."""
    return int(min(n, max(1, np.ceil(retention * n))))


def _veto_threshold(values: np.ndarray, retention: float) -> float:
    """Place a veto so at least `retention` of these positives survive it.

    The engine vetoes on ``value >= threshold``, so surviving means STRICTLY
    below. Returning a plain ``quantile(values, retention)`` gets that right
    only up to interpolation and ties, so the threshold is taken one ULP above
    the k-th smallest positive instead, with k = ceil(retention * n): the
    tightest cut point for which the floor provably holds.

    Getting the direction backwards -- ``ceil((1 - retention) * n)`` -- looks
    identical in the config and destroys the class.
    """
    v = values[np.isfinite(values)]
    if v.size == 0:
        return float('inf')          # nothing to protect: veto stays inactive
    return _veto_threshold_sorted(np.sort(v), retention)


def _veto_threshold_sorted(v: np.ndarray, retention: float) -> float:
    """`_veto_threshold` on an already-sorted, already-finite array.

    The step above the k-th positive is taken in FLOAT32, not float64. mrrsu
    cubes are float32, and numpy compares a float32 array against a Python
    float by demoting the scalar -- so a float64-sized step is rounded straight
    back onto the data value at evaluation time and the k-th positive gets
    vetoed after all. That costs exactly the one pixel the floor promised, and
    only ever shows up as a retention of 0.89.
    """
    if v.size == 0:
        return float('inf')
    k = _order_stat_index(v.size, retention)
    return float(np.nextafter(np.float32(v[k - 1]), np.float32(np.inf)))


def _detect_threshold(values: np.ndarray, retention: float) -> float | None:
    """Place a detection threshold so at least `retention` of these positives
    fire. The engine fires on ``value >= threshold``, so this is the k-th
    LARGEST positive with k = ceil(retention * n)."""
    v = np.sort(values[np.isfinite(values)])
    if v.size == 0:
        return None                  # no usable positives: the class stays inert
    k = _order_stat_index(v.size, retention)
    return float(v[v.size - k])


def _two_sided_window(values: np.ndarray, retention: float
                      ) -> list[float] | None:
    """A window keeping at least `retention` of these positives, trimmed evenly
    from both tails. RPEAK1 is a WAVELENGTH, so "high" is meaningless for it."""
    v = np.sort(values[np.isfinite(values)])
    if v.size == 0:
        return None
    drop = v.size - _order_stat_index(v.size, retention)
    lo_i = drop // 2
    hi_i = v.size - 1 - (drop - drop // 2)
    return [float(v[lo_i]), float(v[max(hi_i, lo_i)])]


def precision_ladder(score, y, thresholds, fires=None) -> list[list[float]]:
    """[(threshold, precision), ...] — precision of the rule at each strictness.

    `fires` restricts the measurement to the pixels the rule actually fires on
    (threshold plus vetoes plus any class-specific gates), so the number that
    comes out is the precision the engine will achieve at that rung and not the
    precision of an unguarded comparison.
    """
    score = np.asarray(score, dtype=float)
    y = np.asarray(y).astype(bool)
    base = np.ones(score.shape, dtype=bool) if fires is None else np.asarray(fires, dtype=bool)
    out = []
    for t in thresholds:
        sel = base & (score >= t)
        n = int(sel.sum())
        prec = float(y[sel].mean()) if n else 0.0
        out.append([float(t), round(prec, 4)])
    return out


def _round_down(x: float, nd: int = 6) -> float:
    """Shorten a cut point without ever making it stricter."""
    scale = 10.0 ** nd
    return float(np.floor(float(x) * scale) / scale)


def _rungs(score: np.ndarray, fires: np.ndarray, base: float) -> list[float]:
    """Ladder cut points: the base (the rule's own firing threshold) plus
    quantiles of the evidence among firing pixels.

    The base is included deliberately. ``_tier_score`` gives 0.0 to a firing
    pixel that clears no rung, and the engine reads a 0.0 as "no mineral" and
    reassigns the pixel to bland -- so a lowest rung above the firing threshold
    punches a silent hole in the detection.
    """
    vals = score[fires & np.isfinite(score)]
    rungs = {float(base)}
    if vals.size:
        # Higher rungs are shortened for readability by rounding DOWN rather
        # than to nearest, so a rung can only ever become more permissive
        # (defensive; nothing depends on it). What the no-gap guarantee DOES
        # rest on is the lowest rung being left exactly at the firing
        # threshold, unrounded -- rounding it up by even 1e-7 would leave the
        # pixels between the two firing with a score of 0.0.
        for q in LADDER_QUANTILES:
            r = _round_down(float(np.quantile(vals, q)))
            if np.isfinite(r) and r > base:
                rungs.add(r)
    return sorted(rungs)


# ─────────────────────────────────────────────────────────────────────────────
# alteration groups, mirroring the engine's disjunction-of-conjunctions
# ─────────────────────────────────────────────────────────────────────────────

def _alteration_evidence(feat: pd.DataFrame, groups: list[dict]
                         ) -> tuple[np.ndarray, np.ndarray]:
    """(fires, strength) exactly as ``evaluate_rules`` computes them: a group
    fires when ALL its required parameters clear their cut points, its strength
    is its WEAKEST member, and the class takes the max over firing groups."""
    n = len(feat)
    fires = np.zeros(n, dtype=bool)
    strength = np.zeros(n, dtype=float)
    for g in groups:
        th = g.get('thresholds')
        hit = np.ones(n, dtype=bool)
        gmin = np.full(n, np.inf)
        for p in g['requires']:
            v = _evidence(feat, p)
            hit &= v >= _cut(th.get(p) if th else None)
            gmin = np.minimum(gmin, v)
        fires |= hit
        strength = np.where(hit, np.maximum(strength, gmin), strength)
    return fires, strength


def _fit_junk_vetoes(feat: pd.DataFrame, protect: np.ndarray, retention: float
                     ) -> tuple[dict, float, float]:
    """Calibrate all four junk cut points against a JOINT retention floor.

    The four vetoes are one gate in the engine -- a single disjunction -- so
    placing each of them independently at the 90th percentile of the protected
    positives lets them compound: four vetoes that each keep 90% of real
    mineral pixels can jointly keep 0.9^4 = 66% of them, which is precisely the
    quiet annihilation the retention floor exists to prevent, only spread
    across four numbers that each look defensible on its own.

    So a single per-veto level `r >= retention` is shared by all four and
    raised (by bisection, deterministic) until the COMBINED junk mask retains
    `retention` of the protected positives. With one active veto this reduces
    exactly to the per-veto rule.

    Returns (thresholds, per_veto_retention, achieved_joint_retention).
    """
    values = {name: _veto_value(feat, params)
              for name, params in VETOES.items()}
    active = {n: v[protect] for n, v in values.items() if v is not None}
    sorted_pos = {n: np.sort(v[np.isfinite(v)]) for n, v in active.items()}

    def build(r: float) -> dict:
        return {n: (_veto_threshold_sorted(s, r) if s.size else float('inf'))
                for n, s in sorted_pos.items()}

    def achieved(th: dict) -> float:
        if not active:
            return 1.0
        vetoed = np.zeros(int(protect.sum()), dtype=bool)
        for n, v in active.items():
            vetoed |= (v > th[n]) if n == 'r770_max' else (v >= th[n])
        return float((~vetoed).mean())

    r = retention
    th = build(r)
    if achieved(th) < retention:
        lo, hi = retention, 1.0      # r = 1.0 vetoes nothing, so hi always works
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if achieved(build(mid)) >= retention:
                hi = mid
            else:
                lo = mid
        r, th = hi, build(hi)
    out = {n: th.get(n, float('inf')) for n in VETOES}
    return out, float(r), achieved(th)


def _class_gate_mask(feat: pd.DataFrame, c: dict, score: np.ndarray,
                     gates: dict) -> np.ndarray:
    """The class's own gates, exactly as ``evaluate_rules`` applies them (the
    junk disjunction is applied separately by the caller)."""
    mask = score >= _cut(gates['threshold'])
    if 'rpeak1_window' in c:
        lo, hi = gates['rpeak1_window'] or (-np.inf, np.inf)
        rp = _evidence(feat, 'RPEAK1')
        hyd = (np.nan_to_num(feat['BD1900R2'].to_numpy(dtype=float), nan=0.0)
               if 'BD1900R2' in feat.columns else np.zeros(len(feat)))
        mask = mask & (rp >= lo) & (rp <= hi) & (hyd < _cut(
            gates['hydration_veto']))
    return mask


def _fit_class_gates(feat: pd.DataFrame, c: dict, score: np.ndarray,
                     pos: np.ndarray, retention: float) -> tuple[dict, float]:
    """Calibrate a class's own cut points against a JOINT retention floor.

    Most classes have one gate and this reduces to placing it at the retention
    quantile. Plagioclase has three -- the BD1300 threshold, the two-sided
    RPEAK1 window (RPEAK1 is a WAVELENGTH, so "high" is meaningless for it) and
    the BD1900R2 dust veto -- and the engine ANDs all three. Placed
    independently at 90% they compound to 0.9^3 = 73%, so the shared per-gate
    level is raised by bisection until the CONJUNCTION keeps `retention` of the
    plagioclase positives. Plagioclase over-firing on bright featureless ground
    is a live failure mode here, so the looser gate is deliberately paired with
    the precision ladder: an admitted marginal pixel lands on a low rung rather
    than being deleted outright.
    """
    def build(r: float) -> dict:
        g = {'threshold': _detect_threshold(score[pos], r)}
        if 'rpeak1_window' in c:
            g['rpeak1_window'] = _two_sided_window(
                _evidence(feat, 'RPEAK1')[pos], r)
            g['hydration_veto'] = _veto_threshold(
                _evidence(feat, 'BD1900R2')[pos], r)
        return g

    def achieved(g: dict) -> float:
        m = _class_gate_mask(feat, c, score, g)[pos]
        return float(m.mean()) if m.size else 0.0

    r = retention
    gates = build(r)
    if achieved(gates) < retention:
        lo, hi = retention, 1.0      # r = 1.0 admits everything, so hi works
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if achieved(build(mid)) >= retention:
                hi = mid
            else:
                lo = mid
        r, gates = hi, build(hi)
    return gates, float(r)


def _junk_mask(feat: pd.DataFrame, junk_cfg: dict) -> np.ndarray:
    """The engine's junk test, evaluated on the training rows."""
    n = len(feat)
    mask = np.zeros(n, dtype=bool)
    for name, params in VETOES.items():
        value = _veto_value(feat, params)
        if value is None:
            continue
        cut = _cut(junk_cfg.get(name))
        # r770_max is a strict `>` in the engine; the rest are `>=`.
        mask |= (value > cut) if name == 'r770_max' else (value >= cut)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate(feat: pd.DataFrame, labels: dict[str, np.ndarray],
              vocab: list[str], retention: float = 0.90) -> dict:
    """Fit every cut point of the expert ruleset to `feat`/`labels`.

    Parameters
    ----------
    feat : one row per labeled pixel, columns named by the REAL mrrsu
        parameter name (OLINDEX3, BD1300, ...). Only the rows passed in are
        read; the caller is responsible for having selected the train split.
    labels : class name -> 0/1 array aligned row-for-row with `feat`.
    vocab : the class list to emit. Rule blocks outside it are dropped so an
        auditor never reads a threshold that is never applied.
    retention : fraction of a class's own positives every cut point must keep.
    """
    if not 0.0 < retention < 1.0:
        raise ValueError(f'retention must be in (0, 1), got {retention}')

    feat = feat.reset_index(drop=True)
    num = feat.select_dtypes(include=[np.number])
    if len(num.columns):
        # Rows with no mrrsu coverage at all are all-NaN (Task 2) and carry no
        # information; they would only distort the audit counts.
        keep = np.isfinite(num.to_numpy(dtype=float)).any(axis=1)
    else:
        keep = np.ones(len(feat), dtype=bool)
    n_dropped = int((~keep).sum())
    feat = feat.loc[keep].reset_index(drop=True)
    labels = {c: np.asarray(y).astype(float)[keep] for c, y in labels.items()}

    cfg = copy.deepcopy(DEFAULT_RULES)
    cfg['vocab'] = list(vocab)
    cfg['retention'] = float(retention)
    cfg['classes'] = {c: b for c, b in cfg['classes'].items() if c in vocab}

    warnings: list[str] = []
    audit: dict = {'retention': float(retention), 'split': 'train',
                   'n_rows': int(len(feat)),
                   'n_rows_dropped_all_nan': n_dropped,
                   'junk': {}, 'classes': {}, 'warnings': warnings}

    # ── vetoes ───────────────────────────────────────────────────────────────
    # Placed to retain the positives of the MINERAL classes. Junk-labeled
    # pixels are excluded: they are precisely what the vetoes exist to reject,
    # and counting them as pixels to retain drags each cut point past the
    # artifact it is meant to catch.
    protect = np.zeros(len(feat), dtype=bool)
    for cls, y in labels.items():
        if cls != 'junk':
            protect |= y.astype(bool)
    if protect.any():
        cuts, per_veto, joint = _fit_junk_vetoes(feat, protect, retention)
    else:
        cuts = {n: float('inf') for n in VETOES}
        per_veto, joint = 1.0, 1.0
        warnings.append('junk: no positives to protect — all vetoes inactive')
    cfg['junk'] = cuts
    audit['junk_per_veto_retention'] = round(per_veto, 6)
    audit['junk_joint_retention'] = round(joint, 4)
    for name, params in VETOES.items():
        value = _veto_value(feat, params)
        cut = cuts[name]
        if value is None:
            audit['junk'][name] = {'params': params, 'status': 'inactive',
                                   'reason': 'parameter absent from the sidecar'}
            warnings.append(f'junk/{name}: parameter absent — left inactive (+inf)')
            continue
        rejected = (value > cut) if name == 'r770_max' else (value >= cut)
        audit['junk'][name] = {
            'params': params,
            'reduce': 'max' if len(params) > 1 else None,
            'threshold': cut,
            'n_protected': int(protect.sum()),
            'retained': round(float((~rejected[protect]).mean()), 4)
            if protect.any() else None,
            'rejects_frac_of_train': round(float(rejected.mean()), 4),
        }

    junk = _junk_mask(feat, cfg['junk'])
    ok = ~junk
    audit['junk_rejects_frac'] = round(float(junk.mean()), 4)
    if junk.mean() > JUNK_REJECT_ALARM:
        warnings.append(
            f'junk vetoes reject {junk.mean():.1%} of the TRAIN rows — the '
            f'cut points are eating the data, not trimming artifacts')

    # ── per-class cut points and ladders ─────────────────────────────────────
    for cls in vocab:
        if cls in DERIVED or cls not in cfg['classes']:
            continue
        c = cfg['classes'][cls]
        y = labels.get(cls)
        n_pos = int(np.nansum(y)) if y is not None else 0
        entry: dict = {'n_positives': n_pos}
        audit['classes'][cls] = entry
        if y is None or n_pos == 0:
            # No positives: leave every cut point at None, which the engine
            # reads as +inf, i.e. the class never fires. Falling back to a
            # quantile of the whole column would paint it everywhere.
            entry['status'] = 'inert (no training positives)'
            warnings.append(f'{cls}: no training positives — left inert')
            continue
        pos = y.astype(bool)

        if 'groups' in c:
            for g in c['groups']:
                g['thresholds'] = {}
                for p in g['requires']:
                    v = _evidence(feat, p)[pos]
                    t = _detect_threshold(v, retention)
                    g['thresholds'][p] = t
                    if t is None:
                        warnings.append(
                            f'{cls}/{g["name"]}: {p} absent or all-NaN over the '
                            f'positives — the group is inert')
            fires_raw, score = _alteration_evidence(feat, c['groups'])
            fires = fires_raw & ok
            base = float(score[fires].min()) if fires.any() else 0.0
            entry['groups'] = {g['name']: dict(g['thresholds'])
                               for g in c['groups']}
        else:
            score = _primary_evidence(feat, c['primary'])
            gates, per_gate = _fit_class_gates(feat, c, score, pos, retention)
            c['primary']['threshold'] = gates['threshold']
            if 'rpeak1_window' in c:
                c['rpeak1_window'] = gates['rpeak1_window']
                c['hydration_veto'] = gates['hydration_veto']
                entry['rpeak1_window'] = gates['rpeak1_window']
                entry['hydration_veto'] = gates['hydration_veto']
                entry['per_gate_retention'] = round(per_gate, 6)
            fires = ok & _class_gate_mask(feat, c, score, gates)
            base = (float(gates['threshold'])
                    if gates['threshold'] is not None else 0.0)
            entry['param'] = c['primary']['param']
            entry['threshold'] = gates['threshold']

        # Two floors are guaranteed separately -- the class's own gates keep
        # `retention` of its positives, and the junk disjunction keeps
        # `retention` of all mineral positives pooled -- so a class survives
        # both at no worse than the union bound. Falling below THAT means the
        # junk gate is landing disproportionately on this one class, which is
        # worth saying out loud; falling merely below `retention` is the
        # expected, documented compounding of two independent gates.
        entry['retained_after_vetoes'] = round(
            float(fires[pos].mean()) if pos.any() else 0.0, 4)
        floor = max(0.0, retention - (1.0 - audit['junk_joint_retention']))
        if entry['retained_after_vetoes'] < floor:
            warnings.append(
                f'{cls}: only {entry["retained_after_vetoes"]:.1%} of its '
                f'positives still fire once the vetoes and gates are applied, '
                f'below the {floor:.1%} the two retention floors bound')

        ladder = precision_ladder(score, y, _rungs(score, fires, base), fires)
        precisions = [p for _t, p in ladder]
        if precisions != sorted(precisions):
            msg = (f'{cls}: precision is NON-MONOTONIC along the ladder '
                   f'{precisions} — the index is badly behaved here')
            print(f'  WARNING {msg}')
            warnings.append(msg)
        c['ladder'] = ladder
        entry['ladder_precisions'] = precisions

    cfg['calibration'] = audit
    return cfg


def fit_from_frames(feat: pd.DataFrame, lab: pd.DataFrame, vocab: list[str],
                    retention: float = 0.90, split: str = 'train') -> dict:
    """Select `split` from a row-aligned (features, labels) pair and calibrate.

    Calibration reads the TRAIN split only: fitting a cut point on val or test
    is leakage that the floor test exists to rule out and that leaves no trace
    in the output.
    """
    if len(feat) != len(lab):
        raise ValueError(
            f'feature rows {len(feat):,} != label rows {len(lab):,} — the '
            f'sidecar is not aligned with the labels')
    feat = feat.reset_index(drop=True)
    lab = lab.reset_index(drop=True)
    if 'split' in lab.columns:
        mask = (lab['split'] == split).to_numpy()
    else:
        raise ValueError('the label frame has no "split" column; refusing to '
                         'calibrate on an unknown mixture of splits')
    if not mask.any():
        raise ValueError(f'no rows with split == {split!r}')
    labels = {c: lab.loc[mask, c].to_numpy() for c in vocab if c in lab.columns}
    return calibrate(feat.loc[mask], labels, vocab, retention)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--features', required=True, help='Task-2 sidecar parquet')
    ap.add_argument('--labels', required=True, help='labeled parquet')
    ap.add_argument('--vocab', choices=('7cls', 'pyx'), default='7cls')
    ap.add_argument('--retention', type=float, default=0.90)
    ap.add_argument('--split', default='train',
                    help='calibration split; anything but "train" is leakage')
    ap.add_argument('--out', default=None,
                    help='default: config/expert_rules_<vocab>.json')
    args = ap.parse_args()

    from data.dataset import _collapse_labels

    feat = pd.read_parquet(args.features)
    lab = _collapse_labels(pd.read_parquet(args.labels))
    vocab = CLASSES_7 if args.vocab == '7cls' else CLASSES_PYX
    # Feature columns already carry real parameter names (Task 2), so the
    # emitted JSON is human-auditable without a decoding step. Drop the
    # sidecar's identity columns so only parameters remain.
    params = [c for c in feat.columns
              if c not in ('tile_id', 'pixel_row', 'pixel_col', 'split')]
    cfg = fit_from_frames(feat[params], lab, vocab, args.retention, args.split)
    print(f"calibrated on {cfg['calibration']['n_rows']:,} "
          f"{args.split.upper()} rows (of {len(lab):,})")
    for w in cfg['calibration']['warnings']:
        print(f'  WARNING {w}')

    out = args.out or os.path.join('config', f'expert_rules_{args.vocab}.json')
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    with open(out, 'w') as f:
        json.dump(cfg, f, indent=2, sort_keys=False)
        f.write('\n')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()

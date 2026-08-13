"""Tests for the expert-ruleset calibrator.

Calibration is the step where a mistake is INVISIBLE. A veto placed on the
wrong side of its own distribution still writes a plausible-looking config
file; the only symptom is a map that is quietly empty for that class. So the
tests below check every cut point by the PROPERTY it is supposed to have --
what fraction of that class's own training positives survives it -- rather
than by its numeric value, and each one is paired in the task report with the
mutation it was verified against.
"""
import copy
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from data.expert_rules import CLASSES_7, CLASSES_PYX, evaluate_rules
from scripts.fit_expert_rules import (calibrate, fit_from_frames,
                                      precision_ladder)

# Band order of a real mrrsu header (see tests/test_expert_rules.py).
NAMES = ['R770','RBR','BD530_2','SH600_2','SH770','BD640_2','BD860_2','BD920_2',
         'RPEAK1','BDI1000VIS','R440','IRR1','R530','R600','BDI1000IR','OLINDEX3',
         'R1330','BD1300','LCPINDEX2','HCPINDEX2','VAR','ISLOPE1','BD1400',
         'BD1435','BD1500_2','ICER1_2','BD1750_2','BD1900_2','BD1900R2','BDI2000',
         'BD2100_2','BD2165','BD2190','MIN2200','BD2210_2','D2200','BD2230',
         'BD2250','MIN2250','BD2265','BD2290','D2300','BD2355','SINDEX2','ICER2_2',
         'MIN2295_2480','MIN2345_2537','BD2500_2','BD3000','BD3100','BD3200',
         'BD3400_2','CINDEX2','BD2600','IRR2','IRR3','R1080','R1506','R2529','R3920']

ALT_PARAMS = ['D2300', 'BD2290', 'BD1900R2', 'BD2210_2', 'MIN2200', 'SINDEX2',
              'BD2500_2']

# Row blocks of the synthetic training set, one per class.
BLOCKS = {'olivine': (0, 200), 'lcp': (200, 350), 'hcp': (350, 500),
          'plagioclase': (500, 600), 'alteration': (600, 700),
          'junk': (700, 750), 'bland': (750, 800)}
N_SYNTH = 800


def _synth(seed=0):
    """A synthetic training set with positives for every 7-class label.

    Every one of the 60 real parameter names is present, so the emitted config
    can be fed straight back into ``evaluate_rules``.
    """
    rng = np.random.default_rng(seed)
    feat = pd.DataFrame(rng.random((N_SYNTH, len(NAMES))) * 0.02,
                        columns=NAMES)
    feat['R770'] = 0.2                       # ordinary, non-saturated
    feat['VAR'] = 1.0 + rng.random(N_SYNTH) * 0.01
    feat['RPEAK1'] = 0.60 + rng.random(N_SYNTH) * 0.02
    feat['ICER1_2'] = rng.random(N_SYNTH) * 0.05
    feat['ICER2_2'] = rng.random(N_SYNTH) * 0.05
    feat['BD1435'] = rng.random(N_SYNTH) * 0.01
    feat['BD3200'] = rng.random(N_SYNTH) * 0.01

    def hi(lo, n, span=0.20):
        return lo + rng.random(n) * span

    for cls, (a, b) in BLOCKS.items():
        n = b - a
        if cls == 'olivine':
            feat.loc[a:b - 1, 'OLINDEX3'] = hi(0.10, n)
        elif cls == 'lcp':
            feat.loc[a:b - 1, 'LCPINDEX2'] = hi(0.10, n)
        elif cls == 'hcp':
            feat.loc[a:b - 1, 'HCPINDEX2'] = hi(0.10, n)
        elif cls == 'plagioclase':
            feat.loc[a:b - 1, 'BD1300'] = hi(0.02, n, 0.08)
            feat.loc[a:b - 1, 'RPEAK1'] = 0.72 + rng.random(n) * 0.06
            feat.loc[a:b - 1, 'BD1900R2'] = rng.random(n) * 0.005
        elif cls == 'alteration':
            for p in ALT_PARAMS:
                feat.loc[a:b - 1, p] = hi(0.10, n)
        elif cls == 'junk':
            feat.loc[a:b - 1, 'ICER1_2'] = 0.60 + rng.random(n) * 0.3

    # float32 like the Task-2 sidecar, so a threshold taken from the data is
    # exactly representable in the float32 cube the engine reads.
    feat = feat.astype(np.float32)
    labels = {c: np.zeros(N_SYNTH, dtype=np.float32) for c in CLASSES_7}
    for cls, (a, b) in BLOCKS.items():
        labels[cls][a:b] = 1.0
    labels['pyx'] = np.maximum(labels['lcp'], labels['hcp'])
    return feat, labels


def _cube(feat):
    """(n, 1, 60) cube in NAMES order from a feature frame."""
    return feat[NAMES].to_numpy(dtype=np.float32)[:, None, :]


def _junk_from(cfg, feat):
    """The engine's junk disjunction, recomputed here from the emitted config
    so the tests do not have to trust the calibrator's own helper."""
    j = cfg['junk']
    icer = np.fmax(feat['ICER1_2'], feat['ICER2_2']).to_numpy()
    co2 = np.fmax(feat['BD1435'], feat['BD3200']).to_numpy()
    return ((icer >= j['icer_high']) | (co2 >= j['co2_ice_high'])
            | (feat['R770'].to_numpy() > j['r770_max'])
            | (feat['VAR'].to_numpy() >= j['var_high']))


# ─────────────────────────────────────────────────────────────────────────────
# The brief's tests
# ─────────────────────────────────────────────────────────────────────────────

def test_precision_ladder_is_non_decreasing_on_separable_data():
    """A higher threshold sees a purer subset, so precision should not fall.
    A violation means the index is badly behaved and must be REPORTED."""
    score = np.concatenate([np.linspace(0, 0.5, 500), np.linspace(0.5, 1, 500)])
    y = (score > 0.5).astype(int)
    ladder = precision_ladder(score, y, np.linspace(0.1, 0.9, 9))
    precisions = [p for _t, p in ladder]
    assert precisions == sorted(precisions), f'non-monotonic: {precisions}'


def test_veto_retention_floor_is_respected():
    """A veto must not silently annihilate its own class."""
    rng = np.random.default_rng(0)
    n = 1000
    feat = pd.DataFrame({'OLINDEX3': rng.random(n), 'ICER1_2': rng.random(n)})
    y = (feat['OLINDEX3'] > 0.7).astype(int).to_numpy()
    cfg = calibrate(feat, {'olivine': y}, vocab=['olivine'], retention=0.90)
    veto = cfg['junk']['icer_high']
    kept = (feat.loc[y == 1, 'ICER1_2'] < veto).mean()
    assert kept >= 0.90, f'veto retained only {kept:.2%} of olivine positives'


def test_calibration_is_deterministic_on_identical_input():
    """Renamed from the brief's `test_calibration_uses_only_the_rows_it_is_given`,
    which called `calibrate` twice with the SAME argument and so could only ever
    test determinism -- it could not fail for its stated reason (reaching past
    the argument into a global split). The real property it named is tested by
    `test_calibration_depends_only_on_the_rows_passed_in` and
    `test_only_train_rows_influence_the_config` below."""
    rng = np.random.default_rng(1)
    feat = pd.DataFrame({'OLINDEX3': rng.random(200), 'ICER1_2': np.zeros(200)})
    y = (feat['OLINDEX3'] > 0.5).astype(int).to_numpy()
    a = calibrate(feat.iloc[:100], {'olivine': y[:100]}, ['olivine'])
    b = calibrate(feat.iloc[:100], {'olivine': y[:100]}, ['olivine'])
    assert a == b, 'calibration is not deterministic on identical input'


# ─────────────────────────────────────────────────────────────────────────────
# Added tests
# ─────────────────────────────────────────────────────────────────────────────

def test_precision_ladder_reports_precision_not_recall():
    """`test_precision_ladder_is_non_decreasing_on_separable_data` pins only the
    ORDER, and recall-instead-of-precision is monotone too (decreasing), so it
    is caught -- but a hit-count, an F1 or a 1-precision would not be. Task 6
    writes these numbers into the probs npz as probabilities, so the quantity
    itself has to be pinned."""
    score = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0, 0, 1, 1])
    (_t, prec), = precision_ladder(score, y, [1.0])
    # sel = {1, 2, 3}: 2 of 3 selected are positive. Recall would be 1.0.
    # Rungs are rounded to 4 dp on purpose -- the JSON is meant to be read.
    assert prec == pytest.approx(2 / 3, abs=1e-4), f'not precision: {prec}'


def test_primary_threshold_retains_the_class():
    """The mirror of the veto floor, on the other side of the comparison: a
    detection threshold placed at the WRONG quantile (the 90th of the positives
    rather than the 10th) leaves the class firing on a tenth of its own training
    pixels, which looks like a working config and an empty map."""
    feat, labels = _synth()
    cfg = calibrate(feat, labels, CLASSES_7, retention=0.90)
    for cls in ('olivine', 'lcp', 'hcp', 'plagioclase'):
        th = cfg['classes'][cls]['primary']['threshold']
        param = cfg['classes'][cls]['primary']['param']
        pos = feat.loc[labels[cls] == 1, param].to_numpy()
        kept = float((pos >= th).mean())
        assert kept >= 0.90, f'{cls} threshold {th} retains only {kept:.2%}'


def test_calibration_depends_only_on_the_rows_passed_in():
    """Same 100 rows, presented as a slice of a 200-row frame and as a
    standalone frame with a different index. Any use of the parent frame, of
    `len(feat)` as a stand-in for the number of rows fitted, or of index LABELS
    where positions were meant, changes the answer."""
    feat, labels = _synth()
    sub = feat.iloc[:300]
    lab_sub = {c: v[:300] for c, v in labels.items()}
    standalone = feat.iloc[:300].reset_index(drop=True).copy()
    standalone.index = standalone.index + 10_000
    a = calibrate(sub, lab_sub, CLASSES_7)
    b = calibrate(standalone, lab_sub, CLASSES_7)
    assert a == b, 'calibration is sensitive to the frame the rows came from'


def test_only_train_rows_influence_the_config():
    """Calibrating on val or test is leakage that the floor test exists to rule
    out, and it is undetectable in the output. The val/test rows here carry a
    deliberately incompatible distribution, so including them moves every
    number."""
    feat, labels = _synth()
    lab = pd.DataFrame({c: labels[c] for c in CLASSES_7})
    lab['split'] = 'train'

    poison = feat.copy()
    poison[NAMES] = 0.9                       # nothing like the train rows
    poison_lab = lab.copy()
    poison_lab['split'] = 'val'
    poison_lab2 = lab.copy()
    poison_lab2['split'] = 'test'

    all_feat = pd.concat([feat, poison, poison], ignore_index=True)
    all_lab = pd.concat([lab, poison_lab, poison_lab2], ignore_index=True)

    from_all = fit_from_frames(all_feat, all_lab, CLASSES_7)
    from_train = fit_from_frames(feat, lab, CLASSES_7)
    assert from_all == from_train, (
        'val/test rows changed the calibration -- the split filter leaks')
    assert from_all['calibration']['n_rows'] == N_SYNTH, (
        f"fitted on {from_all['calibration']['n_rows']} rows, expected "
        f'{N_SYNTH} train rows')


def test_all_four_junk_cut_points_are_calibrated():
    """Four cut points, not three. `co2_ice_high` left at None is a veto that
    never fires, so CO2 frost would be scored as mineralogy."""
    feat, labels = _synth()
    cfg = calibrate(feat, labels, CLASSES_7)
    assert set(cfg['junk']) == {'icer_high', 'co2_ice_high', 'var_high',
                                'r770_max'}
    for name, value in cfg['junk'].items():
        assert value is not None, f'{name} was left uncalibrated'
        assert np.isfinite(value), f'{name} is {value}: the veto never fires'


def test_co2_ice_cut_point_is_on_the_band_depth_scale_not_the_ice_ratio():
    """ICER1_2/ICER2_2 are ice RATIOS, BD1435/BD3200 are BAND DEPTHS. Deriving
    `co2_ice_high` from the ratio columns (or sharing one cut point) puts it on
    the wrong scale by an order of magnitude, with no error anywhere."""
    feat, labels = _synth()
    feat['BD1435'] = feat['BD1435'] * 0.01    # band depths pushed far below
    feat['BD3200'] = feat['BD3200'] * 0.01    # the ice ratios
    cfg = calibrate(feat, labels, CLASSES_7)
    ratio_pos = np.fmax(feat['ICER1_2'], feat['ICER2_2'])
    depth_pos = np.fmax(feat['BD1435'], feat['BD3200'])
    assert cfg['junk']['co2_ice_high'] < ratio_pos.min(), (
        'co2_ice_high sits on the ice-RATIO scale, not the band-depth scale: '
        f"{cfg['junk']['co2_ice_high']} vs depths up to {depth_pos.max()}")
    assert cfg['junk']['co2_ice_high'] > depth_pos.min()
    assert cfg['junk']['icer_high'] != cfg['junk']['co2_ice_high']


def test_the_junk_gate_as_a_whole_retains_the_mineral_positives():
    """Four vetoes that each keep 90% of the mineral positives can jointly keep
    0.9^4 = 66% of them. The engine applies them as ONE disjunction, so the
    floor has to hold for the disjunction -- otherwise the class is annihilated
    a quarter at a time by four numbers that each look defensible alone."""
    feat, labels = _synth()
    cfg = calibrate(feat, labels, CLASSES_7, retention=0.90)
    protect = np.zeros(N_SYNTH, dtype=bool)
    for cls, y in labels.items():
        if cls != 'junk':
            protect |= y.astype(bool)
    kept = float((~_junk_from(cfg, feat)[protect]).mean())
    assert kept >= 0.90, (
        f'the four junk vetoes jointly retain only {kept:.2%} of the mineral '
        'positives; each was placed as if it were the only one')
    assert cfg['calibration']['junk_joint_retention'] >= 0.90


def test_veto_thresholds_hold_against_the_float32_cube_the_engine_reads():
    """The retention floor is a promise about what the ENGINE does, and the
    engine compares a float32 cube against a Python float -- numpy demotes the
    scalar to float32 to do it. A cut point placed one float64 ULP above the
    k-th positive is rounded straight back onto that positive there, and the
    floor comes out one pixel short with no other symptom."""
    from scripts.fit_expert_rules import _veto_threshold
    v32 = np.linspace(0.001, 0.009, 100, dtype=np.float32)
    t = _veto_threshold(v32.astype(np.float64), 0.90)
    kept = float((v32 < t).mean())      # float32 array vs float: t is demoted
    assert kept >= 0.90, (
        f'{kept:.2%} survives in float32; the ULP step was lost on demotion')


def test_ladder_precision_is_the_precision_the_engine_will_achieve():
    """A ladder rung's precision is the number Task 6 writes into the probs npz
    as a probability. Measuring it over every pixel above the threshold rather
    than over the pixels the rule actually FIRES on (vetoes and gates applied)
    reports a different, lower number that no map ever realises."""
    feat, labels = _synth()
    # A block of ice pixels that ALSO respond strongly in OLINDEX3 and are NOT
    # labeled olivine. They clear the threshold but the ice veto stops them, so
    # measuring precision over "everything above the threshold" instead of
    # "everything the rule fires on" counts them as errors the map never makes.
    a, b = BLOCKS['junk']
    feat = feat.copy()
    feat.loc[a:b - 1, 'OLINDEX3'] = np.float32(0.25)
    cfg = calibrate(feat, labels, CLASSES_7)
    th = cfg['classes']['olivine']['primary']['threshold']
    junked = _junk_from(cfg, feat)
    fires = (feat['OLINDEX3'].to_numpy() >= th) & ~junked
    expected = float(labels['olivine'][fires].mean())
    unmasked = float(labels['olivine'][feat['OLINDEX3'].to_numpy() >= th].mean())
    assert expected - unmasked > 0.05, 'the ice block does not move precision'
    base_rung, base_prec = cfg['classes']['olivine']['ladder'][0]
    assert base_rung == th, 'the lowest rung is not the firing threshold'
    assert base_prec == pytest.approx(expected, abs=1e-4), (
        f'lowest rung reports precision {base_prec}, engine achieves '
        f'{expected:.4f} (unmasked would report {unmasked:.4f})')


def test_junk_labeled_pixels_do_not_relax_the_ice_veto():
    """The vetoes are placed to retain the positives of the MINERAL classes. A
    junk-labeled pixel is exactly what the veto exists to reject, so folding it
    into the retained set drags the ice cut point up past the ice it is meant
    to catch."""
    feat, labels = _synth()
    cfg = calibrate(feat, labels, CLASSES_7)
    a, b = BLOCKS['junk']
    icer_junk = np.fmax(feat['ICER1_2'], feat['ICER2_2']).to_numpy()[a:b]
    caught = float((icer_junk >= cfg['junk']['icer_high']).mean())
    assert caught >= 0.95, (
        f'only {caught:.0%} of the junk-labeled ice pixels are vetoed -- junk '
        'positives were counted as pixels the veto must retain')


def test_pyx_list_param_calibrates_on_the_max_of_both_indices():
    """The merged class's `primary.param` is a LIST plus a `reduce` key, not a
    string. Treating it as a column name raises; treating it as a 2-column
    frame silently calibrates on the pooled distribution of both indices, which
    is not the max. The threshold must retain 90% of pyx positives under the
    engine's own elementwise max."""
    feat, labels = _synth()
    cfg = calibrate(feat, labels, CLASSES_PYX)
    prim = cfg['classes']['pyx']['primary']
    assert prim['param'] == ['LCPINDEX2', 'HCPINDEX2']
    assert prim['reduce'] == 'max'
    th = prim['threshold']
    assert th is not None, 'pyx was left uncalibrated'
    ev = np.maximum(feat['LCPINDEX2'], feat['HCPINDEX2']).to_numpy()
    kept = float((ev[labels['pyx'] == 1] >= th).mean())
    assert kept >= 0.90, f'pyx threshold {th} retains only {kept:.2%}'
    lo, hi = BLOCKS['lcp']
    assert float((feat['LCPINDEX2'].to_numpy()[lo:hi] >= th).mean()) >= 0.90
    # Retention alone is one-sided -- a threshold of 0 retains everything, and
    # both a `min` reduction and a pooled two-column quantile land near 0
    # because the OTHER index is background on a single-pyroxene pixel. So the
    # calibrated rule is also required to DISCRIMINATE through the engine.
    out = evaluate_rules(_cube(feat), NAMES, cfg)
    fired = out['pyx'][:, 0] > 0
    pos = labels['pyx'] == 1
    # Exact retention is pinned above; here the engine only has to show that
    # the class still fires broadly (the junk gate has its own floor and the
    # two overlap by chance) and, crucially, that it discriminates.
    live = ~_junk_from(cfg, feat)
    assert float(fired[pos & live].mean()) >= 0.85, 'pyx misses its own positives'
    assert float(fired[~pos].mean()) < 0.10, (
        f'pyx fires on {fired[~pos].mean():.0%} of non-pyroxene ground — the '
        'threshold was fitted to a distribution the engine never evaluates')


def test_out_of_vocab_rule_blocks_are_not_emitted():
    """A 7-class config carrying an inert `pyx` block (or a pyx config carrying
    inert lcp/hcp blocks) invites an auditor to read a threshold that is never
    applied."""
    feat, labels = _synth()
    seven = calibrate(feat, labels, CLASSES_7)
    pyx = calibrate(feat, labels, CLASSES_PYX)
    assert 'pyx' not in seven['classes']
    assert 'lcp' not in pyx['classes'] and 'hcp' not in pyx['classes']
    for cfg, vocab in ((seven, CLASSES_7), (pyx, CLASSES_PYX)):
        assert set(cfg['classes']) <= set(vocab)


def test_class_with_no_positives_is_left_inert_not_firing_everywhere():
    """A `None` threshold reads as +inf in the engine, i.e. never fires. The
    failure this guards against is the opposite reflex -- falling back to a
    quantile of the WHOLE column when a class has no positives, which places
    the cut point in the middle of the data and paints the class everywhere."""
    feat, labels = _synth()
    labels = dict(labels)
    labels['plagioclase'] = np.zeros(N_SYNTH, dtype=np.float32)
    cfg = calibrate(feat, labels, CLASSES_7)
    assert cfg['classes']['plagioclase']['primary']['threshold'] is None
    out = evaluate_rules(_cube(feat), NAMES, cfg)
    assert out['plagioclase'].max() == 0, (
        f"plagioclase fired on {int((out['plagioclase'] > 0).sum())} pixels "
        'with no training positives at all')
    assert out['olivine'].max() > 0, 'the other classes stopped firing too'


def test_every_firing_pixel_receives_a_non_zero_ladder_score():
    """The ladder's lowest rung must not sit above the primary threshold. If it
    does, pixels between the two fire but score 0.0, which the engine then reads
    as "no mineral" and reassigns to BLAND -- a silent hole in the detection,
    invisible in the config."""
    feat, labels = _synth()
    cfg = calibrate(feat, labels, CLASSES_7)
    for cls in ('olivine', 'lcp', 'hcp', 'plagioclase', 'alteration'):
        rungs = [t for t, _p in cfg['classes'][cls]['ladder']]
        assert rungs == sorted(rungs)
        prim = cfg['classes'][cls].get('primary')
        if prim is not None:
            assert rungs[0] <= prim['threshold'], (
                f'{cls}: lowest rung {rungs[0]} is above the primary threshold '
                f"{prim['threshold']} -- firing pixels below it score 0")
    # And behaviourally: the set of pixels scoring > 0 must be EXACTLY the set
    # the rule fires on. A gap shows up as a strict subset, which a
    # ">= 90% of positives" assertion would not distinguish from ordinary
    # veto attrition.
    out = evaluate_rules(_cube(feat), NAMES, cfg)
    junked = _junk_from(cfg, feat)
    th = cfg['classes']['olivine']['primary']['threshold']
    expected = (feat['OLINDEX3'].to_numpy() >= th) & ~junked
    got = out['olivine'][:, 0] > 0
    assert expected.any()
    assert np.array_equal(got, expected), (
        f'{int((expected & ~got).sum())} pixels fire the olivine rule but score '
        '0.0 — the ladder does not cover the firing range')


def test_non_monotonic_precision_is_reported_not_smoothed():
    """A precision that falls as the rule gets stricter means the index is
    badly behaved there, and Task 6 would write a probability that means the
    opposite of what it says. It must be surfaced, not quietly repaired."""
    n = 600
    rng = np.random.default_rng(3)
    feat = pd.DataFrame(rng.random((n, len(NAMES))) * 0.001, columns=NAMES)
    feat['R770'] = 0.2
    feat['VAR'] = 1.0
    # OLINDEX3 anti-correlated with truth above the threshold: the strongest
    # responders are all false positives.
    feat['OLINDEX3'] = np.linspace(0.1, 0.9, n)
    y = np.ones(n, dtype=np.float32)
    y[n // 2:] = 0.0                       # the top half are all negatives
    cfg = calibrate(feat, {'olivine': y}, ['olivine'])
    precisions = [p for _t, p in cfg['classes']['olivine']['ladder']]
    assert precisions != sorted(precisions), 'test data is not non-monotonic'
    # Matching on the class name alone would be satisfied by ANY olivine
    # warning -- the retention notice fires here too -- so the match is on the
    # specific finding.
    warned = [w for w in cfg['calibration']['warnings']
              if 'olivine' in w and 'NON-MONOTONIC' in w]
    assert warned, (
        f'non-monotonic ladder {precisions} was not reported: '
        f"{cfg['calibration']['warnings']}")


def test_alteration_group_thresholds_are_all_filled_and_group_specific():
    """Every group needs a threshold for every parameter it requires; a missing
    key reads as +inf and silently deletes that group (carbonate first, which is
    the Nili Fossae terrain this project exists for)."""
    feat, labels = _synth()
    cfg = calibrate(feat, labels, CLASSES_7)
    groups = cfg['classes']['alteration']['groups']
    assert groups, 'the alteration groups were dropped'
    for g in groups:
        assert g['thresholds'] is not None, f"{g['name']} left uncalibrated"
        assert set(g['thresholds']) == set(g['requires']), (
            f"{g['name']} thresholds {sorted(g['thresholds'])} do not cover "
            f"{g['requires']}")
        for p, t in g['thresholds'].items():
            pos = feat.loc[labels['alteration'] == 1, p].to_numpy()
            assert float((pos >= t).mean()) >= 0.90, (
                f"{g['name']}/{p} threshold {t} retains too few positives")
    # The disjunction over groups must keep the class's own positives. Measured
    # among the rows the junk gate lets through, because that gate carries its
    # own separate floor and compounding the two would hide which one moved.
    out = evaluate_rules(_cube(feat), NAMES, cfg)
    a, b = BLOCKS['alteration']
    live = ~_junk_from(cfg, feat)[a:b]
    kept = float((out['alteration'][a:b, 0][live] > 0).mean())
    assert kept >= 0.90, f'the alteration groups retain only {kept:.2%}'


def test_plagioclase_window_and_hydration_veto_retain_their_positives():
    """RPEAK1 is a WAVELENGTH, so its cut is a two-sided window; the hydration
    veto is one-sided. Both are calibrated, and both must keep the class's own
    positives -- a window fitted to the whole column instead of the positives
    would sit around 0.60 um and reject every plagioclase pixel."""
    feat, labels = _synth()
    cfg = calibrate(feat, labels, CLASSES_7, retention=0.90)
    c = cfg['classes']['plagioclase']
    lo, hi = c['rpeak1_window']
    assert lo < hi
    pos = labels['plagioclase'] == 1
    rp = feat.loc[pos, 'RPEAK1'].to_numpy()
    assert float(((rp >= lo) & (rp <= hi)).mean()) >= 0.90, (
        f'RPEAK1 window [{lo}, {hi}] rejects the plagioclase positives at '
        f'{rp.min():.3f}-{rp.max():.3f}')
    # BOTH tails must be trimmed. Retention alone does not pin this: a window
    # left open at the bottom (lo = the smallest positive) still retains 90%,
    # and the joint-gate floor then quietly accepts it -- while admitting the
    # sub-0.7 um continuum-removal failure regime that is not plagioclase.
    assert lo > rp.min(), f'the RPEAK1 window is open at the bottom ({lo})'
    assert hi < rp.max(), f'the RPEAK1 window is open at the top ({hi})'
    hyd = feat.loc[pos, 'BD1900R2'].to_numpy()
    assert float((hyd < c['hydration_veto']).mean()) >= 0.90
    # The window must actually exclude the non-plagioclase background.
    bg = feat.loc[~pos, 'RPEAK1'].to_numpy()
    assert float(((bg >= lo) & (bg <= hi)).mean()) < 0.5, (
        'the RPEAK1 window admits the whole background population')
    # And the three gates are ANDed by the engine, so the floor has to hold for
    # the CONJUNCTION: three gates at 90% apiece compound to 73%, which is a
    # third of the class quietly gone.
    bd = feat.loc[pos, 'BD1300'].to_numpy()
    joint = float(((bd >= c['primary']['threshold'])
                   & (rp >= lo) & (rp <= hi)
                   & (hyd < c['hydration_veto'])).mean())
    assert joint >= 0.90, (
        f'the plagioclase rule as a whole retains only {joint:.2%}; its three '
        'gates were each placed as if it were the only one')


def test_a_veto_that_rejects_most_of_the_training_set_is_reported():
    """A parameter that is constant across the positives collapses its veto onto
    the data itself, junking nearly everything. That is exactly the config that
    looks fine and produces an empty map, so it is reported."""
    n = 400
    rng = np.random.default_rng(4)
    feat = pd.DataFrame(rng.random((n, len(NAMES))) * 0.01, columns=NAMES)
    feat['R770'] = 0.2
    feat['VAR'] = 1.0
    feat['OLINDEX3'] = rng.random(n)
    y = (feat['OLINDEX3'] > 0.5).astype(np.float32).to_numpy()
    feat.loc[y == 1, 'BD1435'] = 0.0          # constant across the positives
    feat.loc[y == 0, 'BD1435'] = 0.5          # everything else is "CO2 ice"
    cfg = calibrate(feat, {'olivine': y}, ['olivine'])
    assert cfg['calibration']['junk_rejects_frac'] > 0.4
    assert any('junk' in w for w in cfg['calibration']['warnings']), (
        f"a veto junking {cfg['calibration']['junk_rejects_frac']:.0%} of the "
        f"training set was not reported: {cfg['calibration']['warnings']}")


def test_emitted_config_survives_json_and_drives_the_engine():
    """The config is written to disk and read back by Task 6, so every value has
    to be a plain JSON scalar -- a numpy float32 or an int64 raises at
    json.dump, and a NaN silently becomes a threshold that never compares
    true."""
    feat, labels = _synth()
    cfg = calibrate(feat, labels, CLASSES_7)
    text = json.dumps(cfg, indent=2)
    back = json.loads(text)
    assert back == cfg, 'the config does not round-trip through JSON'
    for value in (cfg['junk'].values()):
        assert isinstance(value, float) and not np.isnan(value)
    out = evaluate_rules(_cube(feat), NAMES, back)
    assert list(out.keys()) == CLASSES_7
    for cls in ('olivine', 'lcp', 'hcp', 'plagioclase', 'alteration'):
        a, b = BLOCKS[cls]
        assert out[cls][a:b, 0].max() > 0, f'{cls} never fires on its own data'
    ja, jb = BLOCKS['junk']
    assert float((out['junk'][ja:jb, 0] > 0).mean()) >= 0.95
    # Bland is the residual: its rows carry no mineral response at all, and
    # every one of them is accounted for as either bland or junk.
    ba, bb = BLOCKS['bland']
    for cls in ('olivine', 'lcp', 'hcp', 'plagioclase', 'alteration'):
        assert out[cls][ba:bb, 0].max() == 0, f'{cls} fired on bland ground'
    assert np.all((out['bland'][ba:bb, 0] + out['junk'][ba:bb, 0]) > 0)


def test_calibrate_does_not_mutate_the_shared_default_rules():
    """DEFAULT_RULES is a module-level dict. Editing it in place would make the
    pyx run depend on the 7-class run that preceded it in the same process."""
    from data import expert_rules
    feat, labels = _synth()
    calibrate(feat, labels, CLASSES_7)
    calibrate(feat, labels, CLASSES_PYX)
    # Asserted against the SHIPPED state (every cut point None), not against a
    # snapshot taken at the top of this test: a snapshot is already polluted if
    # any earlier test in the session leaked, so it would pass on a module that
    # mutates DEFAULT_RULES identically every time.
    d = expert_rules.DEFAULT_RULES
    assert all(v is None for v in d['junk'].values()), d['junk']
    for cls, block in d['classes'].items():
        assert block['ladder'] is None, f'{cls} ladder leaked'
        if 'primary' in block:
            assert block['primary']['threshold'] is None, f'{cls} threshold leaked'
        for key in ('rpeak1_window', 'hydration_veto'):
            if key in block:
                assert block[key] is None, f'{cls} {key} leaked'
        for g in block.get('groups', []):
            assert g['thresholds'] is None, f"{cls}/{g['name']} leaked"


def test_main_writes_both_vocabulary_configs_from_the_train_split(tmp_path):
    """End to end at the CLI: the two files Task 6 loads, from a sidecar that is
    mostly val/test."""
    feat, labels = _synth()
    lab = pd.DataFrame({c: labels[c] for c in CLASSES_7})
    lab['olivine_t1'] = lab['olivine']
    lab['olivine_t2'] = 0.0
    lab['other'] = lab['bland']
    lab['split'] = 'train'
    held = lab.copy()
    held['split'] = 'val'
    held_feat = feat.copy()
    held_feat[NAMES] = 0.9

    feat_all = pd.concat([feat, held_feat], ignore_index=True)
    lab_all = pd.concat([lab, held], ignore_index=True)
    fpath = tmp_path / 'feat.parquet'
    lpath = tmp_path / 'lab.parquet'
    feat_all.to_parquet(fpath, index=False)
    lab_all.to_parquet(lpath, index=False)

    outs = {}
    for vocab in ('7cls', 'pyx'):
        out = tmp_path / f'expert_rules_{vocab}.json'
        r = subprocess.run(
            [sys.executable, 'scripts/fit_expert_rules.py',
             '--features', str(fpath), '--labels', str(lpath),
             '--vocab', vocab, '--out', str(out)],
            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        outs[vocab] = json.loads(out.read_text())
        assert outs[vocab]['calibration']['n_rows'] == N_SYNTH, (
            'main calibrated on more than the train split')

    assert outs['7cls']['vocab'] == CLASSES_7
    assert outs['pyx']['vocab'] == CLASSES_PYX
    assert 'pyx' in outs['pyx']['classes']
    assert set(outs['7cls']['junk']) == {'icer_high', 'co2_ice_high',
                                         'var_high', 'r770_max'}

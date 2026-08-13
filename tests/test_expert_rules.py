"""Tests for the expert band-parameter rule engine.

The regression these exist to prevent is EXCLUSIVITY. The vocabulary is
multi-label: a pixel can be olivine AND hcp. Any gate that lets one label
suppress another silently deletes real assemblages from the map.
"""
import copy

import numpy as np
import pytest

from data.expert_rules import CLASSES_PYX, evaluate_rules, DEFAULT_RULES

NAMES = ['R770','RBR','BD530_2','SH600_2','SH770','BD640_2','BD860_2','BD920_2',
         'RPEAK1','BDI1000VIS','R440','IRR1','R530','R600','BDI1000IR','OLINDEX3',
         'R1330','BD1300','LCPINDEX2','HCPINDEX2','VAR','ISLOPE1','BD1400',
         'BD1435','BD1500_2','ICER1_2','BD1750_2','BD1900_2','BD1900R2','BDI2000',
         'BD2100_2','BD2165','BD2190','MIN2200','BD2210_2','D2200','BD2230',
         'BD2250','MIN2250','BD2265','BD2290','D2300','BD2355','SINDEX2','ICER2_2',
         'MIN2295_2480','MIN2345_2537','BD2500_2','BD3000','BD3100','BD3200',
         'BD3400_2','CINDEX2','BD2600','IRR2','IRR3','R1080','R1506','R2529','R3920']


def _cfg():
    """Calibrated config with simple round thresholds, one pixel per scenario."""
    c = copy.deepcopy(DEFAULT_RULES)
    for cls in ('olivine', 'lcp', 'hcp'):
        c['classes'][cls]['primary']['threshold'] = 0.05
        c['classes'][cls]['ladder'] = [[0.05, 0.6], [0.10, 0.9]]
    c['classes']['plagioclase']['rpeak1_window'] = [0.70, 0.80]
    c['classes']['plagioclase']['primary']['threshold'] = 0.01
    c['classes']['plagioclase']['hydration_veto'] = 0.10
    c['classes']['plagioclase']['ladder'] = [[0.01, 0.7]]
    c['classes']['pyx']['primary']['threshold'] = 0.05
    c['classes']['pyx']['ladder'] = [[0.05, 0.6], [0.10, 0.9]]
    c['junk']['icer_high'] = 0.5
    c['junk']['co2_ice_high'] = 0.5
    c['junk']['var_high'] = 1e6
    c['junk']['r770_max'] = 1.0
    for g in c['classes']['alteration']['groups']:
        g['thresholds'] = {k: 0.05 for k in g['requires']}
    c['classes']['alteration']['ladder'] = [[0.05, 0.8]]
    return c


def _cfg_pyx():
    """The same calibrated config under the merged-pyroxene vocabulary."""
    c = _cfg()
    c['vocab'] = CLASSES_PYX
    return c


def _blank(n=1, h=1):
    return np.zeros((h, n, 60), dtype=np.float32)


def _set(cube, param, value, i=0, r=0):
    cube[r, i, NAMES.index(param)] = value


# ─────────────────────────────────────────────────────────────────────────────
# The brief's tests
# ─────────────────────────────────────────────────────────────────────────────

def test_olivine_and_hcp_can_BOTH_fire():
    """THE multi-label guarantee. Olivine-bearing basalt is ordinary; an
    exclusive veto would suppress it silently."""
    cube = _blank()
    _set(cube, 'OLINDEX3', 0.20)
    _set(cube, 'HCPINDEX2', 0.20)
    _set(cube, 'R770', 0.2)
    out = evaluate_rules(cube, NAMES, _cfg())
    assert out['olivine'][0, 0] > 0, 'olivine suppressed by pyroxene presence'
    assert out['hcp'][0, 0] > 0, 'hcp suppressed by olivine presence'


def test_dominance_raises_the_tier_but_does_not_gate():
    """Both pyroxene labels fire when both indices are high; the dominant one
    scores higher. It must NOT zero the weaker one."""
    cube = _blank()
    _set(cube, 'LCPINDEX2', 0.20)
    _set(cube, 'HCPINDEX2', 0.08)
    _set(cube, 'R770', 0.2)
    out = evaluate_rules(cube, NAMES, _cfg())
    assert out['hcp'][0, 0] > 0, 'non-dominant pyroxene was gated to zero'
    assert out['lcp'][0, 0] > out['hcp'][0, 0], 'dominance did not raise the tier'


def test_carbonate_without_hydration_is_still_alteration():
    """Carbonates are anhydrous. A blanket hydration requirement would silently
    reject Mg-carbonate — the Nili Fossae terrain this project exists for."""
    cube = _blank()
    _set(cube, 'BD2500_2', 0.20)
    _set(cube, 'D2300', 0.20)
    _set(cube, 'BD1900R2', 0.0)      # explicitly NOT hydrated
    _set(cube, 'R770', 0.2)
    out = evaluate_rules(cube, NAMES, _cfg())
    assert out['alteration'][0, 0] > 0


def test_ice_is_junk_and_not_alteration():
    cube = _blank()
    _set(cube, 'ICER1_2', 0.9)
    _set(cube, 'D2300', 0.20)
    _set(cube, 'BD2290', 0.20)
    _set(cube, 'BD1900R2', 0.20)
    _set(cube, 'R770', 0.2)
    out = evaluate_rules(cube, NAMES, _cfg())
    assert out['junk'][0, 0] > 0
    assert out['alteration'][0, 0] == 0, 'frost registered as alteration'


def test_plagioclase_needs_rpeak1_inside_the_window_not_merely_high():
    """RPEAK1 is a WAVELENGTH (~0.7-0.8 um for plag), not an amplitude. A
    one-sided 'high' test would admit everything above 0.8."""
    cube = _blank(2)
    for i in (0, 1):
        _set(cube, 'BD1300', 0.05, i)
        _set(cube, 'R770', 0.2, i)
    _set(cube, 'RPEAK1', 0.75, 0)    # inside
    _set(cube, 'RPEAK1', 0.95, 1)    # above the window
    out = evaluate_rules(cube, NAMES, _cfg())
    assert out['plagioclase'][0, 0] > 0
    assert out['plagioclase'][0, 1] == 0


def test_bland_is_the_residual():
    cube = _blank()
    _set(cube, 'R770', 0.2)
    out = evaluate_rules(cube, NAMES, _cfg())
    assert out['bland'][0, 0] > 0
    for c in ('olivine', 'lcp', 'hcp', 'plagioclase', 'alteration'):
        assert out[c][0, 0] == 0


def test_all_nan_pixel_scores_zero_everywhere():
    cube = np.full((1, 1, 60), np.nan, dtype=np.float32)
    out = evaluate_rules(cube, NAMES, _cfg())
    for c, v in out.items():
        assert v[0, 0] == 0, f'{c} scored on an all-NaN pixel'


# ─────────────────────────────────────────────────────────────────────────────
# Added tests. Rationale for each is in the docstring; see the task report for
# the mutations that each one was verified against.
# ─────────────────────────────────────────────────────────────────────────────

def test_olivine_score_is_identical_with_and_without_coexisting_hcp():
    """Multi-label means INDEPENDENT, not merely non-zero. A soft veto that
    merely demoted olivine in the presence of pyroxene (instead of zeroing it)
    would pass `> 0` while still corrupting every olivine-bearing basalt.
    Each label must be scored on its own evidence alone."""
    alone = _blank()
    _set(alone, 'OLINDEX3', 0.20)
    _set(alone, 'R770', 0.2)

    together = _blank()
    _set(together, 'OLINDEX3', 0.20)
    _set(together, 'HCPINDEX2', 0.30)
    _set(together, 'LCPINDEX2', 0.30)
    _set(together, 'R770', 0.2)

    a = evaluate_rules(alone, NAMES, _cfg())['olivine'][0, 0]
    b = evaluate_rules(together, NAMES, _cfg())['olivine'][0, 0]
    assert a > 0
    assert b == a, ('olivine was demoted by coexisting pyroxene: '
                    f'{a} alone vs {b} together')


def test_dominance_modifier_is_actually_applied_within_one_ladder_rung():
    """`test_dominance_raises_the_tier_but_does_not_gate` cannot fail for its
    second stated reason: at LCP=0.20 / HCP=0.08 the two values sit on
    DIFFERENT ladder rungs, so lcp > hcp holds even if the dominance modifier
    is deleted outright. Here both values clear the same top rung, so the two
    scores are equal unless dominance actually demotes the weaker one — and
    still non-zero unless dominance gates."""
    cube = _blank()
    _set(cube, 'LCPINDEX2', 0.12)
    _set(cube, 'HCPINDEX2', 0.11)   # same ladder rung as LCP
    _set(cube, 'R770', 0.2)
    out = evaluate_rules(cube, NAMES, _cfg())
    assert out['hcp'][0, 0] > 0, 'non-dominant pyroxene was gated to zero'
    assert out['lcp'][0, 0] > out['hcp'][0, 0], (
        'dominance modifier not applied: both pyroxenes scored '
        f"{out['lcp'][0, 0]} / {out['hcp'][0, 0]} on the same rung")


def test_plagioclase_rejects_rpeak1_below_the_window():
    """The window is two-sided in BOTH directions. RPEAK1 below ~0.7 um is the
    continuum-removal failure regime documented in data/mrrsu_aux.py, not
    plagioclase; a `<= hi`-only test would admit it."""
    cube = _blank(2)
    for i in (0, 1):
        _set(cube, 'BD1300', 0.05, i)
        _set(cube, 'R770', 0.2, i)
    _set(cube, 'RPEAK1', 0.75, 0)    # inside
    _set(cube, 'RPEAK1', 0.55, 1)    # below the window
    out = evaluate_rules(cube, NAMES, _cfg())
    assert out['plagioclase'][0, 0] > 0
    assert out['plagioclase'][0, 1] == 0, 'RPEAK1 below the window admitted'


def test_nan_pixel_scores_zero_next_to_a_firing_neighbour():
    """`test_all_nan_pixel_scores_zero_everywhere` uses a 1x1 cube, so an
    implementation that returned all-zeros unconditionally, or one that let a
    neighbour's value leak across pixels, is not distinguished. Here the NaN
    pixel must be zero while the pixel beside it fires."""
    cube = _blank(2)
    cube[0, 0, :] = np.nan
    _set(cube, 'OLINDEX3', 0.20, 1)
    _set(cube, 'R770', 0.2, 1)
    out = evaluate_rules(cube, NAMES, _cfg())
    for c in out:
        assert out[c][0, 0] == 0, f'{c} scored on an all-NaN pixel'
    assert out['olivine'][0, 1] > 0, 'the valid neighbour stopped firing'


def test_scores_are_placed_per_pixel_on_a_non_square_grid():
    """Every other test uses H=1, so a flatten/reshape or row/column mixup is
    invisible. A non-square grid with each scenario at a distinct (row, col)
    pins the geometry."""
    cube = _blank(n=5, h=3)
    cube[..., NAMES.index('R770')] = 0.2
    _set(cube, 'OLINDEX3', 0.20, i=0, r=0)
    _set(cube, 'HCPINDEX2', 0.20, i=0, r=0)
    _set(cube, 'BD2500_2', 0.20, i=2, r=1)
    _set(cube, 'D2300', 0.20, i=2, r=1)
    _set(cube, 'ICER1_2', 0.9, i=4, r=2)
    out = evaluate_rules(cube, NAMES, _cfg())

    for c, v in out.items():
        assert v.shape == (3, 5), f'{c} has shape {v.shape}, expected (3, 5)'

    assert out['olivine'][0, 0] > 0 and out['hcp'][0, 0] > 0
    assert np.count_nonzero(out['olivine']) == 1
    assert np.count_nonzero(out['hcp']) == 1
    assert out['alteration'][1, 2] > 0
    assert np.count_nonzero(out['alteration']) == 1
    assert out['junk'][2, 4] > 0
    assert np.count_nonzero(out['junk']) == 1
    # everything else is bland: 15 - 3 scored pixels
    assert np.count_nonzero(out['bland']) == 12
    assert out['bland'][0, 0] == 0 and out['bland'][1, 2] == 0
    assert out['bland'][2, 4] == 0


def test_output_contract_keys_order_dtype_and_range():
    """Task 6 stacks the outputs as `[scores[c] for c in vocab]` into a probs
    array, so the key set, its order and the [0, 1] range are load-bearing."""
    cfg = _cfg()
    cube = _blank(n=4, h=2)
    cube[..., NAMES.index('R770')] = 0.2
    _set(cube, 'OLINDEX3', 0.20, i=1, r=1)
    out = evaluate_rules(cube, NAMES, cfg)
    assert list(out.keys()) == list(cfg['vocab'])
    for c, v in out.items():
        assert v.dtype == np.float32, f'{c} is {v.dtype}, not float32'
        assert v.shape == (2, 4)
        assert np.all(np.isfinite(v)), f'{c} contains non-finite scores'
        assert v.min() >= 0.0 and v.max() <= 1.0, f'{c} outside [0, 1]'


def test_evaluate_rules_does_not_mutate_the_config():
    """The tile loop in Task 6 reuses one config across every tile; an
    in-place edit would make tile N depend on tile N-1."""
    cfg = _cfg()
    before = copy.deepcopy(cfg)
    cube = _blank()
    _set(cube, 'OLINDEX3', 0.20)
    _set(cube, 'R770', 0.2)
    evaluate_rules(cube, NAMES, cfg)
    assert cfg == before, 'evaluate_rules mutated its config argument'


def test_parameters_are_resolved_by_name_not_by_position():
    """Band order comes from each tile's OWN header (Task 1). A rule that
    indexed a fixed position would score the wrong parameter on a reordered
    tile with no error at all."""
    shuffled = list(NAMES)
    # move OLINDEX3 from index 15 to the end
    shuffled.remove('OLINDEX3')
    shuffled.append('OLINDEX3')

    cube = _blank()
    _set(cube, 'R770', 0.2)
    ref = evaluate_rules(cube.copy(), NAMES, _cfg())
    assert ref['olivine'][0, 0] == 0

    shuf_cube = np.zeros((1, 1, 60), dtype=np.float32)
    shuf_cube[0, 0, shuffled.index('R770')] = 0.2
    shuf_cube[0, 0, shuffled.index('OLINDEX3')] = 0.20
    out = evaluate_rules(shuf_cube, shuffled, _cfg())
    assert out['olivine'][0, 0] > 0, 'OLINDEX3 not found via the name list'
    assert out['bland'][0, 0] == 0


def test_pyx_fires_from_either_pyroxene_index_alone():
    """The merged class exists because LCPINDEX2/HCPINDEX2 cross-respond and
    the 2 um band-centre discrimination is unreliable, so either index alone is
    sufficient evidence of pyroxene."""
    cube = _blank(2)
    for i in (0, 1):
        _set(cube, 'R770', 0.2, i)
    _set(cube, 'LCPINDEX2', 0.20, 0)   # LCP only
    _set(cube, 'HCPINDEX2', 0.20, 1)   # HCP only
    out = evaluate_rules(cube, NAMES, _cfg_pyx())
    assert out['pyx'][0, 0] > 0, 'pyx missed an LCP-only pixel'
    assert out['pyx'][0, 1] > 0, 'pyx missed an HCP-only pixel'
    assert list(out.keys()) == list(CLASSES_PYX)
    assert 'lcp' not in out and 'hcp' not in out


def test_pyx_takes_the_max_not_the_min_of_the_two_indices():
    """A `min` reduction would demand BOTH indices be high, which is exactly
    the discrimination the merge exists to avoid relying on."""
    cube = _blank()
    _set(cube, 'LCPINDEX2', 0.20)
    _set(cube, 'HCPINDEX2', 0.06)   # would land a rung lower under `min`
    _set(cube, 'R770', 0.2)
    out = evaluate_rules(cube, NAMES, _cfg_pyx())
    assert out['pyx'][0, 0] == pytest.approx(0.9), (
        'pyx did not score at the stronger index: got '
        f"{out['pyx'][0, 0]}, expected the 0.9 rung of max(0.20, 0.06)")


def test_pyx_is_not_demoted_when_both_indices_are_high():
    """No dominance term on the merged class: there is nothing left to be
    dominant over, so a both-high pixel must score the full top rung, the same
    as a single-index pixel of the same strength."""
    both = _blank()
    _set(both, 'LCPINDEX2', 0.20)
    _set(both, 'HCPINDEX2', 0.20)
    _set(both, 'R770', 0.2)
    one = _blank()
    _set(one, 'LCPINDEX2', 0.20)
    _set(one, 'R770', 0.2)
    a = evaluate_rules(both, NAMES, _cfg_pyx())['pyx'][0, 0]
    b = evaluate_rules(one, NAMES, _cfg_pyx())['pyx'][0, 0]
    assert a == pytest.approx(0.9)
    assert a == b, f'both-high pyx demoted: {a} vs {b} for a single index'
    assert 'dominance_over' not in DEFAULT_RULES['classes']['pyx']


def test_scalar_param_string_still_works_for_the_non_pyx_classes():
    """The list+reduce form is additive: every other class keeps a plain
    parameter-name string and its unchanged behaviour."""
    for cls in ('olivine', 'lcp', 'hcp', 'plagioclase'):
        param = DEFAULT_RULES['classes'][cls]['primary']['param']
        assert isinstance(param, str), f'{cls} primary param became {param!r}'
    cube = _blank()
    _set(cube, 'OLINDEX3', 0.20)
    _set(cube, 'LCPINDEX2', 0.20)
    _set(cube, 'R770', 0.2)
    out = evaluate_rules(cube, NAMES, _cfg())
    assert out['olivine'][0, 0] == pytest.approx(0.9)
    assert out['lcp'][0, 0] == pytest.approx(0.9)


def test_co2_ice_and_ratio_ice_have_independent_thresholds():
    """ICER1_2/ICER2_2 are ice-abundance RATIOS; BD1435/BD3200 are BAND DEPTHS.
    One cut point shared between them cannot be right for both. Same pixel,
    swapped thresholds, opposite verdicts."""
    cube = _blank()
    _set(cube, 'BD1435', 0.30)     # CO2-ice band depth
    _set(cube, 'ICER1_2', 0.0)     # no ice ratio at all
    _set(cube, 'R770', 0.2)

    strict_co2 = _cfg()
    strict_co2['junk']['icer_high'] = 0.9    # permissive on the ratio
    strict_co2['junk']['co2_ice_high'] = 0.05  # strict on the band depth
    assert evaluate_rules(cube, NAMES, strict_co2)['junk'][0, 0] > 0, \
        'CO2-ice band depth did not trigger its own veto'

    strict_ratio = _cfg()
    strict_ratio['junk']['icer_high'] = 0.05   # strict on the ratio
    strict_ratio['junk']['co2_ice_high'] = 0.9  # permissive on the band depth
    assert evaluate_rules(cube, NAMES, strict_ratio)['junk'][0, 0] == 0, (
        'the ice-RATIO threshold was applied to the CO2 band depth — the two '
        'cut points are not independent')


def test_uncalibrated_none_thresholds_are_inert_rather_than_a_typeerror():
    """DEFAULT_RULES ships with every cut point None, and Task 4 leaves a class
    with no training positives at None. That must mean "never fires", not
    `TypeError: '>=' not supported between float and NoneType`."""
    cube = _blank()
    _set(cube, 'OLINDEX3', 0.20)
    _set(cube, 'ICER1_2', 0.9)
    _set(cube, 'BD1435', 0.9)
    _set(cube, 'R770', 0.2)
    out = evaluate_rules(cube, NAMES, copy.deepcopy(DEFAULT_RULES))
    for c in ('olivine', 'lcp', 'hcp', 'plagioclase', 'alteration', 'junk'):
        assert out[c][0, 0] == 0, f'{c} fired on an uncalibrated ruleset'
    assert out['bland'][0, 0] > 0


def test_plagioclase_is_vetoed_by_hydration_even_inside_the_window():
    """The dust veto. Plagioclase over-firing on featureless bright ground is a
    live failure mode in this project (hand plag sits at SAM recall 0.29 and
    competes with bland), so BD1900R2 above the veto must suppress it even when
    RPEAK1 and BD1300 both look right. Mirror case: the same pixel dry fires."""
    cube = _blank(2)
    for i in (0, 1):
        _set(cube, 'RPEAK1', 0.75, i)   # inside the window
        _set(cube, 'BD1300', 0.05, i)   # above the primary threshold
        _set(cube, 'R770', 0.2, i)
    _set(cube, 'BD1900R2', 0.20, 0)     # hydrated: dust / hydrated phase
    _set(cube, 'BD1900R2', 0.00, 1)     # dry
    out = evaluate_rules(cube, NAMES, _cfg())
    assert out['plagioclase'][0, 0] == 0, 'hydrated pixel scored plagioclase'
    assert out['plagioclase'][0, 1] > 0, 'dry plagioclase stopped firing'


def test_non_physical_r770_is_junk():
    """Saturation / calibration blow-ups. The other half of what junk is for;
    only the ice branch was covered."""
    cube = _blank(2)
    _set(cube, 'R770', 1.5, 0)    # above r770_max = 1.0
    _set(cube, 'R770', 0.2, 1)    # control
    out = evaluate_rules(cube, NAMES, _cfg())
    assert out['junk'][0, 0] > 0, 'non-physical R770 was not flagged junk'
    assert out['bland'][0, 0] == 0, 'a saturated pixel was called bland'
    assert out['junk'][0, 1] == 0, 'an ordinary pixel was flagged junk'


def test_extreme_var_is_junk():
    """Dead / noisy spectra: VAR is the spectral-variance artifact flag."""
    cube = _blank(2)
    for i in (0, 1):
        _set(cube, 'R770', 0.2, i)
    _set(cube, 'VAR', 2e6, 0)     # above var_high = 1e6
    _set(cube, 'VAR', 1.0, 1)     # control
    out = evaluate_rules(cube, NAMES, _cfg())
    assert out['junk'][0, 0] > 0, 'extreme VAR was not flagged junk'
    assert out['bland'][0, 0] == 0, 'a noise-dominated pixel was called bland'
    assert out['junk'][0, 1] == 0, 'an ordinary pixel was flagged junk'


def test_alteration_group_strength_is_the_weakest_required_parameter():
    """A group requires ALL its diagnostics, so its strength is its WEAKEST
    member: a max-reduction would let one strong parameter carry a group whose
    other requirement is barely met. Invisible under a single-rung ladder, so
    this uses a multi-rung one — the shape Task 4 will actually emit."""
    cfg = _cfg()
    cfg['classes']['alteration']['ladder'] = [[0.05, 0.5], [0.20, 0.95]]
    cube = _blank()
    # femg_phyllosilicate requires D2300, BD2290, BD1900R2 — deliberately
    # spread across two rungs.
    _set(cube, 'D2300', 0.30)      # top rung
    _set(cube, 'BD1900R2', 0.30)   # top rung
    _set(cube, 'BD2290', 0.08)     # bottom rung: the weakest requirement
    _set(cube, 'R770', 0.2)
    out = evaluate_rules(cube, NAMES, cfg)
    assert out['alteration'][0, 0] == pytest.approx(0.5), (
        'alteration scored the rung of its STRONGEST parameter, not its '
        f"weakest: got {out['alteration'][0, 0]}, expected 0.5 for min(0.30, "
        '0.30, 0.08)')


def test_unknown_parameter_name_raises_instead_of_scoring_zero():
    """A header missing a parameter a rule needs must fail loudly; silently
    scoring the class zero would produce a plausible, wrong map."""
    names = list(NAMES)
    names[names.index('OLINDEX3')] = 'SOMETHING_ELSE'
    cube = _blank()
    _set(cube, 'R770', 0.2)
    with pytest.raises(KeyError, match='OLINDEX3'):
        evaluate_rules(cube, names, _cfg())

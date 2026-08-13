"""Tests for the baseline tile scorer.

The npz is the ONLY interface between a baseline and the floor test, so the
tests below are about the CONTRACT, not about mineralogy:

  * structure is compared against an npz written by the model's own writer
    (``classify_tile_supervised.save_probs``) rather than against a hand-written
    expectation, because a hand-written expectation drifts away from what the
    vectorizer actually reads and stops being a test of anything;
  * the vocabulary and the channel count are the two ways an npz can be loaded
    happily and mislabelled silently, so both must RAISE;
  * ``valid_mask`` must come from the IMPORTED ``load_tile`` -- a mask derived
    from mrrsu instead would change polygon counts for reasons unrelated to the
    method, i.e. a confound that reads as a result;
  * the feature matrix must follow the FITTED column order -- a reordered
    feature vector produces garbage with no error anywhere.
"""
import glob
import json
import os
import sys

import numpy as np
import pytest
from rasterio.transform import Affine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.expert_rules import CLASSES_7
from scripts.classify_tile_baseline import (assemble_feature_matrix,
                                            assemble_npz_payload, score_tile)

# Band order of a real mrrsu header (same list as tests/test_expert_rules.py).
NAMES = ['R770','RBR','BD530_2','SH600_2','SH770','BD640_2','BD860_2','BD920_2',
         'RPEAK1','BDI1000VIS','R440','IRR1','R530','R600','BDI1000IR','OLINDEX3',
         'R1330','BD1300','LCPINDEX2','HCPINDEX2','VAR','ISLOPE1','BD1400',
         'BD1435','BD1500_2','ICER1_2','BD1750_2','BD1900_2','BD1900R2','BDI2000',
         'BD2100_2','BD2165','BD2190','MIN2200','BD2210_2','D2200','BD2230',
         'BD2250','MIN2250','BD2265','BD2290','D2300','BD2355','SINDEX2','ICER2_2',
         'MIN2295_2480','MIN2345_2537','BD2500_2','BD3000','BD3100','BD3200',
         'BD3400_2','CINDEX2','BD2600','IRR2','IRR3','R1080','R1506','R2529','R3920']

H, W = 6, 5
TRANSFORM = Affine(0.1, 0.0, -100.0, 0.0, -0.1, 20.0)


class _StubCRS:
    def to_wkt(self):
        return 'PROJCS["stub"]'


def _cube(fill=0.5):
    """(H, W, 60) cube whose band k is constant k + `fill` -- so a column read
    out of it identifies itself, and a reordering is visible."""
    cube = np.zeros((H, W, len(NAMES)), np.float32)
    for k in range(len(NAMES)):
        cube[:, :, k] = k + fill
    return cube


def _patch_tile_io(monkeypatch, mrral_mask, cube):
    """Make score_tile read a synthetic tile: load_tile returns `mrral_mask`,
    read_mrrsu_cube returns `cube`."""
    import data.mrrsu_bands as mb
    import scripts.classify_tile_supervised as cts

    data = np.zeros((H, W, 59), np.float32)
    monkeypatch.setattr(
        cts, 'load_tile',
        lambda p: (data, mrral_mask.copy(), TRANSFORM, _StubCRS()))
    monkeypatch.setattr(mb, 'read_mrrsu_cube', lambda p: (cube, list(NAMES)))
    monkeypatch.setattr(os.path, 'exists', lambda p: True)


def _rules_config():
    """A minimal but complete 7-class config: every class fires everywhere so
    the plumbing, not the mineralogy, is what the test observes."""
    return {
        'vocab': list(CLASSES_7),
        'junk': {'icer_high': None, 'co2_ice_high': None, 'var_high': None,
                 'r770_max': None},
        'classes': {
            'olivine': {'primary': {'param': 'OLINDEX3', 'threshold': -1e9},
                        'ladder': [[-1e9, 0.6]]},
            'lcp': {'primary': {'param': 'LCPINDEX2', 'threshold': -1e9},
                    'dominance_over': 'HCPINDEX2', 'ladder': [[-1e9, 0.6]]},
            'hcp': {'primary': {'param': 'HCPINDEX2', 'threshold': -1e9},
                    'dominance_over': 'LCPINDEX2', 'ladder': [[-1e9, 0.6]]},
            'plagioclase': {'primary': {'param': 'BD1300', 'threshold': -1e9},
                            'rpeak1_window': [-1e9, 1e9],
                            'hydration_veto': 1e9,
                            'ladder': [[-1e9, 0.6]]},
            'alteration': {'groups': [
                {'name': 'femg_phyllosilicate',
                 'requires': ['D2300', 'BD2290', 'BD1900R2'],
                 'thresholds': {'D2300': -1e9, 'BD2290': -1e9,
                                'BD1900R2': -1e9}}],
                'ladder': [[-1e9, 0.6]]},
        },
    }


# ── the npz contract ─────────────────────────────────────────────────────────

def test_payload_is_structurally_identical_to_a_model_written_npz(
        tmp_path, monkeypatch):
    """Reference is produced by the model's OWN writer, so this test runs on
    every machine rather than skipping wherever no floor test has been run."""
    import scripts.classify_tile_supervised as cts

    monkeypatch.setattr(cts, 'CLASS_NAMES', list(CLASSES_7))
    ref_path = str(tmp_path / 'ref_probs.npz')
    cts.save_probs(ref_path,
                   np.zeros((H, W, 7), np.float32),
                   np.ones((H, W), bool),
                   np.array([TRANSFORM.a, TRANSFORM.b, TRANSFORM.c,
                             TRANSFORM.d, TRANSFORM.e, TRANSFORM.f],
                            dtype=np.float64),
                   'PROJCS["stub"]')
    ref = np.load(ref_path, allow_pickle=True)

    mine_path = str(tmp_path / 'mine_probs.npz')
    np.savez_compressed(mine_path, **assemble_npz_payload(
        probs=np.zeros((H, W, 7), np.float32),
        valid_mask=np.ones((H, W), bool),
        transform_arr=np.arange(6, dtype=np.float64),
        crs_wkt='PROJCS["stub"]',
        class_names=list(CLASSES_7)))
    mine = np.load(mine_path, allow_pickle=True)

    assert set(mine.files) == set(ref.files), (
        f'keys differ: {sorted(mine.files)} vs {sorted(ref.files)}')
    for key in ref.files:
        assert mine[key].dtype == ref[key].dtype, (
            f'{key}: dtype {mine[key].dtype} != model dtype {ref[key].dtype}')
        assert mine[key].shape == ref[key].shape, (
            f'{key}: shape {mine[key].shape} != model shape {ref[key].shape}')
    assert mine['probs'].ndim == 3
    assert [str(x) for x in mine['class_names']] == list(CLASSES_7)


def test_payload_matches_any_real_npz_present_on_this_machine():
    """Belt-and-braces against a genuinely model-written file when one exists."""
    real = sorted(glob.glob('/tmp/floor_test_*/*/*_probs.npz'))
    real += sorted(glob.glob('reports/floor_tests/*/*/*_probs.npz'))
    if not real:
        pytest.skip('no reference probs npz on this machine')
    ref = np.load(real[0], allow_pickle=True)
    payload = assemble_npz_payload(
        probs=np.zeros((H, W, 7), np.float32),
        valid_mask=np.ones((H, W), bool),
        transform_arr=np.arange(6, dtype=np.float64),
        crs_wkt='PROJCS["x"]', class_names=list(CLASSES_7))
    assert set(payload) == set(ref.files), (
        f'keys differ: {sorted(payload)} vs {sorted(ref.files)}')
    assert payload['probs'].dtype == ref['probs'].dtype
    assert payload['valid_mask'].dtype == ref['valid_mask'].dtype
    assert payload['probs'].ndim == ref['probs'].ndim == 3


def test_rejects_a_vocabulary_the_vectorizer_would_not_accept():
    with pytest.raises(ValueError, match='class_names'):
        assemble_npz_payload(
            probs=np.zeros((2, 2, 2), np.float32),
            valid_mask=np.ones((2, 2), bool),
            transform_arr=np.arange(6, dtype=np.float64),
            crs_wkt='x', class_names=['mystery', 'vocab'])


def test_rejects_a_permuted_vocabulary():
    """Same names, wrong ORDER: every channel would be read as another mineral,
    which is exactly the failure that leaves no trace downstream."""
    permuted = list(CLASSES_7)
    permuted[1], permuted[2] = permuted[2], permuted[1]
    with pytest.raises(ValueError, match='class_names'):
        assemble_npz_payload(
            probs=np.zeros((2, 2, 7), np.float32),
            valid_mask=np.ones((2, 2), bool),
            transform_arr=np.arange(6, dtype=np.float64),
            crs_wkt='x', class_names=permuted)


def test_probs_channel_count_must_match_class_names():
    with pytest.raises(ValueError, match='channels'):
        assemble_npz_payload(
            probs=np.zeros((2, 2, 3), np.float32),
            valid_mask=np.ones((2, 2), bool),
            transform_arr=np.arange(6, dtype=np.float64),
            crs_wkt='x', class_names=list(CLASSES_7))


# ── valid_mask identity with the deep model ──────────────────────────────────

def test_valid_mask_is_load_tiles_mask_intersected_with_mrrsu(
        monkeypatch, tmp_path, capsys):
    """The mask must be load_tile's, not one re-derived from the mrrsu cube.

    The synthetic tile is built so the two disagree in BOTH directions, so a
    scorer that used mrrsu validity alone (or mrral validity alone) produces a
    different mask and fails here.
    """
    mrral_mask = np.ones((H, W), bool)
    mrral_mask[0, :] = False          # mrral says nodata, mrrsu says fine
    cube = _cube()
    cube[-1, :, :] = np.nan           # mrrsu says nodata, mrral says fine
    _patch_tile_io(monkeypatch, mrral_mask, cube)

    cfg_path = tmp_path / 'rules.json'
    cfg_path.write_text(json.dumps(_rules_config()))
    payload = score_tile('/nonexistent/t9999_mrral_00n000_0327_4.img',
                         str(cfg_path), model='rules')

    expected = mrral_mask.copy()
    expected[-1, :] = False
    np.testing.assert_array_equal(payload['valid_mask'], expected)
    assert payload['valid_mask'].sum() == (H - 2) * W

    out = capsys.readouterr().out
    assert 'mrral' in out and 'mrrsu' in out, (
        'both footprint counts must be PRINTED so a divergence is visible '
        f'in the log rather than absorbed; got: {out!r}')
    assert str((H - 1) * W) in out and str((H - 1) * W) in out


def test_invalid_pixels_are_zeroed_in_probs(monkeypatch, tmp_path):
    mrral_mask = np.ones((H, W), bool)
    mrral_mask[0, :] = False
    cube = _cube()
    _patch_tile_io(monkeypatch, mrral_mask, cube)
    cfg_path = tmp_path / 'rules.json'
    cfg_path.write_text(json.dumps(_rules_config()))
    payload = score_tile('/nonexistent/t9999_mrral_00n000_0327_4.img',
                         str(cfg_path), model='rules')
    assert payload['probs'][0].max() == 0.0
    assert payload['probs'][1:].max() > 0.0


def test_rules_scoring_prints_the_calibration_caveat(monkeypatch, tmp_path,
                                                     capsys):
    """Ladder precisions were measured over the LABELED-PIXEL population, whose
    positive base rate is far above a real tile's, so these probabilities read
    systematically optimistic against a model scored over whole tiles. Nobody
    should meet the summary tables without that line."""
    _patch_tile_io(monkeypatch, np.ones((H, W), bool), _cube())
    cfg_path = tmp_path / 'rules.json'
    cfg_path.write_text(json.dumps(_rules_config()))
    score_tile('/nonexistent/t9999_mrral_00n000_0327_4.img', str(cfg_path),
               model='rules')
    out = capsys.readouterr().out.lower()
    assert 'optimistic' in out and 'labeled' in out, (
        f'no calibration caveat printed; got: {out!r}')


# ── fitted feature order ─────────────────────────────────────────────────────

def test_feature_matrix_follows_the_fitted_column_order():
    """Band k of the cube holds the value k, so a column that came back in the
    wrong slot announces itself numerically."""
    cube = _cube(fill=0.0)
    cols = ['HCPINDEX2', 'R770', 'BD1300', 'OLINDEX3']
    X = assemble_feature_matrix(cube, list(NAMES), cols)
    assert X.shape == (H * W, len(cols))
    for j, c in enumerate(cols):
        assert np.allclose(X[:, j], NAMES.index(c)), (
            f'column {j} should be {c} (={NAMES.index(c)}) but holds {X[0, j]}')


def test_feature_matrix_raises_on_a_column_absent_from_the_tile():
    with pytest.raises(ValueError, match='NOT_A_PARAM'):
        assemble_feature_matrix(_cube(), list(NAMES), ['R770', 'NOT_A_PARAM'])


def test_ml_path_feeds_the_model_the_fitted_column_order(monkeypatch, tmp_path):
    """End-to-end through score_tile: the matrix handed to the predictor must
    be in meta.json's feature_cols order, not the tile's band order."""
    import joblib

    import scripts.fit_ml_baseline as fml

    cube = _cube(fill=0.0)
    _patch_tile_io(monkeypatch, np.ones((H, W), bool), cube)

    # deliberately NOT the tile's band order, and a strict subset
    cols = ['HCPINDEX2', 'R770', 'BD1300', 'OLINDEX3', 'D2300']
    art = tmp_path / 'ml'
    art.mkdir()
    (art / 'meta.json').write_text(json.dumps(
        {'vocab': list(CLASSES_7), 'feature_cols': cols, 'seed': 0}))

    seen = {}
    monkeypatch.setattr(joblib, 'load', lambda p: 'STUB_MODEL')

    def _fake_predict(model, X, n_classes):
        seen['model'] = model
        seen['X'] = np.asarray(X)
        return np.zeros((len(X), n_classes), np.float32)

    monkeypatch.setattr(fml, 'predict_proba_multilabel', _fake_predict)
    payload = score_tile('/nonexistent/t9999_mrral_00n000_0327_4.img',
                         str(art), model='rf')

    assert seen['model'] == 'STUB_MODEL'
    assert seen['X'].shape == (H * W, len(cols))
    for j, c in enumerate(cols):
        assert np.allclose(seen['X'][:, j], NAMES.index(c)), (
            f'model column {j} should be {c} but holds {seen["X"][0, j]}')
    assert payload['probs'].shape == (H, W, 7)
    assert [str(x) for x in payload['class_names']] == list(CLASSES_7)


def test_ckpt_is_accepted_and_ignored():
    """floor_test.sh always passes --ckpt; the baseline scorer must not choke
    on it, or the whole hook is unusable."""
    from scripts.classify_tile_baseline import build_parser
    args = build_parser().parse_args(
        ['--tile', 't.img', '--baseline', 'b.json', '--save_probs', 'o.npz',
         '--ckpt', '/some/checkpoint.pt', '--no_plot'])
    assert args.ckpt == '/some/checkpoint.pt'
    assert args.tile == 't.img'


def test_floor_test_hook_defaults_to_the_supervised_classifier():
    """With no CLASSIFY_CMD the shipped behaviour must be unchanged, and a
    forked copy of the script must not exist."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(here, 'scripts', 'floor_test.sh')).read()
    assert 'CLASSIFY_CMD:-$PYTHON scripts/classify_tile_supervised.py' in src
    assert '--ckpt "$CKPT"' in src
    # the checkpoint-existence guard still runs for callers without CLASSIFY_CMD
    assert 'ERROR: checkpoint not found' in src


def _run_classify_fragment(env_extra):
    """Execute the classify invocation EXACTLY as floor_test.sh writes it, with
    $PYTHON stubbed to `echo`, and return the command line it would run.

    Asserting on the resolved command rather than on the presence of a string
    tests what the hook DOES: that CLASSIFY_CMD replaces the program and that
    the shipped default is unchanged when it is unset.
    """
    import subprocess
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lines = open(os.path.join(here, 'scripts', 'floor_test.sh')).read().splitlines()
    # the classify invocation: the line naming --tile "$img", plus every
    # backslash-continued line above and below it
    anchor = next(i for i, ln in enumerate(lines) if '--tile "$img"' in ln)
    start = anchor
    while start > 0 and lines[start - 1].rstrip().endswith('\\'):
        start -= 1
    end = anchor
    while lines[end].rstrip().endswith('\\'):
        end += 1
    frag = '\n'.join(lines[start:end + 1])
    script = ('PYTHON=echo; img=IMG; CKPT=CKPT_PATH; npz=NPZ\n' + frag)
    env = dict(os.environ)
    env.pop('CLASSIFY_CMD', None)
    env.pop('CLASSIFY_EXTRA_ARGS', None)
    env.update(env_extra)
    out = subprocess.run(['bash', '-c', script], capture_output=True, text=True,
                         env=env, check=True)
    return ' '.join(out.stdout.split())


def test_hook_default_is_the_supervised_classifier_unchanged():
    assert _run_classify_fragment({}) == (
        'scripts/classify_tile_supervised.py --tile IMG --ckpt CKPT_PATH '
        '--save_probs NPZ --no_plot')


def test_hook_lets_a_baseline_replace_the_classifier_and_still_gets_ckpt():
    cmd = _run_classify_fragment(
        {'CLASSIFY_CMD': 'echo BASELINE --baseline cfg.json --model rules'})
    assert cmd == ('BASELINE --baseline cfg.json --model rules --tile IMG '
                   '--ckpt CKPT_PATH --save_probs NPZ --no_plot')


def test_floor_test_hook_is_the_only_copy_of_the_vectorization():
    """A forked floor_test / vectorizer would drift and silently stop being the
    same comparison."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scripts_dir = os.path.join(here, 'scripts')
    forks = [f for f in os.listdir(scripts_dir)
             if f.startswith('floor_test') and f != 'floor_test.sh']
    assert not forks, f'forked copies of floor_test.sh: {forks}'

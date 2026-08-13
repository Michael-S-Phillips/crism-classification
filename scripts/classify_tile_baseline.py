"""Score one tile with a baseline artifact and emit the standard probs npz.

The npz is the ONLY interface to the floor test, so a baseline written here runs
through floor_test.sh and the vectorizer completely unchanged -- same [0.50..0.99]
threshold ladder, same 3x3 median smoothing, same MIN_PIXELS, same summary
tables. Any difference in polygon counts is then attributable to the METHOD
rather than to the plumbing. That is why the payload is validated against the
vectorizer's accepted vocabularies here instead of being trusted: an npz with a
permuted or unknown class list loads happily and mislabels every polygon.

valid_mask is taken from classify_tile_supervised.load_tile (IMPORTED, not
reimplemented) so the baseline and the deep model mask identically. It is then
intersected with mrrsu validity and BOTH counts are printed: the two footprints
can differ, and a divergence must be visible in the log rather than absorbed
into the comparison as if it were a difference in method quality.

Usage:
    python scripts/classify_tile_baseline.py \\
        --tile .../t1250_mrral_20n078_0327_4.img \\
        --baseline config/expert_rules_7cls.json --model rules \\
        --save_probs /tmp/t1250_probs.npz --no_plot

    CLASSIFY_CMD="conda run -n crism python scripts/classify_tile_baseline.py \\
        --baseline config/expert_rules_7cls.json --model rules" \\
        bash scripts/floor_test.sh /dev/null rules_7cls
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.expert_rules import CLASSES_7, CLASSES_PYX  # noqa: E402

VALID_VOCABS = (CLASSES_7, CLASSES_PYX)

# Printed whenever the rule baseline scores a tile. The ladder precisions were
# measured on the LABELED-PIXEL population (Task 4), whose positive base rate is
# far above a real tile's, so they are not calibrated probabilities over a tile.
# This is inherent to train-split calibration, not a defect of this script.
RULES_CAVEAT = (
    'CAVEAT: rule "probabilities" are ladder precisions measured over the '
    'LABELED-PIXEL population, whose positive base rate is far higher than a '
    'whole tile\'s. Compared at 0.5/0.9/0.99 against a model scored over whole '
    'tiles, the rule baseline reads systematically OPTIMISTIC.')


def assemble_npz_payload(probs, valid_mask, transform_arr, crs_wkt,
                         class_names) -> dict:
    """Build the exact key/dtype set that classify_tile_supervised.save_probs
    writes and the vectorizer reads.

    Raises on any vocabulary the vectorizer would not accept -- including a
    PERMUTATION of an accepted one, which would otherwise read every channel as
    a different mineral without erroring anywhere.
    """
    names = [str(c) for c in class_names]
    if names not in [list(v) for v in VALID_VOCABS]:
        raise ValueError(
            f'class_names {names} is not a vocabulary the vectorizer accepts; '
            f'expected one of {[list(v) for v in VALID_VOCABS]}')
    probs = np.asarray(probs)
    if probs.ndim != 3:
        raise ValueError(f'probs must be (H, W, C); got shape {probs.shape}')
    if probs.shape[-1] != len(names):
        raise ValueError(
            f'probs has {probs.shape[-1]} channels but {len(names)} class_names')
    return {
        'probs': probs.astype(np.float32),
        'valid_mask': np.asarray(valid_mask).astype(bool),
        'transform': np.asarray(transform_arr, dtype=np.float64),
        'crs_wkt': np.array(str(crs_wkt)),
        'class_names': np.array(names),
    }


def assemble_feature_matrix(cube: np.ndarray, names: list[str],
                            feature_cols: list[str]) -> np.ndarray:
    """(H*W, len(feature_cols)) matrix in the FITTED column order.

    `feature_cols` is the column order the model was fitted on. Reshaping the
    cube and feeding it straight in would silently attach every threshold to
    the wrong parameter whenever the parquet column order differs from the
    tile's band order -- garbage output, no error. Columns are therefore
    resolved BY NAME, and a fitted column the tile does not carry is fatal.
    """
    names = list(names)
    missing = [c for c in feature_cols if c not in names]
    if missing:
        raise ValueError(
            f'the model was fitted on columns this tile does not carry: '
            f'{missing}; tile has {len(names)} bands')
    idx = [names.index(c) for c in feature_cols]
    flat = cube.reshape(-1, cube.shape[-1])
    return flat[:, idx]


def resolve_smooth(artifact_smooth: bool, requested: bool | None) -> bool:
    """Decide the smoothing state to score with: the artifact's training-time
    state always wins.

    `requested` is ``None`` when the caller passed no ``--smooth`` opinion, in
    which case the artifact's own recorded state is used silently -- that is
    the correct, non-guessing behaviour, not a gap. It only becomes a problem
    when `requested` is an EXPLICIT, contradicting choice: scoring with the
    wrong smoothing state feeds the model a feature distribution it was never
    fitted on, and nothing else here would catch that, so a real contradiction
    must raise loudly rather than quietly deferring to one side.
    """
    if requested is not None and bool(requested) != artifact_smooth:
        raise ValueError(
            f'--smooth={bool(requested)} contradicts the artifact, which '
            f'was fitted/calibrated with smooth={artifact_smooth}; scoring '
            f'with a different smoothing state than training would feed the '
            f'model a different feature distribution than it was fitted on, '
            f'with no other error to catch the mismatch')
    return artifact_smooth


def score_tile(tile: str, baseline: str, model: str = 'rules',
               smooth: bool | None = None) -> dict:
    """Score `tile` with `baseline` and return the npz payload.

    model='rules' reads a calibrated expert-rules JSON; 'rf'/'histgb' read an
    ML artifact directory (rf.joblib / histgb.joblib + meta.json).

    `smooth` is the CLI's ``--smooth`` opinion (``None`` if not passed). The
    smoothing state actually used to score always comes from the artifact's
    own recorded `smooth` -- fit_ml_baseline.py / fit_expert_rules.py stamp it
    in at train/calibration time -- and `smooth` here is used only to detect
    an explicit contradiction (see `resolve_smooth`). An artifact with no
    `smooth` key predates this change and is refused rather than guessed at.
    """
    import joblib

    import scripts.fit_ml_baseline as fml
    from data import mrrsu_bands
    from data.expert_rules import evaluate_rules
    from scripts import classify_tile_supervised as cts

    mrrsu_img = cts.derive_mrrsu_path(tile)
    if not os.path.exists(mrrsu_img):
        raise SystemExit(f'no co-registered mrrsu tile at {mrrsu_img}')

    # IMPORTED, never reimplemented: the deep model's mask is the reference.
    _data, valid_mask, transform, crs = cts.load_tile(tile)
    cube, names = mrrsu_bands.read_mrrsu_cube(mrrsu_img)
    if cube.shape[:2] != valid_mask.shape:
        raise SystemExit(
            f'mrrsu {cube.shape[:2]} is not co-registered with mrral '
            f'{valid_mask.shape}')

    if model == 'rules':
        with open(baseline) as f:
            cfg = json.load(f)
        if 'smooth' not in cfg:
            raise ValueError(
                f'{baseline} has no "smooth" key -- it was calibrated before '
                f'smoothing provenance was recorded; recalibrate with '
                f'scripts/fit_expert_rules.py rather than guessing whether '
                f'it was smoothed')
        artifact_smooth = bool(cfg['smooth'])
    else:
        with open(os.path.join(baseline, 'meta.json')) as f:
            meta = json.load(f)
        if 'smooth' not in meta:
            raise ValueError(
                f'{baseline}/meta.json has no "smooth" key -- it was fitted '
                f'before smoothing provenance was recorded; refit with '
                f'scripts/fit_ml_baseline.py rather than guessing whether it '
                f'was smoothed')
        artifact_smooth = bool(meta['smooth'])

    effective_smooth = resolve_smooth(artifact_smooth, smooth)
    if effective_smooth:
        from scripts.extract_mrrsu_features import _smooth_nanmean
        cube = _smooth_nanmean(cube)

    mrrsu_valid = np.isfinite(cube).any(axis=-1)
    combined = valid_mask & mrrsu_valid
    print(f'valid pixels - mrral {int(valid_mask.sum()):,}, '
          f'mrrsu {int(mrrsu_valid.sum()):,}, both {int(combined.sum()):,} '
          f'(mrral-only {int((valid_mask & ~mrrsu_valid).sum()):,}, '
          f'mrrsu-only {int((mrrsu_valid & ~valid_mask).sum()):,})')

    if model == 'rules':
        print(RULES_CAVEAT)
        vocab = list(cfg['vocab'])
        scores = evaluate_rules(cube, names, cfg)
        probs = np.stack([scores[c] for c in vocab], axis=-1)
    else:
        vocab = list(meta['vocab'])
        feature_cols = list(meta['feature_cols'])
        art = joblib.load(os.path.join(baseline, f'{model}.joblib'))
        X = assemble_feature_matrix(cube, names, feature_cols)
        H, W = cube.shape[:2]
        p = fml.predict_proba_multilabel(art, X, len(vocab))
        probs = np.asarray(p, np.float32).reshape(H, W, len(vocab))

    probs = np.asarray(probs, np.float32).copy()
    probs[~combined] = 0.0
    transform_arr = np.array(
        [transform.a, transform.b, transform.c,
         transform.d, transform.e, transform.f], dtype=np.float64)
    return assemble_npz_payload(probs, combined, transform_arr,
                                crs.to_wkt() if crs else '', vocab)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tile', required=True, help='mrral .img')
    ap.add_argument('--baseline', required=True,
                    help='expert rules .json, or an ML --out_dir')
    ap.add_argument('--model', choices=('rules', 'rf', 'histgb'),
                    default='rules')
    ap.add_argument('--save_probs', required=True)
    ap.add_argument('--smooth', action='store_true', default=None,
                    help='7x7 NaN-aware mean before scoring. Normally not '
                         'needed: the artifact records whether it was fitted '
                         'on smoothed features and that state is used '
                         'automatically. Passing this explicitly only serves '
                         'as an assertion -- it raises if it contradicts the '
                         'artifact, rather than silently overriding it.')
    ap.add_argument('--no_plot', action='store_true',
                    help='accepted and ignored; this scorer never plots')
    ap.add_argument('--ckpt', default=None,
                    help='accepted and ignored; floor_test.sh always passes it')
    return ap


def main() -> None:
    args = build_parser().parse_args()
    payload = score_tile(args.tile, args.baseline, model=args.model,
                         smooth=args.smooth)
    out_dir = os.path.dirname(os.path.abspath(args.save_probs))
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(args.save_probs, **payload)
    print(f'wrote {args.save_probs}')


if __name__ == '__main__':
    main()

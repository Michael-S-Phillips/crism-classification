"""Classical ML baselines on the 60 mrrsu summary parameters.

Middle rung of the floor test: expert rules (Tasks 3-4) -> classical ML on
expert-designed features (this) -> the deep spatial-spectral model. Two
models, trained the same way and compared the same way:

RandomForest is multi-output natively, so it fits all classes in one call and
never forces exclusivity across them. HistGradientBoosting is single-target,
so it is run ONE-VS-REST per class -- one independent binary model per class,
never an argmax across classes. Both matter for the same reason: a pixel can
be olivine AND hcp (olivine-bearing basalt is ordinary), and the whole expert
ruleset this rung is compared against was built around not forcing
exclusivity. An argmax anywhere here would silently destroy that
co-occurrence and make the ML rung non-comparable with the rules or the deep
model.

HistGB is the second rung specifically because it consumes NaN natively.
mrrsu carries 65535 nodata, which Task 1 converts to NaN; RandomForest cannot
consume NaN, so it needs an imputer. That imputer is fitted on TRAIN ONLY and
frozen into the artifact (``rf.joblib``'s ``impute_median``) -- it is never
recomputed from whatever batch is being scored, because a batch's own median
is a different feature distribution than the one RF trained on, and that
mismatch would not raise an error anywhere; it would just quietly change the
model's answers.

Usage
-----
    python scripts/fit_ml_baseline.py \\
        --features data/mrrsu_features.parquet \\
        --labels   data/mrral_pixels_with_review_v2.parquet \\
        --vocab 7cls --out_dir config/ml_baseline_7cls
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Identity/bookkeeping columns carried by the Task-2 sidecar and the labeled
# parquet; everything else is a real mrrsu parameter name.
KEY_COLS = ('tile_id', 'pixel_row', 'pixel_col', 'split')


def fit_models(X: np.ndarray, Y: np.ndarray, seed: int = 0) -> dict:
    """Fit both classical-ML rungs on (X, Y). Rows are pixels; Y columns are
    independent binary class targets (multi-label, not multi-class).

    Returns ``{'rf': {'model': ..., 'impute_median': ...}, 'histgb': [...]}``.
    The histgb list has one fitted one-vs-rest binary model per class, or
    ``None`` where that class had zero positives in Y (or zero negatives --
    either way there is nothing for a binary classifier to learn, and it must
    not crash the whole fit).
    """
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  RandomForestClassifier)

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y).astype(int)

    # RF cannot consume NaN. The fill value is the per-column median of
    # EXACTLY the rows passed in here (the caller's train split), computed
    # once and carried in the artifact alongside the fitted forest -- so
    # scoring always applies the same fill this forest was trained against.
    impute_median = np.nanmedian(X, axis=0)
    # A column that is all-NaN in train has no median (nan); fall back to 0
    # so the fill itself never introduces a NaN into RF's input.
    impute_median = np.where(np.isfinite(impute_median), impute_median, 0.0)
    X_rf = np.where(np.isfinite(X), X, impute_median)

    rf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    rf.fit(X_rf, Y)

    hist_models: list = []
    for j in range(Y.shape[1]):
        col = Y[:, j]
        if np.unique(col).size < 2:
            # No positives (or no negatives) for this class: a binary
            # classifier has nothing to learn. Leave it inert rather than
            # raise; predict_proba_multilabel reads None as "predict 0".
            hist_models.append(None)
            continue
        m = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
        m.fit(X, col)   # NaN handled natively -- no imputation here
        hist_models.append(m)

    return {
        'rf': {'model': rf, 'impute_median': impute_median},
        'histgb': hist_models,
    }


def predict_proba_multilabel(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    """Independent per-class positive-class probabilities, shape (N, n_classes).

    Never an argmax across the class axis: co-occurring labels must both come
    back > 0.5 when the model actually learned them, because a real pixel can
    be positive for more than one class at once.
    """
    X = np.asarray(X, dtype=np.float64)
    out = np.zeros((len(X), n_classes), dtype=np.float32)

    if isinstance(model, dict) and 'impute_median' in model:
        # RF path. Fill with the median FROZEN at fit time -- never
        # recomputed from this batch, which may have a different (or
        # entirely-NaN) distribution than train did.
        med = model['impute_median']
        Xr = np.where(np.isfinite(X), X, med)
        proba = model['model'].predict_proba(Xr)
        # Multi-output RandomForestClassifier.predict_proba returns a LIST
        # of per-output arrays. An output with zero positives in training
        # has classes_ == [0], so its array is (N, 1) of P(class 0) -- there
        # is no positive-class column to read there, and the correct
        # probability of that (absent) positive class is 0.
        for j, p in enumerate(proba):
            if j >= n_classes:
                break
            out[:, j] = p[:, 1] if p.shape[1] == 2 else 0.0
        return out

    # HistGB path: one one-vs-rest binary model per class, or None where the
    # class had nothing to learn from.
    for j, m in enumerate(model):
        if j >= n_classes or m is None:
            continue
        out[:, j] = m.predict_proba(X)[:, 1]
    return out


def fit_from_frames(feat: pd.DataFrame, lab: pd.DataFrame, vocab: list[str],
                    seed: int = 0, split: str = 'train') -> dict:
    """Select `split` from a row-aligned (features, labels) pair and fit.

    Train on the TRAIN split only: fitting on val or test is leakage the
    floor test exists to rule out, and it leaves no trace in the artifact.

    Returns ``{'models': fit_models(...), 'feature_cols': [...], 'vocab': [...]}``
    -- `feature_cols` is the exact column order X was fitted on, and `vocab`
    is the subset of the requested vocabulary actually present in `lab`
    (same order).
    """
    if len(feat) != len(lab):
        raise ValueError(
            f'feature rows {len(feat):,} != label rows {len(lab):,} -- the '
            f'sidecar is not aligned with the labels')
    feat = feat.reset_index(drop=True)
    lab = lab.reset_index(drop=True)
    if 'split' not in lab.columns:
        raise ValueError('the label frame has no "split" column; refusing '
                         'to fit on an unknown mixture of splits')
    mask = (lab['split'] == split).to_numpy()
    if not mask.any():
        raise ValueError(f'no rows with split == {split!r}')

    feature_cols = [c for c in feat.columns if c not in KEY_COLS]
    X = feat.loc[mask, feature_cols].to_numpy(np.float64)
    present = [c for c in vocab if c in lab.columns]
    Y = (lab.loc[mask, present].to_numpy(float) > 0.4).astype(int)

    models = fit_models(X, Y, seed=seed)
    return {'models': models, 'feature_cols': feature_cols, 'vocab': present}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--features', required=True, help='Task-2 sidecar parquet')
    ap.add_argument('--labels', required=True,
                    help='labeled parquet, row-aligned with --features, '
                         'carrying a "split" column')
    ap.add_argument('--vocab', choices=('7cls', 'pyx'), default='7cls')
    ap.add_argument('--split', default='train',
                    help='training split; anything but "train" is leakage')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    import joblib

    from data.dataset import _collapse_labels
    from data.expert_rules import CLASSES_7, CLASSES_PYX

    feat = pd.read_parquet(args.features)
    lab = _collapse_labels(pd.read_parquet(args.labels))
    vocab = CLASSES_7 if args.vocab == '7cls' else CLASSES_PYX

    out = fit_from_frames(feat, lab, vocab, seed=args.seed, split=args.split)
    n_rows = int((lab['split'] == args.split).sum())
    print(f'fit on {n_rows:,} {args.split.upper()} rows '
          f'({len(out["feature_cols"])} features, {len(out["vocab"])} classes)')

    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(out['models']['rf'], os.path.join(args.out_dir, 'rf.joblib'))
    joblib.dump(out['models']['histgb'],
               os.path.join(args.out_dir, 'histgb.joblib'))
    meta = {
        'vocab': out['vocab'],
        'feature_cols': out['feature_cols'],
        'seed': args.seed,
        'models': ['rf', 'histgb'],
    }
    with open(os.path.join(args.out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
        f.write('\n')
    print(f'wrote {args.out_dir}/{{rf,histgb}}.joblib + meta.json')


if __name__ == '__main__':
    main()

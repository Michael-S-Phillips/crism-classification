"""Tests for the classical-ML floor-test rung (RandomForest + HistGB) on the
60 mrrsu summary parameters.

MULTI-LABEL is the load-bearing property here: a pixel can legitimately be
olivine AND hcp (olivine-bearing basalt), so nothing in this file may collapse
to an argmax across classes -- RandomForest is multi-output natively, and
HistGB (single-target) is run one-vs-rest per class specifically to preserve
that.

HistGB was chosen as the second rung because it consumes NaN natively (mrrsu's
65535 nodata -> NaN). RandomForest cannot, so it needs an imputer -- but that
imputer must be fitted on TRAIN and frozen into the artifact, never
recomputed from whatever batch happens to be scored, or the model sees a
different feature distribution at inference than it trained on.
"""
import numpy as np
import pandas as pd
import pytest

from scripts.fit_ml_baseline import (fit_from_frames, fit_models,
                                     predict_proba_multilabel)


def _separable(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 6)).astype(np.float32)
    Y = np.zeros((n, 3), dtype=int)
    Y[:, 0] = X[:, 0] > 0.6
    Y[:, 1] = X[:, 1] > 0.6
    Y[:, 2] = (X[:, 0] > 0.6) & (X[:, 1] > 0.6)   # genuine co-occurrence
    return X, Y


def test_both_models_learn_a_separable_multilabel_problem():
    X, Y = _separable()
    models = fit_models(X, Y, seed=0)
    for name, m in models.items():
        P = predict_proba_multilabel(m, X, Y.shape[1])
        assert P.shape == (len(X), 3), f'{name} wrong shape {P.shape}'
        assert ((P >= 0) & (P <= 1)).all(), f'{name} produced non-probabilities'
        acc = ((P > 0.5).astype(int) == Y).mean()
        assert acc > 0.9, f'{name} accuracy {acc:.2f} on a separable problem'


def test_co_occurring_labels_are_both_predicted():
    """Multi-label, not multi-class: a pixel positive for two classes must get
    both, not argmax."""
    X, Y = _separable()
    both = np.flatnonzero(Y[:, 2] == 1)[:20]
    models = fit_models(X, Y, seed=0)
    for name, m in models.items():
        P = predict_proba_multilabel(m, X[both], Y.shape[1])
        assert (P[:, 0] > 0.5).mean() > 0.8, f'{name} dropped class 0'
        assert (P[:, 2] > 0.5).mean() > 0.8, f'{name} dropped class 2'


def test_nan_features_do_not_crash_histgb():
    """mrrsu carries 65535 nodata -> NaN. HistGB handles NaN natively; that is
    why it was chosen over an imputer that would bias the comparison."""
    X, Y = _separable()
    X = X.copy(); X[::10, 0] = np.nan
    models = fit_models(X, Y, seed=0)
    P = predict_proba_multilabel(models['histgb'], X, Y.shape[1])
    assert np.isfinite(P).all()


def test_class_with_zero_positives_does_not_crash_and_predicts_zero():
    """A class absent from the training rows (an mc quadrant with no junk
    pixels, say) must not crash either model's fit, and must come out
    predicting 0 -- not 1, which is what naively reading a single-class
    predict_proba column would give."""
    X, Y = _separable()
    Y = Y.copy()
    Y[:, 1] = 0   # class 1 has zero positives anywhere in training
    models = fit_models(X, Y, seed=0)
    for name, m in models.items():
        P = predict_proba_multilabel(m, X, Y.shape[1])
        assert np.isfinite(P).all(), f'{name} produced non-finite output'
        assert (P[:, 1] < 0.5).all(), (
            f'{name} predicted the wholly-absent class as positive')
        # the two real classes must still work fine
        acc0 = ((P[:, 0] > 0.5).astype(int) == Y[:, 0]).mean()
        assert acc0 > 0.9, f'{name} class 0 accuracy {acc0:.2f} degraded'


def test_rf_imputation_is_frozen_at_fit_time_not_recomputed_at_predict():
    """RF's median fill must come from the artifact recorded at TRAIN time.
    Recomputing it from whatever is being scored is a silent-failure path:
    the model would see a different feature distribution at inference than it
    trained on, with no error anywhere."""
    X, Y = _separable()
    models = fit_models(X, Y, seed=0)
    rf = models['rf']
    train_median_col0 = np.nanmedian(X[:, 0])
    assert train_median_col0 < 0.6, 'test setup assumption violated'

    # A predict-time batch whose OWN column-0 median sits on the opposite
    # side of the 0.6 decision boundary from the train median, with half its
    # rows NaN'd out. If the fill is frozen at train time, the NaN rows get
    # the train median (~0.5, class-0 negative). If it is recomputed from
    # this batch instead (the bug), they get ~0.9 (class-0 positive).
    n = 60
    rng = np.random.default_rng(1)
    X_batch = rng.random((n, 6)).astype(np.float32)
    X_batch[:, 0] = 0.90 + rng.random(n) * 0.05
    X_batch[:, 1] = 0.10                      # keep other classes quiet
    nan_rows = np.zeros(n, dtype=bool)
    nan_rows[::2] = True
    X_batch[nan_rows, 0] = np.nan

    P = predict_proba_multilabel(rf, X_batch, Y.shape[1])
    assert (P[nan_rows, 0] < 0.5).mean() > 0.8, (
        'NaN rows read as class-0 positive -- imputation used this batch\'s '
        'own median instead of the frozen train-time median')


def test_only_train_rows_influence_the_fit():
    """Fitting on a frame that adds val/test rows must produce identical
    models to fitting on the train rows alone -- otherwise scores leak across
    the split boundary the floor test depends on being clean."""
    X, Y = _separable(n=300, seed=2)
    cols = [f'f{i}' for i in range(X.shape[1])]
    vocab = ['olivine', 'pyx', 'both']
    feat = pd.DataFrame(X, columns=cols)
    lab = pd.DataFrame(Y, columns=vocab)
    lab['split'] = 'train'

    # Poison rows: inverted features AND inverted labels, so if they leak
    # into the fit the resulting model changes in a way we can detect.
    poison_feat = pd.DataFrame(1.0 - X, columns=cols)
    poison_lab = pd.DataFrame(1 - Y, columns=vocab)
    poison_lab['split'] = 'val'

    all_feat = pd.concat([feat, poison_feat], ignore_index=True)
    all_lab = pd.concat([lab, poison_lab], ignore_index=True)

    out_train = fit_from_frames(feat, lab, vocab, seed=0)
    out_all = fit_from_frames(all_feat, all_lab, vocab, seed=0)

    P_train = predict_proba_multilabel(out_train['models']['rf'], X, 3)
    P_all = predict_proba_multilabel(out_all['models']['rf'], X, 3)
    assert np.allclose(P_train, P_all), (
        'val rows changed the RF fit -- the train-split filter leaks')

    Ph_train = predict_proba_multilabel(out_train['models']['histgb'], X, 3)
    Ph_all = predict_proba_multilabel(out_all['models']['histgb'], X, 3)
    assert np.allclose(Ph_train, Ph_all), (
        'val rows changed the HistGB fit -- the train-split filter leaks')


def test_feature_cols_records_the_fitted_column_order():
    """meta's feature_cols must be the exact order X's columns were fitted
    in -- Task 6 rebuilds the feature vector from this list, and a reordered
    vector at inference produces garbage with no error."""
    X, Y = _separable(n=200, seed=3)
    cols = ['zeta', 'alpha', 'omega', 'beta', 'gamma', 'delta']  # NOT sorted
    vocab = ['olivine', 'pyx', 'both']
    feat = pd.DataFrame(X, columns=cols)
    feat['tile_id'] = 't0001'
    feat['pixel_row'] = 0
    feat['pixel_col'] = 0
    lab = pd.DataFrame(Y, columns=vocab)
    lab['split'] = 'train'

    out = fit_from_frames(feat, lab, vocab, seed=0)
    assert out['feature_cols'] == cols, (
        f'feature_cols {out["feature_cols"]} does not match the column order '
        f'passed in ({cols})')

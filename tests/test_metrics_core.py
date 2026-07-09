"""
Tests for compute_map's `exclude` argument (core stop metric, junk excluded).

Task D of 2026-07-08-unit-balanced-splits: `val_mAP_core` = mAP excluding the
junk class. For a 7-column score matrix, evaluation.metrics._class_names
resolves names via data.dataset.LABEL_COLS_7CLASS, where 'junk' is the last
class (index 6).
"""
import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from data.dataset import LABEL_COLS_7CLASS
from evaluation.metrics import compute_map

N_CLASSES = 7
JUNK_IDX = LABEL_COLS_7CLASS.index('junk')


@pytest.fixture(scope='module')
def synthetic_7cls():
    """7-column y_true/y_score: 6 well-predicted classes, garbage junk column.

    The junk column's scores are anti-correlated with its labels, so its AP is
    near zero and drags the full mAP down.
    """
    assert JUNK_IDX == N_CLASSES - 1  # junk is the last 7-class label
    rng = np.random.default_rng(42)
    n = 400
    y_true = (rng.random((n, N_CLASSES)) < 0.3).astype(np.float32)
    # Every class needs at least one positive so no class is skipped.
    assert (y_true.sum(axis=0) > 0).all()

    noise = rng.normal(0.0, 0.1, size=(n, N_CLASSES))
    y_score = np.clip(0.8 * y_true + 0.1 + noise, 0.0, 1.0)
    # Junk column: anti-correlated (score high where label is 0).
    y_score[:, JUNK_IDX] = np.clip(
        0.8 * (1.0 - y_true[:, JUNK_IDX]) + 0.1 + noise[:, JUNK_IDX], 0.0, 1.0
    )
    return y_true, y_score


def test_exclude_junk_raises_map(synthetic_7cls):
    y_true, y_score = synthetic_7cls
    full = compute_map(y_true, y_score)
    core = compute_map(y_true, y_score, exclude=('junk',))
    assert core > full

    # core must equal the mean over the 6 non-junk columns
    ref = np.mean([
        average_precision_score(
            (y_true[:, i] > 0.4).astype(int), y_score[:, i])
        for i in range(N_CLASSES) if i != JUNK_IDX
    ])
    assert core == pytest.approx(float(ref))


def test_exclude_absent_name_is_noop(synthetic_7cls):
    y_true, y_score = synthetic_7cls
    assert 'notaclass' not in LABEL_COLS_7CLASS
    assert compute_map(y_true, y_score, exclude=('notaclass',)) == \
        compute_map(y_true, y_score)


def test_default_behavior_unchanged(synthetic_7cls):
    """No exclude arg → same result as the pre-change compute_map: the mean
    AP over every column with positives."""
    y_true, y_score = synthetic_7cls
    ref = np.mean([
        average_precision_score(
            (y_true[:, i] > 0.4).astype(int), y_score[:, i])
        for i in range(N_CLASSES)
    ])
    assert compute_map(y_true, y_score) == pytest.approx(float(ref))
    # Explicit empty exclude is identical to omitting it.
    assert compute_map(y_true, y_score, exclude=()) == \
        compute_map(y_true, y_score)

"""
Evaluation metrics for multi-label mineral classification.
All functions accept numpy arrays.
"""
import logging
from typing import Dict, List
import numpy as np
from sklearn.metrics import average_precision_score

import data.dataset

logger = logging.getLogger(__name__)


def _class_names(n_classes: int) -> List[str]:
    """Class names for an n-class score matrix, read at CALL time so the
    train.py / eval-script LABEL_COLS rebind (6-class / 7-class mode) is
    honored even when this module was imported before the rebind."""
    cols = data.dataset.LABEL_COLS
    if n_classes > len(cols):
        # Fallback ladder: try 6-class list, then 7-class list
        cols = data.dataset.LABEL_COLS_WITH_ALTERATION
    if n_classes > len(cols):
        cols = data.dataset.LABEL_COLS_7CLASS
    return list(cols[:n_classes])


def compute_map(y_true: np.ndarray, y_score: np.ndarray,
                exclude: tuple = ()) -> float:
    """Mean Average Precision across all classes. Skips classes with no positives.

    exclude: class names (resolved via ``_class_names(y_score.shape[1])``)
    dropped from the mean, e.g. ``exclude=('junk',)`` for the core stop metric.
    Names not in the resolved label set are ignored (no-op). Default () keeps
    the historical behavior.
    """
    class_names = _class_names(y_score.shape[1]) if exclude else []
    aps = []
    for i in range(y_true.shape[1]):
        if i < len(class_names) and class_names[i] in exclude:
            continue
        if y_true[:, i].sum() > 0:
            aps.append(average_precision_score(
                (y_true[:, i] > 0.4).astype(int), y_score[:, i]
            ))
    return float(np.mean(aps)) if aps else 0.0


def compute_per_class_ap(
    y_true: np.ndarray,
    y_score: np.ndarray
) -> Dict[str, float]:
    """Per-class Average Precision. Returns dict keyed by class name."""
    n_classes = y_score.shape[1]
    class_names = _class_names(n_classes)
    result = {}
    for i, cls in enumerate(class_names):
        if y_true[:, i].sum() > 0:
            result[cls] = float(average_precision_score(
                (y_true[:, i] > 0.4).astype(int), y_score[:, i]
            ))
        else:
            logger.warning("AP[%s]: no positive val examples — reporting 0.0", cls)
            result[cls] = 0.0
    return result


def compute_metrics_by_confidence_tier(
    y_true: np.ndarray,
    y_score: np.ndarray,
    confidence_tiers: List[str],
) -> Dict[str, Dict]:
    """
    Compute mAP and per-class AP broken out by confidence tier.

    Parameters
    ----------
    y_true : (n, 6)
    y_score : (n, 6)
    confidence_tiers : list of str, length n, values in {'High','Moderate','Low'}
    """
    tiers = np.array(confidence_tiers)
    result = {}
    for tier in ['High', 'Moderate', 'Low']:
        mask = tiers == tier
        if mask.sum() == 0:
            result[tier] = {'mAP': float('nan')}
            continue
        result[tier] = {
            'mAP': compute_map(y_true[mask], y_score[mask]),
            'per_class_ap': compute_per_class_ap(y_true[mask], y_score[mask]),
            'n_pixels': int(mask.sum()),
        }
    return result


def compute_full_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    confidence_tiers: List[str],
) -> Dict:
    """Convenience wrapper: overall + per-class + confidence-tier metrics."""
    return {
        'mAP': compute_map(y_true, y_score),
        'per_class_ap': compute_per_class_ap(y_true, y_score),
        'by_confidence': compute_metrics_by_confidence_tier(
            y_true, y_score, confidence_tiers
        ),
    }

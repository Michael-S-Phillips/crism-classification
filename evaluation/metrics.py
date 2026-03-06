"""
Evaluation metrics for multi-label mineral classification.
All functions accept numpy arrays.
"""
from typing import Dict, List
import numpy as np
from sklearn.metrics import average_precision_score

from data.label_parser import CLASSES


def compute_map(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mean Average Precision across all 6 classes. Skips classes with no positives."""
    aps = []
    for i in range(y_true.shape[1]):
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
    result = {}
    for i, cls in enumerate(CLASSES):
        if y_true[:, i].sum() > 0:
            result[cls] = float(average_precision_score(
                (y_true[:, i] > 0.4).astype(int), y_score[:, i]
            ))
        else:
            result[cls] = float('nan')
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

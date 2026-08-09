import torch
import numpy as np
import pytest
from training.losses import WeightedBCEWithLogitsLoss
from evaluation.metrics import (
    compute_map, compute_per_class_ap, compute_metrics_by_confidence_tier
)

def test_weighted_loss_shape():
    loss_fn = WeightedBCEWithLogitsLoss()
    logits = torch.randn(8, 6)
    targets = torch.randint(0, 2, (8, 6)).float()
    weights = torch.ones(8)
    loss = loss_fn(logits, targets, weights)
    assert loss.shape == ()  # scalar
    assert loss.item() > 0

def test_high_weight_increases_loss():
    loss_fn = WeightedBCEWithLogitsLoss()
    # 2 wrong predictions (logit=0, target=1, BCE≈0.693) mixed with
    # 2 near-perfect predictions (logit=10, target=1, BCE≈0.00005).
    # Weighting wrong samples high vs. low changes the weighted mean.
    logits = torch.tensor([[0., 0.], [0., 0.], [10., 10.], [10., 10.]])
    targets = torch.ones(4, 2)
    w_wrong_high = torch.tensor([1.0, 1.0, 0.25, 0.25])  # bad samples weighted high
    w_wrong_low = torch.tensor([0.25, 0.25, 1.0, 1.0])   # good samples weighted high
    assert loss_fn(logits, targets, w_wrong_high) > loss_fn(logits, targets, w_wrong_low)

def test_compute_map_perfect():
    y_true = np.eye(6, dtype=np.float32)
    y_score = np.eye(6, dtype=np.float32)
    mAP = compute_map(y_true, y_score)
    assert mAP == pytest.approx(1.0)

def test_compute_map_range():
    y_true = np.random.randint(0, 2, (100, 6)).astype(np.float32)
    y_score = np.random.rand(100, 6).astype(np.float32)
    mAP = compute_map(y_true, y_score)
    assert 0.0 <= mAP <= 1.0

def test_per_class_ap_keys():
    from data.dataset import LABEL_COLS
    n = len(LABEL_COLS)
    y_true = np.random.randint(0, 2, (50, n)).astype(np.float32)
    y_score = np.random.rand(50, n).astype(np.float32)
    ap_dict = compute_per_class_ap(y_true, y_score)
    assert set(ap_dict.keys()) == set(LABEL_COLS)

def test_metrics_by_confidence_tier():
    y_true = np.random.randint(0, 2, (60, 6)).astype(np.float32)
    y_score = np.random.rand(60, 6).astype(np.float32)
    tiers = ['High'] * 20 + ['Moderate'] * 20 + ['Low'] * 20
    result = compute_metrics_by_confidence_tier(y_true, y_score, tiers)
    assert set(result.keys()) == {'High', 'Moderate', 'Low'}
    for tier_metrics in result.values():
        assert 'mAP' in tier_metrics
        assert 0.0 <= tier_metrics['mAP'] <= 1.0


def test_metrics_by_confidence_tier_reports_review_tiers():
    """Review-derived tiers must appear in their own bucket, not vanish.

    The tier list used to be hardcoded to the three base-parquet tiers, so
    every 'Reviewed-*' row — over a third of the 7-class build once
    Reviewed-Legacy was introduced — was reported in no bucket at all.
    """
    import numpy as np
    from evaluation.metrics import compute_metrics_by_confidence_tier

    y_true = np.random.randint(0, 2, (80, 6)).astype(np.float32)
    y_score = np.random.rand(80, 6).astype(np.float32)
    tiers = (['High'] * 20 + ['Moderate'] * 20
             + ['Reviewed-Legacy'] * 20 + ['Reviewed-High'] * 20)
    result = compute_metrics_by_confidence_tier(y_true, y_score, tiers)

    # Base tiers always present, even 'Low' which has no rows here.
    assert {'High', 'Moderate', 'Low'} <= set(result)
    assert np.isnan(result['Low']['mAP'])
    # Review tiers discovered from the data.
    assert 'Reviewed-Legacy' in result and 'Reviewed-High' in result
    assert result['Reviewed-Legacy']['n_pixels'] == 20
    assert result['Reviewed-High']['n_pixels'] == 20
    # Every row lands in exactly one bucket.
    assert sum(v['n_pixels'] for v in result.values() if 'n_pixels' in v) == 80

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.train_sklearn import train_and_evaluate_sklearn

def make_fake_data(n=200, n_features=60, n_classes=6):
    X = np.random.rand(n, n_features).astype(np.float32)
    y = (np.random.rand(n, n_classes) > 0.7).astype(np.float32)
    w = np.ones(n, dtype=np.float32)
    return X, y, w

def test_logreg_returns_metrics():
    X_tr, y_tr, w_tr = make_fake_data()
    X_v, y_v, w_v = make_fake_data(50)
    metrics = train_and_evaluate_sklearn(
        'logreg', X_tr, y_tr, w_tr, X_v, y_v, w_v,
        use_wandb=False
    )
    assert 'val_mAP' in metrics
    assert 0.0 <= metrics['val_mAP'] <= 1.0

def test_rf_returns_metrics():
    X_tr, y_tr, w_tr = make_fake_data()
    X_v, y_v, w_v = make_fake_data(50)
    metrics = train_and_evaluate_sklearn(
        'rf', X_tr, y_tr, w_tr, X_v, y_v, w_v,
        use_wandb=False, n_estimators=10
    )
    assert 'val_mAP' in metrics

def test_xgb_returns_metrics():
    X_tr, y_tr, w_tr = make_fake_data()
    X_v, y_v, w_v = make_fake_data(50)
    metrics = train_and_evaluate_sklearn(
        'xgb', X_tr, y_tr, w_tr, X_v, y_v, w_v,
        use_wandb=False, n_estimators=10
    )
    assert 'val_mAP' in metrics

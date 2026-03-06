"""
Training and evaluation for sklearn-compatible models.
Supports: logreg, svc, rf, xgb, lgbm
"""
import os, sys, pickle, logging
from typing import Dict, Any
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metrics import compute_full_metrics
from data.label_parser import CLASSES

logger = logging.getLogger(__name__)


def _build_model(model_type: str, **kwargs):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier

    if model_type == 'logreg':
        base = LogisticRegression(max_iter=1000, C=kwargs.get('C', 1.0))
        return MultiOutputClassifier(base)
    elif model_type == 'svc':
        base = LinearSVC(max_iter=2000, C=kwargs.get('C', 1.0))
        return MultiOutputClassifier(base)
    elif model_type == 'rf':
        return RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 200),
            max_depth=kwargs.get('max_depth', None),
            n_jobs=-1, random_state=42
        )
    elif model_type == 'xgb':
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=kwargs.get('n_estimators', 200),
            max_depth=kwargs.get('max_depth', 6),
            learning_rate=kwargs.get('learning_rate', 0.1),
            eval_metric='logloss',
            tree_method='hist',
            random_state=42
        )
    elif model_type == 'lgbm':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=kwargs.get('n_estimators', 200),
            max_depth=kwargs.get('max_depth', -1),
            learning_rate=kwargs.get('learning_rate', 0.1),
            random_state=42, n_jobs=-1, verbose=-1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_and_evaluate_sklearn(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    w_val: np.ndarray,
    confidence_tiers_val: list = None,
    use_wandb: bool = True,
    checkpoint_dir: str = None,
    **model_kwargs
) -> Dict[str, Any]:
    """
    Train a sklearn model and evaluate on validation set.

    For multi-label targets (y shape n x 6), XGB/LGBM train one model per class.
    LogReg/SVC/RF use MultiOutputClassifier.
    """
    if use_wandb:
        import wandb
        wandb.init(
            project='crism-mineral-classification',
            name=f'{model_type}',
            config={'model': model_type, **model_kwargs}
        )

    n_classes = y_train.shape[1]
    models = []

    # Tree models and linear models handle multi-output differently
    if model_type in ('xgb', 'lgbm'):
        # Train one model per class
        for cls_idx in range(n_classes):
            y_col = (y_train[:, cls_idx] > 0.4).astype(int)
            m = _build_model(model_type, **model_kwargs)
            m.fit(X_train, y_col, sample_weight=w_train)
            models.append(m)
    else:
        # MultiOutputClassifier handles all classes at once
        y_hard = (y_train > 0.4).astype(int)
        m = _build_model(model_type, **model_kwargs)
        m.fit(X_train, y_hard, sample_weight=w_train)
        models = [m]

    # Get probability scores for val set
    y_score = _predict_proba(models, X_val, model_type, n_classes)

    if confidence_tiers_val is None:
        confidence_tiers_val = ['High'] * len(y_val)

    metrics = compute_full_metrics(y_val, y_score, confidence_tiers_val)
    flat_metrics = _flatten_metrics(metrics)

    logger.info(f"{model_type} val mAP: {metrics['mAP']:.4f}")
    for cls, ap in metrics['per_class_ap'].items():
        logger.info(f"  {cls}: AP={ap:.4f}")

    if use_wandb:
        import wandb
        wandb.log(flat_metrics)

    # Save checkpoint
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, f'{model_type}_model.pkl')
        with open(ckpt_path, 'wb') as f:
            pickle.dump(models, f)
        logger.info(f"Saved checkpoint to {ckpt_path}")
        if use_wandb:
            import wandb
            artifact = wandb.Artifact(f'{model_type}-model', type='model')
            artifact.add_file(ckpt_path)
            wandb.log_artifact(artifact)

    if use_wandb:
        import wandb
        wandb.finish()

    return {'val_mAP': metrics['mAP'], **flat_metrics}


def _predict_proba(models, X, model_type, n_classes):
    """Get probability scores. Returns array of shape (n, n_classes)."""
    if model_type in ('xgb', 'lgbm'):
        scores = np.stack([m.predict_proba(X)[:, 1] for m in models], axis=1)
    elif model_type == 'svc':
        # LinearSVC has no predict_proba — use decision_function
        m = models[0]
        raw = m.decision_function(X)
        if raw.ndim == 1:
            raw = raw[:, np.newaxis]
        scores = 1 / (1 + np.exp(-raw))
    else:
        m = models[0]
        proba_list = m.predict_proba(X)  # list of (n, 2) arrays
        scores = np.stack([p[:, 1] for p in proba_list], axis=1)
    return scores.astype(np.float32)


def _flatten_metrics(metrics: dict) -> dict:
    """Flatten nested metrics dict for W&B logging."""
    flat = {'val_mAP': metrics['mAP']}
    for cls, ap in metrics['per_class_ap'].items():
        flat[f'val_AP_{cls}'] = ap
    for tier, tm in metrics['by_confidence'].items():
        flat[f'val_mAP_{tier}'] = tm.get('mAP', float('nan'))
    return flat

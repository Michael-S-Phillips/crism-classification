"""
Training loop for PyTorch models (MLP, CNN, ViT).
"""
import os, sys, copy, logging
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import CRISMPixelDataset, CRISMPatchDataset
from training.losses import WeightedBCEWithLogitsLoss
from evaluation.metrics import compute_full_metrics

logger = logging.getLogger(__name__)


def train_torch_model(
    model: torch.nn.Module,
    df: pd.DataFrame,
    model_name: str,
    max_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 10,
    use_wandb: bool = True,
    checkpoint_dir: Optional[str] = None,
    mrrsu_map: Optional[Dict[str, str]] = None,
    patch_size: int = 7,
    cache_dir: Optional[str] = None,
    use_pos_weight: bool = False,
    weight_decay: float = 1e-4,
    device: Optional[str] = None,
    **wandb_config
) -> Dict[str, Any]:
    """
    Train a PyTorch model with early stopping on val mAP.

    Automatically uses CRISMPatchDataset when mrrsu_map is provided (CNN/ViT),
    otherwise uses CRISMPixelDataset (MLP).
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    if use_wandb:
        import wandb as wb
        wb.init(
            project='crism-mineral-classification',
            name=model_name,
            config={'model': model_name, 'lr': lr, 'batch_size': batch_size,
                    'max_epochs': max_epochs, **wandb_config}
        )

    use_patches = mrrsu_map is not None

    # Compute pos_weight from training label prevalence (caps at 20x for stability)
    pos_weight = None
    if use_pos_weight:
        from data.dataset import LABEL_COLS
        train_sub = df[df['split'] == 'train']
        y_tr = train_sub[LABEL_COLS].values.astype('float32')
        n_pos = (y_tr > 0.4).sum(axis=0).clip(min=1)
        n_neg = len(y_tr) - n_pos
        pw = (n_neg / n_pos).clip(max=20.0)
        pos_weight = torch.tensor(pw, dtype=torch.float32).to(device)

    def make_dataset(split):
        sub = df[df['split'] == split]
        if use_patches:
            return CRISMPatchDataset(sub, mrrsu_map, patch_size=patch_size,
                                     cache_dir=cache_dir, split=split)
        return CRISMPixelDataset(sub)

    train_ds = make_dataset('train')
    val_ds = make_dataset('val')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    loss_fn = WeightedBCEWithLogitsLoss()

    val_sub = df[df['split'] == 'val']
    best_val_map = -1.0
    best_state = None
    patience_counter = 0
    stopped_epoch = max_epochs
    metrics = {}

    for epoch in range(1, max_epochs + 1):
        # --- Train ---
        model.train()
        train_losses = []
        for features, labels, weights in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            weights = weights.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, labels, weights, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # --- Validate ---
        model.eval()
        all_logits, all_labels = [], []

        with torch.no_grad():
            for features, labels, weights in val_loader:
                features = features.to(device)
                logits = model(features)
                all_logits.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(labels.numpy())

        y_score = np.concatenate(all_logits)
        y_true = np.concatenate(all_labels)
        conf_tiers = val_sub['confidence_tier'].tolist()

        metrics = compute_full_metrics(y_true, y_score, conf_tiers)
        val_map = metrics['mAP']
        flat = _flatten_metrics(metrics)

        logger.info(f"Epoch {epoch}/{max_epochs} | train_loss={np.mean(train_losses):.4f} | val_mAP={val_map:.4f}")

        if use_wandb:
            import wandb as wb
            wb.log({'epoch': epoch, 'train_loss': np.mean(train_losses), **flat})

        # Early stopping
        if val_map > best_val_map:
            best_val_map = val_map
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                stopped_epoch = epoch
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save checkpoint
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, f'{model_name}_best.pt')
        torch.save({'model_state': best_state, 'val_mAP': best_val_map}, ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}")
        if use_wandb:
            import wandb as wb
            artifact = wb.Artifact(f'{model_name}-model', type='model')
            artifact.add_file(ckpt_path)
            wb.log_artifact(artifact)

    if use_wandb:
        import wandb as wb
        wb.finish()

    return {'val_mAP': best_val_map, 'stopped_epoch': stopped_epoch, **_flatten_metrics(metrics)}


def _flatten_metrics(metrics: dict) -> dict:
    flat = {'val_mAP': metrics['mAP']}
    for cls, ap in metrics['per_class_ap'].items():
        flat[f'val_AP_{cls}'] = ap
    for tier, tm in metrics['by_confidence'].items():
        flat[f'val_mAP_{tier}'] = tm.get('mAP', float('nan'))
    return flat

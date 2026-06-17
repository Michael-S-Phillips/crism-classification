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


def build_class_balanced_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Build per-pixel sampling weights to oversample rare-class positives.

    Each pixel receives weight = max imbalance ratio of any class it is
    positive for. Plagioclase/HCP pixels get ~20–50x the weight of common
    olivine pixels.
    """
    from data.dataset import LABEL_COLS, _collapse_labels
    labels = _collapse_labels(df)[LABEL_COLS].values.astype('float32')
    n_pos = (labels > 0.4).sum(axis=0).clip(min=1)
    n_neg = len(labels) - n_pos
    imbalance = n_neg / n_pos  # higher = rarer class

    pixel_weights = np.ones(len(labels), dtype=np.float32)
    is_pos = labels > 0.4  # (n, 6)
    for i in range(len(labels)):
        if is_pos[i].any():
            pixel_weights[i] = float(imbalance[is_pos[i]].max())
    return pixel_weights


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
    mrral_map: Optional[Dict[str, str]] = None,
    patch_size: int = 7,
    cache_dir: Optional[str] = None,
    use_pos_weight: bool = False,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 0,
    lr_t_max: int = 50,
    lr_schedule: str = 'cosine',
    lr_step_size: int = 10,
    lr_gamma: float = 0.1,
    high_conf_only: bool = False,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    use_asl_loss: bool = False,
    asl_gamma_neg: float = 4.0,
    asl_gamma_pos: float = 0.0,
    asl_clip: float = 0.05,
    use_balanced_sampling: bool = False,
    use_spectral_aug: bool = False,
    aug_noise_std: float = 0.005,
    aug_band_dropout: float = 0.10,
    aug_shift_std: float = 0.005,
    encoder_lr_scale: Optional[float] = None,
    freeze_encoder: bool = False,
    class_weights: Optional[torch.Tensor] = None,
    pos_weight: Optional[torch.Tensor] = None,
    synth_train_cache: Optional[str] = None,
    synth_train_parquet: Optional[str] = None,
    synth_val_cache: Optional[str] = None,
    synth_val_parquet: Optional[str] = None,
    mrrsu_aux_dir: Optional[str] = None,
    is_aux_model: bool = False,
    min_delta: float = 0.0,
    stop_metric: str = 'val_mAP',
    decomp_lambda_recon: float = 1.0,
    decomp_lambda_eps: float = 0.1,
    decomp_lambda_T: float = 0.01,
    decomp_lambda_b: float = 0.01,
    decomp_lambda_smooth: float = 0.001,
    lambda_adv_max: float = 1.0,
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
                    'max_epochs': max_epochs, 'use_asl_loss': use_asl_loss,
                    'asl_gamma_neg': asl_gamma_neg,
                    'encoder_lr_scale': encoder_lr_scale,
                    'freeze_encoder': freeze_encoder,
                    **wandb_config}
        )

    use_patches = mrrsu_map is not None

    # Collapse olivine_t1/t2 → olivine and set uniform confidence weights
    from data.dataset import _collapse_labels
    df = _collapse_labels(df)

    # Split dataframes — train optionally filtered to High-confidence only
    train_df = df[df['split'] == 'train']
    if high_conf_only:
        n_before = len(train_df)
        train_df = train_df[train_df['confidence_tier'] == 'High']
        logger.info(f"high_conf_only: {len(train_df)} High-conf pixels "
                    f"(down from {n_before})")
    val_df = df[df['split'] == 'val']

    # Resolve pos_weight: explicit override wins, then auto-compute, else None.
    if pos_weight is not None:
        pos_weight = pos_weight.to(device, dtype=torch.float32)
        logger.info(f"Using explicit pos_weight: {pos_weight.tolist()}")
    elif use_pos_weight:
        from data.dataset import LABEL_COLS
        y_tr = train_df[LABEL_COLS].values.astype('float32')
        n_pos = (y_tr > 0.4).sum(axis=0).clip(min=1)
        n_neg = len(y_tr) - n_pos
        pw = (n_neg / n_pos).clip(max=20.0)
        pos_weight = torch.tensor(pw, dtype=torch.float32).to(device)
        logger.info(f"Auto-computed pos_weight from prevalence: {pos_weight.tolist()}")

    def make_dataset(sub_df, split_name='train'):
        from data.dataset import MRRAL_BAND_COLS, BAND_COLS, CRISMSpectralDataset, CRISMCombinedDataset
        if mrral_map is not None:
            from data.dataset import CRISMSpectralPatchDataset
            return CRISMSpectralPatchDataset(sub_df, mrral_map, patch_size=patch_size,
                                             cache_dir=cache_dir, split=split_name)
        if use_patches:
            return CRISMPatchDataset(sub_df, mrrsu_map, patch_size=patch_size,
                                     cache_dir=cache_dir, split=split_name)
        has_mrral = MRRAL_BAND_COLS[0] in sub_df.columns
        has_mrrsu = BAND_COLS[0] in sub_df.columns
        if has_mrral and has_mrrsu:
            return CRISMCombinedDataset(sub_df)
        if has_mrral:
            return CRISMSpectralDataset(sub_df)
        return CRISMPixelDataset(sub_df)

    if mrrsu_aux_dir is not None:
        import os as _os
        from data.dataset import MrrsuAuxPatchDataset
        stats = _os.path.join(mrrsu_aux_dir, 'mrrsu_aux_stats.json')
        train_ds = MrrsuAuxPatchDataset(
            train_df, mrral_map, patch_size,
            aux_npy=_os.path.join(mrrsu_aux_dir, 'mrrsu_aux_train.npy'),
            stats_json=stats, cache_dir=cache_dir, split='train')
        val_ds = MrrsuAuxPatchDataset(
            val_df, mrral_map, patch_size,
            aux_npy=_os.path.join(mrrsu_aux_dir, 'mrrsu_aux_val.npy'),
            stats_json=stats, cache_dir=cache_dir, split='val')
    else:
        train_ds = make_dataset(train_df, 'train')
        if synth_train_cache and synth_train_parquet:
            from data.dataset import SyntheticPatchDataset
            from torch.utils.data import ConcatDataset
            synth_ds = SyntheticPatchDataset(synth_train_cache, synth_train_parquet)
            logger.info(f"Concatenating {len(synth_ds)} synthetic plag patches into train set")
            train_ds = ConcatDataset([train_ds, synth_ds])
        val_ds = make_dataset(val_df, 'val')
        if synth_val_cache and synth_val_parquet:
            from data.dataset import SyntheticPatchDataset
            from torch.utils.data import ConcatDataset
            synth_val_ds = SyntheticPatchDataset(
                synth_val_cache, synth_val_parquet, split='val')
            logger.info(f"Concatenating {len(synth_val_ds)} synthetic plag val patches")
            val_ds = ConcatDataset([val_ds, synth_val_ds])

    if use_balanced_sampling:
        if synth_train_cache and synth_train_parquet:
            raise ValueError(
                "use_balanced_sampling is incompatible with synthetic-patch "
                "concatenation: the sampler weights are built from train_df only "
                "and would not align with the concatenated dataset length."
            )
        from torch.utils.data import WeightedRandomSampler
        pw = build_class_balanced_weights(train_df)
        sampler = WeightedRandomSampler(pw, num_samples=len(pw), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    model = model.to(device)
    if encoder_lr_scale is not None and hasattr(model, 'get_param_groups'):
        param_groups = model.get_param_groups(
            head_lr=lr,
            encoder_lr=lr * encoder_lr_scale,
        )
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    if lr_schedule == 'step':
        # Step decay: hold lr for lr_step_size epochs, then ×lr_gamma. Applied
        # to every param group, so the encoder/head differential (encoder_lr_
        # scale) is preserved across the decay. E.g. lr=0.01, step=10, gamma=0.1
        # → 0.01 (ep 0-9) → 0.001 (10-19) → 1e-4 (20-29) → 1e-5 (30-39)...
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_step_size, gamma=lr_gamma)
        logger.info(f"LR schedule: step (size={lr_step_size}, gamma={lr_gamma}, "
                    f"base_lr={lr})")
    else:
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=lr_t_max)
        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
            )
        else:
            scheduler = cosine

    is_decomp = type(model).__name__ == 'DecompSpVit'
    is_decomp_adv = type(model).__name__ == 'DecompSpVitAdv'

    if is_decomp_adv:
        from training.adv_decomp_losses import AdversarialDecompositionLoss
        loss_fn = AdversarialDecompositionLoss(
            lambda_recon=decomp_lambda_recon,
            lambda_smooth=decomp_lambda_smooth,
            asl_gamma_neg=asl_gamma_neg,
            asl_gamma_pos=asl_gamma_pos,
            asl_clip=asl_clip,
        )
        logger.info(
            f"Using AdversarialDecompositionLoss: λ_recon={decomp_lambda_recon}, "
            f"λ_smooth={decomp_lambda_smooth}, λ_adv_max={lambda_adv_max}"
        )
    elif is_decomp:
        from training.decomp_losses import DecompositionLoss
        loss_fn = DecompositionLoss(
            lambda_recon=decomp_lambda_recon,
            lambda_eps=decomp_lambda_eps,
            lambda_T=decomp_lambda_T,
            lambda_b=decomp_lambda_b,
            lambda_smooth=decomp_lambda_smooth,
            asl_gamma_neg=asl_gamma_neg,
            asl_gamma_pos=asl_gamma_pos,
            asl_clip=asl_clip,
        )
        logger.info(
            f"Using DecompositionLoss: λ_recon={decomp_lambda_recon}, "
            f"λ_eps={decomp_lambda_eps}, λ_T={decomp_lambda_T}, "
            f"λ_b={decomp_lambda_b}, λ_smooth={decomp_lambda_smooth}"
        )
    elif use_asl_loss:
        from training.losses import AsymmetricLoss
        loss_fn = AsymmetricLoss(gamma_neg=asl_gamma_neg, gamma_pos=asl_gamma_pos, clip=asl_clip)
    elif use_focal_loss:
        from training.losses import FocalBCEWithLogitsLoss
        loss_fn = FocalBCEWithLogitsLoss(gamma=focal_gamma)
    else:
        loss_fn = WeightedBCEWithLogitsLoss()

    # Per-class loss weights (e.g., boost rare classes like plagioclase, HCP).
    # Moved to device once so the loss can broadcast without per-step transfer.
    if class_weights is not None:
        class_weights = class_weights.to(device)
        logger.info(f"Using per-class loss weights: {class_weights.tolist()}")

    augment = None
    if use_spectral_aug:
        from training.augmentations import SpectralAugmentation
        augment = SpectralAugmentation(
            noise_std=aug_noise_std,
            band_dropout=aug_band_dropout,
            shift_std=aug_shift_std,
        ).to(device)

    val_sub = df[df['split'] == 'val']
    # Extra confidence-tier entries for synth val plag (appended in loader order)
    _synth_val_n = 0
    if synth_val_cache and synth_val_parquet:
        from data.dataset import SyntheticPatchDataset as _SVD
        _synth_val_n = len(_SVD(synth_val_cache, synth_val_parquet, split='val'))
    best_monitored = -1.0
    best_state = None
    best_map = -1.0       # secondary: always tracks val_mAP regardless of stop_metric
    best_map_state = None
    patience_counter = 0
    stopped_epoch = max_epochs
    metrics = {}
    logger.info(f"Early-stop metric: {stop_metric} (patience={patience})")

    for epoch in range(1, max_epochs + 1):
        # --- Train ---
        model.train()
        train_losses = []
        train_loss_components: dict = {}   # decomp-only; ignored for non-decomp models

        # DANN-style lambda_adv warmup for the adversarial decomposition model.
        # Smooth schedule from ~0 at epoch 1 to ~lambda_adv_max at the last epoch.
        if is_decomp_adv:
            import math as _math
            p = (epoch - 1) / max(max_epochs - 1, 1)        # ∈ [0, 1]
            schedule = (2.0 / (1.0 + _math.exp(-10.0 * p))) - 1.0   # ∈ [0, ~1)
            model.lambda_adv = float(lambda_adv_max * schedule)
            logger.info(
                f"epoch {epoch}: lambda_adv = {model.lambda_adv:.4f}"
            )
        for batch in train_loader:
            if is_aux_model:
                features, aux2, labels, weights = batch
                aux2 = aux2.to(device)
            else:
                features, labels, weights = batch
            features = features.to(device)
            if augment is not None:
                augment.train()
                features = augment(features)
            labels = labels.to(device)
            weights = weights.to(device)
            optimizer.zero_grad()
            if is_decomp_adv:
                logits, s_hat, n_hat, x_hat, disc_logits, _, _ = model(features)
                loss, components = loss_fn(
                    x=features,
                    logits=logits, labels=labels, weights=weights,
                    s_hat=s_hat, n_hat=n_hat, x_hat=x_hat,
                    disc_logits=disc_logits,
                    pos_weight=pos_weight, class_weights=class_weights,
                )
                for k, v in components.items():
                    train_loss_components.setdefault(k, []).append(v.item())
            elif is_decomp:
                logits, s_hat, T_hat, b_hat, eps_hat, x_hat = model(features)
                loss, components = loss_fn(
                    x=features,
                    logits=logits, labels=labels, weights=weights,
                    s_hat=s_hat, T_hat=T_hat, b_hat=b_hat,
                    eps_hat=eps_hat, x_hat=x_hat,
                    pos_weight=pos_weight, class_weights=class_weights,
                )
                for k, v in components.items():
                    train_loss_components.setdefault(k, []).append(v.item())
            else:
                logits = model(features, aux2) if is_aux_model else model(features)
                loss = loss_fn(
                    logits, labels, weights,
                    pos_weight=pos_weight, class_weights=class_weights,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # --- Validate ---
        model.eval()
        all_logits, all_labels = [], []
        val_T_means, val_b_means, val_eps_norms = [], [], []
        val_disc_correct, val_disc_total = 0.0, 0
        val_n_norms = []

        with torch.no_grad():
            for batch in val_loader:
                if is_aux_model:
                    features, aux2, labels, weights = batch
                    aux2 = aux2.to(device)
                else:
                    features, labels, weights = batch
                features = features.to(device)
                if is_decomp_adv:
                    logits, _, n_hat, _, disc_logits, _, _ = model(features)
                    val_n_norms.append(n_hat.norm(dim=-1).mean().item())
                    # Multi-label accuracy on the discriminator: per-class accuracy
                    # averaged over classes and samples. A working adversary drives
                    # this DOWN toward the marginal class prior (~prevalence-aware
                    # chance), as the encoder makes n_emb class-uninformative.
                    disc_pred = (torch.sigmoid(disc_logits) > 0.5).float().cpu()
                    target = (labels > 0.4).float()
                    correct = (disc_pred == target).float().mean().item()
                    val_disc_correct += correct * features.size(0)
                    val_disc_total += features.size(0)
                elif is_decomp:
                    logits, _s_hat, T_hat, b_hat, eps_hat, _x_hat = model(features)
                    val_T_means.append(T_hat.mean().item())
                    val_b_means.append(b_hat.mean().item())
                    val_eps_norms.append(eps_hat.norm(dim=-1).mean().item())
                else:
                    logits = model(features, aux2) if is_aux_model else model(features)
                all_logits.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(labels.numpy())

        y_score = np.concatenate(all_logits)
        y_true = np.concatenate(all_labels)
        conf_tiers = val_sub['confidence_tier'].tolist() + ['High'] * _synth_val_n

        metrics = compute_full_metrics(y_true, y_score, conf_tiers)
        val_map = metrics['mAP']
        if val_map > best_map:
            best_map = val_map
            best_map_state = copy.deepcopy(model.state_dict())
        flat = _flatten_metrics(metrics)
        # Pick the scalar we early-stop on. Default 'val_mAP'. Any flat key works.
        if stop_metric == 'val_mAP':
            monitored = val_map
        elif stop_metric in flat:
            monitored = float(flat[stop_metric])
        else:
            raise KeyError(
                f"stop_metric={stop_metric!r} not in available metrics: "
                f"['val_mAP'] + {sorted(flat.keys())}"
            )

        logger.info(
            f"Epoch {epoch}/{max_epochs} | train_loss={np.mean(train_losses):.4f} | "
            f"val_mAP={val_map:.4f} | {stop_metric}={monitored:.4f}"
        )

        if use_wandb:
            import wandb as wb
            log_dict = {'epoch': epoch, 'train_loss': np.mean(train_losses), **flat}
            if is_decomp_adv:
                for k, vals in train_loss_components.items():
                    if vals:
                        log_dict[f'train_loss_{k}'] = float(np.mean(vals))
                if val_n_norms:
                    log_dict['val_n_norm_mean'] = float(np.mean(val_n_norms))
                if val_disc_total > 0:
                    log_dict['val_disc_acc'] = float(val_disc_correct / val_disc_total)
                log_dict['lambda_adv'] = float(model.lambda_adv)
            elif is_decomp:
                # Per-epoch mean of each loss component (training side)
                for k, vals in train_loss_components.items():
                    if vals:
                        log_dict[f'train_loss_{k}'] = float(np.mean(vals))
                # Validation-side physical metrics
                if val_T_means:
                    log_dict['val_T_mean'] = float(np.mean(val_T_means))
                if val_b_means:
                    log_dict['val_b_mean'] = float(np.mean(val_b_means))
                if val_eps_norms:
                    log_dict['val_eps_norm_mean'] = float(np.mean(val_eps_norms))
            wb.log(log_dict)

        # Early stopping with tolerance band on the monitored metric. Values
        # near the running best (within `min_delta`) are treated as plateau,
        # not regression — patience only ticks on a meaningful drop.
        if monitored > best_monitored:
            best_monitored = monitored
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        elif monitored >= best_monitored - min_delta:
            # Inside tolerance — neither update best nor tick patience.
            pass
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch} ({stop_metric} plateaued at {best_monitored:.4f})")
                stopped_epoch = epoch
                break

    # Capture last-epoch weights before restoring best
    last_state = copy.deepcopy(model.state_dict())

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save checkpoints
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Primary: best on stop_metric
        ckpt_path = os.path.join(checkpoint_dir, f'{model_name}_best.pt')
        torch.save(
            {'model_state': best_state, 'stop_metric': stop_metric,
             'best_monitored': best_monitored},
            ckpt_path,
        )
        logger.info(f"Saved checkpoint: {ckpt_path} ({stop_metric}={best_monitored:.4f})")

        # Secondary: best val_mAP — only written when stop_metric differs, so the
        # two files are never identical copies.
        if stop_metric != 'val_mAP' and best_map_state is not None:
            map_ckpt = os.path.join(checkpoint_dir, f'{model_name}_best_map.pt')
            torch.save(
                {'model_state': best_map_state, 'stop_metric': 'val_mAP',
                 'best_monitored': best_map},
                map_ckpt,
            )
            logger.info(f"Saved checkpoint: {map_ckpt} (val_mAP={best_map:.4f})")

        # Last epoch (raw final-epoch weights, before best-restore above)
        last_ckpt = os.path.join(checkpoint_dir, f'{model_name}_last.pt')
        torch.save(
            {'model_state': last_state, 'stop_metric': 'last', 'best_monitored': None},
            last_ckpt,
        )
        logger.info(f"Saved checkpoint: {last_ckpt} (last epoch, stopped_epoch={stopped_epoch})")

        if use_wandb:
            import wandb as wb
            artifact = wb.Artifact(f'{model_name}-model', type='model')
            artifact.add_file(ckpt_path)
            wb.log_artifact(artifact)

    if use_wandb:
        import wandb as wb
        wb.finish()

    return {stop_metric: best_monitored, 'stopped_epoch': stopped_epoch, **_flatten_metrics(metrics)}


def _flatten_metrics(metrics: dict) -> dict:
    flat = {'val_mAP': metrics['mAP']}
    for cls, ap in metrics['per_class_ap'].items():
        flat[f'val_AP_{cls}'] = ap
    for tier, tm in metrics['by_confidence'].items():
        flat[f'val_mAP_{tier}'] = tm.get('mAP', float('nan'))
    return flat

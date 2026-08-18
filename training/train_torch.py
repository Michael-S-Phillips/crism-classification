"""
Training loop for PyTorch models (MLP, CNN, ViT).
"""
import os, sys, copy, logging
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import torch
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from device import get_device
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import CRISMPixelDataset, CRISMPatchDataset
from training.losses import WeightedBCEWithLogitsLoss
from evaluation.metrics import compute_full_metrics, compute_map

logger = logging.getLogger(__name__)


def _core_map(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """val_mAP_core: mAP with the junk class excluded whenever the current
    label set includes one. Name-based (not width-based) so pyx 6-class
    heads (LABEL_COLS_PYX, which has junk) are handled the same as 7-class
    heads; label sets without junk (5/6-class non-pyx) are a no-op and
    core == full mAP."""
    import data.dataset as _d
    if 'junk' in _d.LABEL_COLS[:y_score.shape[1]]:
        return compute_map(y_true, y_score, exclude=('junk',))
    return compute_map(y_true, y_score)


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


def _expected_synth_repr(continuum_removed: bool, dual_cr: bool) -> str:
    """Representation a run's synth patch cache must already be in.

    SyntheticPatchDataset serves its .npy VERBATIM -- it applies no transform --
    so the cache on disk must match whatever the rest of the run serves:
      dual_cr           -> 'dual' (118 channels, hull ⊕ linear)
      continuum_removed -> 'hull' (59 channels, hull-CR). True whether or not
                           cache_is_cr: with it the labeled cache is pre-CR'd on
                           disk, without it CRISMSpectralPatchDataset CRs at read
                           time; either way the SERVED representation is hull-CR.
      otherwise         -> 'raw'  (59 channels, raw reflectance)
    """
    if dual_cr:
        return 'dual'
    return 'hull' if continuum_removed else 'raw'


# Shapes that MUST agree between a _resume.pt and the model being trained.
# (state-dict key, dim to compare, what a mismatch means)
# A resume checkpoint whose tensors happen to be load-compatible is otherwise
# indistinguishable from the right one: a 118-channel run resumed from a 59-channel
# file, or a 7-class run resumed from a 6-class one, would train against the wrong
# representation / mislabelled targets and complete without ever erroring.
_RESUME_GUARDS = (
    ('encoder.band_embed.weight', -1,
     'input channel count (59 = raw/hull-CR, 118 = dual-CR)'),
    ('head.weight', 0, 'label vocabulary width (n_classes)'),
)


def _check_resume_compat(saved_state: dict, current_state: dict, resume_from: str) -> None:
    """Refuse a _resume.pt that belongs to a differently-configured run.

    Raises ValueError naming the offending key and both widths. Keys absent from
    both state dicts are skipped, so models without an `encoder.band_embed` (MLP,
    CNN, the test doubles) are unaffected.
    """
    for key, dim, what in _RESUME_GUARDS:
        in_saved, in_current = key in saved_state, key in current_state
        if not in_saved and not in_current:
            continue
        if in_saved != in_current:
            where = 'checkpoint' if in_saved else 'model being trained'
            raise ValueError(
                f'resume checkpoint mismatch: {key} is present only in the {where} '
                f'({resume_from}). The checkpoint was written by a different model '
                f'architecture; refusing to resume.'
            )
        saved_shape = tuple(saved_state[key].shape)
        current_shape = tuple(current_state[key].shape)
        if saved_shape[dim] != current_shape[dim]:
            raise ValueError(
                f'resume checkpoint mismatch on {key}: {what} is '
                f'{saved_shape[dim]} in the checkpoint but {current_shape[dim]} '
                f'in the model being trained (shapes {saved_shape} vs '
                f'{current_shape}). {resume_from} belongs to a different run; '
                f'refusing to resume.'
            )


def _save_resume_state(path: str, *, model, optimizer, scheduler, epoch: int,
                       best_monitored: float, best_map: float,
                       best_epoch, best_map_epoch,
                       patience_counter: int, stop_metric: str) -> None:
    """Write the full trainer state needed to CONTINUE this run, atomically.

    Deliberately a different file from `{model_name}_last.pt`: _last.pt is a
    RESULT (final-epoch weights, for evaluation), this is MACHINERY. Overloading
    one file for both makes it impossible to tell whether the weights in it are
    the ones to evaluate or the ones to keep training.

    Everything that a fresh process cannot re-derive goes in here. Dropping any
    single field produces a resumed run that completes and looks fine while being
    wrong: no scheduler_state re-runs warmup and restarts the cosine (wrong LR for
    the rest of the run); no best_monitored lets the first resumed epoch overwrite
    a genuinely better _best.pt; no patience_counter means early stopping never
    fires again.

    Atomic: temp path in the SAME directory then os.replace, so a kill mid-write
    cannot leave a truncated .pt where a valid one used to be.
    """
    payload = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'epoch': epoch,
        'best_monitored': best_monitored,
        'best_map': best_map,
        'best_epoch': best_epoch,
        'best_map_epoch': best_map_epoch,
        'patience_counter': patience_counter,
        'stop_metric': stop_metric,
    }
    tmp = f'{path}.tmp{os.getpid()}'
    torch.save(payload, tmp)
    os.replace(tmp, path)


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
    gated_head: bool = False,
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
    continuum_removed: bool = False,
    brightness_aux: bool = False,
    cache_is_cr: bool = False,
    dual_cr: bool = False,
    min_delta: float = 0.0,
    stop_metric: str = 'val_mAP_core',
    checkpoint_every: int = 0,
    resume_from: Optional[str] = None,
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
    Train a PyTorch model with early stopping on val_mAP_core (mAP excluding
    the junk class in 7-class mode; equal to plain val_mAP for 5/6-class).

    Automatically uses CRISMPatchDataset when mrrsu_map is provided (CNN/ViT),
    otherwise uses CRISMPixelDataset (MLP).

    checkpoint_every: write `{model_name}_resume.pt` (full trainer state) every N
        epochs. 0 (default) disables it entirely — nothing is written and the run
        is byte-identical to one from before this option existed.
    resume_from: path to a `{model_name}_resume.pt` to CONTINUE from. Restores
        weights, optimizer, scheduler, best_*/patience bookkeeping and starts at
        the saved epoch + 1.

    max_epochs is TOTAL ACROSS JOBS, not additional: resuming at epoch 123 with
    max_epochs=150 runs 27 more epochs, not 150 more. A resume whose epoch is
    already >= max_epochs logs and returns without training.
    """
    if gated_head and not use_asl_loss:
        raise ValueError('--gated_head requires --asl_loss (GatedAsymmetricLoss '
                         'is the only gated loss implemented)')

    # Gate partition, resolved ONCE and shared by the loss and the validation
    # metric. The composition g*c / (1-g)*c has exactly one implementation
    # (models.gated_classifier.compose_gated_probs) with three call sites:
    # this loss, this validation loop, and scripts/classify_tile_supervised.py.
    # A validation loop that scored raw conditionals instead would silently
    # compare the gate logit against the olivine label and shift every column
    # after it -- no crash, just 24 hours of checkpoint selection on noise.
    gate_partition = None
    if gated_head:
        from data.dataset import LABEL_COLS as _GATE_LABEL_COLS
        from models.gated_classifier import class_partition, compose_gated_probs
        gate_partition = class_partition(_GATE_LABEL_COLS)

    if device is None:
        device = get_device()
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

    # dual_cr (118 channels: hull-CR 0-58 ⊕ linear-CR 59-117) is only served by
    # CRISMSpectralPatchDataset. Anything else reached with dual_cr=True would
    # quietly serve 59 channels, and a wrong-representation run that completes is
    # worse than one that doesn't start — it reads as a falsified hypothesis.
    if dual_cr:
        if not continuum_removed:
            raise ValueError('dual_cr requires continuum_removed=True')
        if mrral_map is None:
            raise ValueError(
                'dual_cr requires the mrral patch path (mrral_map is None), the '
                'only dataset that serves 118 channels')
        if mrrsu_aux_dir is not None:
            raise ValueError(
                'dual_cr is incompatible with mrrsu_aux_dir: MrrsuAuxPatchDataset '
                'serves 59-channel patches')
        # synth_* caches are no longer refused outright: a 118-channel dual plag
        # cache can be built with scripts/convert_synth_cache_representation.py.
        # A 59-channel one is still fatal, and SyntheticPatchDataset raises on it
        # because every construction below declares expect_repr.

    def make_dataset(sub_df, split_name='train'):
        from data.dataset import MRRAL_BAND_COLS, BAND_COLS, CRISMSpectralDataset, CRISMCombinedDataset
        if mrral_map is not None:
            from data.dataset import CRISMSpectralPatchDataset
            return CRISMSpectralPatchDataset(
                sub_df, mrral_map, patch_size=patch_size,
                cache_dir=cache_dir, split=split_name,
                continuum_removed=continuum_removed,
                return_brightness=brightness_aux,
                cache_is_cr=cache_is_cr,
                dual_cr=dual_cr)
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

    # The representation every synth cache in this run must already be in. Only
    # 'raw' can reach the pre-2026-08-11 code path unchanged; 'hull' is what makes
    # a raw cache in a --continuum_removed run fail instead of train.
    synth_repr = _expected_synth_repr(continuum_removed, dual_cr)

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
            # split='train' is LOAD-BEARING. Without it SyntheticPatchDataset
            # serves every row in the parquet, so a synth set carrying its own
            # val/test rows put them in TRAIN while --synth_val_* simultaneously
            # put the val rows in VAL — the model was validated on plagioclase
            # patches it had trained on. Logs show 1,817 into train against 109
            # into val, and the 109 were a subset of the 1,817. Audit 2026-08-08.
            # return_brightness must track brightness_aux exactly. make_dataset
            # gives the labeled half a 4-tuple under --brightness_aux; a synth
            # half still returning a 3-tuple makes default_collate raise "each
            # element in list of batch should be of equal size" on the first
            # mixed batch (2026-08-11, hpc_finetune_handcore).
            synth_ds = SyntheticPatchDataset(synth_train_cache, synth_train_parquet,
                                             split='train', expect_repr=synth_repr,
                                             return_brightness=brightness_aux)
            logger.info(f"Concatenating {len(synth_ds)} synthetic plag patches into train set")
            train_ds = ConcatDataset([train_ds, synth_ds])
        val_ds = make_dataset(val_df, 'val')
        if synth_val_cache and synth_val_parquet:
            from data.dataset import SyntheticPatchDataset
            from torch.utils.data import ConcatDataset
            synth_val_ds = SyntheticPatchDataset(
                synth_val_cache, synth_val_parquet, split='val',
                expect_repr=synth_repr, return_brightness=brightness_aux)
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
    elif use_asl_loss and gated_head:
        from training.gated_losses import GatedAsymmetricLoss
        mineral_idx, non_mineral_idx = gate_partition
        loss_fn = GatedAsymmetricLoss(
            mineral_idx, non_mineral_idx, gamma_neg=asl_gamma_neg,
            gamma_pos=asl_gamma_pos, clip=asl_clip, lambda_gate=1.0)
        logger.info(
            f'Using GatedAsymmetricLoss: gate over minerals {mineral_idx}, '
            f'non-minerals {non_mineral_idx}, clip={asl_clip}, lambda_gate=1.0')
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
        _synth_val_n = len(_SVD(synth_val_cache, synth_val_parquet, split='val',
                               expect_repr=synth_repr))
    best_monitored = -1.0
    best_state = None
    best_epoch = None
    best_map_epoch = None
    best_map = -1.0       # secondary: always tracks val_mAP regardless of stop_metric
    best_map_state = None
    patience_counter = 0
    stopped_epoch = max_epochs
    metrics = {}
    # Default monitored metric is val_mAP_core: mAP excluding the junk class
    # (7-class mode only; identical to val_mAP for 5/6-class heads). Junk's
    # noisy near-zero AP deflates and destabilizes plain val_mAP, so plain
    # 'val_mAP' requests are promoted. Both values are still logged.
    if stop_metric == 'val_mAP':
        stop_metric = 'val_mAP_core'
        logger.info("stop_metric 'val_mAP' promoted to 'val_mAP_core' "
                    "(junk excluded; equal to val_mAP for 5/6-class runs)")
    logger.info(f"Early-stop metric: {stop_metric} (patience={patience})")

    # --- Resume a killed run -------------------------------------------------
    # Restores EVERY piece of state the loop below carries across epochs. Placed
    # after the stop_metric promotion above so the saved and requested metrics are
    # compared in the same (promoted) form, and after the optimizer/scheduler are
    # built so their state dicts can be loaded into the real objects.
    start_epoch = 1
    if resume_from is not None:
        ck = torch.load(resume_from, map_location=device, weights_only=False)
        _check_resume_compat(ck['model_state'], model.state_dict(), resume_from)
        saved_metric = ck.get('stop_metric')
        if saved_metric != stop_metric:
            # best_monitored is a value OF the stop metric. Carrying it across a
            # metric change would compare apples to oranges and either freeze
            # _best.pt forever or overwrite it on the first epoch.
            raise ValueError(
                f'resume checkpoint mismatch on stop_metric: {resume_from} was '
                f'written monitoring {saved_metric!r} but this run monitors '
                f'{stop_metric!r}; best_monitored is not comparable across the '
                f'two. Refusing to resume.'
            )
        model.load_state_dict(ck['model_state'])
        optimizer.load_state_dict(ck['optimizer_state'])
        # LOAD-BEARING. Without this the resumed run re-runs LinearLR warmup and
        # restarts the cosine from scratch, so it trains at the wrong learning
        # rate for every remaining epoch and says nothing about it.
        scheduler.load_state_dict(ck['scheduler_state'])
        best_monitored = float(ck['best_monitored'])
        best_map = float(ck['best_map'])
        best_epoch = ck['best_epoch']
        best_map_epoch = ck['best_map_epoch']
        patience_counter = int(ck['patience_counter'])
        resumed_epoch = int(ck['epoch'])
        start_epoch = resumed_epoch + 1
        stopped_epoch = max_epochs
        logger.info(
            f"Resumed {resume_from}: continuing at epoch {start_epoch}/{max_epochs} "
            f"(max_epochs is TOTAL across jobs, so {max(max_epochs - resumed_epoch, 0)} "
            f"epochs remain) | {stop_metric} best={best_monitored:.4f} @ "
            f"epoch {best_epoch} | val_mAP best={best_map:.4f} @ epoch "
            f"{best_map_epoch} | patience_counter={patience_counter}/{patience} | "
            f"lr={optimizer.param_groups[0]['lr']:.6g}"
        )
        if start_epoch > max_epochs:
            logger.info(
                f"Nothing to do: {resume_from} is already at epoch {resumed_epoch} "
                f"and max_epochs={max_epochs} counts TOTAL epochs across jobs. "
                f"Raise --epochs above {resumed_epoch} to train further. Exiting "
                f"without touching any checkpoint."
            )
            if use_wandb:
                import wandb as wb
                wb.finish()
            return {stop_metric: best_monitored, 'stopped_epoch': resumed_epoch,
                    'resumed_already_complete': True}

    for epoch in range(start_epoch, max_epochs + 1):
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
        val_gates = []      # gated-only; stays empty otherwise

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
                if gate_partition is not None:
                    # Compose exactly as the loss and inference do. Appending
                    # raw sigmoids here would hand compute_map an (N, 8) array
                    # with the GATE in column 0 to score against (N, 7) labels
                    # -- silently misaligned, never an error.
                    probs, gate = compose_gated_probs(logits, *gate_partition)
                    all_logits.append(probs.cpu().numpy())
                    val_gates.append(gate.cpu().numpy())
                else:
                    all_logits.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(labels.numpy())

        y_score = np.concatenate(all_logits)
        y_true = np.concatenate(all_labels)
        conf_tiers = val_sub['confidence_tier'].tolist() + ['High'] * _synth_val_n

        metrics = compute_full_metrics(y_true, y_score, conf_tiers)
        val_map = metrics['mAP']
        # Core mAP: junk excluded whenever the current label set has a junk
        # class (name-based, not width-based -- covers 7-class AND pyx
        # 6-class; label sets without junk get core == full via compute_map).
        val_map_core = _core_map(y_true, y_score)
        if val_map > best_map:
            best_map = val_map
            best_map_state = copy.deepcopy(model.state_dict())
            best_map_epoch = epoch
            # Same walltime-safety as the primary best below: write on improvement
            # rather than only after the loop. Kept even though the end-of-run
            # block re-writes it, because the whole point is surviving a kill.
            if checkpoint_dir and stop_metric != 'val_mAP':
                os.makedirs(checkpoint_dir, exist_ok=True)
                _pm = os.path.join(checkpoint_dir, f'{model_name}_best_map.pt')
                _tm = f'{_pm}.tmp{os.getpid()}'
                torch.save({'model_state': best_map_state,
                            'stop_metric': 'val_mAP', 'best_monitored': best_map,
                            'epoch': epoch}, _tm)
                os.replace(_tm, _pm)
        flat = _flatten_metrics(metrics)
        flat['val_mAP_core'] = val_map_core

        # Gate health (gated runs only). design.md:171-173 flags both failure
        # modes as the arm's main risk: a gate saturating near 1 degenerates
        # to the flat head this arm exists to replace, and one near 0 kills
        # every mineral at once. The gate BCE is unweighted and runs ~3.6x the
        # main ASL term at lambda_gate=1.0, so saturation is a live risk --
        # without these numbers it would stay invisible until a tile is
        # classified a day after the job ends. Quantiles, not just the mean:
        # a bimodal gate (half shut, half open) has an unremarkable mean.
        gate_stats = ''
        if val_gates:
            _g = np.concatenate(val_gates)
            _p10, _p50, _p90 = np.percentile(_g, [10, 50, 90])
            flat['val_gate_mean'] = float(_g.mean())
            flat['val_gate_p10'] = float(_p10)
            flat['val_gate_p50'] = float(_p50)
            flat['val_gate_p90'] = float(_p90)
            gate_stats = (f" | gate_mean={_g.mean():.4f} "
                          f"(p10={_p10:.4f} p50={_p50:.4f} p90={_p90:.4f})")
        # Pick the scalar we early-stop on. Default 'val_mAP_core'. Any flat
        # key works ('val_mAP' itself is promoted to core above).
        if stop_metric == 'val_mAP_core':
            monitored = val_map_core
        elif stop_metric in flat:
            monitored = float(flat[stop_metric])
        else:
            raise KeyError(
                f"stop_metric={stop_metric!r} not in available metrics: "
                f"{sorted(flat.keys())}"
            )

        logger.info(
            f"Epoch {epoch}/{max_epochs} | train_loss={np.mean(train_losses):.4f} | "
            f"val_mAP={val_map:.4f} | val_mAP_core={val_map_core:.4f} | "
            f"{stop_metric}={monitored:.4f}{gate_stats}"
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
        stop_now = False
        if monitored > best_monitored:
            best_monitored = monitored
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
            # WRITE IT NOW, not after the loop. Until 2026-08-13 every
            # checkpoint was written only once training had exited, with
            # best_state held in memory the whole time -- so a job killed by
            # the SLURM walltime saved NOTHING however good its best epoch was.
            # ft_7cls_handcore_reviewup died that way at epoch 123/150 (job
            # 23548837, CANCELLED DUE TO TIME LIMIT) and lost a full day of GPU
            # with no artifact at all. Its sibling arm survived only because it
            # early-stopped at 121 with ~30 min to spare.
            # Atomic: write to a temp path in the same directory and rename, so
            # a kill mid-write cannot leave a truncated .pt where a valid one
            # used to be. Rename is atomic within a filesystem.
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
                _p = os.path.join(checkpoint_dir, f'{model_name}_best.pt')
                _tmp = f'{_p}.tmp{os.getpid()}'
                torch.save({'model_state': best_state, 'stop_metric': stop_metric,
                            'best_monitored': best_monitored, 'epoch': epoch}, _tmp)
                os.replace(_tmp, _p)
                logger.info(f"Checkpointed epoch {epoch}: {_p} "
                            f"({stop_metric}={best_monitored:.4f})")
        elif monitored >= best_monitored - min_delta:
            # Inside tolerance — neither update best nor tick patience.
            pass
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch} ({stop_metric} plateaued at {best_monitored:.4f})")
                stopped_epoch = epoch
                stop_now = True

        # --- Periodic full-state resume checkpoint ---
        # AFTER the early-stopping block on purpose: best_monitored / best_epoch /
        # patience_counter must be this epoch's values. Written a epoch early they
        # would be one epoch stale, and a resume from stale bookkeeping from stale bookkeeping overwrites
        # a better _best.pt on its first epoch. Inert when checkpoint_every == 0.
        if checkpoint_dir and checkpoint_every and epoch % checkpoint_every == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            _rp = os.path.join(checkpoint_dir, f'{model_name}_resume.pt')
            _save_resume_state(
                _rp, model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, best_monitored=best_monitored, best_map=best_map,
                best_epoch=best_epoch, best_map_epoch=best_map_epoch,
                patience_counter=patience_counter, stop_metric=stop_metric)
            logger.info(f"Wrote resume state at epoch {epoch}: {_rp}")

        if stop_now:
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
        if best_state is not None:
            torch.save(
                {'model_state': best_state, 'stop_metric': stop_metric,
                 'best_monitored': best_monitored, 'epoch': best_epoch},
                ckpt_path,
            )
            logger.info(f"Saved checkpoint: {ckpt_path} ({stop_metric}={best_monitored:.4f})")
        else:
            # Only reachable on a RESUMED run in which no epoch beat the restored
            # best_monitored. best_state is None then, and writing it would replace
            # the good _best.pt from the earlier job with {'model_state': None} —
            # silently destroying the very checkpoint resume exists to protect.
            logger.info(
                f"No epoch improved on the resumed best {stop_metric}="
                f"{best_monitored:.4f} (epoch {best_epoch}); leaving {ckpt_path} "
                f"as written by the earlier job."
            )

        # Secondary: best val_mAP — only written when it actually diverges from
        # the monitored metric, so the two files are never identical copies
        # (5/6-class runs have val_mAP_core == val_mAP every epoch).
        if (stop_metric != 'val_mAP' and best_map_state is not None
                and best_map != best_monitored):
            map_ckpt = os.path.join(checkpoint_dir, f'{model_name}_best_map.pt')
            torch.save(
                {'model_state': best_map_state, 'stop_metric': 'val_mAP',
                 'best_monitored': best_map, 'epoch': best_map_epoch},
                map_ckpt,
            )
            logger.info(f"Saved checkpoint: {map_ckpt} (val_mAP={best_map:.4f})")

        # Last epoch (raw final-epoch weights, before best-restore above)
        last_ckpt = os.path.join(checkpoint_dir, f'{model_name}_last.pt')
        torch.save(
            {'model_state': last_state, 'stop_metric': 'last',
             'best_monitored': None, 'epoch': stopped_epoch},
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

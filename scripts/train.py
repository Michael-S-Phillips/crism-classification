"""
Unified training entry point.

Usage:
    conda run -n crism python scripts/train.py --model logreg
    conda run -n crism python scripts/train.py --model rf --n_estimators 300
    conda run -n crism python scripts/train.py --model mlp --lr 1e-3 --epochs 100
    conda run -n crism python scripts/train.py --model cnn --patch_size 7
    conda run -n crism python scripts/train.py --model vit --embed_dim 128

Models: logreg, svc, rf, xgb, lgbm, mlp, cnn, vit
"""
import argparse, os, sys, yaml, logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

SKLEARN_MODELS = {'logreg', 'svc', 'rf', 'xgb', 'lgbm'}
TORCH_MODELS = {'mlp', 'cnn', 'vit', 'spectral_cnn', 'spectral_vit',
                'spectral_hybrid', 'spatial_vit', 'spatial_vit_aux',
                'decomp_spatial_vit', 'decomp_spatial_vit_adv'}

def load_config(config_path):
    try:
        from config_loader import load_config as _load
        return _load(config_path)
    except ImportError:
        with open(config_path) as f:
            return yaml.safe_load(f)

def _build_parser():
    parser = argparse.ArgumentParser(description="Train a mineral classification model.")
    parser.add_argument('--model', required=True, choices=list(SKLEARN_MODELS | TORCH_MODELS))
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--no_wandb', action='store_true')
    # sklearn kwargs
    parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--max_depth', type=int, default=None)
    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--num_leaves', type=int, default=31,
                        help='LightGBM num_leaves')
    parser.add_argument('--subsample', type=float, default=1.0,
                        help='Row subsample ratio for XGB/LGBM')
    # torch kwargs
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=7)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=4)
    # sweep / architecture kwargs
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout rate for CNN/MLP (default: model default)')
    parser.add_argument('--hidden_dims', type=str, default=None,
                        help='MLP hidden dims as comma-separated ints, e.g. 512,256,128')
    parser.add_argument(
        '--with_alteration', action='store_true',
        help='Use the 6-class label set (olivine, lcp, hcp, plagioclase, '
             'other, alteration). Swaps data.dataset.LABEL_COLS to the '
             'WITH_ALTERATION variant and auto-bumps n_classes to 6. The '
             'parquet must contain an `alteration` column (run '
             'scripts/patch_mrral_pixels_with_alteration.py first if the '
             'existing single-file parquet was built before 2026-06-10).')
    parser.add_argument(
        '--seven_class', action='store_true',
        help='Use the 7-class label set (olivine, lcp, hcp, plagioclase, '
             'bland, alteration, junk). Requires a parquet built by '
             'scripts/build_7cls_dataset.py containing bland and junk columns. '
             'Mutually exclusive with --with_alteration.')
    parser.add_argument('--pyx', action='store_true',
                        help='6-class pyroxene-merged vocab '
                             '(olivine/pyx/plagioclase/bland/alteration/junk); '
                             'pyx = max(lcp,hcp). Mutually exclusive with '
                             '--seven_class / --with_alteration.')
    parser.add_argument('--pyx_alt', action='store_true',
                        help='5-class pyroxene-merged vocab for HAND-LABELED-ONLY '
                             'training (olivine/pyx/plagioclase/other/alteration); '
                             'pyx = max(lcp,hcp). Uses the base parquet native '
                             'classes (no review-derived bland/junk). Mutually '
                             'exclusive with --seven_class/--with_alteration/--pyx.')
    parser.add_argument('--use_pos_weight', action='store_true',
                        help='Use pos_weight in loss to upweight rare classes')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='AdamW weight decay for torch models')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Custom wandb run name and checkpoint prefix (default: model name)')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Linear LR warmup epochs before cosine annealing (default: 0)')
    parser.add_argument('--lr_t_max', type=int, default=50,
                        help='CosineAnnealingLR T_max (default: 50)')
    parser.add_argument('--lr_schedule', type=str, default='cosine',
                        choices=['cosine', 'step'],
                        help="LR schedule. 'cosine' (default) = warmup+cosine "
                             "anneal. 'step' = hold lr for --lr_step_size "
                             "epochs then multiply by --lr_gamma; applied to "
                             "all param groups so encoder_lr_scale is preserved.")
    parser.add_argument('--lr_step_size', type=int, default=10,
                        help='StepLR: epochs between LR drops (default 10).')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='StepLR: LR multiplier at each drop (default 0.1).')
    parser.add_argument('--high_conf_only', action='store_true',
                        help='Train on High-confidence pixels only')
    parser.add_argument('--focal_loss', action='store_true',
                        help='Use focal loss instead of BCE')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma (default: 2.0)')
    parser.add_argument('--asl_loss', action='store_true',
                        help='Use asymmetric loss (Wang et al. 2021) instead of focal/BCE')
    parser.add_argument('--asl_gamma_neg', type=float, default=4.0)
    parser.add_argument('--asl_gamma_pos', type=float, default=0.0)
    parser.add_argument('--asl_clip', type=float, default=0.05)
    parser.add_argument('--balanced_sampling', action='store_true',
                        help='Use class-balanced WeightedRandomSampler')
    parser.add_argument('--spectral_aug', action='store_true',
                        help='Apply spectral augmentation during training')
    parser.add_argument('--aug_noise_std', type=float, default=0.005)
    parser.add_argument('--aug_band_dropout', type=float, default=0.10)
    parser.add_argument('--aug_shift_std', type=float, default=0.005)
    parser.add_argument('--pretrain_ckpt', type=str, default=None,
                        help='Path to MAE pretrain checkpoint; loads encoder into spectral_vit')
    parser.add_argument('--init_ckpt', type=str, default=None,
                        help='Path to a finetune checkpoint (model_state) used as a '
                             'warm start for the FULL classifier (encoder + head). '
                             'Use this to continue a finished finetune run for more '
                             'epochs. Mutually exclusive with --pretrain_ckpt.')
    parser.add_argument('--n_classes', type=int, default=5,
                        help='Number of output classes (default 5). Use 1 in '
                             'binary mode (--binary_target_class).')
    parser.add_argument('--binary_target_class', type=str, default=None,
                        choices=[None, 'olivine', 'lcp', 'hcp', 'plagioclase', 'other'],
                        help='Train a binary classifier (target-class vs rest) by '
                             'collapsing the multi-class labels to a single is-X '
                             'column. Set --n_classes 1 with this. When warm-starting '
                             'from a 5-class checkpoint, the head weights are dropped '
                             'and only the encoder is reused.')
    parser.add_argument('--encoder_lr_scale', type=float, default=None,
                        help='LR multiplier for pretrained encoder (e.g. 0.1 → 10× slower than head). '
                             'Only effective when --pretrain_ckpt is set and model has get_param_groups.')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze encoder weights (requires_grad=False). '
                             'Mutually exclusive with --encoder_lr_scale. '
                             'Only effective when --pretrain_ckpt is set.')
    parser.add_argument('--class_weights', type=str, default=None,
                        help='Per-class loss multipliers, comma-separated in '
                             'LABEL_COLS order (olivine,lcp,hcp,plagioclase,other). '
                             'e.g. "1,1,1.5,3,1" to boost HCP and plagioclase. '
                             'Default: uniform 1.0 for all classes.')
    parser.add_argument('--synth_train_cache', type=str, default=None,
                        help='Path to synth_plag_patches_p7.npy to add to the train split.')
    parser.add_argument('--synth_train_parquet', type=str, default=None,
                        help='Path to synth_plag_rows.parquet (row-aligned with the cache).')
    parser.add_argument('--synth_val_cache', type=str, default=None,
                        help='Path to patches.npy used as synth plag val source '
                             '(same file as synth_train_cache; val rows selected by split column).')
    parser.add_argument('--synth_val_parquet', type=str, default=None,
                        help='Path to parquet with split column for filtering val plag rows.')
    parser.add_argument('--mrral_parquets', nargs='+', type=str, default=None,
                        help='One or more mrral pixels parquets to use as the '
                             'training table (e.g. data/mrral_pixels.parquet '
                             'plus data/mrral_pixels_with_review.parquet). '
                             'Concatenated in order. Default: '
                             'cfg.output_dir/mrral_pixels.parquet only. The '
                             'patch cache pointed at by --patch_cache_dir must '
                             'be built from the SAME parquet list, in the same '
                             'order — use cache_mrral_patches.py --parquets ...')
    parser.add_argument('--patch_cache_dir', type=str, default=None,
                        help='Override cfg.patch_cache_dir. Set this when '
                             '--mrral_parquets differs from the default so the '
                             'trainer reads the matching pre-built cache.')
    parser.add_argument('--apply_relabels', type=str, default=None,
                        help='Path to an olivine_relabels.csv; applies soft pyroxene '
                             'targets to the matching polygons before training '
                             '(row order preserved so the patch cache stays valid).')
    parser.add_argument('--mrrsu_aux_dir', type=str, default=None,
                        help='Dir with mrrsu_aux_{split}.npy + mrrsu_aux_stats.json '
                             '(enables the spatial_vit_aux model).')
    parser.add_argument('--continuum_removed', action='store_true',
                        help='Continuum-remove each mrral patch (upper-hull CR over '
                             'the 59-band good-band window) before the encoder sees '
                             'it. Applies to spatial_vit / spatial_vit_aux. CR is '
                             'computed on-read from the raw patch cache unless '
                             '--cache_is_cr says the cache is already CR.')
    parser.add_argument('--brightness_aux', action='store_true',
                        help='Feed the retained per-pixel brightness scalar (mean '
                             'good-band reflectance, pre-CR) as a 1-D late-fusion '
                             'aux to spatial_vit_aux (aux_dim=1). Requires '
                             '--continuum_removed and --model spatial_vit_aux; '
                             'replaces the mrrsu aux (no --mrrsu_aux_dir needed).')
    parser.add_argument('--cache_is_cr', action='store_true',
                        help='The patch cache pointed at by --patch_cache_dir already '
                             'holds CR patches (built with build_global_patch_cache.py '
                             '--continuum_removed or the labeled equivalent); read '
                             'brightness from the *_brightness.npy sidecar rather than '
                             'CR-ing on read.')
    parser.add_argument('--min_delta', type=float, default=0.0,
                        help='Early-stopping tolerance: val_mAP drops up to this '
                             'much below the running best do not tick patience. '
                             'Default 0.0 = strict, any non-improvement counts.')
    parser.add_argument('--stop_metric', type=str, default='val_mAP',
                        help='Metric watched for early stopping and best-checkpoint '
                             'selection. The default val_mAP is auto-promoted to '
                             'val_mAP_core (mAP excluding the noisy junk class in '
                             '7-class mode; identical to val_mAP for 5/6-class runs) '
                             '— both are still logged. Per-class keys still work, '
                             'e.g. val_AP_plagioclase lets a run keep going even when '
                             'overall mAP plateaus, so long as plag is still improving.')
    parser.add_argument('--decomp_lambda_recon', type=float, default=1.0,
                        help='Weight on reconstruction MSE for DecompSpVit.')
    parser.add_argument('--decomp_lambda_eps', type=float, default=0.1,
                        help='Weight on residual L2 for DecompSpVit.')
    parser.add_argument('--decomp_lambda_T', type=float, default=0.01,
                        help='Weight on (T-1)^2 prior for DecompSpVit.')
    parser.add_argument('--decomp_lambda_b', type=float, default=0.01,
                        help='Weight on b^2 prior for DecompSpVit.')
    parser.add_argument('--decomp_lambda_smooth', type=float, default=0.001,
                        help='Weight on spatial-TV smoothness on s_hat for DecompSpVit.')
    parser.add_argument('--lambda_adv_max', type=float, default=1.0,
                        help='Max value of the adversarial GRL multiplier for '
                             'DecompSpVitAdv. Schedule warms from ~0 → this value.')
    return parser


def build_args():
    parser = _build_parser()
    args = parser.parse_args()

    if args.freeze_encoder and args.encoder_lr_scale is not None:
        parser.error('--freeze_encoder and --encoder_lr_scale are mutually exclusive.')
    if args.init_ckpt and args.pretrain_ckpt:
        parser.error('--init_ckpt and --pretrain_ckpt are mutually exclusive '
                     '(--init_ckpt loads the full classifier; --pretrain_ckpt '
                     'loads only the encoder).')
    if args.seven_class and args.with_alteration:
        parser.error('--seven_class and --with_alteration are mutually exclusive.')
    if args.pyx and (args.seven_class or args.with_alteration):
        parser.error('--pyx is mutually exclusive with --seven_class/--with_alteration.')
    if args.pyx_alt and (args.seven_class or args.with_alteration or args.pyx):
        parser.error('--pyx_alt is mutually exclusive with '
                     '--seven_class/--with_alteration/--pyx.')
    if args.brightness_aux and not args.continuum_removed:
        parser.error('--brightness_aux requires --continuum_removed.')
    if args.brightness_aux and args.model != 'spatial_vit_aux':
        parser.error('--brightness_aux requires --model spatial_vit_aux.')
    if args.cache_is_cr and not args.continuum_removed:
        parser.error('--cache_is_cr requires --continuum_removed.')

    # Class-count / LABEL_COLS swap — must happen BEFORE any dataset code reads it.
    if args.seven_class:
        import data.dataset
        data.dataset.LABEL_COLS = list(data.dataset.LABEL_COLS_7CLASS)
        if args.n_classes not in (5, 6, 7):
            logging.warning('--seven_class set but --n_classes=%d; forcing 7.',
                            args.n_classes)
        args.n_classes = 7
        logging.info('7-class mode: LABEL_COLS = %s', data.dataset.LABEL_COLS)
    elif args.with_alteration:
        import data.dataset
        data.dataset.LABEL_COLS = list(data.dataset.LABEL_COLS_WITH_ALTERATION)
        if args.n_classes != 6:
            if args.n_classes != 5:
                logging.warning(
                    f'--with_alteration set but --n_classes={args.n_classes}; '
                    f'forcing n_classes=6.')
            args.n_classes = 6
        logging.info('6-class mode: LABEL_COLS = %s', data.dataset.LABEL_COLS)
    elif args.pyx:
        import data.dataset
        data.dataset.LABEL_COLS = list(data.dataset.LABEL_COLS_PYX)
        args.n_classes = 6
        logging.info('pyx mode: LABEL_COLS = %s', data.dataset.LABEL_COLS)
    elif args.pyx_alt:
        import data.dataset
        data.dataset.LABEL_COLS = list(data.dataset.LABEL_COLS_PYX_ALT)
        args.n_classes = 5
        logging.info('pyx_alt mode: LABEL_COLS = %s', data.dataset.LABEL_COLS)

    return args


def main():
    parser = _build_parser()
    args = build_args()

    # Parse class_weights once for all torch training paths. In binary mode the
    # single value is routed to pos_weight (BCE positive-sample multiplier),
    # not class_weights — which is symmetric across pos/neg and a no-op for
    # n_classes=1.
    class_weights_tensor = None
    binary_pos_weight_tensor = None
    if args.class_weights:
        import torch as _torch
        try:
            vals = [float(v) for v in args.class_weights.split(',')]
        except ValueError as e:
            parser.error(f'--class_weights values must be floats: {e}')
        expected = 1 if args.binary_target_class else args.n_classes
        if len(vals) != expected:
            import data.dataset
            label = ('pos_weight' if args.binary_target_class
                     else ','.join(data.dataset.LABEL_COLS[:expected]))
            parser.error(f'--class_weights must have {expected} value(s) '
                         f'({label}); got {len(vals)}')
        if args.binary_target_class:
            binary_pos_weight_tensor = _torch.tensor(vals, dtype=_torch.float32)
        else:
            class_weights_tensor = _torch.tensor(vals, dtype=_torch.float32)

    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.config
    )
    cfg = load_config(cfg_path)
    parquet_path = os.path.join(cfg['output_dir'], 'pixels.parquet')
    checkpoint_dir = cfg['checkpoints_dir']
    use_wandb = not args.no_wandb

    def _load_mrral_table() -> pd.DataFrame:
        """Resolve --mrral_parquets (or default) and concat in order. The
        patch cache pointed at by --patch_cache_dir or cfg.patch_cache_dir
        MUST have been built from the same parquet list in the same order;
        the cache is row-aligned with the concatenated dataframe."""
        paths = args.mrral_parquets or [
            os.path.join(cfg['output_dir'], 'mrral_pixels.parquet')]
        logging.info(f'Loading {len(paths)} mrral parquet(s): {paths}')
        parts = []
        for p in paths:
            part = pd.read_parquet(p)
            logging.info(f'  {p}: {len(part):,} rows')
            parts.append(part)
        df_out = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        logging.info(f'  concat: {len(df_out):,} rows total')
        return df_out

    if args.patch_cache_dir:
        cfg['patch_cache_dir'] = args.patch_cache_dir
        logging.info(f'patch_cache_dir overridden to {args.patch_cache_dir}')

    if args.model in SKLEARN_MODELS:
        from data.dataset import load_sklearn_arrays
        from training.train_sklearn import train_and_evaluate_sklearn

        df = pd.read_parquet(parquet_path)
        X_tr, y_tr, w_tr, X_v, y_v, w_v, X_te, y_te, w_te = load_sklearn_arrays(parquet_path)
        val_tiers = df[df['split'] == 'val']['confidence_tier'].tolist()

        metrics = train_and_evaluate_sklearn(
            args.model, X_tr, y_tr, w_tr, X_v, y_v, w_v,
            confidence_tiers_val=val_tiers,
            use_wandb=use_wandb,
            checkpoint_dir=checkpoint_dir,
            run_name=args.run_name,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            C=args.C,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            subsample=args.subsample,
        )

    elif args.model in TORCH_MODELS:
        import torch
        from training.train_torch import train_torch_model
        # spatial_vit / spatial_vit_aux load mrral_pixels.parquet directly; skip pixels.parquet
        if args.model not in ('spatial_vit', 'spatial_vit_aux'):
            df = pd.read_parquet(parquet_path)
        run_name = args.run_name or args.model

        if args.model == 'mlp':
            from models.mlp import MLP
            hidden_dims = tuple(int(x) for x in args.hidden_dims.split(',')) \
                if args.hidden_dims else (256, 128)
            dropout = args.dropout if args.dropout is not None else 0.3
            model = MLP(n_features=60, n_classes=5,
                        hidden_dims=hidden_dims, dropout=dropout)
            metrics = train_torch_model(
                model=model, df=df, model_name=run_name,
                max_epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, patience=args.patience,
                use_wandb=use_wandb, checkpoint_dir=checkpoint_dir,
                use_pos_weight=args.use_pos_weight,
                weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs,
                lr_t_max=args.lr_t_max,
                high_conf_only=args.high_conf_only,
                use_focal_loss=args.focal_loss,
                focal_gamma=args.focal_gamma,
                use_balanced_sampling=args.balanced_sampling,
                use_spectral_aug=args.spectral_aug,
                aug_noise_std=args.aug_noise_std,
                aug_band_dropout=args.aug_band_dropout,
                aug_shift_std=args.aug_shift_std,
            )

        elif args.model in ('cnn', 'vit'):
            cache_dir = cfg.get('patch_cache_dir')
            patch_size = args.patch_size
            cache_complete = cache_dir and all(
                os.path.exists(os.path.join(cache_dir, f'{s}_patches_p{patch_size}.npy'))
                for s in ('train', 'val', 'test')
            )
            if cache_complete:
                mrrsu_map = {}
            else:
                from data.extract_pixels import find_tile_pairs
                pairs = find_tile_pairs(cfg['gpkg_dir'], cfg['data_root'])
                mrrsu_map = {tid: p for tid, _, p in pairs}

            dropout = args.dropout if args.dropout is not None else 0.3
            if args.model == 'cnn':
                from models.cnn import SpectralSpatialCNN
                model = SpectralSpatialCNN(n_bands=60, n_classes=5,
                                           patch_size=patch_size, dropout=dropout)
            else:
                from models.vit import SpectralViT
                model = SpectralViT(
                    n_bands=60, n_classes=5, patch_size=patch_size,
                    embed_dim=args.embed_dim, n_heads=args.n_heads,
                    n_layers=args.n_layers, dropout=dropout,
                )

            metrics = train_torch_model(
                model=model, df=df, model_name=run_name,
                max_epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, patience=args.patience,
                use_wandb=use_wandb, checkpoint_dir=checkpoint_dir,
                mrrsu_map=mrrsu_map, patch_size=patch_size,
                cache_dir=cache_dir,
                use_pos_weight=args.use_pos_weight,
                weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs,
                lr_t_max=args.lr_t_max,
                high_conf_only=args.high_conf_only,
                use_focal_loss=args.focal_loss,
                focal_gamma=args.focal_gamma,
                use_balanced_sampling=args.balanced_sampling,
                use_spectral_aug=args.spectral_aug,
                aug_noise_std=args.aug_noise_std,
                aug_band_dropout=args.aug_band_dropout,
                aug_shift_std=args.aug_shift_std,
            )

        elif args.model in ('spectral_cnn', 'spectral_vit'):
            df_mrral = _load_mrral_table()
            dropout = args.dropout if args.dropout is not None else 0.3

            if args.model == 'spectral_cnn':
                from models.spectral_cnn import SpectralCNN1D
                model = SpectralCNN1D(n_bands=59, n_classes=5, dropout=dropout)
            else:
                from models.spectral_transformer import SpectralTransformer
                dropout = args.dropout if args.dropout is not None else 0.1
                model = SpectralTransformer(
                    n_bands=59, n_classes=5,
                    embed_dim=args.embed_dim, n_heads=args.n_heads,
                    n_layers=args.n_layers, dropout=dropout,
                )
                if args.pretrain_ckpt:
                    import logging as _log
                    ckpt = torch.load(args.pretrain_ckpt, map_location='cpu')
                    missing, unexpected = model.load_encoder_state_dict(ckpt['encoder_state'])
                    _log.getLogger(__name__).info(
                        f"Loaded MAE encoder from {args.pretrain_ckpt}. "
                        f"Missing: {missing}, Unexpected: {unexpected}"
                    )

            metrics = train_torch_model(
                model=model, df=df_mrral, model_name=run_name,
                max_epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, patience=args.patience,
                use_wandb=use_wandb, checkpoint_dir=checkpoint_dir,
                use_pos_weight=args.use_pos_weight,
                weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs,
                lr_t_max=args.lr_t_max,
                high_conf_only=args.high_conf_only,
                use_focal_loss=args.focal_loss,
                focal_gamma=args.focal_gamma,
                use_asl_loss=args.asl_loss,
                asl_gamma_neg=args.asl_gamma_neg,
                asl_gamma_pos=args.asl_gamma_pos,
                asl_clip=args.asl_clip,
                use_balanced_sampling=args.balanced_sampling,
                use_spectral_aug=args.spectral_aug,
                aug_noise_std=args.aug_noise_std,
                aug_band_dropout=args.aug_band_dropout,
                aug_shift_std=args.aug_shift_std,
                encoder_lr_scale=args.encoder_lr_scale,
                class_weights=class_weights_tensor,
                min_delta=args.min_delta,
            )

        elif args.model == 'spectral_hybrid':
            from models.hybrid_classifier import SpectralHybridClassifier
            from data.dataset import BAND_COLS
            df_mrral = _load_mrral_table()
            df_mrrsu = pd.read_parquet(parquet_path)  # pixels.parquet has b0..b59
            MERGE_KEYS = ['tile_id', 'polygon_id', 'pixel_row', 'pixel_col']
            df_combined = df_mrral.merge(
                df_mrrsu[MERGE_KEYS + BAND_COLS],
                on=MERGE_KEYS,
                how='inner',
            )
            logging.info(
                f"Combined dataset: {len(df_combined)} pixels "
                f"({len(df_mrral)} mrral ∩ {len(df_mrrsu)} mrrsu)"
            )

            dropout = args.dropout if args.dropout is not None else 0.1
            model = SpectralHybridClassifier(
                n_mrral=59, n_mrrsu=60, n_classes=5,
                embed_dim=args.embed_dim, n_heads=args.n_heads,
                n_layers=args.n_layers, dropout=dropout,
            )
            if args.pretrain_ckpt:
                ckpt = torch.load(args.pretrain_ckpt, map_location='cpu')
                missing, unexpected = model.load_encoder_state_dict(ckpt['encoder_state'])
                logging.info(
                    f"Loaded MAE encoder from {args.pretrain_ckpt}. "
                    f"Missing: {missing}, Unexpected: {unexpected}"
                )

            metrics = train_torch_model(
                model=model, df=df_combined, model_name=run_name,
                max_epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, patience=args.patience,
                use_wandb=use_wandb, checkpoint_dir=checkpoint_dir,
                use_pos_weight=args.use_pos_weight,
                weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs,
                lr_t_max=args.lr_t_max,
                high_conf_only=args.high_conf_only,
                use_focal_loss=args.focal_loss,
                focal_gamma=args.focal_gamma,
                use_asl_loss=args.asl_loss,
                asl_gamma_neg=args.asl_gamma_neg,
                asl_gamma_pos=args.asl_gamma_pos,
                asl_clip=args.asl_clip,
                use_balanced_sampling=args.balanced_sampling,
                use_spectral_aug=args.spectral_aug,
                aug_noise_std=args.aug_noise_std,
                aug_band_dropout=args.aug_band_dropout,
                aug_shift_std=args.aug_shift_std,
                encoder_lr_scale=args.encoder_lr_scale,
                class_weights=class_weights_tensor,
                min_delta=args.min_delta,
            )


        elif args.model == 'spatial_vit':
            import glob as _glob
            data_root = cfg['data_root']
            globs_to_try = [
                os.path.join(data_root, 'mc*', 't*mrral*.hdr'),
                os.path.join(data_root, 't*mrral*.hdr'),
            ]
            mrral_hdrs = []
            for pattern in globs_to_try:
                mrral_hdrs = sorted(_glob.glob(pattern))
                if mrral_hdrs:
                    break
            mrral_map = {}
            for hdr in mrral_hdrs:
                tid = os.path.basename(hdr).split('_mrral_')[0]
                mrral_map[tid] = hdr.replace('.hdr', '.img')
            logging.info(f'mrral_map: {len(mrral_map)} tiles found')

            df_mrral = _load_mrral_table()
            if args.apply_relabels:
                from data.dataset import apply_olivine_relabels
                df_mrral, _n_relabel = apply_olivine_relabels(df_mrral, args.apply_relabels)
                logging.info(f'Applied relabels from {args.apply_relabels}: '
                             f'{_n_relabel} pixels updated with soft pyroxene targets')
            dropout = args.dropout if args.dropout is not None else 0.1

            # Binary mode: collapse multi-class labels to a single is-X column.
            if args.binary_target_class:
                from data.dataset import _collapse_labels
                _coll = _collapse_labels(df_mrral)
                df_mrral = df_mrral.copy()
                df_mrral['__binary__'] = (
                    _coll[args.binary_target_class] > 0
                ).astype('float32')
                # Re-point LABEL_COLS so train_torch_model and the datasets
                # consume the single binary target.
                import data.dataset
                data.dataset.LABEL_COLS = ['__binary__']
                if args.n_classes != 1:
                    logging.warning(
                        f'--binary_target_class set but --n_classes={args.n_classes}; '
                        f'forcing n_classes=1.'
                    )
                    args.n_classes = 1
                n_pos = int(df_mrral['__binary__'].sum())
                logging.info(
                    f'Binary mode: target_class={args.binary_target_class}, '
                    f'positives={n_pos}/{len(df_mrral)} '
                    f'({100 * n_pos / len(df_mrral):.2f}%). '
                    f'LABEL_COLS overridden to [__binary__]; n_classes=1.'
                )

            from models.spatial_spectral_transformer import SpatialSpectralClassifier
            model = SpatialSpectralClassifier(
                n_bands=59, patch_size=args.patch_size, n_classes=args.n_classes,
                embed_dim=args.embed_dim, n_heads=args.n_heads,
                n_layers=args.n_layers, dropout=dropout,
            )
            if args.pretrain_ckpt:
                ckpt = torch.load(args.pretrain_ckpt, map_location='cpu', weights_only=False)
                missing, unexpected = model.load_encoder_state_dict(ckpt['encoder_state'])
                logging.info(
                    f'Loaded spatial MAE encoder from {args.pretrain_ckpt}. '
                    f'Missing: {missing}, Unexpected: {unexpected}'
                )
            elif args.init_ckpt:
                ckpt = torch.load(args.init_ckpt, map_location='cpu', weights_only=False)
                if isinstance(ckpt, dict) and 'model_state' in ckpt:
                    state = ckpt['model_state']
                    prev_val_map = ckpt.get('val_mAP', None)
                else:
                    state = ckpt
                    prev_val_map = None
                # Drop head weights when the loaded head shape doesn't match
                # (e.g. warm-starting a binary classifier from a 5-class one).
                head_w = state.get('head.weight')
                if head_w is not None and head_w.shape[0] != args.n_classes:
                    state = {k: v for k, v in state.items()
                             if not k.startswith('head.')}
                    logging.info(
                        f'Dropping head weights from warm-start ckpt: '
                        f'ckpt head {head_w.shape[0]}-class vs model '
                        f'{args.n_classes}-class. Encoder will be reused.'
                    )
                missing, unexpected = model.load_state_dict(state, strict=False)
                logging.info(
                    f'Warm-started classifier from {args.init_ckpt}. '
                    f'prev val_mAP={prev_val_map}. Missing: {missing}, Unexpected: {unexpected}'
                )
            if args.freeze_encoder:
                for p in model.encoder.parameters():
                    p.requires_grad = False
                logging.info('Encoder frozen (requires_grad=False on all encoder params)')

            mrral_cache_dir = cfg.get('patch_cache_dir')
            metrics = train_torch_model(
                model=model, df=df_mrral, model_name=run_name,
                max_epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, patience=args.patience,
                use_wandb=use_wandb, checkpoint_dir=checkpoint_dir,
                mrral_map=mrral_map, patch_size=args.patch_size,
                cache_dir=mrral_cache_dir,
                use_pos_weight=args.use_pos_weight,
                weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs,
                lr_t_max=args.lr_t_max,
                lr_schedule=args.lr_schedule,
                lr_step_size=args.lr_step_size,
                lr_gamma=args.lr_gamma,
                high_conf_only=args.high_conf_only,
                use_focal_loss=args.focal_loss,
                focal_gamma=args.focal_gamma,
                use_asl_loss=args.asl_loss,
                asl_gamma_neg=args.asl_gamma_neg,
                asl_gamma_pos=args.asl_gamma_pos,
                asl_clip=args.asl_clip,
                use_balanced_sampling=args.balanced_sampling,
                encoder_lr_scale=args.encoder_lr_scale,
                class_weights=class_weights_tensor,
                pos_weight=binary_pos_weight_tensor,
                synth_train_cache=args.synth_train_cache,
                synth_train_parquet=args.synth_train_parquet,
                synth_val_cache=args.synth_val_cache,
                synth_val_parquet=args.synth_val_parquet,
                stop_metric=args.stop_metric,
                min_delta=args.min_delta,
                freeze_encoder=args.freeze_encoder,
                continuum_removed=args.continuum_removed,
                cache_is_cr=args.cache_is_cr,
            )

        elif args.model == 'spatial_vit_aux':
            # Two aux sources: the mrrsu-param aux (--mrrsu_aux_dir, aux_dim=2) or
            # the CR brightness scalar (--brightness_aux, aux_dim=1). Exactly one.
            if args.brightness_aux:
                if args.mrrsu_aux_dir:
                    parser.error('--brightness_aux and --mrrsu_aux_dir are mutually '
                                 'exclusive (both feed the aux head).')
                aux_dim = 1
            else:
                if not args.mrrsu_aux_dir:
                    parser.error('--model spatial_vit_aux requires --mrrsu_aux_dir '
                                 'or --brightness_aux')
                aux_dim = 2
            import glob as _glob
            data_root = cfg['data_root']
            mrral_hdrs = sorted(set(
                _glob.glob(os.path.join(data_root, 'mc*', 't*mrral*.hdr'))
                + _glob.glob(os.path.join(data_root, 't*mrral*.hdr'))))
            mrral_map = {os.path.basename(h).split('_mrral_')[0]: h.replace('.hdr', '.img')
                         for h in mrral_hdrs}
            logging.info(f'mrral_map: {len(mrral_map)} tiles found')
            df_mrral = _load_mrral_table()
            dropout = args.dropout if args.dropout is not None else 0.1
            from models.spatial_spectral_classifier_aux import SpatialSpectralClassifierAux
            model = SpatialSpectralClassifierAux(
                n_bands=59, patch_size=args.patch_size, n_classes=args.n_classes,
                embed_dim=args.embed_dim, n_heads=args.n_heads,
                n_layers=args.n_layers, dropout=dropout, aux_dim=aux_dim,
            )
            if args.pretrain_ckpt:
                ckpt = torch.load(args.pretrain_ckpt, map_location='cpu', weights_only=False)
                missing, unexpected = model.load_encoder_state_dict(ckpt['encoder_state'])
                logging.info(f'Loaded encoder from {args.pretrain_ckpt}. '
                             f'Missing: {missing}, Unexpected: {unexpected}')
            mrral_cache_dir = cfg.get('patch_cache_dir')
            metrics = train_torch_model(
                model=model, df=df_mrral, model_name=run_name,
                max_epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, patience=args.patience,
                use_wandb=use_wandb, checkpoint_dir=checkpoint_dir,
                mrral_map=mrral_map, patch_size=args.patch_size,
                cache_dir=mrral_cache_dir,
                weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs, lr_t_max=args.lr_t_max,
                use_focal_loss=args.focal_loss, focal_gamma=args.focal_gamma,
                use_asl_loss=args.asl_loss, asl_gamma_neg=args.asl_gamma_neg,
                asl_gamma_pos=args.asl_gamma_pos, asl_clip=args.asl_clip,
                encoder_lr_scale=args.encoder_lr_scale,
                class_weights=class_weights_tensor,
                min_delta=args.min_delta,
                stop_metric=args.stop_metric,
                # brightness-aux path: CR patches + brightness scalar via
                # CRISMSpectralPatchDataset (no mrrsu_aux_dir). Otherwise the
                # mrrsu-param aux dataset.
                mrrsu_aux_dir=None if args.brightness_aux else args.mrrsu_aux_dir,
                continuum_removed=args.continuum_removed,
                brightness_aux=args.brightness_aux,
                cache_is_cr=args.cache_is_cr,
                is_aux_model=True,
            )

        elif args.model == 'decomp_spatial_vit':
            import glob as _glob
            data_root = cfg['data_root']
            globs_to_try = [
                os.path.join(data_root, 'mc*', 't*mrral*.hdr'),
                os.path.join(data_root, 't*mrral*.hdr'),
            ]
            mrral_hdrs = []
            for pattern in globs_to_try:
                mrral_hdrs = sorted(_glob.glob(pattern))
                if mrral_hdrs:
                    break
            mrral_map = {}
            for hdr in mrral_hdrs:
                tid = os.path.basename(hdr).split('_mrral_')[0]
                mrral_map[tid] = hdr.replace('.hdr', '.img')
            logging.info(f'mrral_map: {len(mrral_map)} tiles found')

            df_mrral = _load_mrral_table()
            dropout = args.dropout if args.dropout is not None else 0.1

            from models.decomp_spatial_vit import DecompSpVit
            model = DecompSpVit(
                n_bands=59, patch_size=args.patch_size, n_classes=5,
                embed_dim=args.embed_dim, n_heads=args.n_heads,
                n_layers=args.n_layers, dropout=dropout,
            )
            if args.pretrain_ckpt:
                ckpt = torch.load(args.pretrain_ckpt, map_location='cpu', weights_only=False)
                missing, unexpected = model.load_encoder_state_dict(ckpt['encoder_state'])
                logging.info(
                    f'Loaded spatial MAE encoder from {args.pretrain_ckpt}. '
                    f'Missing: {missing}, Unexpected: {unexpected}'
                )
            if args.freeze_encoder:
                for p in model.encoder.parameters():
                    p.requires_grad = False
                logging.info('Encoder frozen (requires_grad=False on all encoder params)')

            mrral_cache_dir = cfg.get('patch_cache_dir')
            metrics = train_torch_model(
                model=model, df=df_mrral, model_name=run_name,
                max_epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, patience=args.patience,
                use_wandb=use_wandb, checkpoint_dir=checkpoint_dir,
                mrral_map=mrral_map, patch_size=args.patch_size,
                cache_dir=mrral_cache_dir,
                use_pos_weight=args.use_pos_weight,
                weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs,
                lr_t_max=args.lr_t_max,
                high_conf_only=args.high_conf_only,
                use_focal_loss=args.focal_loss,
                focal_gamma=args.focal_gamma,
                use_asl_loss=args.asl_loss,
                asl_gamma_neg=args.asl_gamma_neg,
                asl_gamma_pos=args.asl_gamma_pos,
                asl_clip=args.asl_clip,
                use_balanced_sampling=args.balanced_sampling,
                encoder_lr_scale=args.encoder_lr_scale,
                class_weights=class_weights_tensor,
                stop_metric=args.stop_metric,
                min_delta=args.min_delta,
                decomp_lambda_recon=args.decomp_lambda_recon,
                decomp_lambda_eps=args.decomp_lambda_eps,
                decomp_lambda_T=args.decomp_lambda_T,
                decomp_lambda_b=args.decomp_lambda_b,
                decomp_lambda_smooth=args.decomp_lambda_smooth,
                freeze_encoder=args.freeze_encoder,
            )

        elif args.model == 'decomp_spatial_vit_adv':
            import glob as _glob
            data_root = cfg['data_root']
            globs_to_try = [
                os.path.join(data_root, 'mc*', 't*mrral*.hdr'),
                os.path.join(data_root, 't*mrral*.hdr'),
            ]
            mrral_hdrs = []
            for pattern in globs_to_try:
                mrral_hdrs = sorted(_glob.glob(pattern))
                if mrral_hdrs:
                    break
            mrral_map = {}
            for hdr in mrral_hdrs:
                tid = os.path.basename(hdr).split('_mrral_')[0]
                mrral_map[tid] = hdr.replace('.hdr', '.img')
            logging.info(f'mrral_map: {len(mrral_map)} tiles found')

            df_mrral = _load_mrral_table()
            dropout = args.dropout if args.dropout is not None else 0.1

            from models.decomp_spatial_vit_adv import DecompSpVitAdv
            model = DecompSpVitAdv(
                n_bands=59, patch_size=args.patch_size, n_classes=5,
                embed_dim=args.embed_dim, n_heads=args.n_heads,
                n_layers=args.n_layers, dropout=dropout,
                lambda_adv=0.0,   # warms up via the per-epoch schedule
            )
            if args.pretrain_ckpt:
                ckpt = torch.load(args.pretrain_ckpt, map_location='cpu', weights_only=False)
                missing, unexpected = model.load_encoder_state_dict(ckpt['encoder_state'])
                logging.info(
                    f'Loaded spatial MAE encoder from {args.pretrain_ckpt}. '
                    f'Missing: {missing}, Unexpected: {unexpected}'
                )
            if args.freeze_encoder:
                for p in model.encoder.parameters():
                    p.requires_grad = False
                logging.info('Encoder frozen (requires_grad=False on all encoder params)')

            mrral_cache_dir = cfg.get('patch_cache_dir')
            metrics = train_torch_model(
                model=model, df=df_mrral, model_name=run_name,
                max_epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, patience=args.patience,
                use_wandb=use_wandb, checkpoint_dir=checkpoint_dir,
                mrral_map=mrral_map, patch_size=args.patch_size,
                cache_dir=mrral_cache_dir,
                use_pos_weight=args.use_pos_weight,
                weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs,
                lr_t_max=args.lr_t_max,
                high_conf_only=args.high_conf_only,
                use_focal_loss=args.focal_loss,
                focal_gamma=args.focal_gamma,
                use_asl_loss=args.asl_loss,
                asl_gamma_neg=args.asl_gamma_neg,
                asl_gamma_pos=args.asl_gamma_pos,
                asl_clip=args.asl_clip,
                use_balanced_sampling=args.balanced_sampling,
                encoder_lr_scale=args.encoder_lr_scale,
                class_weights=class_weights_tensor,
                stop_metric=args.stop_metric,
                min_delta=args.min_delta,
                decomp_lambda_recon=args.decomp_lambda_recon,
                decomp_lambda_smooth=args.decomp_lambda_smooth,
                lambda_adv_max=args.lambda_adv_max,
                freeze_encoder=args.freeze_encoder,
            )

    print(f"\n=== {run_name if args.model in TORCH_MODELS else args.model} Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == '__main__':
    main()

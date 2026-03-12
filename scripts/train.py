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
TORCH_MODELS = {'mlp', 'cnn', 'vit', 'spectral_cnn', 'spectral_vit', 'spectral_hybrid'}

def load_config(config_path):
    from config_loader import load_config as _load
    return _load(config_path)

def main():
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
    parser.add_argument('--encoder_lr_scale', type=float, default=None,
                        help='LR multiplier for pretrained encoder (e.g. 0.1 → 10× slower than head). '
                             'Only effective when --pretrain_ckpt is set and model has get_param_groups.')
    args = parser.parse_args()

    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.config
    )
    cfg = load_config(cfg_path)
    parquet_path = os.path.join(cfg['output_dir'], 'pixels.parquet')
    checkpoint_dir = cfg['checkpoints_dir']
    use_wandb = not args.no_wandb

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
            mrral_parquet = os.path.join(os.path.dirname(parquet_path), 'mrral_pixels.parquet')
            df_mrral = pd.read_parquet(mrral_parquet)
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
            )

        elif args.model == 'spectral_hybrid':
            from models.hybrid_classifier import SpectralHybridClassifier
            from data.dataset import BAND_COLS
            mrral_parquet = os.path.join(os.path.dirname(parquet_path), 'mrral_pixels.parquet')
            mrrsu_parquet = parquet_path  # pixels.parquet has b0..b59

            df_mrral = pd.read_parquet(mrral_parquet)
            df_mrrsu = pd.read_parquet(mrrsu_parquet)
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
            )

    print(f"\n=== {run_name if args.model in TORCH_MODELS else args.model} Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == '__main__':
    main()

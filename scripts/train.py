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
TORCH_MODELS = {'mlp', 'cnn', 'vit'}

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

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
    # torch kwargs
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--patch_size', type=int, default=7)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=4)
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
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            C=args.C,
            learning_rate=args.learning_rate,
        )

    elif args.model in TORCH_MODELS:
        import torch
        from training.train_torch import train_torch_model
        df = pd.read_parquet(parquet_path)

        if args.model == 'mlp':
            from models.mlp import MLP
            model = MLP(n_features=60, n_classes=6)
            metrics = train_torch_model(
                model=model, df=df, model_name='mlp',
                max_epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, patience=args.patience,
                use_wandb=use_wandb, checkpoint_dir=checkpoint_dir,
            )

        elif args.model in ('cnn', 'vit'):
            from data.extract_pixels import find_tile_pairs
            pairs = find_tile_pairs(cfg['gpkg_dir'], cfg['data_root'])
            mrrsu_map = {tid: p for tid, _, p in pairs}

            if args.model == 'cnn':
                from models.cnn import SpectralSpatialCNN
                model = SpectralSpatialCNN(n_bands=60, n_classes=6, patch_size=args.patch_size)
            else:
                from models.vit import SpectralViT
                model = SpectralViT(
                    n_bands=60, n_classes=6, patch_size=args.patch_size,
                    embed_dim=args.embed_dim, n_heads=args.n_heads, n_layers=args.n_layers
                )

            metrics = train_torch_model(
                model=model, df=df, model_name=args.model,
                max_epochs=args.epochs, batch_size=args.batch_size,
                lr=args.lr, patience=args.patience,
                use_wandb=use_wandb, checkpoint_dir=checkpoint_dir,
                mrrsu_map=mrrsu_map, patch_size=args.patch_size,
            )

    print(f"\n=== {args.model} Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == '__main__':
    main()

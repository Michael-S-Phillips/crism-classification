"""
MAE pre-training on mrral spectral data.

Usage:
    conda run -n crism python scripts/pretrain_mae.py
    conda run -n crism python scripts/pretrain_mae.py --epochs 100 --embed_dim 256

Saves checkpoint to: checkpoints/mae_pretrain_{embed_dim}d_{n_layers}l_best.pt
"""
import argparse
import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--mask_ratio', type=float, default=0.40)
    parser.add_argument('--no_wandb', action='store_true')
    args = parser.parse_args()

    import yaml
    cfg = yaml.safe_load(open(os.path.join(PROJ, 'config.yaml')))
    parquet = os.path.join(cfg['output_dir'], 'mrral_pixels.parquet')
    ckpt_dir = cfg['checkpoints_dir']

    df = pd.read_parquet(parquet)
    # Use all pixels (train+val+test) for pretraining — no labels used
    from data.dataset import CRISMSpectralDataset
    ds = CRISMSpectralDataset(df)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from models.mae import SpectralMAE
    model = SpectralMAE(
        n_bands=59, embed_dim=args.embed_dim, n_heads=args.n_heads,
        n_layers=args.n_layers, mask_ratio=args.mask_ratio,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_name = f'mae_pretrain_{args.embed_dim}d_{args.n_layers}l'
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(project='crism-mineral-classification', name=run_name,
                   config=vars(args))

    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for features, _, _ in loader:
            features = features.to(device)
            optimizer.zero_grad()
            loss, _, _ = model(features)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        mean_loss = np.mean(losses)
        logging.info(f"Epoch {epoch}/{args.epochs} | mae_loss={mean_loss:.5f}")
        if use_wandb:
            import wandb
            wandb.log({'epoch': epoch, 'mae_loss': mean_loss})
        if mean_loss < best_loss:
            best_loss = mean_loss
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f'{run_name}_best.pt')
            torch.save({'encoder_state': model.encoder_state_dict(),
                        'mae_loss': best_loss, 'config': vars(args)}, ckpt_path)

    logging.info(f"Best MAE loss: {best_loss:.5f}")
    logging.info(f"Checkpoint: {ckpt_dir}/{run_name}_best.pt")
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()

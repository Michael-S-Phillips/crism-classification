"""CLI driver for contrastive plag-vs-olivine encoder refinement.

Expects the three patch pools to already exist on disk
(``data/contrastive/{positives,hard_negatives,soft_negatives}/patches.npy``).
Build them first with ``scripts/build_contrastive_data.py``.

Usage:

  conda run -n crism python scripts/train_contrastive.py \\
      --pretrain_ckpt checkpoints/plag_aware_mae_128d_6l_best.pt \\
      --pool_dir data/contrastive \\
      --epochs 30 --batch_size 64 --lr 1e-4 \\
      --tau 0.07 --hard_weight 2.0 --soft_weight 1.0 \\
      --proj_dim 64 --run_name contrastive_plag_v1

Local CPU smoke test:

  conda run -n crism python scripts/train_contrastive.py \\
      --pretrain_ckpt checkpoints/plag_aware_mae_128d_6l_best.pt \\
      --pool_dir /tmp/contrastive_smoke \\
      --epochs 2 --batch_size 8 --device cpu --no_wandb \\
      --run_name smoke
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.contrastive_dataset import ContrastiveTripletDataset
from models.contrastive_encoder import ContrastiveEncoder
from training.contrastive_train import TrainConfig, train_contrastive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pretrain_ckpt', required=True,
                    help='MAE encoder warm-start. Loaded into the underlying '
                         'SpatialSpectralTransformer via load_encoder_state_dict.')
    ap.add_argument('--pool_dir', default='data/contrastive',
                    help='Directory containing positives/, hard_negatives/, '
                         'soft_negatives/ subdirs with patches.npy.')
    ap.add_argument('--output_dir', default='checkpoints',
                    help='Where to write {run_name}_best.pt / _last.pt.')
    ap.add_argument('--run_name', default='contrastive_plag_v1')
    # training
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--encoder_lr_scale', type=float, default=0.01)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    # loss
    ap.add_argument('--tau', type=float, default=0.07)
    ap.add_argument('--hard_weight', type=float, default=2.0)
    ap.add_argument('--soft_weight', type=float, default=1.0)
    ap.add_argument('--n_hard_per_batch', type=int, default=8)
    ap.add_argument('--n_soft_per_batch', type=int, default=8)
    # model
    ap.add_argument('--n_bands', type=int, default=59)
    ap.add_argument('--patch_size', type=int, default=7)
    ap.add_argument('--embed_dim', type=int, default=128)
    ap.add_argument('--n_heads', type=int, default=4)
    ap.add_argument('--n_layers', type=int, default=6)
    ap.add_argument('--proj_dim', type=int, default=64)
    ap.add_argument('--dropout', type=float, default=0.1)
    # noise augmentation
    ap.add_argument('--noise_aug', action='store_true',
                    help='Apply CrismNoiseAugmentation to all patches during '
                         'training (gaussian + 1µm spike + column bias). '
                         'Defaults match the denoising MAE pretraining.')
    ap.add_argument('--noise_sigma_gauss', type=float, default=0.0087)
    ap.add_argument('--noise_sigma_spike', type=float, default=0.0058)
    ap.add_argument('--noise_sigma_column', type=float, default=0.0049)
    # misc
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--no_wandb', action='store_true')
    ap.add_argument('--wandb_project', default='crism-mineral-classification')
    ap.add_argument('--wandb_entity', default='space-imagery-center')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    torch.manual_seed(args.seed)

    # ---------------------------------------------------------- data
    pos = os.path.join(args.pool_dir, 'positives', 'patches.npy')
    hard = os.path.join(args.pool_dir, 'hard_negatives', 'patches.npy')
    soft = os.path.join(args.pool_dir, 'soft_negatives', 'patches.npy')
    for p in (pos, hard, soft):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f'missing patch pool {p}. Build with scripts/build_contrastive_data.py.'
            )
    ds = ContrastiveTripletDataset(
        positives=pos,
        hard_negatives=hard,
        soft_negatives=soft,
        n_hard_per_batch=args.n_hard_per_batch,
        n_soft_per_batch=args.n_soft_per_batch,
        seed=args.seed,
    )
    print(f'positives={len(ds)}  hard={len(ds.hard_negatives)}  '
          f'soft={len(ds.soft_negatives)}')
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )

    # ---------------------------------------------------------- model
    model = ContrastiveEncoder(
        n_bands=args.n_bands, patch_size=args.patch_size,
        embed_dim=args.embed_dim, n_heads=args.n_heads, n_layers=args.n_layers,
        dropout=args.dropout, proj_dim=args.proj_dim,
    )
    ck = torch.load(args.pretrain_ckpt, map_location='cpu', weights_only=False)
    enc_state = ck.get('encoder_state', ck.get('model_state', ck))
    missing, unexpected = model.load_encoder_state_dict(enc_state)
    print(f'warm-started encoder; missing={len(missing)}  unexpected={len(unexpected)}')

    # ---------------------------------------------------------- wandb
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project, entity=args.wandb_entity,
                name=args.run_name, config=vars(args),
            )
        except Exception as e:                                          # pragma: no cover
            print(f'wandb disabled (init failed): {e}')

    # ---------------------------------------------------------- train
    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        encoder_lr_scale=args.encoder_lr_scale,
        tau=args.tau,
        hard_weight=args.hard_weight,
        soft_weight=args.soft_weight,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        grad_clip=args.grad_clip,
        device=args.device,
    )
    # ---------------------------------------------------------- noise aug
    noise_aug = None
    if args.noise_aug:
        from models.noise_augmentation import CrismNoiseAugmentation
        noise_aug = CrismNoiseAugmentation(
            sigma_gauss=args.noise_sigma_gauss,
            sigma_spike=args.noise_sigma_spike,
            sigma_column=args.noise_sigma_column,
            n_bands=args.n_bands,
            patch_size=args.patch_size,
        )
        print(f'noise_aug enabled: σ_gauss={args.noise_sigma_gauss} '
              f'σ_spike={args.noise_sigma_spike} σ_column={args.noise_sigma_column}')

    history = train_contrastive(
        model, loader, cfg,
        noise_aug=noise_aug,
        wandb_run=wandb_run,
        ckpt_dir=args.output_dir,
        run_name=args.run_name,
    )
    if wandb_run is not None:                                           # pragma: no cover
        wandb_run.finish()

    print('done')
    if history:
        print(f'final train_loss = {history[-1]["train_loss"]:.4f}')


if __name__ == '__main__':
    main()

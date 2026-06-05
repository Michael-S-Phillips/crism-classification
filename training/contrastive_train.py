"""InfoNCE loss + training loop for the contrastive plag-vs-olivine refinement.

The loss is a weighted InfoNCE: one positive per anchor, ``N_h`` hard negatives
(MC13-classifier-confident-plag pixels we believe are actually olivine), and
``N_s`` soft negatives (confirmed olivine). Hard negatives carry a larger
weight in the log-sum-exp denominator so the encoder is pushed harder to
separate them from the anchor.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# --------------------------------------------------------------------- loss
def info_nce_loss(
    z_anchor: torch.Tensor,
    z_pos: torch.Tensor,
    z_hard_neg: torch.Tensor,
    z_soft_neg: torch.Tensor,
    tau: float = 0.07,
    hard_weight: float = 2.0,
    soft_weight: float = 1.0,
) -> torch.Tensor:
    """Weighted InfoNCE with explicit hard/soft negative pools.

    All ``z_*`` arguments must be L2-normalised along the last dim.

    Parameters
    ----------
    z_anchor : (B, D)
    z_pos : (B, D) — one positive per anchor
    z_hard_neg : (B, N_h, D)
    z_soft_neg : (B, N_s, D)
    tau : temperature
    hard_weight, soft_weight : multiplicative weights applied (as a log-shift)
        to the hard/soft negative similarities in the denominator. Higher
        ``hard_weight`` pushes the model harder to separate hard negatives.

    Returns
    -------
    scalar loss tensor (mean over the batch)
    """
    if hard_weight <= 0 or soft_weight <= 0:
        raise ValueError("hard_weight and soft_weight must be > 0")
    if tau <= 0:
        raise ValueError("tau must be > 0")

    sim_pos = (z_anchor * z_pos).sum(dim=-1) / tau                       # (B,)
    sim_hard = torch.einsum('bd,bnd->bn', z_anchor, z_hard_neg) / tau    # (B, N_h)
    sim_soft = torch.einsum('bd,bnd->bn', z_anchor, z_soft_neg) / tau    # (B, N_s)

    log_hw = math.log(float(hard_weight))
    log_sw = math.log(float(soft_weight))
    log_denom = torch.logsumexp(
        torch.cat(
            [
                sim_pos.unsqueeze(-1),
                sim_hard + log_hw,
                sim_soft + log_sw,
            ],
            dim=-1,
        ),
        dim=-1,
    )
    return -(sim_pos - log_denom).mean()


# --------------------------------------------------------------------- loop
@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-4
    encoder_lr_scale: float = 0.01
    tau: float = 0.07
    hard_weight: float = 2.0
    soft_weight: float = 1.0
    weight_decay: float = 1e-4
    num_workers: int = 0
    grad_clip: float = 1.0
    log_every: int = 20            # steps between stdout prints
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def _param_groups(model: nn.Module, base_lr: float, encoder_lr_scale: float):
    """Slow LR on encoder, full LR on projection head."""
    proj_params = list(model.proj.parameters())
    proj_ids = {id(p) for p in proj_params}
    enc_params = [p for p in model.parameters() if id(p) not in proj_ids]
    return [
        {'params': enc_params, 'lr': base_lr * encoder_lr_scale},
        {'params': proj_params, 'lr': base_lr},
    ]


def train_contrastive(
    model: nn.Module,
    dataloader: DataLoader,
    cfg: TrainConfig,
    *,
    wandb_run=None,
    val_callback=None,
    val_every_epochs: int = 5,
    ckpt_dir: Optional[str] = None,
    run_name: str = 'contrastive',
    noise_aug: Optional[nn.Module] = None,
):
    """Train ``model`` (a ``ContrastiveEncoder``) with weighted InfoNCE.

    Parameters
    ----------
    model : ContrastiveEncoder
    dataloader : yields (anchor, positive, hard_neg, soft_neg) batches
    cfg : TrainConfig
    wandb_run : optional wandb.Run for logging (we never import wandb here)
    val_callback : optional ``fn(model, epoch) -> dict[str, float]`` invoked
        every ``val_every_epochs`` epochs; returned dict is logged.
    ckpt_dir : if set, writes ``{ckpt_dir}/{run_name}_best.pt`` after each epoch
        that improves the running loss, and ``{run_name}_last.pt`` every epoch.
    """
    device = torch.device(cfg.device)
    model.to(device)
    if noise_aug is not None:
        noise_aug = noise_aug.to(device)

    optim = torch.optim.AdamW(
        _param_groups(model, cfg.lr, cfg.encoder_lr_scale),
        weight_decay=cfg.weight_decay,
    )

    history = []
    best_loss = float('inf')
    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        ep_loss_sum = 0.0
        ep_n = 0
        t0 = time.time()
        for step, (anchor, pos, hard, soft) in enumerate(dataloader):
            anchor = anchor.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            hard = hard.to(device, non_blocking=True)
            soft = soft.to(device, non_blocking=True)

            B = anchor.shape[0]
            N_h = hard.shape[1]
            N_s = soft.shape[1]

            # Optional input-space noise augmentation. The CrismNoiseAugmentation
            # is no-op in eval mode; here we're in train mode so noise is applied.
            if noise_aug is not None:
                anchor = noise_aug(anchor)
                pos = noise_aug(pos)
                hard_flat = noise_aug(hard.reshape(B * N_h, *hard.shape[2:]))
                soft_flat = noise_aug(soft.reshape(B * N_s, *soft.shape[2:]))
            else:
                hard_flat = hard.reshape(B * N_h, *hard.shape[2:])
                soft_flat = soft.reshape(B * N_s, *soft.shape[2:])

            # Encode anchors and positives one patch per row, negatives flat-batched
            z_anchor = model(anchor)                                    # (B, D)
            z_pos = model(pos)                                          # (B, D)
            z_hard = model(hard_flat)                                   # (B*N_h, D)
            z_soft = model(soft_flat)                                   # (B*N_s, D)
            z_hard = z_hard.view(B, N_h, -1)
            z_soft = z_soft.view(B, N_s, -1)

            loss = info_nce_loss(
                z_anchor, z_pos, z_hard, z_soft,
                tau=cfg.tau,
                hard_weight=cfg.hard_weight,
                soft_weight=cfg.soft_weight,
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

            ep_loss_sum += float(loss.item()) * B
            ep_n += B
            global_step += 1
            if wandb_run is not None:
                wandb_run.log({'train/loss': float(loss.item()),
                               'train/epoch': epoch,
                               'train/step': global_step})
            if (step + 1) % cfg.log_every == 0:
                print(f'  epoch {epoch} step {step+1} loss={loss.item():.4f}')

        ep_loss = ep_loss_sum / max(ep_n, 1)
        dt = time.time() - t0
        print(f'epoch {epoch}: train_loss={ep_loss:.4f}  ({dt:.1f}s, n={ep_n})')
        record = {'epoch': epoch, 'train_loss': ep_loss, 'elapsed_sec': dt}

        if val_callback is not None and (epoch % val_every_epochs == 0 or epoch == cfg.epochs):
            try:
                val_metrics = val_callback(model, epoch) or {}
                record.update({f'val/{k}': float(v) for k, v in val_metrics.items()})
                if wandb_run is not None:
                    wandb_run.log({f'val/{k}': float(v) for k, v in val_metrics.items()})
                print(f'  val_callback: {val_metrics}')
            except Exception as e:                                     # pragma: no cover
                print(f'  val_callback failed: {e}')

        if wandb_run is not None:
            wandb_run.log({'train/epoch_loss': ep_loss, 'epoch': epoch})

        history.append(record)

        # Checkpoint handling
        if ckpt_dir is not None:
            os.makedirs(ckpt_dir, exist_ok=True)
            last_path = os.path.join(ckpt_dir, f'{run_name}_last.pt')
            torch.save(
                {
                    'model_state': model.state_dict(),
                    'encoder_state': model.encoder.state_dict(),
                    'epoch': epoch,
                    'train_loss': ep_loss,
                    'config': cfg.__dict__,
                },
                last_path,
            )
            if ep_loss < best_loss:
                best_loss = ep_loss
                best_path = os.path.join(ckpt_dir, f'{run_name}_best.pt')
                torch.save(
                    {
                        'model_state': model.state_dict(),
                        'encoder_state': model.encoder.state_dict(),
                        'epoch': epoch,
                        'train_loss': ep_loss,
                        'config': cfg.__dict__,
                    },
                    best_path,
                )

    return history

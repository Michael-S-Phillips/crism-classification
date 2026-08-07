# scripts/pretrain_plag_aware_mae.py
"""Plag-aware multi-task pretraining: denoising MAE recon + 5-class ASL aux.

Dual stream per step:
  - Stream U: unlabeled global patch cache -> recon loss only
  - Stream L: labeled mrral patches        -> recon loss + aux ASL loss
Total: recon(U) + recon(L) + lambda * ASL(aux_logits_L, labels_L)
lambda ramps 0 -> lambda_target over --aux_warmup epochs.

Warm-starts encoder+decoder from a denoising-MAE checkpoint (--init).

Usage (HPC):
  python scripts/pretrain_plag_aware_mae.py \\
    --init checkpoints/spatial_mae_denoising_128d_6l_best.pt \\
    --epochs 40 --aux_warmup 5 --lambda_target 1.0 \\
    --plag_class_weight 5.0 --run_name plag_aware_mae_128d_6l
"""
import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from device import get_device
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=str, required=True,
                    help="denoising-MAE checkpoint to warm-start from")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=5, help="LR warmup epochs")
    ap.add_argument("--aux_warmup", type=int, default=5, help="lambda ramp epochs")
    ap.add_argument("--lambda_target", type=float, default=1.0)
    ap.add_argument("--plag_class_weight", type=float, default=5.0,
                    help="ASL class_weight on plagioclase (index 3); others 1.0")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--labeled_batch_size", type=int, default=256)
    ap.add_argument("--steps_per_epoch", type=int, default=400)
    ap.add_argument("--monitor_frac", type=float, default=0.03,
                    help="fraction of train split held out for plag-AP checkpoint selection")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--mask_ratio", type=float, default=0.75)
    ap.add_argument("--asl_gamma_neg", type=float, default=4.0)
    ap.add_argument("--asl_gamma_pos", type=float, default=0.0)
    ap.add_argument("--asl_clip", type=float, default=0.05)
    ap.add_argument("--run_name", type=str, default="plag_aware_mae_128d_6l")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--no_wandb", action="store_true")
    args = ap.parse_args()

    from config_loader import load_config
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            args.config)
    cfg = load_config(cfg_path)
    device = get_device()
    log.info(f"device={device}")

    # ── Stream U: unlabeled global cache (recon only) ────────────────────────
    shard_dir = cfg.get("global_patch_cache_dir")
    if not shard_dir:
        raise KeyError("config must define global_patch_cache_dir")
    from data.cached_patch_dataset import CRISMCachedPatchDataset
    ds_u = CRISMCachedPatchDataset(shard_dir=shard_dir, normalize=True, shuffle=True)
    loader_u = DataLoader(ds_u, batch_size=args.batch_size,
                          num_workers=args.num_workers,
                          pin_memory=torch.cuda.is_available(),
                          persistent_workers=args.num_workers > 0,
                          prefetch_factor=4 if args.num_workers > 0 else None)

    # ── Stream L: labeled mrral patches (recon + aux) ────────────────────────
    # Carve a small monitoring slice OUT of the train split (never the official
    # val split — that stays clean for the 3-way fine-tuning comparison). The
    # monitoring slice is used only for checkpoint selection by plag AP.
    from data.dataset import CRISMSpectralPatchDataset, LABEL_COLS
    parquet = os.path.join(cfg["output_dir"], "mrral_pixels.parquet")
    df = pd.read_parquet(parquet)
    train_all = df[df["split"] == "train"].reset_index(drop=True)
    mon_rng = np.random.default_rng(42)
    mon_mask = mon_rng.random(len(train_all)) < args.monitor_frac
    train_mon = train_all[mon_mask].reset_index(drop=True)
    train_core = train_all[~mon_mask].reset_index(drop=True)
    mrral_map = _build_mrral_map(cfg)
    cache_dir = cfg.get("patch_cache_dir")
    # NOTE: the patch-cache memmap is row-aligned to the FULL train split, so the
    # boolean-masked sub-frames must NOT use the cache (indices would misalign).
    # Pass cache_dir=None so these read patches live from tiles via rasterio.
    ds_l = CRISMSpectralPatchDataset(train_core, mrral_map, patch_size=7,
                                     cache_dir=None, split="train")
    ds_mon = CRISMSpectralPatchDataset(train_mon, mrral_map, patch_size=7,
                                       cache_dir=None, split="train")
    loader_l = DataLoader(ds_l, batch_size=args.labeled_batch_size, shuffle=True,
                          num_workers=args.num_workers,
                          pin_memory=torch.cuda.is_available(),
                          persistent_workers=args.num_workers > 0,
                          prefetch_factor=4 if args.num_workers > 0 else None,
                          drop_last=True)
    loader_mon = DataLoader(ds_mon, batch_size=512, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=torch.cuda.is_available())
    log.info(f"labeled train-core rows: {len(ds_l):,}; monitor rows: {len(ds_mon):,}; "
             f"core plag positives {int((train_core['plagioclase'] > 0).sum()):,}")

    # ── Model (warm-start) ───────────────────────────────────────────────────
    from models.multitask_denoising_mae import MultiTaskDenoisingMAE
    model = MultiTaskDenoisingMAE(
        n_bands=59, patch_size=7, embed_dim=args.embed_dim,
        n_heads=args.n_heads, n_layers=args.n_layers,
        decoder_dim=64, decoder_layers=2, mask_ratio=args.mask_ratio, n_classes=5,
    ).to(device)
    ckpt = torch.load(args.init, map_location=device, weights_only=False)
    # warm-start the inherited MAE weights; aux_head stays random
    missing, unexpected = model.load_state_dict(ckpt["mae_state"], strict=False)
    log.info(f"warm-start from {args.init}: missing={missing} unexpected={unexpected}")
    assert all(k.startswith("aux_head") for k in missing), \
        f"unexpected missing keys beyond aux_head: {missing}"

    # ── Loss / optim ─────────────────────────────────────────────────────────
    from training.losses import AsymmetricLoss
    asl = AsymmetricLoss(gamma_neg=args.asl_gamma_neg, gamma_pos=args.asl_gamma_pos,
                         clip=args.asl_clip)
    class_weights = torch.ones(5, device=device)
    class_weights[LABEL_COLS.index("plagioclase")] = args.plag_class_weight

    base_lr = 1.5e-4 * args.batch_size / 256
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.95),
                            weight_decay=0.05)

    def lr_lambda(epoch):
        if epoch < args.warmup:
            return (epoch + 1) / args.warmup
        progress = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project="crism-mineral-classification",
                       entity=cfg.get("wandb", {}).get("entity") or None,
                       name=args.run_name, config=vars(args))
        except Exception as e:
            log.warning(f"wandb off ({e})"); use_wandb = False

    from sklearn.metrics import average_precision_score
    PLAG = LABEL_COLS.index("plagioclase")

    @torch.no_grad()
    def monitor_plag_ap():
        """Plag AP on the held-out train monitoring slice (full-visibility aux head)."""
        model.eval()
        scores, targets = [], []
        for xb, yb, _wb in loader_mon:
            logits = model.forward_aux(xb.to(device))
            scores.append(torch.sigmoid(logits[:, PLAG]).cpu().numpy())
            targets.append(yb[:, PLAG].numpy())
        model.train()
        y = np.concatenate(targets); p = np.concatenate(scores)
        if y.sum() == 0:
            return float("nan")
        return float(average_precision_score(y, p))

    ckpt_dir = cfg.get("checkpoints_dir")
    os.makedirs(ckpt_dir, exist_ok=True)
    it_u, it_l = iter(loader_u), iter(loader_l)
    best_ap = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        lam = args.lambda_target * min(1.0, epoch / max(1, args.aux_warmup))
        rec_losses, aux_losses = [], []
        for _ in range(args.steps_per_epoch):
            try:
                xu = next(it_u)
            except StopIteration:
                it_u = iter(loader_u); xu = next(it_u)
            try:
                xl, yl, wl = next(it_l)
            except StopIteration:
                it_l = iter(loader_l); xl, yl, wl = next(it_l)
            xu = xu.to(device); xl = xl.to(device)
            yl = yl.to(device); wl = wl.to(device)

            opt.zero_grad()
            recon_u, _, _ = model(xu)
            recon_l, _, _ = model(xl)
            recon = recon_u + recon_l
            aux_logits = model.forward_aux(xl)
            aux = asl(aux_logits, yl, wl, class_weights=class_weights)
            loss = recon + lam * aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            rec_losses.append(float(recon)); aux_losses.append(float(aux))

        sched.step()
        mrec, maux = float(np.mean(rec_losses)), float(np.mean(aux_losses))
        mon_ap = monitor_plag_ap()
        lr_now = opt.param_groups[0]["lr"]
        log.info(f"epoch {epoch}/{args.epochs} | recon={mrec:.6f} | aux={maux:.6f} "
                 f"| monitor_plag_AP={mon_ap:.4f} | lambda={lam:.3f} | lr={lr_now:.2e}")
        if use_wandb:
            import wandb
            wandb.log({"epoch": epoch, "recon": mrec, "aux": maux,
                       "monitor_plag_AP": mon_ap, "lambda": lam, "lr": lr_now})

        # Select best by held-out monitoring plag AP (recon logged as a guardrail).
        if mon_ap == mon_ap and mon_ap > best_ap:   # not NaN and improved
            best_ap = mon_ap
            path = os.path.join(ckpt_dir, f"{args.run_name}_best.pt")
            torch.save({"mae_state": model.state_dict(),
                        "encoder_state": model.encoder_state_dict(),
                        "epoch": epoch, "recon": mrec, "aux": maux,
                        "monitor_plag_AP": mon_ap, "config": vars(args)}, path)
            log.info(f"saved best (monitor_plag_AP={mon_ap:.4f}) -> {path}")


def _build_mrral_map(cfg):
    import glob
    data_root = cfg["data_root"]
    hdrs = sorted(set(glob.glob(os.path.join(data_root, "mc*", "t*mrral*.hdr"))
                      + glob.glob(os.path.join(data_root, "t*mrral*.hdr"))))
    return {os.path.basename(h).split("_mrral_")[0]: h.replace(".hdr", ".img")
            for h in hdrs}


if __name__ == "__main__":
    main()

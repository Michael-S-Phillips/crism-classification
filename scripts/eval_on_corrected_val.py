"""Score a fine-tuned checkpoint on the val split, optionally with relabels applied.

Used to put the pre-relabel champion (ft_plag_aware_real_only) and the relabeled
re-run on the SAME corrected-val yardstick, isolating the model gain from the
label-correction effect. Uses the exact metric code (evaluation.metrics) and
target binarization (>0.4) that train_torch.py uses, so numbers are directly
comparable to wandb val_AP.

Usage:
  conda run -n crism python scripts/eval_on_corrected_val.py \\
    --ckpt checkpoints/ft_plag_aware_real_only_best.pt \\
    --apply_relabels data/olivine_relabels.csv          # corrected val
  conda run -n crism python scripts/eval_on_corrected_val.py \\
    --ckpt checkpoints/ft_plag_aware_real_only_best.pt   # original val
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import load_config
from data.dataset import (CRISMSpectralPatchDataset, LABEL_COLS, _collapse_labels,
                          label_cols_for_ckpt,
                          apply_olivine_relabels)
from evaluation.metrics import compute_full_metrics
from models.spatial_spectral_transformer import SpatialSpectralClassifier


def build_mrral_map(cfg):
    data_root = cfg.get('data_root', '/mnt/mrdr')
    hdrs = sorted(set(glob.glob(os.path.join(data_root, 'mc*', 't*mrral*.hdr'))
                      + glob.glob(os.path.join(data_root, 't*mrral*.hdr'))))
    return {os.path.basename(h).split('_mrral_')[0]: h.replace('.hdr', '.img')
            for h in hdrs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--apply_relabels', default=None)
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--batch_size', type=int, default=512)
    args = ap.parse_args()

    cfg = load_config(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the checkpoint FIRST and size the label vocabulary from its head.
    # data.dataset.LABEL_COLS must be rebound before the dataset is built so
    # a 6-class ckpt gets 6-column label tensors (incl. alteration).
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ck['model_state'] if 'model_state' in ck else ck
    global LABEL_COLS
    LABEL_COLS = label_cols_for_ckpt(state)
    import data.dataset
    data.dataset.LABEL_COLS = list(LABEL_COLS)
    print(f'checkpoint head: {len(LABEL_COLS)}-class {LABEL_COLS}')

    df = pd.read_parquet(os.path.join(cfg['output_dir'], 'mrral_pixels.parquet'))
    if args.apply_relabels:
        df, n = apply_olivine_relabels(df, args.apply_relabels)
        print(f'applied relabels: {n} pixels updated (CORRECTED val labels)')
    else:
        print('no relabels (ORIGINAL val labels)')

    df = _collapse_labels(df)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    tiers = val_df['confidence_tier'].astype(str).tolist()

    ds = CRISMSpectralPatchDataset(val_df, build_mrral_map(cfg), patch_size=7,
                                   cache_dir=cfg.get('patch_cache_dir'), split='val')
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SpatialSpectralClassifier(n_bands=59, patch_size=7,
                                      n_classes=len(LABEL_COLS),
                                      embed_dim=128, n_heads=4, n_layers=6).to(device)
    model.load_state_dict(state)
    model.eval()

    ys, ts = [], []
    with torch.no_grad():
        for feats, labels, _w in loader:
            logits = model(feats.to(device))
            ys.append(torch.sigmoid(logits).cpu().numpy())
            ts.append(labels.numpy())
    y_score = np.concatenate(ys); y_true = np.concatenate(ts)

    m = compute_full_metrics(y_true, y_score, tiers)
    print(f'\nckpt: {os.path.basename(args.ckpt)}  '
          f'({"corrected" if args.apply_relabels else "original"} val, n={len(y_true):,})')
    print(f'  val_mAP = {m["mAP"]:.4f}')
    for cls, apv in m['per_class_ap'].items():
        print(f'  val_AP_{cls:<12s} = {apv:.4f}')


if __name__ == '__main__':
    main()

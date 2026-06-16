"""
Evaluate a saved spatial_vit checkpoint on the test split.

Usage:
    conda run -n crism python scripts/eval_test_split.py \\
        --ckpt checkpoints/spvit_lrscale01_v4_fixed_best.pt

Reports per-class test AP, score statistics on HCP-positive pixels (to
detect HCP→LCP cannibalization), and per-tile HCP performance.
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
from data.dataset import (CRISMSpectralPatchDataset, LABEL_COLS,
                          label_cols_for_ckpt)
from evaluation.metrics import compute_per_class_ap, compute_map
from models.spatial_spectral_transformer import SpatialSpectralClassifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to *_best.pt checkpoint")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--parquet", default=None,
                        help="Override default mrral_pixels.parquet path. Use to "
                             "evaluate on a different data lineage (e.g. mc11val).")
    parser.add_argument("--patch_cache_dir", default=None,
                        help="Override cfg.patch_cache_dir. Must match --parquet "
                             "if a non-default parquet is given.")
    args = parser.parse_args()

    cfg = load_config()

    # Load the checkpoint FIRST and size the label vocabulary from its head.
    # data.dataset.LABEL_COLS must be rebound before the dataset is built so
    # a 6-class ckpt gets 6-column label tensors (incl. alteration).
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    global LABEL_COLS
    LABEL_COLS = label_cols_for_ckpt(state)
    import data.dataset
    data.dataset.LABEL_COLS = list(LABEL_COLS)
    print(f"checkpoint head: {len(LABEL_COLS)}-class {LABEL_COLS}")

    parquet_path = args.parquet or os.path.join(cfg["output_dir"], "mrral_pixels.parquet")
    print(f"parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df_test = df[df["split"] == "test"].reset_index(drop=True)
    print(f"test rows: {len(df_test):,}")

    # Build tile_id -> .img path map. Support both nested and flat layouts.
    data_root = cfg["data_root"]
    mrral_hdrs = sorted(set(
        glob.glob(os.path.join(data_root, "mc*", "t*mrral*.hdr"))
        + glob.glob(os.path.join(data_root, "t*mrral*.hdr"))
    ))
    mrral_map = {}
    for hdr in mrral_hdrs:
        tid = os.path.basename(hdr).split("_mrral_")[0]
        mrral_map[tid] = hdr.replace(".hdr", ".img")
    print(f"mrral_map: {len(mrral_map)} tiles")

    cache_dir = args.patch_cache_dir or cfg.get("patch_cache_dir")
    ds = CRISMSpectralPatchDataset(
        df_test, mrral_map, patch_size=7,
        cache_dir=cache_dir, split="test",
    )
    print(f"dataset len: {len(ds)} (cache hit: {ds._cache is not None})")

    model = SpatialSpectralClassifier(
        n_bands=59, patch_size=7, n_classes=len(LABEL_COLS),
        embed_dim=128, n_heads=4, n_layers=6, dropout=0.1,
    )
    model.load_state_dict(state)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"device: {device}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    y_true_l, y_score_l = [], []
    with torch.no_grad():
        for i, (x, y, w) in enumerate(loader):
            x = x.to(device)
            # CRISMSpectralPatchDataset returns (B, 7, 7, 59); SpatialSpectralClassifier
            # expects exactly that format — no permute needed.
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_true_l.append(y.numpy())
            y_score_l.append(probs)
            if i % 20 == 0:
                print(f"  batch {i + 1}/{len(loader)}")

    y_true = np.concatenate(y_true_l, 0)
    y_score = np.concatenate(y_score_l, 0)
    print(f"shapes: y_true={y_true.shape}, y_score={y_score.shape}")

    per_class = compute_per_class_ap(y_true, y_score)
    mAP = compute_map(y_true, y_score)
    print()
    print(f"=== Test metrics for {os.path.basename(args.ckpt)} ===")
    print(f"test_mAP: {mAP:.4f}")
    for cls, ap in per_class.items():
        n_pos = int((y_true[:, LABEL_COLS.index(cls)] > 0.4).sum())
        print(f"  {cls:14s}: AP = {ap:.4f}  (n_pos={n_pos:,})")

    # HCP/LCP cannibalization check
    hcp_idx = LABEL_COLS.index("hcp")
    lcp_idx = LABEL_COLS.index("lcp")
    oli_idx = LABEL_COLS.index("olivine")
    plag_idx = LABEL_COLS.index("plagioclase")

    hcp_mask = y_true[:, hcp_idx] > 0.4
    lcp_mask = y_true[:, lcp_idx] > 0.4
    print()
    print("=== Score statistics on HCP-positive test pixels ===")
    print(f"  n_hcp_pos: {hcp_mask.sum():,}")
    if hcp_mask.sum() > 0:
        print(f"  mean P(hcp) on HCP+: {y_score[hcp_mask, hcp_idx].mean():.4f}")
        print(f"  mean P(lcp) on HCP+: {y_score[hcp_mask, lcp_idx].mean():.4f}")
        print(f"  mean P(olivine) on HCP+: {y_score[hcp_mask, oli_idx].mean():.4f}")
        print(f"  mean P(plagioclase) on HCP+: {y_score[hcp_mask, plag_idx].mean():.4f}")

    print()
    print("=== Score statistics on LCP-positive test pixels ===")
    print(f"  n_lcp_pos: {lcp_mask.sum():,}")
    if lcp_mask.sum() > 0:
        print(f"  mean P(lcp) on LCP+: {y_score[lcp_mask, lcp_idx].mean():.4f}")
        print(f"  mean P(hcp) on LCP+: {y_score[lcp_mask, hcp_idx].mean():.4f}")
        print(f"  mean P(olivine) on LCP+: {y_score[lcp_mask, oli_idx].mean():.4f}")
        print(f"  mean P(plagioclase) on LCP+: {y_score[lcp_mask, plag_idx].mean():.4f}")

    # Per-tile HCP performance
    from sklearn.metrics import average_precision_score
    print()
    print("=== Per-tile HCP performance ===")
    tile_ids = df_test["tile_id"].values
    for tid in sorted(set(tile_ids)):
        mask = tile_ids == tid
        if mask.sum() == 0:
            continue
        yt = (y_true[mask, hcp_idx] > 0.4).astype(int)
        if yt.sum() == 0:
            print(f"  {tid}: no HCP positives ({mask.sum()} pixels)")
            continue
        ap = average_precision_score(yt, y_score[mask, hcp_idx])
        print(f"  {tid}: HCP AP={ap:.4f}  (n_pos={yt.sum()}/{mask.sum()})")


if __name__ == "__main__":
    main()

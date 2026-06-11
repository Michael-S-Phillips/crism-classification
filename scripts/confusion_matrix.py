"""Confusion matrix on the val split: argmax-predicted vs argmax-true class.

Defaults to the *uncorrected* val. Pass --apply_relabels to use the corrected one.
Useful for answering questions like: "for true-plagioclase pixels, what class is
the model actually predicting?"

Usage:
    conda run -n crism python scripts/confusion_matrix.py \\
        --ckpt checkpoints/ft_plag_aware_relabeled_best.pt
"""
import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import load_config
from data.dataset import (CRISMSpectralPatchDataset, LABEL_COLS, _collapse_labels,
                          apply_olivine_relabels)
from models.spatial_spectral_transformer import SpatialSpectralClassifier


def build_mrral_map(cfg):
    data_root = cfg.get("data_root", "/mnt/mrdr")
    hdrs = sorted(set(glob.glob(os.path.join(data_root, "mc*", "t*mrral*.hdr"))
                      + glob.glob(os.path.join(data_root, "t*mrral*.hdr"))))
    return {os.path.basename(h).split("_mrral_")[0]: h.replace(".hdr", ".img")
            for h in hdrs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--apply_relabels", default=None,
                    help="Path to relabel CSV (use corrected val). Omit for original val.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--min_label", type=float, default=0.4,
                    help="Skip pixels where max(soft labels) < this (ambiguous).")
    ap.add_argument("--out_dir", default="reports")
    args = ap.parse_args()

    cfg = load_config(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_parquet(os.path.join(cfg["output_dir"], "mrral_pixels.parquet"))
    val_kind = "original"
    if args.apply_relabels:
        df, n = apply_olivine_relabels(df, args.apply_relabels)
        print(f"applied relabels: {n} pixels updated (CORRECTED val labels)")
        val_kind = "corrected"
    else:
        print("no relabels (ORIGINAL val labels)")

    df = _collapse_labels(df)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    print(f"val pixels (pre-filter): {len(val_df):,}")

    ds = CRISMSpectralPatchDataset(val_df, build_mrral_map(cfg), patch_size=7,
                                   cache_dir=cfg.get("patch_cache_dir"), split="val")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SpatialSpectralClassifier(n_bands=59, patch_size=7, n_classes=5,
                                      embed_dim=128, n_heads=4, n_layers=6).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"] if "model_state" in ck else ck)
    model.eval()

    ys, ts = [], []
    with torch.no_grad():
        for feats, labels, _w in loader:
            logits = model(feats.to(device))
            ys.append(torch.sigmoid(logits).cpu().numpy())
            ts.append(labels.numpy())
    y_score = np.concatenate(ys)
    y_true = np.concatenate(ts)
    assert y_true.shape[1] == len(LABEL_COLS), f"label cols mismatch: {LABEL_COLS}"

    # argmax class indices
    pred_idx = y_score.argmax(axis=1)
    true_idx = y_true.argmax(axis=1)
    max_lab = y_true.max(axis=1)

    keep = max_lab >= args.min_label
    dropped = int((~keep).sum())
    print(f"dropped {dropped:,} pixels with max(soft_label) < {args.min_label} (ambiguous)")
    pred_idx = pred_idx[keep]
    true_idx = true_idx[keep]
    n = len(true_idx)
    print(f"scored pixels: {n:,}")

    K = len(LABEL_COLS)
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(true_idx, pred_idx):
        cm[t, p] += 1
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

    # ---- print summary ----
    print(f"\n=== Confusion matrix (rows = true class, cols = predicted class) ===")
    header = "true\\pred  " + " ".join(f"{c:>12}" for c in LABEL_COLS) + "    total"
    print(header)
    for i, cls in enumerate(LABEL_COLS):
        row = " ".join(f"{cm[i, j]:>12,d}" for j in range(K))
        print(f"{cls:<10}" + row + f"  {cm[i].sum():>8,d}")

    print(f"\n=== Row-normalized (each row sums to 1) ===")
    print(header.replace("    total", ""))
    for i, cls in enumerate(LABEL_COLS):
        row = " ".join(f"{cm_norm[i, j]:>12.3f}" for j in range(K))
        print(f"{cls:<10}" + row)

    # overall accuracy
    acc = float(cm.diagonal().sum() / max(cm.sum(), 1))
    print(f"\noverall argmax accuracy: {acc:.4f}")

    # ---- save outputs ----
    stem = os.path.splitext(os.path.basename(args.ckpt))[0]
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir,
                            f"confusion_matrix_{stem}_{val_kind}.csv")
    png_path = os.path.join(args.out_dir,
                            f"confusion_matrix_{stem}_{val_kind}.png")
    pd.DataFrame(cm, index=LABEL_COLS, columns=LABEL_COLS).to_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, mat, title, fmt in (
        (axes[0], cm, f"Raw counts (n={n:,})", "d"),
        (axes[1], cm_norm, "Row-normalized (P(pred | true))", ".2f"),
    ):
        im = ax.imshow(mat, cmap="viridis", aspect="auto")
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        ax.set_xticklabels(LABEL_COLS, rotation=30, ha="right")
        ax.set_yticklabels(LABEL_COLS)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(title)
        for i in range(K):
            for j in range(K):
                v = mat[i, j]
                color = "white" if v < (mat.max() * 0.6) else "black"
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        color=color, fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(f"{stem}  ·  {val_kind} val  ·  acc={acc:.3f}", fontsize=11)
    plt.tight_layout()
    fig.savefig(png_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\nwrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()

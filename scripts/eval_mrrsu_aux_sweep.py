"""Collate val-split metrics across the mrrsu-aux normalization sweep.

For each normalization mode (zscore / minmax / pertile_zscore), expects a
checkpoint at ``checkpoints/ft_mrrsu_aux_{mode}_best.pt`` plus the matching aux
cache at ``data/patch_cache_{mode}/`` (built by the slurm job). Runs the same
val-split scoring loop as ``eval_on_corrected_val.py`` but for the aux model
(SpatialSpectralClassifierAux), then writes a single markdown report with mAP
and per-class AP for each mode.

Usage:
  conda run -n crism python scripts/eval_mrrsu_aux_sweep.py
  conda run -n crism python scripts/eval_mrrsu_aux_sweep.py --apply_relabels \\
      data/olivine_relabels.csv
  conda run -n crism python scripts/eval_mrrsu_aux_sweep.py --dry_run
"""
import argparse
import glob
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


MODES = ("zscore", "minmax", "pertile_zscore")
REPORT_PATH = os.path.join("reports", "mrrsu_aux_norm_sweep_results.md")


def _checkpoint_path(mode: str) -> str:
    return os.path.join("checkpoints", f"ft_mrrsu_aux_{mode}_best.pt")


def _aux_dir(mode: str) -> str:
    return os.path.join("data", f"patch_cache_{mode}")


def _build_mrral_map(cfg):
    data_root = cfg.get('data_root', '/Volumes/Mars_GIS/CRISM/MRDR')
    hdrs = sorted(set(
        glob.glob(os.path.join(data_root, 'mc*', 't*mrral*.hdr'))
        + glob.glob(os.path.join(data_root, 't*mrral*.hdr'))
    ))
    return {os.path.basename(h).split('_mrral_')[0]: h.replace('.hdr', '.img')
            for h in hdrs}


def evaluate_one(ckpt: str, aux_dir: str, cfg, apply_relabels: Optional[str],
                 batch_size: int) -> dict:
    """Score a single (ckpt, aux_dir) pair on the val split."""
    import numpy as np
    import pandas as pd
    import torch
    import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from device import get_device
    from torch.utils.data import DataLoader

    from data.dataset import (LABEL_COLS, MrrsuAuxPatchDataset, _collapse_labels,
                              apply_olivine_relabels)
    from evaluation.metrics import compute_full_metrics
    from models.spatial_spectral_classifier_aux import SpatialSpectralClassifierAux

    device = get_device()

    df = pd.read_parquet(os.path.join(cfg['output_dir'], 'mrral_pixels.parquet'))
    if apply_relabels:
        df, n = apply_olivine_relabels(df, apply_relabels)
        print(f'  applied relabels: {n} pixels updated')
    df = _collapse_labels(df)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    tiers = val_df['confidence_tier'].astype(str).tolist()

    mrral_map = _build_mrral_map(cfg)
    aux_npy = os.path.join(aux_dir, 'mrrsu_aux_val.npy')
    stats_json = os.path.join(aux_dir, 'mrrsu_aux_stats.json')
    ds = MrrsuAuxPatchDataset(
        val_df, mrral_map, patch_size=7,
        aux_npy=aux_npy, stats_json=stats_json,
        cache_dir=cfg.get('patch_cache_dir'), split='val',
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SpatialSpectralClassifierAux(
        n_bands=59, patch_size=7, n_classes=5,
        embed_dim=128, n_heads=4, n_layers=6,
    ).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state'] if 'model_state' in ck else ck)
    model.eval()

    ys, ts = [], []
    with torch.no_grad():
        for feats, aux, labels, _w in loader:
            logits = model(feats.to(device), aux.to(device))
            ys.append(torch.sigmoid(logits).cpu().numpy())
            ts.append(labels.numpy())
    y_score = np.concatenate(ys)
    y_true = np.concatenate(ts)
    m = compute_full_metrics(y_true, y_score, tiers)
    return {
        "mAP": float(m['mAP']),
        "per_class_ap": {k: float(v) for k, v in m['per_class_ap'].items()},
        "n_val": int(len(y_true)),
        "checkpoint": ckpt,
        "ckpt_val_mAP": (float(ck.get('val_mAP'))
                        if isinstance(ck, dict) and ck.get('val_mAP') is not None
                        else None),
    }


def write_report(results: dict, out_path: str, apply_relabels: Optional[str]) -> None:
    """Write a markdown summary of mAP + per-class AP per mode."""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    classes = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']
    lines = []
    lines.append("# mrrsu-aux normalization ablation results\n")
    lines.append(f"Val labels: {'corrected (relabels applied)' if apply_relabels else 'original'}\n")

    # Summary table
    lines.append("| mode | status | val_mAP | " + " | ".join(f"AP_{c}" for c in classes) + " |")
    lines.append("|" + "---|" * (len(classes) + 3))
    for mode in MODES:
        r = results.get(mode)
        if r is None or r.get("error"):
            err = r.get("error") if r else "missing"
            lines.append(f"| {mode} | ERROR ({err}) | -- | " + " | ".join("--" for _ in classes) + " |")
            continue
        per = r["per_class_ap"]
        ap_cells = " | ".join(f"{per.get(c, float('nan')):.4f}" for c in classes)
        lines.append(f"| {mode} | ok | {r['mAP']:.4f} | {ap_cells} |")
    lines.append("")

    # Detail blocks
    for mode in MODES:
        r = results.get(mode)
        lines.append(f"## {mode}")
        if r is None:
            lines.append("_(no result -- checkpoint or cache missing)_\n")
            continue
        if r.get("error"):
            lines.append(f"_ERROR: {r['error']}_\n")
            continue
        lines.append(f"- checkpoint: `{r['checkpoint']}`")
        lines.append(f"- n_val: {r['n_val']:,}")
        if r.get("ckpt_val_mAP") is not None:
            lines.append(f"- ckpt-recorded val_mAP: {r['ckpt_val_mAP']:.4f}")
        lines.append(f"- recomputed val_mAP: {r['mAP']:.4f}")
        for c in classes:
            lines.append(f"  - AP_{c}: {r['per_class_ap'].get(c, float('nan')):.4f}")
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--apply_relabels', default=None,
                    help="optional CSV of olivine relabels (see eval_on_corrected_val.py)")
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--output', default=REPORT_PATH,
                    help="path to the output markdown report")
    ap.add_argument('--dry_run', action='store_true',
                    help="don't load any checkpoints; just write a placeholder "
                         "markdown noting which sweep slots are missing.")
    args = ap.parse_args()

    if args.dry_run:
        placeholder = {
            mode: {"error": "dry_run (no evaluation performed)"}
            for mode in MODES
        }
        write_report(placeholder, args.output, args.apply_relabels)
        return

    from config_loader import load_config
    cfg = load_config(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.config))

    results = {}
    for mode in MODES:
        ckpt = _checkpoint_path(mode)
        aux_dir = _aux_dir(mode)
        print(f"\n=== mode={mode} ===")
        if not os.path.exists(ckpt):
            print(f"  SKIP: missing checkpoint {ckpt}")
            results[mode] = {"error": f"missing checkpoint {ckpt}"}
            continue
        if not os.path.exists(os.path.join(aux_dir, 'mrrsu_aux_stats.json')):
            print(f"  SKIP: missing aux cache {aux_dir}")
            results[mode] = {"error": f"missing aux cache {aux_dir}"}
            continue
        try:
            results[mode] = evaluate_one(
                ckpt, aux_dir, cfg, args.apply_relabels, args.batch_size,
            )
            print(f"  val_mAP={results[mode]['mAP']:.4f}")
        except Exception as e:  # pragma: no cover - operational
            print(f"  ERROR evaluating {mode}: {e}")
            results[mode] = {"error": str(e)}

    write_report(results, args.output, args.apply_relabels)


if __name__ == '__main__':
    main()

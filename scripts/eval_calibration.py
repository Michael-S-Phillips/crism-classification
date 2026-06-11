"""Calibration metrics for CRISM mineral classifiers on the val split.

For each of the 5 classes:

  * Brier score: mean squared error between predicted positive-class probability
    and the {0,1} positive indicator.
  * Expected Calibration Error (ECE): 15 equal-width confidence bins, weighted
    mean of |bin_confidence − bin_empirical_positive_rate|.
  * Reliability diagram: a single 2x3 grid of per-class diagrams + an overall
    one, plotted into ``reliability.png``.

We use the same val frame, label collapse, optional relabel path, and patch
dataset as ``eval_on_corrected_val.py`` so calibration numbers line up
1-to-1 with the AP numbers wandb reports.

For the contrastive encoder path we accept the same ``--probe_head_path`` /
inline-training hooks as ``eval_polygon_accuracy.py``.

Usage:

  conda run -n crism python scripts/eval_calibration.py \\
      --ckpt checkpoints/ft_plag_aware_real_only_best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import load_config
from data.dataset import (CRISMSpectralPatchDataset, LABEL_COLS,
                          _collapse_labels, apply_olivine_relabels)

# Reuse the model-loading machinery from the polygon harness so we have one
# canonical code path for "classifier vs contrastive encoder + probe".
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_polygon_accuracy import (build_mrral_map, detect_ckpt_kind,
                                   load_model_auto)

N_CLASSES = len(LABEL_COLS)


def _make_val_loader(cfg, batch_size: int, apply_relabels: str | None,
                     debug_rows: int | None = None):
    df = pd.read_parquet(os.path.join(cfg['output_dir'], 'mrral_pixels.parquet'))
    if apply_relabels:
        df, n = apply_olivine_relabels(df, apply_relabels)
        print(f'applied relabels: {n} pixels updated')
    df = _collapse_labels(df)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    if debug_rows is not None:
        val_df = val_df.head(debug_rows).reset_index(drop=True)
    cache_dir = None if debug_rows else cfg.get('patch_cache_dir')
    ds = CRISMSpectralPatchDataset(val_df, build_mrral_map(cfg),
                                    patch_size=7, cache_dir=cache_dir, split='val')
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return loader, val_df


def score_val(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    ys, ts = [], []
    model.eval()
    with torch.no_grad():
        for feats, labels, _w in loader:
            logits = model(feats.to(device))
            ys.append(torch.sigmoid(logits).cpu().numpy())
            ts.append(labels.numpy())
    return np.concatenate(ys), np.concatenate(ts)


def brier_per_class(y_true_bin: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """Per-class Brier score = mean((p - y)^2)."""
    return np.mean((y_score - y_true_bin) ** 2, axis=0)


def ece_per_class(y_true_bin: np.ndarray, y_score: np.ndarray,
                  n_bins: int = 15) -> np.ndarray:
    """Per-class Expected Calibration Error.

    For each class c we bin predictions ``y_score[:, c]`` into ``n_bins``
    equal-width bins on [0, 1]; per bin we compute
        |mean(predicted_prob_in_bin) − empirical_positive_rate_in_bin|
    and weight by bin population.
    """
    n_samples, n_cls = y_score.shape
    out = np.zeros(n_cls, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for c in range(n_cls):
        p = y_score[:, c]
        y = y_true_bin[:, c]
        bin_idx = np.clip(np.digitize(p, bin_edges) - 1, 0, n_bins - 1)
        ece = 0.0
        for b in range(n_bins):
            sel = bin_idx == b
            if not sel.any():
                continue
            conf = p[sel].mean()
            acc = y[sel].mean()
            weight = sel.mean()
            ece += weight * abs(conf - acc)
        out[c] = ece
    return out


def reliability_curves(y_true_bin: np.ndarray, y_score: np.ndarray,
                       n_bins: int = 10) -> list[dict]:
    """Per-class reliability data: list of dicts with arrays of bin centers,
    predicted-mean, empirical-positive-rate, count."""
    out = []
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    for c in range(y_score.shape[1]):
        p = y_score[:, c]
        y = y_true_bin[:, c]
        bin_idx = np.clip(np.digitize(p, bin_edges) - 1, 0, n_bins - 1)
        pred = np.full(n_bins, np.nan, dtype=np.float64)
        emp = np.full(n_bins, np.nan, dtype=np.float64)
        counts = np.zeros(n_bins, dtype=np.int64)
        for b in range(n_bins):
            sel = bin_idx == b
            counts[b] = int(sel.sum())
            if counts[b]:
                pred[b] = p[sel].mean()
                emp[b] = y[sel].mean()
        out.append({
            'class': LABEL_COLS[c],
            'bin_centers': centers,
            'predicted_mean': pred,
            'empirical_rate': emp,
            'counts': counts,
        })
    return out


def plot_reliability(curves: list[dict], out_path: str, title: str = ''):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.ravel()
    for i, c in enumerate(curves):
        ax = axes[i]
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='perfect')
        valid = ~np.isnan(c['predicted_mean'])
        ax.plot(c['predicted_mean'][valid], c['empirical_rate'][valid],
                'o-', label='model')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(f"{c['class']} (n={int(c['counts'].sum())})")
        ax.set_xlabel('mean predicted probability')
        ax.set_ylabel('empirical positive rate')
        # Bar chart of bin populations beneath each curve
        ax2 = ax.twinx()
        ax2.bar(c['bin_centers'], c['counts'], width=0.08, alpha=0.2,
                color='C1', label='count')
        ax2.set_ylabel('# val pixels')
        ax.legend(loc='upper left', fontsize=8)
    # Hide unused subplot if any
    for j in range(len(curves), len(axes)):
        axes[j].axis('off')
    fig.suptitle(title or 'Per-class reliability diagrams')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--apply_relabels', default=None)
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--ece_bins', type=int, default=15)
    ap.add_argument('--reliability_bins', type=int, default=10)
    ap.add_argument('--label_threshold', type=float, default=0.4,
                    help='Soft-label > threshold == positive (same convention as '
                         'evaluation.metrics).')
    ap.add_argument('--debug_rows', type=int, default=None,
                    help='Use only the first N val rows (smoke test).')
    # contrastive-encoder hooks (delegated to eval_polygon_accuracy.load_model_auto)
    ap.add_argument('--probe_head_path', default=None)
    ap.add_argument('--probe_epochs', type=int, default=5)
    ap.add_argument('--probe_lr', type=float, default=1e-3)
    ap.add_argument('--probe_debug_rows', type=int, default=None)
    ap.add_argument('--no_probe_train', action='store_true')
    ap.add_argument('--extra_plag_dir',
                    default='/mnt/mrdr/crism_classification/data/contrastive/extra_plag_roi')
    ap.add_argument('--mrrsu_aux_npy', default=None)
    args = ap.parse_args()

    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.config)
    cfg = load_config(cfg_path)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'device: {device}')

    ckpt_stem = Path(args.ckpt).stem
    out_dir = args.out_dir or os.path.join(
        cfg.get('reports_dir',
                os.path.join(os.path.dirname(__file__), '..', 'reports')),
        f'calibration_{ckpt_stem}',
    )
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f'output dir: {out_dir}')

    print('loading model…')
    model, kind = load_model_auto(args, cfg, device)
    print(f'  detected kind: {kind}')

    print('building val loader…')
    loader, val_df = _make_val_loader(cfg, args.batch_size, args.apply_relabels,
                                       debug_rows=args.debug_rows)
    print(f'  val rows: {len(val_df):,}')

    print('scoring…')
    y_score, y_soft = score_val(model, loader, device)
    y_true_bin = (y_soft > args.label_threshold).astype(np.float32)

    brier = brier_per_class(y_true_bin, y_score)
    ece = ece_per_class(y_true_bin, y_score, n_bins=args.ece_bins)
    curves = reliability_curves(y_true_bin, y_score, n_bins=args.reliability_bins)

    df = pd.DataFrame({
        'class': LABEL_COLS,
        'brier': brier,
        'ece': ece,
        'n_positives': y_true_bin.sum(axis=0).astype(int),
        'pos_rate': y_true_bin.mean(axis=0),
        'mean_pred_prob': y_score.mean(axis=0),
    })
    df.to_csv(os.path.join(out_dir, 'calibration.csv'), index=False)

    # Per-class reliability arrays as parquet for downstream re-plotting
    rel_rows = []
    for c in curves:
        for b in range(len(c['bin_centers'])):
            rel_rows.append({
                'class': c['class'],
                'bin_center': float(c['bin_centers'][b]),
                'predicted_mean': float(c['predicted_mean'][b])
                    if not np.isnan(c['predicted_mean'][b]) else None,
                'empirical_rate': float(c['empirical_rate'][b])
                    if not np.isnan(c['empirical_rate'][b]) else None,
                'count': int(c['counts'][b]),
            })
    pd.DataFrame(rel_rows).to_parquet(
        os.path.join(out_dir, 'reliability.parquet'), index=False)
    plot_reliability(curves, os.path.join(out_dir, 'reliability.png'),
                     title=f'Reliability — {ckpt_stem}'
                           + (' (corrected val)' if args.apply_relabels else ' (raw val)'))

    # Markdown summary
    lines = []
    lines.append(f'# Calibration report')
    lines.append('')
    lines.append(f'- **Checkpoint:** `{args.ckpt}`')
    lines.append(f'- **Detected kind:** `{kind}`')
    lines.append(f'- **Val rows:** {len(val_df):,}')
    lines.append(f'- **Label threshold:** soft > {args.label_threshold}')
    lines.append(f'- **ECE bins:** {args.ece_bins}')
    lines.append(f'- **Reliability bins:** {args.reliability_bins}')
    if args.apply_relabels:
        lines.append(f'- **Relabels:** `{args.apply_relabels}`')
    lines.append('')
    lines.append('## Per-class calibration')
    lines.append('')
    # Re-use the polygon-eval markdown helper so we don't double-define it.
    from eval_polygon_accuracy import _df_to_markdown
    lines.append(_df_to_markdown(df, floatfmt='.4f'))
    lines.append('')
    lines.append(f'**Mean Brier:** {brier.mean():.4f}')
    lines.append('')
    lines.append(f'**Mean ECE:** {ece.mean():.4f}')
    lines.append('')
    with open(os.path.join(out_dir, 'summary.md'), 'w') as f:
        f.write('\n'.join(lines))

    payload = {
        'ckpt': args.ckpt,
        'kind': kind,
        'n_val': int(len(val_df)),
        'label_threshold': args.label_threshold,
        'per_class': df.to_dict(orient='records'),
        'mean_brier': float(brier.mean()),
        'mean_ece': float(ece.mean()),
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(payload, f, indent=2)

    print(f'mean Brier = {brier.mean():.4f}   mean ECE = {ece.mean():.4f}')
    print(f'wrote {out_dir}/summary.md')


if __name__ == '__main__':
    main()

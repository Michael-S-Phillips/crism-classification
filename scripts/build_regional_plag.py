"""Fit the regional plagioclase-likelihood scorer (full-spectra, region-mean).

Plagioclase is near its per-pixel detection floor in MRDR (~0.14 AP): per-pixel
spectra are too noisy to isolate plag's subtle signature from the other mafic
minerals. But REGIONAL aggregation (region-mean of the full 59-band spectrum)
denoises it into a strongly separable signal — tile-disjoint plag-vs-rest reaches
AUC ~0.92 / AP ~0.71 at the polygon level.

This fits a logistic regression on polygon-mean 59-band mrral reflectance to
predict plagioclase-vs-rest, with a TILE-DISJOINT split for honest validation.
The saved scorer is applied to tile superpixels by scripts/apply_regional_plag.py.

(NOTE: the 2 mrrsu params RPEAK1+BD1300 alone are NOT used — they give only a
pairwise plag-vs-olivine signal and are ~random for plag-vs-rest. The full
spectrum is required.)

Usage:
  conda run -n crism python scripts/build_regional_plag.py
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             roc_curve, precision_recall_curve)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import load_config

N_BANDS = 59
BAND_COLS = [f'm{i}' for i in range(N_BANDS)]


def load_polygon_means(parquet):
    """Group labeled pixels into polygons; return (X_polygon_mean, is_plag, tile)."""
    df = pd.read_parquet(parquet, columns=BAND_COLS + ['plagioclase', 'tile_id', 'polygon_id'])
    X = df[BAND_COLS].to_numpy(np.float32)
    ok = np.isfinite(X).all(axis=1) & (np.abs(X) < 1e4).all(axis=1)
    df = df[ok].reset_index(drop=True)
    g = df.groupby(['tile_id', 'polygon_id'])
    agg = g[BAND_COLS].mean().reset_index()
    is_plag = (g['plagioclase'].max() > 0).astype(int).values
    tiles = g['tile_id'].first().values
    return agg[BAND_COLS].to_numpy(np.float32), is_plag, tiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--out_scorer', default='data/regional_plag_scorer.json')
    ap.add_argument('--out_fig', default='reports/regional_plag_validation.png')
    ap.add_argument('--test_size', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.config))
    parquet = os.path.join(cfg['output_dir'], 'mrral_pixels.parquet')

    X, y, groups = load_polygon_means(parquet)
    print(f'polygons: {len(X):,}  (plag {int(y.sum()):,}, {y.mean():.1%}); '
          f'tiles {len(np.unique(groups))}')

    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    tr, te = next(gss.split(X, y, groups))
    print(f'train {len(tr):,} ({len(np.unique(groups[tr]))} tiles) | '
          f'val {len(te):,} ({len(np.unique(groups[te]))} tiles), val plag {y[te].mean():.1%}')

    scaler = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=5000, class_weight='balanced', C=1.0)
    clf.fit(scaler.transform(X[tr]), y[tr])
    scores = clf.decision_function(scaler.transform(X[te]))
    auc = float(roc_auc_score(y[te], scores))
    apv = float(average_precision_score(y[te], scores))
    print(f'tile-disjoint validation: plag-vs-rest AUC={auc:.4f}  AP={apv:.4f}')

    probs = clf.predict_proba(scaler.transform(X[te]))[:, 1]
    prec, rec, thr = precision_recall_curve(y[te], probs)
    target_p = 0.7
    cand = np.where(prec[:-1] >= target_p)[0]
    if len(cand):
        idx = cand[np.argmax(rec[cand])]          # highest recall at >=70% precision
        thr_sel = float(thr[idx]); p_sel = float(prec[idx]); r_sel = float(rec[idx])
    else:
        idx = int(np.argmax(prec[:-1])); thr_sel = float(thr[idx])
        p_sel = float(prec[idx]); r_sel = float(rec[idx])
    print(f'threshold {thr_sel:.4f}: precision {p_sel:.2%}, recall {r_sel:.2%}')

    scorer = {
        'model': 'logistic_regression',
        'features': 'region_mean_mrral_59band',
        'band_cols': BAND_COLS,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_std': scaler.scale_.tolist(),
        'coef': clf.coef_[0].tolist(),
        'intercept': float(clf.intercept_[0]),
        'val_auc': auc, 'val_ap': apv,
        'prob_threshold': thr_sel,
        'threshold_precision': p_sel, 'threshold_recall': r_sel,
        'note': 'plag-vs-rest on region-mean 59-band mrral spectra; tile-disjoint val',
    }
    os.makedirs(os.path.dirname(args.out_scorer), exist_ok=True)
    with open(args.out_scorer, 'w') as f:
        json.dump(scorer, f, indent=2)
    print(f'wrote {args.out_scorer}')

    fpr, tpr, _ = roc_curve(y[te], scores)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, lw=2); axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.4)
    axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
    axes[0].set_title(f'ROC — regional plag-vs-rest (AUC={auc:.3f})')
    axes[1].plot(rec, prec, lw=2)
    axes[1].axhline(y[te].mean(), color='k', ls='--', alpha=0.4,
                    label=f'base rate {y[te].mean():.1%}')
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
    axes[1].set_title(f'PR — regional plag (AP={apv:.3f})'); axes[1].legend()
    fig.suptitle('Regional plagioclase scorer — polygon-mean 59-band spectra, tile-disjoint val')
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out_fig), exist_ok=True)
    fig.savefig(args.out_fig, dpi=150, bbox_inches='tight')
    print(f'wrote {args.out_fig}')


if __name__ == '__main__':
    main()

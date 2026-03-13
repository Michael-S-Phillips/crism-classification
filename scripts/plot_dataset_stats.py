"""
Figure 4: Dataset statistics — class prevalence and confidence tier breakdown.

Output: reports/fig_dataset_stats.png
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Insert project root so package imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fig_style import FIGSIZE_WIDE, DPI, MINERAL_COLORS, apply_style, despine
from config_loader import load_config
from data.dataset import _collapse_labels, LABEL_COLS

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJ, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

TIER_COLORS = {'High': '#43a047', 'Moderate': '#ffa726', 'Low': '#ef5350'}


def main():
    apply_style()
    cfg = load_config()
    parquet_path = os.path.join(cfg['output_dir'], 'pixels.parquet')
    df = pd.read_parquet(parquet_path)
    df = _collapse_labels(df)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # --- Left panel: class prevalence in train split ---
    train_df  = df[df['split'] == 'train']
    n_train   = len(train_df)
    percentages = []
    counts      = []
    for cls in LABEL_COLS:
        n_pos = int((train_df[cls] > 0.4).sum())
        percentages.append(100.0 * n_pos / n_train if n_train > 0 else 0.0)
        counts.append(n_pos)

    y_left  = range(len(LABEL_COLS))
    colors  = [MINERAL_COLORS[cls] for cls in LABEL_COLS]
    ax_left.barh(list(y_left), percentages, color=colors, edgecolor='white', height=0.6)
    for i, (pct, cnt) in enumerate(zip(percentages, counts)):
        ax_left.text(pct + 0.5, i, f'n={cnt:,}', va='center', fontsize=9)
    ax_left.set_yticks(list(y_left))
    ax_left.set_yticklabels(LABEL_COLS)
    ax_left.set_xlabel('% of train pixels')
    max_pct = max(percentages) if percentages else 0
    ax_left.set_xlim(0, max_pct * 1.3 if max_pct > 0 else 100)
    ax_left.set_title('Class Prevalence (train split)')
    despine(ax_left)

    # --- Right panel: confidence tier breakdown per split ---
    splits     = ['train', 'val', 'test']
    tier_order = ['High', 'Moderate', 'Low']
    y_right    = range(len(splits))
    lefts      = np.zeros(len(splits))

    for tier in tier_order:
        vals = np.array(
            [(df[df['split'] == s]['confidence_tier'] == tier).sum() for s in splits],
            dtype=float,
        )
        ax_right.barh(list(y_right), vals, left=lefts,
                      color=TIER_COLORS[tier], label=tier, edgecolor='white', height=0.6)
        lefts += vals

    ax_right.set_yticks(list(y_right))
    ax_right.set_yticklabels(splits)
    ax_right.set_xlabel('Pixel count')
    ax_right.set_title('Confidence Tier by Split')
    ax_right.legend(loc='upper right', fontsize=9)
    despine(ax_right)

    plt.tight_layout()
    out = os.path.join(REPORTS_DIR, 'fig_dataset_stats.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()

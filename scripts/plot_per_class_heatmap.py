"""
Figure 2: Per-class AP heatmap for v6 models.

Output: reports/fig_per_class_heatmap.png
"""
import os, sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Insert project root so 'scripts.fig_style' is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fig_style import LABEL_COLS, DPI, apply_style, despine

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJ, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# Hardcoded per-class AP values (from v6 sweep logs), sorted by mAP descending.
# Columns: name, olivine, lcp, hcp, plagioclase, other, mAP
MODELS = [
    ('shybrid_asl_diffr_v6', 0.97, 0.85, 0.51, 0.26, 0.48, 0.614),
    ('shybrid_asl_v6',       0.97, 0.86, 0.51, 0.22, 0.48, 0.608),
    ('svit_asl_v6',          0.90, 0.90, 0.55, 0.22, 0.48, 0.534),
    ('svit_asl_diffr_v6',    0.97, 0.65, 0.38, 0.29, 0.32, 0.523),
    ('scnn_asl_v6',          0.90, 0.10, 0.22, 0.20, 0.24, 0.333),
]


def main():
    apply_style()

    model_names = [m[0] for m in MODELS]
    # AP matrix: rows=models, cols=classes (olivine, lcp, hcp, plagioclase, other)
    ap_matrix = np.array([[m[1], m[2], m[3], m[4], m[5]] for m in MODELS])
    map_vals   = [m[6] for m in MODELS]

    fig = plt.figure(figsize=(9, 4))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[5, 0.6], figure=fig, wspace=0.08)
    ax   = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1])

    # --- Heatmap ---
    im = ax.imshow(ap_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')

    # Cell text
    n_models, n_classes = ap_matrix.shape
    for i in range(n_models):
        for j in range(n_classes):
            val = ap_matrix[i, j]
            color = 'white' if val >= 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=9, color=color)

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(LABEL_COLS, fontsize=9)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_names, fontsize=9)
    ax.set_title('Per-Class Average Precision (v6 models, ASL loss)')
    # No colorbar — the mAP sidebar and cell text provide sufficient value indication.

    # --- Right mAP column ---
    # Derive y-limits from matrix shape to stay independent of axes state.
    # imshow places row i at y=i with extent [i-0.5, i+0.5], so limits are [-0.5, n-0.5]
    # inverted (top row = lowest y in display) → [n-0.5, -0.5].
    ax_r.set_ylim(n_models - 0.5, -0.5)
    ax_r.set_xlim(0, 1)
    ax_r.axis('off')
    ax_r.set_title('mAP', fontsize=10)
    for i, mval in enumerate(map_vals):
        ax_r.text(0.1, i, f'{mval:.3f}', va='center', fontsize=10)

    plt.tight_layout()
    out = os.path.join(REPORTS_DIR, 'fig_per_class_heatmap.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()

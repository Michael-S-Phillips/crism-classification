"""
Figure 1: Model Progression horizontal bar chart.

Output: reports/fig_model_progression.png
"""
import os, sys
import matplotlib.pyplot as plt
import numpy as np

# Insert project root so 'scripts.fig_style' is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fig_style import DPI, apply_style, despine

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJ, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# Hardcoded sweep results (stable; not loaded from files)
# Format: (run_name, val_mAP, group)
RUNS = [
    ('scnn_base',            0.554, 'v3'),
    ('scnn_focal',           0.615, 'v3'),
    ('svit_base',            0.522, 'v3'),
    ('svit_mae',             0.507, 'v3'),
    ('svit_base_v5',         0.561, 'v5'),
    ('svit_mae_v5',          0.520, 'v5'),
    ('scnn_asl_v6',          0.333, 'v6'),
    ('svit_asl_v6',          0.534, 'v6'),
    ('svit_asl_diffr_v6',    0.523, 'v6'),
    ('shybrid_asl_v6',       0.608, 'v6'),
    ('shybrid_asl_diffr_v6', 0.614, 'v6'),
]

GROUP_COLORS = {'v3': '#e3f2fd', 'v5': '#e8f5e9', 'v6': '#fff3e0'}
GROUP_LABELS = {'v3': 'v3\n(6-class)', 'v5': 'v5\n(5-class)', 'v6': 'v6\n(5-class)'}

# Annotation text placed just above the boundary after each group
GROUP_TRANSITION_TEXT = {
    'v3': '+mrral spectral input (ViT)',
    'v5': '+ASL loss',
}


def main():
    apply_style()

    names  = [r[0] for r in RUNS]
    maps   = [r[1] for r in RUNS]
    groups = [r[2] for r in RUNS]
    n = len(RUNS)

    # y_pos: 0=bottom, n-1=top. We want table-order top-to-bottom,
    # so table row 0 (scnn_base) = y = n-1 (top), table row n-1 = y = 0 (bottom).
    y_pos = [n - 1 - i for i in range(n)]

    fig, ax = plt.subplots(figsize=(8, 6))

    # --- Group shading (axhspan) ---
    for g in ['v3', 'v5', 'v6']:
        indices = [i for i, r in enumerate(RUNS) if r[2] == g]
        ymin = min(y_pos[i] for i in indices) - 0.4
        ymax = max(y_pos[i] for i in indices) + 0.4
        ax.axhspan(ymin, ymax, color=GROUP_COLORS[g], alpha=0.5, zorder=0)

    # --- Horizontal bars ---
    ax.barh(y_pos, maps, color='#1976d2', edgecolor='white', height=0.6, zorder=2)

    # --- Y-axis tick labels (run names) ---
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)

    # --- Group labels on far left via ax.text ---
    for g in ['v3', 'v5', 'v6']:
        indices = [i for i, r in enumerate(RUNS) if r[2] == g]
        mid = (min(y_pos[i] for i in indices) + max(y_pos[i] for i in indices)) / 2
        ax.text(-0.03, mid, GROUP_LABELS[g], fontsize=8, color='#444444',
                va='center', ha='right', transform=ax.get_yaxis_transform())

    # --- Between-group boundaries and intervention labels ---
    for i in range(len(RUNS) - 1):
        if RUNS[i][2] != RUNS[i + 1][2]:
            boundary_y = (y_pos[i] + y_pos[i + 1]) / 2
            ax.axhline(y=boundary_y, linestyle='--', color='#aaaaaa', linewidth=0.8, zorder=1)
            from_group = RUNS[i][2]
            if from_group in GROUP_TRANSITION_TEXT:
                ax.text(0.02, boundary_y + 0.15, GROUP_TRANSITION_TEXT[from_group],
                        fontsize=8, color='#555555', va='bottom')

    # --- Axes formatting ---
    ax.set_xlim(0, 0.72)
    ax.set_xlabel('val mAP')
    ax.set_title('Model Progression by Sweep Family')
    despine(ax)

    # --- Footnote ---
    fig.text(
        0.01, 0.01,
        '† v3 mAP computed over 6 classes (olivine_t1/t2 separate); '
        'v5/v6 over 5 classes (olivine collapsed). Values not directly comparable.',
        fontsize=7, color='#666666', va='bottom',
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(REPORTS_DIR, 'fig_model_progression.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()

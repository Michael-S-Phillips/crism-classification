"""
Figure 5: Ablation waterfall — cumulative val mAP by training intervention.

Output: reports/fig_ablation_waterfall.png
"""
import os, sys
import matplotlib.pyplot as plt

# Insert project root so package imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.fig_style import FIGSIZE_SINGLE, DPI, apply_style, despine

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJ, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# Hardcoded intervention sequence: (step_index, x_label, val_mAP)
STEPS = [
    (0, 'Baseline\n(scnn, BCE)',      0.554),
    (1, '+ Focal loss',               0.615),
    (2, '+ mrral\nViT',               0.561),
    (3, '+ MAE\npretrain',            0.562),
    (4, '+ ASL loss',                 0.534),
    (5, '+ Hybrid\n(mrrsu feats)',    0.608),
    (6, '+ Hybrid\n+ diff LR',        0.614),
]

COLOR_BASELINE = '#9e9e9e'
COLOR_GAIN     = '#4caf50'
COLOR_REGRESS  = '#ef5350'


def main():
    apply_style()

    step_indices = [s[0] for s in STEPS]
    labels       = [s[1] for s in STEPS]
    maps         = [s[2] for s in STEPS]

    colors = [COLOR_BASELINE]
    for i in range(1, len(maps)):
        colors.append(COLOR_GAIN if maps[i] >= maps[i - 1] else COLOR_REGRESS)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax.bar(step_indices, maps, color=colors, edgecolor='white', width=0.6)

    # Delta annotations for steps 1–6
    for i in range(1, len(maps)):
        delta = round(maps[i] - maps[i - 1], 2)
        delta_str = f'+{delta:.2f}' if delta >= 0 else f'{delta:.2f}'
        ax.text(step_indices[i], maps[i] + 0.01, delta_str,
                ha='center', va='bottom', fontsize=9)

    ax.set_xticks(step_indices)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel('val mAP')
    ax.set_title('Cumulative val mAP by training intervention')
    despine(ax)

    plt.tight_layout()
    out = os.path.join(REPORTS_DIR, 'fig_ablation_waterfall.png')
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    print(f'Saved {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()

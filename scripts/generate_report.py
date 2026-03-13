"""
Generate a comprehensive sweep report from wandb run metrics.

Pulls all runs from the 'crism-mineral-classification' wandb project,
produces visualizations, and writes a markdown report to reports/.

Usage:
    conda run -n crism python scripts/generate_report.py
    conda run -n crism python scripts/generate_report.py --output reports/sweep_report.md
"""
import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import wandb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(PROJ, 'reports')

CLASSES = ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']

# Baseline runs (original single train per model family)
BASELINE_RUNS = {'mlp', 'cnn', 'vit', 'logreg', 'svc', 'rf', 'xgb', 'lgbm'}

# Sweep runs
SWEEP_RUNS = {
    'mlp_sw1', 'mlp_sw2', 'mlp_sw3', 'mlp_sw4',
    'cnn_sw1', 'cnn_sw2', 'cnn_sw3', 'cnn_sw4',
    'vit_sw1', 'vit_sw2', 'vit_sw3',
}


def fetch_runs(project: str = 'crism-mineral-classification') -> pd.DataFrame:
    api = wandb.Api()
    runs = api.runs(f'space-imagery-center/{project}')
    records = []
    for run in runs:
        s = run.summary._json_dict
        cfg = dict(run.config)
        record = {
            'run_name': run.name,
            'run_id': run.id,
            'state': run.state,
            'val_mAP': s.get('val_mAP', np.nan),
        }
        for cls in CLASSES:
            record[f'val_AP_{cls}'] = s.get(f'val_AP_{cls}', np.nan)
        record.update({
            'model': cfg.get('model', run.name.split('_')[0]),
            'lr': cfg.get('lr', np.nan),
            'batch_size': cfg.get('batch_size', np.nan),
            'dropout': cfg.get('dropout', np.nan),
            'use_pos_weight': cfg.get('use_pos_weight', False),
            'weight_decay': cfg.get('weight_decay', np.nan),
            'stopped_epoch': s.get('stopped_epoch', np.nan),
        })
        records.append(record)

    df = pd.DataFrame(records)
    df = df.dropna(subset=['val_mAP'])
    df = df.sort_values('val_mAP', ascending=False).reset_index(drop=True)
    return df


def fetch_history(run_id: str, project: str = 'crism-mineral-classification') -> pd.DataFrame:
    api = wandb.Api()
    run = api.run(f'space-imagery-center/{project}/{run_id}')
    hist = run.history(keys=['epoch', 'train_loss', 'val_mAP'], x_axis='epoch')
    return pd.DataFrame(hist)


def plot_model_comparison(df: pd.DataFrame, out_path: str):
    """Horizontal bar chart: all runs ranked by val_mAP."""
    df_plot = df.sort_values('val_mAP').tail(30)  # top 30
    fig, ax = plt.subplots(figsize=(10, max(6, len(df_plot) * 0.35)))

    colors = []
    for name in df_plot['run_name']:
        if name in BASELINE_RUNS:
            colors.append('#4e79a7')
        elif 'mlp' in name:
            colors.append('#f28e2b')
        elif 'cnn' in name:
            colors.append('#59a14f')
        elif 'vit' in name:
            colors.append('#e15759')
        else:
            colors.append('#76b7b2')

    bars = ax.barh(df_plot['run_name'], df_plot['val_mAP'], color=colors)
    ax.set_xlabel('val mAP', fontsize=12)
    ax.set_title('Model Comparison — val mAP (all runs)', fontsize=13)
    ax.axvline(0.6, color='gray', linestyle='--', alpha=0.5, label='mAP=0.60')
    ax.axvline(0.65, color='gray', linestyle=':', alpha=0.5, label='mAP=0.65')

    for bar, val in zip(bars, df_plot['val_mAP']):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4e79a7', label='Baseline'),
        Patch(facecolor='#f28e2b', label='MLP sweep'),
        Patch(facecolor='#59a14f', label='CNN sweep'),
        Patch(facecolor='#e15759', label='ViT sweep'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    ax.set_xlim(0, min(1.0, df_plot['val_mAP'].max() + 0.05))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Saved: {out_path}')


def plot_per_class_heatmap(df: pd.DataFrame, out_path: str):
    """Heatmap: runs × classes, color = AP."""
    ap_cols = [f'val_AP_{c}' for c in CLASSES]
    df_heat = df[['run_name'] + ap_cols].copy().set_index('run_name')
    df_heat.columns = CLASSES

    # Keep top 20 runs by mAP
    top_names = df.sort_values('val_mAP', ascending=False).head(20)['run_name'].tolist()
    df_heat = df_heat.loc[[n for n in top_names if n in df_heat.index]]

    fig, ax = plt.subplots(figsize=(10, max(6, len(df_heat) * 0.4)))
    data = df_heat.values.astype(float)
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0.2, vmax=0.85)

    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=30, ha='right', fontsize=10)
    ax.set_yticks(range(len(df_heat)))
    ax.set_yticklabels(df_heat.index, fontsize=9)
    ax.set_title('Per-class AP — top 20 runs', fontsize=13)

    for i in range(len(df_heat)):
        for j in range(len(CLASSES)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color='black' if 0.3 < val < 0.75 else 'white')

    plt.colorbar(im, ax=ax, label='AP')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Saved: {out_path}')


def plot_learning_curves(df: pd.DataFrame, out_path: str):
    """Learning curves (val_mAP vs epoch) for all neural sweep runs."""
    neural_runs = df[df['run_name'].str.match(r'(mlp|cnn|vit)_sw\d+')].copy()
    if neural_runs.empty:
        print('  No neural sweep runs found — skipping learning curves')
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    families = [('mlp', axes[0], '#f28e2b'), ('cnn', axes[1], '#59a14f'), ('vit', axes[2], '#e15759')]

    api = wandb.Api()
    for family, ax, color in families:
        family_runs = neural_runs[neural_runs['run_name'].str.startswith(family)]
        ax.set_title(f'{family.upper()} sweep', fontsize=12)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('val mAP')
        ax.set_ylim(0.3, 0.85)
        ax.grid(alpha=0.3)

        for _, row in family_runs.iterrows():
            try:
                run = api.run(f'space-imagery-center/crism-mineral-classification/{row["run_id"]}')
                hist = run.history(keys=['val_mAP'])
                hist_df = pd.DataFrame(hist)
                if 'val_mAP' in hist_df.columns and len(hist_df) > 1:
                    hist_df = hist_df.dropna(subset=['val_mAP'])
                    ax.plot(hist_df.index, hist_df['val_mAP'],
                            label=row['run_name'], alpha=0.8)
            except Exception as e:
                print(f'    Warning: could not fetch history for {row["run_name"]}: {e}')

        ax.legend(fontsize=8)

    plt.suptitle('Learning Curves — Neural Sweep Runs', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Saved: {out_path}')


def plot_baseline_vs_best(df: pd.DataFrame, out_path: str):
    """Side-by-side: baseline vs best sweep run per family."""
    families = ['mlp', 'cnn', 'vit']
    baselines = {}
    bests = {}

    for fam in families:
        base_row = df[df['run_name'] == fam]
        if not base_row.empty:
            baselines[fam] = base_row.iloc[0]['val_mAP']
        sweep_rows = df[df['run_name'].str.match(f'{fam}_sw\\d+')]
        if not sweep_rows.empty:
            bests[fam] = sweep_rows['val_mAP'].max()

    ap_cols = [f'val_AP_{c}' for c in CLASSES]
    fig, axes = plt.subplots(1, len(families), figsize=(15, 5))

    for ax, fam in zip(axes, families):
        base_row = df[df['run_name'] == fam]
        sweep_rows = df[df['run_name'].str.match(f'{fam}_sw\\d+')]

        if base_row.empty or sweep_rows.empty:
            ax.set_title(f'{fam.upper()} — no data')
            continue

        best_sweep = sweep_rows.loc[sweep_rows['val_mAP'].idxmax()]
        base = base_row.iloc[0]

        x = np.arange(len(CLASSES))
        w = 0.35
        base_vals = [base.get(f'val_AP_{c}', 0) for c in CLASSES]
        best_vals = [best_sweep.get(f'val_AP_{c}', 0) for c in CLASSES]

        ax.bar(x - w/2, base_vals, w, label=f'baseline ({base["val_mAP"]:.3f})', alpha=0.8, color='#4e79a7')
        ax.bar(x + w/2, best_vals, w, label=f'{best_sweep["run_name"]} ({best_sweep["val_mAP"]:.3f})',
               alpha=0.8, color='#e15759')
        ax.set_xticks(x)
        ax.set_xticklabels(CLASSES, rotation=30, ha='right', fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('AP')
        ax.set_title(f'{fam.upper()} — Baseline vs Best Sweep')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Saved: {out_path}')


def write_markdown_report(df: pd.DataFrame, plots: dict, out_path: str):
    """Write the full markdown report."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    total_runs = len(df)
    best_run = df.iloc[0]

    baseline_neural = df[df['run_name'].isin({'mlp', 'cnn', 'vit'})]
    sweep_neural = df[df['run_name'].str.match(r'(mlp|cnn|vit)_sw\d+')]

    best_mlp = df[df['run_name'].str.match(r'mlp(_sw\d+)?')].iloc[0] if not df[df['run_name'].str.match(r'mlp(_sw\d+)?')].empty else None
    best_cnn = df[df['run_name'].str.match(r'cnn(_sw\d+)?')].iloc[0] if not df[df['run_name'].str.match(r'cnn(_sw\d+)?')].empty else None
    best_vit = df[df['run_name'].str.match(r'vit(_sw\d+)?')].iloc[0] if not df[df['run_name'].str.match(r'vit(_sw\d+)?')].empty else None

    lines = [
        f'# CRISM Classification Sweep Report',
        f'',
        f'**Generated:** {now}  ',
        f'**Runs evaluated:** {total_runs}  ',
        f'**Best run:** `{best_run["run_name"]}` — val mAP = **{best_run["val_mAP"]:.4f}**',
        f'',
        '---',
        '',
        '## 1. Model Comparison',
        '',
        f'![Model Comparison]({os.path.basename(plots["comparison"])})',
        '',
    ]

    # Summary table
    lines += [
        '### Top 15 Runs',
        '',
        '| Rank | Run | val mAP | Olivine | HCP | LCP | Phyllosilicate | Carbonate | Plagioclase |',
        '|------|-----|---------|---------|-----|-----|----------------|-----------|-------------|',
    ]
    for i, row in df.head(15).iterrows():
        aps = ' | '.join(
            f'{row.get(f"val_AP_{c}", float("nan")):.3f}' for c in CLASSES
        )
        lines.append(
            f'| {i+1} | `{row["run_name"]}` | **{row["val_mAP"]:.4f}** | {aps} |'
        )
    lines.append('')

    # Per-class heatmap
    lines += [
        '## 2. Per-class AP Heatmap',
        '',
        f'![Per-class AP heatmap]({os.path.basename(plots["heatmap"])})',
        '',
    ]

    # Baseline vs best
    lines += [
        '## 3. Baseline vs Best Sweep (per family)',
        '',
        f'![Baseline vs Best]({os.path.basename(plots["comparison_fam"])})',
        '',
    ]

    # Learning curves
    lines += [
        '## 4. Learning Curves',
        '',
        f'![Learning Curves]({os.path.basename(plots["learning"])})',
        '',
    ]

    # Per-family analysis
    lines += ['## 5. Per-family Summary', '']
    for fam, best in [('MLP', best_mlp), ('CNN', best_cnn), ('ViT', best_vit)]:
        if best is None:
            continue
        base_row = df[df['run_name'] == fam.lower()]
        baseline_map = base_row.iloc[0]['val_mAP'] if not base_row.empty else float('nan')
        delta = best['val_mAP'] - baseline_map if not np.isnan(baseline_map) else float('nan')
        delta_str = f'+{delta:.4f}' if delta >= 0 else f'{delta:.4f}'
        lines += [
            f'### {fam}',
            f'- Baseline mAP: `{baseline_map:.4f}`  ',
            f'- Best sweep run: `{best["run_name"]}` — mAP: `{best["val_mAP"]:.4f}` ({delta_str})',
            f'- Config: lr={best.get("lr")}, dropout={best.get("dropout")}, '
            f'use_pos_weight={best.get("use_pos_weight")}, batch_size={best.get("batch_size")}',
            '',
        ]

    # Chronic weak classes
    lines += ['## 6. Class-level Analysis', '']
    for cls in CLASSES:
        col = f'val_AP_{cls}'
        best_ap = df[col].max()
        worst_ap = df[col].min()
        mean_ap = df[col].mean()
        lines.append(f'- **{cls}**: best={best_ap:.3f}, mean={mean_ap:.3f}, worst={worst_ap:.3f}')
    lines.append('')

    # Next steps
    best_overall_ap = {cls: df[f'val_AP_{cls}'].max() for cls in CLASSES}
    weak_classes = [c for c, ap in best_overall_ap.items() if ap < 0.55]

    lines += [
        '## 7. Next Steps',
        '',
        f'**Weak classes** (best AP < 0.55 across all runs): {", ".join(weak_classes) if weak_classes else "none"}',
        '',
        '### Recommended actions:',
        '',
    ]

    if weak_classes:
        lines += [
            f'1. **Focal loss / asymmetric loss** for {", ".join(weak_classes)} — '
            f'pos_weight helps but focal loss better handles hard negatives',
            '2. **Label smoothing** — soft targets may reduce overconfident false negatives',
        ]
    lines += [
        '3. **Deeper ViT / CNN search** — expand best config ± 1 hyperparameter at a time',
        '4. **Data augmentation** — spectral noise injection, random band dropout for CNN/ViT',
        '5. **Ensemble** — average top-3 sweep models; expected +2–4 mAP points',
        '6. **Test-set evaluation** — run best model on held-out test split for final numbers',
        '',
        '---',
        f'*Report generated by `scripts/generate_report.py`*',
    ]

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=None,
                        help='Output markdown path (default: reports/sweep_report_YYYYMMDD.md)')
    parser.add_argument('--project', default='crism-mineral-classification')
    args = parser.parse_args()

    os.makedirs(REPORTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_md = args.output or os.path.join(REPORTS_DIR, f'sweep_report_{stamp}.md')
    img_dir = os.path.dirname(out_md)

    print('Fetching wandb runs...')
    df = fetch_runs(args.project)
    print(f'  Found {len(df)} runs with valid val_mAP')
    print(df[['run_name', 'val_mAP']].to_string(index=False))

    print('\nGenerating plots...')
    plots = {
        'comparison': os.path.join(img_dir, f'comparison_{stamp}.png'),
        'heatmap': os.path.join(img_dir, f'heatmap_{stamp}.png'),
        'learning': os.path.join(img_dir, f'learning_curves_{stamp}.png'),
        'comparison_fam': os.path.join(img_dir, f'baseline_vs_best_{stamp}.png'),
    }

    plot_model_comparison(df, plots['comparison'])
    plot_per_class_heatmap(df, plots['heatmap'])
    plot_learning_curves(df, plots['learning'])
    plot_baseline_vs_best(df, plots['comparison_fam'])

    print('\nWriting markdown report...')
    write_markdown_report(df, plots, out_md)

    print(f'\nReport complete: {out_md}')


if __name__ == '__main__':
    main()

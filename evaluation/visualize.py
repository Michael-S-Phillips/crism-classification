"""
Visualization utilities for model evaluation.
"""
import os
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import precision_recall_curve
import rasterio

from data.label_parser import CLASSES


CLASS_COLORS = {
    'olivine_t1':  '#2ca02c',
    'olivine_t2':  '#98df8a',
    'lcp':         '#1f77b4',
    'hcp':         '#aec7e8',
    'plagioclase': '#d62728',
    'other':       '#7f7f7f',
}


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    confidence_tiers: List[str],
    output_path: Optional[str] = None,
):
    """
    Plot per-class precision-recall curves, one subplot per class,
    with separate lines for each confidence tier.
    """
    tiers = ['High', 'Moderate', 'Low']
    tier_colors = {'High': '#d62728', 'Moderate': '#ff7f0e', 'Low': '#1f77b4'}
    tier_arr = np.array(confidence_tiers)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for cls_idx, cls_name in enumerate(CLASSES):
        ax = axes[cls_idx]
        for tier in tiers:
            mask = tier_arr == tier
            if mask.sum() == 0 or y_true[mask, cls_idx].sum() == 0:
                continue
            y_t = (y_true[mask, cls_idx] > 0.4).astype(int)
            y_s = y_score[mask, cls_idx]
            precision, recall, _ = precision_recall_curve(y_t, y_s)
            ax.plot(recall, precision, color=tier_colors[tier], label=tier, linewidth=2)

        ax.set_title(cls_name, fontsize=12)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend(fontsize=9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)

    plt.suptitle('Precision-Recall Curves by Class and Confidence Tier', fontsize=14)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig


def plot_prediction_map(
    prob_dir: str,
    output_path: Optional[str] = None,
):
    """
    Create a false-color map from per-class probability GeoTIFFs.
    Colours each pixel by its highest-probability class.
    """
    class_arrays = []
    for cls_name in CLASSES:
        tif_path = os.path.join(prob_dir, f'{cls_name}_prob.tif')
        with rasterio.open(tif_path) as src:
            class_arrays.append(src.read(1))

    probs = np.stack(class_arrays)  # (n_classes, H, W)
    best = probs.argmax(axis=0)

    cmap = mcolors.ListedColormap(list(CLASS_COLORS.values()))
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(best, cmap=cmap, vmin=0, vmax=len(CLASSES) - 1, origin='upper')
    cbar = plt.colorbar(im, ax=ax, ticks=range(len(CLASSES)))
    cbar.set_ticklabels(CLASSES)
    ax.set_title('Mineral Classification Map')
    ax.axis('off')

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    return fig

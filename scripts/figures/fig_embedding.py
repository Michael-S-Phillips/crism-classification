"""
Generate fig_v5_embedding.png — encoder embedding visualization. For a
representative pixel per class, compute the 128-d center-pixel embedding
from the fine-tuned classifier's encoder and show as a heatmap. Includes
a similarity matrix that demonstrates whether the encoder separates classes.

Usage:
    conda run -n crism python scripts/figures/fig_embedding.py
"""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, '/Volumes/Mars_GIS/CRISM/MRDR/crism_classification')
from models.spatial_spectral_transformer import SpatialSpectralClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _utils import (
    CLASS_COLORS, CRISM_LABEL_COLS, build_mrral_map,
    find_representative_pixels, load_mrral_parquet, read_patch_from_tile,
)

OUT_PATH = '/Volumes/Mars_GIS/CRISM/MRDR/crism_classification/reports/v5/fig_v5_embedding.png'

# Use the fine-tuned classifier (encoder + head) — we want post-fine-tune
# embeddings since that's what's actually used at inference time.
# Use the local v4 best (lrscale001 — pre-label-collapse-fix). The v4_fixed
# and v5 checkpoints live on HPC; this local one is the freshest available
# for figure generation.
CLASSIFIER_CKPT = '/Volumes/Mars_GIS/CRISM/MRDR/crism_classification/checkpoints/spvit_lrscale001_v4_best.pt'


def load_classifier_encoder():
    """Return the fine-tuned encoder ready for inference + a function that
    extracts a (B, 128) center-pixel embedding from a (B, 7, 7, 59) input.
    """
    model = SpatialSpectralClassifier(
        n_bands=59, patch_size=7, n_classes=5,
        embed_dim=128, n_heads=4, n_layers=6, dropout=0.1,
    )
    ckpt = torch.load(CLASSIFIER_CKPT, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state', ckpt.get('state_dict', ckpt))
    model.load_state_dict(state)
    model.eval()

    @torch.no_grad()
    def embed(x_bhwc: torch.Tensor) -> torch.Tensor:
        # SpatialSpectralTransformer's forward returns (B, n_tokens+1, embed_dim).
        # We want the center pixel token. Use the encoder's forward to get all
        # tokens, then index the center-pixel token (slot N//2 + 1 because of CLS).
        out = model.encoder(x_bhwc)
        n_tokens = 7 * 7
        center_idx = n_tokens // 2 + 1   # +1 for CLS
        return out[:, center_idx]

    return model, embed


def main():
    df = load_mrral_parquet()
    mrral_map = build_mrral_map()
    pixels = find_representative_pixels(df, n_per_class=3, seed=0)

    model, embed = load_classifier_encoder()
    print(f'classifier ckpt: {os.path.basename(CLASSIFIER_CKPT)}')

    embeddings = {}   # cls -> (n_pixels_for_cls, 128)
    sample_info = {}  # cls -> list of (tid, pr, pc)
    for cls in CRISM_LABEL_COLS:
        embs = []
        for tid, pr, pc in pixels.get(cls, []):
            mrral = mrral_map.get(tid)
            if not (mrral and os.path.exists(mrral)):
                continue
            patch = read_patch_from_tile(mrral, pr, pc, patch_size=7, n_bands=59)
            x = torch.from_numpy(patch).unsqueeze(0)
            e = embed(x)[0].numpy()
            embs.append(e)
        if embs:
            embeddings[cls] = np.stack(embs)
            sample_info[cls] = pixels.get(cls, [])[:len(embs)]
        else:
            print(f'  no embedding for {cls}')

    # Figure layout: one row per class showing the embedding heatmap +
    # a cosine-similarity matrix on the right.
    fig = plt.figure(figsize=(13.5, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.4, 1], height_ratios=[2.1, 1])

    # --- Top-left: stacked embeddings, one row per (class, instance) ---
    ax_emb = fig.add_subplot(gs[0, 0])
    stacked = []
    row_labels = []
    row_colors = []
    for cls in CRISM_LABEL_COLS:
        if cls not in embeddings:
            continue
        for i, e in enumerate(embeddings[cls]):
            stacked.append(e)
            row_labels.append(f'{cls} #{i+1}')
            row_colors.append(CLASS_COLORS[cls])
    stacked = np.array(stacked)
    # Normalize each row to [-1, 1] for visibility
    vmin, vmax = np.percentile(stacked, (2, 98))
    vlim = max(abs(vmin), abs(vmax))
    im = ax_emb.imshow(stacked, aspect='auto', cmap='RdBu_r',
                       vmin=-vlim, vmax=vlim, interpolation='nearest')
    ax_emb.set_yticks(range(len(row_labels)))
    ax_emb.set_yticklabels(row_labels, fontsize=9)
    for tick, color in zip(ax_emb.get_yticklabels(), row_colors):
        tick.set_color(color)
    ax_emb.set_xlabel('Embedding dimension (0–127)')
    ax_emb.set_title('Center-pixel encoder embeddings (128-d), 3 representative pixels per class\n'
                     'classifier: spvit_lrscale001_v4_fixed (post fine-tune)',
                     fontsize=10)
    plt.colorbar(im, ax=ax_emb, label='Embedding value', shrink=0.85)

    # --- Top-right: per-class mean embedding bar plot for one example pixel ---
    ax_bar = fig.add_subplot(gs[0, 1])
    for i, cls in enumerate(CRISM_LABEL_COLS):
        if cls not in embeddings:
            continue
        mean_e = embeddings[cls].mean(axis=0)
        ax_bar.plot(np.arange(128), mean_e + i * 2.5,    # vertical offset per class
                    color=CLASS_COLORS[cls], linewidth=1.0)
        ax_bar.text(-3, i * 2.5, cls, ha='right', va='center',
                    color=CLASS_COLORS[cls], fontsize=10)
    ax_bar.set_xlabel('Embedding dimension')
    ax_bar.set_yticks([])
    ax_bar.set_title('Mean embedding per class (offset vertically)', fontsize=10)
    ax_bar.set_xlim(-30, 128)
    ax_bar.grid(axis='x', alpha=0.3)

    # --- Bottom: cosine similarity between class-mean embeddings ---
    classes_with_emb = [c for c in CRISM_LABEL_COLS if c in embeddings]
    means = np.stack([embeddings[c].mean(axis=0) for c in classes_with_emb])
    means_norm = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-8)
    sim = means_norm @ means_norm.T

    ax_sim = fig.add_subplot(gs[1, 0])
    im2 = ax_sim.imshow(sim, vmin=-1, vmax=1, cmap='RdBu_r')
    ax_sim.set_xticks(range(len(classes_with_emb)))
    ax_sim.set_xticklabels(classes_with_emb, rotation=15)
    ax_sim.set_yticks(range(len(classes_with_emb)))
    ax_sim.set_yticklabels(classes_with_emb)
    for i in range(len(classes_with_emb)):
        for j in range(len(classes_with_emb)):
            ax_sim.text(j, i, f'{sim[i, j]:.2f}', ha='center', va='center',
                        color='white' if abs(sim[i, j]) > 0.6 else 'black',
                        fontsize=9)
    ax_sim.set_title('Cosine similarity between class-mean embeddings\n'
                     '(low off-diagonal = encoder separates classes)', fontsize=10)
    plt.colorbar(im2, ax=ax_sim, label='Cosine similarity', shrink=0.85)

    # --- Bottom-right: explanatory text + sample info ---
    ax_txt = fig.add_subplot(gs[1, 1])
    ax_txt.axis('off')
    info = (
        "How to read this figure:\n\n"
        "• Top heatmap shows the 128-d vector each pixel\n"
        "  produces after the encoder's 6 transformer blocks.\n"
        "  Same-class rows should look similar; different-\n"
        "  class rows should look distinct.\n\n"
        "• Bottom cosine-similarity matrix summarises the\n"
        "  same observation: diagonal = 1.0 (self-similar);\n"
        "  small off-diagonal = good class separation;\n"
        "  large off-diagonal = the encoder hasn't fully\n"
        "  resolved that class pair (e.g., HCP↔LCP risk).\n\n"
        "• These embeddings feed a single linear+sigmoid\n"
        "  head to produce the per-class probabilities."
    )
    ax_txt.text(0.0, 1.0, info, va='top', fontsize=9, family='monospace')

    fig.suptitle('Encoder embedding vectors — what the SpatialSpectralClassifier sees '
                 'just before the classification head',
                 fontsize=12, y=1.02)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {OUT_PATH}')


if __name__ == '__main__':
    main()

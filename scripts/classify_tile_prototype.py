"""
Classify a CRISM mrral tile using prototype-based cosine similarity.

Loads per-class prototype embeddings (built by build_prototypes.py), embeds
all tile pixels via the stored encoder, scores each pixel against the
prototypes via cosine similarity. Output is a (H,W,5) .npz identical in
schema to classify_tile_supervised.py --save_probs output.

Optionally compares two prototype files (e.g. fine-tuned vs MAE encoder)
in a side-by-side figure.

Usage:
    # Single encoder
    conda run -n crism python scripts/classify_tile_prototype.py \\
        --tile /mnt/mrdr/mc26/t0435_mrral_40s323_0327_4.img \\
        --proto_a data/prototypes/proto_finetuned_all.npz \\
        --save_probs /tmp/t0435_proto_finetuned_probs.npz

    # Comparison (proto_a vs proto_b, optional supervised baseline)
    conda run -n crism python scripts/classify_tile_prototype.py \\
        --tile /mnt/mrdr/mc26/t0435_mrral_40s323_0327_4.img \\
        --proto_a data/prototypes/proto_finetuned_all.npz \\
        --proto_b data/prototypes/proto_mae_all.npz \\
        --supervised_probs /tmp/t0435_mrral_40s323_0327_4_probs.npz \\
        --save_probs /tmp/t0435_proto_finetuned_probs.npz \\
        --out reports/fig_prototype_t0435.png
"""
import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.build_prototypes import normalize_patches, load_encoder
from scripts.classify_tile_supervised import load_tile, save_probs
from scripts.fig_style import MINERAL_COLORS

# ── constants ────────────────────────────────────────────────────────────────
N_BANDS     = 59
PATCH_SIZE  = 7
PAD         = PATCH_SIZE // 2
CENTER_IDX  = PATCH_SIZE ** 2 // 2 + 1   # = 25
EMBED_DIM   = 128
NODATA      = 65535.0
CLIP_MAX    = 0.5
N_CLASSES   = 5
CLASS_NAMES = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']
# Use consistent mineral colors from fig_style (same as plot_vector_mineral_maps.py)
CLASS_COLORS = [MINERAL_COLORS[n] for n in CLASS_NAMES]
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── pure helpers (importable for testing) ─────────────────────────────────────

def cosine_similarity_classify(
    embeddings: np.ndarray,
    prototypes: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between query embeddings and prototypes.

    Args:
        embeddings: (N, embed_dim) float32, L2-normalized
        prototypes: (n_classes, embed_dim) float32, L2-normalized

    Returns:
        (N, n_classes) float32 similarities clipped to [0, 1]
    """
    sims = embeddings @ prototypes.T   # (N, n_classes) in [-1, 1]
    return np.clip(sims, 0.0, 1.0).astype(np.float32)


def apply_valid_mask(
    sims_flat: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Reshape flat similarities to (H, W, n_classes) and zero invalid pixels.

    Args:
        sims_flat: (H*W, n_classes) float32
        valid_mask: (H, W) bool

    Returns:
        (H, W, n_classes) float32 — invalid pixels set to 0.0
    """
    H, W = valid_mask.shape
    result = sims_flat.reshape(H, W, -1).copy()
    result[~valid_mask] = 0.0
    return result


def apply_min_similarity(
    sims_hw5: np.ndarray,
    valid_mask: np.ndarray,
    min_similarity: float | None,
) -> np.ndarray:
    """Zero out pixels whose max class similarity falls below min_similarity.

    Args:
        sims_hw5: (H, W, 5) float32, output of apply_valid_mask
        valid_mask: (H, W) bool — used to exclude invalid pixels from percentile
        min_similarity: explicit threshold in [0, 1], or None to auto-compute
            as the 10th percentile of max-similarity across valid pixels.

    Returns:
        (H, W, 5) float32 — low-confidence pixels zeroed on all channels.
    """
    result = sims_hw5.copy()
    max_sims = result.max(axis=-1)                       # (H, W)
    if min_similarity is None:
        valid_max = max_sims[valid_mask]                 # (N_valid,)
        threshold = float(np.percentile(valid_max, 10))
    else:
        threshold = min_similarity
    low_conf = max_sims < threshold                      # (H, W) bool
    result[low_conf] = 0.0
    return result


def load_prototype_npz(path: str) -> Tuple[np.ndarray, List[str], str]:
    """Load prototype .npz produced by build_prototypes.py.

    Returns:
        prototypes: (n_classes, embed_dim) float32
        class_names: list of class name strings
        encoder_ckpt: checkpoint path string (for loading the encoder)
    """
    data = np.load(path, allow_pickle=True)
    prototypes = data['prototypes']
    if prototypes.ndim != 2 or prototypes.shape[1] != 128:
        raise ValueError(
            f"Unexpected prototypes shape {prototypes.shape}; expected (n_classes, 128)"
        )
    class_names = [str(x) for x in data['class_names']]
    encoder_ckpt = str(data['encoder_ckpt'])
    return prototypes, class_names, encoder_ckpt


# ── patch extraction ─────────────────────────────────────────────────────────

def extract_patches_batched(tile: np.ndarray, batch_size: int = 512):
    """Yield (patches, flat_indices) batches over all tile pixels."""
    H, W, C = tile.shape
    padded = np.pad(tile, ((PAD, PAD), (PAD, PAD), (0, 0)), mode='constant')
    n_pixels = H * W
    for start in range(0, n_pixels, batch_size):
        end = min(start + batch_size, n_pixels)
        rows = np.arange(start, end) // W
        cols = np.arange(start, end) % W
        batch = np.stack([
            padded[r:r + PATCH_SIZE, c:c + PATCH_SIZE, :]
            for r, c in zip(rows, cols)
        ])
        yield batch.astype(np.float32), np.arange(start, end)


# ── tile embedding ────────────────────────────────────────────────────────────

def embed_tile(
    tile: np.ndarray,
    encoder: torch.nn.Module,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Embed all tile pixels via encoder; return (H*W, 128) L2-normalized.

    Args:
        tile: (H, W, 59) float32, pre-processed (nodata→0, clipped)
        encoder: SpatialSpectralTransformer in eval mode
        device: torch device
        batch_size: number of pixels per inference batch

    Returns:
        (H*W, embed_dim) float32, L2-normalized
    """
    from tqdm import tqdm
    H, W, _ = tile.shape
    n_pixels = H * W
    n_batches = (n_pixels + batch_size - 1) // batch_size
    embeddings = np.zeros((n_pixels, EMBED_DIM), dtype=np.float32)

    with torch.no_grad():
        for patches, idx in tqdm(extract_patches_batched(tile, batch_size),
                                  total=n_batches, desc='Embedding'):
            patches = normalize_patches(patches)
            x = torch.from_numpy(patches).to(device)
            out = encoder(x)                         # (B, 50, 128)
            emb = out[:, CENTER_IDX]                  # (B, 128)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            embeddings[idx] = emb.cpu().numpy()

    return embeddings


# ── figure ────────────────────────────────────────────────────────────────────

def _argmax_rgb(sims_hw5: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Return (H, W, 3) uint8 RGB argmax class map. Invalid pixels → white."""
    import matplotlib.colors as mc
    H, W = valid_mask.shape
    rgb = np.ones((H, W, 3), dtype=np.float32)  # white = invalid
    dom = np.argmax(sims_hw5, axis=2)            # (H, W)
    for ci, color in enumerate(CLASS_COLORS):
        r, g, b = mc.to_rgb(color)
        mask = valid_mask & (dom == ci)
        rgb[mask] = [r, g, b]
    return (rgb * 255).astype(np.uint8)


def make_comparison_figure(
    panels_argmax: List[Tuple[str, np.ndarray]],   # [(label, (H,W,3) uint8), ...]
    panels_sims: List[Tuple[str, np.ndarray]],     # [(label, (H,W,5) float32), ...]
) -> plt.Figure:
    """Build the 2-section comparison figure.

    Top section: argmax class maps (one per encoder/supervised).
    Bottom section: per-class similarity heatmaps (one row per encoder).

    Args:
        panels_argmax: list of (label, rgb_image) tuples for top section
        panels_sims: list of (label, sims_hw5) tuples for bottom section
    """
    n_top   = len(panels_argmax)
    n_enc   = len(panels_sims)
    n_cols  = max(n_top, N_CLASSES)
    cell_w, cell_h = 3.5, 3.0
    fig_w = cell_w * n_cols
    fig_h = cell_h * (1 + n_enc)

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = gridspec.GridSpec(
        1 + n_enc, n_cols,
        figure=fig,
        hspace=0.45, wspace=0.25,
    )

    # Top row: argmax maps
    for i, (label, rgb) in enumerate(panels_argmax):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(rgb)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.axis('off')
    # Hide unused top-row cells
    for i in range(n_top, n_cols):
        fig.add_subplot(gs[0, i]).axis('off')

    # Bottom rows: per-class similarity heatmaps
    for enc_i, (enc_label, sims_hw5) in enumerate(panels_sims):
        last_ax = None
        for ci, class_name in enumerate(CLASS_NAMES):
            ax = fig.add_subplot(gs[1 + enc_i, ci])
            im = ax.imshow(sims_hw5[:, :, ci], vmin=0, vmax=1, cmap='viridis', aspect='auto')
            if enc_i == 0:
                ax.set_title(class_name, fontsize=9)
            if ci == 0:
                ax.set_ylabel(enc_label, fontsize=9, rotation=90, labelpad=4)
            ax.axis('off')
            last_ax = ax
        plt.colorbar(im, ax=last_ax, fraction=0.03, pad=0.02)

    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Classify a CRISM tile via cosine similarity to prototypes.')
    parser.add_argument('--tile',   required=True, help='mrral .img path')
    parser.add_argument('--proto_a', required=True, help='Primary prototype .npz')
    parser.add_argument('--proto_b', default=None,
                        help='Second prototype .npz for comparison figure')
    parser.add_argument('--save_probs', default=None, metavar='PATH',
                        help='Save (H,W,5) similarity .npz (proto_a only)')
    parser.add_argument('--supervised_probs', default=None,
                        help='Existing supervised .npz to include as 3rd argmax panel')
    parser.add_argument('--out', default=None,
                        help='Figure output path (default: reports/fig_prototype_<tile>.png)')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument(
        '--min_similarity', type=float, default=None,
        help='Minimum max cosine similarity to classify a pixel. '
             'Default: 10th percentile of tile max-similarity distribution. '
             'Pass 0.0 to disable.',
    )
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tile_id = os.path.splitext(os.path.basename(args.tile))[0]

    if args.out is None:
        args.out = os.path.join(PROJ, 'reports', f'fig_prototype_{tile_id}.png')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # Load tile
    print(f'Loading tile: {args.tile}')
    tile, valid_mask, transform, crs = load_tile(args.tile)
    H, W = valid_mask.shape
    print(f'  {H}×{W}, {valid_mask.sum():,} valid pixels')

    transform_arr = np.array([transform.a, transform.b, transform.c,
                               transform.d, transform.e, transform.f], dtype=np.float64)
    crs_wkt = crs.to_wkt()

    def run_proto(proto_path: str) -> Tuple[np.ndarray, np.ndarray, str]:
        """Embed tile + cosine similarity for one prototype file.
        Returns (sims_hw5, embeddings_flat, encoder_tag).
        """
        protos, _, ckpt_path = load_prototype_npz(proto_path)
        tag = os.path.splitext(os.path.basename(proto_path))[0]
        print(f'Loading encoder for {tag}: {ckpt_path}')
        encoder, _ = load_encoder(ckpt_path, device)
        print(f'Embedding tile ({tag})...')
        embs = embed_tile(tile, encoder, device, args.batch_size)
        sims_flat = cosine_similarity_classify(embs, protos)
        sims_hw5  = apply_valid_mask(sims_flat, valid_mask)
        sims_hw5  = apply_min_similarity(sims_hw5, valid_mask, args.min_similarity)
        return sims_hw5, embs, tag

    # Run proto_a (always)
    sims_a, _, tag_a = run_proto(args.proto_a)

    # Save probs from proto_a
    if args.save_probs:
        save_probs(args.save_probs, sims_a, valid_mask, transform_arr, crs_wkt)
        print(f'Saved probs → {args.save_probs}')

    # Build figure panels
    panels_argmax = [(f'Proto-A\n{tag_a}', _argmax_rgb(sims_a, valid_mask))]
    panels_sims   = [(tag_a, sims_a)]

    if args.proto_b:
        sims_b, _, tag_b = run_proto(args.proto_b)
        panels_argmax.append((f'Proto-B\n{tag_b}', _argmax_rgb(sims_b, valid_mask)))
        panels_sims.append((tag_b, sims_b))

    if args.supervised_probs:
        sup = np.load(args.supervised_probs)
        sup_hw5 = sup['probs']
        panels_argmax.append(('Supervised\n(sigmoid)', _argmax_rgb(sup_hw5, valid_mask)))

    fig = make_comparison_figure(panels_argmax, panels_sims)
    fig.suptitle(f'Prototype classifier — {tile_id}', fontsize=12, y=1.01)
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f'Saved figure → {args.out}')


if __name__ == '__main__':
    main()

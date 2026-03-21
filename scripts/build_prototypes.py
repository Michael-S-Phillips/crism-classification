"""
Build per-class prototype embeddings from labeled CRISM pixels.

Loads mrral_pixels.parquet, filters by confidence tier and split,
extracts 7×7 patch embeddings via the SpatialSpectralTransformer encoder,
computes one L2-normalized mean embedding per mineral class.

Usage:
    conda run -n crism python scripts/build_prototypes.py \\
        --ckpt checkpoints/spvit_lrscale0005_best.pt \\
        --out data/prototypes/proto_finetuned_all.npz

    conda run -n crism python scripts/build_prototypes.py \\
        --ckpt checkpoints/spatial_mae_128d_6l_best.pt \\
        --out data/prototypes/proto_mae_all.npz

    # High-confidence only
    conda run -n crism python scripts/build_prototypes.py \\
        --ckpt checkpoints/spvit_lrscale0005_best.pt \\
        --confidence_tiers High \\
        --out data/prototypes/proto_finetuned_high.npz
"""
import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.spatial_spectral_transformer import (
    SpatialSpectralClassifier,
    SpatialSpectralTransformer,
)

# ── constants ────────────────────────────────────────────────────────────────
N_BANDS    = 59
PATCH_SIZE = 7
CENTER_IDX = PATCH_SIZE ** 2 // 2 + 1   # = 25  (CLS is slot 0, spatial token i is slot i+1)
EMBED_DIM  = 128
N_HEADS    = 4
N_LAYERS   = 6
CLASS_NAMES = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── pure helpers (importable for testing) ────────────────────────────────────

def normalize_patches(patches: np.ndarray) -> np.ndarray:
    """Per-patch mean/std normalization.

    Args:
        patches: (B, H, W, C) float32 — finite, no NaN

    Returns:
        (B, H, W, C) float32 — zero mean, unit std per patch
    """
    B = patches.shape[0]
    flat = patches.reshape(B, -1)
    mu    = flat.mean(axis=1, keepdims=True)
    sigma = flat.std(axis=1,  keepdims=True)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    return ((flat - mu) / sigma).reshape(patches.shape)


def filter_parquet(df: pd.DataFrame,
                   splits: List[str],
                   confidence_tiers: List[str]) -> pd.DataFrame:
    """Return rows matching the given splits and confidence tiers.

    Also collapses olivine_t1/t2 → olivine column (max).
    Does NOT reset_index — original row positions are preserved for memmap lookup.

    Args:
        df: raw mrral_pixels parquet DataFrame
        splits: e.g. ['train', 'val']
        confidence_tiers: e.g. ['High', 'Moderate', 'Low']

    Returns:
        Filtered DataFrame with added 'olivine' column. Original index preserved.
    """
    out = df.copy()
    out['olivine'] = out[['olivine_t1', 'olivine_t2']].max(axis=1)
    mask = out['split'].isin(splits) & out['confidence_tier'].isin(confidence_tiers)
    return out[mask]


def compute_prototypes(
    embeddings_per_class: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Compute one L2-normalized prototype per class.

    Args:
        embeddings_per_class: dict mapping class name → (N, embed_dim) float32
            array of L2-normalized per-pixel embeddings.
            N may be 0 — this raises ValueError naming the class.

    Returns:
        prototypes: (n_classes, embed_dim) float32 — each row has unit L2 norm
        n_pixels_per_class: dict mapping class name → int pixel count
    """
    n_classes  = len(CLASS_NAMES)
    embed_dim  = next(v.shape[1] for v in embeddings_per_class.values() if v.shape[0] > 0)
    prototypes = np.zeros((n_classes, embed_dim), dtype=np.float32)
    n_pixels   = {}

    for ci, name in enumerate(CLASS_NAMES):
        embs = embeddings_per_class[name]
        if embs.shape[0] == 0:
            raise ValueError(
                f"No hard-positive pixels for class '{name}'. "
                f"Widen --confidence_tiers or --splits."
            )
        n_pixels[name] = embs.shape[0]
        # Normalize each embedding to unit sphere before averaging (Fréchet mean)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        embs = embs / norms
        mean_emb = embs.mean(axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm < 1e-8:
            raise ValueError(f"Prototype for '{name}' has near-zero norm — check data.")
        prototypes[ci] = mean_emb / norm

    return prototypes, n_pixels


# ── encoder loading ───────────────────────────────────────────────────────────

def load_encoder(ckpt_path: str, device: torch.device) -> Tuple[torch.nn.Module, str]:
    """Load encoder from a fine-tuned or MAE checkpoint.

    Fine-tuned checkpoints have 'model_state' key → load SpatialSpectralClassifier,
    return its .encoder (drops the linear head).

    MAE checkpoints have 'encoder_state' key → load directly into a bare
    SpatialSpectralTransformer via load_encoder_state_dict().

    Returns:
        (encoder module in eval mode, ckpt_tag string for metadata)
    """
    ckpt_tag = os.path.splitext(os.path.basename(ckpt_path))[0]
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(state, dict) and 'model_state' in state:
        # Fine-tuned SpatialSpectralClassifier
        classifier = SpatialSpectralClassifier(
            n_bands=N_BANDS, patch_size=PATCH_SIZE, n_classes=5,
            embed_dim=EMBED_DIM, n_heads=N_HEADS, n_layers=N_LAYERS,
        ).to(device)
        classifier.load_state_dict(state['model_state'])
        encoder = classifier.encoder
    elif isinstance(state, dict) and 'encoder_state' in state:
        # MAE pre-trained encoder
        encoder = SpatialSpectralTransformer(
            n_bands=N_BANDS, patch_size=PATCH_SIZE,
            embed_dim=EMBED_DIM, n_heads=N_HEADS, n_layers=N_LAYERS,
        ).to(device)
        encoder.load_encoder_state_dict(state['encoder_state'])
    else:
        raise ValueError(
            f"Unrecognized checkpoint format. Keys: {list(state.keys())[:8]}\n"
            "Expected 'model_state' (fine-tuned) or 'encoder_state' (MAE)."
        )

    encoder.eval()
    return encoder, ckpt_tag


# ── patch extraction from cache ───────────────────────────────────────────────

def embed_labeled_pixels(
    df: pd.DataFrame,
    splits: List[str],
    confidence_tiers: List[str],
    cache_dir: str,
    encoder: torch.nn.Module,
    device: torch.device,
    batch_size: int = 512,
) -> Dict[str, np.ndarray]:
    """Extract L2-normalized center-token embeddings for all hard-positive pixels.

    IMPORTANT: memmap row i == row i within the FULL (unfiltered) split slice of
    the parquet. Therefore this function slices the full split first (reset_index),
    then applies confidence_tier and class filters to get the correct row indices.

    Args:
        df: full mrral_pixels parquet DataFrame with 'olivine' column already added
            (via filter_parquet or manual max). Must NOT be pre-filtered by tier.
        splits: the splits to process ('train', 'val')
        confidence_tiers: e.g. ['High', 'Moderate', 'Low'] — applied per-split
        cache_dir: path containing mrral_{split}_patches_p7.npy files
        encoder: SpatialSpectralTransformer in eval mode
        device: torch device
        batch_size: inference batch size

    Returns:
        dict mapping class name → (N, 128) float32 L2-normalized embeddings.
        N may be 0 if no hard-positive pixels found for that class/tier combo.
    """
    embeddings_per_class: Dict[str, List[np.ndarray]] = {n: [] for n in CLASS_NAMES}

    for split in splits:
        # Full split slice — reset to 0-based so index == memmap row number
        df_split = df[df['split'] == split].reset_index(drop=True)
        if len(df_split) == 0:
            continue

        cache_file = os.path.join(cache_dir, f'mrral_{split}_patches_p{PATCH_SIZE}.npy')
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Patch cache not found: {cache_file}")

        # Open memmap with full split row count — row i == df_split.iloc[i]
        cache = np.memmap(
            cache_file, dtype='float32', mode='r',
            shape=(len(df_split), PATCH_SIZE, PATCH_SIZE, N_BANDS),
        )

        tier_mask = df_split['confidence_tier'].isin(confidence_tiers)

        for class_name in CLASS_NAMES:
            # Row must match confidence tier AND be hard-positive for this class
            pos_mask = (df_split[class_name] == 1.0) & tier_mask
            indices = df_split.index[pos_mask].to_numpy()  # 0-based == memmap rows
            if len(indices) == 0:
                continue

            class_embs: List[np.ndarray] = []
            for start in tqdm(range(0, len(indices), batch_size),
                               desc=f'{split}/{class_name}', leave=False):
                batch_idx = indices[start:start + batch_size]
                patches = np.array(cache[batch_idx])         # materialize from memmap
                patches = normalize_patches(patches)
                x = torch.from_numpy(patches).to(device)
                with torch.no_grad():
                    out = encoder(x)                         # (B, 50, 128)
                    emb = out[:, CENTER_IDX]                  # (B, 128)
                    emb = torch.nn.functional.normalize(emb, dim=-1)
                class_embs.append(emb.cpu().numpy())

            if class_embs:
                embeddings_per_class[class_name].append(np.concatenate(class_embs, axis=0))

    # Consolidate across splits
    return {
        name: np.concatenate(chunks, axis=0) if chunks else np.zeros((0, EMBED_DIM), dtype=np.float32)
        for name, chunks in embeddings_per_class.items()
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Build per-class prototype embeddings from labeled CRISM pixels.')
    parser.add_argument('--ckpt', required=True,
                        help='Encoder checkpoint path (fine-tuned or MAE)')
    parser.add_argument('--confidence_tiers', nargs='+',
                        default=['High', 'Moderate', 'Low'],
                        choices=['High', 'Moderate', 'Low'],
                        help='Labeled-pixel confidence tiers to include (default: all)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'],
                        choices=['train', 'val'],
                        help='Parquet splits to use (default: train val; never test)')
    parser.add_argument('--out', default=None,
                        help='Output .npz path (default: data/prototypes/proto_<tag>_<tiers>.npz)')
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load encoder
    print(f'Loading encoder from {args.ckpt}')
    encoder, ckpt_tag = load_encoder(args.ckpt, device)
    print(f'  encoder tag: {ckpt_tag}')

    # Output path
    tiers_tag = '_'.join(t[0].lower() for t in sorted(args.confidence_tiers))  # e.g. 'hml'
    if args.out is None:
        out_dir = os.path.join(PROJ, 'data', 'prototypes')
        os.makedirs(out_dir, exist_ok=True)
        args.out = os.path.join(out_dir, f'proto_{ckpt_tag}_{tiers_tag}.npz')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # Load parquet (full — embed_labeled_pixels handles tier filtering internally)
    parquet_path = os.path.join(PROJ, 'data', 'mrral_pixels.parquet')
    print(f'Loading {parquet_path}')
    df = pd.read_parquet(parquet_path)
    # Add olivine column (max of t1/t2) needed for class hard-positive selection
    df['olivine'] = df[['olivine_t1', 'olivine_t2']].max(axis=1)
    # Report filtered count for user info
    df_filtered = filter_parquet(df, splits=args.splits, confidence_tiers=args.confidence_tiers)
    print(f'  {len(df_filtered):,} pixels match filters '
          f'(tiers={args.confidence_tiers}, splits={args.splits})')

    # Embed labeled pixels — pass FULL df so memmap row indices are correct
    cache_dir = os.path.join(PROJ, 'data', 'patch_cache')
    print('Extracting embeddings for labeled pixels...')
    embeddings_per_class = embed_labeled_pixels(
        df, args.splits, args.confidence_tiers, cache_dir, encoder, device, args.batch_size,
    )

    for name, embs in embeddings_per_class.items():
        print(f'  {name:12s}: {len(embs):,} hard-positive pixels')

    # Compute prototypes
    print('Computing prototypes...')
    prototypes, n_pixels_per_class = compute_prototypes(embeddings_per_class)

    # Save
    np.savez_compressed(
        args.out,
        prototypes=prototypes,
        class_names=np.array(CLASS_NAMES),
        encoder_ckpt=np.array(args.ckpt),
        confidence_tiers_used=np.array(args.confidence_tiers),
        splits_used=np.array(args.splits),
        n_pixels_per_class=np.array([n_pixels_per_class[n] for n in CLASS_NAMES]),
    )
    print(f'Saved → {args.out}')
    for i, name in enumerate(CLASS_NAMES):
        print(f'  {name:12s}: norm={np.linalg.norm(prototypes[i]):.6f}, '
              f'n={n_pixels_per_class[name]:,}')


if __name__ == '__main__':
    main()

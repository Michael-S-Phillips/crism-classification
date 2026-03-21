# Prototype Classifier Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a cosine-similarity prototype classifier that replaces the over-predicting linear head with per-class anchor embeddings computed from labeled pixels, supporting comparison between fine-tuned and MAE encoders.

**Architecture:** Two scripts: `build_prototypes.py` extracts labeled pixel embeddings and computes one L2-normalized prototype per class; `classify_tile_prototype.py` embeds all tile pixels and scores them against the prototypes via cosine similarity, outputting the same `(H,W,5)` `.npz` format used by the downstream vectorization pipeline. The comparison figure uses a GridSpec layout with argmax maps on top and per-class similarity heatmaps below.

**Tech Stack:** PyTorch, numpy, pandas, matplotlib, tqdm. Models from `models/spatial_spectral_transformer.py`. Patch cache memmaps in `data/patch_cache/`. Labeled data in `data/mrral_pixels.parquet`.

---

## Chunk 1: build_prototypes.py

## Task 1: Tests for prototype math (pure functions, no GPU needed)

**Files:**
- Create: `tests/test_build_prototypes.py`

These tests cover the math-only functions. They use synthetic data and run fast without any checkpoint or patch cache.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_build_prototypes.py
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_normalize_patches_zero_mean():
    """normalize_patches produces zero mean per patch."""
    from scripts.build_prototypes import normalize_patches
    rng = np.random.default_rng(0)
    patches = rng.random((8, 7, 7, 59)).astype(np.float32)
    out = normalize_patches(patches)
    B = patches.shape[0]
    flat = out.reshape(B, -1)
    np.testing.assert_allclose(flat.mean(axis=1), 0.0, atol=1e-5)


def test_normalize_patches_unit_std():
    """normalize_patches produces unit std per patch (when input has variance)."""
    from scripts.build_prototypes import normalize_patches
    rng = np.random.default_rng(1)
    patches = rng.random((8, 7, 7, 59)).astype(np.float32)
    out = normalize_patches(patches)
    B = patches.shape[0]
    flat = out.reshape(B, -1)
    np.testing.assert_allclose(flat.std(axis=1), 1.0, atol=1e-4)


def test_normalize_patches_constant_patch_no_nan():
    """normalize_patches handles constant patches without producing NaN."""
    from scripts.build_prototypes import normalize_patches
    patches = np.ones((4, 7, 7, 59), dtype=np.float32) * 0.3
    out = normalize_patches(patches)
    assert np.isfinite(out).all(), "constant patch should not produce NaN"


def test_compute_prototypes_shape():
    """compute_prototypes returns (n_classes, embed_dim) array."""
    from scripts.build_prototypes import compute_prototypes, CLASS_NAMES
    rng = np.random.default_rng(2)
    embed_dim = 128
    # 5 classes, ~20 embeddings each, already L2-normalized
    emb_per_class = {}
    for name in CLASS_NAMES:
        raw = rng.random((20, embed_dim)).astype(np.float32)
        emb_per_class[name] = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    protos, counts = compute_prototypes(emb_per_class)
    assert protos.shape == (len(CLASS_NAMES), embed_dim)


def test_compute_prototypes_l2_normalized():
    """Each prototype has unit L2 norm."""
    from scripts.build_prototypes import compute_prototypes, CLASS_NAMES
    rng = np.random.default_rng(3)
    embed_dim = 128
    emb_per_class = {}
    for name in CLASS_NAMES:
        raw = rng.random((15, embed_dim)).astype(np.float32)
        emb_per_class[name] = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    protos, _ = compute_prototypes(emb_per_class)
    norms = np.linalg.norm(protos, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_compute_prototypes_pixel_counts():
    """n_pixels_per_class matches the number of embeddings supplied."""
    from scripts.build_prototypes import compute_prototypes, CLASS_NAMES
    rng = np.random.default_rng(4)
    embed_dim = 128
    counts_in = {name: rng.integers(10, 50) for name in CLASS_NAMES}
    emb_per_class = {}
    for name, n in counts_in.items():
        raw = rng.random((n, embed_dim)).astype(np.float32)
        emb_per_class[name] = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    _, counts_out = compute_prototypes(emb_per_class)
    for name in CLASS_NAMES:
        assert counts_out[name] == counts_in[name]


def test_compute_prototypes_zero_embeddings_raises():
    """compute_prototypes raises ValueError when a class has no embeddings."""
    from scripts.build_prototypes import compute_prototypes, CLASS_NAMES
    rng = np.random.default_rng(5)
    embed_dim = 128
    emb_per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        if i == 2:  # leave hcp empty
            emb_per_class[name] = np.zeros((0, embed_dim), dtype=np.float32)
        else:
            raw = rng.random((10, embed_dim)).astype(np.float32)
            emb_per_class[name] = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    with pytest.raises(ValueError, match='hcp'):
        compute_prototypes(emb_per_class)


def test_confidence_tier_filtering():
    """filter_parquet returns different rows for High vs all tiers."""
    from scripts.build_prototypes import filter_parquet
    import pandas as pd
    # Synthetic parquet with 3 tiers and 2 splits
    rng = np.random.default_rng(6)
    n = 60
    df = pd.DataFrame({
        'split': ['train'] * 30 + ['val'] * 30,
        'confidence_tier': ['High'] * 20 + ['Moderate'] * 20 + ['Low'] * 20,
        'olivine_t1': rng.choice([0.0, 1.0], size=n),
        'olivine_t2': rng.choice([0.0, 1.0], size=n),
        'lcp': rng.choice([0.0, 1.0], size=n),
        'hcp': rng.choice([0.0, 1.0], size=n),
        'plagioclase': rng.choice([0.0, 1.0], size=n),
        'other': rng.choice([0.0, 1.0], size=n),
    })
    df_all = filter_parquet(df, splits=['train', 'val'], confidence_tiers=['High', 'Moderate', 'Low'])
    df_high = filter_parquet(df, splits=['train', 'val'], confidence_tiers=['High'])
    assert len(df_high) < len(df_all)
    assert set(df_high['confidence_tier'].unique()) == {'High'}


def test_filter_parquet_preserves_original_index():
    """filter_parquet does NOT reset_index — original row positions are preserved for memmap lookup."""
    from scripts.build_prototypes import filter_parquet
    import pandas as pd
    rng = np.random.default_rng(7)
    n = 30
    df = pd.DataFrame({
        'split': ['train'] * n,
        'confidence_tier': ['High'] * 10 + ['Moderate'] * 10 + ['Low'] * 10,
        'olivine_t1': np.zeros(n), 'olivine_t2': np.zeros(n),
        'lcp': np.zeros(n), 'hcp': np.zeros(n),
        'plagioclase': np.zeros(n), 'other': np.zeros(n),
    })
    # Original index after reset is 0..29
    df_filtered = filter_parquet(df, splits=['train'], confidence_tiers=['High'])
    # Filtered rows should retain their original integer index values (0..9 for High)
    assert list(df_filtered.index) == list(range(10)), \
        "filter_parquet must not reset_index; memmap lookup depends on original positions"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism python -m pytest tests/test_build_prototypes.py -v 2>&1 | head -30
```

Expected: ImportError or ModuleNotFoundError — `build_prototypes` does not exist yet.

---

## Task 2: Implement `build_prototypes.py`

**Files:**
- Create: `scripts/build_prototypes.py`

- [ ] **Step 3: Create `scripts/build_prototypes.py` with the testable functions first**

```python
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

    Args:
        df: raw mrral_pixels parquet DataFrame
        splits: e.g. ['train', 'val']
        confidence_tiers: e.g. ['High', 'Moderate', 'Low']

    Returns:
        Filtered DataFrame with added 'olivine' column.
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
```

- [ ] **Step 4: Run the tests**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism python -m pytest tests/test_build_prototypes.py -v 2>&1 | tail -20
```

Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_prototypes.py tests/test_build_prototypes.py
git commit -m "feat: add build_prototypes.py with per-class prototype extraction"
```

---

## Task 3: Smoke-test build_prototypes.py end-to-end

This verifies the script runs against real data and produces valid output. It is NOT a unit test — it requires checkpoints and the patch cache.

- [ ] **Step 6: Run with fine-tuned encoder (all tiers)**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism python scripts/build_prototypes.py \
    --ckpt checkpoints/spvit_lrscale0005_best.pt \
    --out data/prototypes/proto_finetuned_all.npz 2>&1 | tail -15
```

Expected output (approximate):
```
Device: cuda
Loading encoder from checkpoints/spvit_lrscale0005_best.pt
  encoder tag: spvit_lrscale0005_best
  1967871 pixels after filtering ...
Extracting embeddings for labeled pixels...
  olivine     : 123,456 hard-positive pixels
  lcp         : 456,789 hard-positive pixels
  ...
Saved → data/prototypes/proto_finetuned_all.npz
  olivine     : norm=1.000000, n=...
```

- [ ] **Step 7: Verify prototype norms in Python**

```bash
conda run -n crism python -c "
import numpy as np
d = np.load('data/prototypes/proto_finetuned_all.npz', allow_pickle=True)
print('prototypes shape:', d['prototypes'].shape)
print('norms:', np.linalg.norm(d['prototypes'], axis=1))
print('class_names:', d['class_names'])
print('n_pixels_per_class:', d['n_pixels_per_class'])
"
```

Expected: shape `(5, 128)`, all norms ≈ 1.0.

- [ ] **Step 8: Run with MAE encoder**

```bash
conda run -n crism python scripts/build_prototypes.py \
    --ckpt checkpoints/spatial_mae_128d_6l_best.pt \
    --out data/prototypes/proto_mae_all.npz 2>&1 | tail -10
```

- [ ] **Step 9: Run with High confidence only**

```bash
conda run -n crism python scripts/build_prototypes.py \
    --ckpt checkpoints/spvit_lrscale0005_best.pt \
    --confidence_tiers High \
    --out data/prototypes/proto_finetuned_high.npz 2>&1 | tail -10
```

- [ ] **Step 10: Commit**

```bash
git add data/prototypes/.gitkeep 2>/dev/null || true
git commit -m "feat: build prototype files for finetuned and MAE encoders (not committed — gitignored)"
```

(Note: the `.npz` files themselves are not committed — they will be added to `.gitignore` in Chunk 2.)

---

## Chunk 2: classify_tile_prototype.py

## Task 4: Tests for cosine similarity (pure functions)

**Files:**
- Create: `tests/test_classify_tile_prototype.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_classify_tile_prototype.py
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_cosine_sim_shape():
    """cosine_similarity_classify returns (n_pixels, 5) array."""
    from scripts.classify_tile_prototype import cosine_similarity_classify
    rng = np.random.default_rng(0)
    embs = rng.random((100, 128)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    protos = rng.random((5, 128)).astype(np.float32)
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    result = cosine_similarity_classify(embs, protos)
    assert result.shape == (100, 5)


def test_cosine_sim_range():
    """All similarity values are in [0, 1] after clipping."""
    from scripts.classify_tile_prototype import cosine_similarity_classify
    rng = np.random.default_rng(1)
    embs = rng.standard_normal((200, 128)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    protos = rng.standard_normal((5, 128)).astype(np.float32)
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    result = cosine_similarity_classify(embs, protos)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_cosine_sim_perfect_match():
    """Query equal to a prototype gets similarity 1.0 for that class."""
    from scripts.classify_tile_prototype import cosine_similarity_classify
    rng = np.random.default_rng(2)
    protos = rng.standard_normal((5, 128)).astype(np.float32)
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    # Use prototype 2 (hcp) as query
    query = protos[2:3].copy()  # (1, 128) — already unit norm
    result = cosine_similarity_classify(query, protos)
    assert result.shape == (1, 5)
    assert abs(result[0, 2] - 1.0) < 1e-5, f"Expected 1.0, got {result[0, 2]}"


def test_invalid_pixels_masked():
    """Pixels outside valid_mask are set to 0.0 in output."""
    from scripts.classify_tile_prototype import apply_valid_mask
    H, W = 10, 12
    sims = np.random.rand(H * W, 5).astype(np.float32)
    valid_mask = np.ones((H, W), dtype=bool)
    valid_mask[0, 0] = False
    valid_mask[3, 5] = False
    result = apply_valid_mask(sims, valid_mask)
    assert result.shape == (H, W, 5)
    np.testing.assert_array_equal(result[0, 0], np.zeros(5, dtype=np.float32))
    np.testing.assert_array_equal(result[3, 5], np.zeros(5, dtype=np.float32))
    # Valid pixels should be unchanged
    assert result[1, 1].sum() > 0


def test_load_prototype_npz(tmp_path):
    """load_prototype_npz returns prototypes array, class_names, and ckpt_tag."""
    from scripts.classify_tile_prototype import load_prototype_npz
    protos = np.random.rand(5, 128).astype(np.float32)
    np.savez_compressed(
        str(tmp_path / 'test_proto.npz'),
        prototypes=protos,
        class_names=np.array(['olivine', 'lcp', 'hcp', 'plagioclase', 'other']),
        encoder_ckpt=np.array('checkpoints/spvit_lrscale0005_best.pt'),
        confidence_tiers_used=np.array(['High', 'Moderate', 'Low']),
        splits_used=np.array(['train', 'val']),
        n_pixels_per_class=np.array([100, 200, 50, 75, 150]),
    )
    p, names, ckpt_path = load_prototype_npz(str(tmp_path / 'test_proto.npz'))
    assert p.shape == (5, 128)
    assert list(names) == ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']
    assert 'spvit_lrscale0005' in ckpt_path
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n crism python -m pytest tests/test_classify_tile_prototype.py -v 2>&1 | head -20
```

Expected: ImportError — `classify_tile_prototype` does not exist yet.

---

## Task 5: Implement `classify_tile_prototype.py`

**Files:**
- Create: `scripts/classify_tile_prototype.py`

- [ ] **Step 3: Create `scripts/classify_tile_prototype.py`**

```python
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
from typing import List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.spatial_spectral_transformer import (
    SpatialSpectralClassifier,
    SpatialSpectralTransformer,
)
from scripts.build_prototypes import normalize_patches, load_encoder
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


def load_prototype_npz(path: str) -> Tuple[np.ndarray, List[str], str]:
    """Load prototype .npz produced by build_prototypes.py.

    Returns:
        prototypes: (n_classes, embed_dim) float32
        class_names: list of class name strings
        encoder_ckpt: checkpoint path string (for loading the encoder)
    """
    data = np.load(path, allow_pickle=True)
    prototypes = data['prototypes']
    class_names = [str(x) for x in data['class_names']]
    encoder_ckpt = str(data['encoder_ckpt'])
    return prototypes, class_names, encoder_ckpt


# ── tile loading & patch extraction ──────────────────────────────────────────

def load_tile(img_path: str):
    """Load mrral tile; returns (data HWC, valid_mask HW, transform, crs)."""
    import rasterio
    with rasterio.open(img_path) as src:
        data = src.read(list(range(1, N_BANDS + 1))).astype(np.float32)
        transform = src.transform
        crs = src.crs
    nodata_mask = (data == NODATA) | ~np.isfinite(data)
    data = np.clip(data, 0.0, CLIP_MAX)
    data[nodata_mask] = 0.0
    valid_mask = ~nodata_mask.any(axis=0)
    return data.transpose(1, 2, 0), valid_mask, transform, crs


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
        for ci, class_name in enumerate(CLASS_NAMES):
            ax = fig.add_subplot(gs[1 + enc_i, ci])
            im = ax.imshow(sims_hw5[:, :, ci], vmin=0, vmax=1, cmap='viridis', aspect='auto')
            if enc_i == 0:
                ax.set_title(class_name, fontsize=9)
            if ci == 0:
                ax.set_ylabel(enc_label, fontsize=9, rotation=90, labelpad=4)
            ax.axis('off')
        plt.colorbar(im, ax=fig.axes[-1], fraction=0.03, pad=0.02)

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
        return sims_hw5, embs, tag

    # Run proto_a (always)
    sims_a, _, tag_a = run_proto(args.proto_a)

    # Save probs from proto_a
    if args.save_probs:
        from scripts.classify_tile_supervised import save_probs
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
```

- [ ] **Step 4: Run the tests**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism python -m pytest tests/test_classify_tile_prototype.py -v 2>&1 | tail -20
```

Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/classify_tile_prototype.py tests/test_classify_tile_prototype.py
git commit -m "feat: add classify_tile_prototype.py with cosine similarity inference"
```

---

## Task 6: Update .gitignore and run full test suite

**Files:**
- Modify: `.gitignore`

- [ ] **Step 6: Add `data/prototypes/` to .gitignore**

Open `.gitignore` and add:
```
data/prototypes/
```
Place it near the existing `data/vector/` line.

- [ ] **Step 7: Commit .gitignore**

```bash
git add .gitignore
git commit -m "chore: gitignore data/prototypes/ (derived from gitignored checkpoints)"
```

- [ ] **Step 8: Run full test suite**

```bash
cd /mnt/mrdr/crism_classification
conda run -n crism python -m pytest tests/ -v 2>&1 | tail -30
```

Expected: All tests pass (no regressions).

---

## Task 7: End-to-end smoke test

- [ ] **Step 9: Classify T0435 with proto_a (fine-tuned)**

```bash
conda run -n crism python scripts/classify_tile_prototype.py \
    --tile /mnt/mrdr/mc26/t0435_mrral_40s323_0327_4.img \
    --proto_a data/prototypes/proto_finetuned_all.npz \
    --save_probs /tmp/t0435_proto_finetuned_probs.npz 2>&1 | tail -10
```

- [ ] **Step 10: Classify T0435 with comparison figure (proto_a vs MAE)**

```bash
conda run -n crism python scripts/classify_tile_prototype.py \
    --tile /mnt/mrdr/mc26/t0435_mrral_40s323_0327_4.img \
    --proto_a data/prototypes/proto_finetuned_all.npz \
    --proto_b data/prototypes/proto_mae_all.npz \
    --supervised_probs /tmp/t0435_mrral_40s323_0327_4_probs.npz \
    --save_probs /tmp/t0435_proto_finetuned_probs.npz \
    --out reports/fig_prototype_t0435.png 2>&1 | tail -10
```

Expected: `Saved figure → reports/fig_prototype_t0435.png`

- [ ] **Step 11: Repeat for T0434**

```bash
conda run -n crism python scripts/classify_tile_prototype.py \
    --tile /mnt/mrdr/mc26/t0434_mrral_40s318_0327_4.img \
    --proto_a data/prototypes/proto_finetuned_all.npz \
    --proto_b data/prototypes/proto_mae_all.npz \
    --supervised_probs /tmp/t0434_mrral_40s318_0327_4_probs.npz \
    --save_probs /tmp/t0434_proto_finetuned_probs.npz \
    --out reports/fig_prototype_t0434.png 2>&1 | tail -10
```

- [ ] **Step 12: Verify .npz shape**

```bash
conda run -n crism python -c "
import numpy as np
d = np.load('/tmp/t0435_proto_finetuned_probs.npz')
print('probs shape:', d['probs'].shape)
print('probs range:', d['probs'].min(), d['probs'].max())
print('valid pixels:', d['valid_mask'].sum())
"
```

Expected: shape `(H, W, 5)`, values in `[0, 1]`.

- [ ] **Step 13: Final commit**

```bash
git add reports/fig_prototype_t0435.png reports/fig_prototype_t0434.png
git commit -m "feat: generate prototype classifier comparison figures for T0435 and T0434"
```

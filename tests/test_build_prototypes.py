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
    counts_in = {name: int(rng.integers(10, 50)) for name in CLASS_NAMES}
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
    df_filtered = filter_parquet(df, splits=['train'], confidence_tiers=['High'])
    assert list(df_filtered.index) == list(range(10)), \
        "filter_parquet must not reset_index; memmap lookup depends on original positions"

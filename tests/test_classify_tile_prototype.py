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

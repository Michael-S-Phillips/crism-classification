import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_encoder_forward_shape():
    from models.spatial_spectral_transformer import SpatialSpectralTransformer
    enc = SpatialSpectralTransformer(n_bands=59, patch_size=7, embed_dim=64, n_heads=4, n_layers=2)
    x = torch.randn(4, 7, 7, 59)
    out = enc(x)
    # CLS + 49 spatial tokens
    assert out.shape == (4, 50, 64), f"Got {out.shape}"


def test_encoder_encode_visible():
    from models.spatial_spectral_transformer import SpatialSpectralTransformer
    enc = SpatialSpectralTransformer(n_bands=59, patch_size=7, embed_dim=64, n_heads=4, n_layers=2)
    x = torch.randn(4, 7, 7, 59)
    # Keep 12 visible tokens out of 49
    visible_ids = torch.stack([torch.randperm(49)[:12] for _ in range(4)])
    out = enc.encode_visible(x, visible_ids)
    assert out.shape == (4, 13, 64), f"Got {out.shape}"  # CLS + 12 visible


def test_classifier_output_shape():
    from models.spatial_spectral_transformer import SpatialSpectralClassifier
    clf = SpatialSpectralClassifier(n_bands=59, patch_size=7, n_classes=5,
                                     embed_dim=64, n_heads=4, n_layers=2)
    x = torch.randn(4, 7, 7, 59)
    out = clf(x)
    assert out.shape == (4, 5), f"Got {out.shape}"


def test_classifier_param_groups():
    from models.spatial_spectral_transformer import SpatialSpectralClassifier
    clf = SpatialSpectralClassifier(n_bands=59, patch_size=7, n_classes=5,
                                     embed_dim=64, n_heads=4, n_layers=2)
    groups = clf.get_param_groups(head_lr=1e-3, encoder_lr=1e-4)
    assert len(groups) == 2
    assert groups[0]['lr'] == 1e-4   # encoder
    assert groups[1]['lr'] == 1e-3   # head
    # All params accounted for
    all_ids = {id(p) for g in groups for p in g['params']}
    assert all_ids == {id(p) for p in clf.parameters()}


def test_classifier_deterministic_in_eval():
    from models.spatial_spectral_transformer import SpatialSpectralClassifier
    clf = SpatialSpectralClassifier(n_bands=59, patch_size=7, n_classes=5,
                                     embed_dim=64, n_heads=4, n_layers=2, dropout=0.1)
    clf.eval()
    x = torch.randn(4, 7, 7, 59)
    with torch.no_grad():
        assert torch.allclose(clf(x), clf(x)), "eval() should be deterministic"

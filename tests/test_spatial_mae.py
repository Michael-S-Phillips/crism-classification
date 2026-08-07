import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_mae_forward_returns_scalar_loss():
    from models.spatial_mae import SpatialSpectralMAE
    model = SpatialSpectralMAE(n_bands=59, patch_size=7, embed_dim=64,
                                n_heads=4, n_layers=2, mask_ratio=0.75)
    x = torch.randn(4, 7, 7, 59)
    loss, recon, mask = model(x)
    assert loss.shape == (), f"Loss must be scalar, got {loss.shape}"
    assert loss.item() > 0


def test_mae_reconstruction_shape():
    from models.spatial_mae import SpatialSpectralMAE
    model = SpatialSpectralMAE(n_bands=59, patch_size=7, embed_dim=64,
                                n_heads=4, n_layers=2, mask_ratio=0.75)
    x = torch.randn(4, 7, 7, 59)
    loss, recon, mask = model(x)
    assert recon.shape == (4, 49, 59), f"Got {recon.shape}"
    assert mask.shape == (4, 49), f"Got {mask.shape}"


def test_mae_mask_ratio():
    from models.spatial_mae import SpatialSpectralMAE
    model = SpatialSpectralMAE(n_bands=59, patch_size=7, embed_dim=64,
                                n_heads=4, n_layers=2, mask_ratio=0.75)
    x = torch.randn(8, 7, 7, 59)
    _, _, mask = model(x)
    # Each sample should have int(49 * 0.75) = 36 tokens masked
    per_sample = mask.float().sum(dim=1)
    assert (per_sample == 36).all(), f"Expected 36 masked per sample, got {per_sample}"


def test_mae_encoder_state_dict_loads_into_classifier():
    from models.spatial_mae import SpatialSpectralMAE
    from models.spatial_spectral_transformer import SpatialSpectralClassifier
    mae = SpatialSpectralMAE(n_bands=59, patch_size=7, embed_dim=64,
                              n_heads=4, n_layers=2)
    clf = SpatialSpectralClassifier(n_bands=59, patch_size=7, n_classes=5,
                                     embed_dim=64, n_heads=4, n_layers=2)
    state = mae.encoder_state_dict()
    missing, unexpected = clf.load_encoder_state_dict(state)
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"
    assert len(missing) == 0, f"Missing keys: {missing}"


def test_mae_encode_no_masking():
    from models.spatial_mae import SpatialSpectralMAE
    model = SpatialSpectralMAE(n_bands=59, patch_size=7, embed_dim=64,
                                n_heads=4, n_layers=2)
    x = torch.randn(4, 7, 7, 59)
    emb = model.encode(x)
    # encode() returns center-pixel embedding
    assert emb.shape == (4, 64), f"Got {emb.shape}"

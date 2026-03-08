import torch
import pytest

def test_mlp_output_shape():
    from models.mlp import MLP
    model = MLP(n_features=60, n_classes=6)
    x = torch.randn(8, 60)
    out = model(x)
    assert out.shape == (8, 6)

def test_mlp_no_sigmoid_in_forward():
    """MLP should return logits, not probabilities."""
    from models.mlp import MLP
    model = MLP()
    x = torch.zeros(4, 60)
    out = model(x)
    # If sigmoid applied, all outputs would be 0.5 for zero input
    # Logits for zero input after linear layers will be near 0 but not exactly 0.5
    assert not torch.allclose(out, torch.full_like(out, 0.5))

def test_cnn_output_shape():
    from models.cnn import SpectralSpatialCNN
    model = SpectralSpatialCNN(n_bands=60, n_classes=6, patch_size=7)
    x = torch.randn(4, 60, 7, 7)
    out = model(x)
    assert out.shape == (4, 6)

def test_vit_output_shape():
    from models.vit import SpectralViT
    model = SpectralViT(n_bands=60, n_classes=6, patch_size=7, embed_dim=64, n_heads=4, n_layers=2)
    x = torch.randn(4, 60, 7, 7)
    out = model(x)
    assert out.shape == (4, 6)

def test_cnn_dropout_parameter():
    """CNN should accept a dropout parameter and apply it."""
    from models.cnn import SpectralSpatialCNN
    model = SpectralSpatialCNN(n_bands=60, n_classes=6, patch_size=7, dropout=0.5)
    model.train()
    x = torch.randn(4, 60, 7, 7)
    out1 = model(x)
    out2 = model(x)
    assert not torch.allclose(out1, out2), "Dropout should cause stochastic outputs in train mode"
    model.eval()
    out3 = model(x)
    out4 = model(x)
    assert torch.allclose(out3, out4), "No dropout in eval mode"

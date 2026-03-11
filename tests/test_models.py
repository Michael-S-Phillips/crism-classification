import torch
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def test_spectral_cnn1d_forward_shape():
    from models.spectral_cnn import SpectralCNN1D
    model = SpectralCNN1D(n_bands=59, n_classes=6)
    x = torch.randn(4, 59)
    out = model(x)
    assert out.shape == (4, 6)


def test_spectral_cnn1d_dropout_parameter():
    from models.spectral_cnn import SpectralCNN1D
    m = SpectralCNN1D(n_bands=59, n_classes=6, dropout=0.4)
    assert m is not None


def test_spectral_transformer_forward_shape():
    from models.spectral_transformer import SpectralTransformer
    model = SpectralTransformer(n_bands=59, n_classes=6, embed_dim=64, n_heads=4, n_layers=4)
    x = torch.randn(4, 59)
    out = model(x)
    assert out.shape == (4, 6)


def test_spectral_transformer_get_param_groups():
    from models.spectral_transformer import SpectralTransformer
    model = SpectralTransformer(n_bands=59, n_classes=5, embed_dim=64, n_heads=2, n_layers=2)
    groups = model.get_param_groups(head_lr=3e-4, encoder_lr=3e-5)
    assert len(groups) == 2
    assert groups[0]['lr'] == 3e-5, "encoder group should get slow LR"
    assert groups[1]['lr'] == 3e-4, "head group should get fast LR"
    head_param_ids = {id(p) for p in model.head.parameters()}
    encoder_param_ids = {id(p) for p in groups[0]['params']}
    assert not (head_param_ids & encoder_param_ids), "head params must not appear in encoder group"
    all_group_ids = encoder_param_ids | {id(p) for p in groups[1]['params']}
    all_model_ids = {id(p) for p in model.parameters()}
    assert all_group_ids == all_model_ids, "All parameters must be in exactly one group"


def test_train_torch_differential_lr():
    """Training with encoder_lr_scale should not raise and should produce valid metrics."""
    import pandas as pd, numpy as np
    from training.train_torch import train_torch_model
    from models.spectral_transformer import SpectralTransformer
    rng = np.random.default_rng(1)
    n = 120
    df = pd.DataFrame({
        **{f'm{i}': rng.random(n).astype('float32') for i in range(59)},
        'olivine_t1': rng.integers(0, 2, n).astype('float32'),
        'olivine_t2': rng.integers(0, 2, n).astype('float32'),
        'lcp':  rng.integers(0, 2, n).astype('float32'),
        'hcp':  rng.integers(0, 2, n).astype('float32'),
        'plagioclase': rng.integers(0, 2, n).astype('float32'),
        'other': rng.integers(0, 2, n).astype('float32'),
        'confidence_weight': np.ones(n, dtype='float32'),
        'confidence_tier': ['High'] * n,
        'split': ['train'] * 80 + ['val'] * 40,
    })
    model = SpectralTransformer(n_bands=59, n_classes=5, embed_dim=32, n_heads=2, n_layers=2)
    metrics = train_torch_model(
        model=model, df=df, model_name='test_diffr',
        max_epochs=2, batch_size=32, lr=3e-4,
        patience=5, use_wandb=False, checkpoint_dir=None,
        encoder_lr_scale=0.1,
    )
    assert 'val_mAP' in metrics
    assert 0.0 <= metrics['val_mAP'] <= 1.0


def test_spectral_transformer_mask_token():
    from models.spectral_transformer import SpectralTransformer
    model = SpectralTransformer(n_bands=59, n_classes=6)
    x = torch.randn(2, 59)
    x[:, 10:20] = 0.0   # simulate masked bands
    out = model(x)
    assert out.shape == (2, 6)


def test_hybrid_classifier_output_shape():
    from models.hybrid_classifier import SpectralHybridClassifier
    model = SpectralHybridClassifier(
        n_mrral=59, n_mrrsu=60, n_classes=5,
        embed_dim=64, n_heads=2, n_layers=2,
    )
    x = torch.randn(4, 119)
    out = model(x)
    assert out.shape == (4, 5), f"Expected (4, 5), got {out.shape}"


def test_hybrid_classifier_get_param_groups():
    from models.hybrid_classifier import SpectralHybridClassifier
    model = SpectralHybridClassifier(
        n_mrral=59, n_mrrsu=60, n_classes=5, embed_dim=64, n_heads=2, n_layers=2,
    )
    groups = model.get_param_groups(head_lr=3e-4, encoder_lr=3e-5)
    assert len(groups) == 2
    assert groups[0]['lr'] == 3e-5
    assert groups[1]['lr'] == 3e-4
    all_group_ids = set()
    for g in groups:
        ids = {id(p) for p in g['params']}
        assert not (ids & all_group_ids), "Param groups must not overlap"
        all_group_ids |= ids
    all_model_ids = {id(p) for p in model.parameters()}
    assert all_group_ids == all_model_ids


def test_hybrid_classifier_load_encoder_state_dict():
    """load_encoder_state_dict should load pretrained encoder weights without error."""
    from models.hybrid_classifier import SpectralHybridClassifier
    m1 = SpectralHybridClassifier(n_mrral=59, n_mrrsu=60, n_classes=5, embed_dim=32, n_heads=2, n_layers=2)
    m2 = SpectralHybridClassifier(n_mrral=59, n_mrrsu=60, n_classes=5, embed_dim=32, n_heads=2, n_layers=2)
    encoder_state = {k: v for k, v in m1.encoder.state_dict().items()
                     if not k.startswith('head.')}
    missing, unexpected = m2.load_encoder_state_dict(encoder_state)
    assert len(missing) == 0, f"Unexpected missing keys: {missing}"
    for k, v in m1.encoder.state_dict().items():
        if not k.startswith('head.'):
            assert torch.allclose(m2.encoder.state_dict()[k], v), f"Weight {k} not loaded"


def test_hybrid_classifier_returns_logits_not_probs():
    from models.hybrid_classifier import SpectralHybridClassifier
    model = SpectralHybridClassifier(n_mrral=59, n_mrrsu=60, n_classes=5, embed_dim=32, n_heads=2, n_layers=2)
    model.eval()
    x = torch.zeros(2, 119)
    out = model(x)
    assert not torch.allclose(out, torch.full_like(out, 0.5))

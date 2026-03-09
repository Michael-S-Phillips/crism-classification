import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_spectral_mae_forward():
    from models.mae import SpectralMAE
    model = SpectralMAE(n_bands=59, embed_dim=128, n_heads=4, n_layers=4, mask_ratio=0.4)
    x = torch.randn(4, 59)
    loss, pred, mask = model(x)
    assert loss.shape == (), "Loss should be scalar"
    assert pred.shape == (4, 59), f"Pred shape {pred.shape}"
    assert mask.shape == (4, 59), f"Mask shape {mask.shape}"
    assert 0.3 < mask.float().mean().item() < 0.5, "~40% should be masked"


def test_spectral_mae_encoder_extract():
    from models.mae import SpectralMAE
    model = SpectralMAE(n_bands=59, embed_dim=128, n_heads=4, n_layers=4)
    x = torch.randn(4, 59)
    embed = model.encode(x)
    assert embed.shape == (4, 128), f"Encoder output shape {embed.shape}"


def test_mae_pretrained_weights_loadable_into_spectral_transformer():
    from models.mae import SpectralMAE
    from models.spectral_transformer import SpectralTransformer
    mae = SpectralMAE(n_bands=59, embed_dim=128, n_heads=4, n_layers=4)
    classifier = SpectralTransformer(n_bands=59, n_classes=6,
                                     embed_dim=128, n_heads=4, n_layers=4)
    state = mae.encoder_state_dict()
    missing, unexpected = classifier.load_encoder_state_dict(state)
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"

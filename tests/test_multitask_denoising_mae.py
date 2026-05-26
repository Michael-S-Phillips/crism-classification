# tests/test_multitask_denoising_mae.py
import torch

from models.multitask_denoising_mae import MultiTaskDenoisingMAE


def _model():
    return MultiTaskDenoisingMAE(
        n_bands=59, patch_size=7, embed_dim=128, n_heads=4, n_layers=6,
        decoder_dim=64, decoder_layers=2, mask_ratio=0.75, n_classes=5,
        sigma_gauss=0.0087, sigma_spike=0.0058, sigma_column=0.0049,
    )


def test_recon_path_unchanged():
    m = _model()
    x = torch.randn(4, 7, 7, 59) * 0.1
    loss, recon, mask = m(x)
    assert loss.ndim == 0 and recon.shape == (4, 49, 59) and mask.shape == (4, 49)


def test_aux_forward_shape():
    m = _model()
    x = torch.randn(3, 7, 7, 59) * 0.1
    logits = m.forward_aux(x)
    assert logits.shape == (3, 5)


def test_aux_uses_full_visibility_center_token():
    # forward_aux must be deterministic given fixed weights (no random masking)
    m = _model().eval()
    x = torch.randn(2, 7, 7, 59) * 0.1
    a = m.forward_aux(x)
    b = m.forward_aux(x)
    torch.testing.assert_close(a, b)


def test_encoder_state_dict_loads_into_classifier():
    from models.spatial_spectral_transformer import SpatialSpectralClassifier
    m = _model()
    clf = SpatialSpectralClassifier(
        n_bands=59, patch_size=7, n_classes=5, embed_dim=128, n_heads=4, n_layers=6,
    )
    missing, unexpected = clf.load_encoder_state_dict(m.encoder_state_dict())
    assert missing == [] and unexpected == []

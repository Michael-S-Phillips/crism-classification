# tests/test_spatial_spectral_classifier_aux.py
import torch

from models.spatial_spectral_classifier_aux import SpatialSpectralClassifierAux


def _model():
    return SpatialSpectralClassifierAux(
        n_bands=59, patch_size=7, n_classes=5, embed_dim=128,
        n_heads=4, n_layers=6, aux_dim=2, aux_hidden=16,
    )


def test_forward_shape():
    m = _model()
    x = torch.randn(4, 7, 7, 59) * 0.1
    aux = torch.randn(4, 2)
    out = m(x, aux)
    assert out.shape == (4, 5)


def test_encoder_loads_from_mae_checkpoint():
    from models.spatial_spectral_transformer import SpatialSpectralTransformer
    m = _model()
    enc = SpatialSpectralTransformer(n_bands=59, patch_size=7, embed_dim=128,
                                     n_heads=4, n_layers=6)
    missing, unexpected = m.load_encoder_state_dict(enc.state_dict())
    assert missing == [] and unexpected == []


def test_param_groups_split_encoder_vs_head():
    m = _model()
    groups = m.get_param_groups(head_lr=1e-3, encoder_lr=1e-6)
    assert len(groups) == 2
    enc_ids = {id(p) for p in m.encoder.parameters()}
    head_group = [g for g in groups if g['lr'] == 1e-3][0]
    head_ids = {id(p) for p in head_group['params']}
    assert enc_ids.isdisjoint(head_ids)
    aux_ids = {id(p) for p in m.aux_mlp.parameters()} | {id(p) for p in m.head.parameters()}
    assert aux_ids == head_ids

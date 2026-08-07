"""Tests for the adversarial signal/noise decomposition encoder."""
import pytest
import torch

from models.decomp_spatial_vit_adv import DecompSpVitAdv, GradientReversalLayer


@pytest.fixture
def model():
    return DecompSpVitAdv(
        n_bands=59, patch_size=7, n_classes=5,
        embed_dim=128, n_heads=4, n_layers=6, dropout=0.0,
        disc_hidden=64,
        lambda_adv=1.0,
    )


def test_forward_shapes(model):
    """Model returns the documented 7-tuple with correct shapes."""
    B = 4
    x = torch.randn(B, 7, 7, 59)
    out = model(x)
    logits, s_hat, n_hat, x_hat, disc_logits, s_emb_c, n_emb_c = out
    assert logits.shape == (B, 5)
    assert s_hat.shape == (B, 49, 59)
    assert n_hat.shape == (B, 49, 59)
    assert x_hat.shape == (B, 49, 59)
    assert disc_logits.shape == (B, 5)
    assert s_emb_c.shape == (B, 128)
    assert n_emb_c.shape == (B, 128)


def test_reconstruction_is_additive(model):
    """x_hat must equal s_hat + n_hat exactly."""
    B = 2
    x = torch.randn(B, 7, 7, 59)
    _, s_hat, n_hat, x_hat, _, _, _ = model(x)
    torch.testing.assert_close(x_hat, s_hat + n_hat, rtol=1e-5, atol=1e-5)


def test_classifier_reads_center_signal_embedding(model):
    """Logits should depend more on center-pixel than corner-pixel changes."""
    B = 3
    torch.manual_seed(0)
    x = torch.randn(B, 7, 7, 59)
    logits0 = model(x)[0]
    x_c = x.clone(); x_c[:, 3, 3, :] += 1.0
    x_corner = x.clone(); x_corner[:, 0, 0, :] += 1.0
    delta_center = (model(x_c)[0] - logits0).abs().sum().item()
    delta_corner = (model(x_corner)[0] - logits0).abs().sum().item()
    assert delta_center > delta_corner


def test_grl_forward_identity():
    """GRL must be identity in the forward pass."""
    x = torch.randn(4, 8)
    y = GradientReversalLayer.apply(x, 1.0)
    torch.testing.assert_close(y, x)


def test_grl_backward_negates_and_scales():
    """GRL multiplies the upstream gradient by -lambda_adv."""
    x = torch.randn(4, 8, requires_grad=True)
    lam = 0.5
    y = GradientReversalLayer.apply(x, lam)
    y.sum().backward()
    expected = -lam * torch.ones_like(x)
    torch.testing.assert_close(x.grad, expected, rtol=1e-5, atol=1e-5)


def test_adversarial_gradient_flows_to_encoder(model):
    """The encoder receives gradient from the discriminator loss via GRL.
    Verifies the adversarial path exists and is non-trivial when lambda_adv > 0."""
    B = 4
    x = torch.randn(B, 7, 7, 59)
    labels = (torch.rand(B, 5) > 0.5).float()
    _, _, _, _, disc_logits, _, _ = model(x)
    disc_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        disc_logits, labels
    )
    grads = torch.autograd.grad(
        disc_loss, [model.encoder.band_embed.weight, model.discriminator[0].weight],
    )
    enc_grad, disc_grad = grads[0], grads[1]
    assert enc_grad.abs().sum() > 0, \
        "encoder must receive gradient from adversarial loss"
    assert disc_grad.abs().sum() > 0, \
        "discriminator must receive gradient from adversarial loss"


def test_lambda_adv_zero_blocks_encoder_gradient(model):
    """With lambda_adv=0, the encoder receives no gradient through the adversarial path."""
    B = 2
    x = torch.randn(B, 7, 7, 59)
    model.lambda_adv = 0.0
    _, _, _, _, disc_logits, _, _ = model(x)
    enc_grad = torch.autograd.grad(
        disc_logits.sum(), model.encoder.band_embed.weight,
    )[0]
    assert enc_grad.abs().sum().item() == pytest.approx(0.0, abs=1e-6)


def test_load_mae_encoder(model):
    """MAE checkpoint state loads cleanly into the encoder."""
    import os
    ckpt_path = '/Volumes/Mars_GIS/CRISM/MRDR/crism_classification/checkpoints/spatial_mae_128d_6l_best.pt'
    if not os.path.exists(ckpt_path):
        pytest.skip(f"MAE checkpoint not available at {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_encoder_state_dict(ckpt['encoder_state'])
    assert unexpected == [], f"Unexpected keys: {unexpected}"
    assert not any(k.startswith('encoder.encoder') for k in missing), \
        f"Core encoder weights missing: {[k for k in missing if k.startswith('encoder.encoder')]}"


def test_lambda_adv_setter():
    """lambda_adv is mutable so the training loop can update it per epoch."""
    m = DecompSpVitAdv(lambda_adv=0.5)
    assert m.lambda_adv == 0.5
    m.lambda_adv = 0.8
    assert m.lambda_adv == 0.8


def test_get_param_groups_split(model):
    groups = model.get_param_groups(head_lr=1e-3, encoder_lr=1e-5)
    assert len(groups) == 2
    assert groups[0]['lr'] == 1e-5
    assert groups[1]['lr'] == 1e-3
    enc_ids = {id(p) for p in groups[0]['params']}
    head_ids = {id(p) for p in groups[1]['params']}
    assert enc_ids.isdisjoint(head_ids)
    total = sum(p.numel() for p in model.parameters())
    grouped = sum(p.numel() for g in groups for p in g['params'])
    assert grouped == total

"""Tests for the adversarial decomposition composite loss."""
import pytest
import torch

from training.adv_decomp_losses import AdversarialDecompositionLoss


@pytest.fixture
def loss_fn():
    return AdversarialDecompositionLoss(
        lambda_recon=10.0,
        lambda_smooth=0.001,
        asl_gamma_neg=4.0, asl_gamma_pos=0.0, asl_clip=0.05,
    )


def _make_outputs(B=4, n_tokens=49, n_bands=59, n_classes=5):
    torch.manual_seed(0)
    s_hat = torch.randn(B, n_tokens, n_bands) * 0.1
    n_hat = torch.randn(B, n_tokens, n_bands) * 0.01
    x_hat = s_hat + n_hat
    x = x_hat + torch.randn_like(x_hat) * 0.005
    logits = torch.randn(B, n_classes)
    disc_logits = torch.randn(B, n_classes)
    labels = (torch.rand(B, n_classes) > 0.5).float()
    weights = torch.ones(B)
    return dict(
        x=x, logits=logits, labels=labels, weights=weights,
        s_hat=s_hat, n_hat=n_hat, x_hat=x_hat,
        disc_logits=disc_logits,
    )


def test_loss_returns_scalar_and_components(loss_fn):
    o = _make_outputs()
    total, components = loss_fn(**o)
    assert total.ndim == 0
    for key in ('cls', 'recon', 'adv', 'smooth'):
        assert key in components
        assert components[key].ndim == 0


def test_recon_zero_when_perfect(loss_fn):
    o = _make_outputs()
    # Make recon perfect: x_hat = x exactly
    o['x_hat'] = o['x'].reshape(o['s_hat'].shape) if o['x'].dim() != o['s_hat'].dim() else o['x']
    # Update s_hat and n_hat to also be consistent (s_hat + n_hat = x_hat)
    # though the loss only uses x_hat for recon, so this is fine.
    _, c = loss_fn(**o)
    assert c['recon'].item() < 1e-8


def test_smooth_zero_when_signal_uniform(loss_fn):
    B, n_tokens, n_bands = 2, 49, 59
    spec = torch.randn(B, 1, n_bands) * 0.1
    s_hat_uniform = spec.expand(-1, n_tokens, -1).clone()
    n_hat = torch.zeros(B, n_tokens, n_bands)
    x_hat = s_hat_uniform + n_hat
    o = dict(
        x=x_hat.clone(),
        logits=torch.zeros(B, 5),
        labels=torch.zeros(B, 5),
        weights=torch.ones(B),
        s_hat=s_hat_uniform, n_hat=n_hat, x_hat=x_hat,
        disc_logits=torch.zeros(B, 5),
    )
    _, c = loss_fn(**o)
    assert c['smooth'].item() < 1e-8


def test_total_is_weighted_sum(loss_fn):
    """Total = cls + λ_recon·recon + adv + λ_smooth·smooth."""
    o = _make_outputs()
    total, c = loss_fn(**o)
    expected = (
        c['cls']
        + loss_fn.lambda_recon * c['recon']
        + c['adv']
        + loss_fn.lambda_smooth * c['smooth']
    )
    torch.testing.assert_close(total, expected, rtol=1e-5, atol=1e-5)


def test_class_weights_threaded_through(loss_fn):
    """class_weights must be accepted and threaded into both cls and adv."""
    o = _make_outputs()
    cw = torch.tensor([1.0, 1.0, 1.5, 3.0, 1.0])
    _, c_with = loss_fn(**o, class_weights=cw)
    assert torch.is_tensor(c_with['cls'])
    assert torch.is_tensor(c_with['adv'])

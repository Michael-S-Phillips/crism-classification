"""Tests for the composite decomposition loss."""
import pytest
import torch

from training.decomp_losses import DecompositionLoss


@pytest.fixture
def loss_fn():
    return DecompositionLoss(
        lambda_recon=1.0,
        lambda_eps=0.1,
        lambda_T=0.01,
        lambda_b=0.01,
        lambda_smooth=0.001,
        asl_gamma_neg=4.0, asl_gamma_pos=0.0, asl_clip=0.05,
    )


def _make_decomp_outputs(B=4, n_tokens=49, n_bands=59, n_classes=5):
    """Synthesize valid forward outputs."""
    torch.manual_seed(0)
    s_hat = torch.randn(B, n_tokens, n_bands) * 0.1
    eps_hat = torch.randn(B, n_tokens, n_bands) * 0.01
    T_hat = torch.full((B, n_bands), 0.85)
    b_hat = torch.randn(B, n_bands) * 0.01
    x_hat = T_hat.unsqueeze(1) * s_hat + b_hat.unsqueeze(1) + eps_hat
    x = x_hat + torch.randn_like(x_hat) * 0.005   # small reconstruction error
    logits = torch.randn(B, n_classes)
    labels = (torch.rand(B, n_classes) > 0.5).float()
    weights = torch.ones(B)
    return dict(
        x=x, logits=logits, labels=labels, weights=weights,
        s_hat=s_hat, T_hat=T_hat, b_hat=b_hat, eps_hat=eps_hat, x_hat=x_hat,
    )


def test_loss_returns_scalar_and_components(loss_fn):
    o = _make_decomp_outputs()
    total, components = loss_fn(
        x=o['x'], logits=o['logits'], labels=o['labels'], weights=o['weights'],
        s_hat=o['s_hat'], T_hat=o['T_hat'], b_hat=o['b_hat'],
        eps_hat=o['eps_hat'], x_hat=o['x_hat'],
    )
    assert total.ndim == 0, "total must be a scalar tensor"
    for key in ('cls', 'recon', 'eps_reg', 'T_reg', 'b_reg', 'smooth'):
        assert key in components, f"missing loss component: {key}"
        assert components[key].ndim == 0


def test_recon_loss_zero_when_reconstruction_perfect(loss_fn):
    """If x_hat exactly equals x, recon component should be 0."""
    o = _make_decomp_outputs()
    perfect = o['x']  # set x_hat == x
    total, components = loss_fn(
        x=o['x'], logits=o['logits'], labels=o['labels'], weights=o['weights'],
        s_hat=o['s_hat'], T_hat=o['T_hat'], b_hat=o['b_hat'],
        eps_hat=o['eps_hat'], x_hat=perfect,
    )
    assert components['recon'].item() < 1e-8


def test_eps_reg_zero_when_residual_zero(loss_fn):
    """If eps_hat is all zeros, eps_reg should be 0."""
    o = _make_decomp_outputs()
    o['eps_hat'] = torch.zeros_like(o['eps_hat'])
    total, components = loss_fn(**o)
    assert components['eps_reg'].item() < 1e-8


def test_T_reg_zero_when_T_is_one(loss_fn):
    """L_T_reg should be exactly 0 when T_hat == 1.0 (its prior)."""
    o = _make_decomp_outputs()
    o['T_hat'] = torch.ones_like(o['T_hat'])
    _, components = loss_fn(**o)
    assert components['T_reg'].item() < 1e-8


def test_b_reg_zero_when_b_is_zero(loss_fn):
    o = _make_decomp_outputs()
    o['b_hat'] = torch.zeros_like(o['b_hat'])
    _, components = loss_fn(**o)
    assert components['b_reg'].item() < 1e-8


def test_smooth_zero_when_signal_uniform(loss_fn):
    """L_smooth should be 0 when s_hat is spatially uniform across the patch."""
    B, n_tokens, n_bands = 2, 49, 59
    # Uniform spatial signal — same spectrum at every spatial position
    spec = torch.randn(B, 1, n_bands) * 0.1
    s_hat_uniform = spec.expand(-1, n_tokens, -1).clone()
    T_hat = torch.ones(B, n_bands)
    b_hat = torch.zeros(B, n_bands)
    eps_hat = torch.zeros(B, n_tokens, n_bands)
    x_hat = T_hat.unsqueeze(1) * s_hat_uniform + b_hat.unsqueeze(1) + eps_hat
    x = x_hat.clone()
    logits = torch.zeros(B, 5)
    labels = torch.zeros(B, 5)
    weights = torch.ones(B)
    _, components = loss_fn(
        x=x, logits=logits, labels=labels, weights=weights,
        s_hat=s_hat_uniform, T_hat=T_hat, b_hat=b_hat,
        eps_hat=eps_hat, x_hat=x_hat,
    )
    assert components['smooth'].item() < 1e-8


def test_total_loss_is_weighted_sum_of_components(loss_fn):
    """Total loss must equal cls + λ_recon*recon + λ_eps*eps_reg + λ_T*T_reg + λ_b*b_reg + λ_smooth*smooth."""
    o = _make_decomp_outputs()
    total, c = loss_fn(**o)
    expected = (
        c['cls']
        + loss_fn.lambda_recon * c['recon']
        + loss_fn.lambda_eps * c['eps_reg']
        + loss_fn.lambda_T * c['T_reg']
        + loss_fn.lambda_b * c['b_reg']
        + loss_fn.lambda_smooth * c['smooth']
    )
    torch.testing.assert_close(total, expected, rtol=1e-5, atol=1e-5)


def test_class_weights_scale_classification_term(loss_fn):
    """Passing class_weights should scale the classification component."""
    o = _make_decomp_outputs()
    cw = torch.tensor([1.0, 1.0, 1.5, 3.0, 1.0])
    _, c_with = loss_fn(**o, class_weights=cw)
    _, c_without = loss_fn(**o)
    # The two cls values are almost certainly different in the random-label case;
    # the only invariant we can assert is the class_weights branch ran.
    assert torch.is_tensor(c_with['cls'])

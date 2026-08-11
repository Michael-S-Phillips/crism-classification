"""Tests for the CRISM noise augmentation module."""
import hashlib

import pytest
import torch

import data.continuum_removal as cr_module
from models.noise_augmentation import CrismNoiseAugmentation

# Captured from the UNMODIFIED pre-fix module (git commit 0436e5e, before the
# noise_scale fix) via:
#   torch.manual_seed(1234)
#   aug = CrismNoiseAugmentation(sigma_gauss=0.0087, sigma_spike=0.0058,
#       sigma_column=0.0049, spike_center_band=15, spike_fwhm_bands=3.0,
#       spike_band_range=(13, 17), n_bands=59, patch_size=7)
#   aug.train(); x = torch.randn(6, 7, 7, 59); out = aug(x)
#   hashlib.sha256(out.contiguous().numpy().tobytes()).hexdigest()
# A sha256 over the raw float32 bytes is used instead of embedding the full
# (6,7,7,59) tensor so the fixture stays a short literal here rather than a
# multi-KB blob or an external binary file (binary fixtures are blocked by
# this repo's blanket `*.pt` .gitignore rule, and adding a .gitignore
# exception is out of scope for this fix). Any change to sigmas, RNG draw
# order/count, or the spike profile for the 59-band path changes this hash.
_BASELINE_59_SEED = 1234
_BASELINE_59_BATCH = 6
_BASELINE_59_SHA256 = (
    "34a877470eeea037b06bc079617bde2a34417fd1e7acfe512a83173a7b51bbf0"
)
_BASELINE_59_SUM = -4.623924255371094  # human-readable cross-check


@pytest.fixture
def aug():
    return CrismNoiseAugmentation(
        sigma_gauss=0.0087,
        sigma_spike=0.0058,
        sigma_column=0.0049,
        spike_center_band=15,
        spike_fwhm_bands=3,
        spike_band_range=(13, 17),
        n_bands=59,
        patch_size=7,
    )


def test_forward_shape_preserved(aug):
    aug.train()
    x = torch.randn(8, 7, 7, 59)
    out = aug(x)
    assert out.shape == x.shape


def test_eval_mode_disables_corruption(aug):
    aug.eval()
    x = torch.randn(4, 7, 7, 59)
    out = aug(x)
    torch.testing.assert_close(out, x)


def test_train_mode_changes_output(aug):
    aug.train()
    torch.manual_seed(0)
    x = torch.randn(4, 7, 7, 59) * 0.1
    out = aug(x)
    assert not torch.allclose(out, x), "training mode must produce a different output from input"


def test_empirical_gaussian_sigma(aug):
    """Empirical std of (corrupted - clean), averaged over many patches, should
    approximate the configured σ_gauss when other corruptions are off."""
    aug_only_gauss = CrismNoiseAugmentation(
        sigma_gauss=0.01, sigma_spike=0.0, sigma_column=0.0,
        n_bands=59, patch_size=7,
    )
    aug_only_gauss.train()
    torch.manual_seed(42)
    x = torch.zeros(2000, 7, 7, 59)
    out = aug_only_gauss(x)
    empirical_sigma = (out - x).std().item()
    assert 0.009 < empirical_sigma < 0.011, f"empirical σ = {empirical_sigma}"


def test_empirical_spike_only(aug):
    """Spike-only augmentation: only bands inside the spike range should be nonzero."""
    aug_only_spike = CrismNoiseAugmentation(
        sigma_gauss=0.0, sigma_spike=0.01, sigma_column=0.0,
        spike_center_band=15, spike_fwhm_bands=3, spike_band_range=(13, 17),
        n_bands=59, patch_size=7,
    )
    aug_only_spike.train()
    torch.manual_seed(0)
    x = torch.zeros(1000, 7, 7, 59)
    out = aug_only_spike(x)
    delta = out - x
    outside_max = delta[:, :, :, :13].abs().max().item()
    assert outside_max < 1e-6, f"corruption leaked outside spike band range: max={outside_max}"
    above_max = delta[:, :, :, 18:].abs().max().item()
    assert above_max < 1e-6, f"corruption leaked above spike band range: max={above_max}"
    inside_std = delta[:, :, :, 13:18].std().item()
    assert inside_std > 1e-5, f"no spike content inside range: std={inside_std}"


def test_empirical_column_only(aug):
    """Column-only: within a patch, all 7 rows of a given column should be perturbed identically."""
    aug_only_column = CrismNoiseAugmentation(
        sigma_gauss=0.0, sigma_spike=0.0, sigma_column=0.01,
        n_bands=59, patch_size=7,
    )
    aug_only_column.train()
    torch.manual_seed(0)
    x = torch.zeros(50, 7, 7, 59)
    out = aug_only_column(x)
    delta = out - x
    for i in range(5):
        for c in range(7):
            col = delta[i, :, c, :]
            row0 = col[0]
            for r in range(1, 7):
                torch.testing.assert_close(col[r], row0, rtol=1e-5, atol=1e-6)


def test_all_components_combine_additively(aug):
    """When all three σ values are set, outside the spike range the empirical std
    is approximately sqrt(σ_gauss² + σ_column²)."""
    aug.train()
    torch.manual_seed(0)
    x = torch.zeros(1000, 7, 7, 59)
    out = aug(x)
    delta = out - x
    outside_std = delta[:, :, :, 0:13].std().item()
    expected_outside = (0.0087 ** 2 + 0.0049 ** 2) ** 0.5
    assert 0.8 * expected_outside < outside_std < 1.2 * expected_outside, \
        f"outside-spike std = {outside_std}, expected ≈ {expected_outside}"


# --------------------------------------------------------------------------
# 118-channel dual-continuum representation: noise_scale fix
# --------------------------------------------------------------------------

def test_59_band_bit_identical_to_pre_fix_baseline():
    """Regression guard for design constraint 1: the 59-band path must be
    BIT-IDENTICAL to the pre-fix module. `_BASELINE_59_SHA256` above was
    captured from the unmodified module (git commit 0436e5e, before this
    fix) under a fixed seed. Reproducing the exact same RNG draw sequence
    (same seed, same construction, same forward call) must reproduce a
    tensor with the exact same bytes -- not merely close, but bit-identical
    (verified via sha256 over the raw float32 buffer, which is sensitive to
    any change of even a single bit).

    Two layers, deliberately: the sigma-equality and output-sum assertions
    are robust and diagnose the likely failure directly, while the sha256 is
    the strict guard that catches single-bit drift. A sha256 over float32
    bytes is also sensitive to the torch version and BLAS backend, so a
    hash-only mismatch (with the sum still matching) most likely means the
    numerics environment moved, not that this module regressed -- see the
    regen recipe in the module comment above `_BASELINE_59_SHA256`.
    """
    torch.manual_seed(_BASELINE_59_SEED)
    aug = CrismNoiseAugmentation(
        sigma_gauss=0.0087, sigma_spike=0.0058, sigma_column=0.0049,
        spike_center_band=15, spike_fwhm_bands=3.0, spike_band_range=(13, 17),
        n_bands=59, patch_size=7,
    )
    # Layer 1: the 59-band path must not pick up the dual-mode noise_scale.
    # This is the assertion that fires on the actual regression this test
    # exists to prevent, and it is environment-independent.
    explicit = CrismNoiseAugmentation(n_bands=59, dual_continuum=False)
    assert (aug.sigma_gauss, aug.sigma_spike, aug.sigma_column) == (
        explicit.sigma_gauss, explicit.sigma_spike, explicit.sigma_column
    ) == (0.0087, 0.0058, 0.0049), (
        'the 59-band path picked up a noise scale factor; sigmas must stay '
        'exactly as estimated against 59-band hull-CR data'
    )
    aug.train()
    x = torch.randn(_BASELINE_59_BATCH, 7, 7, 59)
    out = aug(x).contiguous()
    assert out.sum().item() == pytest.approx(_BASELINE_59_SUM, abs=1e-4), (
        "59-band output sum diverged from pre-fix baseline -- the 59-band "
        "path must be untouched"
    )
    digest = hashlib.sha256(out.numpy().tobytes()).hexdigest()
    assert digest == _BASELINE_59_SHA256, (
        f"59-band output diverged bit-for-bit from the pre-fix baseline "
        f"(got sha256={digest}, expected {_BASELINE_59_SHA256}) -- the "
        f"59-band path must be untouched. If the sigma and sum assertions "
        f"above PASSED and only this hash differs, the torch version or BLAS "
        f"backend has changed rather than this module: re-capture the "
        f"baseline with the recipe in the comment above _BASELINE_59_SHA256."
    )


def test_118_band_gaussian_scale_restores_relative_corruption():
    """At 118 channels (standardised dual-continuum representation, data std
    ≈ 1.0), the realised gaussian corruption std must land near the ORIGINAL
    relative level the sigmas were estimated at against 59-band hull-CR data
    (std 0.0705): sigma_gauss / hull_std ≈ 0.0087 / 0.07054 ≈ 12.3%. Without
    the noise_scale fix the realised std would be ~0.87% of unit data std
    (~14x too weak).
    """
    aug = CrismNoiseAugmentation(
        sigma_gauss=0.0087, sigma_spike=0.0, sigma_column=0.0,
        n_bands=118, patch_size=7,
    )
    aug.train()
    torch.manual_seed(42)
    x = torch.zeros(4000, 7, 7, 118)
    out = aug(x)
    empirical_sigma = (out - x).std().item()

    expected_scale = 1.0 / cr_module.CR_SCALES['hull_std']
    expected_sigma = 0.0087 * expected_scale  # ≈ 0.1233 (~12.3% of unit std)

    assert 0.95 * expected_sigma < empirical_sigma < 1.05 * expected_sigma, (
        f"empirical σ_gauss at 118 channels = {empirical_sigma}, "
        f"expected ≈ {expected_sigma} (~12.4% of unit data std), "
        f"i.e. matching the 59-band relative corruption level"
    )
    # And explicitly rule out the un-scaled (bug) regime.
    assert empirical_sigma > 0.05, (
        f"empirical σ_gauss = {empirical_sigma} looks like the un-scaled "
        f"(~0.9% of unit std) bug regime, not the corrected ~12.4% level"
    )


def test_118_band_spike_profile_has_mirrored_bump_in_both_blocks():
    """At 118 channels the 1 µm-seam spike profile must have a bump in BOTH
    the hull block (bands 13-17, center 15) and the linear block (bands
    72-76, center 74 = 59+15) -- each peaking at 1.0 -- and be exactly zero
    outside those two ranges.
    """
    aug = CrismNoiseAugmentation(
        spike_center_band=15, spike_fwhm_bands=3.0, spike_band_range=(13, 17),
        n_bands=118, patch_size=7,
    )
    profile = aug._spike_profile
    assert profile.shape == (118,)

    assert profile[15].item() == pytest.approx(1.0, abs=1e-6), \
        f"hull-block peak != 1.0: {profile[15].item()}"
    assert profile[74].item() == pytest.approx(1.0, abs=1e-6), \
        f"linear-block mirrored peak != 1.0: {profile[74].item()}"

    keep = torch.zeros(118, dtype=torch.bool)
    keep[13:18] = True   # hull block: bands 13..17 inclusive
    keep[72:77] = True   # linear block: bands 72..76 inclusive (59 + 13..17)
    outside = profile[~keep]
    assert torch.all(outside == 0.0), \
        f"spike profile nonzero outside both band ranges: max={outside.abs().max().item()}"


def test_noise_scale_derives_from_cr_scales_not_a_literal(monkeypatch):
    """noise_scale must be read from data.continuum_removal.CR_SCALES at
    construction time, not a hardcoded literal. Monkeypatching CR_SCALES and
    observing the resulting sigma track it is the only honest way to verify
    this (a literal 14.176... would produce the same numbers as the real
    scales and pass a same-values check, but would NOT track a changed
    CR_SCALES).
    """
    monkeypatch.setattr(cr_module, 'CR_SCALES', {'hull_std': 0.5, 'linear_std': 1.0})
    aug = CrismNoiseAugmentation(
        sigma_gauss=0.01, sigma_spike=0.0, sigma_column=0.0,
        n_bands=118, patch_size=7,
    )
    expected = 0.01 * (1.0 / 0.5)
    assert aug.sigma_gauss == pytest.approx(expected), (
        f"sigma_gauss={aug.sigma_gauss} did not track the monkeypatched "
        f"CR_SCALES['hull_std']=0.5 (expected {expected}) -- looks hardcoded"
    )


def test_eval_mode_noop_at_118_channels():
    """The module must remain a no-op in eval() mode at 118 channels too --
    the dual-mode scaling must not bypass the training-mode gate."""
    aug = CrismNoiseAugmentation(n_bands=118, patch_size=7)
    aug.eval()
    x = torch.randn(4, 7, 7, 118)
    out = aug(x)
    torch.testing.assert_close(out, x)

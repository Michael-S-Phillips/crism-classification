# tests/test_synthetic_plag.py
import numpy as np
import pytest

from data.synthetic_plag import interp_to_mrral_wavelengths


def test_interp_basic_linear():
    # library wavelengths 400..500, reflectance = wl/1000 (so 0.4..0.5)
    lib_wl = np.array([400.0, 450.0, 500.0])
    lib_refl = np.array([0.40, 0.45, 0.50])
    target_wl = np.array([425.0, 475.0])
    out = interp_to_mrral_wavelengths(lib_wl, lib_refl, target_wl)
    assert out.shape == (2,)
    np.testing.assert_allclose(out, [0.425, 0.475], atol=1e-6)


def test_interp_drops_sentinel_and_nan():
    # 65535 wavelength sentinel and NaN reflectance bands must be ignored
    lib_wl = np.array([400.0, 450.0, 65535.0, 500.0])
    lib_refl = np.array([0.40, np.nan, 0.99, 0.50])
    target_wl = np.array([450.0])
    out = interp_to_mrral_wavelengths(lib_wl, lib_refl, target_wl)
    # only (400,0.40) and (500,0.50) are valid → interp at 450 = 0.45
    np.testing.assert_allclose(out, [0.45], atol=1e-6)


from data.synthetic_plag import synthesize_patches


def test_synthesize_patches_shape_and_clip():
    rng = np.random.default_rng(0)
    spectrum = np.full(59, 0.2, dtype=np.float32)
    patches = synthesize_patches(spectrum, n_aug=8, rng=rng)
    assert patches.shape == (8, 7, 7, 59)
    assert patches.dtype == np.float32
    assert patches.min() >= 0.0 and patches.max() <= 0.5  # clipped to [0, CLIP_MAX]


def test_synthesize_patches_not_flat():
    # per-pixel noise must make neighbours differ (no flat-tile shortcut)
    rng = np.random.default_rng(1)
    spectrum = np.full(59, 0.2, dtype=np.float32)
    patches = synthesize_patches(spectrum, n_aug=4, rng=rng)
    # within a single patch, the 49 center-band values should not be identical
    band0 = patches[0, :, :, 0].ravel()
    assert band0.std() > 1e-4


def test_synthesize_patches_centered_on_spectrum():
    # mean over many augmentations/pixels should track the source spectrum
    rng = np.random.default_rng(2)
    spectrum = np.linspace(0.1, 0.4, 59).astype(np.float32)
    patches = synthesize_patches(spectrum, n_aug=200, rng=rng,
                                 noise_sigma=0.005, jitter_sigma=0.003,
                                 continuum_scale_range=(0.97, 1.03))
    mean_spec = patches.mean(axis=(0, 1, 2))
    np.testing.assert_allclose(mean_spec, spectrum, atol=0.02)


# append to tests/test_synthetic_plag.py
from data.synthetic_plag import build_synth_rows


def test_build_synth_rows_schema():
    import pandas as pd
    rng = np.random.default_rng(3)
    spectra = {
        "FRT00008842_07_#1_Plagioclase": np.full(59, 0.2, dtype=np.float32),
        "FRT000092B4_07_#1_Plagioclase": np.full(59, 0.25, dtype=np.float32),
    }
    patches, df = build_synth_rows(spectra, n_aug=5, rng=rng,
                                   confidence_tier="High")
    assert patches.shape == (10, 7, 7, 59)              # 2 spectra * 5 aug
    assert len(df) == 10
    # schema must match mrral_pixels.parquet label/meta columns
    for col in ["tile_id", "polygon_id", "pixel_row", "pixel_col",
                "olivine_t1", "olivine_t2", "lcp", "hcp", "plagioclase",
                "other", "confidence_tier", "split"]:
        assert col in df.columns
    assert (df["plagioclase"] == 1).all()
    assert (df["other"] == 0).all()
    assert (df["lcp"] == 0).all() and (df["hcp"] == 0).all()
    assert (df["olivine_t1"] == 0).all() and (df["olivine_t2"] == 0).all()
    assert (df["split"] == "train").all()              # train-only, never val/test
    assert df["tile_id"].str.startswith("SYNTH_PLAG_").all()
    assert [f"m{i}" for i in range(59)] == [c for c in df.columns if c.startswith("m")]


def test_synth_train_injection_excludes_val_and_test_rows(tmp_path):
    """The synth TRAIN set must not contain the synth VAL/TEST rows.

    Regression for the 2026-08-08 plagioclase leak: train_torch built the synth
    train dataset with no `split` argument, so SyntheticPatchDataset served every
    row — including the val and test rows that --synth_val_* then put into VAL.
    The model was validated on plagioclase patches it had trained on, which
    inflated plag val AP in every run using both flags.
    """
    import numpy as np
    import pandas as pd
    from data.dataset import SyntheticPatchDataset

    n = 30
    splits = ['train'] * 18 + ['val'] * 6 + ['test'] * 6
    rows = {'tile_id': [f't{i:04d}' for i in range(n)],
            'polygon_id': list(range(n)),
            'pixel_row': list(range(n)), 'pixel_col': list(range(n)),
            'olivine_t1': [0.0] * n, 'olivine_t2': [0.0] * n, 'lcp': [0.0] * n,
            'hcp': [0.0] * n, 'plagioclase': [1.0] * n, 'other': [0.0] * n,
            'confidence_weight': [1.0] * n, 'confidence_tier': ['High'] * n,
            'split': splits}
    pq = tmp_path / 'rows.parquet'
    pd.DataFrame(rows).to_parquet(pq, index=False)
    npy = tmp_path / 'patches.npy'
    np.save(npy, np.random.rand(n, 7, 7, 59).astype(np.float32))

    train_ds = SyntheticPatchDataset(str(npy), str(pq), split='train')
    val_ds = SyntheticPatchDataset(str(npy), str(pq), split='val')
    unsplit = SyntheticPatchDataset(str(npy), str(pq))

    assert len(train_ds) == 18
    assert len(val_ds) == 6
    # The bug: no split argument serves everything, so val rows land in train.
    assert len(unsplit) == n, 'unfiltered behaviour changed'
    assert len(train_ds) + len(val_ds) < len(unsplit), (
        'train and val must be disjoint subsets, not overlapping views')

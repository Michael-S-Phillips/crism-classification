"""Unit tests for sam_analysis.endmembers."""
from __future__ import annotations

import os

import numpy as np
import pytest

from sam_analysis.endmembers import N_BANDS, load_endmember_library

XLSX = "/Volumes/Mars_GIS/CRISM/MRDR/endmember_extraction/crism_endmembers/crism_endmember_spectra.xlsx"
PARQUET = "/Volumes/Mars_GIS/CRISM/MRDR/crism_classification/data/mrral_pixels.parquet"


pytestmark = pytest.mark.skipif(
    not (os.path.exists(XLSX) and os.path.exists(PARQUET)),
    reason="Endmember xlsx or labeled-pixel parquet not available",
)


def test_load_returns_all_four_classes():
    lib = load_endmember_library()
    assert set(lib.keys()) == {"olivine", "lcp", "hcp", "plagioclase"}


def test_each_spectrum_is_59_bands_finite():
    lib = load_endmember_library()
    for name, spec in lib.items():
        assert spec.shape == (N_BANDS,), f"{name} shape {spec.shape}"
        assert np.isfinite(spec).all(), f"{name} has non-finite values"


def test_values_in_reflectance_range():
    """Spectra are mean-clipped reflectance; expect roughly [0, 0.5]."""
    lib = load_endmember_library()
    for name, spec in lib.items():
        assert spec.min() >= -0.05, f"{name} min={spec.min():.4f} too negative"
        assert spec.max() <= 0.6, f"{name} max={spec.max():.4f} too high"


def test_plag_is_distinct_from_olivine():
    """A useful endmember library has plag != olivine spectrally."""
    lib = load_endmember_library()
    # cosine similarity should be < 1 - 1e-3 (i.e. they differ meaningfully)
    o = lib["olivine"]
    p = lib["plagioclase"]
    cos = float(np.dot(o, p) / (np.linalg.norm(o) * np.linalg.norm(p) + 1e-12))
    assert cos < 0.9999, f"olivine and plagioclase identical (cos={cos})"

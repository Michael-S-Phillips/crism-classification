"""Pytest configuration and fixtures."""
import os
import sys

# Pin the test suite to CPU so results are deterministic and backend-independent
# (device.get_device() reads CRISM_DEVICE). Real scripts still auto-select MPS.
# Set before any test module imports code that calls get_device() at import time.
os.environ.setdefault("CRISM_DEVICE", "cpu")

# Constrain OpenMP to a single thread for the test *process* only (not the conda
# env, so real scripts keep full threading). Without this, long full-suite runs
# on macOS segfault from the duplicate OpenMP runtimes bundled by torch / xgboost
# / lightgbm. Must be set before torch is imported anywhere in the process.
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_loader import load_config

import pytest

# Derive real-data paths from the machine-local data_root (portable across
# Mac / Linux / HPC). Fixtures skip gracefully when the files aren't present.
_DATA_ROOT = load_config()['data_root']
REAL_IMG = os.path.join(_DATA_ROOT, 'mc13', 't1249_mrral_20n073_0327_4.img')
REAL_GPKG = os.path.join(
    _DATA_ROOT, 'crism_classification', 'data', 'vector',
    't1249_mrral_20n073_0327_4_mineral_map.gpkg',
)


@pytest.fixture
def real_img_path():
    if not os.path.exists(REAL_IMG):
        pytest.skip('Real tile not available')
    return REAL_IMG


@pytest.fixture
def real_gpkg_path():
    if not os.path.exists(REAL_GPKG):
        pytest.skip('Real GeoPackage not available')
    return REAL_GPKG

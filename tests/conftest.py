"""Pytest configuration and fixtures."""
import os
import pytest

REAL_IMG = '/mnt/mrdr/mc13/t1249_mrral_20n073_0327_4.img'
REAL_GPKG = '/mnt/mrdr/crism_classification/data/vector/t1249_mrral_20n073_0327_4_mineral_map.gpkg'


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

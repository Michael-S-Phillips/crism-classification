# tests/test_spectrum.py
import numpy as np
import pytest
import geopandas as gpd
from app.spectrum import extract_polygon_spectrum, extract_roi_spectrum


def test_extract_polygon_spectrum_returns_dict(real_img_path, real_gpkg_path):
    gdf = gpd.read_file(real_gpkg_path, layer='olivine')
    poly = gdf.iloc[0].geometry
    result = extract_polygon_spectrum(real_img_path, poly)
    assert 'wavelengths' in result
    assert 'mean' in result and 'std' in result
    assert len(result['wavelengths']) == len(result['mean'])
    assert all(500 <= w <= 2600 for w in result['wavelengths'])


def test_extract_polygon_spectrum_no_nan_in_mean(real_img_path, real_gpkg_path):
    gdf = gpd.read_file(real_gpkg_path, layer='olivine')
    poly = gdf.iloc[0].geometry
    result = extract_polygon_spectrum(real_img_path, poly)
    assert not any(np.isnan(result['mean']))


def test_extract_roi_spectrum_returns_dict(real_img_path):
    result = extract_roi_spectrum(real_img_path, row=100, col=100, radius=5)
    assert 'wavelengths' in result
    assert len(result['wavelengths']) > 0


def test_extract_roi_spectrum_radius_zero(real_img_path):
    result = extract_roi_spectrum(real_img_path, row=100, col=100, radius=0)
    assert len(result['wavelengths']) > 0

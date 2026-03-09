"""
Tests for mrral spectral extraction functions.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

CFG_DATA_ROOT = '/mnt/gigas/CRISM/MRDR'
CFG_GPKG_DIR = '/mnt/gigas/CRISM/MRDR/categorized_mineral_units'


def test_find_mrral_pairs_returns_mrral_paths():
    from data.extract_pixels import find_mrral_pairs
    pairs = find_mrral_pairs(CFG_GPKG_DIR, CFG_DATA_ROOT)
    assert len(pairs) > 0
    for tile_id, gpkg_path, mrral_path in pairs:
        assert 'mrral' in mrral_path.lower(), f"Expected mrral in path: {mrral_path}"
        assert os.path.exists(mrral_path), f"File not found: {mrral_path}"


def test_mrral_records_have_59_spectral_columns():
    from data.extract_pixels import find_mrral_pairs, extract_mrral_pixels_from_pair
    pairs = find_mrral_pairs(CFG_GPKG_DIR, CFG_DATA_ROOT)
    tile_id, gpkg_path, mrral_path = pairs[0]
    records = extract_mrral_pixels_from_pair(tile_id, mrral_path, gpkg_path)
    assert len(records) > 0, "Expected at least one pixel record"
    r = records[0]
    for i in range(59):
        assert f'm{i}' in r, f'm{i} missing from record'
    for i in range(59, 72):
        assert f'm{i}' not in r, f'm{i} should not be present (wavelength > 2500 nm)'


def test_mrral_records_no_nodata():
    from data.extract_pixels import find_mrral_pairs, extract_mrral_pixels_from_pair
    pairs = find_mrral_pairs(CFG_GPKG_DIR, CFG_DATA_ROOT)
    tile_id, gpkg_path, mrral_path = pairs[0]
    records = extract_mrral_pixels_from_pair(tile_id, mrral_path, gpkg_path)
    for r in records[:500]:
        vals = [r[f'm{i}'] for i in range(59)]
        assert all(v < 65535 for v in vals), "NODATA value (65535) not filtered"


def test_mrral_records_have_label_and_metadata_columns():
    from data.extract_pixels import find_mrral_pairs, extract_mrral_pixels_from_pair
    pairs = find_mrral_pairs(CFG_GPKG_DIR, CFG_DATA_ROOT)
    tile_id, gpkg_path, mrral_path = pairs[0]
    records = extract_mrral_pixels_from_pair(tile_id, mrral_path, gpkg_path)
    r = records[0]
    for col in ['tile_id', 'polygon_id', 'pixel_row', 'pixel_col',
                'confidence_weight', 'confidence_tier',
                'olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']:
        assert col in r, f"Missing metadata column: {col}"

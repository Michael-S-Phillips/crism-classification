"""Tests for tile rendering."""
import pytest
from app.tile_renderer import render_false_color, px_to_rowcol


def test_render_false_color_returns_bytes(real_img_path):
    png_bytes, meta = render_false_color(real_img_path)
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 1000


def test_render_false_color_meta_keys(real_img_path):
    _, meta = render_false_color(real_img_path)
    for key in ('width', 'height', 'scale_x', 'scale_y', 'src_width', 'src_height'):
        assert key in meta


def test_render_false_color_max_dim(real_img_path):
    _, meta = render_false_color(real_img_path)
    assert max(meta['width'], meta['height']) <= 1024


def test_px_to_rowcol_roundtrip():
    meta = {'scale_x': 0.5, 'scale_y': 0.5}
    row, col = px_to_rowcol(img_x=100, img_y=80, meta=meta)
    assert row == 160
    assert col == 200

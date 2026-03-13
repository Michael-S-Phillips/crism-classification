"""Smoke tests for visualization figure scripts."""
import os, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFigStyle:
    def test_mineral_colors_keys(self):
        from scripts.fig_style import MINERAL_COLORS, LABEL_COLS
        assert set(MINERAL_COLORS.keys()) == {'olivine', 'lcp', 'hcp', 'plagioclase', 'other'}

    def test_label_cols(self):
        from scripts.fig_style import LABEL_COLS
        assert LABEL_COLS == ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']

    def test_dpi(self):
        from scripts.fig_style import DPI
        assert DPI == 300

    def test_apply_style_sets_font_size(self):
        import matplotlib
        import matplotlib.pyplot as plt
        from scripts.fig_style import apply_style
        original = dict(plt.rcParams)
        apply_style()
        assert plt.rcParams['font.size'] == 11
        matplotlib.rcdefaults()  # reset to avoid state leak into later tests

    def test_despine_hides_top_right(self):
        import matplotlib.pyplot as plt
        from scripts.fig_style import despine
        fig, ax = plt.subplots()
        despine(ax)
        assert not ax.spines['top'].get_visible()
        assert not ax.spines['right'].get_visible()
        plt.close(fig)


class TestModelProgression:
    def test_creates_png(self, tmp_path, monkeypatch):
        import scripts.plot_model_progression as m
        monkeypatch.setattr(m, 'REPORTS_DIR', str(tmp_path))
        m.main()
        assert (tmp_path / 'fig_model_progression.png').exists()


class TestHeatmap:
    def test_creates_png(self, tmp_path, monkeypatch):
        import scripts.plot_per_class_heatmap as m
        monkeypatch.setattr(m, 'REPORTS_DIR', str(tmp_path))
        m.main()
        assert (tmp_path / 'fig_per_class_heatmap.png').exists()


def _make_mrral_df():
    """Minimal mrral_pixels.parquet fixture — 50 rows, 59 bands."""
    rng = np.random.default_rng(42)
    n = 50
    data = {f'm{i}': rng.random(n).astype('float32') for i in range(59)}
    for cls in ['olivine', 'lcp', 'hcp', 'plagioclase', 'other']:
        # ~30% positive per class; seed 42 guarantees ≥1 positive per class with n=50
        labels = np.where(rng.random(n) > 0.7, 1.0, 0.0).astype('float32')
        assert labels.sum() >= 1, f"Fixture seed produced 0 positives for {cls}"
        data[f'label_{cls}'] = labels
    data['split'] = ['train'] * 30 + ['val'] * 10 + ['test'] * 10
    data['confidence_tier'] = ['High'] * 20 + ['Moderate'] * 20 + ['Low'] * 10
    data['tile_id']    = ['t001'] * n
    data['polygon_id'] = list(range(n))
    data['pixel_row']  = list(range(n))
    data['pixel_col']  = list(range(n))
    return pd.DataFrame(data)


class TestClassSpectra:
    def test_creates_png(self, tmp_path, monkeypatch):
        import scripts.plot_class_spectra_v2 as m
        from unittest.mock import patch

        monkeypatch.setattr(m, 'REPORTS_DIR', str(tmp_path))

        # Write fixture parquet to tmp_path
        df = _make_mrral_df()
        parquet_path = tmp_path / 'mrral_pixels.parquet'
        df.to_parquet(parquet_path)

        cfg = {'output_dir': str(tmp_path), 'data_root': str(tmp_path)}
        fixed_wavelengths = np.linspace(410, 2457, 59)

        with patch('scripts.plot_class_spectra_v2.load_config', return_value=cfg), \
             patch('scripts.plot_class_spectra_v2.get_wavelengths',
                   return_value=fixed_wavelengths):
            m.main()

        assert (tmp_path / 'fig_class_spectra_v2.png').exists()


def _make_pixels_df():
    """Minimal pixels.parquet fixture — 60 rows with 6-class labels before collapse."""
    rng = np.random.default_rng(42)
    n = 60
    data = {}
    for cls in ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']:
        data[cls] = np.where(rng.random(n) > 0.6, 1.0, 0.0).astype('float32')
    data['split']           = ['train'] * 40 + ['val'] * 10 + ['test'] * 10
    data['confidence_tier'] = ['High'] * 20 + ['Moderate'] * 20 + ['Low'] * 20
    return pd.DataFrame(data)


class TestDatasetStats:
    def test_creates_png(self, tmp_path, monkeypatch):
        import scripts.plot_dataset_stats as m
        from unittest.mock import patch

        monkeypatch.setattr(m, 'REPORTS_DIR', str(tmp_path))

        df = _make_pixels_df()
        parquet_path = tmp_path / 'pixels.parquet'
        df.to_parquet(parquet_path)

        cfg = {'output_dir': str(tmp_path)}
        with patch('scripts.plot_dataset_stats.load_config', return_value=cfg):
            m.main()

        assert (tmp_path / 'fig_dataset_stats.png').exists()

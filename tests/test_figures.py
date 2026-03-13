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

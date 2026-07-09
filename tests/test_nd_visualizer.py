"""Tests for the N-D Visualizer app (Component 3).

Builds a tiny synthetic viz parquet + endmembers csv in tmp_path, points the
app's module-level defaults at them via env vars (read at import), and drives
the app through Streamlit's AppTest harness.
"""
import importlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "label_quant", "nd_visualizer_app.py")

BAND_COLS = [f"m{i}" for i in range(2, 59)]
NB = len(BAND_COLS)  # 57


def _make_corpus(tmp_path):
    """Write a synthetic viz parquet + endmembers csv; return (viz, em) paths."""
    rng = np.random.RandomState(0)
    classes = ["olivine", "lcp", "hcp", "plagioclase", "alteration", "bland"]
    sources = ["hand", "confirmed", "reassigned", "tag"]
    rows = []
    for ci, cls in enumerate(classes):
        # each class a distinct offset so clusters are separable
        center = np.linspace(0.1, 0.4, NB) + 0.05 * ci
        for j in range(40):
            spec = center + rng.normal(0, 0.01, NB)
            rows.append({
                "class": cls,
                "source": sources[j % len(sources)],
                "tile_id": f"t{ci:02d}",
                "polygon_id": j % 5,
                "confidence_weight": [0.5, 0.75, 1.0][j % 3],
                "multi": bool(j % 7 == 0),
                **{c: v for c, v in zip(BAND_COLS, spec)},
            })
    viz = pd.DataFrame(rows)
    viz_path = tmp_path / "viz.parquet"
    viz.to_parquet(viz_path)

    em_rows = []
    for ci, cls in enumerate(["olivine", "lcp", "hcp"]):
        center = np.linspace(0.1, 0.4, NB) + 0.05 * ci
        em_rows.append({
            "class": cls, "kind": "medoid", "source": "hand",
            "tile_id": f"t{ci:02d}", "polygon_id": 0, "n_px": 99,
            **{c: v for c, v in zip(BAND_COLS, center)},
        })
    em = pd.DataFrame(em_rows)
    em_path = tmp_path / "endmembers.csv"
    em.to_csv(em_path, index=False)
    return str(viz_path), str(em_path)


@pytest.fixture
def app(tmp_path, monkeypatch):
    viz_path, em_path = _make_corpus(tmp_path)
    monkeypatch.setenv("NDVIZ_PARQUET", viz_path)
    monkeypatch.setenv("NDVIZ_ENDMEMBERS", em_path)
    monkeypatch.setenv("NDVIZ_WAVELENGTHS", str(tmp_path / "missing.json"))
    return AppTest.from_file(APP_PATH, default_timeout=60)


def test_runs_without_exception(app):
    app.run()
    assert not app.exception


def test_projection_modes(app):
    app.run()
    assert not app.exception
    for mode in ["PCA", "Raw bands", "Random projection"]:
        app.radio(key="proj_mode").set_value(mode).run()
        assert not app.exception, f"exception in projection mode {mode}"


def test_pca_component_selection(app):
    app.run()
    assert not app.exception
    # PCA is the default projection mode; choose PC2/PC3/PC4 per axis
    # (synthetic data has enough samples for >=4 components).
    app.selectbox(key="pca_x").set_value(1).run()
    app.selectbox(key="pca_y").set_value(2).run()
    app.selectbox(key="pca_z").set_value(3).run()
    assert not app.exception
    assert app.selectbox(key="pca_x").value == 1
    assert app.selectbox(key="pca_z").value == 3


def test_angle_to_endmember(app):
    app.run()
    app.radio(key="color_mode").set_value("angle to endmember").run()
    assert not app.exception
    # a chosen endmember must exist
    assert app.selectbox(key="em_color").value is not None


def test_angle_math_finite(tmp_path, monkeypatch):
    viz_path, em_path = _make_corpus(tmp_path)
    monkeypatch.setenv("NDVIZ_PARQUET", viz_path)
    monkeypatch.setenv("NDVIZ_ENDMEMBERS", em_path)
    mod = importlib.import_module("scripts.label_quant.nd_visualizer_app")
    importlib.reload(mod)
    em = pd.read_csv(em_path)
    viz = pd.read_parquet(viz_path)
    X = viz[BAND_COLS].to_numpy(dtype=float)
    v = em.iloc[0][BAND_COLS].to_numpy(dtype=float)
    angles = mod.spectral_angles_deg(X, v)
    assert angles.shape == (len(X),)
    assert np.all(np.isfinite(angles))
    assert np.all(angles >= 0) and np.all(angles <= 180)


def _load_module():
    mod = importlib.import_module("scripts.label_quant.nd_visualizer_app")
    importlib.reload(mod)
    return mod


def test_resolve_clicked_points(tmp_path, monkeypatch):
    viz_path, _ = _make_corpus(tmp_path)
    monkeypatch.setenv("NDVIZ_PARQUET", viz_path)
    mod = _load_module()
    df = pd.read_parquet(viz_path).reset_index(drop=True)
    # customdata first element is the positional row-id into df
    event = {"selection": {"points": [
        {"customdata": [5, "t02", 3, "hcp", "hand", 1.0]},
        {"customdata": [12, "t04", 1, "alteration", "tag", 0.75]},
    ]}}
    out = mod.resolve_clicked_points(event, df)
    assert list(out.index) == [5, 12]
    assert set(BAND_COLS).issubset(out.columns)


def test_resolve_clicked_points_empty_and_dedupe(tmp_path, monkeypatch):
    viz_path, _ = _make_corpus(tmp_path)
    monkeypatch.setenv("NDVIZ_PARQUET", viz_path)
    mod = _load_module()
    df = pd.read_parquet(viz_path).reset_index(drop=True)
    assert mod.resolve_clicked_points(None, df).empty
    assert mod.resolve_clicked_points({}, df).empty
    # out-of-range and duplicate row-ids are dropped
    event = {"selection": {"points": [
        {"customdata": [3]}, {"customdata": [3]},
        {"customdata": [10 ** 9]},
    ]}}
    out = mod.resolve_clicked_points(event, df)
    assert list(out.index) == [3]


def test_py_compile():
    r = subprocess.run([sys.executable, "-m", "py_compile", APP_PATH],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

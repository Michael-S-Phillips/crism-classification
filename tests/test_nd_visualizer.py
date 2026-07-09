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


def _make_full_corpus(tmp_path):
    """Write a synthetic *full* pixel corpus (labeled_spectra-style, m2..m58
    WITH real pixel_row/pixel_col) with a few multi-pixel polygons; return
    (path, specs). Each polygon's pixels get distinct in-bounds coords."""
    rng = np.random.RandomState(3)
    rows = []
    specs = [("lcp", "hand", "t01", 6, 20),
             ("hcp", "reassigned", "t02", 3, 12),
             ("olivine", "hand", "t03", 1, 8)]
    for ci, (cls, source, tile, pid, n) in enumerate(specs):
        center = np.linspace(0.1, 0.4, NB) + 0.05 * ci
        for k in range(n):
            spec = center + rng.normal(0, 0.01, NB)
            rows.append({
                "class": cls, "source": source, "tile_id": tile,
                "polygon_id": pid, "confidence_weight": 1.0, "multi": False,
                "pixel_row": np.int32(ci * 5 + k),
                "pixel_col": np.int32(ci * 3 + (k % 4)),
                **{c: v for c, v in zip(BAND_COLS, spec)},
            })
    corpus = pd.DataFrame(rows)
    path = tmp_path / "labeled_spectra.parquet"
    corpus.to_parquet(path)
    return str(path), specs


def _make_tile_fixture(tmp_path, tile_id, n_bands=72, h=64, w=64):
    """Write a tiny multi-band ENVI raster whose band1/band2 (m0/m1) are known
    deterministic functions of (row, col), and return a glob pattern that
    resolves to it (matching the app's {tile_id} substitution)."""
    import rasterio
    from rasterio.transform import from_origin
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    data = np.zeros((n_bands, h, w), dtype=np.float32)
    data[0] = (rr * 1000 + cc).astype(np.float32)          # band1 = m0 marker
    data[1] = (rr * 1000 + cc + 0.5).astype(np.float32)    # band2 = m1 marker
    tile_dir = tmp_path / "mc99"
    tile_dir.mkdir(exist_ok=True)
    img = tile_dir / f"{tile_id}_mrral_00n000_0327_4.img"
    with rasterio.open(
            img, "w", driver="ENVI", height=h, width=w, count=n_bands,
            dtype="float32", transform=from_origin(0, 0, 1, 1)) as ds:
        ds.write(data)
    # glob pattern with {tile_id} placeholder (like DEFAULT_TILE_GLOB)
    return str(tmp_path / "mc*" / "{tile_id}_mrral_*_0327_4.img")


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


def test_seven_class_synthetic(tmp_path, monkeypatch):
    """Synthetic corpus with junk + bland: all 7 classes selectable, minerals
    on / reference clouds off by default, app clean."""
    rng = np.random.RandomState(1)
    classes = ["olivine", "lcp", "hcp", "plagioclase", "alteration", "bland",
               "junk"]
    rows = []
    for ci, cls in enumerate(classes):
        center = np.linspace(0.1, 0.4, NB) + 0.05 * ci
        for j in range(30):
            spec = center + rng.normal(0, 0.01, NB)
            rows.append({
                "class": cls, "source": "hand", "tile_id": f"t{ci:02d}",
                "polygon_id": j % 4, "confidence_weight": 1.0, "multi": False,
                **{c: v for c, v in zip(BAND_COLS, spec)},
            })
    viz_path = tmp_path / "viz7.parquet"
    pd.DataFrame(rows).to_parquet(viz_path)
    monkeypatch.setenv("NDVIZ_PARQUET", str(viz_path))
    monkeypatch.setenv("NDVIZ_ENDMEMBERS", str(tmp_path / "missing.csv"))
    monkeypatch.setenv("NDVIZ_WAVELENGTHS", str(tmp_path / "missing.json"))
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception
    cf = at.multiselect(key="f_classes")
    assert set(cf.options) == set(classes), "class filter missing options"
    # minerals on, bland+junk off by default
    assert "junk" not in cf.value and "bland" not in cf.value
    assert {"olivine", "lcp", "hcp", "plagioclase", "alteration"} <= set(cf.value)


def test_projection_modes(app):
    app.run()
    assert not app.exception
    for mode in ["PCA", "Raw bands", "Random projection"]:
        app.radio(key="proj_mode").set_value(mode).run()
        assert not app.exception, f"exception in projection mode {mode}"


def test_2d_pick_mode(app):
    app.run()
    assert not app.exception
    # 3-D default renders a chart
    assert len(app.get("plotly_chart")) >= 2
    # toggle 2-D pick mode: chart still renders, app stays clean, selection key present
    app.checkbox(key="pick2d").set_value(True).run()
    assert not app.exception
    charts = app.get("plotly_chart")
    assert len(charts) >= 2
    # the selectable chart is keyed "scatter2d" (AppTest embeds key in id suffix)
    assert any(str(getattr(c, "id", "")).endswith("-scatter2d") for c in charts)


def _decode_y(y):
    if isinstance(y, dict) and "bdata" in y:
        dt = {"f8": "<f8", "f4": "<f4", "i8": "<i8"}[y["dtype"]]
        return np.frombuffer(_b64(y["bdata"]), dtype=dt)
    return np.array(y, dtype=float)


def _b64(s):
    import base64
    return base64.b64decode(s)


def test_spectra_panel_renders_traces(app):
    import json
    app.run()
    assert not app.exception
    charts = app.get("plotly_chart")
    # spectra panel is the last plotly chart
    spec = json.loads(charts[-1].spec)
    scat = [t for t in spec["data"] if t.get("type") == "scatter"]
    assert len(scat) > 0, "spectra panel rendered no line traces"
    finite = 0
    for t in scat:
        if t.get("y") is not None:
            arr = _decode_y(t["y"])
            if arr.size and np.all(np.isfinite(arr)):
                finite += 1
    assert finite > 0, "spectra panel has no finite-y traces"


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


def test_band_exclusion_pca_input_masking(tmp_path, monkeypatch):
    """The PCA/random fit matrix uses apply_band_mask: 53 good cols when the
    exclusion toggle is ON, 57 when OFF."""
    viz_path, _ = _make_corpus(tmp_path)
    monkeypatch.setenv("NDVIZ_PARQUET", viz_path)
    mod = _load_module()
    M = np.random.RandomState(0).rand(12, 57)
    mod.set_band_exclusion(True)
    assert mod.apply_band_mask(M).shape[1] == 53
    mod.set_band_exclusion(False)
    assert mod.apply_band_mask(M).shape[1] == 57
    mod.set_band_exclusion(True)  # restore module default


def test_band_exclusion_toggle_runs_clean(app):
    app.run()
    assert not app.exception
    # default ON → toggle OFF → back ON, app clean in each state
    app.checkbox(key="exclude_bands").set_value(False).run()
    assert not app.exception
    app.checkbox(key="exclude_bands").set_value(True).run()
    assert not app.exception


def test_angle_coloring_finite_both_toggle_states(app):
    app.run()
    app.radio(key="color_mode").set_value("angle to endmember").run()
    for state in (True, False):
        app.checkbox(key="exclude_bands").set_value(state).run()
        assert not app.exception
        # spectra panel angle captions computed without error implies finite;
        # also directly exercise the masked angle math in this toggle state
    # direct finite check via the module helper in both states
    mod = _load_module()
    em = pd.read_csv(os.environ["NDVIZ_ENDMEMBERS"])
    X = np.random.RandomState(1).rand(20, 57)
    v = em.iloc[0][BAND_COLS].to_numpy(dtype=float)
    for state in (True, False):
        mod.set_band_exclusion(state)
        ang = mod.spectral_angles_deg(X, v)
        assert np.all(np.isfinite(ang))
    mod.set_band_exclusion(True)


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


def test_apply_relabel_mineral_and_discard(tmp_path, monkeypatch):
    corpus_path, specs = _make_full_corpus(tmp_path)
    tile_glob = _make_tile_fixture(tmp_path, "t01")
    _make_tile_fixture(tmp_path, "t03")  # tile for the discard polygon
    relabels_dir = str(tmp_path / "ndviz_relabels")
    monkeypatch.setenv("NDVIZ_CORPUS", corpus_path)
    monkeypatch.setenv("NDVIZ_RELABELS_DIR", relabels_dir)
    monkeypatch.setenv("NDVIZ_TILE_GLOB", tile_glob)
    mod = _load_module()

    # polygon 0: lcp/hand → reassign to hcp (Moderate); polygon 2: olivine → discard
    lcp = {"class": "lcp", "source": "hand", "tile_id": "t01", "polygon_id": 6}
    oli = {"class": "olivine", "source": "hand", "tile_id": "t03", "polygon_id": 1}
    w1 = mod.apply_relabel([lcp], "hcp", "Moderate", corpus_path, relabels_dir,
                           tile_glob=tile_glob)
    w2 = mod.apply_relabel([oli], "discard", "High", corpus_path, relabels_dir,
                           tile_glob=tile_glob)
    assert len(w1) == 1 and w1[0]["n_px"] == 20 and w1[0]["m01_ok"] is True
    assert len(w2) == 1 and w2[0]["n_px"] == 8 and w2[0]["m01_ok"] is True

    hn = pd.read_parquet(os.path.join(relabels_dir, "hard_negatives"))
    assert {"m0", "m1", "m58"}.issubset(hn.columns)

    # real coords preserved from the corpus (lcp polygon: rows 0..19, cols k%4)
    reassigned = hn[hn["negative_of"] == ""].sort_values("pixel_row")
    discarded = hn[hn["negative_of"] == "olivine"]
    assert len(reassigned) == 20 and len(discarded) == 8
    assert list(reassigned["pixel_row"]) == list(range(20))
    assert list(reassigned["pixel_col"]) == [k % 4 for k in range(20)]
    # m0/m1 read from the fixture raster: band1 = row*1000+col, band2 = +0.5
    exp_m0 = reassigned["pixel_row"].to_numpy() * 1000 + reassigned["pixel_col"].to_numpy()
    assert np.allclose(reassigned["m0"], exp_m0)
    assert np.allclose(reassigned["m1"], exp_m0 + 0.5)
    # m2..m58 still sourced from corpus (non-zero)
    assert (reassigned["m2"] != 0).all()

    # mineral reassignment: hcp positive, Moderate weight (0.75)
    assert (reassigned["hcp"] == 1.0).all()
    assert (reassigned["lcp"] == 0.0).all()
    assert np.allclose(reassigned["confidence_weight"], 0.75)
    assert (reassigned["confidence_tier"] == "Reviewed-Moderate").all()
    label_cols = ["olivine_t1", "olivine_t2", "lcp", "hcp", "plagioclase",
                  "other", "alteration"]
    assert (discarded[label_cols].to_numpy() == 0.0).all()
    assert np.allclose(discarded["confidence_weight"], 1.0)

    # decisions.csv: two rows with correct provenance, clean layer (no :nom01)
    dec = pd.read_csv(os.path.join(relabels_dir, "decisions.csv"))
    assert len(dec) == 2
    assert set(dec["source_gpkg"]) == {"ndviz"}
    assert set(dec["predicted_class"]) == {"lcp", "olivine"}
    assert dec.set_index("predicted_class").loc["lcp", "corrected_class"] == "hcp"
    assert (dec["area_m2"] == 0).all()
    assert not dec["layer"].astype(str).str.contains(":nom01").any()
    assert mod.relabel_session_count(relabels_dir) == 2


def test_apply_relabel_missing_tile_fallback(tmp_path, monkeypatch):
    corpus_path, _ = _make_full_corpus(tmp_path)
    # glob points at a dir with no matching tile → m0/m1 fall back to 0.0
    empty_glob = str(tmp_path / "no_tiles" / "{tile_id}_mrral_*_0327_4.img")
    relabels_dir = str(tmp_path / "ndviz_relabels")
    monkeypatch.setenv("NDVIZ_CORPUS", corpus_path)
    monkeypatch.setenv("NDVIZ_RELABELS_DIR", relabels_dir)
    mod = _load_module()
    warnings = []
    poly = {"class": "lcp", "source": "hand", "tile_id": "t01", "polygon_id": 6}
    w = mod.apply_relabel([poly], "hcp", "High", corpus_path, relabels_dir,
                          tile_glob=empty_glob, warn=warnings.append)
    assert w[0]["m01_ok"] is False
    assert warnings, "expected a warning on missing tile"
    hn = pd.read_parquet(os.path.join(relabels_dir, "hard_negatives"))
    assert (hn["m0"] == 0.0).all() and (hn["m1"] == 0.0).all()
    # real coords still preserved even in fallback
    assert list(hn.sort_values("pixel_row")["pixel_row"]) == list(range(20))
    # degraded rows flagged via ':nom01' in the decision log layer
    dec = pd.read_csv(os.path.join(relabels_dir, "decisions.csv"))
    assert dec["layer"].astype(str).str.endswith(":nom01").all()


def test_apply_relabel_tag_semantics(tmp_path, monkeypatch):
    corpus_path, _ = _make_full_corpus(tmp_path)
    relabels_dir = str(tmp_path / "ndviz_relabels")
    monkeypatch.setenv("NDVIZ_CORPUS", corpus_path)
    monkeypatch.setenv("NDVIZ_RELABELS_DIR", relabels_dir)
    mod = _load_module()
    poly = {"class": "hcp", "source": "reassigned", "tile_id": "t02",
            "polygon_id": 3}
    mod.apply_relabel([poly], "alteration", "Low", corpus_path, relabels_dir)
    hn = pd.read_parquet(os.path.join(relabels_dir, "hard_negatives"))
    # alteration is a TAG: negative_of='alteration', no positive labels
    assert (hn["negative_of"] == "alteration").all()
    label_cols = ["olivine_t1", "olivine_t2", "lcp", "hcp", "plagioclase",
                  "other", "alteration"]
    assert (hn[label_cols].to_numpy() == 0.0).all()
    assert np.allclose(hn["confidence_weight"], 0.5)  # Low


def test_relabel_panel_renders(tmp_path, monkeypatch):
    viz_path, em_path = _make_corpus(tmp_path)
    corpus_path, _ = _make_full_corpus(tmp_path)
    relabels_dir = str(tmp_path / "ndviz_relabels")
    monkeypatch.setenv("NDVIZ_PARQUET", viz_path)
    monkeypatch.setenv("NDVIZ_ENDMEMBERS", em_path)
    monkeypatch.setenv("NDVIZ_WAVELENGTHS", str(tmp_path / "missing.json"))
    monkeypatch.setenv("NDVIZ_CORPUS", corpus_path)
    monkeypatch.setenv("NDVIZ_RELABELS_DIR", relabels_dir)
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    # seed a clicked pixel identifying a real corpus polygon, enable 2-D mode
    at.session_state["clicked"] = [{
        "class": "lcp", "source": "hand", "tile_id": "t01", "polygon_id": 6,
        "confidence_weight": 1.0,
        **{c: 0.2 for c in BAND_COLS},
    }]
    at.checkbox(key="pick2d").set_value(True).run()
    assert not at.exception
    # the relabel selectbox must be present
    assert at.selectbox(key="relabel_as").value in (
        "olivine", "lcp", "hcp", "plagioclase", "alteration", "bland",
        "ambiguous", "discard")


def test_py_compile():
    r = subprocess.run([sys.executable, "-m", "py_compile", APP_PATH],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

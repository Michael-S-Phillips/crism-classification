"""Tests for the SAM endmember analysis (Component 2).

Strict TDD: written before the implementation. Synthetic corpora are built in
memory as DataFrames matching the labeled-spectra schema (class, source,
tile_id, polygon_id, confidence_weight, multi, m2..m58) and exercised through
the public functions of ``sam_endmembers``.
"""
import numpy as np
import pandas as pd
import pytest

from scripts.label_quant.sam_endmembers import (
    analyze,
    angle_between,
    spectral_angle_matrix,
    BAND_COLS,
)

RNG = np.random.default_rng(1234)
NB = len(BAND_COLS)  # 57


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _rows(cls, source, tile_id, polygon_id, spectra, multi=False, weight=1.0):
    """Build a list of pixel-row dicts for one polygon.

    ``spectra`` is an (n_px, NB) array; each row becomes one pixel row.
    """
    out = []
    for px in spectra:
        rec = {
            "class": cls,
            "source": source,
            "tile_id": tile_id,
            "polygon_id": polygon_id,
            "confidence_weight": weight,
            "multi": multi,
        }
        for b, col in enumerate(BAND_COLS):
            rec[col] = float(px[b])
        out.append(rec)
    return out


def _cluster(direction, n_poly, n_px, noise, rng):
    """Yield n_poly polygons of n_px pixels scattered tightly around
    ``direction`` (a unit vector in NB-space)."""
    direction = direction / np.linalg.norm(direction)
    polys = []
    for _ in range(n_poly):
        base = direction + noise * rng.standard_normal(NB)
        px = base[None, :] + 0.3 * noise * rng.standard_normal((n_px, NB))
        # NB: no DC offset — a large constant across all bands would collapse
        # every planted direction toward the all-ones vector and shrink angles.
        polys.append(px)
    return polys


def _frame(row_lists):
    return pd.DataFrame([r for rl in row_lists for r in rl])


# --------------------------------------------------------------------------- #
# 1. Angle correctness
# --------------------------------------------------------------------------- #
def test_angle_identical_is_zero():
    a = np.array([1.0, 2.0, 3.0])
    assert angle_between(a, a) == pytest.approx(0.0, abs=1e-9)


def test_angle_orthogonal_is_ninety():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert np.degrees(angle_between(a, b)) == pytest.approx(90.0, abs=1e-9)


def test_angle_scaled_copy_is_zero():
    a = np.array([1.0, 2.0, 3.0])
    b = 2.0 * a
    assert angle_between(a, b) == pytest.approx(0.0, abs=1e-9)


def test_angle_matrix_matches_pairwise():
    X = RNG.standard_normal((6, NB)) + 5.0
    M = spectral_angle_matrix(X)
    assert M.shape == (6, 6)
    # diagonal ~0
    assert np.allclose(np.diag(M), 0.0, atol=1e-6)
    # symmetric
    assert np.allclose(M, M.T, atol=1e-9)
    # matches angle_between for a couple of pairs
    assert M[0, 3] == pytest.approx(angle_between(X[0], X[3]), abs=1e-9)


# --------------------------------------------------------------------------- #
# 2. Medoid recovery on synthetic clusters
# --------------------------------------------------------------------------- #
def test_medoid_recovery_and_matrix():
    rng = np.random.default_rng(7)
    # three distinct directions
    d = {}
    d["olivine"] = np.zeros(NB); d["olivine"][0] = 1.0
    d["lcp"] = np.zeros(NB); d["lcp"][1] = 1.0
    v = np.zeros(NB); v[2] = 1.0; v[3] = 1.0
    d["hcp"] = v
    planted = {}
    for (c1, v1) in d.items():
        for (c2, v2) in d.items():
            planted[(c1, c2)] = np.degrees(angle_between(v1, v2))

    rls = []
    for cls, direction in d.items():
        for i, polypx in enumerate(_cluster(direction, 8, 40, 0.005, rng)):
            rls.append(_rows(cls, "hand", "T1", f"{cls}_{i}", polypx))
    res = analyze(_frame(rls), min_px=10)

    # medoid of each class lands in its own cluster (closest to its direction)
    for cls, direction in d.items():
        med = res["medoids"][cls]
        own = np.degrees(angle_between(med, direction))
        for other, odir in d.items():
            if other == cls:
                continue
            assert own < np.degrees(angle_between(med, odir))
        assert own < 3.0

    # inter-class matrix approximates planted angles (deg) within 2 deg
    mat = res["angle_matrix"]
    for c1 in d:
        for c2 in d:
            assert mat.loc[c1, c2] == pytest.approx(planted[(c1, c2)], abs=2.0)


# --------------------------------------------------------------------------- #
# 3. Planted mislabel -> negative margin, suspect
# --------------------------------------------------------------------------- #
def test_planted_mislabel_is_suspect():
    rng = np.random.default_rng(11)
    dl = np.zeros(NB); dl[0] = 1.0     # lcp direction
    dh = np.zeros(NB); dh[1] = 1.0     # hcp direction
    rls = []
    for i, px in enumerate(_cluster(dl, 8, 40, 0.02, rng)):
        rls.append(_rows("lcp", "hand", "T1", f"lcp_{i}", px))
    for i, px in enumerate(_cluster(dh, 8, 40, 0.02, rng)):
        rls.append(_rows("hcp", "hand", "T1", f"hcp_{i}", px))
    # one lcp-LABELED polygon whose spectrum sits in the hcp cluster
    bad = _cluster(dh, 1, 40, 0.02, rng)[0]
    rls.append(_rows("lcp", "hand", "T1", "lcp_BAD", bad))

    res = analyze(_frame(rls), min_px=10)
    pur = res["purity"]
    row = pur[(pur["class"] == "lcp") & (pur["polygon_id"] == "lcp_BAD")]
    assert len(row) == 1
    row = row.iloc[0]
    assert row["margin_deg"] < 0
    assert bool(row["suspect"]) is True
    assert row["nearest_other_class"] == "hcp"


# --------------------------------------------------------------------------- #
# 4. Multi exclusion + min_px
# --------------------------------------------------------------------------- #
def test_multi_and_minpx_exclusion():
    rng = np.random.default_rng(21)
    dl = np.zeros(NB); dl[0] = 1.0
    dh = np.zeros(NB); dh[1] = 1.0
    rls = []
    for i, px in enumerate(_cluster(dl, 6, 40, 0.02, rng)):
        rls.append(_rows("lcp", "hand", "T1", f"lcp_{i}", px))
    for i, px in enumerate(_cluster(dh, 6, 40, 0.02, rng)):
        rls.append(_rows("hcp", "hand", "T1", f"hcp_{i}", px))
    # a multi-label polygon (should be excluded from ALL class-level analysis)
    mpx = _cluster(dl, 1, 40, 0.02, rng)[0]
    rls.append(_rows("lcp", "hand", "T1", "lcp_MULTI", mpx, multi=True))
    # a 5-px degenerate lcp polygon (excluded from medoid math, kept in purity)
    spx = _cluster(dl, 1, 5, 0.02, rng)[0]
    rls.append(_rows("lcp", "hand", "T1", "lcp_SMALL", spx))

    res = analyze(_frame(rls), min_px=10)

    # endmember (medoid/candidate/discriminative) rows never reference the
    # degenerate or multi polygons
    em = res["endmembers"]
    refd = set(em["polygon_id"])
    assert "lcp_SMALL" not in refd
    assert "lcp_MULTI" not in refd

    pur = res["purity"]
    ids = set(pur["polygon_id"])
    # degenerate polygon IS in the purity report, flagged
    assert "lcp_SMALL" in ids
    small = pur[pur["polygon_id"] == "lcp_SMALL"].iloc[0]
    assert small["n_px"] == 5
    assert bool(small["degenerate"]) is True
    # multi polygon excluded everywhere
    assert "lcp_MULTI" not in ids


# --------------------------------------------------------------------------- #
# 5. Cross-source coherence
# --------------------------------------------------------------------------- #
def test_cross_source_coherence():
    rng = np.random.default_rng(31)
    da = np.zeros(NB); da[0] = 1.0            # hand direction
    db = np.zeros(NB); db[0] = 1.0; db[5] = 1.0  # confirmed direction (offset)
    planted = np.degrees(angle_between(da, db))  # ~45 deg
    rls = []
    for i, px in enumerate(_cluster(da, 6, 40, 0.02, rng)):
        rls.append(_rows("lcp", "hand", "T1", f"h_{i}", px))
    for i, px in enumerate(_cluster(db, 6, 40, 0.02, rng)):
        rls.append(_rows("lcp", "confirmed", "T2", f"c_{i}", px))
    # a second class so corpus has >1 class
    dh = np.zeros(NB); dh[1] = 1.0
    for i, px in enumerate(_cluster(dh, 6, 40, 0.02, rng)):
        rls.append(_rows("hcp", "hand", "T1", f"hcp_{i}", px))

    res = analyze(_frame(rls), min_px=10)
    cs = res["cross_source"]
    row = cs[(cs["class"] == "lcp")
             & (((cs["source_a"] == "hand") & (cs["source_b"] == "confirmed"))
                | ((cs["source_a"] == "confirmed") & (cs["source_b"] == "hand")))]
    assert len(row) == 1
    assert row.iloc[0]["angle_deg"] == pytest.approx(planted, abs=3.0)
    assert row.iloc[0]["angle_deg"] > 20.0


# --------------------------------------------------------------------------- #
# 6. Dynamic class discovery (classes derived from the corpus, not hardcoded)
# --------------------------------------------------------------------------- #
def test_dynamic_class_discovery():
    rng = np.random.default_rng(41)
    # two canonical mineral classes plus two novel diagnostic classes not in
    # any hardcoded 5-class list (bland + a totally new label)
    dirs = {}
    for k, band in [("lcp", 0), ("hcp", 1), ("bland", 2), ("weirdclass", 3)]:
        v = np.zeros(NB); v[band] = 1.0
        dirs[k] = v
    rls = []
    for cls, direction in dirs.items():
        for i, px in enumerate(_cluster(direction, 6, 40, 0.01, rng)):
            rls.append(_rows(cls, "hand", "T1", f"{cls}_{i}", px))
    res = analyze(_frame(rls), min_px=10)

    # every present class appears in medoids, angle matrix, purity, spread
    for cls in dirs:
        assert cls in res["medoids"]
        assert cls in res["angle_matrix"].index
        assert cls in res["angle_matrix"].columns
        assert cls in set(res["purity"]["class"])
        assert cls in set(res["intra_spread"]["class"])
    # matrix is square over all discovered classes
    assert res["angle_matrix"].shape == (len(dirs), len(dirs))


# --------------------------------------------------------------------------- #
# 7. Canonical ordering of the 8 corpus classes (minerals, then blands, junk)
# --------------------------------------------------------------------------- #
def test_canonical_class_ordering():
    rng = np.random.default_rng(51)
    # present in a scrambled input order; expect canonical order in the matrix
    order_in = ["junk", "bland_reject", "lcp", "bland_dust", "olivine",
                "hcp", "plagioclase", "alteration"]
    dirs = {c: np.zeros(NB) for c in order_in}
    for i, c in enumerate(order_in):
        dirs[c][i] = 1.0
    rls = []
    for cls in order_in:  # feed in scrambled order
        for j, px in enumerate(_cluster(dirs[cls], 6, 40, 0.01, rng)):
            rls.append(_rows(cls, "hand", "T1", f"{cls}_{j}", px))
    res = analyze(_frame(rls), min_px=10)
    expected = ["olivine", "lcp", "hcp", "plagioclase", "alteration",
                "bland_dust", "bland_reject", "junk"]
    assert list(res["angle_matrix"].index) == expected

"""The bland gate: require p_bland < gate before a pixel counts as any mineral.

Why this exists. On t1321 the e87 dual-CR model fires lcp >= 0.99 on 125,757 px,
and 35% of those have LCPINDEX2 ~ 0 -- bright red dust, not pyroxene (RBR 6.0 vs
3.8, R770 0.26 vs 0.16 against the pixels that do show the band). The model's own
bland channel separates the two populations at AUC 0.93, but at absolute values
(median 0.087 vs 0.0067) far too small to cross any threshold, because
multi-label sigmoids never force p_lcp down when p_bland rises. The gate reads
that latent signal out.

These tests exercise the real function the vectorizer calls. They do not
reimplement the comparison locally -- an earlier test in this project did that
for a different fix and passed with the implementation deleted.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.vectorize_per_mineral_thresholds_nili_6cls import (
    GATE_CHANNEL_CANDIDATES,
    bland_gate_mask,
)


def _probs(bland_col, channels):
    """(1, N, C) prob cube whose bland channel is `bland_col`."""
    n = len(bland_col)
    p = np.zeros((1, n, len(channels)), dtype=np.float32)
    p[0, :, channels.index('bland' if 'bland' in channels else 'other')] = bland_col
    return p


CH7 = ['olivine', 'lcp', 'hcp', 'plagioclase', 'bland', 'alteration', 'junk']
CH5 = ['olivine', 'lcp', 'hcp', 'plagioclase', 'other', 'alteration']


def test_gate_of_none_is_the_identity():
    """Gating off must return the caller's mask untouched, not a copy-with-changes.
    Every existing product was built with no gate; that path must not shift."""
    valid = np.array([[True, True, False, True]])
    p = _probs([0.9, 0.0, 0.5, 0.02], CH7)
    out = bland_gate_mask(valid, p, CH7, None)
    assert np.array_equal(out, valid)


def test_pixels_at_or_above_the_gate_are_dropped():
    valid = np.ones((1, 5), dtype=bool)
    p = _probs([0.001, 0.029, 0.030, 0.031, 0.500], CH7)
    out = bland_gate_mask(valid, p, CH7, 0.03)
    # strict <: 0.030 is NOT kept
    assert out.tolist() == [[True, True, False, False, False]]


def test_gate_only_removes_never_adds():
    """A pixel already invalid must stay invalid even with p_bland = 0."""
    valid = np.array([[False, False, True]])
    p = _probs([0.0, 0.0, 0.0], CH7)
    out = bland_gate_mask(valid, p, CH7, 0.03)
    assert out.tolist() == [[False, False, True]]


def test_five_class_vocab_gates_on_other():
    """The 5-class vocab calls the same class 'other'. Silently skipping the gate
    there would produce an ungated product that claims to be gated."""
    valid = np.ones((1, 3), dtype=bool)
    p = _probs([0.01, 0.20, 0.02], CH5)
    out = bland_gate_mask(valid, p, CH5, 0.03)
    assert out.tolist() == [[True, False, True]]


def test_missing_bland_channel_raises_rather_than_silently_passing():
    """If no gate channel exists, returning valid_mask unchanged would mean the
    run reports a gate it did not apply. Fail loudly instead."""
    ch = ['olivine', 'lcp', 'hcp']
    valid = np.ones((1, 2), dtype=bool)
    p = np.zeros((1, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match='bland'):
        bland_gate_mask(valid, p, ch, 0.03)


def test_gate_channel_candidates_are_the_two_known_names():
    assert GATE_CHANNEL_CANDIDATES == ('bland', 'other')


def test_measured_lcp_case_from_t1321():
    """The real numbers: a dusty false-lcp pixel (p_bland 0.087, the measured
    median) is dropped; a true-lcp pixel (0.0067) is kept. If the gate were
    inverted or the comparison non-strict, this pair would not separate."""
    valid = np.ones((1, 2), dtype=bool)
    p = _probs([0.087, 0.0067], CH7)
    out = bland_gate_mask(valid, p, CH7, 0.03)
    assert out.tolist() == [[False, True]]

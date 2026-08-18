"""The gate makes mineral and non-mineral mutually exclusive by construction."""
from __future__ import annotations

import pytest
import torch

from models.gated_classifier import (
    GatedSpatialSpectralClassifierAux, class_partition, compose_gated_probs,
)

CLASSES = ['olivine', 'lcp', 'hcp', 'plagioclase', 'bland', 'alteration', 'junk']
MIN_IDX, NON_IDX = [0, 1, 2, 3, 5], [4, 6]


def test_partition_matches_the_spec():
    m, n = class_partition(CLASSES)
    assert m == MIN_IDX
    assert n == NON_IDX


def test_partition_rejects_an_unknown_vocabulary():
    """A silent mis-partition would gate the wrong classes."""
    with pytest.raises(ValueError):
        class_partition(['olivine', 'pyx', 'plagioclase'])


def test_the_exclusivity_constraint_holds_for_random_logits():
    """The whole point: max(p_mineral) + max(p_non-mineral) <= 1. The current
    flat head violates this on 35.4% of t1321's valid pixels, peaking at 1.935."""
    torch.manual_seed(0)
    logits = torch.randn(512, 8) * 5
    probs, _ = compose_gated_probs(logits, MIN_IDX, NON_IDX)
    s = probs[:, MIN_IDX].max(1).values + probs[:, NON_IDX].max(1).values
    assert s.max().item() <= 1.0 + 1e-5


def test_all_probabilities_are_in_range():
    torch.manual_seed(1)
    probs, gate = compose_gated_probs(torch.randn(256, 8) * 8, MIN_IDX, NON_IDX)
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    assert gate.min() >= 0.0 and gate.max() <= 1.0


def test_a_closed_gate_zeroes_every_mineral():
    logits = torch.zeros(1, 8)
    logits[0, 0] = -30.0            # gate shut
    probs, gate = compose_gated_probs(logits, MIN_IDX, NON_IDX)
    assert gate.item() == pytest.approx(0.0, abs=1e-6)
    assert probs[0, MIN_IDX].max().item() == pytest.approx(0.0, abs=1e-6)


def test_an_open_gate_zeroes_bland_and_junk():
    logits = torch.zeros(1, 8)
    logits[0, 0] = 30.0             # gate open
    probs, gate = compose_gated_probs(logits, MIN_IDX, NON_IDX)
    assert gate.item() == pytest.approx(1.0, abs=1e-6)
    assert probs[0, NON_IDX].max().item() == pytest.approx(0.0, abs=1e-6)


def test_co_occurrence_survives_the_gate():
    """olivine+hcp is a real assemblage and 18.8% of training rows carry 2+
    minerals. If the gate forced competition among minerals this would fail."""
    logits = torch.zeros(1, 8)
    logits[0, 0] = 10.0                          # gate open
    logits[0, 1 + CLASSES.index('olivine')] = 10.0
    logits[0, 1 + CLASSES.index('hcp')] = 10.0
    probs, _ = compose_gated_probs(logits, MIN_IDX, NON_IDX)
    assert probs[0, CLASSES.index('olivine')].item() > 0.99
    assert probs[0, CLASSES.index('hcp')].item() > 0.99


def test_composition_equals_the_plain_product():
    """Log-space maths must agree with the naive product where the naive one is
    still accurate -- otherwise a bug hides behind 'numerical stability'."""
    torch.manual_seed(2)
    logits = torch.randn(64, 8)
    probs, gate = compose_gated_probs(logits, MIN_IDX, NON_IDX)
    c = torch.sigmoid(logits[:, 1:])
    g = torch.sigmoid(logits[:, 0:1])
    assert torch.allclose(probs[:, MIN_IDX], (g * c)[:, MIN_IDX], atol=1e-6)
    assert torch.allclose(probs[:, NON_IDX], ((1 - g) * c)[:, NON_IDX], atol=1e-6)


def test_model_emits_eight_logits():
    m = GatedSpatialSpectralClassifierAux(
        n_bands=118, patch_size=7, embed_dim=32, n_heads=2, n_layers=1,
        aux_dim=1, aux_hidden=16)
    out = m(torch.randn(4, 7, 7, 118), torch.randn(4, 1))
    assert out.shape == (4, 8)


def test_model_head_is_one_wider_than_the_class_count():
    m = GatedSpatialSpectralClassifierAux(
        n_bands=118, patch_size=7, embed_dim=32, n_heads=2, n_layers=1,
        aux_dim=1, aux_hidden=16)
    assert m.head.out_features == 8

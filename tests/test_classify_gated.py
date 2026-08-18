"""An 8-wide head must be read as 7 gated classes, never as 8 classes."""
from __future__ import annotations

import pytest
import torch

import scripts.classify_tile_supervised as cts


def test_eight_wide_head_is_rejected_when_not_gated():
    """Loud failure is correct: silently inventing a class shifts every
    downstream index."""
    cts.GATED_MODE = False
    with pytest.raises(ValueError, match='unsupported head size'):
        cts._set_n_classes({'head.weight': torch.zeros(8, 272)})


def test_eight_wide_head_is_seven_classes_when_gated():
    cts.GATED_MODE = True
    try:
        cts._set_n_classes({'head.weight': torch.zeros(8, 272)})
        assert cts.N_CLASSES == 7
        assert list(cts.CLASS_NAMES) == ['olivine', 'lcp', 'hcp', 'plagioclase',
                                         'bland', 'alteration', 'junk']
    finally:
        cts.GATED_MODE = False


def test_gated_probs_written_are_composed_not_raw_conditionals():
    """Raw conditionals look like probabilities, sum plausibly, and are wrong."""
    from models.gated_classifier import class_partition, compose_gated_probs
    names = ['olivine', 'lcp', 'hcp', 'plagioclase', 'bland', 'alteration', 'junk']
    m, n = class_partition(names)
    logits = torch.zeros(1, 8)
    logits[0, 0] = -8.0                      # gate mostly shut
    logits[0, 1 + names.index('lcp')] = 8.0  # conditional very confident
    probs, _ = compose_gated_probs(logits, m, n)
    assert probs[0, names.index('lcp')].item() < 0.01, \
        'a shut gate must suppress a confident conditional'

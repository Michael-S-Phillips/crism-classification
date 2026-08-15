"""Threshold-token formatting: legacy stability + high-ladder distinctness.

`polygon_uid` is `{tile_id}::{thresh_token}::{index}` and keys decisions.csv,
so the token must (a) never change for the thresholds already recorded in
historical decisions, and (b) be injective over the ladder actually in use.
"""
import pytest

from scripts.review.polygon_queue import _canonical_layer
from scripts.threshold_names import fmt_threshold
from scripts.vectorize_per_mineral_thresholds_nili_6cls import _fmt_thresh

# The rungs that appear in the shipped decisions.csv files (data/mc13_review*,
# vector_mc13_7cls_v3_lrscale001). Their tokens are frozen forever.
LEGACY_GRID = [0.50, 0.60, 0.75, 0.85, 0.90, 0.95, 0.97, 0.99]

# The 8-rung high ladder used by the mc_deploy_pyx vectorization.
HIGH_LADDER = [0.50, 0.85, 0.97, 0.99, 0.995, 0.999, 0.9995, 0.9999]


def test_legacy_tokens_are_byte_identical():
    """Every historically-recorded threshold still renders exactly as before.

    Hard-coded expected strings, NOT `f'{t:.2f}'` — a test that recomputes the
    old formula would pass no matter what the formatter does.
    """
    expected = {
        0.50: 'thresh_0.50', 0.60: 'thresh_0.60', 0.75: 'thresh_0.75',
        0.85: 'thresh_0.85', 0.90: 'thresh_0.90', 0.95: 'thresh_0.95',
        0.97: 'thresh_0.97', 0.99: 'thresh_0.99',
    }
    for t in LEGACY_GRID:
        assert _canonical_layer(t) == expected[t]
    # The four rungs the task calls out explicitly.
    assert _canonical_layer(0.50) == 'thresh_0.50'
    assert _canonical_layer(0.85) == 'thresh_0.85'
    assert _canonical_layer(0.97) == 'thresh_0.97'
    assert _canonical_layer(0.99) == 'thresh_0.99'


def test_high_ladder_tokens_are_all_distinct():
    """All eight rungs of the deployment ladder map to distinct uid tokens.

    Under the old `f'{prob:.2f}'`: 0.99/0.995 -> 'thresh_0.99' and
    0.999/0.9995/0.9999 -> 'thresh_1.00', i.e. 8 rungs -> 5 tokens.
    """
    tokens = [_canonical_layer(t) for t in HIGH_LADDER]
    assert len(set(tokens)) == len(HIGH_LADDER), (
        f'collision in {dict(zip(HIGH_LADDER, tokens))}')
    assert tokens == [
        'thresh_0.50', 'thresh_0.85', 'thresh_0.97', 'thresh_0.99',
        'thresh_0.995', 'thresh_0.999', 'thresh_0.9995', 'thresh_0.9999',
    ]


def test_tokens_round_trip_to_the_original_float():
    for t in HIGH_LADDER + LEGACY_GRID:
        token = _canonical_layer(t)
        assert float(token.removeprefix('thresh_')) == pytest.approx(t, abs=1e-12)


def test_vectorizer_and_queue_agree():
    """The physical layer name and the uid token come from one implementation.

    If these ever diverge, gpkg layers and decisions.csv references drift apart
    silently.
    """
    assert _fmt_thresh is fmt_threshold
    for t in HIGH_LADDER:
        assert _canonical_layer(t) == f'thresh_{_fmt_thresh(t)}'

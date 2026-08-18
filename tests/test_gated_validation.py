"""The validation metric must compose the gate, exactly as the loss does.

C2: training/train_torch.py appended ``torch.sigmoid(logits)`` unconditionally.
For a gated model that is an (N, 8) array of RAW conditionals with the GATE
sitting in column 0, scored against (N, 7) labels. It never crashes --
compute_map loops ``range(y_true.shape[1])`` = 7 and indexes into the 8-wide
array, and evaluation.metrics._class_names(8) falls through and returns 7
names -- so the gate is silently compared against the olivine label, the
olivine conditional against lcp, and so on down the row. val_mAP,
val_mAP_core, best-checkpoint selection and early stopping all run on that
misaligned array for the full 24-hour job.

These tests drive the REAL train_torch_model validation loop and capture the
array that actually reaches compute_full_metrics, rather than re-deriving the
composition in the test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

import data.dataset
import training.train_torch as tt
from models.gated_classifier import class_partition, compose_gated_probs

LABELS_7 = ['olivine', 'lcp', 'hcp', 'plagioclase', 'bland', 'alteration', 'junk']
N_FEATURES = 60


class _LinearHead(torch.nn.Module):
    """Minimal stand-in for the real encoder: the validation composition is
    model-agnostic, so a Linear that emits n_out logits exercises the same
    path as GatedSpatialSpectralClassifierAux without needing a patch cache."""

    def __init__(self, n_out):
        super().__init__()
        self.head = torch.nn.Linear(N_FEATURES, n_out)

    def forward(self, x):
        return self.head(x)


def _fake_df(n=180, seed=0):
    rng = np.random.default_rng(seed)
    data = {f'b{i}': rng.random(n).astype(np.float32) for i in range(N_FEATURES)}
    for col in ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase',
                'other', 'alteration', 'bland', 'junk']:
        data[col] = (rng.random(n) > 0.6).astype(np.float32)
    data['confidence_weight'] = np.ones(n, dtype=np.float32)
    data['confidence_tier'] = ['High'] * n
    data['tile_id'] = 't0001'
    data['polygon_id'] = 0
    data['pixel_row'] = 0
    data['pixel_col'] = 0
    n_train, n_val = int(n * 0.6), int(n * 0.2)
    data['split'] = (['train'] * n_train + ['val'] * n_val
                     + ['test'] * (n - n_train - n_val))
    return pd.DataFrame(data)


@pytest.fixture
def seven_class_labels(monkeypatch):
    """scripts/train.py rebinds data.dataset.LABEL_COLS for 7-class mode
    before any dataset code reads it; mirror that here."""
    monkeypatch.setattr(data.dataset, 'LABEL_COLS', list(LABELS_7))
    return LABELS_7


@pytest.fixture
def captured_val_scores(monkeypatch):
    """Capture the y_score array train_torch_model hands to the metric --
    the array that drives val_mAP, checkpoint selection and early stopping."""
    seen = {}
    real = tt.compute_full_metrics

    def _spy(y_true, y_score, conf_tiers, *a, **kw):
        seen['y_true'] = y_true
        seen['y_score'] = y_score
        return real(y_true, y_score, conf_tiers, *a, **kw)

    monkeypatch.setattr(tt, 'compute_full_metrics', _spy)
    return seen


def _run(model, df, **kw):
    return tt.train_torch_model(
        model=model, df=df, model_name='gated_val_test', max_epochs=1,
        batch_size=32, lr=1e-3, use_wandb=False, checkpoint_dir=None,
        device='cpu', **kw)


def test_gated_validation_scores_are_composed_and_seven_wide(
        seven_class_labels, captured_val_scores):
    """The array scored against the (N, 7) labels must itself be (N, 7)
    composed probabilities -- not the (N, 8) raw conditionals."""
    torch.manual_seed(0)
    df = _fake_df()
    _run(_LinearHead(8), df, use_asl_loss=True, gated_head=True)

    y_score = captured_val_scores['y_score']
    y_true = captured_val_scores['y_true']
    assert y_score.shape[1] == len(LABELS_7), (
        f'validation scored a {y_score.shape[1]}-wide array against '
        f'{y_true.shape[1]} labels -- the gate logit is being compared to a '
        f'mineral label and every column after it is shifted')
    assert y_score.shape == y_true.shape


def test_gated_validation_obeys_the_exclusivity_constraint(
        seven_class_labels, captured_val_scores):
    """The composed array must carry the gate's guarantee. Raw conditionals
    are 7 free sigmoids and routinely break it -- this is the property the
    whole gate exists to enforce, checked on the metric's own input."""
    torch.manual_seed(1)
    _run(_LinearHead(8), _fake_df(seed=1), use_asl_loss=True, gated_head=True)

    y_score = captured_val_scores['y_score']
    mineral_idx, non_mineral_idx = class_partition(LABELS_7)
    s = (y_score[:, mineral_idx].max(axis=1)
         + y_score[:, non_mineral_idx].max(axis=1))
    assert s.max() <= 1.0 + 1e-5


def test_gated_validation_matches_compose_gated_probs_exactly(
        seven_class_labels, captured_val_scores):
    """One implementation, three call sites: the validation loop must use the
    same compose_gated_probs as the loss and inference, not a re-derivation."""
    torch.manual_seed(2)
    df = _fake_df(seed=2)
    model = _LinearHead(8)
    _run(model, df, use_asl_loss=True, gated_head=True)

    y_score = captured_val_scores['y_score']
    mineral_idx, non_mineral_idx = class_partition(LABELS_7)

    from data.dataset import CRISMPixelDataset
    val_ds = CRISMPixelDataset(df[df['split'] == 'val'])
    feats = torch.stack([val_ds[i][0] for i in range(len(val_ds))])
    model.eval()
    with torch.no_grad():
        expected, _ = compose_gated_probs(
            model(feats), mineral_idx, non_mineral_idx)
    np.testing.assert_allclose(y_score, expected.numpy(), rtol=1e-5, atol=1e-6)


def test_ungated_validation_is_still_a_plain_sigmoid(captured_val_scores):
    """Backward compatibility: with gated_head absent the validation array
    must be bit-identical to what every existing run has always produced."""
    torch.manual_seed(3)
    df = _fake_df(seed=3)
    model = _LinearHead(len(data.dataset.LABEL_COLS))
    _run(model, df, use_asl_loss=True)

    y_score = captured_val_scores['y_score']
    from data.dataset import CRISMPixelDataset
    val_ds = CRISMPixelDataset(df[df['split'] == 'val'])
    feats = torch.stack([val_ds[i][0] for i in range(len(val_ds))])
    model.eval()
    with torch.no_grad():
        expected = torch.sigmoid(model(feats)).numpy()
    np.testing.assert_array_equal(y_score, expected)


# --- M3: the gate distribution must be observable --------------------------
# design.md:171-173 requires monitoring it: a gate saturating near 1
# degenerates to the flat head, near 0 kills every mineral at once. The gate
# BCE is unweighted and runs ~3.6x the main ASL term at lambda_gate=1.0, so a
# saturated gate is a live risk that would otherwise stay invisible until a
# tile is classified at hour 25.

def test_gate_statistics_are_logged_each_epoch(
        seven_class_labels, captured_val_scores, caplog):
    torch.manual_seed(4)
    with caplog.at_level('INFO', logger='training.train_torch'):
        _run(_LinearHead(8), _fake_df(seed=4), use_asl_loss=True,
             gated_head=True)
    text = '\n'.join(r.message for r in caplog.records)
    assert 'gate_mean' in text, f'no gate statistics in epoch log:\n{text}'
    for q in ('p10', 'p50', 'p90'):
        assert q in text, f'gate quantile {q} missing from epoch log:\n{text}'


def test_no_gate_statistics_for_an_ungated_run(captured_val_scores, caplog):
    """The existing epoch line must not change for non-gated runs."""
    torch.manual_seed(5)
    with caplog.at_level('INFO', logger='training.train_torch'):
        _run(_LinearHead(len(data.dataset.LABEL_COLS)), _fake_df(seed=5),
             use_asl_loss=True)
    text = '\n'.join(r.message for r in caplog.records)
    assert 'gate_mean' not in text

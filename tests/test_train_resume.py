"""Tests for training resume (--checkpoint_every / --resume).

A resumed run that is subtly wrong is worse than one that crashes: it completes,
writes a checkpoint, and reports metrics, while having trained at the wrong
learning rate or having overwritten a better checkpoint with a worse model. Every
test here targets one such silent failure and was verified by MUTATING
training/train_torch.py to reintroduce the bug and watching the test fail.

The LR-parity test is the load-bearing one. It uses warmup_epochs > 0 with the
cosine schedule so SequentialLR is actually exercised — with warmup 0 (plain
CosineAnnealingLR) or lr_schedule='step' the test would pass even with
scheduler.load_state_dict() removed entirely, which is exactly the trap.
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.train_torch import train_torch_model  # noqa: E402
from data.dataset import LABEL_COLS, MRRAL_BAND_COLS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_BANDS = len(MRRAL_BAND_COLS)          # 59
N_CLASSES = len(LABEL_COLS)             # 5 by default


def make_df(n=120, seed=0):
    """Tiny synthetic multilabel table served by CRISMSpectralDataset (flat 59-band
    pixels), so no rasterio, no patch cache and no real tiles are touched."""
    rng = np.random.default_rng(seed)
    data = {c: rng.random(n).astype('float32') for c in MRRAL_BAND_COLS}
    # _collapse_labels() requires the raw olivine_t1/olivine_t2 pair.
    for col in ['olivine_t1', 'olivine_t2', 'lcp', 'hcp', 'plagioclase', 'other']:
        data[col] = (rng.random(n) > 0.6).astype('float32')
    data['confidence_weight'] = np.ones(n, dtype='float32')
    data['confidence_tier'] = ['High'] * n
    n_train = int(n * 0.67)
    data['split'] = ['train'] * n_train + ['val'] * (n - n_train)
    return pd.DataFrame(data)


class LRProbeModel(torch.nn.Module):
    """Flat-input encoder/head model that records the optimizer LR once per epoch.

    train_torch_model calls model.train() exactly once per epoch (and model.eval()
    before validation), so overriding train() is a per-epoch hook that needs no
    changes to the trainer. The optimizer reference is injected by the test via an
    AdamW capture, because the trainer builds the optimizer itself.
    """

    def __init__(self, n_in=N_BANDS, n_out=N_CLASSES, hidden=8):
        super().__init__()
        self.encoder = torch.nn.Module()
        # Named band_embed so the resume compatibility guard (which keys off
        # encoder.band_embed.weight) is exercised by the mismatch tests.
        self.encoder.band_embed = torch.nn.Linear(n_in, hidden)
        self.head = torch.nn.Linear(hidden, n_out)
        self.lr_log = []
        self.optimizer = None

    def forward(self, x):
        return self.head(torch.relu(self.encoder.band_embed(x)))

    def get_param_groups(self, head_lr, encoder_lr):
        head_params = list(self.head.parameters())
        head_ids = {id(p) for p in head_params}
        enc_params = [p for p in self.parameters() if id(p) not in head_ids]
        return [{'params': enc_params, 'lr': encoder_lr},
                {'params': head_params, 'lr': head_lr}]

    def train(self, mode=True):
        if mode and self.optimizer is not None:
            self.lr_log.append(tuple(g['lr'] for g in self.optimizer.param_groups))
        return super().train(mode)


def run(model, df, *, ckpt_dir, name, max_epochs, **kw):
    """train_torch_model with the LR probe wired up and the scheduler config that
    makes SequentialLR (LinearLR warmup → CosineAnnealingLR) the live code path."""
    orig_adamw = torch.optim.AdamW

    def capture(params, **kwargs):
        opt = orig_adamw(params, **kwargs)
        model.optimizer = opt
        return opt

    cfg = dict(lr=1e-2, batch_size=32, patience=10_000, use_wandb=False,
               warmup_epochs=3, lr_t_max=6, lr_schedule='cosine',
               encoder_lr_scale=0.01)
    cfg.update(kw)
    with mock.patch('torch.optim.AdamW', side_effect=capture):
        return train_torch_model(model=model, df=df, model_name=name,
                                 checkpoint_dir=ckpt_dir, max_epochs=max_epochs,
                                 **cfg)


# ---------------------------------------------------------------------------
# 1. LR parity — the critical test
# ---------------------------------------------------------------------------

def test_resumed_lr_trajectory_matches_uninterrupted_run(tmp_path):
    """The LR sequence of split(K, N) training must equal that of one N-epoch run.

    Dropping scheduler.load_state_dict() makes the resumed half re-run LinearLR
    warmup and restart the cosine: the run completes, logs nothing unusual, and
    trains at the wrong LR for every remaining epoch.
    """
    N, K = 8, 4
    df = make_df()

    torch.manual_seed(0)
    whole = LRProbeModel()
    run(whole, df, ckpt_dir=str(tmp_path / 'whole'), name='whole', max_epochs=N)
    assert len(whole.lr_log) == N, whole.lr_log

    torch.manual_seed(0)
    first = LRProbeModel()
    run(first, df, ckpt_dir=str(tmp_path / 'split'), name='split',
        max_epochs=K, checkpoint_every=K)
    resume_path = tmp_path / 'split' / 'split_resume.pt'
    assert resume_path.exists(), 'no _resume.pt written'

    torch.manual_seed(123)   # deliberately different: LR must not depend on init
    second = LRProbeModel()
    run(second, df, ckpt_dir=str(tmp_path / 'split'), name='split',
        max_epochs=N, resume_from=str(resume_path))

    split_log = first.lr_log + second.lr_log
    assert len(split_log) == N, (first.lr_log, second.lr_log)
    diffs = [abs(a - b)
             for row_a, row_b in zip(whole.lr_log, split_log)
             for a, b in zip(row_a, row_b)]
    assert max(diffs) < 1e-12, (
        f'LR trajectories diverge (max abs diff {max(diffs):.3e}).\n'
        f'uninterrupted: {[r[0] for r in whole.lr_log]}\n'
        f'resumed:       {[r[0] for r in split_log]}')
    # Guard the guard: the schedule must actually vary, or "they match" is vacuous.
    assert len({r[0] for r in whole.lr_log}) == N


# ---------------------------------------------------------------------------
# 2. best_monitored survives
# ---------------------------------------------------------------------------

def test_resume_does_not_overwrite_a_better_best_checkpoint(tmp_path):
    """A resumed run whose epochs are all worse must leave _best.pt alone.

    Two ways to break it, both silent: reset best_monitored to -1.0 on resume (the
    first resumed epoch then "improves" and overwrites the good weights), or write
    the end-of-run _best.pt when best_state is None (writing model_state=None over
    a valid checkpoint).
    """
    ckpt_dir = tmp_path / 'ck'
    df = make_df()
    torch.manual_seed(0)
    model = LRProbeModel()
    run(model, df, ckpt_dir=str(ckpt_dir), name='m', max_epochs=2, checkpoint_every=2)

    resume_path = ckpt_dir / 'm_resume.pt'
    ck = torch.load(resume_path, map_location='cpu', weights_only=False)
    # An unbeatable best: mAP <= 1.0 always, so no resumed epoch can improve.
    ck['best_monitored'] = 1.0
    ck['best_epoch'] = 2
    torch.save(ck, resume_path)

    best_path = ckpt_dir / 'm_best.pt'
    good = torch.load(best_path, map_location='cpu', weights_only=False)
    good['best_monitored'] = 1.0
    good['sentinel'] = 'written-by-the-first-job'
    torch.save(good, best_path)

    torch.manual_seed(7)
    model2 = LRProbeModel()
    run(model2, df, ckpt_dir=str(ckpt_dir), name='m', max_epochs=6,
        checkpoint_every=2, resume_from=str(resume_path))

    after = torch.load(best_path, map_location='cpu', weights_only=False)
    assert after.get('sentinel') == 'written-by-the-first-job', (
        'the resumed run overwrote a better _best.pt')
    assert after['best_monitored'] == 1.0
    assert after['model_state'] is not None


# ---------------------------------------------------------------------------
# 3. patience_counter survives
# ---------------------------------------------------------------------------

def test_resume_preserves_patience_counter(tmp_path):
    """Early stopping must fire on schedule after a resume.

    patience_counter restored at 2 with patience=3 means the FIRST resumed epoch
    stops the run. Reset it to 0 and the run keeps going for 3 more epochs — on a
    real run, early stopping effectively never fires again.
    """
    ckpt_dir = tmp_path / 'ck'
    df = make_df()
    torch.manual_seed(0)
    model = LRProbeModel()
    run(model, df, ckpt_dir=str(ckpt_dir), name='p', max_epochs=2, checkpoint_every=2)

    resume_path = ckpt_dir / 'p_resume.pt'
    ck = torch.load(resume_path, map_location='cpu', weights_only=False)
    # Unreachable best (>1.0) so every resumed epoch is a regression and the
    # counter ticks; min_delta=0 keeps it outside the tolerance band too.
    ck['best_monitored'] = 1.5
    ck['patience_counter'] = 2
    torch.save(ck, resume_path)

    torch.manual_seed(7)
    model2 = LRProbeModel()
    metrics = run(model2, df, ckpt_dir=str(ckpt_dir), name='p', max_epochs=20,
                  patience=3, resume_from=str(resume_path))
    assert metrics['stopped_epoch'] == 3, (
        f"expected early stop at epoch 3 (counter 2/3 + one regression), got "
        f"{metrics['stopped_epoch']} — patience_counter was not restored")


# ---------------------------------------------------------------------------
# 4. Config-mismatch resumes raise, naming the mismatch
# ---------------------------------------------------------------------------

def _resume_file(tmp_path, name='x'):
    df = make_df()
    torch.manual_seed(0)
    run(LRProbeModel(), df, ckpt_dir=str(tmp_path), name=name,
        max_epochs=1, checkpoint_every=1)
    return str(tmp_path / f'{name}_resume.pt')


def test_resume_channel_mismatch_raises(tmp_path):
    """A 59-channel _resume.pt must not load into a 118-channel (dual-CR) run."""
    path = _resume_file(tmp_path, 'chan')
    wide = LRProbeModel(n_in=118)
    with pytest.raises(ValueError) as exc:
        run(wide, make_df(), ckpt_dir=str(tmp_path), name='chan',
            max_epochs=2, resume_from=path)
    msg = str(exc.value)
    assert 'encoder.band_embed.weight' in msg
    assert '59' in msg and '118' in msg, msg


def test_resume_vocab_mismatch_raises(tmp_path):
    """A 5-class _resume.pt must not load into a 7-class run: the head would be
    reinitialised silently and every target would be mislabelled."""
    path = _resume_file(tmp_path, 'vocab')
    seven = LRProbeModel(n_out=7)
    with pytest.raises(ValueError) as exc:
        run(seven, make_df(), ckpt_dir=str(tmp_path), name='vocab',
            max_epochs=2, resume_from=path)
    msg = str(exc.value)
    assert 'head.weight' in msg
    assert str(N_CLASSES) in msg and '7' in msg, msg


# ---------------------------------------------------------------------------
# 5. Mutual exclusion at parse time
# ---------------------------------------------------------------------------

def _parse(argv):
    """Run scripts/train.build_args() with argv in a subprocess; no I/O happens
    there, so this exercises the parser.error guards only."""
    code = ("import sys; sys.argv=['x'] + %r; "
            "import scripts.train as t; t.build_args(); print('PARSED_OK')" % argv)
    p = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                       capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def test_resume_with_pretrain_ckpt_is_refused_at_parse_time():
    base = ['--model', 'spatial_vit_aux', '--mrral_parquets', '_',
            '--resume', '/tmp/run_resume.pt']
    rc, out = _parse(base + ['--pretrain_ckpt', '/tmp/enc.pt'])
    assert rc != 0, out
    assert '--resume' in out and '--pretrain_ckpt' in out, out

    rc, out = _parse(base + ['--init_ckpt', '/tmp/cls.pt'])
    assert rc != 0, out
    assert '--resume' in out and '--init_ckpt' in out, out

    # ...and --resume on its own still parses.
    rc, out = _parse(base)
    assert rc == 0 and 'PARSED_OK' in out, out


def test_every_torch_branch_forwards_the_resume_flags():
    """--checkpoint_every / --resume must reach train_torch_model from EVERY model
    branch. A flag some branches accept and silently drop is worse than a missing
    flag: the command looks right and the run looks fine (2026-08-10, --synth_*)."""
    src = open(os.path.join(ROOT, 'scripts', 'train.py')).read()
    n_calls = src.count('= train_torch_model(')
    assert n_calls >= 8, n_calls
    assert src.count('checkpoint_every=args.checkpoint_every,') == n_calls
    assert src.count('resume_from=args.resume,') == n_calls


# ---------------------------------------------------------------------------
# 6. Inert default
# ---------------------------------------------------------------------------

def test_checkpoint_every_zero_changes_nothing(tmp_path):
    """checkpoint_every=0 (the default) writes no _resume.pt and leaves the run
    identical to one that never passes the new arguments at all."""
    df = make_df()
    a_dir, b_dir = tmp_path / 'a', tmp_path / 'b'

    torch.manual_seed(0)
    m_a = LRProbeModel()
    metrics_a = run(m_a, df, ckpt_dir=str(a_dir), name='r', max_epochs=3)

    torch.manual_seed(0)
    m_b = LRProbeModel()
    metrics_b = run(m_b, df, ckpt_dir=str(b_dir), name='r', max_epochs=3,
                    checkpoint_every=0, resume_from=None)

    expected = ['r_best.pt', 'r_best_map.pt', 'r_last.pt']
    assert sorted(os.listdir(a_dir)) == expected
    assert sorted(os.listdir(b_dir)) == expected, 'a _resume.pt was written anyway'
    assert metrics_a.keys() == metrics_b.keys()
    for k in metrics_a:                       # nan == nan is False; compare reprs
        assert repr(metrics_a[k]) == repr(metrics_b[k]), k
    for fname in expected:
        sa = torch.load(a_dir / fname, map_location='cpu', weights_only=False)
        sb = torch.load(b_dir / fname, map_location='cpu', weights_only=False)
        assert sa.keys() == sb.keys()
        for k in sa['model_state']:
            assert torch.equal(sa['model_state'][k], sb['model_state'][k]), k

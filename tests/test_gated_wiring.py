"""--gated_head must reach the model and the loss, not just parse."""
from __future__ import annotations

import inspect

import scripts.train as train_mod
import training.train_torch as tt


def test_train_torch_model_accepts_gated_head():
    assert 'gated_head' in inspect.signature(tt.train_torch_model).parameters


def test_gated_head_defaults_off():
    """Every existing run must be unaffected."""
    assert inspect.signature(tt.train_torch_model).parameters['gated_head'].default is False


def test_cli_exposes_gated_head():
    src = inspect.getsource(train_mod)
    assert "'--gated_head'" in src or '"--gated_head"' in src


def test_gated_head_forwarded_wherever_asl_is():
    """A flag that parses but is not forwarded silently trains the flat head
    while the log claims a gated run -- the worst outcome available here.
    scripts/train.py has 8 train_torch_model call sites; the 6 that pass
    use_asl_loss are exactly the ones that must also pass gated_head."""
    src = inspect.getsource(train_mod)
    n_asl = src.count('use_asl_loss=args.asl_loss')
    n_gated = src.count('gated_head=args.gated_head')
    assert n_asl == 6, f'call-site count changed ({n_asl}); re-read train.py'
    assert n_gated == n_asl, (
        f'{n_asl} call sites pass use_asl_loss but only {n_gated} pass gated_head')

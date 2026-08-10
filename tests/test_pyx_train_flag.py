import subprocess, sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def test_pyx_flag_sets_6class_vocab():
    code = ("import data.dataset as d, sys; "
            "sys.argv=['x','--pyx','--model','spatial_vit','--mrral_parquets','_']; "
            "import scripts.train as t; a=t.build_args(); "
            "print(a.n_classes, ','.join(d.LABEL_COLS))")
    out = subprocess.run([sys.executable,'-c',code], cwd=ROOT, capture_output=True, text=True)
    assert '6 olivine,pyx,plagioclase,bland,alteration,junk' in out.stdout, out.stdout+out.stderr


def test_every_model_branch_forwards_the_synth_flags():
    """--synth_* must reach train_torch_model for EVERY branch that accepts them.

    Regression for 2026-08-10: the spatial_vit_aux branch omitted all four
    synth_* kwargs while spatial_vit passed them, so --synth_train_cache and
    friends were accepted by argparse and then silently dropped. The
    ft_7cls_handcore_* runs trained with zero MTRDR plagioclase despite being
    explicitly configured to use it, and nothing in the log said so.

    A parser flag that some code paths ignore is worse than a missing flag: the
    command looks right, the run looks fine, and the intervention never happened.
    """
    import re
    import os

    src = open(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'scripts', 'train.py')).read()

    # Every train_torch_model(...) call site, sliced from the call to its close.
    calls = []
    for m in re.finditer(r'train_torch_model\(', src):
        depth, i = 0, m.end() - 1
        while i < len(src):
            if src[i] == '(':
                depth += 1
            elif src[i] == ')':
                depth -= 1
                if depth == 0:
                    calls.append(src[m.start():i + 1])
                    break
            i += 1
    assert calls, 'found no train_torch_model call sites'

    patch_calls = [c for c in calls if 'mrral_map=' in c]
    assert len(patch_calls) >= 2, f'expected >=2 patch-based call sites, got {len(patch_calls)}'

    KW = ('synth_train_cache', 'synth_train_parquet',
          'synth_val_cache', 'synth_val_parquet')
    forwarding = [c for c in patch_calls if all(f'{k}=' in c for k in KW)]
    partial = [c for c in patch_calls
               if any(f'{k}=' in c for k in KW) and c not in forwarding]
    assert not partial, 'a call site forwards SOME synth_* kwargs but not all'
    assert len(forwarding) >= 2, (
        'expected at least the spatial_vit and spatial_vit_aux branches to '
        'forward synth_*')

    # Branches that do NOT forward them must be rejected at parse time, so the
    # flag can never silently do nothing.
    assert '_SYNTH_SUPPORTED' in src, (
        'branches that ignore synth_* must be guarded by a parser.error, not '
        'left to drop the flags silently')
    assert 'is not supported by --model' in src

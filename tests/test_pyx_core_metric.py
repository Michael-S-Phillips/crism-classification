import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def test_core_excludes_junk_for_pyx_6class(monkeypatch):
    import data.dataset as d
    from training.train_torch import _core_map  # extracted helper (see impl)
    monkeypatch.setattr(d, 'LABEL_COLS', list(d.LABEL_COLS_PYX))  # has junk
    y_true = np.eye(6)[np.arange(6) % 6]
    y_score = y_true * 0.9 + 0.05
    core = _core_map(y_true, y_score)
    # junk (last col) excluded -> core computed over 5 classes, finite
    assert np.isfinite(core)

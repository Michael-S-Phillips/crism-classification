import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import _collapse_labels, LABEL_COLS_PYX

def test_pyx_is_max_of_lcp_hcp_and_preserves_them():
    df = pd.DataFrame({'olivine_t1':[0,0.,0], 'olivine_t2':[0,0.,0],
                       'lcp':[0.2,0.0,0.7], 'hcp':[0.0,0.9,0.6],
                       'plagioclase':[0,0,0.], 'bland':[0,0,0.],
                       'alteration':[0,0,0.], 'junk':[0,0,0.],
                       'confidence_tier':['high','high','high']})
    out = _collapse_labels(df)
    assert np.allclose(out['pyx'], [0.2, 0.9, 0.7])   # max(lcp,hcp)
    assert 'lcp' in out.columns and 'hcp' in out.columns  # preserved for Spec B
    assert LABEL_COLS_PYX == ['olivine','pyx','plagioclase','bland','alteration','junk']

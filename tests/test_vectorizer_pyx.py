import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.vectorize_per_mineral_thresholds_nili_6cls as V
class NPZ(dict):
    @property
    def files(self): return list(self.keys())
def test_pyx_channels_selected():
    d = NPZ(class_names=np.array(['olivine','pyx','plagioclase','bland','alteration','junk']))
    V._check_npz_channels(d, None)
    assert 'pyx' in V.MINERAL_NAMES and V.PROB_CHANNELS[1] == 'pyx'

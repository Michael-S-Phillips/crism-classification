import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.classify_tile_supervised as C
def test_pyx_overrides_altertion_6class():
    C.PYX_MODE = True
    C._set_n_classes({'head.weight': __import__('numpy').zeros((6,128))})
    assert C.CLASS_NAMES == ['olivine','pyx','plagioclase','bland','alteration','junk']

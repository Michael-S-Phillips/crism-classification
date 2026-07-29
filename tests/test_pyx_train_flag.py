import subprocess, sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def test_pyx_flag_sets_6class_vocab():
    code = ("import data.dataset as d, sys; "
            "sys.argv=['x','--pyx','--model','spatial_vit','--mrral_parquets','_']; "
            "import scripts.train as t; a=t.build_args(); "
            "print(a.n_classes, ','.join(d.LABEL_COLS))")
    out = subprocess.run([sys.executable,'-c',code], cwd=ROOT, capture_output=True, text=True)
    assert '6 olivine,pyx,plagioclase,bland,alteration,junk' in out.stdout, out.stdout+out.stderr

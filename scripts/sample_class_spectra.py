"""Sample representative spectra per class from EXACTLY the sources the
hand-core policy admits. Mirrors build_7cls_dataset's source policy."""
import sys, glob, re; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from data.continuum_removal import continuum_removed, good_band_mask_59

BANDS = [f'm{i}' for i in range(59)]
RNG = np.random.default_rng(42)
N = 4000   # pixels sampled per class

# wavelengths from a real header
from data.continuum_removal import WAVELENGTHS_59 as WAV  # authoritative 59-band grid

def take(df, mask, n=N):
    idx = np.flatnonzero(np.asarray(mask))
    if len(idx) == 0: return None
    if len(idx) > n: idx = RNG.choice(idx, n, replace=False)
    return df.iloc[idx][BANDS].to_numpy(np.float32)

out = {}

# ---- hand-labeled base: olivine / lcp / hcp / plagioclase / alteration ----
base = pd.read_parquet('data/mrral_pixels.parquet',
                       columns=BANDS+['olivine_t1','olivine_t2','lcp','hcp',
                                      'plagioclase','other','alteration'])
oliv = (base['olivine_t1']>0)|(base['olivine_t2']>0)
out['olivine']     = [('hand', take(base, oliv))]
out['lcp']         = [('hand', take(base, base['lcp']>0))]
out['hcp']         = [('hand', take(base, base['hcp']>0))]
out['plagioclase'] = [('hand', take(base, base['plagioclase']>0))]
out['alteration']  = [('hand', take(base, base['alteration']>0))]
del base

# ---- v3 review (High+Moderate only) ----
KEEP = {'Reviewed-High','Reviewed-Moderate'}
cf = pd.concat([pd.read_parquet(f) for f in
                sorted(glob.glob('data/mc13_review_7cls_v3/confirmed_pixels/*.parquet'))],
               ignore_index=True)
cf = cf[cf['confidence_tier'].isin(KEEP)]
for cls, m in [('olivine', (cf['olivine_t1']>0)|(cf['olivine_t2']>0)),
               ('lcp', cf['lcp']>0), ('hcp', cf['hcp']>0),
               ('alteration', cf['alteration']>0)]:
    s = take(cf, m)
    if s is not None: out[cls].append(('v3 review', s))
del cf

hn = pd.concat([pd.read_parquet(f) for f in
                sorted(glob.glob('data/mc13_review_7cls_v3/hard_negatives/*.parquet'))],
               ignore_index=True)
hn = hn[hn['confidence_tier'].isin(KEEP)]
out['bland'] = [('v3 review', take(hn, hn['negative_of']==''))]
out['junk']  = [('v3 review', take(hn, hn['negative_of']=='ambiguous'))]
del hn

# ---- MTRDR plag (real, targeted observations) + synthetic plag library ----
for lbl, path in [('MTRDR', 'data/patch_cache/mtrdr_plag_rows.parquet'),
                  ('synth', 'data/patch_cache/synth_plag_rows.parquet')]:
    try:
        r = pd.read_parquet(path, columns=BANDS+['plagioclase'])
    except Exception as e:
        print(f'  skip {lbl}: {e}'); continue
    arr = take(r, r['plagioclase'] > 0)
    if arr is not None:
        out['plagioclase'].append((lbl, arr))

np.savez_compressed(
    '/tmp/claude-1000/-mnt-mars-gis-CRISM-MRDR/932813d4-7224-4166-ad78-a68e5521e8fe/scratchpad/spectra.npz',
    wav=WAV, good=good_band_mask_59(),
    **{f'{c}__{src}': arr for c, lst in out.items() for src, arr in lst if arr is not None})
for c, lst in out.items():
    for src, arr in lst:
        if arr is not None:
            print(f'{c:12} {src:10} n={len(arr):>5}  mean I/F {np.nanmean(arr):.4f}')

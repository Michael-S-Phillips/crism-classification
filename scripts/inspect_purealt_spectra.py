"""Pull example spectra from the pure-alteration training signal to verify
they look like alteration minerals (hydration bands at 1.4/1.9/2.2/2.3 um).

The pure-alt train signal is dominated by the 103,895 'Reviewed'-tier MC11
review-tagged alteration pixels. This plots a sample of them (raw + mean-
normalized) against olivine and bland class means for contrast.
"""
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HDR = '/mnt/mrdr/mc11/t1086_mrral_05n338_0327_4.hdr'
PARQUET = 'data/mrral_pixels_with_review_v2.parquet'
NODATA = 65535.0
MC11 = {'t1086', 't1087', 't1088', 't1090'}

hdr = open(HDR).read()
m = re.search(r'wavelength\s*=\s*\{([^}]*)\}', hdr, re.S)
wl = np.array([float(x) for x in m.group(1).replace('\n', '').split(',')
               if x.strip()])[:59]

bands = [f'm{i}' for i in range(59)]
cols = bands + ['split', 'alteration', 'olivine_t1', 'olivine_t2', 'other',
                'confidence_tier', 'tile_id']
df = pd.read_parquet(PARQUET, columns=cols)
tr = df[df.split == 'train']


def clean(sub):
    X = np.array(sub[bands].to_numpy(np.float32), copy=True)
    X[X >= NODATA] = np.nan
    X[X <= 0] = np.nan
    return X


pure = tr[(tr.alteration > 0.5) & (tr.confidence_tier == 'Reviewed')
          & (tr.tile_id.isin(MC11))]
gpkg = tr[(tr.alteration > 0.5) & (tr.confidence_tier != 'Reviewed')]
oli = tr[(tr.olivine_t1 > 0.5) | (tr.olivine_t2 > 0.5)]
bland = tr[tr.other > 0.5]

Xpure = clean(pure)
Xgpkg = clean(gpkg)
Xoli = clean(oli.sample(min(5000, len(oli)), random_state=0))
Xbl = clean(bland.sample(min(5000, len(bland)), random_state=0))

rng = np.random.default_rng(1)
ex = rng.choice(len(Xpure), 6, replace=False)


def norm(X):
    return X / np.nanmean(X, axis=1, keepdims=True)


fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
ax = axes[0]
for i in ex:
    ax.plot(wl, Xpure[i], lw=0.8, alpha=0.55)
ax.plot(wl, np.nanmean(Xpure, 0), 'k-', lw=2.5,
        label=f'MC11 review-alt mean (n={len(Xpure):,})')
ax.plot(wl, np.nanmean(Xoli, 0), color='tab:red', lw=2, ls='--', label='olivine mean')
ax.plot(wl, np.nanmean(Xbl, 0), color='gray', lw=2, ls=':', label='bland mean')
ax.set_title('Raw reflectance: 6 example MC11 review-alt + class means')
ax.set_xlabel('wavelength (nm)'); ax.set_ylabel('I/F reflectance')
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(wl, np.nanmean(norm(Xpure), 0), 'k-', lw=2.5,
        label=f'MC11 review-alt (n={len(Xpure):,})')
ax.plot(wl, np.nanmean(norm(Xgpkg), 0), color='tab:purple', lw=2,
        label=f'gpkg-alt Argyre/Hellas (n={len(Xgpkg):,})')
ax.plot(wl, np.nanmean(norm(Xoli), 0), color='tab:red', lw=2, ls='--', label='olivine')
ax.plot(wl, np.nanmean(norm(Xbl), 0), color='gray', lw=2, ls=':', label='bland')
y1 = ax.get_ylim()[1]
for w, lab in [(1400, '1.4'), (1900, '1.9'), (2200, '2.2'), (2300, '2.3')]:
    ax.axvline(w, color='c', lw=0.7, alpha=0.5)
    ax.text(w, y1, lab, fontsize=7, color='c', ha='center', va='bottom')
ax.set_title('Mean-normalized shape (hydration bands, um, cyan)')
ax.set_xlabel('wavelength (nm)'); ax.set_ylabel('reflectance / mean')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('reports/purealt_train_spectra.png', dpi=130)
print('saved reports/purealt_train_spectra.png')

print('\n=== sanity stats ===')
print(f'MC11 review-alt pixels: {len(Xpure):,}   '
      f'all-NaN rows: {int(np.isnan(Xpure).all(1).sum())}   '
      f'any-NaN frac: {np.isnan(Xpure).any(1).mean():.3f}')
mp = np.nanmean(Xpure, 0)
for um in (1.0, 1.9, 2.3):
    j = int(np.argmin(np.abs(wl - um * 1000)))
    print(f'  mean reflectance @ {um:.1f}um (band {j}, {wl[j]:.0f}nm): {mp[j]:.4f}')
# crude band-depth check: 1.9um relative to 1.7/2.1 shoulders
def bd(target, s1, s2):
    jt = int(np.argmin(np.abs(wl - target)))
    j1 = int(np.argmin(np.abs(wl - s1)))
    j2 = int(np.argmin(np.abs(wl - s2)))
    cont = (mp[j1] + mp[j2]) / 2
    return 1 - mp[jt] / cont
print(f'  1.9um band depth (vs 1750/2100 shoulders): {bd(1900,1750,2100):+.4f}')
print(f'  2.3um band depth (vs 2150/2400 shoulders): {bd(2300,2150,2400):+.4f}')

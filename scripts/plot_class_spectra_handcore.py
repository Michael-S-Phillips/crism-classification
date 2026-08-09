import sys; sys.path.insert(0, '/mnt/mars-gis/CRISM/MRDR/crism_classification')
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data.continuum_removal import continuum_removed, good_band_mask_59

S='/tmp/claude-1000/-mnt-mars-gis-CRISM-MRDR/932813d4-7224-4166-ad78-a68e5521e8fe/scratchpad'
d=np.load(f'{S}/spectra.npz'); WAV=d['wav']
GOOD=good_band_mask_59()   # bands 16-19 (1021-1056nm) are the detector-overlap
                          # window: CR sets them to 1.0 by construction, not measured
HAND, REVIEW = '#0072B2', '#D55E00'          # CVD-validated pair (worst ΔE 28.8)
INK, MUTED, GRID = '#1a1a1a', '#5c5c5c', '#d8d8d8'

CLASSES=[('olivine','olivine'),('lcp','lcp'),('hcp','hcp'),
         ('plagioclase','plagioclase'),('alteration','alteration'),
         ('bland','bland / dust'),('junk','junk')]
SRC=[('hand','hand-labeled',HAND),('v3 review','v3 review',REVIEW)]

def prep(a):
    """Mask nodata exactly as the training reader does, then CR."""
    a=a.astype(np.float32).copy()
    a[(a>1.0)|(a==65535)|(~np.isfinite(a))]=np.nan
    a=np.clip(a,0,0.5)
    ok=np.isfinite(a).all(1); a=a[ok]
    return a, continuum_removed(a[:,None,None,:].copy())[:,0,0,:]

fig,axes=plt.subplots(2,4,figsize=(19,8.6),sharex=True)
fig.patch.set_facecolor('white')
for ax,(key,title) in zip(axes.ravel(),CLASSES):
    n_tot=0
    for skey,slabel,col in SRC:
        k=f'{key}__{skey}'
        if k not in d.files: continue
        raw,cr=prep(d[k]); n_tot+=len(raw)
        med=np.nanmedian(cr,0); lo=np.nanpercentile(cr,25,0); hi=np.nanpercentile(cr,75,0)
        med=np.where(GOOD,med,np.nan); lo=np.where(GOOD,lo,np.nan); hi=np.where(GOOD,hi,np.nan)
        ax.fill_between(WAV,lo,hi,color=col,alpha=0.16,linewidth=0)
        ax.plot(WAV,med,color=col,lw=2.0,label=f'{slabel}  (n={len(raw):,})',
                solid_capstyle='round')
    ax.set_title(title,fontsize=13,color=INK,fontweight='600',loc='left',pad=7)
    ax.axhline(1.0,color=GRID,lw=1,zorder=0)
    ax.grid(True,color=GRID,lw=0.6,alpha=0.7); ax.set_axisbelow(True)
    for s in ('top','right'): ax.spines[s].set_visible(False)
    for s in ('left','bottom'): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED,labelsize=9)
    ax.set_ylim(0.55,1.06)
    ax.legend(frameon=False,fontsize=9,loc='lower left',labelcolor=INK)

ax=axes.ravel()[7]; ax.axis('off')
ax.text(0,0.95,'How to read this',fontsize=12,fontweight='600',color=INK,va='top')
ax.text(0,0.86,'Continuum-removed reflectance — what the model\nsees. 1.0 = continuum; dips are absorptions.\nLine = median, shading = IQR (25–75%).\n\nBlue vs orange is the SAME class from two\nsources. Divergence = review is teaching\nsomething different from the hand labels.\n\nGap at ~1030 nm: the detector-overlap window,\nwhich CR sets to 1.0 by construction.\n\nplagioclase has no review line — zero plag\nconfirms exist in either review session.\n\nbland is POST-REPAIR: t1444 (72% of the class)\nhad a zero-filled 2.25–2.46 µm tail until it\nwas re-extracted from the source tile today.',
        fontsize=9.5,color=MUTED,va='top',linespacing=1.55)
for ax in axes.ravel()[4:7]: ax.set_xlabel('wavelength (nm)',fontsize=10,color=MUTED)
axes[0,0].set_ylabel('CR reflectance',fontsize=10,color=MUTED)
axes[1,0].set_ylabel('CR reflectance',fontsize=10,color=MUTED)
fig.suptitle('Hand-core dataset — representative spectra by class and source',
             fontsize=16,color=INK,fontweight='600',x=0.008,ha='left',y=0.985)
fig.text(0.008,0.945,'Sampled from exactly the rows the hand-core policy admits (v3 review filtered to High+Moderate). '
         'Nodata masked and clipped to [0, 0.5] as the training reader does.',fontsize=10,color=MUTED,ha='left')
fig.tight_layout(rect=[0,0,1,0.93])
out='reports/handcore_spectra_by_class.png'
fig.savefig(out,dpi=150,facecolor='white'); print('wrote',out)

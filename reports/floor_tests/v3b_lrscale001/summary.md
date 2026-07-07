# Floor test: v3b_lrscale001

- checkpoint: `checkpoints/ft_7cls_v3b_lrscale001_best.pt`
- date: 2026-07-07T22:57Z
- tiles: nili t1249 t1250 t1321 t1322 | argyre t0434 t0435

## nili — per-mineral × threshold polygon counts
```
Per-mineral × threshold polygon counts:
  mineral        0.50 0.60 0.75 0.85 0.90 0.95 0.97 0.99
  olivine         1,136  1,599  3,594  1,870  1,246    884    740    685
  lcp               325    238    160     65     40     12      7      4
  hcp               371    230    101     37     17      1      0      0
  plagioclase         7      3      3      1      0      0      0      0
  alteration        176    105     43      7      3      0      0      0
```

gpkg sizes:
```
    536576  /mnt/mrdr/crism_classification/reports/floor_tests/v3b_lrscale001/nili/alteration.gpkg
    983040  /mnt/mrdr/crism_classification/reports/floor_tests/v3b_lrscale001/nili/hcp.gpkg
   1114112  /mnt/mrdr/crism_classification/reports/floor_tests/v3b_lrscale001/nili/lcp.gpkg
  16080896  /mnt/mrdr/crism_classification/reports/floor_tests/v3b_lrscale001/nili/olivine.gpkg
    176128  /mnt/mrdr/crism_classification/reports/floor_tests/v3b_lrscale001/nili/plagioclase.gpkg
```

## argyre — per-mineral × threshold polygon counts
```
Per-mineral × threshold polygon counts:
  mineral        0.50 0.60 0.75 0.85 0.90 0.95 0.97 0.99
  olivine            82    122    372    794  1,153  1,001    621    242
  lcp                12      9      7      1      1      0      0      0
  hcp               148     94     41      2      0      0      0      0
  plagioclase         0      0      0      0      0      0      0      0
  alteration         32     10      4      0      0      0      0      0
```

gpkg sizes:
```
    188416  /mnt/mrdr/crism_classification/reports/floor_tests/v3b_lrscale001/argyre/alteration.gpkg
    454656  /mnt/mrdr/crism_classification/reports/floor_tests/v3b_lrscale001/argyre/hcp.gpkg
    229376  /mnt/mrdr/crism_classification/reports/floor_tests/v3b_lrscale001/argyre/lcp.gpkg
   7081984  /mnt/mrdr/crism_classification/reports/floor_tests/v3b_lrscale001/argyre/olivine.gpkg
```

No previous floor test found — this is the baseline.

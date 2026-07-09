# Tiered SAM Classification of Standard Test Sites

Endmembers: 8 medoids from reports/label_quantification/endmembers.csv (kind=='medoid'), 57-band window m2..m58 (raster bands 3..59).

Conservative rule: a pixel is labelled mineral M only if its nearest of ALL 8 endmembers (incl. bland_dust/bland_reject/junk background) is M AND its spectral angle to M <= the layer threshold. Output classes: olivine, lcp, hcp, plagioclase, alteration.

Angle ladder (deg, loosest->tightest): 3, 2, 1.5, 1.25, 1, 0.8, 0.6, 0.5. min polygon size = 4 px.

Values below are POLYGON counts (>= min px), pooled across each region's tiles. Counts are non-monotonic in angle: loose thresholds yield few large blobs, mid thresholds fragment into many polygons, tight thresholds vanish.


## Nili Fossae (t1249, t1250, t1321, t1322) — 154s

### Polygon counts per mineral x angle

| mineral | ang_3.0 | ang_2.0 | ang_1.5 | ang_1.25 | ang_1.0 | ang_0.8 | ang_0.6 | ang_0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| olivine | 3518 | 2660 | 1905 | 992 | 296 | 28 | 0 | 0 |
| lcp | 9664 | 8420 | 7486 | 6075 | 2367 | 344 | 18 | 0 |
| hcp | 5365 | 3862 | 2192 | 1224 | 429 | 29 | 0 | 0 |
| plagioclase | 9048 | 7673 | 6676 | 3765 | 1172 | 107 | 1 | 0 |
| alteration | 4856 | 1498 | 32 | 1 | 0 | 0 | 0 | 0 |


### Assigned-pixel angle distribution (pixels whose nearest endmember is the mineral)

| mineral | n_pixels (argmin=mineral) | p10 angle (deg) | p50 angle (deg) |
| --- | --- | --- | --- |
| olivine | 224206 | 1.27 | 1.99 |
| lcp | 2151631 | 1.24 | 2.17 |
| hcp | 336750 | 1.31 | 2.43 |
| plagioclase | 759068 | 1.17 | 1.64 |
| alteration | 118494 | 1.81 | 2.17 |


## Argyre (t0434, t0435) — 92s

### Polygon counts per mineral x angle

| mineral | ang_3.0 | ang_2.0 | ang_1.5 | ang_1.25 | ang_1.0 | ang_0.8 | ang_0.6 | ang_0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| olivine | 3258 | 3784 | 4358 | 4290 | 2152 | 406 | 10 | 0 |
| lcp | 4769 | 6397 | 8474 | 8769 | 5808 | 1285 | 36 | 0 |
| hcp | 11026 | 11549 | 10984 | 8528 | 3170 | 465 | 12 | 0 |
| plagioclase | 9469 | 9408 | 9478 | 9450 | 7999 | 3650 | 463 | 47 |
| alteration | 44 | 16 | 4 | 0 | 0 | 0 | 0 | 0 |


### Assigned-pixel angle distribution (pixels whose nearest endmember is the mineral)

| mineral | n_pixels (argmin=mineral) | p10 angle (deg) | p50 angle (deg) |
| --- | --- | --- | --- |
| olivine | 302478 | 0.95 | 1.35 |
| lcp | 1202217 | 0.96 | 1.45 |
| hcp | 543605 | 0.98 | 1.37 |
| plagioclase | 633861 | 0.76 | 1.03 |
| alteration | 1095 | 1.61 | 2.04 |

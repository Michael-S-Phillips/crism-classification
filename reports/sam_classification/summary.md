# Tiered SAM Classification of Standard Test Sites

Endmembers: 8 medoids from reports/label_quantification/endmembers.csv (kind=='medoid'), 57-band window m2..m58 (raster bands 3..59).

Conservative rule: a pixel is labelled mineral M only if its nearest of ALL 8 endmembers (incl. bland_dust/bland_reject/junk background) is M AND its spectral angle to M <= the layer threshold. Output classes: olivine, lcp, hcp, plagioclase, alteration.

Angle ladder (deg, loosest->tightest): 3, 2, 1.5, 1.25, 1, 0.8, 0.6, 0.5. min polygon size = 4 px.

Values below are POLYGON counts (>= min px), pooled across each region's tiles. Counts are non-monotonic in angle: loose thresholds yield few large blobs, mid thresholds fragment into many polygons, tight thresholds vanish.


## Nili Fossae (t1249, t1250, t1321, t1322) — 153s

### Polygon counts per mineral x angle

| mineral | ang_3.0 | ang_2.0 | ang_1.5 | ang_1.25 | ang_1.0 | ang_0.8 | ang_0.6 | ang_0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| olivine | 3438 | 2652 | 2104 | 1308 | 564 | 116 | 1 | 0 |
| lcp | 9421 | 8072 | 6883 | 5711 | 3649 | 996 | 62 | 5 |
| hcp | 4775 | 3941 | 2577 | 1676 | 1030 | 301 | 6 | 0 |
| plagioclase | 7931 | 6412 | 5520 | 3653 | 1378 | 362 | 4 | 0 |
| alteration | 5948 | 3373 | 128 | 3 | 0 | 0 | 0 | 0 |


### Assigned-pixel angle distribution (pixels whose nearest endmember is the mineral)

| mineral | n_pixels (argmin=mineral) | p10 angle (deg) | p50 angle (deg) |
| --- | --- | --- | --- |
| olivine | 223284 | 1.14 | 1.95 |
| lcp | 1839287 | 1.06 | 1.73 |
| hcp | 331052 | 1.06 | 1.85 |
| plagioclase | 629536 | 1.08 | 1.51 |
| alteration | 182083 | 1.71 | 2.06 |


## Argyre (t0434, t0435) — 88s

### Polygon counts per mineral x angle

| mineral | ang_3.0 | ang_2.0 | ang_1.5 | ang_1.25 | ang_1.0 | ang_0.8 | ang_0.6 | ang_0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| olivine | 2600 | 2725 | 3173 | 3442 | 2606 | 1068 | 99 | 12 |
| lcp | 3706 | 4439 | 5434 | 6259 | 6793 | 3566 | 148 | 10 |
| hcp | 5537 | 5577 | 5264 | 5005 | 4040 | 1329 | 79 | 6 |
| plagioclase | 5054 | 5052 | 5063 | 5601 | 5805 | 3487 | 215 | 4 |
| alteration | 31 | 18 | 5 | 0 | 0 | 0 | 0 | 0 |


### Assigned-pixel angle distribution (pixels whose nearest endmember is the mineral)

| mineral | n_pixels (argmin=mineral) | p10 angle (deg) | p50 angle (deg) |
| --- | --- | --- | --- |
| olivine | 258869 | 0.83 | 1.19 |
| lcp | 1389422 | 0.83 | 1.15 |
| hcp | 292771 | 0.81 | 1.04 |
| plagioclase | 715124 | 0.76 | 0.99 |
| alteration | 744 | 1.47 | 1.82 |

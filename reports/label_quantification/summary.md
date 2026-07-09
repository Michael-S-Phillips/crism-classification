# Label Quantification — SAM Endmember Analysis

Analysis window: 57 bands (m2..m58, 534-2457 nm), raw reflectance.

min_px = 10. Angles in DEGREES. Class-level math on polygon mean spectra, single-label only.

Runtime: 13.6 s.


## Interpretation caveat

Raw-reflectance spectral angles are continuum/albedo-dominated (all
inter-class medoid angles come out <3 deg), so absolute suspect counts
are structurally inflated. Read margins as a RELATIVE worst-offenders
ranking, not a mislabel census. Continuum removal is the planned v2
that would make these angles mineralogical.


## Per-source purity (headline)

Count of polygons, median angle to own-class medoid, and % with negative margin (closer to another class).


| class | source | n_polygons | median_own_angle_deg | pct_suspect |
| --- | --- | --- | --- | --- |
| alteration | confirmed | 8 | 3.57 | 0.00 |
| alteration | tag | 76 | 2.15 | 36.84 |
| bland | hand | 8 | 10.05 | 100.00 |
| bland | reassigned | 251 | 6.53 | 76.49 |
| hcp | confirmed | 28 | 4.76 | 92.86 |
| hcp | hand | 575 | 1.48 | 68.70 |
| junk | tag | 116 | 5.03 | 40.52 |
| lcp | confirmed | 166 | 6.76 | 93.37 |
| lcp | hand | 706 | 1.54 | 36.97 |
| olivine | confirmed | 52 | 2.67 | 88.46 |
| olivine | hand | 562 | 1.63 | 49.11 |
| olivine | reassigned | 93 | 5.82 | 16.13 |
| plagioclase | hand | 519 | 1.59 | 58.57 |


## Cross-source coherence per class

Angle between per-source sub-medoids — quantifies source disagreement within a class.


| class | source_a | source_b | n_a | n_b | angle_deg |
| --- | --- | --- | --- | --- | --- |
| alteration | confirmed | tag | 5 | 71 | 1.42 |
| bland | hand | reassigned | 8 | 243 | 10.13 |
| hcp | confirmed | hand | 26 | 573 | 4.51 |
| lcp | confirmed | hand | 153 | 695 | 6.22 |
| olivine | confirmed | reassigned | 50 | 82 | 5.35 |
| olivine | hand | reassigned | 549 | 82 | 4.86 |
| olivine | confirmed | hand | 50 | 549 | 2.48 |


## Inter-class medoid angle matrix (deg)


| class | olivine | lcp | hcp | plagioclase | alteration | bland | junk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| olivine | 0.00 | 2.52 | 0.88 | 1.20 | 2.34 | 7.37 | 9.35 |
| lcp | 2.52 | 0.00 | 1.96 | 1.55 | 2.98 | 5.53 | 7.63 |
| hcp | 0.88 | 1.96 | 0.00 | 1.01 | 2.42 | 6.79 | 8.73 |
| plagioclase | 1.20 | 1.55 | 1.01 | 0.00 | 2.42 | 6.49 | 8.60 |
| alteration | 2.34 | 2.98 | 2.42 | 2.42 | 0.00 | 7.73 | 9.78 |
| bland | 7.37 | 5.53 | 6.79 | 6.49 | 7.73 | 0.00 | 2.63 |
| junk | 9.35 | 7.63 | 8.73 | 8.60 | 9.78 | 2.63 | 0.00 |


## Per-class intra-class spread (angle to own medoid, deg)


| class | n_polygons | mean_deg | p50_deg | p90_deg |
| --- | --- | --- | --- | --- |
| olivine | 681 | 2.87 | 1.85 | 5.34 |
| lcp | 848 | 3.04 | 1.77 | 6.53 |
| hcp | 599 | 2.35 | 1.54 | 3.47 |
| plagioclase | 507 | 1.98 | 1.58 | 2.94 |
| alteration | 76 | 4.07 | 2.24 | 10.14 |
| bland | 251 | 8.14 | 6.83 | 19.11 |
| junk | 105 | 6.70 | 5.11 | 10.69 |


## Bland & junk (diagnostic classes)

`bland` and `junk` are catch-all classes, not mineral endmembers, so
their purity is DIAGNOSTIC rather than a quality gate. The interesting
signal is which mineral class each hugs spectrally (the
nearest-other-class column below). `junk` is single-source (`tag`), so
cross-source coherence is n/a for it. Note also that base (`hand`)
`bland` uses a constant polygon_id (0): with the (class, source,
tile_id, polygon_id) key this collapses to one huge tile-mean polygon
per tile — expected, and it does not affect the medoid math.


### `junk` nearest-other-class distribution (116 polygons)


| nearest_other_class | n_polygons | pct |
| --- | --- | --- |
| bland | 92 | 79.31 |
| lcp | 9 | 7.76 |
| hcp | 7 | 6.03 |
| olivine | 4 | 3.45 |
| plagioclase | 2 | 1.72 |
| alteration | 2 | 1.72 |


### `bland` nearest-other-class distribution (259 polygons)


| nearest_other_class | n_polygons | pct |
| --- | --- | --- |
| junk | 176 | 67.95 |
| alteration | 45 | 17.37 |
| olivine | 14 | 5.41 |
| plagioclase | 12 | 4.63 |
| lcp | 11 | 4.25 |
| hcp | 1 | 0.39 |



## Suspect polygons (negative margin)

Total suspects: 1753 / 3160 polygons (55.5%).


Suspects by class:


| class | n_suspect | n_polygons |
| --- | --- | --- |
| alteration | 28 | 84 |
| bland | 200 | 259 |
| hcp | 421 | 603 |
| junk | 47 | 116 |
| lcp | 416 | 872 |
| olivine | 337 | 707 |
| plagioclase | 304 | 519 |


### Top-20 worst-margin suspects (with provenance)


| class | source | tile_id | polygon_id | n_px | confidence_weight | own_angle_deg | nearest_other_class | nearest_other_angle_deg | margin_deg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| alteration | tag | t1159 | 1600979008 | 7 | 0.50 | 13.79 | junk | 5.48 | -8.31 |
| alteration | tag | t1159 | 801889504 | 624 | 0.50 | 14.94 | junk | 6.95 | -7.99 |
| alteration | tag | t1160 | 82877845 | 46 | 0.50 | 11.89 | junk | 4.09 | -7.81 |
| alteration | tag | t1160 | 1248938873 | 12 | 0.50 | 10.55 | junk | 2.85 | -7.70 |
| alteration | tag | t1160 | 1192594267 | 9 | 0.50 | 11.36 | junk | 3.70 | -7.66 |
| alteration | tag | t1159 | 148205312 | 489 | 0.50 | 15.23 | junk | 7.79 | -7.43 |
| alteration | tag | t1160 | 226399413 | 482 | 0.50 | 11.38 | junk | 4.02 | -7.36 |
| alteration | tag | t1160 | 4147629625 | 51 | 0.50 | 11.48 | junk | 4.25 | -7.23 |
| alteration | tag | t1160 | 3804725248 | 11 | 0.50 | 10.35 | junk | 3.18 | -7.17 |
| junk | tag | t1370 | 3044974737 | 45 | 1.00 | 10.52 | plagioclase | 3.68 | -6.84 |
| olivine | hand | t0360 | 13 | 411 | 1.00 | 36.43 | junk | 29.61 | -6.82 |
| olivine | hand | t0360 | 26 | 540 | 1.00 | 37.50 | junk | 30.69 | -6.81 |
| olivine | hand | t0360 | 22 | 41 | 1.00 | 36.41 | junk | 29.60 | -6.81 |
| olivine | hand | t0360 | 16 | 73 | 1.00 | 36.38 | junk | 29.72 | -6.66 |
| alteration | tag | t1160 | 1632260864 | 2216 | 0.50 | 11.43 | junk | 4.82 | -6.61 |
| olivine | hand | t0360 | 25 | 67 | 1.00 | 36.37 | junk | 29.80 | -6.57 |
| bland | reassigned | t1101 | 3112597758 | 436 | 1.00 | 10.69 | alteration | 4.13 | -6.56 |
| bland | reassigned | t1030 | 2677539686 | 70 | 1.00 | 9.76 | alteration | 3.25 | -6.51 |
| bland | reassigned | t1028 | 638435218 | 66 | 1.00 | 12.45 | olivine | 5.94 | -6.51 |
| bland | reassigned | t1030 | 632208560 | 2344 | 1.00 | 9.98 | alteration | 3.51 | -6.47 |

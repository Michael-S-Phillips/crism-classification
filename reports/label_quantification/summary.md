# Label Quantification — SAM Endmember Analysis

Analysis window: 57 bands (m2..m58, 534-2457 nm), raw reflectance.

min_px = 10. Angles in DEGREES. Class-level math on polygon mean spectra, single-label only.

Runtime: 14.5 s.


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
| bland_dust | hand | 8 | 0.22 | 0.00 |
| bland_reject | reassigned | 251 | 6.53 | 76.49 |
| hcp | confirmed | 28 | 4.76 | 92.86 |
| hcp | hand | 575 | 1.48 | 68.17 |
| junk | tag | 116 | 5.03 | 40.52 |
| lcp | confirmed | 166 | 6.76 | 93.37 |
| lcp | hand | 706 | 1.54 | 36.69 |
| olivine | confirmed | 52 | 2.67 | 88.46 |
| olivine | hand | 562 | 1.61 | 51.42 |
| olivine | reassigned | 93 | 5.82 | 16.13 |
| plagioclase | hand | 519 | 1.58 | 58.38 |


## Cross-source coherence per class

Angle between per-source sub-medoids — quantifies source disagreement within a class.


| class | source_a | source_b | n_a | n_b | angle_deg |
| --- | --- | --- | --- | --- | --- |
| alteration | confirmed | tag | 5 | 71 | 1.42 |
| hcp | confirmed | hand | 26 | 573 | 4.51 |
| lcp | confirmed | hand | 153 | 695 | 6.22 |
| olivine | confirmed | reassigned | 50 | 82 | 5.35 |
| olivine | hand | reassigned | 549 | 82 | 4.86 |
| olivine | confirmed | hand | 50 | 549 | 2.48 |


## Inter-class medoid angle matrix (deg)


| class | olivine | lcp | hcp | plagioclase | alteration | bland_dust | bland_reject | junk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| olivine | 0.00 | 2.52 | 0.88 | 1.20 | 2.34 | 3.90 | 7.37 | 9.35 |
| lcp | 2.52 | 0.00 | 1.96 | 1.55 | 2.98 | 5.33 | 5.53 | 7.63 |
| hcp | 0.88 | 1.96 | 0.00 | 1.01 | 2.42 | 4.59 | 6.79 | 8.73 |
| plagioclase | 1.20 | 1.55 | 1.01 | 0.00 | 2.42 | 4.19 | 6.49 | 8.60 |
| alteration | 2.34 | 2.98 | 2.42 | 2.42 | 0.00 | 3.81 | 7.73 | 9.78 |
| bland_dust | 3.90 | 5.33 | 4.59 | 4.19 | 3.81 | 0.00 | 10.13 | 12.38 |
| bland_reject | 7.37 | 5.53 | 6.79 | 6.49 | 7.73 | 10.13 | 0.00 | 2.63 |
| junk | 9.35 | 7.63 | 8.73 | 8.60 | 9.78 | 12.38 | 2.63 | 0.00 |


## Per-class intra-class spread (angle to own medoid, deg)


| class | n_polygons | mean_deg | p50_deg | p90_deg |
| --- | --- | --- | --- | --- |
| olivine | 681 | 2.41 | 1.84 | 5.14 |
| lcp | 848 | 2.96 | 1.77 | 6.52 |
| hcp | 599 | 1.96 | 1.52 | 3.32 |
| plagioclase | 507 | 1.77 | 1.58 | 2.87 |
| alteration | 76 | 4.07 | 2.24 | 10.14 |
| bland_dust | 8 | 0.36 | 0.22 | 0.92 |
| bland_reject | 243 | 8.08 | 6.62 | 19.13 |
| junk | 105 | 6.70 | 5.11 | 10.69 |


## Bland & junk (diagnostic classes)

`bland_dust`, `bland_reject` and `junk` are catch-all classes, not
mineral endmembers, so their purity is DIAGNOSTIC rather than a quality
gate. The interesting signal is which class each hugs spectrally (the
nearest-other-class column below). `junk` is single-source (`tag`), so
cross-source coherence is n/a for it. Note also that `bland_dust`
(hand dust-tiles) uses a constant polygon_id (0): with the (class,
source, tile_id, polygon_id) key this collapses to one huge tile-mean
polygon per tile — expected, and it does not affect the medoid math.


### `junk` nearest-other-class distribution (116 polygons)


| nearest_other_class | n_polygons | pct |
| --- | --- | --- |
| bland_reject | 92 | 79.31 |
| lcp | 9 | 7.76 |
| hcp | 7 | 6.03 |
| olivine | 4 | 3.45 |
| plagioclase | 2 | 1.72 |
| alteration | 2 | 1.72 |


### `bland_dust` nearest-other-class distribution (8 polygons)


| nearest_other_class | n_polygons | pct |
| --- | --- | --- |
| alteration | 8 | 100.00 |


### `bland_reject` nearest-other-class distribution (251 polygons)


| nearest_other_class | n_polygons | pct |
| --- | --- | --- |
| junk | 176 | 70.12 |
| bland_dust | 53 | 21.12 |
| lcp | 11 | 4.38 |
| plagioclase | 9 | 3.59 |
| olivine | 1 | 0.40 |
| hcp | 1 | 0.40 |



## Suspect polygons (negative margin)

Total suspects: 1752 / 3160 polygons (55.4%).


Suspects by class:


| class | n_suspect | n_polygons |
| --- | --- | --- |
| alteration | 28 | 84 |
| bland_dust | 0 | 8 |
| bland_reject | 192 | 251 |
| hcp | 418 | 603 |
| junk | 47 | 116 |
| lcp | 414 | 872 |
| olivine | 350 | 707 |
| plagioclase | 303 | 519 |


### Top-20 worst-margin suspects (with provenance)


| class | source | tile_id | polygon_id | n_px | confidence_weight | own_angle_deg | nearest_other_class | nearest_other_angle_deg | margin_deg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bland_reject | reassigned | t1101 | 3112597758 | 436 | 1.00 | 10.69 | bland_dust | 0.70 | -9.99 |
| bland_reject | reassigned | t1028 | 423515327 | 52 | 1.00 | 10.48 | bland_dust | 0.56 | -9.92 |
| bland_reject | reassigned | t1316 | 4123051822 | 1084 | 1.00 | 10.79 | bland_dust | 0.86 | -9.92 |
| bland_reject | reassigned | t1029 | 1764188128 | 170 | 1.00 | 10.39 | bland_dust | 0.51 | -9.89 |
| bland_reject | reassigned | t1100 | 638916467 | 741 | 1.00 | 10.56 | bland_dust | 0.72 | -9.83 |
| bland_reject | reassigned | t1028 | 638435218 | 66 | 1.00 | 12.45 | bland_dust | 2.62 | -9.83 |
| bland_reject | reassigned | t1388 | 3812673900 | 1859 | 1.00 | 10.48 | bland_dust | 0.65 | -9.83 |
| bland_reject | reassigned | t1316 | 3522647741 | 91 | 1.00 | 11.00 | bland_dust | 1.18 | -9.82 |
| bland_reject | reassigned | t1100 | 1942545568 | 405 | 1.00 | 10.85 | bland_dust | 1.05 | -9.80 |
| bland_reject | reassigned | t1028 | 2506744414 | 212 | 1.00 | 10.14 | bland_dust | 0.39 | -9.75 |
| bland_reject | reassigned | t1028 | 1027517789 | 116 | 1.00 | 10.14 | bland_dust | 0.41 | -9.73 |
| bland_reject | reassigned | t1173 | 3948553120 | 374 | 1.00 | 10.18 | bland_dust | 0.46 | -9.72 |
| bland_reject | reassigned | t1316 | 443655571 | 284 | 1.00 | 10.53 | bland_dust | 0.82 | -9.71 |
| bland_reject | reassigned | t1030 | 2645620482 | 393 | 1.00 | 10.07 | bland_dust | 0.36 | -9.71 |
| bland_reject | reassigned | t1100 | 2028014889 | 9154 | 1.00 | 10.37 | bland_dust | 0.66 | -9.71 |
| bland_reject | reassigned | t1316 | 3216383249 | 1615 | 1.00 | 10.25 | bland_dust | 0.55 | -9.70 |
| bland_reject | reassigned | t1100 | 896449925 | 1045 | 1.00 | 10.57 | bland_dust | 0.89 | -9.68 |
| bland_reject | reassigned | t1028 | 2724471204 | 297 | 1.00 | 10.07 | bland_dust | 0.41 | -9.65 |
| bland_reject | reassigned | t1028 | 2769586980 | 82 | 1.00 | 10.14 | bland_dust | 0.50 | -9.64 |
| bland_reject | reassigned | t1030 | 896309542 | 288 | 1.00 | 9.98 | bland_dust | 0.36 | -9.62 |

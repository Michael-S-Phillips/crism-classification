# Label Quantification — SAM Endmember Analysis

Analysis window: 57 bands (m2..m58, 534-2457 nm), raw reflectance.

min_px = 10. Angles in DEGREES. Class-level math on polygon mean spectra, single-label only.

Runtime: 15.9 s.


## Interpretation caveat

1 um VNIR/IR detector-overlap band exclusion: ON. Bands with
wavelength in [1000-1065] nm (m16,m17,m18,m19 — the 1021/1023 nm
junction duplicate pair and the high-variance ~1056 nm band) are
dropped BEFORE L2 normalization in ALL angle math, so they contribute
nothing to the spectral angle. Endmember CSV still stores all 57 bands.

Raw-reflectance spectral angles are continuum/albedo-dominated (all
inter-class medoid angles come out <3 deg), so absolute suspect counts
are structurally inflated. Read margins as a RELATIVE worst-offenders
ranking, not a mislabel census. Continuum removal is the planned v2
that would make these angles mineralogical.


## Per-source purity (headline)

Count of polygons, median angle to own-class medoid, and % with negative margin (closer to another class).


| class | source | n_polygons | median_own_angle_deg | pct_suspect |
| --- | --- | --- | --- | --- |
| alteration | confirmed | 8 | 3.18 | 37.50 |
| alteration | tag | 76 | 2.11 | 36.84 |
| bland_dust | hand | 8 | 0.23 | 0.00 |
| bland_reject | reassigned | 251 | 6.58 | 77.29 |
| hcp | confirmed | 28 | 4.77 | 92.86 |
| hcp | hand | 575 | 1.34 | 69.22 |
| junk | tag | 116 | 4.28 | 41.38 |
| lcp | confirmed | 166 | 6.85 | 96.99 |
| lcp | hand | 706 | 1.32 | 31.16 |
| olivine | confirmed | 52 | 2.56 | 88.46 |
| olivine | hand | 562 | 1.42 | 54.09 |
| olivine | reassigned | 93 | 5.66 | 16.13 |
| plagioclase | hand | 519 | 1.35 | 58.38 |


## Cross-source coherence per class

Angle between per-source sub-medoids — quantifies source disagreement within a class.


| class | source_a | source_b | n_a | n_b | angle_deg |
| --- | --- | --- | --- | --- | --- |
| alteration | confirmed | tag | 5 | 71 | 1.28 |
| hcp | confirmed | hand | 26 | 573 | 4.60 |
| lcp | confirmed | hand | 153 | 695 | 6.10 |
| olivine | confirmed | reassigned | 50 | 82 | 5.29 |
| olivine | hand | reassigned | 549 | 82 | 5.09 |
| olivine | confirmed | hand | 50 | 549 | 2.39 |


## Inter-class medoid angle matrix (deg)


| class | olivine | lcp | hcp | plagioclase | alteration | bland_dust | bland_reject | junk |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| olivine | 0.00 | 2.50 | 0.89 | 0.97 | 2.40 | 3.82 | 7.01 | 9.30 |
| lcp | 2.50 | 0.00 | 1.83 | 1.84 | 2.99 | 5.41 | 5.21 | 7.66 |
| hcp | 0.89 | 1.83 | 0.00 | 0.86 | 2.48 | 4.50 | 6.27 | 8.58 |
| plagioclase | 0.97 | 1.84 | 0.86 | 0.00 | 2.47 | 4.01 | 6.52 | 8.90 |
| alteration | 2.40 | 2.99 | 2.48 | 2.47 | 0.00 | 3.72 | 7.38 | 9.76 |
| bland_dust | 3.82 | 5.41 | 4.50 | 4.01 | 3.72 | 0.00 | 10.23 | 12.66 |
| bland_reject | 7.01 | 5.21 | 6.27 | 6.52 | 7.38 | 10.23 | 0.00 | 2.61 |
| junk | 9.30 | 7.66 | 8.58 | 8.90 | 9.76 | 12.66 | 2.61 | 0.00 |


## Per-class intra-class spread (angle to own medoid, deg)


| class | n_polygons | mean_deg | p50_deg | p90_deg |
| --- | --- | --- | --- | --- |
| olivine | 681 | 2.23 | 1.66 | 4.88 |
| lcp | 848 | 2.81 | 1.56 | 6.62 |
| hcp | 599 | 1.80 | 1.37 | 3.13 |
| plagioclase | 507 | 1.59 | 1.35 | 2.90 |
| alteration | 76 | 4.10 | 2.21 | 10.41 |
| bland_dust | 8 | 0.35 | 0.23 | 0.86 |
| bland_reject | 243 | 8.09 | 6.68 | 19.96 |
| junk | 105 | 6.36 | 4.61 | 10.75 |


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
| bland_reject | 91 | 78.45 |
| lcp | 9 | 7.76 |
| hcp | 8 | 6.90 |
| olivine | 5 | 4.31 |
| alteration | 2 | 1.72 |
| plagioclase | 1 | 0.86 |


### `bland_dust` nearest-other-class distribution (8 polygons)


| nearest_other_class | n_polygons | pct |
| --- | --- | --- |
| alteration | 8 | 100.00 |


### `bland_reject` nearest-other-class distribution (251 polygons)


| nearest_other_class | n_polygons | pct |
| --- | --- | --- |
| junk | 177 | 70.52 |
| bland_dust | 50 | 19.92 |
| lcp | 11 | 4.38 |
| plagioclase | 8 | 3.19 |
| alteration | 3 | 1.20 |
| olivine | 1 | 0.40 |
| hcp | 1 | 0.40 |



## Suspect polygons (negative margin)

Total suspects: 1746 / 3160 polygons (55.3%).


Suspects by class:


| class | n_suspect | n_polygons |
| --- | --- | --- |
| alteration | 31 | 84 |
| bland_dust | 0 | 8 |
| bland_reject | 194 | 251 |
| hcp | 424 | 603 |
| junk | 48 | 116 |
| lcp | 381 | 872 |
| olivine | 365 | 707 |
| plagioclase | 303 | 519 |


### Top-20 worst-margin suspects (with provenance)


| class | source | tile_id | polygon_id | n_px | confidence_weight | own_angle_deg | nearest_other_class | nearest_other_angle_deg | margin_deg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bland_reject | reassigned | t1028 | 1257586822 | 76 | 1.00 | 10.49 | bland_dust | 0.34 | -10.16 |
| bland_reject | reassigned | t1100 | 1942545568 | 405 | 1.00 | 11.08 | bland_dust | 0.93 | -10.15 |
| bland_reject | reassigned | t1101 | 3112597758 | 436 | 1.00 | 10.82 | bland_dust | 0.68 | -10.14 |
| bland_reject | reassigned | t1028 | 423515327 | 52 | 1.00 | 10.65 | bland_dust | 0.51 | -10.14 |
| bland_reject | reassigned | t1100 | 896449925 | 1045 | 1.00 | 10.81 | bland_dust | 0.68 | -10.12 |
| bland_reject | reassigned | t1100 | 638916467 | 741 | 1.00 | 10.77 | bland_dust | 0.65 | -10.12 |
| bland_reject | reassigned | t1100 | 2028014889 | 9154 | 1.00 | 10.59 | bland_dust | 0.48 | -10.11 |
| bland_reject | reassigned | t1028 | 2506744414 | 212 | 1.00 | 10.32 | bland_dust | 0.22 | -10.10 |
| bland_reject | reassigned | t1316 | 4123051822 | 1084 | 1.00 | 10.96 | bland_dust | 0.86 | -10.10 |
| bland_reject | reassigned | t1029 | 1764188128 | 170 | 1.00 | 10.59 | bland_dust | 0.50 | -10.09 |
| bland_reject | reassigned | t1028 | 2769586980 | 82 | 1.00 | 10.32 | bland_dust | 0.25 | -10.07 |
| bland_reject | reassigned | t1028 | 1027517789 | 116 | 1.00 | 10.30 | bland_dust | 0.23 | -10.07 |
| bland_reject | reassigned | t1388 | 3812673900 | 1859 | 1.00 | 10.67 | bland_dust | 0.62 | -10.05 |
| bland_reject | reassigned | t1316 | 443655571 | 284 | 1.00 | 10.72 | bland_dust | 0.67 | -10.05 |
| bland_reject | reassigned | t1028 | 2724471204 | 297 | 1.00 | 10.23 | bland_dust | 0.19 | -10.05 |
| bland_reject | reassigned | t1316 | 3216383249 | 1615 | 1.00 | 10.46 | bland_dust | 0.42 | -10.04 |
| bland_reject | reassigned | t1316 | 3522647741 | 91 | 1.00 | 11.13 | bland_dust | 1.10 | -10.03 |
| bland_reject | reassigned | t1028 | 638435218 | 66 | 1.00 | 12.65 | bland_dust | 2.63 | -10.02 |
| bland_reject | reassigned | t1101 | 1939191300 | 234 | 1.00 | 10.36 | bland_dust | 0.38 | -9.98 |
| bland_reject | reassigned | t1173 | 3948553120 | 374 | 1.00 | 10.32 | bland_dust | 0.37 | -9.95 |

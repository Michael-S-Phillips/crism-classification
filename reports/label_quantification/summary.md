# Label Quantification — SAM Endmember Analysis

Analysis window: 57 bands (m2..m58, 534-2457 nm), raw reflectance.

min_px = 10. Angles in DEGREES. Class-level math on polygon mean spectra, single-label only.

Runtime: 4.7 s.


## Per-source purity (headline)

Count of polygons, median angle to own-class medoid, and % with negative margin (closer to another class).


| class | source | n_polygons | median_own_angle_deg | pct_suspect |
| --- | --- | --- | --- | --- |
| alteration | confirmed | 8 | 3.57 | 0.00 |
| alteration | tag | 76 | 2.15 | 36.84 |
| hcp | confirmed | 28 | 4.76 | 92.86 |
| hcp | hand | 575 | 1.48 | 68.70 |
| lcp | confirmed | 166 | 6.76 | 0.00 |
| lcp | hand | 706 | 1.54 | 34.42 |
| olivine | confirmed | 52 | 2.67 | 88.46 |
| olivine | hand | 562 | 1.63 | 49.11 |
| olivine | reassigned | 93 | 5.82 | 16.13 |
| plagioclase | hand | 519 | 1.59 | 58.57 |


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


| class | olivine | lcp | hcp | plagioclase | alteration |
| --- | --- | --- | --- | --- | --- |
| olivine | 0.00 | 2.52 | 0.88 | 1.20 | 2.34 |
| lcp | 2.52 | 0.00 | 1.96 | 1.55 | 2.98 |
| hcp | 0.88 | 1.96 | 0.00 | 1.01 | 2.42 |
| plagioclase | 1.20 | 1.55 | 1.01 | 0.00 | 2.42 |
| alteration | 2.34 | 2.98 | 2.42 | 2.42 | 0.00 |


## Per-class intra-class spread (angle to own medoid, deg)


| class | n_polygons | mean_deg | p50_deg | p90_deg |
| --- | --- | --- | --- | --- |
| olivine | 681 | 2.87 | 1.85 | 5.34 |
| lcp | 848 | 3.04 | 1.77 | 6.53 |
| hcp | 599 | 2.35 | 1.54 | 3.47 |
| plagioclase | 507 | 1.98 | 1.58 | 2.94 |
| alteration | 76 | 4.07 | 2.24 | 10.14 |


## Suspect polygons (negative margin)

Total suspects: 1333 / 2785 polygons (47.9%).


Suspects by class:


| class | n_suspect | n_polygons |
| --- | --- | --- |
| alteration | 28 | 84 |
| hcp | 421 | 603 |
| lcp | 243 | 872 |
| olivine | 337 | 707 |
| plagioclase | 304 | 519 |


### Top-20 worst-margin suspects (with provenance)


| class | source | tile_id | polygon_id | n_px | confidence_weight | own_angle_deg | nearest_other_class | nearest_other_angle_deg | margin_deg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| alteration | tag | t1159 | 1600979008 | 7 | 0.50 | 13.79 | lcp | 11.39 | -2.39 |
| alteration | tag | t1160 | 1248938873 | 12 | 0.50 | 10.55 | lcp | 8.16 | -2.39 |
| alteration | tag | t1160 | 1192594267 | 9 | 0.50 | 11.36 | lcp | 9.01 | -2.35 |
| alteration | tag | t1160 | 3804725248 | 11 | 0.50 | 10.35 | lcp | 8.01 | -2.34 |
| alteration | tag | t1160 | 82877845 | 46 | 0.50 | 11.89 | lcp | 9.57 | -2.33 |
| alteration | tag | t1160 | 226399413 | 482 | 0.50 | 11.38 | lcp | 9.06 | -2.32 |
| alteration | tag | t1159 | 801889504 | 624 | 0.50 | 14.94 | lcp | 12.62 | -2.32 |
| alteration | tag | t1232 | 3232856689 | 73 | 0.50 | 4.81 | lcp | 2.50 | -2.30 |
| alteration | tag | t1160 | 4147629625 | 51 | 0.50 | 11.48 | lcp | 9.19 | -2.29 |
| alteration | tag | t1160 | 1632260864 | 2216 | 0.50 | 11.43 | lcp | 9.18 | -2.25 |
| alteration | tag | t1159 | 148205312 | 489 | 0.50 | 15.23 | lcp | 13.00 | -2.23 |
| alteration | tag | t1090 | 2945707304 | 62 | 0.50 | 5.20 | lcp | 3.11 | -2.09 |
| lcp | hand | t0433 | 375 | 10 | 1.00 | 5.09 | olivine | 3.00 | -2.09 |
| olivine | hand | t0360 | 26 | 540 | 1.00 | 37.50 | lcp | 35.46 | -2.04 |
| lcp | hand | t0433 | 343 | 10 | 1.00 | 4.43 | olivine | 2.44 | -1.99 |
| olivine | hand | t0290 | 72 | 286 | 1.00 | 3.66 | lcp | 1.68 | -1.98 |
| olivine | hand | t0433 | 423 | 45 | 1.00 | 4.95 | lcp | 3.00 | -1.95 |
| olivine | hand | t0361 | 17 | 529 | 1.00 | 3.36 | lcp | 1.42 | -1.95 |
| olivine | hand | t0360 | 13 | 411 | 1.00 | 36.43 | lcp | 34.50 | -1.93 |
| lcp | hand | t0433 | 287 | 22 | 1.00 | 4.66 | olivine | 2.73 | -1.93 |

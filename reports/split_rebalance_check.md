# Split rebalance verification

_Generated: 2026-07-09 08:30:52 by `scripts/verify_split_rebalance.py`_

- Base parquet: `/mnt/mrdr/crism_classification/data/mrral_pixels.parquet`
- Confirmed dirs: `/mnt/mrdr/crism_classification/data/mc13_review/confirmed_pixels`, `/mnt/mrdr/crism_classification/data/mc13_review_7cls_v3/confirmed_pixels`
- Hard-negative dirs: `/mnt/mrdr/crism_classification/data/mc13_review/hard_negatives`, `/mnt/mrdr/crism_classification/data/mc13_review_7cls_v3/hard_negatives`
- LINK_DEG = 0.25, MIN_HOLDOUT_FRAC = 0.05, targets = 70/15/15

## Union assertions (min-holdout + zero leakage): **PASS**

Every class with >0 pixels holds >=5% of its pixels in both val and test, and no class has a val/train same-class polygon pair within 0.25 deg.

The union below is JOINTLY RE-SPLIT (`b7._joint_resplit`), mirroring `build_7cls_dataset.main()`; the per-source splits in the detail section are provisional diagnostics.

## Union (joint re-split) — positive-pixel counts per split

| class | train | val | test | total |
| --- | --- | --- | --- | --- |
| olivine_t1 | 461,533 | 98,903 | 98,906 | 659,342 |
| olivine_t2 | 92,321 | 19,792 | 19,793 | 131,906 |
| lcp | 500,283 | 107,206 | 107,210 | 714,699 |
| hcp | 472,747 | 101,310 | 101,306 | 675,363 |
| plagioclase | 247,103 | 52,960 | 52,960 | 353,023 |
| bland | 629,992 | 135,003 | 135,005 | 900,000 |
| alteration | 186,306 | 39,923 | 39,930 | 266,159 |
| junk | 121,266 | 27,332 | 28,917 | 177,515 |

## Union (joint re-split) — achieved split fractions

| class | train | val | test |
| --- | --- | --- | --- |
| olivine_t1 | 0.700 | 0.150 | 0.150 |
| olivine_t2 | 0.700 | 0.150 | 0.150 |
| lcp | 0.700 | 0.150 | 0.150 |
| hcp | 0.700 | 0.150 | 0.150 |
| plagioclase | 0.700 | 0.150 | 0.150 |
| bland | 0.700 | 0.150 | 0.150 |
| alteration | 0.700 | 0.150 | 0.150 |
| junk | 0.683 | 0.154 | 0.163 |

## Cross-source leakage (val/train same-class polygon pairs within 0.25 deg)

Computed on the jointly re-split union: polygons from different sources within LINK_DEG of each other share a geographic unit, hence a split. Asserted zero.

| class | leaking_pairs | val_polys | train_polys |
| --- | --- | --- | --- |
| olivine_t1 | 0 | 85 | 475 |
| olivine_t2 | 0 | 45 | 250 |
| lcp | 0 | 93 | 778 |
| hcp | 0 | 79 | 616 |
| plagioclase | 0 | 38 | 453 |
| bland | 0 | 75 | 228 |
| alteration | 0 | 72 | 199 |
| junk | 0 | 18 | 115 |

**Total leaking val/train pairs across all classes: 0**

## Per-source detail (provisional per-source splits, overridden by the joint re-split)

### base (2,043,036 rows)

Counts:

| class | train | val | test | total |
| --- | --- | --- | --- | --- |
| olivine_t1 | 62,398 | 13,370 | 13,695 | 89,463 |
| olivine_t2 | 92,313 | 19,807 | 19,786 | 131,906 |
| lcp | 382,153 | 81,906 | 81,904 | 545,963 |
| hcp | 303,997 | 65,152 | 65,156 | 434,305 |
| plagioclase | 246,472 | 53,665 | 52,886 | 353,023 |
| bland | 222,624 | 38,713 | 38,663 | 300,000 |
| alteration | 79,747 | 15,909 | 16,169 | 111,825 |
| junk | 0 | 0 | 0 | 0 |

Fractions:

| class | train | val | test |
| --- | --- | --- | --- |
| olivine_t1 | 0.697 | 0.149 | 0.153 |
| olivine_t2 | 0.700 | 0.150 | 0.150 |
| lcp | 0.700 | 0.150 | 0.150 |
| hcp | 0.700 | 0.150 | 0.150 |
| plagioclase | 0.698 | 0.152 | 0.150 |
| bland | 0.742 | 0.129 | 0.129 |
| alteration | 0.713 | 0.142 | 0.145 |
| junk | 0.000 | 0.000 | 0.000 |

### confirmed (885,894 rows)

Counts:

| class | train | val | test | total |
| --- | --- | --- | --- | --- |
| olivine_t1 | 372,135 | 79,935 | 79,599 | 531,669 |
| olivine_t2 | 0 | 0 | 0 | 0 |
| lcp | 117,104 | 25,101 | 26,531 | 168,736 |
| hcp | 162,736 | 34,788 | 34,940 | 232,464 |
| plagioclase | 0 | 0 | 0 | 0 |
| bland | 0 | 0 | 0 | 0 |
| alteration | 30,757 | 5,504 | 5,816 | 42,077 |
| junk | 0 | 0 | 0 | 0 |

Fractions:

| class | train | val | test |
| --- | --- | --- | --- |
| olivine_t1 | 0.700 | 0.150 | 0.150 |
| olivine_t2 | 0.000 | 0.000 | 0.000 |
| lcp | 0.694 | 0.149 | 0.157 |
| hcp | 0.700 | 0.150 | 0.150 |
| plagioclase | 0.000 | 0.000 | 0.000 |
| bland | 0.000 | 0.000 | 0.000 |
| alteration | 0.731 | 0.131 | 0.138 |
| junk | 0.000 | 0.000 | 0.000 |

### reassigned (38,210 rows)

Counts:

| class | train | val | test | total |
| --- | --- | --- | --- | --- |
| olivine_t1 | 17,238 | 15,254 | 5,718 | 38,210 |
| olivine_t2 | 0 | 0 | 0 | 0 |
| lcp | 0 | 0 | 0 | 0 |
| hcp | 0 | 8,493 | 101 | 8,594 |
| plagioclase | 0 | 0 | 0 | 0 |
| bland | 0 | 0 | 0 | 0 |
| alteration | 0 | 0 | 0 | 0 |
| junk | 0 | 0 | 0 | 0 |

Fractions:

| class | train | val | test |
| --- | --- | --- | --- |
| olivine_t1 | 0.451 | 0.399 | 0.150 |
| olivine_t2 | 0.000 | 0.000 | 0.000 |
| lcp | 0.000 | 0.000 | 0.000 |
| hcp | 0.000 | 0.988 | 0.012 |
| plagioclase | 0.000 | 0.000 | 0.000 |
| bland | 0.000 | 0.000 | 0.000 |
| alteration | 0.000 | 0.000 | 0.000 |
| junk | 0.000 | 0.000 | 0.000 |

### mc13_bland (300,000 rows)

Counts:

| class | train | val | test | total |
| --- | --- | --- | --- | --- |
| olivine_t1 | 0 | 0 | 0 | 0 |
| olivine_t2 | 0 | 0 | 0 | 0 |
| lcp | 0 | 0 | 0 | 0 |
| hcp | 0 | 0 | 0 | 0 |
| plagioclase | 0 | 0 | 0 | 0 |
| bland | 209,991 | 45,005 | 45,004 | 300,000 |
| alteration | 0 | 0 | 0 | 0 |
| junk | 0 | 0 | 0 | 0 |

Fractions:

| class | train | val | test |
| --- | --- | --- | --- |
| olivine_t1 | 0.000 | 0.000 | 0.000 |
| olivine_t2 | 0.000 | 0.000 | 0.000 |
| lcp | 0.000 | 0.000 | 0.000 |
| hcp | 0.000 | 0.000 | 0.000 |
| plagioclase | 0.000 | 0.000 | 0.000 |
| bland | 0.700 | 0.150 | 0.150 |
| alteration | 0.000 | 0.000 | 0.000 |
| junk | 0.000 | 0.000 | 0.000 |

### mc11_bland (300,000 rows)

Counts:

| class | train | val | test | total |
| --- | --- | --- | --- | --- |
| olivine_t1 | 0 | 0 | 0 | 0 |
| olivine_t2 | 0 | 0 | 0 | 0 |
| lcp | 0 | 0 | 0 | 0 |
| hcp | 0 | 0 | 0 | 0 |
| plagioclase | 0 | 0 | 0 | 0 |
| bland | 208,835 | 45,144 | 46,021 | 300,000 |
| alteration | 0 | 0 | 0 | 0 |
| junk | 0 | 0 | 0 | 0 |

Fractions:

| class | train | val | test |
| --- | --- | --- | --- |
| olivine_t1 | 0.000 | 0.000 | 0.000 |
| olivine_t2 | 0.000 | 0.000 | 0.000 |
| lcp | 0.000 | 0.000 | 0.000 |
| hcp | 0.000 | 0.000 | 0.000 |
| plagioclase | 0.000 | 0.000 | 0.000 |
| bland | 0.696 | 0.150 | 0.153 |
| alteration | 0.000 | 0.000 | 0.000 |
| junk | 0.000 | 0.000 | 0.000 |

### junk (177,515 rows)

Counts:

| class | train | val | test | total |
| --- | --- | --- | --- | --- |
| olivine_t1 | 0 | 0 | 0 | 0 |
| olivine_t2 | 0 | 0 | 0 | 0 |
| lcp | 0 | 0 | 0 | 0 |
| hcp | 0 | 0 | 0 | 0 |
| plagioclase | 0 | 0 | 0 | 0 |
| bland | 0 | 0 | 0 | 0 |
| alteration | 0 | 0 | 0 | 0 |
| junk | 118,503 | 30,095 | 28,917 | 177,515 |

Fractions:

| class | train | val | test |
| --- | --- | --- | --- |
| olivine_t1 | 0.000 | 0.000 | 0.000 |
| olivine_t2 | 0.000 | 0.000 | 0.000 |
| lcp | 0.000 | 0.000 | 0.000 |
| hcp | 0.000 | 0.000 | 0.000 |
| plagioclase | 0.000 | 0.000 | 0.000 |
| bland | 0.000 | 0.000 | 0.000 |
| alteration | 0.000 | 0.000 | 0.000 |
| junk | 0.668 | 0.170 | 0.163 |

### alteration (112,257 rows)

Counts:

| class | train | val | test | total |
| --- | --- | --- | --- | --- |
| olivine_t1 | 0 | 0 | 0 | 0 |
| olivine_t2 | 0 | 0 | 0 | 0 |
| lcp | 0 | 0 | 0 | 0 |
| hcp | 0 | 0 | 0 | 0 |
| plagioclase | 0 | 0 | 0 | 0 |
| bland | 0 | 0 | 0 | 0 |
| alteration | 78,417 | 16,995 | 16,845 | 112,257 |
| junk | 0 | 0 | 0 | 0 |

Fractions:

| class | train | val | test |
| --- | --- | --- | --- |
| olivine_t1 | 0.000 | 0.000 | 0.000 |
| olivine_t2 | 0.000 | 0.000 | 0.000 |
| lcp | 0.000 | 0.000 | 0.000 |
| hcp | 0.000 | 0.000 | 0.000 |
| plagioclase | 0.000 | 0.000 | 0.000 |
| bland | 0.000 | 0.000 | 0.000 |
| alteration | 0.699 | 0.151 | 0.150 |
| junk | 0.000 | 0.000 | 0.000 |


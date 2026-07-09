# Split rebalance verification

_Generated: 2026-07-08 18:35:47 by `scripts/verify_split_rebalance.py`_

- Base parquet: `/mnt/mrdr/crism_classification/data/mrral_pixels.parquet`
- Confirmed dirs: `/mnt/mrdr/crism_classification/data/mc13_review/confirmed_pixels`, `/mnt/mrdr/crism_classification/data/mc13_review_7cls_v3/confirmed_pixels`
- Hard-negative dirs: `/mnt/mrdr/crism_classification/data/mc13_review/hard_negatives`, `/mnt/mrdr/crism_classification/data/mc13_review_7cls_v3/hard_negatives`
- LINK_DEG = 0.25, MIN_HOLDOUT_FRAC = 0.05, targets = 70/15/15

## Union min-holdout assertion: **PASS**

Every class with >0 pixels holds >=5% of its pixels in both val and test.

## Union — positive-pixel counts per split

| class | train | val | test | total |
| --- | --- | --- | --- | --- |
| olivine_t1 | 457,441 | 101,727 | 100,174 | 659,342 |
| olivine_t2 | 92,313 | 19,807 | 19,786 | 131,906 |
| lcp | 499,257 | 107,007 | 108,435 | 714,699 |
| hcp | 466,708 | 108,432 | 100,223 | 675,363 |
| plagioclase | 246,472 | 53,665 | 52,886 | 353,023 |
| bland | 608,334 | 128,862 | 162,804 | 900,000 |
| alteration | 157,652 | 71,494 | 37,013 | 266,159 |
| junk | 118,503 | 30,095 | 28,917 | 177,515 |

## Union — achieved split fractions

| class | train | val | test |
| --- | --- | --- | --- |
| olivine_t1 | 0.694 | 0.154 | 0.152 |
| olivine_t2 | 0.700 | 0.150 | 0.150 |
| lcp | 0.699 | 0.150 | 0.152 |
| hcp | 0.691 | 0.161 | 0.148 |
| plagioclase | 0.698 | 0.152 | 0.150 |
| bland | 0.676 | 0.143 | 0.181 |
| alteration | 0.592 | 0.269 | 0.139 |
| junk | 0.668 | 0.170 | 0.163 |

## Cross-source leakage (val/train same-class polygon pairs within 0.25 deg)

Sources are split independently, so a same-class polygon in one source can land in val while a nearby same-class polygon in another source lands in train. Reported, not asserted.

| class | leaking_pairs | val_polys | train_polys |
| --- | --- | --- | --- |
| olivine_t1 | 0 | 84 | 428 |
| olivine_t2 | 0 | 81 | 209 |
| lcp | 0 | 178 | 694 |
| hcp | 0 | 192 | 492 |
| plagioclase | 0 | 45 | 428 |
| bland | 0 | 46 | 268 |
| alteration | 218 | 129 | 125 |
| junk | 0 | 2 | 131 |

**Total leaking val/train pairs across all classes: 218**

## Per-source detail

### base (2,043,036 rows)

Counts:

| class | train | val | test | total |
| --- | --- | --- | --- | --- |
| olivine_t1 | 62,398 | 13,370 | 13,695 | 89,463 |
| olivine_t2 | 92,313 | 19,807 | 19,786 | 131,906 |
| lcp | 382,153 | 81,906 | 81,904 | 545,963 |
| hcp | 303,997 | 65,152 | 65,156 | 434,305 |
| plagioclase | 246,472 | 53,665 | 52,886 | 353,023 |
| bland | 189,508 | 38,713 | 71,779 | 300,000 |
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
| bland | 0.632 | 0.129 | 0.239 |
| alteration | 0.713 | 0.142 | 0.145 |
| junk | 0.000 | 0.000 | 0.000 |

### confirmed (885,894 rows)

Counts:

| class | train | val | test | total |
| --- | --- | --- | --- | --- |
| olivine_t1 | 372,188 | 79,886 | 79,595 | 531,669 |
| olivine_t2 | 0 | 0 | 0 | 0 |
| lcp | 117,104 | 25,101 | 26,531 | 168,736 |
| hcp | 162,711 | 34,809 | 34,944 | 232,464 |
| plagioclase | 0 | 0 | 0 | 0 |
| bland | 0 | 0 | 0 | 0 |
| alteration | 7,560 | 30,186 | 4,331 | 42,077 |
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
| alteration | 0.180 | 0.717 | 0.103 |
| junk | 0.000 | 0.000 | 0.000 |

### reassigned (38,210 rows)

Counts:

| class | train | val | test | total |
| --- | --- | --- | --- | --- |
| olivine_t1 | 22,855 | 8,471 | 6,884 | 38,210 |
| olivine_t2 | 0 | 0 | 0 | 0 |
| lcp | 0 | 0 | 0 | 0 |
| hcp | 0 | 8,471 | 123 | 8,594 |
| plagioclase | 0 | 0 | 0 | 0 |
| bland | 0 | 0 | 0 | 0 |
| alteration | 0 | 0 | 0 | 0 |
| junk | 0 | 0 | 0 | 0 |

Fractions:

| class | train | val | test |
| --- | --- | --- | --- |
| olivine_t1 | 0.598 | 0.222 | 0.180 |
| olivine_t2 | 0.000 | 0.000 | 0.000 |
| lcp | 0.000 | 0.000 | 0.000 |
| hcp | 0.000 | 0.986 | 0.014 |
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
| alteration | 70,345 | 25,399 | 16,513 | 112,257 |
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
| alteration | 0.627 | 0.226 | 0.147 |
| junk | 0.000 | 0.000 | 0.000 |


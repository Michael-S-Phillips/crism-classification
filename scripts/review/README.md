# MC13 polygon review

Streamlit app for confirming/rejecting MC13 model-predicted polygons and
harvesting their interior pixels into an alternative training set.

## Run

```bash
conda run -n crism streamlit run scripts/review/app.py
```

Defaults (overridable in sidebar):
- gpkg dir   = `data/vector_mc13_relabeled`
- mrral dir  = `/mnt/mrdr/mc13`
- output dir = `data/mc13_review`

## Outputs

- `data/mc13_review/decisions.csv` — append-only ledger (source of truth).
- `data/mc13_review/confirmed_pixels.parquet` — schema matches `data/mrral_pixels.parquet`.
- `data/mc13_review/hard_negatives.parquet` — same schema + `negative_of` column.

## UI categories ↔ parquet columns

The "if rejected, actually:" dropdown maps UI-friendly names to the
existing `mrral_pixels.parquet` schema (which keeps its historical `other`
column name for downstream compatibility):

| Dropdown value | Where it lands | Schema effect |
|---|---|---|
| `(blank)` | `hard_negatives.parquet` | All labels 0; `negative_of` = predicted_class. |
| `olivine` | `hard_negatives.parquet` | `olivine_t1 = 1.0`; `negative_of` = blank. |
| `lcp` / `hcp` | `hard_negatives.parquet` | That column = 1.0; `negative_of` = blank. |
| `bland` | `hard_negatives.parquet` | `other = 1.0` (schema alias); `negative_of` = blank. |
| `alteration` | `hard_negatives.parquet` | All labels 0; `negative_of` = `"alteration"`. Use for clays / sulfates / opal / prehnite / chlorite and other 2.3-2.5 µm features the model commonly misreads as HCP. |
| `ambiguous` | `hard_negatives.parquet` | All labels 0; `negative_of` = `"ambiguous"`. |

`bland` is the UI name for the dust-dominated / featureless spectra harvested
from known dust regions. The schema column stays `other` because
`mrral_pixels.parquet` already uses that name.

## Consuming the outputs for training

When you concatenate the review parquets onto the existing training set,
the filter rules are:

1. **Positive training examples**
   - From `confirmed_pixels.parquet`: include every row. The active label
     column (one of `olivine_t1`, `lcp`, `hcp`, `other`) tells you the class.
   - From `hard_negatives.parquet`: include rows where `negative_of` is
     blank/null (these are reject-with-correction rows; the corrected
     mineral's label column is set). They are positive examples for the
     corrected class.

2. **Hard negatives**
   - From `hard_negatives.parquet`, rows where `negative_of in
     ('olivine', 'lcp', 'hcp', 'plagioclase')` → per-class hard negatives.
     Use them as "definitely NOT this class" examples for that class only.
   - From `hard_negatives.parquet`, rows where `negative_of == 'ambiguous'`
     → extra-strong negatives applicable to **every** class (we know it's
     not any of our minerals, but we don't know what it is). Useful for
     reducing overconfidence on out-of-distribution pixels.
   - From `hard_negatives.parquet`, rows where `negative_of == 'alteration'`
     → same as ambiguous (universal negative) but with a more specific
     provenance tag. Filter by this tag to extract the spectra that
     looked like HCP/LCP but were actually alteration minerals; they're
     a candidate seed pool for a future alteration-mineral classifier.

3. **Bland / dust class**
   - The `other` label column is populated by both confirmed bland pixels
     and reject-as-bland rows in hard_negatives. Either way: positive
     examples for the `other` class. There is no separate bland-as-negative
     concept in the schema.

4. **Pre-classifier masks (NOT done in this app)**
   - Shadows and CO₂ frost are masked **before** classification, not as
     review categories. Use a low-broadband-reflectance threshold for
     shadow, and the BD1435 mrrsu parameter for CO₂ ice. Those pixels
     never reach the model.

## Design

See `docs/superpowers/specs/2026-06-07-mc13-polygon-review-app-design.md`
and `docs/superpowers/plans/2026-06-07-mc13-polygon-review-app.md`.

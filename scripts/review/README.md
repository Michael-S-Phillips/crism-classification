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

## Design

See `docs/superpowers/specs/2026-06-07-mc13-polygon-review-app-design.md`
and `docs/superpowers/plans/2026-06-07-mc13-polygon-review-app.md`.

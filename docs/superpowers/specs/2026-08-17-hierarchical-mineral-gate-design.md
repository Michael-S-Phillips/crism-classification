# Hierarchical mineral-present gate — design

**Date:** 2026-08-17
**Status:** approved (design), awaiting spec review
**Creates:** `models/gated_classifier.py`, `training/gated_losses.py`,
`scripts/hpc_finetune_dualcr_gated.slurm`, `tests/test_gated_classifier.py`,
`tests/test_gated_losses.py`
**Touches:** `scripts/train.py` (one new `--gated_head` flag),
`training/train_torch.py` (model + loss selection)

---

## Problem

The classifier emits seven independent sigmoids. Nothing couples them, so
"this is lcp" and "this is featureless" are free to be asserted at once — and
they are. On Nili t1321 the e87 model puts mean `p_lcp = 0.996` and mean
`p_bland = 0.067` on the *same* pixels, and across the tile:

| `max(p_mineral) + max(p_non-mineral)` | value |
|---|---|
| median | 0.949 |
| p90 | 1.154 |
| max | **1.935** |
| share of valid px exceeding 1.0 | **35.4%** |

A pixel at 1.935 is simultaneously ~97% mineral and ~97% non-mineral. That is
not a calibration error, it is the absence of a constraint.

It also explains a measured oddity: `p_bland` separates dust-driven false lcp
from real lcp at **AUC 0.93**, yet sits at absolute values (0.087 vs 0.0067) far
below any usable threshold. The ranking information exists; the architecture has
no channel through which it can suppress the mineral head. The `--bland_gate`
shipped in `683d833` bolts that coupling on at vectorization time, after the fact
and at a cost of 51% of Nili lcp pixels. This learns it instead.

## The partition

Verified against `data/mrral_pixels.parquet` (2,619,784 rows), labels binary:

- **mineral** — olivine, lcp, hcp, plagioclase, alteration — 66.3% of rows
- **non-mineral** — bland, junk — 33.5%
- **contradictions (both): 0 rows.** The partition is clean in the data, not
  merely by intent.
- neither: 5,467 rows (0.21%) — unlabelled; see edge cases.

**18.8% of rows (492,585) carry two or more minerals simultaneously**, and
112,821 carry three. Multi-label co-occurrence within minerals is substantial and
must survive: olivine+hcp is a real assemblage, as noted when the expert rules
were designed. This rules out a softmax over the seven classes.

## Design

One gate logit and seven conditional logits from the same pooled embedding:

```
g   = sigmoid(z_gate)                     P(any mineral present)
c_k = sigmoid(z_k)                        P(class k | mineral present)
b_k = sigmoid(z_k)                        P(class k | no mineral present)

p_k = g * c_k          for k in {olivine, lcp, hcp, plagioclase, alteration}
p_k = (1 - g) * b_k    for k in {bland, junk}
```

Conditionals stay **independent sigmoids within each branch**, so olivine and hcp
can both be high given the gate — the co-occurrence above is preserved exactly.
Only the mineral/non-mineral split is made exclusive, which is the split the data
says is exclusive.

The constraint follows by construction: `p_mineral ≤ g` and `p_non-mineral ≤ 1-g`,
so `max(p_mineral) + max(p_non-mineral) ≤ 1`. The 35.4% of pixels currently
violating that become unrepresentable. To keep `p_lcp = 0.99` the model must
drive `g ≥ 0.99`, which forces `p_bland ≤ 0.01` — it has to choose.

Head shape: `head` becomes `Linear(272, 8)` — one gate logit plus seven
conditionals — against the current `Linear(272, 7)`. Everything upstream of the
head is untouched, so the encoder checkpoint stays interchangeable with every
other arm.

## Gate supervision

The gate receives gradient implicitly through `p_k`, but implicit-only training
lets it drift toward a constant while the conditionals absorb everything. Add an
explicit auxiliary term:

```
y_gate = 1 if any mineral label is positive else 0
L = L_main(p, y) + lambda_gate * BCE(g, y_gate)
```

with `lambda_gate = 1.0`. `y_gate` is derived, not a new annotation — the
partition above is exactly the derivation, and it is contradiction-free.

## Loss and numerics

`AsymmetricLoss` consumes logits and applies its own sigmoid. The gated head
produces **probabilities**, and `logit(g * c)` is unstable near both ends, so the
loss cannot simply be reused as-is.

`training/gated_losses.py` provides an ASL variant taking `p` directly, with the
identical formula (clip, `gamma_neg`, `gamma_pos`, class weights, per-sample
weights) so the only difference between this arm and its comparator is the head.
`p` is computed as `exp(logsigmoid(z_g) + logsigmoid(z_k))`, accurate in the
small-`p` tail where the naive product loses precision.

**This arm inherits `--asl_clip 0.0`** from the clip finding
(`tests/test_asl_clip_gradient.py`, `MODELS.md`). Running it at the old
`clip=0.05` would spend a 24-hour run reproducing a defect already measured.

## Relationship to the other arms

Three arms, one axis of variation each, all off the same data:

| arm | head | clip | tests |
|---|---|---|---|
| `dualcr_level` (e87) | flat | 0.05 | the existing baseline |
| `dualcr_noclip` | flat | **0.0** | does restoring the gradient fix persistence? |
| `dualcr_gated` | **gated** | 0.0 | does structural exclusivity add anything on top? |

`dualcr_gated` is read against `dualcr_noclip`, not against e87 — the clip
differs between gated and e87, so an e87 comparison would confound head and loss.

## Evaluation

Inherited from the hard-negatives spec so all arms share one yardstick:

1. **The constraint holds.** `max(p_mineral) + max(p_non-mineral) ≤ 1` for every
   valid pixel, up to float tolerance. This is an assertion about the
   implementation, checkable on the first tile classified — if it fails, the head
   is wired wrong.
2. **t1321 false share:** of pixels at `lcp ≥ 0.99`, the fraction with
   LCPINDEX2 below tile p40. **Now 35%; target < 10%.**
3. **t1249 over-correction guard:** confident-lcp count must not fall more than
   ~15% on a genuinely LCP-rich tile.
4. **Floor test**, compared in **pixels retained**, not polygon counts — a
   subtractive change fragments regions and inflates polygon counts (measured:
   Nili lcp @0.50 went 1,675 → 3,622 under the bland gate while losing 51% of
   pixels).
5. **`--bland_gate` should become unnecessary.** If the learned gate works, the
   vectorizer gate at 0.03 should remove far fewer pixels than the 51% it
   currently removes on Nili lcp. That is the cleanest signal that the coupling
   moved from post-hoc filter into the model.

## Edge cases

- **The 5,467 unlabelled rows (0.21%)** get `y_gate = 0`, matching how the
  current loss already treats them (all classes negative). Consistent, and
  flagged rather than silently assumed — if the fraction ever grows, revisit,
  because asserting "no mineral" on unlabelled pixels is a real claim.
- **Hard negatives from the dust spec** are labelled bland, so `y_gate = 0`.
  They therefore train the gate directly, which is the mechanism by which the two
  interventions reinforce rather than merely coexist.
- **Checkpoint compatibility — verified, and it fails loudly.**
  `classify_tile_supervised._set_n_classes` infers the vocabulary from
  `head.weight.shape[0]` and ends in
  `raise ValueError(f'unsupported head size {n} (expected 5, 6, or 7)')`. An
  8-wide gated head therefore **aborts at load** rather than being misread as an
  8-class model — the dangerous outcome (silently inventing a class and shifting
  every downstream index) cannot happen.

  Inference needs a `--gated_head` flag that does two things: rebind the vocab to
  the seven real classes despite the 8-wide head, and apply the `g * c` /
  `(1-g) * b` composition before writing probabilities. Without the composition
  the npz would contain raw conditionals, which look like probabilities, sum
  plausibly, and are wrong. `MODELS.md` must record which checkpoints are gated,
  because the file name will not say so.

## Risks

- **The gate becomes a bottleneck.** One scalar now controls every mineral. If it
  saturates near 1 the architecture degenerates to the flat head with extra
  steps; if it saturates near 0 all minerals die at once. Monitor the gate's
  distribution during training, not only the loss.
- **A shared gate couples unrelated failures.** Suppressing dust false-lcp also
  suppresses everything else on those pixels. That is intended where the pixel
  really is non-mineral, and harmful where the gate is wrong — which is why
  criterion 3 exists.
- **Alteration is the awkward member.** It is grouped as a mineral, but altered
  terrain is often bright and dusty, so the gate may learn "bright ⇒ no mineral"
  and take alteration down with the dust. Watch alteration specifically in the
  floor test; MC11 is the altered/dusty probe.
- **This may be redundant with `dualcr_noclip`.** If restoring the gradient alone
  fixes the false positives, the gate adds complexity for nothing. The arm
  ordering above is what tells us; run noclip first and read it before judging
  the gate.

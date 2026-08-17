# Model Registry

Single source of truth for **kept** CRISM classifier checkpoints and the backbones
they warm-start from. One row per model that is a real result or a reuse candidate.
Dead sweep arms and architecture bake-offs are summarized in Tier 3, not listed
per-file.

**Maintenance:** update this file (and the wiki `Experiments & Results.md`) as the
*closing step* of every experiment — add the new checkpoint, its metrics, and its
floor/visual verdict. If it drifts, it's useless.

**Metrics caveat:** `val_mAP` numbers are **not comparable across data-build eras** —
splits, class vocab, and val composition changed (esp. the 2026-07-09 honest
unit-balanced split rebuild, which *lowered* honest val_mAP and added
`val_mAP_core` = mAP excluding junk). The **floor test** (8 fixed tiles: 4 Nili, 2 Argyre,
2 MC11 — see `scripts/floor_test.sh`)
is the only leakage-immune cross-era comparator. See `reports/floor_tests/`.

**Floor-test caveat — everything before 2026-08-17 ran with a buggy reader.** Until `82afe80`,
`classify_tile_supervised.load_tile` clipped physically impossible reflectance to 0.5 instead of
masking it, so inference saw values the model was never trained on (fine-tuning always masked
them). All eight floor-test tiles sit in the affected quadrants. Measured effect on the deployed
maps: **olivine −13% to −20% at 0.97–0.995**, pyroxene ~unchanged, and on the worst single tile
(t1389) *every* ≥0.99 olivine detection was artefact (84,371 px → 0). `dualcr_level_e87` is the
first floor test run with the fix; comparisons against any earlier floor test therefore confound
the model change with the reader change. To compare cleanly, re-run the older checkpoint rather
than trusting its stored summary.

Files live in `checkpoints/`. `_best.pt` = best on the stop metric; `_best_map.pt`
= best on plain `val_mAP`; `_last.pt` = final epoch.

---

## Naming convention (new checkpoints only)

Existing filenames are **not** renamed (175 files referenced across slurm/scripts —
renaming is risky churn). Going forward, name new runs:

```
ft_{Ncls}_{databuild}_{backbone}_{lr}[_variant]
```

e.g. `ft_7cls_v3b_denoiseMAE_lr001` = 7-class, v3b data build, denoising-MAE
backbone, encoder_lr_scale 0.01. A name should tell you what the model *is*.

---

> ## ⚠ Plagioclase val AP is INFLATED for every run using MTRDR patches
>
> **Audit 2026-08-08.** `train_torch` built the synthetic-patch TRAIN set with no
> `split` filter, so it served *every* row of the synth parquet — including the
> val and test rows that `--synth_val_*` simultaneously put into VAL. The model
> was validated on plagioclase patches it had trained on. Logs show
> `Concatenating 1817 synthetic plag patches into train set` against
> `Concatenating 109 synthetic plag val patches`, and the 109 are a subset of
> the 1,817.
>
> **Affected:** any run passing both `--synth_train_*` and `--synth_val_*` —
> `hpc_finetune_7cls_v3bland.slurm` (the 7-class champion's lineage),
> `hpc_finetune_7cls_reviewonly.slurm`, `hpc_finetune_pyx.slurm`, and the
> ablations that copy them. Read their `val_AP_plagioclase` as an **upper
> bound**, not a measurement — and note it still only reached ~0.148, so true
> plagioclase performance is *worse* than these numbers, not better.
>
> Fixed in `e83a827` (synth train now filters `split=='train'`), with a
> regression test in `tests/test_synthetic_plag.py`. Checkpoints were NOT
> re-run; the caveat stands rather than the numbers being corrected.
>
> **Related, and the likelier root problem:** a spectral-angle confusion matrix
> (`scripts/sam_confusion_matrix.py`) puts hand-labelled plagioclase at recall
> 0.29 (k=1) / 0.41 (k=5), confused with alteration and lcp at 0.22 each, while
> MTRDR plagioclase reaches 0.87 / **0.97** with a 1.09° self-angle. Allowing
> 5 endmembers rescues `junk` (0.50 → 0.81), so the method detects multi-modality
> when present — hand plagioclase does not recover, meaning it is genuinely
> entangled with other classes rather than several clean modes. Plag AP has sat
> at 0.069–0.152 across every architecture tried; mislabelled hand polygons
> explain that better than class imbalance or model capacity.

## Tier 1 — Milestone classifiers

### ft_review_mtrdr  — champion (5-class)
- file: `ft_review_mtrdr_best.pt`
- parent: `ft_bland_v3_lrscale0001_cont1` ← `spatial_mae_denoising_128d_6l`
- classes (5): olivine · lcp · hcp · plagioclase · bland
- val_mAP **0.7644** (stop=val_mAP) · Phase 8, review-augmented + MTRDR-plag patches
- floor: — · visual: MC13 clean; HCP noisier than v3-bland lineage (relabeled 9c/14r)
- status: **champion of the 5-class line**; no alteration/junk → misfires on altered MC11

### ft_bland_v3_lrscale0001_cont1  (5-class)
- file: `ft_bland_v3_lrscale0001_cont1_best.pt`
- parent: `ft_bland_v3_lrscale0001` ← `spatial_mae_denoising_128d_6l`
- classes (5): olivine · lcp · hcp · plagioclase · bland
- val_mAP **0.7262** (Phase 6 bland-relabel) · warm-started from lrscale0001 (+100 ep)
- floor: — · visual: **cleanest MC13 HCP (7 confirm / 0 reject)** — best HCP visuals on record
- status: superseded on val_mAP by review_mtrdr, but its **encoder is the reuse workhorse**
  (feeds the 7cls line). Extracted encoder = `cont1_encoder_only.pt`.

### ft_bland_v3_lrscale0001  (5-class)
- file: `ft_bland_v3_lrscale0001_best.pt` · val_mAP 0.7211 · parent `spatial_mae_denoising`
- status: predecessor of cont1; kept for lineage.

### ft_with_review  (5-class)
- file: `ft_with_review_best.pt` · val_mAP 0.7643
- status: first review-augmented FT; **regressed −2.9 pp mAP / −14 pp HCP** before the
  per-polygon-cap + MTRDR remediation that produced `ft_review_mtrdr`. Kept as the
  cautionary baseline.

### ft_v3_denoising_lrscale001  (5-class)
- file: `ft_v3_denoising_lrscale001_best.pt` · val_mAP ~0.60 (Phase 5, stratified split)
- status: earlier **production vector-product** checkpoint (`data/vector_mc*_v3_denoising/`).
  Superseded but still referenced by deployed MC11/MC13 products.

### ft_7cls_v3b_lrscale001  — 7-class winner (honest splits)
- file: `ft_7cls_v3b_lrscale001_best.pt`
- parent (encoder): `ft_bland_v3_lrscale0001_cont1`
- classes (7): olivine · lcp · hcp · plagioclase · bland · alteration · junk
- val_mAP_core **0.7954** @ep86 (stop=val_mAP_core; plain val_mAP 0.756) · honest unit-balanced splits
- per-class val AP: oliv 0.897 · lcp 0.838 · hcp 0.876 · plag 0.383 · alt 0.793 · bland 0.985 · junk 0.519
- floor (`v4honest_lrscale001`, 2026-07-13): **LCP → 0 polygons at Nili & Argyre** despite
  val 0.838 (two-population problem confirmed, not split geometry); plag improved 1→67 @Nili;
  olivine over-predicted (mafic→olivine collapse worsened); Argyre HCP cratered 187→2.
- status: best **raw-mrral** 7-class by val core, but **superseded as best-so-far by
  `ft_7cls_cr_lrscale0001`** (below) on the floor test — its LCP collapses to 0 OOD, which
  the CR representation fixes.

### ft_7cls_v3b_lrscale0001 / _lrscale01  (7-class, sweep arms)
- files: `ft_7cls_v3b_lrscale0001_best.pt` (core 0.7767 @ep104) ·
  `ft_7cls_v3b_lrscale01_best.pt` (core 0.7883 @ep6, then degraded)
- status: honest-splits recovery sweep siblings of the winner above.

### ft_7cls_cr_lrscale0001  — best of the single-CR era (7-class, continuum-removed)
- file: `ft_7cls_cr_lrscale0001_best.pt`  (classify with `--continuum_removed --brightness_aux --embed_dim 256`)
- representation: **continuum-removed mrral** (59-band upper-hull CR, fed UN-z-scored to
  preserve absolute band depth) + per-pixel brightness as aux (aux_dim=1); CR-native 256d encoder.
- classes (7): olivine · lcp · hcp · plagioclase · bland · alteration · junk
- val_mAP_core **0.560** / val_mAP 0.551 — **NOT comparable** to raw-mrral runs (CR changes the
  input distribution; low val is expected here, the floor test is the arbiter).
- per-class val AP: oliv 0.745 · lcp 0.412 · hcp 0.734 · plag 0.069 · alt 0.745 · bland 0.659 · junk 0.492
- floor (`cr_lrscale0001`, 2026-07-28): **Nili LCP RESTORED — 1,197 @0.50 → 513 @0.99** (every
  raw-mrral 7-class model collapsed LCP to 0 OOD; this is the first to fix the two-population
  collapse). Nili oliv 1,092→475, hcp 879→208, alt 1,027→1, plag 0. Argyre: **olivine diffuse**
  (425 @0.50 → 44 @0.99, peaks low) — the known CR weakness; lcp 312→8, hcp 743→15.
- deployed: full MC11 map at `data/vector_mc11_cr_lrscale0001/`.
- status: best-so-far as of 2026-07-30 (visual/floor judgment); **superseded 2026-08-14 by the
  dual-CR line below** — `ft_6cls_handcore_dualcr_pyx_level` is what ships the current maps.
  Kept as the entry that established CR as the right representation. Solves the LCP OOD collapse —
  the project's biggest open problem — at the cost of diffuse Argyre olivine. val_mAP is low and
  non-comparable; do not rank it against raw runs on val. (pyx-merge line under evaluation — see
  floor tests `pyx_lrscale001`/`pyx_lrscale0001`, not yet promoted.)

### ft_5cls_pyxalt_cr_*  — **REJECTED on floor test** (5-class, pyx merge + CR)
- files: `ft_5cls_pyxalt_cr_lrscale01_best.pt` (0.5923) · `_lrscale0001_best.pt` (0.5848)
  · `_lrscale001_best.pt` (0.5524).  Classify with
  `--continuum_removed --brightness_aux --embed_dim 256 --pyx_alt`.
- backbone: `spatial_mae_cr_denoising_256d_6l` (MAE on unlabeled tiles — no label leakage)
- classes (5, `--pyx_alt`): olivine · **pyx** (lcp+hcp merged) · plagioclase · other · alteration
- data: **base parquet only** (`mrral_pixels.parquet`) — hand labels, no review concat, no
  relabels, no MTRDR plag. Single-variable test of "does pyx+CR hold on hand labels alone?"
- val_mAP_core 0.5923 (best arm, encoder_lr_scale 0.1) — non-comparable across eras, and moot.
- floor (`pyxalt_cr_lrscale01`, 2026-08-08): **FAIL.** vs `ft_6cls_pyxcr_lrscale001_best`,
  four metrics moved >2× the wrong way — Argyre alteration 226→899 (4.0×), Argyre plag @0.99
  39→206 (5.3×), Nili alteration 635→1,399, Nili plag 1,564→2,943. Pyx floods Nili at
  2,824 @0.50 / 21.6 MB (known-bad "v2 flood" is 2,772 / 10.5 MB). Argyre plag should be ≈0,
  is 1,733. Argyre olivine peaks at 0.97 instead of 0.85–0.90.
- **diagnosis — probability saturation:** Argyre pyx polygon count *rises* with threshold
  (414 @0.50 → 1,505 @0.99). Thresholds should merge/remove regions, never create them; rising
  counts mean a near-1.0 probability field fragmenting. **Present in `ft_6cls_pyxcr` too** — a
  CR-pyx-family trait, not something `--pyx_alt` introduced.
- status: **rejected, do not promote.** Kept only as the negative result for the pyx+CR
  hand-labels-only hypothesis. Diagnose family-level saturation before running more pyx arms;
  lr_scale is not the variable that matters. Other two arms not floor-tested (val_mAP_core
  spread <0.04; this was not a near-miss). Report:
  `reports/floor_tests/pyxalt_cr_lrscale01/summary.md`.
- **caveat on this lineage:** all three arms first crashed on an incomplete
  `patch_cache_base_cr` (train-only). The dataset silently fell back to on-the-fly reads while
  `cache_is_cr` suppressed CR — see `c6e12fd`. Any earlier run pairing `--cache_is_cr` with an
  incomplete cache and *no* `--brightness_aux` would have trained on raw patches **silently**.

### The handcore lineage (parents of the dual-CR line)

Not previously registered; recorded here because the dual-CR entries below are
unreadable without them. Data build `mrral_pixels_7cls_handcore.parquet`
(2,753,758 rows / 128 tiles) — hand-cored labels, `weight scheme: level`.

- `ft_7cls_handcore_level_best.pt` — val_mAP_core **0.7219**, floor `handcore_level` (2026-08-10)
- `ft_7cls_handcore_reviewup_best.pt` — val_mAP_core **0.7239** @ep127, floor `handcore_reviewup`
  (2026-08-10). This is the run that hit the 24 h walltime and **saved nothing**; the
  checkpoint-on-improvement + resume machinery (`75007a2`, `bec520b`) was written in response,
  and the checkpoints now present are from the re-run.
- Both are 59-band single-CR (`--continuum_removed --brightness_aux --embed_dim 256`).
- **Nili HCP @0.50: 2,329 and 2,560** — i.e. the HCP flood that the dual-CR children inherit
  starts here. It is a property of this label set, not of the dual representation.

### ft_7cls_handcore_dualcr_level  — dual-CR, 7-class · **highest monitored metric on record**
- files: `ft_7cls_handcore_dualcr_level_best.pt` (ep87) · `_best_map.pt` (ep87, val_mAP 0.8413)
  · `_last.pt` (ep127) · `_resume.pt` (ep125, carries optimizer + scheduler + patience state)
- classify with `--continuum_removed --dual_cr --brightness_aux --embed_dim 256`
- backbone: `spatial_mae_dualcr_denoising_256d_6l_best.pt` (loaded clean — Missing [], Unexpected [])
- representation: **118-channel dual continuum removal** — hull-CR in channels 0–58 ⊕
  **linear-CR** (per-spectrum least-squares line) in 59–117, standardised per block
  (`hull_std` 0.070541, `linear_std` 0.172122, measured over 51,463 spectra); plus the
  per-pixel brightness scalar as aux (1 → 16-d MLP, so `head` is 7×**272** = 256+16).
  Motivation: hull-CR destroys alteration's broad 1–2 µm arch, which made alteration a false
  attractor; linear-CR preserves it. See `docs/superpowers/specs/` dual-CR design.
- data: train cache 1,988,871 × 7×7×118, val 392,113; + MTRDR plag 8,671 and
  extra_plag_roi 1,817 dual patches with brightness sidecars
- classes (7): olivine · lcp · hcp · plagioclase · bland · alteration · junk
- val_mAP_core **0.8796** @ep87 (stop=val_mAP_core, patience 40) · val_mAP 0.8413
- per-class val AP — **final epoch (127), NOT the best checkpoint.** The run only emits
  per-class AP in the wandb end-of-run summary, so no per-class numbers exist for ep87:
  oliv 0.987 · lcp 0.967 · hcp 0.830 · plag 0.769 · alt 0.635 · bland 0.968 · junk 0.591
- floor (`dualcr_level_e87`, 2026-08-17): **HCP FAIL** — Nili 2,179 @0.50 / 10.5 MB,
  Argyre 1,111 / 6.1 MB, against the known-bad v2 flood of 2,772 / 10.5 MB. All four other
  criteria PASS: Nili LCP alive (1,675 @0.50), Nili olivine confident (763 @0.99), Argyre
  olivine peaks @0.85 (1,427), Argyre plag 0 at every rung ≥0.95.
- vs `dualcr_level_e68` (same file, 19 epochs earlier): **down at 0.50, up at 0.99** in every
  class and region — Nili lcp @0.99 1,113→1,493, hcp 336→480, oliv 713→763. Nothing moved >2×.
  Sharper, but "more mass at 0.99" is not self-evidently better for a model that saturates.
- status: **not deployed.** Best monitored metric on record, but the HCP flood is unresolved and
  is a lineage property — diagnose it at the label-set level before spending another training arm.

### ft_6cls_handcore_dualcr_pyx_level  — dual-CR + pyx merge · **the currently deployed model**
- files: `ft_6cls_handcore_dualcr_pyx_level_best.pt` (ep34) · `_best_map.pt` (ep24, val_mAP 0.8134)
  · `_last.pt` (ep75) · `_resume.pt` (ep75)
- classify with `--continuum_removed --dual_cr --brightness_aux --embed_dim 256 --pyx`
- backbone / representation / data build: identical to `ft_7cls_handcore_dualcr_level` above,
  differing only in the vocabulary (`head` is 6×272).
- classes (6, `--pyx`): olivine · **pyx** (lcp+hcp merged) · plagioclase · bland · alteration · junk
- val_mAP_core **0.8458** @ep34 · val_mAP 0.8080. Trained to ep75 without improving on ep34.
- per-class val AP — again **final epoch (75), not ep34**: oliv 0.986 · pyx 0.961 ·
  plag 0.763 · alt 0.546 · bland 0.964 · junk 0.562
- floor (`dualcr_pyx_e34` + `dualcr_pyx_e34_hi`, 2026-08-14): Nili @0.99 oliv 692 · pyx 1,681 ·
  plag 9 · alt 32. The merge recovers pyroxene the split model loses at the lcp/hcp boundary
  (1,681 vs 1,113+336=1,449 on Nili).
- **saturation, confirmed:** on Nili, pyx counts are flat-to-RISING across the ladder
  (1,555 @0.50 → 1,806 @0.97 → 1,681 @0.99). Thresholds must never create regions. This is the
  same family trait already recorded for `ft_5cls_pyxalt_cr` and `ft_6cls_pyxcr` — dual-CR did
  **not** fix it. Practical consequence: the ladder only separates signal above ~0.999, which is
  why the deployed products use the extended ladder
  `[0.5, 0.85, 0.97, 0.99, 0.995, 0.999, 0.9995, 0.9999]` rather than the default 8-rung grid.
- deployed: **MC11 + MC13 + MC26**, 183 tiles. Corrected products (post-`PHYS_MAX` fix) at
  `data/mc_deploy_pyx_physmax/probs/` and `reports/mc_deploy_pyx_physmax/`; the superseded
  pre-fix run is retained at `data/mc_deploy_pyx/` + `reports/mc_deploy_pyx/` for comparison.
- status: **the model behind the current maps.** Preferred over the 7-class arm on visual
  judgment despite a lower monitored metric (0.8458 vs 0.8796) — note the two are **not
  directly comparable**, different vocabularies.

> ### ⚠ Plagioclase val AP jumped to ~0.77 in the handcore/dual-CR line — read it carefully
>
> Plag AP sat at **0.069–0.152** across every architecture before this line; both dual-CR runs
> report **~0.77**. This is *not* the `e83a827` synthetic-patch leak — that fix (2026-08-08)
> predates these runs and `train_torch.py:351` passes `split='train'`, so train and val synth
> patches are disjoint.
>
> The likelier reading is a **change of measurement, not of skill**: val plag positives now
> include MTRDR- and ROI-derived patches, which a spectral-angle test puts at recall 0.87–0.97
> versus 0.29–0.41 for hand-labelled plagioclase (see the audit box above). The logs also warn
> `AP[plagioclase]: no positive val examples` repeatedly, so the metric rests on sparse positives.
>
> The floor test is the check, and it disagrees with the val number: Nili plag **16 polygons
> @0.99**, Argyre **0 at every rung ≥0.95**. Treat plag ~0.77 as "finds MTRDR plag," not as a
> solved class.

### ft_6cls_mc11val_denoise  — best MC11-alteration model (6-class)
- file: `ft_6cls_mc11val_denoise_best.pt`
- backbone: **fresh** `spatial_mae_denoising_128d_6l` (--pretrain_ckpt, new head)
- classes (6): olivine · lcp · hcp · plagioclase · bland · alteration
- val_mAP **0.8138** on MC11-inclusive val (stop=val_mAP, best-ckpt follows val_AP_alteration)
- status: **0.8138 DEBUNKED (2026-07-14 MC11 visual test).** On the most-altered MC11
  tile (t1450) it predicts 69.5% olivine + 21% plag + **0.0% alteration** = the classic
  MC11 false-mineral failure. Val was leaky/inflated (pre-honest-splits). NOT a usable
  MC11 model. See [[reviewonly-and-mc11]].

### ft_6cls_mc11val_spend  (6-class)
- file: `ft_6cls_mc11val_spend_best.pt` · val_mAP 0.8061 · fresh `spatial_mae_spend_128d_6l`
- status: 2nd-best MC11-alteration; SPEND smoothing bias.

### ft_6cls_mc11val_champion  (6-class) — "warm-start champion → MC11" experiment
- file: `ft_6cls_mc11val_champion_best.pt`
- backbone: **warm-start `ft_review_mtrdr`** (--init_ckpt) + fresh 6th (alteration) head
- val_mAP **0.796** on MC11-inclusive val
- status: **this IS the "start from the 5-class champion and adapt to MC11" idea.**
  Underperformed both fresh-MAE arms (0.8138 / 0.8061) → warm-starting the champion
  classifier did *not* help vs a fresh head on the same backbone. Pre-honest-splits era.

### ft_6cls_purealt_ls001 / ls0001  (6-class)
- files: `ft_6cls_purealt_ls001_best.pt` (0.6412) · `ft_6cls_purealt_ls0001_best.pt` (0.6478)
- status: pure-alteration-focused variants; lower overall mAP.

### ft_plag_aware_relabeled(_mtrdr)  (5-class)
- files: `ft_plag_aware_relabeled_best.pt` (0.7589) · `ft_plag_aware_relabeled_mtrdr_best.pt` (0.7564)
- backbone: `plag_aware_mae_128d_6l` (multi-task plag-aware MAE) + olivine→HCP soft relabels
- status: plag-aware pretraining barely helped plag (encoder-limited finding).

### ft_mrrsu_aux(_zscore)  (5-class)
- files: `ft_mrrsu_aux_best.pt` (0.7408) · `ft_mrrsu_aux_zscore_best.pt` (0.7427)
- status: RPEAK1/BD1300 mrrsu aux-channel injection experiment (per-tile z-score norm variants).

---

## Tier 2 — Backbones / MAE encoders (warm-start sources)

| encoder | file | kind | note |
|---|---|---|---|
| **denoising MAE** | `spatial_mae_denoising_128d_6l_best.pt` | DenoisingSpatialSpectralMAE, ep160, loss 0.0116 | **workhorse** — feeds bland_v3, review_mtrdr, 7cls, mc11val_denoise |
| base spatial MAE | `spatial_mae_128d_6l_best.pt` | SpatialSpectralMAE, ep194, loss 0.016 | original 128d/6-layer encoder |
| continued spatial MAE | `spatial_mae_128d_6l_cont_epoch{200..350}.pt` | continued pretrain | longer-trained base |
| SPEND MAE | `spatial_mae_spend_128d_6l_best.pt` | SPEND, ep199, loss 0.0236 | smoothing bias; feeds mc11val_spend |
| plag-aware MAE | `plag_aware_mae_128d_6l_best.pt` | multi-task (recon+aux), ep40, monitor_plag_AP 0.899 | plag-aware pretraining |
| cont1 encoder | `cont1_encoder_only.pt` | encoder extracted from ft_bland_v3_cont1 | convenience export |
| CR denoising MAE (256d) | `spatial_mae_cr_denoising_256d_6l_best.pt` | DenoisingSpatialSpectralMAE, 256d/6L, **59-ch hull-CR** | feeds `ft_7cls_cr_*`, `ft_6cls_pyxcr_*`, `ft_5cls_pyxalt_cr_*` |
| **dual-CR denoising MAE** | `spatial_mae_dualcr_denoising_256d_6l_best.pt` | DenoisingSpatialSpectralMAE, 256d/6L/4-head, decoder 64d×2, mask 0.75, **118-ch dual-CR** (`n_channel_blocks=2`), ep**154** of 200, mae_loss **0.0469** | feeds BOTH dual-CR classifiers; epoch snapshots at 50/100/150/200 |

Injected pretrain noise for the dual-CR MAE: `sigma_gauss` 0.0087, `sigma_spike` 0.0058,
`sigma_column` 0.0049, spike centred on band 15 with FWHM 3 bands.

Encoder input width is **not** uniform: the original line is 128-dim / 59-band raw mrral; the
CR line is 256-dim / 59-channel hull-CR; the dual-CR line is 256-dim / **118**-channel. A
checkpoint only warm-starts a classifier of matching width — `band_embed.weight` is
`(embed_dim, n_channels)`, so read that tensor's shape if a checkpoint's lineage is unclear.

> **⚠ The dual-CR MAE was pretrained on unmasked blue-edge pixels.**
> `scripts/build_global_patch_cache.py:119–121` masks only `== 65535` and non-finite, then clips
> to `CLIP_MAX = 0.5`. It does **not** apply `PHYS_MAX = 1.0`, so a 410 nm blue-edge value of
> ~2400–3900 I/F entered the pretrain cache as a plausible-looking **0.5** rather than being
> masked to 0.0 the way `CRISMSpectralPatchDataset` does on the labeled path. Scope on deployed
> tiles: 832,731 px, 0.274%, in 159 of 183 tiles (`reports/phys_max_contamination.csv`).
> Fine-tuning and (since `82afe80`) inference both mask correctly; **pretraining still does not.**
> Rebuilding the global cache and re-pretraining is an open decision, not a done fix.

---

## Tier 3 — Archived exploration (not individually tracked)

Early architecture bake-offs and dead sweep arms, superseded by the SpatialSpectral
line above. Kept on disk for reproducibility; see wiki `Experiments & Results.md`
(Phases 1–5) for context. Families:

- `cnn_sw*`, `mlp_sw*`, `vit_sw*` — spectral-window architecture comparison (single-pixel era)
- `scnn_*`, `svit_*`, `shybrid_*` — spatial CNN / ViT / hybrid variants (ASL, focal, aug)
- `spvit_*`, `spvit_decomp_*`, `spvit_frozen_*` — SpatialViT lr/decomp/frozen sweeps
- `contrastive_plag_*` — contrastive-learning plag encoder experiments
- `mae_pretrain_128d_4l`, `spatial_mae_64d_2l`, `*_smoke`, `*UNKNOWN*` — early/smoke pretrains

---

*Last updated: 2026-08-17 — added the dual-CR line (encoder + both classifiers), the handcore parents, the
PHYS_MAX reader caveat, and the plagioclase-AP caveat. Previous: 2026-07-13 (initial registry).*

# Methods

*Draft — 2026-08-14. Numbers verified against the code and trained checkpoints;
provenance for each is noted in the accompanying comment blocks where it is not
obvious. Sections marked **[PENDING]** await runs still in progress.*

## 2.1 Data

We used Multispectral Reduced Data Records (MRDR) from the Compact
Reconnaissance Imaging Spectrometer for Mars (CRISM) aboard Mars Reconnaissance
Orbiter, obtained from the Planetary Data System Geosciences Node. MRDR products
are map-projected mosaics organised by Mars Chart (MC) quadrangle in a Mars 2000
equidistant-cylindrical projection (IAU sphere, semi-major axis 3 396 190 m,
inverse flattening 169.8944472). Each quadrangle contains 36 to 75 tiles of
approximately 1538 × 1636 pixels at 181.1 m per pixel.

Two co-registered products were used. The `mrral` product supplies calibrated
I/F reflectance; we retained a 59-band subset spanning 410.1 to 2456.8 nm,
excluding bands beyond 2.5 µm where thermal emission and low signal-to-noise
degrade the spectra. The `mrrsu` product supplies the 60 standard CRISM summary
parameters of Viviano-Beck et al. (2014) and was used only by the baselines
described in Section 2.7. A third product, `mrrde`, supplies observation geometry
and MOLA-derived elevation and was used only for the atmospheric diagnostic in
Section 2.9. All three are pixel-for-pixel co-registered within a tile, which we
verified rather than assumed.

Pixels flagged with the CRISM no-data sentinel (65535) or carrying reflectance
above 1.0 I/F were masked. The latter is not merely a sanity bound: band 0
(410 nm) carries a known blue-edge artefact reaching approximately 1180 I/F.
Retained reflectances were clipped to the interval [0, 0.5].

## 2.2 Labels and class vocabulary

Training labels were per-pixel polygon annotations drawn over `mrral` tiles by a
planetary spectroscopist, supplemented by two independent review campaigns in
which candidate detections were confirmed, rejected, or reclassified. Each
annotation carries a confidence tier of High, Moderate, or Low; review-derived
annotations carry a parallel Reviewed-High, Reviewed-Moderate, or Reviewed-Low
tier. Tiers enter training as per-pixel loss weights (Section 2.6).

Labelling is **multi-label rather than multi-class**: a pixel may carry several
mineral labels simultaneously, because mineral assemblages co-occur at the
~181 m pixel scale. Olivine-bearing basalt, for instance, legitimately carries
both olivine and pyroxene. This property constrains the architecture, the loss,
and the baselines, and we return to it repeatedly below.

Two vocabularies were evaluated. The seven-class vocabulary comprises olivine,
low-calcium pyroxene (LCP), high-calcium pyroxene (HCP), plagioclase, bland,
alteration, and junk. The six-class vocabulary merges the two pyroxenes into a
single `pyx` class, defined per pixel as the elementwise maximum of the LCP and
HCP labels. The merge is motivated in Section 2.7: the band parameters that
discriminate LCP from HCP differ only in the centre position of a single 2 µm
absorption, and that discrimination proves unreliable. The `bland` class denotes
spectrally featureless ground and `junk` denotes artefacts, ice, and unusable
spectra; both are retained so that the model can decline to assign a mineral
rather than being forced to choose among minerals everywhere.

## 2.3 Train, validation and test partitioning

Because adjacent CRISM pixels are near-duplicates at this resolution, a random
per-pixel split would place near-identical spectra in both training and
evaluation and inflate every metric. We therefore partitioned by **geographic
unit**. Polygon centroids were computed in projected coordinates, and polygons
were agglomerated by single linkage at 0.25° (approximately 15 km at the Mars
equator), unioned with any pair of polygons sharing at least one literal pixel.
Whole units were then assigned greedily to train, validation, and test with
target fractions of 0.70, 0.15, and 0.15, balancing per-class positive pixel
counts rather than raw pixel counts so that rare classes are represented in every
partition. A minimum-holdout guard forces the smallest donor unit into the
holdouts while any class falls below 5% in validation or test.

This procedure eliminates pixel-level and polygon-level leakage. It does not
eliminate scene-level overlap: a unit is a spatial cluster, whereas an MRDR tile
spans 5°, so one tile may contribute units to more than one partition. In our
data, 8 of 20 test tiles also appear in training, accounting for 38.6% of test
pixels. We therefore report test metrics both overall and restricted to the
scene-disjoint subset, and recommend the latter as the more conservative figure.

## 2.4 Spectral representation: dual continuum removal

Continuum removal is standard practice for isolating absorption features, and
upper-hull continuum removal — dividing a spectrum by its upper convex hull — is
the usual choice. It imposes an invariance to albedo that a network fed raw
reflectance will not adopt voluntarily, because albedo is predictive
in-distribution while tracking dust, illumination, and atmospheric path, all of
which correlate with location and therefore fail out-of-distribution.

Upper-hull removal has a specific and severe cost for one class. A broad convex
arch over 1–2 µm is approximately its own convex hull, so hull removal divides
that feature out. Measured on our training data, hull removal retains only 41% of
the 1–2 µm arch amplitude for alteration, against 84% for bland. The consequence
is that hull-removed alteration becomes the flattest class in the vocabulary
(mean band depth 0.0381, flatter than bland at 0.0412) and therefore acts as an
attractor for featureless ground.

We therefore introduce a **dual representation of 118 channels**, concatenating
the 59-band hull-removed spectrum (channels 0–58) with a 59-band
**linear continuum removal** (channels 59–117). Linear removal divides each
spectrum by a per-spectrum least-squares line fitted over the 55 good bands,
excluding the 1021–1056 nm detector-overlap window. Because a line has no
curvature, this removes level and slope — the albedo nuisance — while preserving
curvature exactly. Using the 1–2 µm arch alone to separate alteration from all
other classes, per-pixel AUC is 0.991 for raw reflectance, 0.990 under linear
removal, and 0.856 under hull removal, with the hull-removed figure degrading
worst against HCP (0.719), which is the confusion the classifier actually makes.
Linear removal also renders curvature a *signed* feature: alteration is convex at
+0.174 while bland is concave at −0.118, on opposite sides of zero rather than
merely far apart.

Hull removal is retained alongside rather than replaced, because its stronger
invariance is the most plausible explanation for the one unambiguous
out-of-distribution success in our earlier experiments, and no experiment
isolates that causally.

Linear-removed values were clipped to [0, 2] before caching; the 99.99th
percentile is 1.415, so this retains all real signal while removing tails that
would dominate the reconstruction gradient. Each 59-band block was then divided
by its own global standard deviation, computed once over 51 463 labelled spectra
(0.070541 for the hull block, 0.172122 for the linear block), placing the two
blocks on a comparable scale before the encoder. We note in Section 4 that equal
target variance does not produce equal reconstruction loss.

## 2.5 Architecture and self-supervised pre-training

The encoder is a spatial-spectral transformer operating on 7 × 7 pixel patches.
Each of the 49 spatial positions is embedded from its 118-channel spectrum by a
single linear layer to 256 dimensions, augmented with a learned positional
embedding and a class token, and passed through six pre-norm transformer encoder
layers with four attention heads.

The encoder was pre-trained as a **denoising masked autoencoder**. In each step,
75% of the 49 spatial positions were masked; the visible positions were encoded
and a lightweight decoder (64 dimensions, two layers) reconstructed all
positions. Reconstruction loss was mean squared error over all positions, not
only the masked ones.

Corruption followed a CRISM-specific noise model applied to the clean patch
before encoding, comprising three additive components: independent Gaussian noise
per pixel and band; a band-localised spike centred on the 1 µm detector seam with
one magnitude per patch, shaped as a Gaussian bump of 3-band FWHM; and a
per-column, per-band bias broadcast across rows, representing CRISM's
column-correlated artefacts. The three standard deviations were estimated from
labelled spectra as 0.0087, 0.0058, and 0.0049 in hull-removed units. Because
these are absolute quantities and the dual representation is standardised to
approximately unit variance, they were rescaled by the reciprocal of the
hull-block standard deviation (a factor of 14.176) so that the corruption-to-signal
ratio matches the value at which they were estimated; the seam spike was
mirrored into the linear block. Without this rescaling the corruption would have
been approximately 14 times too weak and the model nominally, but not
effectively, denoising.

Pre-training ran for 200 epochs over 200 000 patches per epoch at batch size
1024, with 10 warmup epochs and a cosine schedule, sampling patches from all
available tiles rather than only labelled ones. The best checkpoint by
reconstruction loss occurred at epoch 154 (loss 0.0469).

## 2.6 Supervised fine-tuning

The pre-trained encoder was transferred to a classifier that concatenates the
encoder's class-token output with a small multilayer perceptron embedding of a
single auxiliary scalar — the centre-pixel brightness of the patch, computed from
the raw spectrum before continuum removal — and maps the concatenation to
per-class logits through a linear head. Brightness is supplied explicitly because
continuum removal deliberately discards it, yet absolute albedo carries
information a spectroscopist uses.

Because labels are multi-label, the model emits an independent sigmoid per class
and is trained with **asymmetric loss** (Ridnik et al., 2021), which down-weights
easy negatives and is well suited to the extreme negative-to-positive imbalance
of per-pixel mineral detection. We used γ⁻ = 4.0, γ⁺ = 0.0, and a probability
margin of 0.05. Per-pixel loss weights were assigned from the annotation
confidence tier as 1.0, 0.85, and 0.70 for High, Moderate, and Low respectively.

Optimisation used AdamW at a base learning rate of 5 × 10⁻⁴ with the encoder
learning rate scaled by 10⁻³ relative to the head, batch size 256, for up to 150
epochs with early stopping on validation mean average precision after a patience
of 40 epochs and a minimum improvement of 5 × 10⁻⁴. The early-stopping metric
excludes the `junk` class, whose noisy near-zero average precision destabilises
the mean without carrying scientific interest.

Plagioclase is the weakest class in the hand-labelled data, and we augmented it
with an independent pool of 8671 plagioclase-bearing pixels from MTRDR products,
validated against a disjoint pool of 1817 pixels sharing no tiles or pixels with
the training pool. These auxiliary patches were converted to the same
representation as the main training data before injection, since the dataset
serves them without transformation.

## 2.7 Baselines

Because prior work in this area compares model checkpoints against one another,
we established two independent reference points, both emitting per-pixel
probability rasters in the same format as the neural model so that all methods
pass through an identical downstream vectorisation and are therefore directly
comparable.

### 2.7.1 Expert band-parameter rules

The first baseline encodes domain practice as an explicit, auditable ruleset over
the 60 CRISM summary parameters. The logical structure was fixed by mineralogy
and never fitted; only threshold values were calibrated from training data.

Olivine is detected by OLINDEX3, LCP by LCPINDEX2, HCP by HCPINDEX2, and
plagioclase jointly by RPEAK1 and BD1300, the criterion used in manual mapping.
RPEAK1 is the wavelength of the reflectance peak rather than an amplitude, and
plagioclase occupies a window near 0.7–0.8 µm; it is therefore applied as a
two-sided window and, because it is a regional rather than per-pixel
discriminant, computed as a 7 × 7 mean. Alteration is a disjunction over five
mineral groups, each requiring a specific diagnostic together with evidence of
hydration, with the deliberate exception of carbonate, which is anhydrous and
would be silently rejected by a blanket hydration requirement. Ice indices veto
alteration so that seasonal frost cannot register as a hydrated mineral, and
contribute instead to `junk` alongside non-physical reflectance and extreme
spectral variance. `bland` is the residual.

Two design decisions follow from the multi-label property. First, no rule
excludes another mineral: an earlier formulation in which olivine vetoed pyroxene
was rejected because olivine-bearing basalts are common. Second, the
cross-response between LCPINDEX2 and HCPINDEX2 is handled as a *tier modifier*
rather than a gate — where both indices are elevated, both labels fire and the
dominant index merely scores higher.

Thresholds were calibrated on the training partition alone. Each veto is placed
so that it retains at least 90% of the class's own training positives, jointly
across all vetoes applied to that class, making a veto structurally unable to
eliminate the class it protects. The detection threshold is then swept to
generate a ladder, and each rung is assigned the **empirical precision** of the
rule at that strictness on training data. A rung therefore reports the fraction
of detections that are correct, placing the rule on the same axis as a model
probability. We note that this precision is measured over labelled pixels, whose
positive base rate exceeds that of a whole tile, so it is optimistic.

### 2.7.2 Classical machine learning

The second baseline trains a random forest (300 trees, minimum leaf size 5) and a
histogram gradient-boosting classifier (200 iterations) on the same 60 summary
parameters at each labelled pixel, using the same training partition. Gradient
boosting was selected as the second learner specifically because it accepts
missing values natively, and the summary parameters contain no-data pixels whose
imputation would otherwise bias the comparison; the random forest requires
imputation, for which the training-set median was computed once and stored with
the model rather than recomputed at inference. Multi-label output is preserved by
native multi-output support in the forest and by one-versus-rest fitting for the
boosted trees; no arg-max is taken at any point.

## 2.8 Inference and vectorisation

At inference the full 118-channel representation is computed once for a padded
tile and 7 × 7 patches are extracted from it, which is equivalent to per-patch
computation because the least-squares fit is row-independent. Per-class
probability rasters are median-filtered with a 3 × 3 kernel and thresholded at a
ladder of absolute probabilities; connected components of at least 9 pixels are
polygonised, simplified to a 200 m tolerance, and written as per-mineral
GeoPackage layers with the mean probability and the mean 59-band spectrum of each
polygon retained as attributes, so that any detection can be inspected
spectrally.

The threshold ladder is configurable. For models exhibiting probability
saturation we extend it above 0.99 to 0.995, 0.999, 0.9995 and 0.9999, since a
class whose polygon count barely falls across those rungs is saturated rather
than selective.

## 2.9 Evaluation

Three complementary evaluations were used.

A fixed set of eight tiles across three regions — four at Nili Fossae, two at
Argyre, and two in a dusty, altered terrain used as an out-of-distribution probe
— serves as a qualitative acceptance check, read as maps rather than scored. Two
of these tiles are deliberately training tiles, on the principle that a model
which does not look clean on terrain it has seen cannot be trusted elsewhere.

Quantitative comparison uses average precision per class on the held-out test
partition, reported both overall and on the scene-disjoint subset (Section 2.3).

Finally, because HCPINDEX2 lies in the 2 µm region where residual atmospheric
CO₂ absorption survives the standard volcano-scan correction, and no CRISM
summary parameter tracks gaseous CO₂ (the CO₂-named parameters all track CO₂
*ice*), we assess this confound through atmospheric path length. Elevation and
the air-mass factor sec(i) + sec(e) were taken from the `mrrde` product, and HCP
detection rate was tabulated by decile of each. Detections concentrating at low
elevation and high air mass would indicate residual CO₂ rather than
clinopyroxene. This diagnostic is reported rather than applied as a correction,
since genuine HCP also occurs at low elevation.

---

## Figures required

Not generated here; each needs data or a schematic that must be accurate rather
than illustrative.

1. **Representation schematic** — a spectrum shown raw, hull-removed, and
   linearly removed, with the 1–2 µm arch annotated. This figure carries the
   paper's central argument and should be drawn from real alteration and bland
   spectra, not idealised.
2. **Architecture diagram** — patch → per-position embedding → transformer →
   class token ⊕ brightness → per-class sigmoid, with the MAE pre-training path
   shown separately.
3. **Partitioning map** — geographic units coloured by split, showing that
   holdout units are spatially separated and making the residual tile-level
   overlap visible rather than buried in text.
4. **Threshold-ladder comparison** — polygon count against threshold for each
   method and class, which is where saturation becomes visible.

## Open items

- **[PENDING]** Final fine-tuned checkpoint metrics; runs in progress.
- **[PENDING]** Test-partition average precision for all methods, overall and
  scene-disjoint.
- The baselines were calibrated on an earlier label build than the one used for
  the neural models. A headline comparison requires both on the same labels; this
  is a re-run of extraction and fitting with no code change.
- Citations are named but not yet formatted; Viviano-Beck et al. (2014) and
  Ridnik et al. (2021) are the two load-bearing references so far, and the
  asymmetric-loss and MAE citations should be verified against the published
  versions rather than from memory.

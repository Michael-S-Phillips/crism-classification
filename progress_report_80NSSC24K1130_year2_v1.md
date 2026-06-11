# Annual Research Performance Progress Report

---

**Federal Agency:** NASA, Science Mission Directorate (SMD)

**Award Number:** 80NSSC24K1130

**Project Title:** From Olivine to Feldspar: Tracing Martian Crustal Evolution with CRISM Mapping Data

**PI:** Michael Phillips, Assistant Research Professor
**Email:** phillipsm@arizona.edu | **Phone:** (513) 218-0597

**Submitting Official:** Same as PI

**Submission Date:** 06/09/2026

**UEI:** ED44Y3W6P7B9
**EIN:** 74-2652689

**Recipient Organization:** University of Arizona, 845 N Park AVE RM 538, Tucson, AZ 85721

**Recipient Identifying Number / Account Number:** 3056490

**Period of Performance:** 08/01/2024 – 07/31/2027

**Reporting Period End Date:** 07/31/2026

**Report Frequency:** Annual

**Final Report:** No

---

## 1. Accomplishments

### 1.1 What are the major goals of the project?

The goal of this investigation is to understand the global distribution and stratigraphic relationships of primary rock-forming minerals on Mars using CRISM MRDR v4 data. Two specific objectives guide the effort:

**Objective 1:** Investigate the regional and global spatial distribution of primary minerals on Mars (olivine, LCP, HCP, plagioclase) and their relationship to geologic ages and units.

**Objective 2:** Reconstruct a global stratigraphy of primary minerals through a targeted investigation of impact excavated materials.

Both objectives require the completion of "Task 0" — the production of the first global primary mineral maps derived from CRISM MRDR v4 products. The Year 1 work plan was devoted to Task 0 (spectral parameter production, vectorized mineral indicator map generation, and iterative testing). The Year 2 work plan focuses on analysis of the mineral maps (Tasks 1a–1c) to address Objective 1.

### 1.2 What was accomplished under these goals?

Significant progress was made in the reporting period. Below we summarize accomplishments organized by task.

#### Task 0a – Spectral Parameter Rasters (Complete)

All 1,764 CRISM MRDR v4 tiles across Mars Charts MC02–MC30 were downloaded and processed. The MRDR v4 archive includes pre-computed spectral summary parameter products (mrrsu files) containing key parameters for primary mineral detection, including BD1300 (olivine), OLINDEX3 (olivine), LCPINDEX2 (low-calcium pyroxene), and HCPINDEX2 (high-calcium pyroxene). A global virtual mosaic (VRT) was constructed from all tiles using GDAL and a 13.2 GB cloud-optimized GeoTIFF (COG) of the four primary mafic mineral parameters was generated to enable efficient global-scale visualization and analysis. Additionally, incremental PCA was applied across the full mrral (hyperspectral reflectance) dataset to explore spectral variability.

#### Task 0b – Vectorized Mineral Indicator Maps (Substantially Advanced)

We developed a Python-based processing pipeline to convert per-pixel mineral classifications into GIS-ready GeoPackage vector products, as proposed. However, rather than relying solely on the threshold-based spectral parameter approach outlined in the proposal, we developed a significantly more capable **deep learning classification system** that provides per-pixel, multi-label mineral probabilities across five classes (olivine, LCP, HCP, plagioclase, other). This approach is expected to deliver substantially more accurate and reliable mineral indicator maps than the originally proposed parameter-thresholding method.

The classification pipeline ("crism_classification") includes the following components:

**Labeled training dataset.** We manually mapped mineral unit polygons in QGIS across multiple CRISM MRDR tiles, producing ~1.97 million labeled pixels stored as GeoPackage files. Each pixel carries multi-label confidence-weighted annotations (High / Moderate / Low confidence) for each mineral class. Training, validation, and test splits (70/15/15) were established with a fixed random seed.

**Self-supervised pre-training with a Masked Autoencoder (MAE).** We designed a Spatial-Spectral Masked Autoencoder (SpatialSpectralMAE) that pre-trains on the **full global MRDR dataset** (~3.87 billion pixels across all 1,764 tiles) by streaming random 7×7 spatial patches of 59-band mrral reflectance data. The MAE masks 75% of spatial tokens (pixels) and trains a transformer encoder to reconstruct the masked spectra from the visible context. (A "denoising" variant of the MAE additionally corrupts the visible spectra with realistic CRISM noise — Gaussian σ ≈ 0.0087, a 1 µm spike σ ≈ 0.0058, and a column-bias σ ≈ 0.0049 — and trains the model to reconstruct the clean signal at every position. The current champion classifier uses the plain MAE encoder; the v3 denoising variant is shown in Figure 9.) This forces the encoder to learn both spectral signatures of minerals and their spatial geological context. Pre-training was conducted on an HPC cluster for 194 epochs (100,000 patches/epoch), reaching a reconstruction loss of 0.01598. The pre-trained encoder serves as the foundation for downstream classification.

**SpatialSpectralTransformer architecture.** The encoder treats each pixel in a 7×7 spatial patch as a token, projecting its 59-band spectrum into a 128-dimensional embedding via a linear layer, then processing through 6 layers of pre-norm transformer encoder blocks (4 attention heads, 512-dimensional feed-forward). A learnable CLS token and positional embeddings capture spatial relationships. For classification, the center-pixel token (which has attended to all 48 surrounding pixels) is passed through a linear head to produce 5-class logits. This design captures both per-pixel spectral information and spatial geological context from the surrounding neighborhood.

**Systematic model development and hyperparameter optimization.** Over 50 training runs were conducted using Weights & Biases experiment tracking, progressing through sklearn baselines (LogReg, Random Forest, XGBoost, LightGBM), 1D spectral models (SpectralCNN, SpectralTransformer), spatial models on summary parameters, and ultimately the spatial-spectral transformer on hyperspectral data. Key innovations that improved performance include: Asymmetric Loss (ASL) for handling severe class imbalance in the multi-label setting; differential learning rate scheduling (very slow encoder updates at 0.001× head learning rate to preserve pre-trained representations); and spectral augmentation (noise injection, band dropout, spectral shift).

**Current best result:** Through extensive methodological iteration during the reporting period the spatial-spectral ViT pipeline has advanced from validation mean Average Precision (mAP) of 0.7175 (the result reported earlier in the year) to **val_mAP = 0.7579** with the current best configuration (`ft_plag_aware_relabeled`: a fine-tuned classifier built on top of a plagioclase-aware multi-task MAE encoder). Per-class performance has improved across the board: olivine AP = 0.935, LCP = 0.924, HCP = 0.792 (up from 0.51), plagioclase = 0.138, and a near-perfect AP of 1.0 on the "other" / spectrally bland or dusty catch-all class. The HCP gain (0.51 → 0.79) reflects a series of targeted interventions described in the next subsection. Plagioclase remains the hardest class — at 4.3% prevalence, with subtle 1.25 µm Fe²⁺ features that overlap olivine, it is fundamentally encoder-limited rather than head-limited (verified by linear-probe diagnostics).

#### Task 0b.1 – Methodology Advances Since Mid-Year

The original validation-mAP metric is computed against the same labeled-polygon dataset used for training and is therefore biased upward by label noise. To get a more reliable picture of real-world classification quality, a **polygon-level evaluation framework** was developed during the reporting period. For each labeled polygon, every interior pixel is classified, per-class probabilities are aggregated, and the polygon is scored as a single labeled unit against the expert mapper's category. The framework also computes calibration metrics (Brier score, Expected Calibration Error) which surfaced an important finding: the prior best model was systematically overconfident (ECE = 0.15), with predicted probabilities not matching empirical positive rates. A subsequent contrastive-refinement step (described below) reduced ECE to 0.05, a 3× calibration improvement, with substantially better polygon-level plagioclase recall (0.29 → 0.41) and HCP recall (0.64 → 0.80) even where the overall val_mAP changed only modestly.

The following techniques were introduced and integrated during this reporting period:

1. **Plag-aware multi-task MAE pretraining.** The masked autoencoder was augmented with a binary plagioclase-detection auxiliary head, supervised by ~6,500 synthetic plagioclase patches generated from known mineralogy. This gave the encoder a plag-relevant inductive bias prior to fine-tuning. The current best classifier uses this encoder.

2. **Contrastive refinement with hard-negative mining.** A second-stage training pass uses InfoNCE contrastive loss over anchor/positive/hard-negative pixel triplets. Anchors are plag-positive pixels harvested from manually-vetted ROI files contributed by an Co-I Viviano (1,817 plag positive patches across 31 CRISM observations). Hard negatives are olivine pixels that could be easily confused with plagioclase pixels and were identified by a Spectral Angle Mapper diagnostic (55% of model-predicted plag in test tiles is actually olivine). The contrastive encoder achieved the +16 percentage-point HCP recall improvement and the 3× calibration improvement noted above.

3. **Auxiliary mrrsu summary-parameter injection.** Selected mrrsu spectral parameters (BD1300 = 1.3 µm band depth, RPEAK1 = reflectance-peak wavelength) were injected as auxiliary channels into the classifier head. The reflectance peak position is a known plag/olivine discriminator that the raw 59-band encoder struggles to derive from spectra alone.

4. **Manual label refinement.** An interactive Streamlit application was built to walk a researcher through model-predicted polygons one at a time, displaying mean spectra with ±1σ envelopes alongside false-color RGB thumbnails for spatial context. The reviewer confirms, rejects with optional reclassification, or skips each polygon. Confirmed polygons are harvested into a parquet schema compatible with the training set; rejections are tagged and used as labeled hard negatives. As of this reporting period, ~169 polygons across MC13 have been reviewed, yielding ~1.18 million manually-verified positive pixels (olivine, LCP, HCP) plus 58,000 "bland" / dust-dominated negative pixels and 34,000 "ambiguous" negatives (i.e., not bland/dusty but also not a mafic mineral; usually shadowed regions, CO2 frost, or spurious pixels).

5. **Targeted plagioclase training supplementation.** Sensor-space ROI files identifying plagioclase-bearing pixels in 31 CRISM observations were ingested and resampled to the 59-band mrral convention, producing a high-confidence plag positive pool to drive the contrastive refinement.

**Vectorization pipeline (Vectroscopy).** A complete pipeline converts per-pixel probability maps into GeoPackage vector mineral maps with confidence-tiered polygons. Global percentile-based thresholds ensure consistent class boundaries across tiles. Six tiles in the Nili Fossae and MC13 regions have been fully vectorized as a demonstration (Figure 1). These vector products are GIS-ready and loadable in QGIS and ArcGIS Pro and will be part of the published deliverables from this effort.

#### Task 0c – Iteration and Testing (Ongoing)

The test suite has grown to **>60 unit tests** validating all components of the pipeline — model architectures, data loading, loss functions, metric computation, tile-level inference, the polygon-level evaluation harness, the manual-review app's queue iterator and persistence layer, and the cross-process determinism of derived parquet outputs. Prototype and supervised classification approaches have been compared on demonstration tiles (Figure 2).

Iteration during this reporting period focused on two interlocking activities:

1. **Architectural and training-recipe sweeps** on the labeled training pool (described in §Task 0b above), driving val_mAP from 0.72 → 0.76 and polygon-level recall metrics substantially higher.

2. **Active-learning data harvest.** Because the polygon-level evaluation revealed that the training-set label noise was a primary bottleneck (the model was overconfident in regions where the gpkg labels disagreed with what the spectra actually showed), the focus shifted from architectural changes to expanding and cleaning the training data through manual review. The MC13 quadrant review yielded a 1.18-million-pixel manually-verified positive pool and a 100-thousand-pixel labeled negative pool, with detailed provenance (decision timestamp, source vector product, predicted class, corrected class, source polygon UID).

Notable observations from the MC13 review:
- **No genuine plagioclase detections in MC13.** Of 48 model-flagged plag polygons reviewed, 46 were rejected (27 reclassified as olivine, 19 tagged ambiguous). This validates the encoder-bottleneck hypothesis: the contrastive model is still false-positiving plag on olivine-bearing spectra, and MC13 does not contain regions where this is balanced by true positives.
- **The v3 denoising-MAE classifier produces materially cleaner HCP polygons than the prior best model** (review confirm-rate 7/11 vs 9/39). This influenced the choice of model for the next-region review (MC11, in progress).
- **The "other" class label has a known systematic mismatch** between training (dust-region pixels) and human-mapped polygons (which often contain real but unclassified minerals). This is being explicitly handled by renaming the training label to `bland` in the review app while keeping the parquet schema unchanged for downstream compatibility, and by planning explicit pre-classifier masks for shadow and CO₂-frost regions instead of overloading a catch-all class.

#### Task 1a – Global and Regional Distribution (Initiated)

With the classification pipeline nearing production readiness, we have begun preliminary investigation of global mineral distributions using the mrrsu summary parameter products. Global visualization tools (QGIS project with VRT mosaics, COG basemaps, and Mars Chart quadrant overlays) are in place to support the spatial analysis planned for the remainder of Year 2 and into Year 3.

### 1.3 What opportunities for training and professional development has the project provided?

Samuel Cartwright (Graduate Student # 1) has received extensive training in python coding for remote sensing, CRISM data processing, Python-based GIS development, and experiment design. He has led the development of the vectorization pipeline and will lead global analysis of the output mineral unit maps. This work will form a core chapter of the Sam's dissertation. He has gained proficiency in skills that complement his planetary geology training.

### 1.4 How have the results been disseminated to communities of interest?

Preliminary results have been presented at the Lunar an Planetary Science Conference and at the Geological Society of America Fall meeting.

### 1.5 What do you plan to do during the next reporting period to accomplish the goals?

The MC13 review effort has demonstrated the value and limitations of the manual-review pipeline: clean labels improve calibration but a single-Mars Chart pool concentrates the loss signal on too few unique polygons. The immediate next step (in progress at the time of this report) is to deploy the strongest classifier (the v3 denoising-MAE variant) on a second Mars Chart quadrant (MC11, a region with substantially different geology from MC13) and run the same review pipeline against the resulting predictions. This will provide cross-region label diversity, which our negative-result diagnosis indicates is the missing ingredient for meaningful training-set augmentation. Once the cross-region pool is assembled, we will retrain the classifier with appropriate per-polygon subsampling and a balanced confidence-weighting scheme, and run inference on all 1,764 MRDR tiles to produce global mineral probability maps and vectorized mineral indicator products. We expect this to be finished by the end of June 2026.

During Year 3 (08/2026–07/2027), we plan to:

1. **Complete Objective 1 analyses** (Tasks 1a–1d). With global mineral maps in hand, we will perform spatial clustering and outlier analyses, calculate summary statistics by geologic age and terrain unit (intersecting with the Tanaka et al. global geologic map), and conduct the focused investigation of spectrally feldspathic outcrops. Two publications will be compiled from this effort. Frist, the results of the analysis of primary minerals with the global geology of Mars will be compiled into the first publication and presented on at a Fall meeting (likely GSA). Second, a global investigation into the distrubtion of spectrally feldspathic outcrops will be published and presented on at a Spring conference (likely LPSC).

2. **Complete Objective 2** (Tasks 2a–2e). We will generate GIS shape layers of crater-related units using the Robbins crater database, associate mineral detections with crater materials, and begin stratigraphic reconstruction using crater scaling laws. Results from this effort will be compiled into a third publication and presented on at LPSC in the Spring.

---

## 2. Products

### 2.1 Publications and Conference Presentations

1. The global distribution of primary rock-forming minerals on Mars, Cartwright et al., *in prep*.
2. Cartwright S, Jandai T, et al (2025) Uncovering Martian Crustal Evolution: Preliminary Results from Global Primary Mineral Mapping with CRISM. pp 2025AM – 10213, GSA Connects 2025 Meeting (https://gsameetings.secure-platform.com:443/connects25/gallery/rounds/82002/details/16214)
3. Phillips MS, Cartwright SFA, Viviano CE, et al (2026) Diversity, Distribution, and Petrogenesis of Spectrally Feldspathic Outcrops on Mars. In: 57th Lunar and Planetary Science Conference. Lunar and Planetary Institute, The Woodlands, TX, Abstract #1828



### 2.2 Technologies or Techniques

**Deep learning classification pipeline for CRISM hyperspectral data.** A complete, open-source Python pipeline ("crism_classification") for per-pixel multi-label mineral classification of CRISM MRDR mrral data. The pipeline includes:
- `SpatialSpectralTransformer` — a vision transformer adapted for spatial patches of hyperspectral data
- `SpatialSpectralMAE` — masked autoencoder pre-training on global CRISM data, with a plag-aware multi-task variant introduced during the reporting period
- `SpatialSpectralClassifier` — fine-tuned mineral classifier with differential learning rates, asymmetric loss, and optional mrrsu auxiliary-channel injection
- `ContrastiveEncoder` + InfoNCE contrastive-refinement training stage with hard-negative mining
- Vectroscopy pipeline for converting pixel-level probabilities to GIS-ready vector mineral maps with per-class threshold ladders
- **Polygon-level evaluation harness** — independent of train/val pixel-level metrics, produces calibration analysis (Brier, ECE, reliability diagrams) and confusion matrices on the polygon unit
- **Streamlit-based manual review application** — paginates through model-predicted polygons with mean spectra + spatial context thumbnails, harvests confirmed pixels into parquet, writes corrected-class rejections to a hard-negatives parquet, and supports cross-session navigation and re-decisioning with atomic-write durability
- Interactive tile viewer (FastAPI web application)

### 2.3 Databases

**Labeled mineral pixel dataset.** ~1.97 million pixels from manually mapped polygons with multi-label confidence-weighted mineral annotations across 5 classes (olivine, LCP, HCP, plagioclase, other), stored as Parquet files with full provenance (tile ID, pixel coordinates, polygon ID, confidence tier). During the reporting period this was supplemented by ~1.18 million manually-reviewed positive pixels and ~92 thousand labeled hard-negative pixels harvested through the polygon review application (per-pixel provenance preserves the source vector product, predicted class, corrected class if any, and decision timestamp).

**Global CRISM MRDR v4 COG.** A 13.2 GB cloud-optimized GeoTIFF containing the four key mafic mineral summary parameters (BD1300, OLINDEX3, LCPINDEX2, HCPINDEX2) mosaicked across all 1,764 tiles.

### 2.4 Software

The crism_classification Python package (hosted on GitHub) includes all model definitions, training scripts, inference scripts, and visualization tools. Key dependencies: PyTorch, rasterio, geopandas, scikit-learn, Weights & Biases.

### 2.5 Data or Other Products

**Pre-trained model checkpoints:**
- `spatial_mae_128d_6l_best.pt` — MAE encoder pre-trained on the full global MRDR dataset (epoch 194, mae_loss = 0.01598)
- `spatial_mae_denoising_128d_6l_best.pt` — Denoising-MAE variant pretrained during the reporting period; basis for the current best classifier
- `plag_aware_mae_128d_6l_best.pt` — Plag-aware multi-task MAE encoder (adds a binary plag-detection auxiliary head during pretraining)
- `ft_plag_aware_relabeled_best.pt` — Current champion classifier (val_mAP = 0.7579, polygon-level olivine recall = 0.96)
- `ft_v3_denoising_lrscale001_best.pt` — v3 denoising-MAE classifier (best HCP polygon precision in manual review)
- `contrastive_plag_v1_best.pt` — Contrastive-refinement encoder (3× better calibration, +16 pp polygon HCP recall)
- Multiple ablation checkpoints documenting the model development trajectory

**Vectorized mineral maps:** GeoPackage files for the full MC13 quadrant (54 tiles, four mineral classes per source model variant) plus six earlier demonstration tiles in Nili Fossae and Argyre. MC11 is being added at the time of this report.

**Mars Chart quadrant boundary files:** GeoPackage vector files for the 30 Mars Chart quadrants, generated to support spatial analysis.

---

## 3. Participants & Other Collaborating Organizations

### 3.1 What individuals have worked on the project?

| Name | Role | Contribution |
|------|------|-------------|
| Michael Phillips (University of Arizona) | PI | Project oversight, classification pipeline development, spectral analysis, mineral mapping methodology |
| Sam Cartwirght (CU Boulder) | Graduate Student #1 | Vectorization pipeline, spectral parameter analysis, primary mineral global geology analysis |
| Christopher Hamilton (University of Arizona) | Co-I #1 | expert assessment of petrogensis and volcanic origin for primary minerals |
| Christina Viviano (JHU/APL) | Co-I #2 | [TO BE FILLED: spectral parameter expertise] |
| Frank Seelos (JHU/APL) | Co-I #3 | [TO BE FILLED: CRISM data processing] |
| Mallory Kinczyk (JHU/APL) | Co-I #4 | [TO BE FILLED: crater stratigraphy] |
| Patrick O'Brien  (CU Boulder) | Co-I #5 | new institutional PI at CU Boulder, advising Sam Cartwright |

### 3.2 What other organizations have been involved?

N/A

### 3.3 Have other collaborators or contacts been involved?

N/A

---

## 4. Impact (Optional)

### 4.1 What is the impact on the development of the principal discipline(s) of the project?

The methodology advances developed during this reporting period — particularly the polygon-level evaluation framework with calibration analysis, the contrastive-refinement strategy with hard-negative mining for spectrally-confusable minerals, and the active-learning manual-review pipeline — extend beyond CRISM mineral mapping to any hyperspectral classification problem where (a) labeled training data is noisy or label definitions are inconsistent across sources, (b) the relevant per-class evaluation metric is the integral over a polygon rather than per-pixel accuracy, and (c) a small number of difficult classes drive utility. We expect these techniques to be of interest to the broader planetary remote sensing community working with VNIR/IR hyperspectral data (e.g., M3, OMEGA, other CRISM datasets and future Mars missions).

### 4.2 What is the impact on other disciplines?

The polygon-level evaluation harness and active-learning review tool are general-purpose: they take a vector polygon source and a per-pixel classifier and produce calibration and confusion-matrix diagnostics with no CRISM-specific assumptions. We anticipate they will be reusable for terrestrial Earth-observation hyperspectral analyses by future students, given that the labeled-data quality issue is universal across remote-sensing classification problems.

### 4.3 What is the impact on the development of human resources?

See §1.3.

### 4.4 What is the impact on physical, institutional, and information resources that form infrastructure?

The pipeline code and pre-trained model checkpoints are versioned on GitHub. The full ~3.87 billion-pixel global mrral dataset, the labeled training pool, the polygon-review outputs, and the global mineral-parameter COG together constitute a research-ready infrastructure asset hosted at the University of Arizona and accessible by collaborators. Data will be released on Zenodo at the time of publication.

### 4.5 What is the impact on technology transfer?

Not applicable in this reporting period.

### 4.6 What is the impact on society beyond science and technology?

Not applicable in this reporting period.

---

## 5. Changes/Problems (Optional)

### 5.1 Changes in approach and reasons for change

One scope refinement was made during this reporting period:

1. **Active-learning manual-review pipeline added to Task 0c.** The original Year 2 work plan assumed that the existing labeled-polygon dataset (~1.97M pixels) would be sufficient to train a production-quality classifier, with iteration on architecture. Polygon-level evaluation results during the reporting period revealed that label noise in the existing polygon dataset was a meaningful contributor to remaining errors, motivating the construction of an interactive review application to harvest higher-quality labels at the pixel level. This is an addition to the planned methodology, not a substitution; it has delayed the global-production milestone by approximately 2–3 months but should yield a measurably better global map.

### 5.2 Actual or anticipated problems or delays and actions to resolve them

The active-learning pivot described above has delayed scaling from demonstration tiles to full global production by approximately 2–3 months. We do not anticipate this impacting the overall Period of Performance timeline, as global inference itself is well-understood and bounded (~hours of HPC GPU time once a final classifier is selected). 

### 5.3 Changes that have a significant impact on expenditures

None.

### 5.4 Significant changes in use or care of human subjects, vertebrate animals, biohazards, and/or select agents

Not applicable.

### 5.5 Change of primary performance site location from that originally proposed

None.

---

## 6. Project Outcomes (Optional)

The Year 2 reporting period closed with a classification pipeline including a calibrated, polygon-level-evaluated mineral classifier (val_mAP = 0.7579, polygon-level olivine recall = 0.96, HCP recall = 0.80 with the contrastive-refined variant), an active-learning data-curation workflow, and a vectorized mineral indicator product for the full MC13 quadrant. The pipeline is ready to support both the global production planned for the remainder of Y2 (deliverable Task 0 finalization), the regional and MC-scale geologic analyses planned in Tasks 1a–1d, and the stratigraphic reconstruction planned for Task 2.

---

## Figures

### Figure 1: Vectorized Mineral Maps — Per-Class Threshold Ladders (Tile t1249, Nili Fossae)

For each detected mafic mineral on a representative tile (t1249, Nili Fossae region, MC26), the figure shows a four-panel summary: (top-left) MAF false-color browse for spatial context, (top-right) vectorized mineral polygons at five confidence-tier thresholds overlaid on the same scene, (bottom-left) the per-threshold pixel-count distribution, and (bottom-right) the per-threshold mean reflectance spectrum. This panel format demonstrates the complete end-to-end pipeline from per-pixel model output, through threshold-based vectorization, to GIS-ready GeoPackage products with explicit confidence tiers. Plagioclase is omitted from this tile because t1249 contains no genuine plagioclase detections; the discriminator for plagioclase is the focus of the manual-review and contrastive-refinement work described in §1.2 Task 0b.1.

**(a) Olivine**

![Olivine threshold ladder for t1249](reports/per_mineral_nili_v3_bland/nili_t1249_olivine.png)

**(b) Low-Calcium Pyroxene (LCP)**

![LCP threshold ladder for t1249](reports/per_mineral_nili_v3_bland/nili_t1249_lcp.png)

**(c) High-Calcium Pyroxene (HCP)**

![HCP threshold ladder for t1249](reports/per_mineral_nili_v3_bland/nili_t1249_hcp.png)

Across the three mafic minerals shown, the threshold-ladder approach lets downstream analyses select the appropriate confidence/coverage trade-off — strict thresholds (0.97) yield small, highly-confident detection sets useful for endmember characterization, while permissive thresholds (0.85) yield broader regions suitable for context mapping. The mean-spectrum panels (bottom-right of each sub-figure) confirm that the detected pixels show the diagnostic spectral signatures of each class.

### Figure 2: Model Architecture

![Model Architecture](reports/fig_v3_denoise_architecture.png)

Schematic representation of the encoder (A) and classifier (B) models used to produce the mineral maps shown in Figure 1. The encoding model was trained using masked autoencoding where 75-85% of input spatial mixels are masked and the model is tasked with reconstructing the unmasked input. This is an unsupervised approach that allows the encoder to learn general CRISM spectral representations. For classification, the decoder model from pre-training is removed and a linear classification head is attached to the encoder. The encoder is allowed to be adjusted (fine-tuned) at a fraction of the learning rate of the classifier model. The classifier model learns to predict among 5 output classes: olivine, low-Ca pyroxene (LCP), high-Ca pyroxene (HCP), plagioclase, and "bland". 

### Figure 3: Full MC13 Quadrant Classification

**(a) Olivine**

![Olivine threshold ladder for MC13](reports/per_mineral_mc13_v3_bland/mc13_olivine.png)

**(b) Low-Calcium Pyroxene (LCP)**

![LCP threshold ladder for MC13](reports/per_mineral_mc13_v3_bland/mc13_lcp.png)

**(c) High-Calcium Pyroxene (HCP)**

![HCP threshold ladder for MC13](reports/per_mineral_mc13_v3_bland/mc13_hcp.png)

Seamless mosaic comparison of the deep-learning classifier output (right) against the conventional spectral-parameter approach (left) across the entire MC13 Mars Chart quadrant (54 mrral tiles, 5–30°N × 45–90°E). The deep-learning classifier produces a substantially cleaner, more spatially coherent product than direct parameter thresholding, validating the methodological choice described in §1.2 Task 0b. 

### Figure 4: Polygon-Level Evaluation — Confusion Matrix

![Polygon-level confusion matrix (champion classifier)](reports/polygon_eval_ft_plag_aware_relabeled_best/confusion_matrix.png)

Polygon-level confusion matrix for the current best classifier evaluated on the full 3,386 single-mineral test-polygon set. Each polygon is classified by its mean predicted-class probability and scored against the expert mapper's category. This polygon-level view (in contrast to the per-pixel val_mAP metric) provides the more reliable accuracy estimate referenced in §1.2 Task 0b.1 and was the diagnostic that motivated the calibration-improving contrastive-refinement step.

---

*Prepared under NASA Grant 80NSSC24K1130.*

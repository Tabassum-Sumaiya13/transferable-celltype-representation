
# Cross-Cohort Spatial Proteomics Cell-Type Annotation

## 1. Research question

The project tests whether a model can learn cell-type representations that remain biologically meaningful when the data change across:

- cohort and patient population;
- tissue and disease;
- acquisition platform;
- antibody panel and missing markers;
- raw-value scale and preprocessing;
- native label names and label granularity.

The strongest test is not random cell-level cross-validation. Cells from the same slide are technically related, so a random split can make the task look easier than it is. The main evaluation is therefore leave-one-cohort-out (LOCO): one complete cohort is hidden, the model is trained on the remaining cohorts, and the hidden cohort is scored only after training and model-selection decisions are complete.

## 2. Cohort design

The registry in `celltype_transfer/config.py` describes each cohort declaratively. The current roster contains CRC and UPMC CODEX data, Keren MIBI-TOF data, Ferguson IMC data, Phillips CODEX data, Danenberg IMC data, and Sorin IMC data. Each specification records the tissue, disease, platform, pixel size, value scale, source tables, marker columns, coordinates, patient identifiers, and native labels.

The loader converts every source into one standard table:

`cell_id, cohort, image_id, patient_id, x_px, y_px, area_px2, native_label, label_confidence`

Stage 0 keeps all cells and columns. It does not silently remove unusual labels, stains, or channels. It audits missing values, marker counts, label counts, slide sizes, and coordinate plots. A dataset is rejected only for a documented structural reason, such as the absence of recoverable cell coordinates.

This separation matters. Data ingestion should not decide that a label is biologically invalid before the marker evidence has been examined.

## 3. Marker and panel harmonization

Different studies use different names for the same protein, duplicate reagents, different epitopes, and non-protein channels such as DNA stains. Stage 0b resolves marker names through HGNC and UniProt into a canonical triple:

`gene_or_complex | epitope | modification`

The triple, rather than the raw column name, identifies a marker. This keeps biologically different markers such as CD45, CD45RA, and CD45RO separate while allowing equivalent names to match. Duplicate reagents that resolve to the same triple are combined according to the panel policy. Non-protein channels are excluded from the biological marker vocabulary.

The result is a frozen vocabulary in `work/panel.json`. Every marker has one stable index. A cohort that does not measure a marker does not receive a biological zero; it receives an explicit absent-marker state. A later cohort can use only the vocabulary that was frozen during model development. New markers are reported as dropped rather than silently changing the meaning or shape of an existing checkpoint.

## 4. Continuous value normalization

The source datasets arrive on incompatible scales: raw fluorescence, arcsinh values, z-scores, and quantized uint8 values. Comparing these values directly would make technical scale differences look like biology.

For each cohort and marker, the pipeline computes a mid-rank empirical cumulative distribution function (ECDF):

$$
u_{c,m}(x_i) = \frac{\operatorname{rank}(x_i)}{n_c}
$$

where $c$ is the cohort, $m$ is the marker, and ties receive their average rank. This is a monotone, bin-free transform. It preserves the ordering of cells within a cohort while putting markers from different technical scales onto a comparable $[0,1]$ scale.

The normalization decision is selected by a predeclared bake-off. The tested alternatives include image-level ECDF, cohort-level ECDF, learned slide correction, and combinations of image and cohort ranks. The winning representation is the one that gives the best cross-cohort masked-marker reconstruction without using cell-type labels from the held-out cohort.

Image-level normalization is treated carefully. If a slide is dominated by tumour cells, ranking inside that slide can create an artificial median and distort prevalence. Therefore the cohort-level rank is retained as the main representation, while image statistics are tested as an additional signal rather than assumed to be harmless.

## 5. Self-supervised panel-aware representation learning

After normalization, a cell is represented as a set of marker tokens rather than as a fixed-width vector whose missing entries look like negative expression.

Each measured marker token contains:

1. the normalized marker value;
2. a learned marker-identity embedding;
3. an explicit absent-marker embedding for vocabulary slots not measured by the cohort.

The token sequence is processed by a small Transformer-style set encoder. Attention lets the model use marker combinations, not only independent marker thresholds. Mean pooling produces a 128-dimensional cell embedding, $z_{cell}$.

Before supervised cell-type training, the token encoder is pretrained with masked-marker reconstruction. A fraction of eligible marker values is hidden, and the model predicts the hidden values from the remaining marker set. The objective is label-free, so all cohorts can contribute before their native labels have been aligned:

$$
\mathcal{L}_{mask} = \frac{1}{|\mathcal{H}|}\sum_{m \in \mathcal{H}} \left(\hat{u}_m-u_m\right)^2
$$

where $\mathcal{H}$ is the set of hidden, measurable, non-degenerate markers. Markers with insufficient rank variation are excluded from this loss because their reconstruction denominator is effectively zero and would create unstable, uninformative gradients.

This stage answers a useful intermediate question: does the representation transfer marker relationships across panels before any shared cell-type label is imposed?

## 6. Automatic shared label space

Native labels cannot be compared by spelling. The same name can mean different biology in different cohorts, while different names can describe the same type. Stage 1b therefore builds a shared label space from marker signatures rather than text.

For every `(cohort, native_label)` pair, it computes:

- marker quantiles on the cohort-level ECDF scale;
- mean marker ranks;
- label prevalence and cell count;
- within-label marker co-expression;
- the marker evidence available in that cohort.

Cross-cohort comparison uses within-cohort, between-label scaling. This prevents a cohort's cell composition from becoming a false cohort fingerprint. A symmetric distance is used to decide whether labels should merge. A separate directed containment relation represents broad parent types and narrower subtypes. Cycles are contracted into strongly connected components before the nesting graph is used.

Granularity is selected per branch. A proposed split is retained only when it reproduces under a held-out-cohort test and passes minimum support rules. Cluster names are assigned from discriminative markers for readability; text names do not determine similarity or clustering.

The label-space procedure is evaluated without any hand mapping. The project's own hand mapping was removed entirely in protocol v2 (as a method, a gate and a validation source); agreement is measured only against published references (CellMarker 2.0 and the Cell Ontology). This prevents the method from tuning its ontology against the answer it is later judged on.

## 7. Supervised cell-type training

Once the shared label space exists, cells whose native labels map into it can supervise the encoder. The main model uses the pretrained token layer and predicts shared clusters from $z_{cell}$.

The training objective in Stage 6 contains:

- a prototype-based cell-type loss, with one learned prototype per shared cluster;
- masked-marker reconstruction as an auxiliary representation loss;
- confidence weighting where a cohort supplies label confidence;
- optional VICReg and other controls, tested by ablation rather than assumed to help.

Prototype initialization uses the marker signatures from the label-space stage. This gives each prototype a biologically interpretable starting point and makes prototype drift measurable. Training uses class-balanced sampling and early stopping on validation slides from the training cohorts. The held-out cohort is never used for early stopping.

## 8. Domain-shift control

Stage 3 tests whether the cell embedding contains technical slide information. A gradient-reversal adversary tries to predict slide identity while the encoder receives the reversed gradient and is encouraged to remove slide-specific information.

The adversary targets slide identity, not cohort identity. Cohort is confounded with tissue and disease in this dataset, so removing cohort information could also remove real biology. Only slide distinctions that can be compared within shared patients are used for the strongest technical-invariance check.

The adversary is selected by a lambda sweep. A fresh discriminator is trained from scratch on the frozen embedding and evaluated on unseen slides. This is more reliable than reading only the discriminator trained jointly with the encoder, because a jointly trained discriminator can fail for optimization reasons while slide information remains present.

The final shipped configuration is determined by the gates. If the adversarial arm does not improve cross-cohort macro-F1, the non-adversarial arm is kept. Domain invariance is a constraint, not a goal that should erase tissue-specific biology.

## 9. Evaluation protocol

The primary protocol is seven-fold LOCO over the registered cohorts:

1. select one complete cohort as the test cohort;
2. build training-only label-space and model artifacts for the other cohorts;
3. split the training cohorts by patient into train, validation, and test patients (a patient's slides never fall on both sides);
4. fit normalization probes, the token encoder, and the cell-type model using training data only;
5. select epochs and ablations using training-cohort validation data only;
6. evaluate once on every cell or a declared reproducible sample from the held-out cohort;
7. repeat until each cohort has served as the unseen test cohort.

The headline metric is macro-F1 over reliable shared clusters:

$$
F1_{macro} = \frac{1}{K}\sum_{k=1}^{K} \frac{2P_kR_k}{P_k+R_k}
$$

Macro-F1 prevents abundant cell types from hiding failure on rare types. The report also includes per-class precision, recall, F1, balanced accuracy where applicable, confusion patterns, support, and the majority and random baselines. The generalization gap is reported as:

$$
\Delta_{gen} = F1_{in\text{-}distribution} - F1_{held\text{-}out}
$$

The gap is often more informative than one accuracy number because it separates ordinary classification quality from robustness to domain shift.

## 10. Completely unseen cohort test

An older Ferguson frozen-holdout protocol (train on five cohorts, score Ferguson as pure test data) and a new-cohort script (place a new cohort's labels into the frozen partition and run a forward pass with no retraining) were part of `pipeline2`. They were written for the old 5+1 roster and are archived, not ported, in `dropped_past_works/pipeline2_stale/` (`s7_eval.py`, `s9_newcohort.py`). The seven-fold LOCO protocol, with a label space built inside each fold without its held-out cohort, is the only protocol in `celltype_transfer/`. Do not mix the two in one report, and do not present the whole-roster label space as blind to any cohort: it was built with every cohort present.

## 11. What would count as biological success?

The central claim is supported only if all of the following hold together:

- held-out macro-F1 is clearly above majority and random baselines;
- performance is stable across held-out cohorts, not driven by one easy dataset;
- per-cell-type results show transfer of biologically meaningful rare types, not only broad abundant classes;
- removing or masking panel markers causes a measured, interpretable degradation rather than a collapse caused by missing-value coding;
- the representation cannot be strongly decoded for technical slide identity while retaining cell-type performance;
- newly observed labels can be rejected as novel instead of being forced into an incorrect known class;
- the result remains after excluding unreliable ontology branches and after reporting the exact label-space construction used in each fold.

Together, these tests distinguish biological generalization from memorizing cohort-specific marker scales, panels, label names, or slide artifacts.

## 12. Current limitations

- The spatial-context stage (Gate 4) passed on its declared checks, but its paired 95% interval for (neighbourhood − cell) spans zero over 7 folds. So "spatial context improves transfer" is not yet an earned claim. Its `cell` baseline also uses the 3-loss configuration, not the shipped 2-loss one.
- Pixel-size values marked `ASSUMED` are measurement risks and should be verified before using physical distances.
- The shared ontology is inferred from marker signatures. It can merge biologically distinct labels when the available panel lacks discriminative markers.
- A frozen vocabulary cannot use genuinely novel proteins unless the model is retrained and the benchmark is repeated.
- The old Ferguson holdout path and the current seven-fold LOCO registry must be kept separate in reports.

## 13. How to run

The code is in `celltype_transfer/`. `python celltype_transfer/run.py` lists every step in order; `python celltype_transfer/run.py cpu` runs the local steps 1–8. Steps 9–12 need a GPU and run on Kaggle (`celltype_transfer/gpu/README.md`).

| # | Step (file) | What it does | Gate |
|---|---|---|---|
| 1 | `load_cohorts.py --build` | reads each dataset into one standard table (`work/raw/`) | 0 |
| 2 | `resolve_markers.py --offline` | resolves marker names to (gene, epitope, modification) triples | 0b |
| 3 | `harmonise_values.py` (`--bakeoff`) | per-cohort rank values, 40,000-cell sample | 1 |
| 4 | `build_label_space.py --expect gate1b_v4_expect.csv` | the shared label space, from marker signatures | 1b |
| 5 | `build_marker_vocabulary.py` | the frozen 109-triple vocabulary and the wide value tables | – |
| 6 | `build_label_confidence.py` | per-cell label confidence (only UPMC has a real one); needs step 5 | – |
| 7 | `build_fold_label_spaces.py --folds` | one label space per fold, built without the held-out cohort | 1b-fold |
| 8 | `build_neighbour_graph.py` | 15 nearest neighbours per cell, from the full raw tables | 4 (checks 6–7) |
| 9 | `pretrain_masked_markers.py --check`, `--loto` | masked-marker pretraining; fold-local warm starts | 2 |
| 10 | `train_prototype_classifier.py --ablate-losses` | the headline LOCO classifier and its loss ablation | 6 |
| 11 | `train_adversarial_encoder.py --lambda-sweep` | the slide adversary, λ sweep | 3 |
| 12 | `train_spatial_context.py --gate` | does the spatial neighbourhood help? | 4 |
| 13 | `compare_external_baseline.py --gate` | the published MAPS method on the same folds | 10 |
| 14 | `compare_labels_to_clusters.py` | native labels vs unsupervised clusters | – |

Shared helpers: `config.py` (the cohort registry and every path), `splits.py` (the patient split), `metrics.py`, `loaders.py`, `panel_utils.py`, `models/` (the network pieces). Pre-registered gate rules live in `celltype_transfer/declared/gate*_expect.csv`. `celltype_transfer/tests/golden.py --check` proves a code change did not change any weight or score.

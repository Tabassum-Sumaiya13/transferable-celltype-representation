# Chapter 4 — Results

**Cross-cohort cell-type annotation for spatial proteomics, evaluated leave-one-cohort-out**

All numbers in this chapter come from the full re-run of 2026-09-18/19. That run reproduced the earlier run exactly (Section 4.10). The protocol is seven-fold leave-one-cohort-out (LOCO) with fold-local label spaces and patient-level splits (Chapter 3). Unless stated otherwise, "macro-F1" means the **core macro-F1** averaged over the seven folds, and every paired test has $n = 7$ folds.

---

## 4.0 Summary of findings

| # | Finding | Status |
|---|---|---|
| 1 | The shipped model transfers to unseen cohorts: LOCO macro-F1 **0.3151**, about 5× random (0.0591) and far above majority (0.0065). It beats both on 7 of 7 folds. | Supported (paired 95% CI excludes zero) |
| 2 | Gate 6 **fails** its pre-registered check 4: the prototype head beats a linear head by +0.0192, against a required +0.02. | Pre-registered failure, not waived |
| 3 | Masked-marker pretraining with an `[ABSENT]` token transfers marker relationships across panels (Gate 2 PASS, $R^2$ 0.2646 vs 0.2192 for a 9-marker control). | Supported |
| 4 | Zero-filling unmeasured markers destroys transfer: MAPS with zero-fill scores 0.0166 against 0.3324 on the shared 9 markers. | Supported (CI excludes zero) |
| 5 | The published MAPS baseline and this project are **statistically indistinguishable** (difference −0.0173, CI [−0.0986, +0.0640]). | Tie |
| 6 | Spatial context, the slide adversary, VICReg and confidence weighting do **not** measurably improve the headline. | Not supported |
| 7 | The label space shipped **under waiver**: 4 of 9 declared cases passed, 5 failed and were waived in writing. | Pre-registered failure, waived |
| 8 | The Danenberg fold is weak for every method (0.05–0.10): 16 of its 29 labels have no matching cluster in its fold space. | Limitation |
| 9 | Post-hoc: the main loss comes from having no labels in the target cohort, not from the encoder, decoder or label space. | Exploratory |

The main message is limited: transfer is real and clearly above chance, but with seven folds no design choice in this project can be shown to beat another, including the published baseline.

---

## 4.1 Evaluation metrics and how results were checked

### 4.1.1 Primary metric: macro-F1

For each class $k$ present in the held-out cohort's truth, with true positives $TP_k$, false positives $FP_k$ and false negatives $FN_k$:

$$
P_k = \frac{TP_k}{TP_k + FP_k}, \qquad
R_k = \frac{TP_k}{TP_k + FN_k}, \qquad
F1_k = \frac{2 P_k R_k}{P_k + R_k}
$$

$$
\text{macro-F1} = \frac{1}{|\mathcal K|}\sum_{k \in \mathcal K} F1_k
$$

**Why macro-F1.** Every class counts equally, so a model cannot score well by getting the common classes right and ignoring rare ones. On this data a model that always predicts the most common class has high accuracy but macro-F1 near zero.

**Which classes are in $\mathcal K$.** $\mathcal K$ is the set of classes present in the held-out truth. Classes that do not appear in the truth are not scored as zero, because that would measure how many classes the cohort happens to contain, not accuracy. Predicting such a class is still an error: it lowers the precision of the class that was wrongly predicted.

Two versions are reported:

| Name | Classes scored | Use |
|---|---|---|
| **core macro-F1** | present in truth, **minus** clusters flagged `unreliable` | headline |
| all macro-F1 | present in truth, all | reported beside the headline |

NOVEL held-out labels (no cluster in the fold's space admits them) are not in either score. They are reported separately as coverage.

The headline is the unweighted mean of the seven per-fold core macro-F1 values. Folds have different class sets because each fold has its own label space.

### 4.1.2 Secondary annotation metrics

All use the same class set as core macro-F1:

- **Balanced accuracy** — the mean of per-class recall.
- **Cohen's kappa** (Cohen, 1960) — agreement above chance: $\kappa = (p_o - p_e)/(1 - p_e)$, where $p_o$ is observed agreement and $p_e$ is agreement expected from the class frequencies.
- **Weighted F1** — per-class F1 weighted by class support. It rewards the common classes, so it is an upper-side contrast to macro-F1.
- **Support-stratified F1** — mean F1 for classes with < 50, 50–200, 200–1,000 and ≥ 1,000 test cells.
- **Zero-F1 classes** — how many scored classes the model never gets right.

### 4.1.3 Reference predictors

Every fold also scores two trivial predictors with the same metric, on the same cells and class set:

- **Majority** — always predicts the most common class in the **training** cohorts. It never looks at the held-out cohort.
- **Random** — predicts a class uniformly at random (fixed seed).

Both are needed. Under macro-F1, a uniform guess can beat a majority guess when classes are unbalanced, so beating only the majority predictor is not enough.

### 4.1.4 Metrics used by individual stages

| Metric | Stage | What it measures |
|---|---|---|
| Resolution coverage; `never_merge` / `must_merge` pass rate | Gate 0b | whether marker names map to correct molecular identities |
| Kolmogorov–Smirnov (KS) statistic | Gate 1 | largest gap between two cumulative distributions; lower = more similar across cohorts |
| AUROC of "label vs rest of its cohort" on a named marker | Gate 1 | whether a label is high on the marker its name implies (threshold 0.75) |
| Masked-marker reconstruction $R^2 = 1 - \text{MSE}_{\text{model}}/\text{MSE}_{\text{train mean}}$ | Gates 1, 2 | how well hidden marker values are predicted from the rest; 0 = no better than the mean |
| Adjusted Rand index (ARI; Hubert & Arabie, 1985) | Gate 1b | agreement between two partitions, corrected for chance (0 = chance, 1 = identical) |
| `cohort_ari` | Gate 1b | ARI between clusters and cohort IDs; high = clusters are just cohorts |
| LOCO stability | Gate 1b | mean ARI between the full clustering and re-derivations with one cohort removed |
| Normalised mutual information (NMI; Strehl & Ghosh, 2002) | Labels vs clusters | shared information between two labellings, 0 to 1 |
| Retained slide bits $= \log_2 N_{\text{slides}} - \text{CE}_{\text{bits}}$ | Gate 3 | slide information left in the embedding; 0 = chance |
| Cohort-probe accuracy | Gates 2, 3 | how well a classifier recovers cohort identity from the embedding |
| Minimum pairwise prototype distance; auxiliary $\log\sigma$ movement | Gate 6 | prototype collapse and loss switch-off |

### 4.1.5 Paired statistics

Every method comparison is paired by fold: same held-out cohort, same cells, same label space, one thing changed. For per-fold differences $\delta_1, \dots, \delta_7$:

- **Mean difference** $\bar\delta$.
- **95% t-interval:** $\bar\delta \pm t_{0.975,\,6} \cdot s_\delta / \sqrt 7$, where $s_\delta$ is the sample standard deviation.
- **Exact sign-flip test:** all $2^7 = 128$ ways of flipping the signs of the $\delta_i$ are enumerated. The p-value is the share whose mean has absolute value at least $|\bar\delta|$ (Good, 2005). It makes no assumption about the shape of the differences. With 7 folds, the smallest possible p-value is $2/128 = 0.016$.

**Decision rule (pre-registered in `gate4_expect.csv` and `gate10_expect.csv`).** "A is better than B" is written only if the 95% interval excludes zero. If it spans zero, the result is reported as "not established", whatever the mean says.

### 4.1.6 How results were checked

1. **Pre-registered gates.** Thresholds were written before each run (Chapter 3, Section 3.1.4). A gate result is PASS, FAIL or waived. Waivers carry written evidence and are never presented as passes.
2. **Controls measured in the same run.** A baseline is re-measured inside the run that tests the method (for example the linear head in Gate 6, the core-9 control in Gate 2), so that only one thing changes.
3. **Leakage assertions in code.** For example: no held-out row in a fold's label space; warm starts only from checkpoints trained on subsets of the fold's training cohorts; no neighbour edge across images.
4. **One metric library.** Every arm, including the external baseline, is scored by the same `metrics.py`, from saved per-cell predictions, so no arm is scored by its own reporting code.
5. **Reproduction.** A full re-run from raw data compared every artefact and weight with the previous run (Section 4.10).

---

## 4.2 Data and marker identity (Gates 0 and 0b)

### 4.2.1 Gate 0 — loading

All seven cohorts loaded: **6,055,589 cells, 1,931 images, 1,381 patients**, 141 native labels. Missing data was negligible: 0.001% of Sorin cells lacked coordinates; no patient IDs were missing. Only UPMC ships a per-cell label confidence. One candidate dataset (Risom et al., 2022) was rejected because its public release has no cell coordinates.

### 4.2.2 Gate 0b — marker resolution: **PASS**

| Check | Result |
|---|---|
| Columns resolved automatically (target ≥ 90%) | **298 / 298 (100%)** — 240 proteins, 39 complexes/families, 19 non-protein channels |
| `never_merge` pairs kept apart | **11 / 11** |
| `must_merge` spelling groups unified | **5 / 5** |
| Non-protein channels resolved to a gene | **0 of 22** |
| Markers shared by ≥ 2 cohorts: raw names → resolved triples | **49 → 62 (+13)** |

Resolution sources: HGNC symbol 92, HGNC alias 59, complex table 58, HGNC previous symbol 43, manual 27, UniProt 19. The run used the recorded cache (571 hits, 0 web calls).

**Panel overlap after resolution** (109 protein triples):

| Present in at least N cohorts | 7 | 6 | 5 | 4 | 3 | 2 | 1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Markers | 9 | 13 | 18 | 31 | 39 | 59 | 109 |

The 9 markers in every cohort are CD3, CD4, CD8A, CD68, CD20 (*MS4A1*), FOXP3, HLA-DR, CD31 (*PECAM1*) and pan-keratin. **50 of 109 markers are measured by exactly one cohort.** This is the panel-mismatch problem the token design addresses.

**Meaning.** Resolving names to triples adds 13 shared markers without merging any declared-distinct pair. The largest overlap gain is still small: most of the vocabulary is cohort-specific.

---

## 4.3 Value harmonisation (Gate 1)

### 4.3.1 Distribution overlap (mean pairwise KS across cohorts, 8 shared markers)

| | Raw | V1 | V3 | V2a | V2b | V1+V3 | V2a+V1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mean KS | **0.702** | 0.169 | 0.259 | 0.235 | 0.163 | 0.169 | 0.233 |

Every rank-based arm cuts cross-cohort distribution mismatch by about 3–4×. This confirms that the arrival-scale problem is fixed. It **cannot rank the arms**: a per-group ECDF makes each group's distribution uniform by construction, so the remaining differences mostly reflect sampling.

### 4.3.2 Composition skew (median shift of the same label on skewed vs balanced slides)

Image-level ranking (V1) invents negative values on tumour-dominated slides, as predicted. The shift for V1 was negative in all 6 cohorts tested (for example Keren −0.336, Sorin −0.220), while the cohort-level V3 was much smaller (Keren −0.134, Sorin −0.040).

### 4.3.3 Label–marker consistency (AUROC ≥ 0.75, 40 declared assertions)

| | V1 | **V3** | V2a | V2b | V1+V3 | V2a+V1 |
|---|---:|---:|---:|---:|---:|---:|
| Assertions passing | 33 | **34** | 33 | 25 | 33 | 33 |

The recurring failures are biologically explainable: CD20 is weak in several IMC/MIBI panels, pan-keratin is shared by sibling tumour labels (which dilutes a "vs all others" AUROC), and Ferguson's CD31 does not mark its own endothelial label (AUROC 0.463; CD13 marks it at 0.83), a known data defect.

### 4.3.4 LOCO masked-marker $R^2$ (the decision metric)

| Held out | V1 | **V3** | V2a | V2b | V1+V3 | V2a+V1 |
|---|---:|---:|---:|---:|---:|---:|
| CRC | 0.119 | 0.138 | 0.124 | 0.189 | 0.146 | 0.110 |
| UPMC | 0.141 | 0.198 | 0.197 | 0.188 | 0.189 | 0.196 |
| Keren | 0.091 | 0.152 | 0.130 | 0.104 | 0.146 | 0.141 |
| Ferguson | 0.162 | 0.452 | 0.442 | 0.517 | 0.437 | 0.457 |
| Phillips | 0.192 | 0.207 | 0.204 | 0.237 | 0.192 | 0.198 |
| Danenberg | 0.135 | 0.281 | 0.276 | 0.327 | 0.264 | 0.263 |
| Sorin | 0.157 | 0.153 | 0.163 | 0.104 | 0.156 | 0.135 |
| **Mean** | 0.142 | **0.226** | 0.219 | **0.238** | 0.218 | 0.214 |

### 4.3.5 Decision

- **Image-level ranking (V1) is clearly worst** (0.142), consistent with the composition-skew result.
- **V2b has the highest mean** (0.238 vs 0.226 for V3), but the lead is small (+0.012), it loses 3 of 7 folds, its FiLM correction sits at the bound (|γ−1| = 0.30) in every fold, which suggests it is fitting cohort identity, and it passes the fewest biology assertions (25/40).
- The protocol pre-registered **V3**, and V3 is used throughout. This is a declared deviation from the bake-off's raw winner, not a silent one.

---

## 4.4 Shared label space (Gate 1b)

### 4.4.1 Whole-roster build

- **Input:** 130 native labels after QC (11 labels and 233,808 cells removed as unlabelled, not-a-cell, unassigned, ambiguous or under 100 cells).
- **Cut:** $\tau = 0.800$, the most stable cut among those passing all guards. **32 of 35** candidate cuts were rejected by the guards.
- **At the cut:** stability 0.794, `cohort_ari` 0.078 (cap 0.2), 99.3% of cells in clusters spanning ≥ 2 cohorts.
- **Output:** **21 clusters**, 16 spanning two or more cohorts and 5 held by a single cohort.

Clusters that recover a cell type across many cohorts:

| Cluster (named by markers) | Cohorts | Native labels merged |
|---|---:|---|
| `MS4A1+ PTPRC+` (B cells) | **7** | B cells of all 7 cohorts |
| `CD8A+ CD3+ LAG3+` (CD8 T) | **7** | CD8 T cells of all 7 cohorts, plus Phillips intraepithelial tumour cells |
| `CD3+ CD4+ PTPRC+` (CD4 T / Treg) | **7** | 16 labels: CD4 T, CD3 T and Tregs across cohorts |
| `CD163+ CD68+ CD274+` (macrophages) | **7** | 11 labels, including all Phillips M1/M2 macrophages |
| `CD34+ COL4+` (endothelium) | **7** | 9 labels: vasculature, Vessel, Endothelial, EC, lymphatics |
| pan-keratin+ (tumour / epithelium) | 6 | 10 labels: CRC tumour, 5 UPMC tumour sub-labels, Keren keratin+ tumour, Ferguson SC, Phillips epithelium, Sorin Cancer |

The derived space also contains **problem clusters**:

- The **largest cluster** (21 labels, 4 cohorts) mixes Danenberg epithelial sub-types with CRC stroma, smooth muscle, granulocytes and nerves.
- **Cluster 14** is a residual (named only by a negative marker, `CD274−`) that holds 15 labels, including fibroblasts, Sorin monocytes, neutrophils and NK cells.
- Check 2b found **4 clusters that break a biological exclusion** (for example a T-cell label and an epithelial label in one cluster), covering 37 labels and 20.8% of cells.

### 4.4.2 Declared hard cases

| Case | Expected | Result |
|---|---|---|
| endothelium (across cohorts) | same | PASS |
| B cells | same | PASS |
| CD8 T cells | same | PASS |
| CD4 vs CD8 T | different | PASS |
| same name, different type (CRC vs Phillips `tumor cells`) | different | PASS |
| tumour vs T cell | different | PASS |
| tumour / epithelium across tissues | same | **FAIL, waived** — UPMC `Tumor` joins Keren `Tumor`; UPMC `Tumor` is only 8th of 16 UPMC labels on keratin |
| stroma | same | **FAIL, waived** — fibroblast-specific markers (PDGFRB, FSP1) exist in only one cohort (Danenberg); without Danenberg, LOCO ARI of the stroma split is −0.018 |
| fibroblasts within Danenberg | same | **FAIL, waived** — four fibroblast labels split three ways |
| stroma bridged via Danenberg | same | **FAIL, waived** — they do share a cluster, but it is the residual cluster, which scoring does not credit |
| tumour including Danenberg | same | **FAIL, waived** — Danenberg grades its epithelium by keratin level, so its CK-low/medium labels sit mid-range |
| macrophages (additional case) | same | FAIL (split across two clusters) |
| nesting (2 cases) | nested | FAIL |

Of the 9 **required** cases, 4 passed and 5 failed and were waived after the run.

**Verdict: Gate 1b SHIPPED UNDER WAIVER — NOT A PASS.** The other checks passed: the cohort guard, 100% of label pairs directly comparable (≥ 8 informative shared markers), 4 of 15 per-branch splits accepted, 21 of 21 clusters coherent, and an acyclic nesting graph with 34 edges.

**Meaning.** Marker-signature alignment recovers the major immune lineages and endothelium across all seven cohorts **with no text and no hand mapping**. It fails where the needed markers are missing from most panels (fibroblasts) and where cohorts annotate one compartment at very different depths (Danenberg's epithelium).

### 4.4.3 Fold-local label spaces

| Held out | $\tau$ | Cut rule | Clusters | Unreliable clusters | Held-out labels | NOVEL labels | Share of held-out cells NOVEL |
|---|---:|---|---:|---:|---:|---:|---:|
| CRC | 0.825 | nearest-feasible | 16 | 4 | 23 | 1 | 0.1% |
| Danenberg | 0.600 | shipped rule | 41 | 4 | 29 | **16** | **42.5%** |
| Keren | 0.800 | nearest-feasible | 20 | 3 | 16 | 0 | 0% |
| Phillips | 0.775 | shipped rule | 21 | 3 | 21 | 1 | 0.4% |
| Sorin | 0.825 | shipped rule | 23 | 4 | 16 | 0 | 0% |
| UPMC | 0.775 | nearest-feasible | 23 | 4 | 16 | 0 | 0% |
| Ferguson | 0.800 | shipped rule | 19 | 3 | 9 | 1 | 9.1% |

- Fold spaces agree well with the whole-roster space (cell-weighted ARI 0.78–0.93 on training labels).
- **Three folds (CRC, Keren, UPMC) had no cut that passed every guard** and use the post-hoc nearest-feasible cut (guard violation 0.9–3.7%). Their numbers carry that flag.
- **Danenberg is the extreme case.** Without Danenberg, the training cohorts cannot place 16 of its 29 labels, including its B cells and most epithelial sub-types, so 42.5% of its cells are outside the closed-set score.

### 4.4.4 Native labels vs derived clusters (descriptive)

| Measure | Value |
|---|---:|
| Cell-weighted ARI (native label vs cluster) | 0.401 |
| Cell-weighted NMI | 0.741 |
| Unweighted NMI | 0.716 |

The high NMI with a moderate ARI means clusters are mostly **unions** of native labels: the derived space is coarser than native annotation, as intended, but it rarely splits one native label across clusters.

---

## 4.5 Masked-marker pretraining (Gate 2): **PASS**

### 4.5.1 Gate checks

| Check | Result | Detail |
|---|---|---|
| 1. Per-marker reconstruction $R^2$ on held-out patients | PASS | median $R^2$ **0.5087** over 262 kept (cohort, marker) pairs; none below zero |
| 2. Flat-marker exclusion is justified | PASS | the 14 excluded pairs have median $R^2$ 0.089 (vs 0.506 kept), with extreme values down to −9,212 |
| 3. Full panel beats the 9-marker core | PASS | full panel **0.2646** vs core-9 control 0.2192 vs Gate 1 MLP 0.169 |
| 4. `[ABSENT]` token (Arm B) vs measured-only (Arm A) | decided | ship **Arm B** |

### 4.5.2 Within-cohort reconstruction (held-out patients)

| Cohort | Markers kept | Median $R^2$ | Min | Max |
|---|---:|---:|---:|---:|
| CRC | 56 | 0.603 | 0.261 | 0.813 |
| Danenberg | 36 | 0.472 | 0.101 | 0.804 |
| Keren | 28 | 0.450 | 0.002 | 0.836 |
| Phillips | 57 | 0.479 | 0.016 | 0.769 |
| Sorin | 12 | 0.343 | 0.060 | 0.604 |
| UPMC | 39 | 0.418 | 0.155 | 0.751 |
| Ferguson | 34 | 0.643 | 0.323 | 0.852 |
| **All** | **262** | **0.509** | 0.002 | 0.852 |

### 4.5.3 Cross-cohort (LOCO) reconstruction on the 9 core markers

| Held out | Arm A (measured only) | **Arm B (`[ABSENT]`)** | Core-9 control | B − control |
|---|---:|---:|---:|---:|
| CRC | 0.304 | 0.308 | 0.137 | +0.171 |
| UPMC | 0.214 | 0.207 | 0.192 | +0.016 |
| Keren | 0.214 | 0.232 | 0.215 | +0.018 |
| Ferguson | 0.434 | 0.432 | 0.348 | +0.084 |
| Phillips | 0.279 | 0.254 | 0.188 | +0.066 |
| Danenberg | 0.247 | 0.321 | 0.273 | +0.048 |
| Sorin | **0.039** | **0.097** | 0.183 | −0.085 |
| **Mean** | 0.247 | **0.265** | 0.219 | **+0.045** |

**Meaning.**

- A wider panel helps transfer (+0.045 over the same architecture on 9 markers). The gain comes from panel width, not from swapping the MLP for a transformer.
- The `[ABSENT]` token matters most on **Sorin**, the smallest panel (17 markers vs 36–57 elsewhere), where Arm B more than doubles Arm A. With `[ABSENT]` every cohort presents 109 tokens, so the encoder never meets a set size it did not train on.
- Sorin is also the one fold where the 9-marker control wins. Transfer to a very small panel remains the weakest point.
- A 7-way cohort probe on the embedding reaches accuracy 1.000 for **both** arms (chance 0.143). So the `[ABSENT]` token is not the only source of cohort information: the identity embeddings of the *measured* markers already fingerprint the panel.

**Leave-one-tissue-out.** Holding out a whole tissue instead of one cohort changed core-9 $R^2$ by only −0.006 to +0.014 (breast and skin). This is too small to show a tissue effect on reconstruction.

---

## 4.6 Headline classifier (Gate 6): **FAIL (check 4)**

### 4.6.1 Gate checks

| Check | Result | Detail |
|---|---|---|
| 1. Auxiliary sigma does not run away | PASS | largest log-σ move 1.129 (cap 3.0) |
| 2. Prototypes do not collapse | PASS | minimum distance never below floor 0.0233 (start 0.0466); no pair frozen |
| 3. VICReg earns its place | **drop VICReg** | 3 losses 0.2956 vs 2 losses 0.3151 (−0.0195) |
| 3b. Slide adversary earns its place | **drop adversary** | 0.2964 vs 0.3151 (−0.0187) |
| 4. Prototype head beats linear head by ≥ 0.02 | **FAIL** | 0.3151 vs 0.2958, margin **+0.0192** |
| 6. Confidence weighting (diagnostic) | no effect | on − off +0.0001, CI [−0.034, +0.034], p = 1.000 (6 folds) |

**Shipped configuration:** `proto2` — prototype head, cell-type + masked-marker losses. **Headline LOCO macro-F1: 0.3151.** The gate fails check 4 by 0.0008. The failure is **not waived**.

### 4.6.2 Per-fold results (core macro-F1)

| Held out | Cut rule | Classes in space | `proto2` (shipped) | `proto3` (+VICReg) | `linear` | `proto2adv` | Majority | Random |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CRC | nearest-feasible | 16 | 0.3042 | 0.3053 | 0.2800 | 0.2720 | 0.0000 | 0.0696 |
| UPMC | nearest-feasible | 23 | **0.4127** | 0.4251 | 0.4006 | 0.3470 | 0.0000 | 0.0605 |
| Keren | nearest-feasible | 20 | 0.2830 | 0.2618 | 0.3108 | 0.3080 | 0.0000 | 0.0638 |
| Ferguson | shipped rule | 19 | **0.3942** | 0.2636 | 0.2686 | 0.3000 | 0.0299 | 0.0740 |
| Phillips | shipped rule | 21 | 0.3455 | 0.3354 | 0.3776 | 0.3489 | 0.0000 | 0.0523 |
| Danenberg | shipped rule | 41 | **0.0513** | 0.0631 | 0.0963 | 0.0984 | 0.0000 | 0.0385 |
| Sorin | shipped rule | 23 | **0.4147** | 0.4147 | 0.3369 | 0.4002 | 0.0159 | 0.0548 |
| **Mean** | | | **0.3151** | 0.2956 | 0.2958 | 0.2964 | 0.0065 | 0.0591 |

"All-class" macro-F1 (unreliable clusters included) for `proto2` is 0.2975.

### 4.6.3 Paired comparisons

| Comparison | Mean Δ | 95% CI | Sign-flip p | Folds won |
|---|---:|---|---:|---:|
| `proto2` − random | **+0.2560** | **[+0.1450, +0.3671]** | **0.016** | 7/7 |
| `proto2` − majority | **+0.3085** | **[+0.1951, +0.4220]** | **0.016** | 7/7 |
| `proto2` − `linear` (check 4) | +0.0193 | [−0.0390, +0.0775] | 0.469 | 4/7 |
| `proto2` − `proto3` (check 3) | +0.0195 | [−0.0271, +0.0661] | 0.500 | 3/7 |
| `proto2` − `proto2adv` (check 3b) | +0.0187 | [−0.0273, +0.0647] | 0.375 | 4/7 |
| conf on − off (check 6) | +0.0001 | [−0.0343, +0.0344] | 1.000 | — |

(The first two rows and the last three were computed from the per-fold values above with the project's `metrics.paired` function. They were not part of the pre-registered gate report.)

**Meaning.**

- **Transfer is real.** The model beats both trivial predictors on every fold, and these are the only intervals in the project that exclude zero.
- **The ablation decisions are not statistically resolved.** The gate rules chose `proto2` over `proto3`, `linear` and `proto2adv` by mean score, as declared. But all three differences are about 0.02, and all three intervals span zero. The ranking of these arms could change with a different set of cohorts.

### 4.6.4 Secondary metrics for the shipped model

| Held out | Test cells | Scored classes | Zero-F1 classes | Core macro-F1 | Balanced accuracy | Cohen's κ | Weighted F1 | Median class F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CRC | 6,447 | 8 | 1 | 0.304 | 0.369 | 0.331 | 0.317 | 0.271 |
| UPMC | 7,879 | 8 | 0 | 0.413 | 0.382 | 0.638 | 0.522 | 0.423 |
| Keren | 8,000 | 11 | 0 | 0.283 | 0.299 | 0.382 | 0.308 | 0.207 |
| Ferguson | 5,402 | 6 | 0 | 0.394 | 0.443 | 0.430 | 0.353 | 0.402 |
| Phillips | 8,000 | 9 | 2 | 0.346 | 0.368 | 0.540 | 0.429 | 0.427 |
| Danenberg | 4,334 | 11 | 6 | 0.051 | 0.038 | 0.233 | 0.058 | 0.000 |
| Sorin | 7,390 | 9 | 3 | 0.415 | 0.501 | 0.613 | 0.411 | 0.555 |
| **Mean** | 6,779 | 8.9 | 1.7 | **0.315** | **0.343** | **0.452** | **0.343** | 0.326 |

Kappa is higher than macro-F1 on most folds. This means the large classes are predicted better than the small ones; macro-F1 is pulled down by a few classes the model misses entirely. **12 of the 62 scored classes (19%) have F1 = 0**, and 6 of them are in Danenberg.

### 4.6.5 Per-class results, grouped by lineage

Classes are grouped by the dominant native label of the held-out cohort in each cluster. The grouping is for reading only.

| Lineage | Classes | Mean F1 | Median F1 | Zero-F1 |
|---|---:|---:|---:|---:|
| B cells | 5 | **0.625** | 0.599 | 0 |
| CD8+ / cytotoxic T | 7 | **0.543** | 0.658 | 1 |
| Tumour / epithelial | 8 | 0.428 | 0.468 | 1 |
| Endothelial / lymphatic | 4 | 0.327 | 0.291 | 0 |
| CD4+ / helper / regulatory T | 9 | 0.279 | 0.292 | 1 |
| Macrophage / monocyte / granulocyte | 13 | 0.278 | 0.309 | 1 |
| Dendritic cells | 5 | 0.093 | 0.062 | 2 |
| Stromal / fibroblast | 4 | 0.070 | 0.070 | 1 |
| Mixed / other | 7 | 0.012 | 0.000 | 5 |

Selected per-class results:

| Fold | Class (held-out native labels) | Support | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| Sorin | B cell | 459 | 0.885 | 0.704 | **0.784** |
| Keren | CD8_T | 710 | 0.639 | 0.958 | **0.767** |
| Sorin | Treg | 449 | 0.710 | 0.760 | **0.734** |
| Phillips | epithelium | 704 | 0.729 | 0.734 | **0.732** |
| UPMC | six tumour labels | 2,985 | 0.839 | 0.622 | **0.715** |
| UPMC | Vessel | 496 | 0.633 | 0.819 | **0.714** |
| CRC | CD8 T, CD4 T CD45RO+, Tregs, CD4 T | 1,628 | 0.553 | 0.933 | 0.694 |
| Ferguson | BC (B cells) | 447 | 0.620 | 0.770 | 0.687 |
| CRC | stroma, vasculature, adipocytes, lymphatics | 1,281 | 0.255 | 0.077 | 0.118 |
| Ferguson | TC_CD4 | 962 | 0.467 | 0.007 | 0.014 |
| Phillips | tumour cells (CTCL); CD4 T | 1,087 | 0.786 | 0.010 | 0.020 |
| Sorin | Th; T other | 1,046 | 0.000 | 0.000 | 0.000 |
| Danenberg | CD8+ T cells | 415 | 0.000 | 0.000 | 0.000 |

**Meaning.**

- **The model transfers cell types that have a strong, widely measured lineage marker.** B cells (CD20), CD8 T cells (CD8A), endothelium (CD31/CD34) and keratin-positive tumour are all measured by all or nearly all cohorts.
- **It fails where the defining markers are rare or the labels mix lineages.** Stroma and fibroblasts have no fibroblast-specific marker outside Danenberg. Dendritic cells are rare and inconsistently defined. "Mixed / other" clusters combine unlike labels.
- **CD4 T cells are the most unstable immune class.** In Ferguson, Sorin and Phillips, CD4 T cells are mostly predicted as another class. In Phillips this is partly biological: its "tumour cells" are malignant CD4 T cells (cutaneous T-cell lymphoma).
- **Many failures are low-recall, high-precision** (for example Ferguson TC_CD4: precision 0.47, recall 0.007). The model rarely predicts these classes; the cells are absorbed by a neighbouring class.
- **Support is not the explanation.** 47 of 62 scored classes have 200–1,000 test cells, and zero-F1 classes are not especially small.

### 4.6.6 Training diagnostics

- **Sigma:** the masked-marker log-σ moved at most 1.129 from its start, so the auxiliary loss was never switched off.
- **Collapse:** the minimum prototype distance stayed above half its initial value in every fold; no pair was frozen.
- **Drift:** several prototypes moved far from their initial marker signatures (up to 1.08 cosine distance, for a `FUT4+ NCAM1+ B3GAT1+` cluster). This means the model disagrees with the label space on where those clusters lie, which fits the label-space problems in Section 4.4.

---

## 4.7 Slide adversary (Gate 3): **PASS** (not shipped)

Gate 3 uses a plain linear head, so its absolute numbers are not comparable with Gate 6's prototype head.

### 4.7.1 λ sweep

| λ | LOCO macro-F1 | Fresh-probe slide bits | Co-trained slide bits | Cohort accuracy |
|---:|---:|---:|---:|---:|
| 0 | 0.2755 | 4.46 | 3.18 | 0.780 |
| **0.01** | **0.3119** | 4.38 | 2.13 | 0.567 |
| 0.03 | 0.3054 | 4.26 | −2.70 | 0.278 |
| 0.1 | 0.2549 | 3.63 | −0.79 | 0.209 |
| 0.3 | 0.2501 | 2.74 | −3.51 | 0.177 |

Majority 0.0065 and random 0.0591 on the same folds; cohort-accuracy chance is 0.167.

| Check | Result |
|---|---|
| 1. Beats majority by ≥ 0.05 and beats random | PASS (+0.305 and +0.253) |
| 2. λ selection | ships **λ = 0.01** (+0.036 over λ = 0; wins 6 of 7 folds) |
| 3. Slide information falls as λ rises (fresh probe) | yes: 4.46 → 4.38 → 4.26 → 3.63 → 2.74 bits |
| 4. Cohort guard (inverted: must stay **above** 0.25) | OK: 0.567 |
| 5. Hiding vs removing | at λ = 0.01 the fresh probe recovers 4.38 bits vs 2.13 for the co-trained head |

### 4.7.2 Meaning

- **A small adversary helps a linear head** (+0.036).
- **It mostly hides slide information rather than removing it.** A fresh probe still recovers almost all the slide bits (4.38 vs 4.46 at λ = 0), while the co-trained head it fought drops to 2.13. The negative co-trained values at higher λ show the encoder beating its own discriminator, not erasing the signal.
- **Larger λ erases tissue biology.** At λ ≥ 0.1, cohort accuracy approaches chance and macro-F1 falls. This is the expected failure when cohort is confounded with tissue.
- **The adversary does not transfer to the shipped head.** Added to the prototype classifier at the same λ, it **lowers** macro-F1 by 0.0187 (Section 4.6). Why it helps one head and hurts the other has not been established. One untested explanation: prototypes initialised from marker signatures already give the classifier cohort-neutral targets.

---

## 4.8 Spatial context (Gate 4): **PASS, but the effect is not established**

### 4.8.1 Arms and gate

| Arm | What it sees | LOCO macro-F1 |
|---|---|---:|
| `cell` | the cell's own markers | 0.2956 |
| `neigh` | + 15 neighbours, pooled | 0.3063 |
| `shuffle` | + another cell's neighbourhood from the same image | 0.2932 |
| `ctx` | `neigh` + context loss | **0.3164** |

| Check | Result | Detail |
|---|---|---|
| 1. `neigh` beats `cell` | PASS | +0.0107 |
| 2. `neigh` beats `shuffle` | PASS | +0.0130 |
| 2b. `shuffle` falls back toward `cell` | PASS | |shuffle − cell| = 0.0023 |
| 3. Paired 95% CI on (`neigh` − `cell`) | **spans zero** | [−0.0239, +0.0453], p = 0.547 |
| 4. Context loss earns its place (optional) | ship `ctx` | +0.0101 over `neigh` |
| 5. Verdict survives UPMC pixel size × 1.3 and × 0.77 | PASS | check 1 delta +0.0106 and +0.0109 |
| 8. Shuffle is a real derangement | PASS | 100% of cells moved |

### 4.8.2 Per fold

| Held out | `cell` | `neigh` | `shuffle` | `ctx` |
|---|---:|---:|---:|---:|
| CRC | 0.3053 | 0.2751 | 0.2972 | 0.3162 |
| UPMC | 0.4251 | 0.3983 | 0.3337 | 0.4037 |
| Keren | 0.2618 | 0.2708 | 0.2908 | 0.3268 |
| Ferguson | 0.2636 | 0.2906 | 0.2961 | 0.2988 |
| Phillips | 0.3354 | **0.4168** | 0.3677 | 0.3803 |
| Danenberg | 0.0631 | 0.0661 | 0.0685 | 0.0850 |
| Sorin | 0.4147 | 0.4261 | 0.3987 | 0.4037 |

### 4.8.3 Paired comparisons

| Comparison | Mean Δ | 95% CI | Sign-flip p |
|---|---:|---|---:|
| `neigh` − `cell` | +0.0107 | [−0.0239, +0.0453] | 0.547 |
| `neigh` − `shuffle` | +0.0130 | [−0.0187, +0.0448] | 0.375 |
| `ctx` − `neigh` | +0.0101 | [−0.0201, +0.0403] | 0.422 |
| `ctx` − `proto2` (shipped model) | +0.0013 | [−0.0429, +0.0455] | 0.969 |
| `neigh` − `proto2` | −0.0088 | [−0.0578, +0.0401] | 0.625 |

### 4.8.4 Neighbour graph

Median edge lengths were 15.8–24.9 µm across cohorts, inside the declared 10–30 µm range. No edge crossed an image. The homotypic fraction (share of neighbours with the same native label) was 0.12–0.26. A descriptive check before any fit agreed with biology: in Keren, tumour cells had the lowest boundary score `d_self` (0.061–0.069, uniform nests) and Tregs, neutrophils and NK cells the highest (0.099–0.116, scattered among unlike cells).

### 4.8.5 Meaning

- The controls behave as designed: the real neighbourhood beats a shuffled one, and the shuffled arm falls back to the cell-only level. So what `neigh` gains is not image-level composition leaking in.
- **But the sentence "spatial context improves transfer" is not earned.** The gain is about +0.01, it is driven mainly by one fold (Phillips, +0.081), and the interval spans zero.
- **Gate 4 compared against a weaker baseline than the shipped one.** Its `cell` arm uses the 3-loss setup (its weights are identical to Gate 6 `proto3`). Against the shipped 2-loss `proto2`, `ctx` is a tie (+0.0013) and `neigh` is slightly worse (−0.0088).

---

## 4.9 External baseline: MAPS (Gate 10): **FAIL (check 1), a statistical tie**

### 4.9.1 Arms

| Method | Markers used | LOCO macro-F1 |
|---|---|---:|
| **This project** (`proto2`) | all 109 (with `[ABSENT]`) | 0.3151 |
| MAPS `core9` | 9 shared by all cohorts | **0.3324** |
| MAPS `shared` | 9–11 shared with the held-out cohort | 0.3090 |
| MAPS `zerofill` | 109, unmeasured = 0 | **0.0166** |

### 4.9.2 Gate

| Check | Result | Detail |
|---|---|---|
| 6. Same cells, split and label space as ours | PASS | matched on label-space hash, class count and split fingerprint in every fold |
| 1. Ours beats MAPS `core9` | **FAIL** | −0.0173 |
| 2b. Ours beats MAPS `shared` | PASS | +0.0061 |
| 3. Does zero-fill rescue MAPS? | no | −0.3158 vs `core9` |
| 4. Paired CI on (ours − `core9`) | spans zero | [−0.0986, +0.0640], p = 0.609 |
| 5. MAPS `core9` beats majority on ≥ 5/7 folds | PASS | 7/7, so the baseline works |

### 4.9.3 Per fold

| Held out | Ours | MAPS `core9` | MAPS `shared` | MAPS `zerofill` |
|---|---:|---:|---:|---:|
| CRC | **0.3042** | 0.2946 | 0.2792 | 0.0000 |
| UPMC | **0.4127** | 0.2660 | 0.2660 | 0.0091 |
| Keren | 0.2830 | **0.4098** | 0.4098 | 0.0178 |
| Phillips | **0.3455** | 0.3272 | 0.3272 | 0.0000 |
| Sorin | 0.4147 | **0.4943** | 0.4307 | 0.0400 |
| Danenberg | 0.0513 | **0.0893** | 0.0893 | 0.0399 |
| Ferguson | 0.3942 | **0.4454** | 0.3604 | 0.0090 |

| Comparison | Mean Δ | 95% CI | Sign-flip p |
|---|---:|---|---:|
| ours − `core9` | −0.0173 | [−0.0986, +0.0640] | 0.609 |
| ours − `shared` | +0.0061 | [−0.0705, +0.0827] | 0.766 |
| `zerofill` − `core9` | **−0.3158** | **[−0.4436, −0.1880]** | **0.016** |

### 4.9.4 Meaning

- **This project does not beat the published baseline.** It wins 3 of 7 folds against `core9` and the interval is wide in both directions. The correct statement is: **a published method ties this pipeline**.
- **Zero-filling is harmful, and this is the one method-level difference in the project with an interval that excludes zero.** Filling unmeasured markers with 0 cuts MAPS from 0.33 to 0.02. This supports the design principle that an unmeasured marker must not be encoded as a negative value. The comparison is confounded, though: it compares two MAPS arms, not `[ABSENT]` against zero-fill inside one model.
- The 9 markers every cohort measures carry most of the transferable signal. The 100 extra markers this project can use do not produce a measurable gain in annotation, even though they help masked-marker reconstruction (Section 4.5).

---

## 4.10 Reproducibility

The full pipeline was re-run from the raw datasets on 2026-09-18/19 with the renamed and cleaned code (`celltype_transfer/`). The declared pass mark was "7-fold mean within ±0.015 of the previous value".

| Compared | Items | Outcome |
|---|---:|---|
| CPU artefacts (steps 1–8) | 77 files | 74 byte-identical; 2 identical in content (differ only in a GPU-arm key and a wall-clock column); 1 metadata label differs (Ferguson `role`) |
| Gate 2 pretraining | 26 fits | 26/26 bit-identical weights |
| Gate 6 classifier | 21 fits | 21/21 bit-identical weights |
| Gate 3 encoder | 35 fits | 35/35 bit-identical weights |
| Gate 4 spatial | 30 fits | 30/30 identical scores |
| Gate 10 MAPS | 21 fits | 21/21 identical scores |
| Labels vs clusters | — | identical (ARI 0.401, NMI 0.741) |

Every arm reproduced at **±0.0000**. The run also found and fixed three problems in how steps were *run* (not in the method): markers had been resolved against live databases instead of the recorded cache; the label space was being scored against an old rules file, which silently dropped 5 `unreliable` flags; and label confidence ran before the tables it filters by existed.

**Meaning.** The results are not an accident of one run or one machine setup. **Limit:** bit-identical reproduction shows stability to re-running with the same seed. It does not show stability across seeds; no multi-seed run was done.

---

## 4.11 Post-hoc diagnostic analyses (exploratory, not pre-registered)

These analyses were run after the gates, on the cached shipped checkpoints, with no retraining. As a sanity check, each harness first reproduced the headline 0.3151 exactly. No threshold was declared for them, so they are **hypothesis-generating, not confirmatory**.

### 4.11.1 Where the gap to supervised performance goes

Using the MAPS MLP as a common decoder (Danenberg excluded because its class set differs between runs):

| Step | Mean macro-F1 | Cost |
|---|---:|---:|
| Ceiling: supervised in-cohort, on harmonised values | 0.684 | — |
| Encoder embedding, cohort seen in training | 0.648 | −0.036 (compression) |
| Encoder embedding, held-out cohort, supervised probe | 0.570 | −0.078 (domain shift) |
| Actual zero-shot LOCO pipeline | 0.315 | **−0.255** (no target labels) |

Further tests separated the pieces of the last step:

| Component | Cost | How measured |
|---|---:|---|
| Encoder compression | 0.036 | values vs in-distribution embedding |
| Domain shift | 0.078 | in-distribution vs held-out embedding |
| Decoder design | ≤ 0.020 | prototype (0.3151), best retrieval/kNN (0.2998), linear (0.2958) |
| Harmonised label vocabulary | **−0.053 (it helps)** | supervised probe → native labels 0.5622 vs → harmonised clusters 0.6147 |
| **No labels from the target cohort** | **≈ 0.30** | supervised vs zero-shot, same space and decoder |

**Meaning.** Every part of the pipeline that could be redesigned costs little, or helps. Almost all of the gap is the price of zero-shot transfer itself.

### 4.11.2 Class coverage governs transfer

For 42 ordered (source cohort → target cohort) pairs, each decoded with one source cohort:

| Predictor of transfer macro-F1 | r (n = 42) |
|---|---:|
| **Share of the target's classes present in the source** | **+0.753** (t = 7.25, p < 0.0001) |
| Source class count | −0.246 |
| Same tissue | −0.139 |
| Source panel size | +0.122 |
| Marker overlap (Jaccard) | +0.120 |
| Same platform | +0.049 |

Pooling six training cohorts (0.3073) did **not** beat the best single source cohort (0.3177). The best single source is an oracle choice, but pooling was better on only 3 of 7 targets.

**Meaning.** Transfer depends on whether the training cohorts contain the target's cell types, not on tissue, platform or panel overlap. This explains the Danenberg fold: no training cohort covers its classes.

A label-free proxy for coverage (agreement between six source decoders) was tested with a pre-declared drop rule and **failed**: it correlated with per-class F1 less well than the model's own confidence (+0.449 vs +0.493). Six sources can agree confidently on the same wrong class.

### 4.11.3 Cost of harmonisation within a cohort

Within-cohort, patient-split logistic regression on native labels:

| | ECDF $u^{\text{coh}}$ | Raw (log1p + z) | Both |
|---|---:|---:|---:|
| Mean macro-F1 (7 cohorts) | 0.619 | 0.675 | **0.695** |

Raw values beat ECDF values in **7 of 7 cohorts** (+0.027 to +0.093). The ECDF buys cross-cohort comparability at the cost of absolute intensity. The two are complementary, which makes feeding both into the encoder a direct next experiment.

### 4.11.4 Few-shot exchange rate

Using the fold embedding that never saw the target, with N labelled cells per native type from the target's training patients (5 seeds, scored on native labels):

| Labelled cells per type | 0 (zero-shot pipeline) | 1 | 2 | 5 | 10 | 50 | 250 | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Macro-F1 | 0.315 | 0.256 | 0.321 | 0.378 | 0.422 | 0.492 | 0.542 | 0.562 |

The zero-shot pipeline is worth more than one labelled cell per type and less than two. The median cohort matches it with about 4–5 labelled cells per type. Two caveats apply: this curve scores native labels while LOCO scores harmonised clusters, and the upper rows are optimistic because rare types could not supply the full N cells.

### 4.11.5 LOCO cannot rank methods with seven cohorts

| Quantity | Value |
|---|---:|
| Spread of mean macro-F1 across six methods tried | 0.037 |
| Standard deviation across folds for one method | 0.127 |
| Standard error of a 7-fold mean | ≈ 0.048 |

Fold-to-fold variation is about 3× larger than the full range of methods tried. As a rough guide, resolving a 0.02 difference at this variance would need on the order of a hundred cohorts. **LOCO with 7 cohorts can show that transfer exists; it cannot choose between methods whose scores differ by about 0.02.**

---

## 4.12 Claims: earned and not earned

| Claim | Earned? | Evidence |
|---|---|---|
| The model transfers to an unseen cohort above chance | **Yes** | 7/7 folds; CI vs random [+0.145, +0.367] |
| Marker-signature alignment recovers B, CD8 T, CD4 T, macrophage and endothelial types across all 7 cohorts without text | **Yes (descriptive)** | Section 4.4.1 |
| Masked pretraining with a wider panel transfers marker relationships better than the 9-marker core | **Yes, on reconstruction** | Gate 2 check 3 |
| Zero-filling unmeasured markers is harmful | **Yes (in MAPS)** | CI [−0.444, −0.188] |
| The prototype head beats a linear head | **No** | Gate 6 check 4 FAIL; CI spans zero |
| This project beats the published baseline | **No — a tie** | CI [−0.099, +0.064] |
| Spatial context improves transfer | **No** | CI spans zero; tie against shipped model |
| A slide adversary helps | **No** (helps a linear head, hurts the shipped head) | Gate 3 vs Gate 6 check 3b |
| Confidence weighting helps | **No** | +0.0001 |
| The shared label space is correct | **No — it is a hypothesis, shipped under waiver** | 5/9 required cases waived |

---

## 4.13 Limitations

1. **n = 7 folds.** Differences of about ±0.04 are within noise, and every intervention's interval spans zero.
2. **One seed.** Reproduction was exact, but variation across seeds and cell draws was not measured.
3. **Label space under waiver.** 5 of 9 required cases failed. Three fold spaces (CRC, Keren, UPMC) use a post-hoc cut rule.
4. **Danenberg.** 42.5% of its cells are NOVEL in its fold, and every method scores 0.05–0.10 on it. Its fold pulls the mean down by about 0.04.
5. **Fibroblast markers.** PDGFRB and FSP1 exist in only one cohort, so stroma cannot be aligned across cohorts.
6. **Assumed pixel sizes** for Phillips and UPMC. The spatial verdict survived ×1.3 and ×0.77 rescaling for UPMC.
7. **Gate 4 baseline.** The spatial comparison was made against the 3-loss model, not the shipped 2-loss model.
8. **External ontology validation not repeated.** The protocol calls for comparison with CellMarker 2.0 (Hu et al., 2023) and the Cell Ontology (Diehl et al., 2016). That stage belonged to the older pipeline and was not re-run on this roster, so the harmonised space has no external validation in this chapter.
9. **Open-set behaviour.** NOVEL labels are excluded from the closed-set score, but abstention (rejecting a novel cell at prediction time) was not evaluated on this roster.

---

## References

- Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1), 37–46.
- Danenberg, E., Bardwell, H., Zanotelli, V. R. T., et al. (2022). Breast tumor microenvironment structures are associated with genomic features and clinical outcome. *Nature Genetics*, 54, 660–669. https://doi.org/10.1038/s41588-022-01041-y
- Diehl, A. D., Meehan, T. F., Bradford, Y. M., et al. (2016). The Cell Ontology 2016: enhanced content, modularization, and ontology interoperability. *Journal of Biomedical Semantics*, 7, 44.
- Ferguson, A. L., Sharman, A. R., Allison, R. O., et al. (2022). High-dimensional and spatial analysis reveals immune landscape–dependent progression in cutaneous squamous cell carcinoma. *Clinical Cancer Research*, 28(21), 4677–4688. https://doi.org/10.1158/1078-0432.CCR-22-1332
- Good, P. (2005). *Permutation, Parametric and Bootstrap Tests of Hypotheses* (3rd ed.). Springer.
- Hanley, J. A., & McNeil, B. J. (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology*, 143(1), 29–36.
- Hu, C., Li, T., Xu, Y., et al. (2023). CellMarker 2.0: an updated database of manually curated cell markers in human/mouse and web tools based on scRNA-seq data. *Nucleic Acids Research*, 51(D1), D870–D876.
- Hubert, L., & Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2(1), 193–218.
- Keren, L., Bosse, M., Marquez, D., et al. (2018). A structured tumor-immune microenvironment in triple negative breast cancer revealed by multiplexed ion beam imaging. *Cell*, 174(6), 1373–1387. https://doi.org/10.1016/j.cell.2018.08.039
- Massey, F. J. (1951). The Kolmogorov–Smirnov test for goodness of fit. *Journal of the American Statistical Association*, 46(253), 68–78.
- Phillips, D., Matusiak, M., Gutierrez, B. R., et al. (2021). Immune cell topography predicts response to PD-1 blockade in cutaneous T cell lymphoma. *Nature Communications*, 12, 6726. https://doi.org/10.1038/s41467-021-26974-6
- Risom, T., Glass, D. R., Averbukh, I., et al. (2022). Transition to invasive breast cancer is associated with progressive changes in the structure and composition of tumor stroma. *Cell*, 185(2), 299–310. https://doi.org/10.1016/j.cell.2021.12.023
- Schürch, C. M., Bhate, S. S., Barlow, G. L., et al. (2020). Coordinated cellular neighborhoods orchestrate antitumoral immunity at the colorectal cancer invasive front. *Cell*, 182(5), 1341–1359. https://doi.org/10.1016/j.cell.2020.07.005
- Shaban, M., Bai, Y., Qiu, H., et al. (2024). MAPS: pathologist-level cell type annotation from tissue images through machine learning. *Nature Communications*, 15, 28.
- Sorin, M., Rezanejad, M., Karimi, E., et al. (2023). Single-cell spatial landscapes of the lung tumour immune microenvironment. *Nature*, 614, 548–554. https://doi.org/10.1038/s41586-022-05672-3
- Strehl, A., & Ghosh, J. (2002). Cluster ensembles — a knowledge reuse framework for combining multiple partitions. *Journal of Machine Learning Research*, 3, 583–617.
- Wu, Z., Trevino, A. E., Wu, E., et al. (2022). Graph deep learning for the characterization of tumour microenvironments from spatial protein profiles in tissue specimens. *Nature Biomedical Engineering*, 6, 1435–1448. https://doi.org/10.1038/s41551-022-00951-w

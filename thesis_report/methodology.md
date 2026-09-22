# Chapter 3 — Methodology

**Cross-cohort cell-type annotation for spatial proteomics, evaluated leave-one-cohort-out**

---

## 3.1 Problem statement and study design

### 3.1.1 The problem

Multiplexed spatial proteomics platforms (CODEX, imaging mass cytometry (IMC) and MIBI-TOF) measure tens of proteins per cell while keeping each cell's position in the tissue. Every published study annotates its own cells, and a model trained on one study usually fails on another. Three kinds of shift happen at the same time:

1. **Feature-space shift (panel mismatch).** Each study uses its own antibody panel. In this study the panels range from 18 to 59 marker columns (17 to 57 protein markers after non-protein channels are removed). Of the 109 protein identities in the combined vocabulary, only 9 are measured by all seven cohorts and 50 are measured by exactly one.
2. **Covariate shift (value scale).** Values arrive on different scales: raw fluorescence (CODEX), ion counts (IMC), z-scores (MIBI-TOF), arcsinh-transformed values, and 8-bit quantised values.
3. **Label shift (vocabulary mismatch).** Every study names its cell types independently. The same name can mean different biology (CRC and Phillips both use `tumor cells`, but Phillips is a T-cell lymphoma), and different names can mean the same biology (`vasculature`, `Vessel`, `Endothelial`, `EC`).

In the terms of the dataset-shift literature (Quiñonero-Candela et al., 2009; Ben-David et al., 2010), this is joint covariate, label and feature-space shift.

### 3.1.2 Research question

> Can a model learn a cell representation that transfers to a cohort it has never seen, when that cohort differs in tissue, platform, panel, value scale and label vocabulary?

### 3.1.3 Evaluation principle: leave-one-cohort-out

Random cell-level cross-validation is not a valid test of this question. Cells from the same slide share staining, segmentation and patient biology, so random splits make the task look easier than it is. The primary protocol is therefore **seven-fold leave-one-cohort-out (LOCO)**:

1. Pick one cohort as the held-out cohort.
2. Build every data-dependent artefact that touches labels (label space, prototypes, pretrained weights, classifier) from the other six cohorts only.
3. Make all model-selection decisions (early stopping, loss ablations, hyperparameters) on validation patients from the six training cohorts.
4. Score the held-out cohort once.
5. Repeat until each cohort has been held out.

Section 3.12 lists every step taken to prevent information from the held-out cohort leaking into training.

### 3.1.4 Pre-registration

Every stage of the pipeline ends in a **gate**: a set of checks with pass thresholds. The thresholds and expected cases were written to files (`celltype_transfer/declared/gate*_expect.csv`) **before** the stage was run, and each stage scores itself against its file. Three rules apply:

- A threshold is never changed to make a run pass. If a rule is found to be wrong, the old row stays in the file marked `REPLACED`, with the reason.
- A failure that is accepted anyway stays visible: it is marked `waived = 1`, with written evidence.
- Failed attempts are kept in code comments and reports, not deleted.

This is modelled on pre-registration in clinical and psychological research (Nosek et al., 2018). Its purpose is to stop the method being tuned against the results it is judged on.

### 3.1.5 Pipeline overview

| Step | Stage | Output | Gate |
|---|---|---|---|
| 1 | Load cohorts into one standard table | `work/raw/{cohort}.parquet` | 0 |
| 2 | Resolve marker names to molecular identities | `marker_registry.csv` | 0b |
| 3 | Harmonise values (per-cohort mid-rank ECDF) | `work/values/{cohort}.parquet` | 1 |
| 4 | Derive a shared label space from marker signatures | `label_map.csv`, `prototypes.npy` | 1b |
| 5 | Freeze the marker vocabulary; build wide tables | `panel.json` | — |
| 6 | Per-cell label confidence | `work/label_conf/` | — |
| 7 | Fold-local label spaces (one per LOCO fold) | `work/spaces/` | 1b-fold |
| 8 | Spatial neighbour graph | `{cohort}_neighbours.npz` | 4 (checks 6–7) |
| 9 | Masked-marker pretraining | `pretrain_*.pt` | 2 |
| 10 | Prototype classifier and loss ablation | `classifier_*.pt` | 6 |
| 11 | Encoder with slide adversary (λ sweep) | `encoder_*.pt` | 3 |
| 12 | Spatial context | `spatial_*.pt` | 4 |
| 13 | External baseline (MAPS) | `external_*.pt` | 10 |
| 14 | Native labels vs derived clusters | report only | — |

Steps 1–8 run on CPU. Steps 9–13 ran on Kaggle (2 × NVIDIA Tesla T4).

---

## 3.2 Data

### 3.2.1 Cohorts

Seven public cohorts were used. All are cancer tissue with cell coordinates and author-provided cell-type labels.

| Cohort | Platform | Tissue | Cells | Images | Patients | Marker columns | Native labels | Pixel size (µm) | Values arrive as | Source |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| CRC | CODEX | colorectal | 258,385 | 140 | 35 | 58 | 29 | 0.377 | raw | Schürch et al., 2020 |
| Danenberg | IMC | breast | 1,123,466 | 794 | 718 | 39 | 32 | 1.0 | raw | Danenberg et al., 2022 |
| Keren | MIBI-TOF | breast | 197,678 | 40 | 40 | 49 | 17 | 0.391 | z-score | Keren et al., 2018 |
| Phillips | CODEX | skin (CTCL) | 117,170 | 69 | 14 | 59 | 21 | 0.377* | raw | Phillips et al., 2021 |
| Sorin | IMC | lung | 2,141,875 | 536 | 476 | 18 | 17 | 1.0 | raw uint8 | Sorin et al., 2023 |
| UPMC | CODEX | head and neck | 2,061,102 | 308 | 81 | 39 | 16 | 0.377* | arcsinh | Wu et al., 2022 |
| Ferguson | IMC | skin (cSCC) | 155,913 | 44 | 17 | 36 | 9 | 1.0 | raw | Ferguson et al., 2022 |
| **Total** | | | **6,055,589** | **1,931** | **1,381** | | **141** | | | |

\* Assumed, not published. Phillips and UPMC were imaged on the same Nolan-lab CODEX/Keyence stack as CRC, so CRC's published value was used. For IMC, 1 µm is the laser ablation spot size. Pixel size only matters in the spatial stage; Section 3.11.6 describes the sensitivity test for this assumption.

**Excluded dataset.** Risom et al. (2022) (breast DCIS, MIBI-TOF) was considered and rejected because its public release has no per-cell centroid coordinates, so no spatial graph can be built.

### 3.2.2 Standard table

A single generic loader (`loaders.py`) reads every cohort using a declarative per-cohort specification (`config.py`: base table, joins, marker-column rule, pixel size, value scale). No code anywhere branches on the cohort name, so adding a cohort means adding one specification. Every cohort becomes one table:

`cell_id, cohort, image_id, patient_id, x_px, y_px, area_px2, native_label, label_confidence, <marker columns>`

Stage 0 keeps **every** cell and every label, including labels such as `dirt` or `undefined`. Whether a label is usable as a cell type is decided later, in the label-space stage (Section 3.5.1), not at load time. Stage 0 also audits missing coordinates (about 0.001% of cells, all in Sorin), missing patient IDs (none), and plots the median-sized slide of each cohort to confirm that the coordinate columns were read correctly.

### 3.2.3 Subsampling

From Stage 1 onward, each cohort is represented by a **40,000-cell stratified subsample**. The sample is drawn round-robin over (patient × native label) strata, so rare labels are kept whole and one very large stratum cannot dominate. Normalisation references (the ECDFs) are computed from **all** cells; only the table written to disk is subsampled. The spatial stage is the one exception (Section 3.11.1).

---

## 3.3 Marker identity resolution

### 3.3.1 The identity triple

Marker column names are not identities. The same protein is spelled many ways (`panCK`, `Pan-Keratin`, `Keratin`), and similar names can mean different molecules (CD45, CD45RA and CD45RO are all the gene *PTPRC*, but they mark opposite T-cell states). Each marker column is therefore resolved to a triple:

$$
t = (\text{gene or complex},\ \text{epitope},\ \text{modification})
$$

For example CD45RA → (`HGNC:9666`, `RA`, `none`), phospho-S6 → (`HGNC:10429`, `pan`, `phospho`), and pan-keratin → (`FAMILY:KRT_PAN`, `pan`, `none`). Two columns are merged only when their triples are identical.

### 3.3.2 Resolution procedure

Names are looked up in this order: HGNC approved symbols, HGNC aliases, HGNC previous symbols (Seal et al., 2023), UniProt (UniProt Consortium, 2023), a declared table of protein complexes and families (for example CD3 = CD3D|CD3E|CD3G, HLA-DR), and finally a small declared list of manual overrides. Non-protein channels (DNA dyes, elemental channels such as Na, Fe, Au, background) are flagged `non_protein` and kept out of the biological vocabulary.

The web answers were recorded in a cache (`work/api_cache.json`). The final pipeline resolves **offline** from this cache, because live database answers can change over time and would silently change the vocabulary.

### 3.3.3 Declared correctness checks (Gate 0b)

- **Coverage** ≥ 90% of marker columns resolved automatically.
- **`never_merge`**: 11 declared pairs that must resolve to *different* triples (for example CD45 vs CD45RA, panCK vs CK5, H3K9ac vs H3K27me3).
- **`must_merge`**: 5 declared spelling groups that must resolve to *one* triple (for example the six spellings of pan-keratin).
- **No non-protein leaks**: 22 declared non-protein names must never resolve to a gene. This check exists because `Na` (sodium) once resolved to the gene *XK*, whose previous HGNC symbol is `NA`, after pandas read the string "NA" as a missing value.
- Resolution must share more markers across cohorts than verbatim string matching does.

### 3.3.4 Frozen vocabulary

After resolution the vocabulary is frozen as **109 protein triples**, each with a stable index, and fingerprinted (`109:66147d20`). Every checkpoint stores this fingerprint; a checkpoint with a different fingerprint is refitted, not loaded. A new cohort can only use markers already in the frozen vocabulary; any new marker is reported as dropped rather than changing the meaning of an existing model input.

---

## 3.4 Value harmonisation

### 3.4.1 Per-cohort mid-rank ECDF

Raw values cannot be compared across cohorts: an IMC ion count of 5 and a CODEX fluorescence of 5,000 carry no shared meaning. Each (cohort, marker) is therefore mapped through its empirical cumulative distribution function (ECDF), using mid-ranks for ties:

$$
u_{i,m} \;=\; \frac{\operatorname{rank}^{\text{avg}}\!\left(x_{i,m}\ \middle|\ \{x_{j,m} : j \in c\}\right)}{N_c} \;\in\; (0, 1]
$$

where $x_{i,m}$ is the raw value of cell $i$ on marker $m$, $c$ is the cell's cohort, $N_c$ is the number of cells in the cohort, and $\operatorname{rank}^{\text{avg}}$ gives tied values their average rank (Conover, 1999).

Properties that motivated this choice:

- **Monotone and bin-free.** The order of cells within a cohort is kept exactly, and no histogram bins are introduced (bins create step artefacts in any model that reads the value).
- **Scale-free.** Every marker of every cohort lands on the same $(0,1]$ scale, whatever the platform.
- **Tie-safe.** Sorin arrives as 8-bit integers, so on many markers more than half of its cells share the value 0. Mid-ranks give all of them one value instead of spreading them across an artificial range.

This is a per-cohort form of quantile normalisation (Bolstad et al., 2003). Unlike full quantile normalisation, it does not force cohorts onto a shared reference distribution; it only makes positions comparable.

### 3.4.2 Why cohort level and not image level

Ranking inside each image removes more slide-to-slide drift, but it creates a known artefact. On a slide that is 90% tumour, ranking pushes half of the tumour cells below the median on keratin, so an identical cell reads lower only because of its neighbours. Ranking over the whole cohort avoids this.

### 3.4.3 Normalisation bake-off (Gate 1)

The choice was tested, not assumed. Six arms were compared:

| Arm | Value channel(s) | Learned slide correction |
|---|---|---|
| V1 | image-level ECDF $u^{\text{img}}$ | none |
| **V3** | **cohort-level ECDF $u^{\text{coh}}$** | **none** |
| V2a | $u^{\text{coh}}$ | FiLM, scalar (4 parameters per marker) |
| V2b | $u^{\text{coh}}$ | FiLM, MLP over the whole slide-statistics vector |
| V1+V3 | $u^{\text{img}}, u^{\text{coh}}$ | none |
| V2a+V1 | $u^{\text{coh}}, u^{\text{img}}$ | FiLM, scalar |

FiLM (feature-wise linear modulation; Perez et al., 2018) applies $\tilde u = \gamma u + \beta$ with $\gamma = 1 + \varepsilon \tanh(\cdot)$, $\beta = \varepsilon \tanh(\cdot)$ and $\varepsilon = 0.3$, driven by per-slide summary statistics. The output bound and zero initialisation mean the worst case is "FiLM does nothing", which equals V3.

Arms were scored on the 9 markers measured by every cohort, using five checks: (1) cross-cohort distribution overlap (mean pairwise Kolmogorov–Smirnov statistic); (2) robustness to slide composition; (3) label–marker consistency (AUROC ≥ 0.75 for 40 declared assertions such as "B cells are high on CD20"); (4) LOCO masked-marker reconstruction $R^2$ (the decision metric); (5) how hard FiLM pushes against its bound.

**Decision.** The protocol (`benchmark_protocol.yaml`, `value_transform`) pre-registers **V3, the cohort-level ECDF**, and V3 is used throughout. On the 7-cohort roster V2b has the highest mean $R^2$ (Chapter 4, Section 4.3), but it was not adopted. Its lead was small (+0.012), it lost 3 of 7 folds, it hit its correction bound on every fold (a sign that it was fitting cohort identity rather than slide drift), and it passed only 25 of 40 marker-biology assertions against 34 of 40 for V3.

A second channel proposed in the design (a tanh-squashed robust z-score) was built, measured and dropped. It saturated (5.5–17.5% of cells at |value| > 0.99), and it was redundant: any per-cell function of the raw value built from cohort statistics is a monotone transform of that value, so it carries the same information as $u^{\text{coh}}$ (measured correlation 0.89–0.97).

---

## 3.5 Deriving a shared label space

The seven cohorts have 141 native labels (130 after the label quality control of Section 3.5.1) and no shared dictionary. The shared label space is derived **from marker signatures only**. Text is used only to give clusters readable names; it never enters a distance. Text embeddings were rejected because they fail in both directions: `SC` and `EC` differ by one character but mean squamous carcinoma and endothelial cell, while `CD4 T cell` and `CD8 T cell` are almost identical strings for opposite cell types.

No hand-written mapping is used anywhere, as a method, as a gate, or as a validation source. An earlier hand mapping was removed in protocol v2 because it was written by the same person as the method and so tested nothing external.

Three kinds of label are kept apart throughout this thesis:

- **native** — the author's label, the raw input, never assumed correct;
- **harmonised** — this project's derived clusters, a hypothesis, never called "the correct space";
- **reference** — external, published ontologies.

### 3.5.1 Label quality control

Before alignment, labels are removed if they fall below 100 cells (quantiles on fewer cells are noise) or if a declared QC table (`declared/label_qc.csv`) marks them as not a cell (`dirt`), unassigned (`undefined`, `Unidentified`), unlabelled (Sorin `nan`), or ambiguous multi-type mixtures (for example `tumor cells / immune cells`). Eleven labels (233,808 cells) were removed.

### 3.5.2 Label signatures

For every (cohort, label, marker) the stage stores nine quantiles of $u^{\text{coh}}$ (at 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98), the **mean rank**, the label's prevalence, and the within-label marker co-expression matrix.

The **mean rank** is used as the label's position on each marker. It is the Mann–Whitney statistic, equivalent to the AUROC of "this label vs the rest of its cohort" (Mann & Whitney, 1947; Hanley & McNeil, 1982). The median was tried first and failed on zero-inflated data: in Sorin most cells tie at 0 on most markers, so almost every label had the same median, and only 5 of 17 Sorin markers showed any between-label spread.

### 3.5.3 Within-cohort rescaling

Raw positions are not comparable across cohorts, because an ECDF position depends on cohort composition. Keren is 50% keratin-positive tumour, so every other Keren label is pushed low on keratin. The first build compared raw positions and produced clusters that tracked cohorts, not cell types.

What *is* comparable is a label's position **relative to the other labels in its own cohort**. For each (cohort $c$, marker $m$), with label positions $P_{\ell,m}$:

$$
\tilde P_{\ell,m} = \frac{P_{\ell,m} - \tfrac12\left(\max_{\ell'} P_{\ell',m} + \min_{\ell'} P_{\ell',m}\right)}{\max\!\left(\max_{\ell'} P_{\ell',m} - \min_{\ell'} P_{\ell',m},\ 0.05\right)}
$$

The midrange is used as the centre because it depends only on the most negative and most positive label. The earlier median-label centre moved whenever a cohort split one compartment into many sub-labels: UPMC has six tumour sub-labels, and under median centring UPMC `Tumor` and Ferguson `SC` ended 14× apart on keratin despite similar raw ranks. The same affine map is applied to all nine quantiles of each label.

### 3.5.4 Label-to-label distance

A marker is **informative** for a pair of labels $(a, b)$ from cohorts $c_a, c_b$ when both cohorts measure it and its between-label spread $W_{c,m} = \max_\ell P_{\ell,m} - \min_\ell P_{\ell,m}$ is at least 0.1 in both cohorts. Its weight is

$$
w_{ab,m} = \min(W_{c_a,m},\ W_{c_b,m}) \cdot \mathbb 1\!\left[\min(W_{c_a,m}, W_{c_b,m}) \ge 0.1\right]
$$

The distance is the weighted root-mean-square difference of rescaled positions:

$$
d(a,b) = \sqrt{\frac{\sum_m w_{ab,m}\,\big(\tilde P_{a,m}-\tilde P_{b,m}\big)^2}{\sum_m w_{ab,m}}}
$$

Two further steps complete it:

- **Block normalisation.** For every pair of cohorts, distances are divided by the median distance in that cohort-pair block, $\hat d(a,b) = d(a,b) / \operatorname{median}_{\text{block}}(d)$. This removes a constant offset between cohort pairs that share many or few markers.
- **Evidence floor.** If fewer than $k = 8$ informative markers are shared, the pair is set to a large distance (5.0) and can only join through a third label comparable to both.

The similarity reported in tables is $s(a,b) = \exp(-\hat d(a,b))$.

Several alternatives were built and measured, and they are kept in the code as switches so they are not proposed again: a floored-geometric-mean "veto" distance, positive-fraction agreement, lineage-weighted markers, and robust (p90–p10) scaling. None beat the weighted RMS once declared cases were scored in a way that cannot be satisfied by one giant cluster (Section 3.5.7).

### 3.5.5 Clustering and choice of the cut

Labels are grouped by **average-linkage agglomerative clustering** (UPGMA; Sokal & Michener, 1958; Murtagh & Contreras, 2012) on $\hat d$, cut at height $\tau$. Average linkage was chosen over Leiden community detection because modularity on a dense similarity graph of about 100 nodes returned one giant community plus singletons.

The cut $\tau$ is **never tuned against a reference**. It is chosen over a grid (0.300 to 1.150 in steps of 0.025) by **leave-one-cohort-out stability**: the mean adjusted Rand index (ARI; Hubert & Arabie, 1985) between the full clustering and each re-derivation with one cohort removed. Stability alone is not enough: a clustering that simply equals the cohort partition is perfectly stable (the first build scored 0.972 while mixing tumour, macrophages, T cells and B cells). So cuts must first pass three hard guards:

1. `cohort_ari` ≤ 0.2 (the clustering must not be the cohort partition);
2. no cluster holds more than a quarter of all labels;
3. at least 95% of cells sit in clusters that span two or more cohorts.

Among the cuts that pass, the most stable one is chosen.

### 3.5.6 Per-branch refinement, nesting and naming

- **Per-branch splits.** Each cluster is offered a binary split. The split is kept only if both sides hold at least two labels, both sides span at least two cohorts, at least two cohorts have labels on *both* sides (so a split can never be a cohort boundary in disguise), and the split reproduces with a cohort held out (LOCO ARI ≥ 0.5).
- **Nesting.** A separate *directed* relation records broad-parent/narrow-child structure. The containment of $a$ in $b$ is the fraction of $a$'s q10–q90 range that lies inside $b$'s, combined across markers with a floored geometric mean. An edge needs containment ≥ 0.75 and a clear asymmetry. The graph is made acyclic by contracting strongly connected components (Tarjan, 1972). This graph is reported but **not used in any loss**, because it failed its declared cases.
- **Naming.** Clusters are named from their top discriminative markers (for example `MS4A1+ PTPRC+`). Names are for readability only.

### 3.5.7 Declared hard cases and waivers

Before the run, 9 required biological cases were declared (`gate1b_v4_expect.csv`). Examples: B cells from all cohorts should merge; CD4 and CD8 T cells should stay apart; the same name meaning different types (CRC vs Phillips `tumor cells`) should stay apart; tumour/epithelial labels should merge across tissues. `same` cases are scored **blob-proof**: a pass counts only if the shared cluster is not the largest cluster and is named by at least one positive marker.

Four cases passed and five failed. The five failures were **waived in writing** after the run, with the measured cause recorded for each. The label space therefore **shipped under waiver and is not a pass**. Clusters holding a member of a waived failed case are flagged `unreliable` and are left out of the headline score (Section 3.13.2). They are still used in training.

### 3.5.8 Fold-local label spaces

The whole-roster label space was built with every cohort present, so it can never be used for a LOCO headline. For each LOCO fold a separate label space is built from the **six training cohorts only**, with the same code, switches and cut rule. The held-out cohort's labels are then placed into that frozen partition by an **admission rule**: a held-out label joins the cluster with the smallest average distance to it, if that distance is at most the fold's $\tau$. Otherwise it is marked **NOVEL** and is left out of the closed-set score. It is never forced into a known class.

Two assertions are enforced in code before a fold map is written: no held-out row reaches any fold table, and the training-to-training geometry of every fold is bit-identical to the full build.

**Post-hoc change (declared as such).** In three folds (CRC, Keren, UPMC) no cut passes all guards. The originally declared fallback (the most stable cut over the whole grid) produced degenerate spaces. It was replaced, after this was seen, by the **nearest-feasible cut**: the cut with the smallest relative guard violation. Every number from these three folds carries the flag `nearest_feasible`.

---

## 3.6 Within-cohort data splits

Inside each training cohort, cells are split **by patient** into train / validation / test at 68% / 12% / 20% of patients (`splits.split_masks`). A patient's slides never appear on both sides. Patient-level splitting matters because slide-level splitting leaked patient biology: under the earlier slide split, 100% of CRC, Phillips and Ferguson test cells had a patient with training slides. Every checkpoint stores the split fingerprint (`patient-v1:015916c8`), and a mismatch forces a refit.

Per fold, up to 15,000 training cells and 3,750 validation cells are drawn per training cohort, and up to 8,000 cells from the **held-out cohort's test patients** are scored.

---

## 3.7 Panel-aware cell encoder

### 3.7.1 Why not a fixed feature matrix

A fixed cell × marker matrix forces a value into every column. For a marker a cohort never measured, the only available value is a fill-in, and a zero is a real claim: "this cell is negative." The previous pipeline measured the cost: widening a fixed 19-marker table hurt marker-poor cohorts (Ferguson −0.085 macro-F1). This project instead treats **an unmeasured marker as unknown, not as zero**.

### 3.7.2 Marker tokens

Each cell is a set of 109 tokens, one per vocabulary slot. For cell $i$, slot $m$ and cohort panel $\mathcal M_c$, the input token is

$$
\mathbf h^{(0)}_{i,m} =
\begin{cases}
f_{\text{val}}(u_{i,m}) + \mathbf e_{m} & m \in \mathcal M_c,\ m \notin \mathcal H_i \quad\text{(measured)}\\[3pt]
\mathbf e_{\text{mask}} + \mathbf e_{m} & m \in \mathcal M_c,\ m \in \mathcal H_i \quad\text{(hidden for training)}\\[3pt]
\mathbf e_{\text{absent}} + \mathbf e_{m} & m \notin \mathcal M_c \quad\text{(never measured)}
\end{cases}
$$

- $f_{\text{val}}: \mathbb R \to \mathbb R^{64}$ is a two-layer MLP (1 → 64 → 64, GELU; Hendrycks & Gimpel, 2016) **shared by all markers**, so a marker the value pathway has never seen can still be encoded.
- $\mathbf e_m \in \mathbb R^{64}$ is a learned identity embedding for triple $m$. A hidden token keeps its identity; otherwise the model could not know which marker it is asked to predict.
- $\mathbf e_{\text{mask}}$ and $\mathbf e_{\text{absent}}$ are two separate learned vectors. "Hidden for training" and "never measured" are different states.
- $\mathcal H_i$ is the set of slots hidden for the masked-marker objective (Section 3.8).

### 3.7.3 Set encoder

The tokens pass through $B = 2$ pre-norm Transformer blocks (Vaswani et al., 2017; Xiong et al., 2020) with 4 attention heads and width 64:

$$
\mathbf x' = \mathbf x + \operatorname{MHA}\big(\operatorname{LN}(\mathbf x)\big), \qquad
\mathbf x'' = \mathbf x' + \operatorname{FFN}\big(\operatorname{LN}(\mathbf x')\big)
$$

where LN is layer normalisation (Ba et al., 2016) and FFN is 64 → 128 → 64 with GELU. There is no positional encoding, so the encoder is permutation-equivariant over marker tokens, as in set models (Zaheer et al., 2017; Lee et al., 2019). Layer normalisation is per token and there is no batch normalisation, so no batch statistics are read at test time. After a final LayerNorm, the tokens are mean-pooled and projected to a 128-dimensional cell embedding:

$$
\mathbf z_i = g\!\left(\frac{1}{M}\sum_{m=1}^{M}\operatorname{LN}\big(\mathbf x^{(B)}_{i,m}\big)\right) \in \mathbb R^{128}, \qquad g:\ 64 \to 128 \to 128 \ \text{(GELU)}
$$

The attention width is kept at 64 because attention cost grows with the square of the token count, while the cheap output projection is widened to 128.

### 3.7.4 Batching

Batches are drawn from one cohort at a time, in random cohort order, so all cells in a batch share one panel and one `present` mask. No ragged padding is needed.

---

## 3.8 Masked-marker pretraining (Gate 2)

### 3.8.1 Objective

Before any label is used, the encoder is pretrained to reconstruct hidden marker values, in the style of masked language and image modelling (Devlin et al., 2019; He et al., 2022). For each cell, 15% of its **eligible** markers (at least one) are hidden and predicted from the rest through a linear head on each token:

$$
\mathcal L_{\text{mask}} = \frac{1}{|\mathcal H|}\sum_i \sum_{m \in \mathcal H_i}\big(\hat u_{i,m} - u_{i,m}\big)^2
$$

The objective uses no labels, so every cohort can contribute before labels are aligned.

### 3.8.2 Eligible markers

A (cohort, marker) pair is excluded from the loss when its **rank spread** is below 0.2:

$$
\text{rank\_spread}_{c,m} = \frac{\operatorname{Var}(u^{\text{coh}}_{\cdot,m})}{1/12}
$$

$1/12$ is the variance of an untied uniform rank, so the ratio says how much of a normal marker's range survives ties. Below the floor, the $R^2$ denominator collapses and the loss gradient is noise. 14 of 278 pairs are excluded (9 Keren, 5 Sorin).

This rule **replaced** the declared rule (`tie_mass` ≥ 0.5) before any training run. The old rule excluded none of the UPMC markers it was written for (the ECDF had already fixed them) and would have removed 16 of 17 Sorin markers, including CD3, CD4, CD8A, CD20 and CD68. The old row is kept in the expect file, marked `REPLACED`.

### 3.8.3 Reconstruction $R^2$

$$
R^2 = 1 - \frac{\operatorname{MSE}_{\text{model}}}{\operatorname{MSE}_{\text{baseline}}}
$$

The baseline predicts the **mean of the training cohorts**. The mean is the best constant predictor under squared error; the median, named in the original design, is weaker and would inflate every $R^2$.

### 3.8.4 Arms and controls

- **Arm A** — tokens only for the cohort's measured markers (a variable-size set).
- **Arm B** — all 109 slots, with $\mathbf e_{\text{absent}}$ for unmeasured slots (fixed size).
- **Core-9 control** — the same set transformer on only the 9 markers shared by all cohorts. This separates the effect of architecture (set transformer vs Gate 1's MLP) from the effect of panel width.

The arm was decided on **cross-cohort LOCO $R^2$**, not within-cohort $R^2$, because an `[ABSENT]` token can only matter for a panel the model has not trained on.

### 3.8.5 Fold-local warm starts

A pretrained model trained on all seven cohorts would leak the held-out cohort into every LOCO fold. So one pretraining model is fitted per LOCO fold on its six training cohorts (plus one per leave-one-tissue-out fold). Downstream stages get weights only through `warm_for(train_cohorts)`, which returns a checkpoint whose recorded training cohorts are a subset of the fit's, and raises an error if none exists.

### 3.8.6 Settings

Adam (Kingma & Ba, 2015), learning rate 3 × 10⁻³, batch 512, at most 30 epochs, early stopping on the validation patients of the training cohorts with patience 4; 15,000 training cells per cohort.

---

## 3.9 Prototype classifier (Gate 6) — the headline model

### 3.9.1 Prototypes

Each cluster $\ell$ of the fold's label space gets a learnable prototype $\boldsymbol\pi_\ell \in \mathbb R^{128}$. Prototypes are **initialised from marker signatures, not at random**. Each cluster's mean marker signature over the 109-slot vocabulary is passed through the warm-started encoder as a synthetic cell, with that cluster's own `present` mask (markers none of its cohorts measured get the `[ABSENT]` token), and the output becomes $\boldsymbol\pi_\ell^{(0)}$. The encoder is loaded first and prototypes are read out of that same encoder, so they start in the space the model uses.

### 3.9.2 Scoring and loss

Logits are temperature-scaled cosine similarities (Snell et al., 2017; Gidaris & Komodakis, 2018; Wang et al., 2018):

$$
s_{i\ell} = \frac{1}{T}\cdot\frac{\mathbf z_i^\top \boldsymbol\pi_\ell}{\lVert\mathbf z_i\rVert\,\lVert\boldsymbol\pi_\ell\rVert}, \qquad T = 0.1
$$

Cosine similarity stops the loss from being reduced just by making $\lVert \mathbf z\rVert$ larger. The classification loss is cross-entropy, optionally weighted per cell by label confidence $w_i$:

$$
\mathcal L_{\text{cls}} = \frac{\sum_i w_i\,\big[-\log \operatorname{softmax}(\mathbf s_i)_{y_i}\big]}{\sum_i w_i}
$$

Only UPMC ships a per-cell confidence (a kNN probability from 0.14 to 1.0). All other cohorts have $w_i = 1$, which is plain cross-entropy.

### 3.9.3 Multi-task weighting

Auxiliary losses are weighted by homoscedastic uncertainty (Kendall et al., 2018). With a learned $s_k = \log\sigma_k$ for each auxiliary loss:

$$
\mathcal L = \mathcal L_{\text{cls}} + \sum_{k \in \text{aux}} \Big( e^{-2 s_k}\,\mathcal L_k + s_k \Big)
$$

The cell-type weight is **pinned at 1 and not learned**. Uncertainty weighting can switch off a task whose labels look noisy, and labels pooled from six annotation schemes look very noisy. Left free, the one task that matters would be the most likely one to be switched off.

Candidate auxiliary losses were:

- $\mathcal L_{\text{mask}}$ — the masked-marker loss of Section 3.8, computed from the per-token outputs;
- $\mathcal L_{\text{VIC}}$ — the variance and covariance terms of VICReg (Bardes et al., 2022). The invariance term is not used because there is no declared augmentation for protein values.

### 3.9.4 Guards

- **Sigma guard (check 1).** No auxiliary $\log\sigma$ may move more than 3.0 from its start. A runaway sigma deletes its loss while the training curve still looks healthy.
- **Collapse guard (check 2).** Two learnable prototypes can merge silently. The minimum pairwise cosine distance between prototypes is tracked each epoch. If it falls below half its starting value, the pair is frozen and logged rather than the run being stopped, because a merge may be a true statement about the label space.
- **Drift diagnostic.** The cosine distance each prototype moves from its initial signature is reported. A large move means the model disagrees with the label space about where that cluster lies.

### 3.9.5 Training

- **Class-balanced sampling.** Every epoch, each training cohort contributes an equal number of cells per class (with replacement for small classes), then shuffled.
- Adam, learning rate 10⁻³, batch 512, at most 30 epochs.
- **Early stopping on macro-F1** over the validation patients of the training cohorts (patience 4). Loss is not used because it is not comparable across loss sets. The held-out cohort is never used for early stopping.

### 3.9.6 Ablations decided by Gate 6

| Arm | Losses | Adversary | Confidence | Role |
|---|---|---|---|---|
| `proto2` | cls + mask | no | on | 2-loss model |
| `proto3` | cls + mask + VICReg | no | on | check 3: does VICReg earn its place? |
| `proto2adv` | cls + mask | λ = 0.01 | on | check 3b: does the adversary earn its place? |
| `linear` | plain linear head (+ mask, VICReg) | no | on | check 4: does the prototype head beat a linear head by ≥ 0.02? |
| `noconf` | cls + mask | no | off | check 6: confidence weighting (6 folds where UPMC trains) |

The rule, declared before the run, was that an extra component ships only if it improves LOCO macro-F1. **The shipped configuration is `proto2`: prototype head, cell-type + masked-marker losses, no VICReg, no adversary.**

Two parts of the original design are deliberately not built: a descendant-tolerant cross-entropy (it needs the nesting graph, which failed its declared cases), and spatial context in the headline model (it was tested separately, Section 3.11).

---

## 3.10 Domain-adversarial encoder (Gate 3)

### 3.10.1 Gradient reversal

A gradient reversal layer (GRL; Ganin & Lempitsky, 2015; Ganin et al., 2016) is the identity in the forward pass and multiplies the gradient by $-\lambda$ in the backward pass:

$$
\mathcal R_\lambda(\mathbf z) = \mathbf z, \qquad \frac{\partial \mathcal R_\lambda}{\partial \mathbf z} = -\lambda\,\mathbf I
$$

A head placed after the GRL learns to predict the domain, while the encoder before it learns to make that prediction impossible.

### 3.10.2 Slide, not cohort

The adversary targets **slide identity**, not cohort identity. On this roster cohort is confounded with tissue (colorectal, head and neck, breast ×2, lung, skin ×2). An embedding that cannot tell cohorts apart also cannot tell colon from lung, and colon and lung tumour cells really differ. Slide-to-slide variation inside a cohort is close to pure batch effect (same tissue, machine and disease), so removing it is safe. A cohort head is kept at only $0.1\lambda$ and read as an **inverted guard**: if cohort accuracy falls toward chance, that warns that tissue biology is being erased.

Total loss: $\mathcal L = \mathcal L_{\text{cls}} + \mathcal L_{\text{slide}}(\mathcal R_\lambda(\mathbf z)) + \mathcal L_{\text{cohort}}(\mathcal R_{0.1\lambda}(\mathbf z))$, with a plain linear cell-type head and class-balanced sampling so that only one thing (the adversary) changes. λ ramps linearly from 0 to its value over the first 5 epochs. λ = 0 is exactly the plain encoder.

### 3.10.3 Measuring what the adversary removed

Slide accuracy is not usable: chance on about 1,100 slides is about 0.1%. The measure is **retained slide information in bits**:

$$
\text{retained\_bits} = \log_2 N_{\text{slides}} - \text{CE}_{\text{bits}}
$$

This is zero at chance. It is reported for two discriminators:

- the **co-trained** head, which the encoder fought during training;
- a **fresh probe** (a two-hidden-layer MLP, trained from scratch on the frozen encoder, and scored on held-out cells from the same slides).

A co-trained discriminator can be beaten without the information being gone, because the encoder only has to hide it from that one head. If the fresh probe recovers more than the co-trained head, the adversary is hiding, not removing.

### 3.10.4 Sweep

λ ∈ {0, 0.01, 0.03, 0.1, 0.3}, seven LOCO folds each, fold-local label spaces and warm starts. The λ with the highest LOCO macro-F1 ships from Gate 3, and it is then re-tested inside Gate 6 as arm `proto2adv`.

---

## 3.11 Spatial context (Gate 4)

### 3.11.1 Neighbour graph

For every cell, the $k = 15$ nearest neighbours are found by Euclidean distance **within the same image**, using the **full** raw tables, not the 40,000-cell sample. Inside a 1.9% sample, the "nearest neighbours" are not neighbours: UPMC's mean spacing would rise from 11.7 µm to about 79 µm. Two checks are asserted when the graph is built: no edge crosses an image boundary (check 6), and every cohort's median edge length lies within 10–30 µm (check 7).

### 3.11.2 Neighbourhood features

For each cell $i$ with valid neighbours $\mathcal K(i)$:

- **Pooled profile** — the plain (unweighted) mean of the neighbours' harmonised values:
  $$\bar u_{i,m} = \frac{1}{|\mathcal K(i)|}\sum_{j\in\mathcal K(i)} u_{j,m}$$
  This is a single mean-aggregation step, like the mean aggregator of GraphSAGE (Hamilton et al., 2017) or one round of message passing (Gilmer et al., 2017). No learned edge weights or attention are used.
- **Heterogeneity** `het` — the standard deviation across neighbours, averaged over the cohort's measured markers (a uniform nest vs a mixed zone).
- **Boundary score** `d_self` — the mean absolute difference between the cell's own profile and $\bar{\mathbf u}_i$ over measured markers.
- **Geometry** — the log median edge length in µm, the median edge length relative to the image's median nearest-neighbour spacing, the anisotropy of neighbour offsets (0 = blob, 1 = line, as in vessels) and a flag for whether it is defined, and the fraction of the 15 slots holding a real neighbour.

`het` and `d_self` exist because a mean cannot express "is a vessel touching me" (a maximum question) or "am I on a tumour boundary" (a variance question). Without them, a failed gate could not tell "no spatial signal" from "the pooling threw it away."

**Deliberately excluded:** the fraction of neighbours with the same native label. On the held-out cohort those labels are exactly what is being predicted.

### 3.11.3 Fusion

The pooled profile is passed through **the same encoder** as the cell (so unmeasured slots still get `[ABSENT]`), giving $\mathbf z^{\text{nbr}}_i$. It is added as a residual through a gate whose last layer starts at zero:

$$
\mathbf z_i = \mathbf z^{\text{cell}}_i + \phi\big([\,\mathbf z^{\text{nbr}}_i \,;\, \psi(\mathbf g_i)\,]\big)
$$

where $\psi$ embeds the 7 context scalars (7 → 32 → 128) and $\phi$ is 256 → 128 → 128 with the final layer zero-initialised. At epoch 0, arm `neigh` is therefore bit-identical to arm `cell`, and the neighbourhood must earn any margin. Concatenation was rejected because it would change the prototype width and break prototype initialisation.

### 3.11.4 Context loss (arm `ctx`)

An auxiliary head predicts the pooled neighbourhood from the cell's own embedding, so neighbourhood information is pushed into $\mathbf z^{\text{cell}}$:

$$
\mathcal L_{\text{ctx}} = \frac{1}{|\mathcal M_c|}\sum_{m \in \mathcal M_c}\big(\hat{\bar u}_{i,m}(\mathbf z^{\text{cell}}_i) - \bar u_{i,m}\big)^2
$$

Only measured slots are scored. It enters the uncertainty weighting as a third auxiliary loss.

### 3.11.5 Arms and controls

| Arm | Sees |
|---|---|
| `cell` | the cell's own markers only |
| `neigh` | + its 15 neighbours, pooled |
| `shuffle` | + **another** cell's neighbourhood from the same image (a within-image derangement: every cell moved) |
| `ctx` | `neigh` + $\mathcal L_{\text{ctx}}$ |

The `shuffle` arm is the leakage control. It keeps the image's overall composition but breaks the link between a cell and its real surroundings. If `shuffle` scored as well as `neigh`, the gain would come from image-level composition, not from spatial neighbourhoods. `d_self` is computed **after** shuffling, so the control truly has no spatial signal.

All Gate 4 arms use the 3-loss configuration (cls + mask + VICReg). The `cell` arm has identical weights to Gate 6 `proto3`.

### 3.11.6 Pixel-size sensitivity (check 5)

Because UPMC's pixel size is assumed, the UPMC `neigh` fold was refitted with it multiplied by 1.3 and by 0.77, to test whether check 1's verdict moves.

---

## 3.12 Leakage controls — summary

| Risk | Control |
|---|---|
| Held-out labels shape the label space | Fold-local label spaces; asserted that no held-out row reaches a fold table |
| Held-out cells in pretraining | Fold-local warm starts via `warm_for`, which raises when no safe checkpoint exists |
| Held-out cohort used for model selection | Early stopping and ablations use training-cohort validation patients only |
| Patient biology shared across train and test | Patient-level splits, recorded fingerprint |
| Held-out slides in the adversary | Slide vocabulary built from training cohorts only |
| Neighbours' labels used as input | `nbr_same` excluded |
| Neighbours crossing images | Asserted at graph build |
| Stale artefacts | Vocabulary, split and label-space hashes stored in every checkpoint; mismatch forces refit |

---

## 3.13 Evaluation

The metrics are defined in full in Chapter 4, Section 4.1. In short:

### 3.13.1 Primary metric

Macro-F1 over the classes **present in the held-out cohort's truth**. Classes absent from the truth are not scored as zero (that would measure label coverage, not accuracy), but predicting them still costs precision.

### 3.13.2 Headline score

**Core macro-F1** leaves out clusters flagged `unreliable` (Section 3.5.7) and held-out labels marked NOVEL (Section 3.5.8). It is averaged over the seven LOCO folds. Because each fold has its own label space, folds have different class sets.

### 3.13.3 Reference predictors

- **Majority**: always predicts the most common class in the training cohorts.
- **Random**: predicts uniformly at random.

Both are scored with the same metric, on the same cells and class set.

### 3.13.4 Paired statistics

Each comparison between two arms is paired by fold (same cells, same label space, one thing changed). Two statistics are reported:

- a 95% **t-interval** on the mean per-fold difference;
- an **exact sign-flip permutation test** over all $2^7 = 128$ sign assignments (Good, 2005). It makes no assumption about the shape of the differences.

With $n = 7$ folds, the sentence "method A improves on method B" is written only if the interval excludes zero. This rule was declared in `gate4_expect.csv` and `gate10_expect.csv`.

---

## 3.14 External baseline: MAPS (Gate 10)

MAPS (Shaban et al., 2024) is a published MLP cell phenotyper. Its network class is **imported from the vendored MAPS source, not reimplemented**; if the import fails, the stage stops. It runs on exactly the same 7 folds, cells, patient splits, fold-local label spaces and metric as the headline model.

MAPS needs a fixed input size, so it cannot represent "this marker was not measured". Three arms were run:

| Arm | Markers | Meaning |
|---|---|---|
| `core9` | the 9 markers every cohort measures | the fair fixed-input setting |
| `shared` | markers shared by the training cohorts and the held-out cohort (9–11) | uses the held-out panel's composition (metadata known before any label, stated as a generosity) |
| `zerofill` | all 109, unmeasured set to 0 | what zero-filling does |

Two declared deviations favour MAPS: it gets the same class-balanced sampler (plain cross-entropy collapses onto common classes under macro-F1), and it gets a larger epoch budget (max 60, patience 8) than the headline model (30, 4).

---

## 3.15 Descriptive comparison: native labels vs derived clusters

The whole-roster label space is compared with the native labels by cell-weighted ARI and normalised mutual information (NMI; Strehl & Ghosh, 2002). This is descriptive only: native labels are not ground truth.

---

## 3.16 Reproducibility

- **One seed, per-purpose generators.** All randomness comes from `config.rng(*purpose)` seeded from `SEED = 20260810`, so each random draw (sampling, shuffling, masking, baselines) has its own reproducible stream.
- **Checkpoint cache with fingerprints.** Every fit is cached with its vocabulary, split and label-space hashes.
- **Golden test.** `tests/golden.py --check` fits tiny CPU versions of every model stage and compares weight hashes with a stored reference. Any code refactor must leave it passing.
- **Full re-run.** The whole pipeline was re-run from the raw datasets on 2026-09-18/19 with renamed, cleaned code. The pass mark was "7-fold mean within ±0.015 of the published value" (Chapter 4, Section 4.10).

---

## 3.17 Final shipped configuration

| Component | Setting |
|---|---|
| Value transform | per-cohort mid-rank ECDF ($u^{\text{coh}}$, V3) |
| Vocabulary | 109 protein triples, frozen |
| Unmeasured markers | learned `[ABSENT]` token (Arm B) |
| Encoder | 2 pre-norm Transformer blocks, 4 heads, token width 64, mean pool, 128-d embedding |
| Pretraining | masked-marker, 15% of eligible markers, fold-local warm start |
| Label space | fold-local, weighted-RMS distance on rescaled mean ranks, average linkage, stability-chosen cut |
| Head | cosine prototypes, $T = 0.1$, initialised from marker signatures |
| Losses | cell-type CE (weight 1, confidence-weighted where available) + masked-marker (uncertainty-weighted) |
| Not shipped | VICReg, slide adversary, spatial context, descendant-tolerant CE |
| Optimiser | Adam, lr 10⁻³, batch 512, ≤ 30 epochs, early stop on validation macro-F1 (patience 4) |
| Sampling | class-balanced per cohort, one cohort per batch |

---

## References

**Datasets**

- Danenberg, E., Bardwell, H., Zanotelli, V. R. T., et al. (2022). Breast tumor microenvironment structures are associated with genomic features and clinical outcome. *Nature Genetics*, 54, 660–669. https://doi.org/10.1038/s41588-022-01041-y
- Ferguson, A. L., Sharman, A. R., Allison, R. O., et al. (2022). High-dimensional and spatial analysis reveals immune landscape–dependent progression in cutaneous squamous cell carcinoma. *Clinical Cancer Research*, 28(21), 4677–4688. https://doi.org/10.1158/1078-0432.CCR-22-1332
- Keren, L., Bosse, M., Marquez, D., et al. (2018). A structured tumor-immune microenvironment in triple negative breast cancer revealed by multiplexed ion beam imaging. *Cell*, 174(6), 1373–1387. https://doi.org/10.1016/j.cell.2018.08.039
- Phillips, D., Matusiak, M., Gutierrez, B. R., et al. (2021). Immune cell topography predicts response to PD-1 blockade in cutaneous T cell lymphoma. *Nature Communications*, 12, 6726. https://doi.org/10.1038/s41467-021-26974-6
- Risom, T., Glass, D. R., Averbukh, I., et al. (2022). Transition to invasive breast cancer is associated with progressive changes in the structure and composition of tumor stroma. *Cell*, 185(2), 299–310. https://doi.org/10.1016/j.cell.2021.12.023
- Schürch, C. M., Bhate, S. S., Barlow, G. L., et al. (2020). Coordinated cellular neighborhoods orchestrate antitumoral immunity at the colorectal cancer invasive front. *Cell*, 182(5), 1341–1359. https://doi.org/10.1016/j.cell.2020.07.005
- Sorin, M., Rezanejad, M., Karimi, E., et al. (2023). Single-cell spatial landscapes of the lung tumour immune microenvironment. *Nature*, 614, 548–554. https://doi.org/10.1038/s41586-022-05672-3
- Wu, Z., Trevino, A. E., Wu, E., et al. (2022). Graph deep learning for the characterization of tumour microenvironments from spatial protein profiles in tissue specimens. *Nature Biomedical Engineering*, 6, 1435–1448. https://doi.org/10.1038/s41551-022-00951-w

**Databases and baseline**

- Seal, R. L., Braschi, B., Gray, K., et al. (2023). Genenames.org: the HGNC resources in 2023. *Nucleic Acids Research*, 51(D1), D1003–D1009.
- Shaban, M., Bai, Y., Qiu, H., et al. (2024). MAPS: pathologist-level cell type annotation from tissue images through machine learning. *Nature Communications*, 15, 28.
- The UniProt Consortium (2023). UniProt: the Universal Protein Knowledgebase in 2023. *Nucleic Acids Research*, 51(D1), D523–D531.

**Methods**

- Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization. *arXiv:1607.06450*.
- Bardes, A., Ponce, J., & LeCun, Y. (2022). VICReg: Variance-invariance-covariance regularization for self-supervised learning. *International Conference on Learning Representations (ICLR)*.
- Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., & Vaughan, J. W. (2010). A theory of learning from different domains. *Machine Learning*, 79(1–2), 151–175.
- Bolstad, B. M., Irizarry, R. A., Åstrand, M., & Speed, T. P. (2003). A comparison of normalization methods for high density oligonucleotide array data based on variance and bias. *Bioinformatics*, 19(2), 185–193.
- Conover, W. J. (1999). *Practical Nonparametric Statistics* (3rd ed.). Wiley.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *Proceedings of NAACL-HLT*, 4171–4186.
- Ganin, Y., & Lempitsky, V. (2015). Unsupervised domain adaptation by backpropagation. *Proceedings of the 32nd International Conference on Machine Learning (ICML)*, 1180–1189.
- Ganin, Y., Ustinova, E., Ajakan, H., et al. (2016). Domain-adversarial training of neural networks. *Journal of Machine Learning Research*, 17(59), 1–35.
- Gidaris, S., & Komodakis, N. (2018). Dynamic few-shot visual learning without forgetting. *Proceedings of IEEE/CVF CVPR*, 4367–4375.
- Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). Neural message passing for quantum chemistry. *Proceedings of ICML*, 1263–1272.
- Good, P. (2005). *Permutation, Parametric and Bootstrap Tests of Hypotheses* (3rd ed.). Springer.
- Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
- Hanley, J. A., & McNeil, B. J. (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology*, 143(1), 29–36.
- He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked autoencoders are scalable vision learners. *Proceedings of IEEE/CVF CVPR*, 16000–16009.
- Hendrycks, D., & Gimpel, K. (2016). Gaussian error linear units (GELUs). *arXiv:1606.08415*.
- Hubert, L., & Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2(1), 193–218.
- Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. *Proceedings of IEEE/CVF CVPR*, 7482–7491.
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
- Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019). Set Transformer: A framework for attention-based permutation-invariant neural networks. *Proceedings of ICML*, 3744–3753.
- Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *Annals of Mathematical Statistics*, 18(1), 50–60.
- Murtagh, F., & Contreras, P. (2012). Algorithms for hierarchical clustering: an overview. *WIREs Data Mining and Knowledge Discovery*, 2(1), 86–97.
- Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T. (2018). The preregistration revolution. *Proceedings of the National Academy of Sciences*, 115(11), 2600–2606.
- Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM: Visual reasoning with a general conditioning layer. *Proceedings of the AAAI Conference on Artificial Intelligence*, 32(1).
- Quiñonero-Candela, J., Sugiyama, M., Schwaighofer, A., & Lawrence, N. D. (Eds.) (2009). *Dataset Shift in Machine Learning*. MIT Press.
- Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical networks for few-shot learning. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
- Sokal, R. R., & Michener, C. D. (1958). A statistical method for evaluating systematic relationships. *University of Kansas Science Bulletin*, 38, 1409–1438.
- Strehl, A., & Ghosh, J. (2002). Cluster ensembles — a knowledge reuse framework for combining multiple partitions. *Journal of Machine Learning Research*, 3, 583–617.
- Tarjan, R. (1972). Depth-first search and linear graph algorithms. *SIAM Journal on Computing*, 1(2), 146–160.
- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
- Wang, H., Wang, Y., Zhou, Z., et al. (2018). CosFace: Large margin cosine loss for deep face recognition. *Proceedings of IEEE/CVF CVPR*, 5265–5274.
- Xiong, R., Yang, Y., He, D., et al. (2020). On layer normalization in the Transformer architecture. *Proceedings of ICML*, 10524–10533.
- Zaheer, M., Kottur, S., Ravanbakhsh, S., Poczos, B., Salakhutdinov, R., & Smola, A. J. (2017). Deep sets. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.

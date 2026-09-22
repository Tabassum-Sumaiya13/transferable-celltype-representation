# Diagnostic probes — where the LOCO score actually goes

Measured 2026-09-13, CPU, from the **cached shipped checkpoints** (`work/ckpt/s6_proto2_fold_*.pt`).
No training happened. Every number below reproduces the Gate 6 headline **0.3151** exactly as a
sanity check, so the harness is reading the same cells, splits and label spaces the gates did.

These are **diagnostics, not a gate**. Nothing here was pre-registered, so no threshold in
`pipeline2/panel/` is scored against it. Its purpose is to tell the next redesign where the loss
is, so effort is not spent on a stage that is already fine.

---

## 0. HEADLINE — transfer is governed by label-space coverage, nothing else

`pairwise.py`. For each **target** cohort B the encoder from fold B is used (it never saw B); a
linear decoder is fitted on **one source cohort A at a time** in fold B's own label space and
scored on B's test patients. **42 ordered pairs** — the only part of this project with real
statistical power.

### Correlations over the 42 pairs

| correlate | r | significant |
|---|---|---|
| **`class_cover`** — fraction of the target's classes the source contains | **+0.753** | **t = 7.25, n = 42, p < 0.0001** |
| `src_classes` | −0.246 | no |
| `same_tissue` | **−0.139** | no — and negative |
| `n_src` (source panel size) | +0.122 | no |
| `jaccard` (marker overlap) | +0.120 | no |
| `shared_markers` | +0.110 | no |
| `same_platform` | +0.049 | no |

**Result.** Coverage is the only significant correlate. Tissue match does not help. Platform does
not matter. Marker overlap barely matters.

**Meaning.** Cross-cohort transfer is a **roster composition** problem, not a representation or
architecture problem. If the training cohorts contain the target's cell types, transfer works; if
they do not, no encoder, decoder or label space can recover them.

**Reason.** A classifier cannot predict a class it was never shown. This is why 23% of classes
score exactly zero (section 3) and why Danenberg collapses to 0.0513 — every source transfers to
it at 0.016–0.230, so nobody covers its classes. That is a roster failure, not a model failure.

**This retires two earlier findings in this document as n=7 artifacts.** The panel-size correlation
(−0.879) and the CODEX/IMC platform split both **wash out at n=42** (+0.122 and +0.049). Section 5
should be read with that correction. The separability correlate (+0.878, n=7) is very likely
`class_cover` in disguise.

### Pooling six cohorts buys nothing over one well-chosen cohort

| target | best single source | pooled 6 | LOCO prototype | pooled − best |
|---|---|---|---|---|
| ferguson | **0.4270** | 0.3412 | 0.3942 | **−0.0858** |
| Sorin | 0.4086 | 0.3870 | 0.4147 | −0.0216 |
| UPMC | 0.3797 | 0.4135 | 0.4127 | +0.0338 |
| Phillips | 0.3516 | 0.3617 | 0.3455 | +0.0101 |
| CRC | 0.2983 | 0.2962 | 0.3042 | −0.0021 |
| Keren | 0.2805 | 0.2988 | 0.2830 | +0.0183 |
| Danenberg | 0.0781 | 0.0530 | 0.0513 | −0.0251 |
| **MEAN** | **0.3177** | **0.3073** | **0.3151** | |

**Result.** The best single source beats pooling all six (0.3177 vs 0.3073). Pooling wins on only
**3 of 7** targets. On ferguson it costs 0.086.

**Meaning.** Pooling is **averaging, not accumulating**. Adding cohorts does not add coverage
proportionally — it dilutes. All three numbers sit within 0.01, i.e. inside the noise floor, so
the honest claim is that six cohorts give **no measurable advantage over one well-chosen cohort**.

**Caveat.** "Best single source" is an oracle — choosing it requires target labels. The point is
not that one cohort is enough in practice; it is that pooling is not additive.

**Next.** Optimise the training roster for **coverage of the target's classes**, not for tissue
match, panel size, or cohort count. And report coverage alongside any LOCO number, because it
predicts the result better than anything about the method.

---

## 0a. NEGATIVE — coverage cannot be estimated without labels via source agreement

`coverage_claim1.py`. Section 0 showed coverage governs transfer (r = +0.753, n = 42) — but that
`class_cover` was computed **with labels**. This tests whether a **label-free** proxy reproduces it.

Method: six independent source decoders vote on every target cell;
`coverage = fraction agreeing with the modal vote`. Compared against the shipped model's own
max-softmax `confidence`. Both aggregated per true class, correlated against that class's F1.

**Pre-registered before the run**: *"confidence ≥ coverage → the method is a relabelled
active-learning baseline. Drop it."*

| signal | corr with per-class F1 | t | n |
|---|---|---|---|
| coverage (source agreement) | +0.449 | 3.86 | 61 |
| **confidence (max softmax)** | **+0.493** | **4.35** | 61 |

**Result. Confidence wins by 0.044. The declared threshold fires — the method is dropped.**

**Reason — the predicted failure mode, confirmed.** The risk declared when the method was specced
was that all six sources would agree *for the wrong reason*, each mapping an uncovered type to the
same nearest wrong neighbour. That is what happens:

| target | cluster | n | F1 | coverage | confidence |
|---|---|---|---|---|---|
| Sorin | 13 | 341 | **0.000** | **0.924** | 0.843 |
| Sorin | 18 | 170 | **0.000** | **0.825** | 0.731 |
| Sorin | 11 | 1046 | **0.000** | 0.718 | 0.780 |
| Danenberg | 4 | 415 | **0.000** | 0.693 | 0.757 |

Six of six sources agree at 92% on a class scoring exactly zero. **Disagreement is not the
signature of absence.**

**Neither signal is usable in any case.** r ≈ 0.45–0.49 explains ~20–25% of variance. Zero-F1
classes average 0.614 coverage against 0.687 for the rest — a tendency, not a router.

**Meaning.** The finding in section 0 stands; the *method* built on it does not. There is a real
gap between "coverage predicts transfer" and "coverage can be measured without labels", and this
project has now measured both sides of it.

---

## 1. The headline decomposition

Where the distance between "what is achievable" and "what LOCO scores" is spent.
MAPS MLP as the decoder throughout. **Danenberg excluded** — its class count differs between
runs (13 vs 29) because the two used different fold label spaces, so its rows are not comparable.

| step | mean macro-F1 | cost | what it is |
|---|---|---|---|
| ceiling: MAPS on `u_coh`, in-cohort supervised | **0.684** | — | the most any model gets with full target labels |
| encoder, in-distribution (`emb_seen`) | 0.648 | −0.036 | **compression** by the encoder |
| encoder, held-out cohort (`emb_held`) | 0.570 | −0.078 | **domain shift** |
| the actual LOCO pipeline | **0.315** | **−0.255** | **zero-shot decoding** |

**Result.** The zero-shot decoding step costs 0.255 of the total 0.369 — about **70%**.
Encoder compression and domain shift together cost 0.114.

**Meaning.** The representation is not the bottleneck. Effort spent on encoders, adversaries or
spatial context is spent on the part that already works.

**Reason.** The decoder compresses every class to one prototype, matches through a harmonized
vocabulary, and has no target supervision at all. Those three are still bundled inside the 0.255.

**Next.** Retrieval vs prototype, zero-shot, same folds — that ablation splits "decoder design"
from "the irreducible price of zero target labels".

### 1a. The 0.255 is NOT the decoder — measured, four families

`retrieval.py`. Same embedding, same cells, same fold-local space, same metric, same
unreliable-cluster drop. Only the decoder changes. No training.

| decoder family | mean macro-F1 | vs shipped |
|---|---|---|
| **prototype, 2 losses (shipped)** | **0.3151** | — |
| retrieval, best of 8 variants (`bal100`) | 0.2998 | −0.0153 |
| linear head (cached Gate 6 arm) | 0.2958 | −0.0193 |
| prototype, 3 losses (cached Gate 6 arm) | 0.2956 | −0.0195 |

Retrieval variants tested: plain kNN vote and class-balanced per-class mean similarity, at
k = 1, 5, 25, 100. **Every one lost.** Five of seven cohorts lost; only Phillips (+0.0133) and
Danenberg (+0.0090) gained.

**Result.** Four decoder families span 0.0195 in total, and the shipped prototype is the best.

**Meaning.** The decoder is not the bottleneck. "One prototype per class destroys multi-modality"
was the hypothesis this test was built to confirm, and it is **false** on this data — the
multi-modal, class-balanced retrieval variant lost too.

**Reason.** Retrieval keys on individual training cells, which carry cohort-specific
idiosyncrasies. A prototype averages over six cohorts, so what it discards is largely domain
shift. **Under domain shift, compression helps** — the prototype's lossiness is doing useful work.

**Next.** By elimination, the ~0.235 that remains is the price of having **zero target labels**,
bundled with the **harmonized vocabulary**. Those two are still not separated. The test that
splits them: probe the held-out embedding → *harmonized* labels and compare to → *native* labels
(0.5622). A large gap means the vocabulary costs; a small one means it is pure supervision.

### 1b. The harmonized vocabulary is EASIER than native — the whole gap is supervision

`vocabcost.py`. Same embedding, same cells, same patient split, same decoder. **Only the target
changes.** Both fit on the held-out cohort, so supervision is held constant.

| held | k native | k harm | → native | → harmonized | delta | LOCO zero-shot |
|---|---|---|---|---|---|---|
| CRC | 22 | 9 | 0.4337 | 0.5470 | **+0.1133** | 0.3042 |
| UPMC | 16 | 10 | 0.5050 | 0.6070 | **+0.1020** | 0.4127 |
| Phillips | 20 | 11 | 0.4590 | 0.5369 | +0.0780 | 0.3455 |
| Keren | 16 | 12 | 0.5949 | 0.6702 | +0.0753 | 0.2830 |
| ferguson | 8 | 7 | 0.6459 | 0.6566 | +0.0107 | 0.3942 |
| Sorin | 16 | 12 | 0.8136 | 0.8119 | −0.0018 | 0.4147 |
| Danenberg | 13 | 12 | 0.4835 | 0.4735 | −0.0100 | 0.0513 |
| **MEAN** | | | **0.5622** | **0.6147** | **+0.0525** | **0.3151** |

**Result.** Predicting harmonized clusters is **easier** than predicting native labels, by 0.0525.
Five of seven cohorts positive.

**Meaning.** The harmonization is not destroying signal. It reduces class count (CRC 22 → 9,
Phillips 20 → 11), and the coarser target is easier. **The label space is not the bottleneck, and
redesigning it — tree-build included — is not the lever.**

**Reason.** Merging native labels that a classifier could not separate anyway removes distinctions
that were costing macro-F1 without carrying information.

**Next.** With the vocabulary ruled out, the residual is isolated:
**0.6147 (supervised, same space) − 0.3151 (zero-shot, same space) = 0.2996.**
That is the price of having no target labels, and nothing else.

### 1c. Complete accounting

| component | cost | how it was measured |
|---|---|---|
| encoder compression | 0.036 | MAPS on values vs in-distribution embedding |
| domain shift | 0.078 | in-distribution vs held-out embedding |
| decoder design | ≤0.020 | four families: prototype, linear, kNN, balanced-kNN |
| harmonized vocabulary | **−0.053 — it helps** | probe → native vs → harmonized |
| **absence of target labels** | **~0.300** | supervised vs zero-shot, same space, same decoder |

**Every component this project could redesign has now been measured, and each is small or
beneficial. The entire gap is the one thing zero-shot forbids by definition: labels from the
target cohort.**

The few-shot curve (section 4) prices that directly — 2–5 labelled cells per type matches the
whole zero-shot apparatus.

Per cohort, the encoder/shift split varies enormously:

| cohort | ceiling | seen | held | compression | shift |
|---|---|---|---|---|---|
| Sorin | 0.837 | 0.834 | 0.833 | −0.003 | −0.001 |
| Keren | 0.740 | 0.744 | 0.603 | **+0.004** | −0.141 |
| CRC | 0.565 | 0.555 | 0.420 | −0.010 | −0.135 |
| ferguson | 0.770 | 0.636 | 0.605 | **−0.134** | −0.031 |
| Phillips | 0.598 | 0.538 | 0.480 | −0.061 | −0.058 |
| UPMC | 0.591 | 0.580 | 0.477 | −0.011 | −0.103 |

Sorin loses essentially nothing at any stage. Keren's compression is *negative*. ferguson is the
only cohort where compression, not shift, dominates. Any claim about "the encoder" that does not
survive this spread is a claim about one cohort.

---

## 1d. LOCO detects transfer but cannot rank methods

| fold | proto2 | majority | random |
|---|---|---|---|
| Sorin | 0.4147 | 0.0159 | 0.0548 |
| UPMC | 0.4127 | 0.0000 | 0.0605 |
| ferguson | 0.3942 | 0.0299 | 0.0740 |
| Phillips | 0.3455 | 0.0000 | 0.0523 |
| CRC | 0.3042 | 0.0000 | 0.0696 |
| Keren | 0.2830 | 0.0000 | 0.0638 |
| Danenberg | 0.0513 | 0.0000 | 0.0385 |
| **MEAN** | **0.3151** | **0.0065** | **0.0591** |

**Result, part one — the protocol works as an existence proof.** 0.3151 is ~5× random and ~48×
majority, and beats random on 7 of 7 folds (sign test p = 0.008). Transfer is real.

**Result, part two — it cannot discriminate methods.**

| quantity | value |
|---|---|
| spread across all six methods measured | **0.0368** |
| SD across folds for a single method | **0.1179** |
| Gate 10's paired 95% CI | [−0.0986, +0.0640] |

The fold-to-fold noise is **3× larger than the entire range of every method tried** — prototype,
linear, retrieval, MAPS core-9, MAPS shared, 3-loss.

**Meaning.** Every method decision this project has made sits inside the noise floor:

| decision | margin |
|---|---|
| Gate 6 FAIL | 0.0008 |
| Gate 10 FAIL (MAPS ahead) | 0.0173 |
| Gate 4 "spatial helps" | 0.0107 (its own check 3 already flagged this) |
| retrieval vs prototype | 0.0153 |

**Gate 10 therefore does not say MAPS beats this project — it says indistinguishable** (p = 0.609,
CI spanning zero). "A published method ties our pipeline" is the accurate sentence.

**Reason.** With SD 0.118 and n = 7, the standard error on any mean is 0.045. Resolving a 0.02
difference would need roughly **130 cohorts**. This is structural, not a tuning problem.

**Next.** Keep LOCO as the headline existence proof, always with its interval attached. Stop using
it to select between methods. Select in the **within-cohort supervised setting** — signal is 2×
larger (0.61 vs 0.31), variance is lower, and every clean result in this document came from there.
Then confirm the chosen method on LOCO once, as a check rather than a selector.

---

## 2. Stage 1's ECDF harmonisation costs signal — 7 of 7 cohorts

`harmcost.py`. Within-cohort, patient-split, logistic regression → that cohort's own native
labels. The value tables carry **both** feature sets already:
`u_coh::<triple>` (Stage 1 mid-rank ECDF) and `raw::<triple>` (original intensity).

| cohort | k | `u_coh` | `raw` (log1p+z) | **both** | raw − u_coh |
|---|---|---|---|---|---|
| Danenberg | 32 | 0.4370 | 0.5297 | 0.5509 | **+0.0927** |
| Keren | 17 | 0.7194 | 0.7860 | **0.8615** | +0.0667 |
| Phillips | 21 | 0.5916 | 0.6529 | 0.6652 | +0.0613 |
| Sorin | 17 | 0.8346 | 0.8885 | 0.8939 | +0.0539 |
| CRC | 29 | 0.4520 | 0.5039 | 0.5258 | +0.0520 |
| ferguson | 9 | 0.6721 | 0.7078 | 0.7055 | +0.0357 |
| UPMC | 16 | 0.6260 | 0.6531 | 0.6634 | +0.0271 |
| **MEAN** | | **0.6190** | **0.6746** | **0.6952** | **+0.0556** |

**Result.** Every cohort loses under harmonisation. Concatenating both beats either alone
(0.6952); Keren gains +0.14 over harmonised alone.

**Meaning.** The two representations are **complementary**, not redundant — ECDF adds
within-cohort rank and destroys absolute intensity. Stage 2 currently tokenises `u_coh` only.

**Reason.** ECDF exists to solve the **cross-cohort** scale problem. Within one cohort there is no
scale problem for it to solve, so this is exactly the setting where it can only cost. This does
**not** show harmonisation is wrong — it prices it: 0.056 of within-cohort signal bought
cross-cohort comparability.

**Next.** Feed `raw::` alongside `u_coh::` into Stage 2 and re-run Gate 6. The columns already
exist in every value table. Measured headroom +0.076 under a linear decoder.

---

## 3. A quarter of all classes score exactly zero

Read straight from `per_cls` in the cached checkpoints — no computation.

| held | scored classes | F1 | classes at exactly 0 | median class F1 |
|---|---|---|---|---|
| CRC | 9 | 0.3042 | 2 (22%) | 0.125 |
| UPMC | 10 | 0.4127 | 0 | 0.376 |
| Keren | 12 | 0.2830 | 0 | 0.253 |
| ferguson | 7 | 0.3942 | 0 | 0.432 |
| Phillips | 11 | 0.3455 | 3 (27%) | 0.427 |
| **Danenberg** | 12 | **0.0513** | **7 (58%)** | 0.004 |
| Sorin | 12 | 0.4147 | 5 (42%) | 0.306 |

**Across 73 scored classes: 17 score exactly 0 (23%), 29 below 0.10 (40%).**
Dropping the zeros lifts the mean class F1 from 0.288 to 0.3753.

**Result.** The failure is concentrated, not broad.

**Meaning.** A quarter of classes are being sent somewhere else entirely. Danenberg's 0.0513 is
almost entirely this.

**Reason.** *Not rarity* — median support of the zero classes is **302 cells** vs 528 for the
rest. Three hundred cells is not a sample-size problem. Something systematic.

**Next.** Confusion matrices for those 17 classes. Highest-information cheap experiment available.

---

## 4. Few-shot exchange rate — how many labelled cells buy how much

`fewshot.py`. Embedding from the fold that **held that cohort out**; N cells per native type drawn
from its train patients; scored on its test patients; 5 seeds.

| cells per type | mean macro-F1 |
|---|---|
| 0 (the pipeline) | 0.3151 |
| 1 | 0.2558 |
| 2 | 0.3205 |
| 5 | 0.3777 |
| 10 | 0.4224 |
| 20 | 0.4568 |
| 50 | 0.4923 |
| 100 | 0.5144 |
| 250 | 0.5416 |
| all (15,000) | 0.5622 |

Per cohort, cells/type needed to match the zero-shot pipeline: ferguson 1, Sorin 1, Keren ~2,
UPMC ~4, Danenberg ~6, Phillips ~6, CRC ~15. Median ≈ 4–5.

**Two flaws in this table, both real:**

1. **Different target.** The curve scores **native** labels; LOCO scores **harmonized** clusters.
   Not the same task. Native k is higher than LOCO's scored class count in all seven folds, so the
   bias runs *against* the headline rather than for it — but the comparison is still invalid until
   the curve is re-run against harmonized labels.
2. **Rare-type capping.** The sampler used `min(n, available)`, so the n=100 and n=250 rows are not
   100 or 250 cells for every type. Macro-F1 weights those types equally. Upper rows are sound;
   the last two are optimistic by an unknown amount.

The zero-shot pipeline beats 1-shot (0.3151 > 0.2558) and loses to 2-shot. So the machinery is
doing real work — it is worth more than one labelled cell per type and less than two.

---

## 5. What predicts cohort difficulty

Correlations against few-shot @250, with leave-one-out robustness:

| predictor | all 7 | drop Sorin | drop CRC |
|---|---|---|---|
| class count *k* | −0.395 | −0.795 | −0.204 |
| within-cohort separability (`lr_raw`) | **+0.878** | **+0.756** | **+0.861** |
| panel size | −0.879 | −0.718 | −0.853 |

**Class count is not the driver** — its correlation swings from −0.20 to −0.80 depending on which
single cohort is dropped. The decisive evidence is three cohorts at **identical k = 16**:

| cohort (k=16) | few-shot @250 | separability | markers | platform |
|---|---|---|---|---|
| UPMC | 0.487 | 0.616 | 39 | CODEX |
| Keren | 0.566 | 0.720 | 39 | MIBI-TOF |
| Sorin | **0.790** | **0.824** | **17** | IMC |

Same class count, 0.30 spread, ordering matches separability exactly.

**Panel size is confounded with platform.** The three largest panels are the three CODEX cohorts.
CODEX mean 0.454, IMC mean 0.621. `corr(is_CODEX) = −0.644`. With n=7 these cannot be separated
observationally.

**The only causal test available**: subsample markers *within* a cohort (CRC 56 → 40 → 30 → 17),
same cells, same platform, same annotator. If performance rises as markers are removed, panel size
is causal. If flat or falling, the −0.879 is confounded and the driver travels with platform.
Not yet run.

---

## 6. ARI probe — and why it misled

`ari_probe.py`, `ari_controls.py`. ARI between k-means on the embedding and the cohort's own
native labels. Needs no shared vocabulary.

| | raw markers, no model | trained embedding |
|---|---|---|
| held-out cohort | 0.1392 | 0.1924 |
| cohort seen in training | 0.0483 | 0.2903 |

**This result caused a wrong conclusion and is recorded as a warning.** ARI 0.19 was read as "the
representation is weak", which pointed at redesigning the encoder. The linear probe then showed
the embedding supports macro-F1 0.5622 on the same cells.

**Reason.** ARI measures whether the **dominant unsupervised structure** matches the labels. Label
information can be present but subordinate. ARI cannot see it.

**Lesson.** Do not diagnose representation quality with an unsupervised clustering metric. Probe
it supervised. On its own, the ARI number would have sent this project to redesign the part that
works.

---

## 7. Ceiling estimates, and how much to trust them

| decoder | features | mean macro-F1 |
|---|---|---|
| logistic regression | `u_coh` | 0.619 – 0.659 |
| MAPS MLP | `u_coh` | **0.678** |
| logistic regression | `raw` | 0.675 |
| logistic regression | `raw` + `u_coh` | **0.695** |

A published, tuned MLP buys **+0.019** over logistic regression. The label information is
essentially **linearly decodable** — there is no large nonlinear reservoir a better model unlocks.
On UPMC, logistic regression beats MAPS.

**So the ceiling is ~0.70, not 1.0.** LOCO's 0.3151 is about **45% of achievable**, not 32% of
perfect. Report against the measured ceiling, not against 1.0.

**This is still not the true ceiling.** Label noise is unmeasured. Inter-annotator agreement would
give it and no second annotation exists — but `label_confidence` *is* a column in the raw tables
(`s6_train.load_conf` reads it; UPMC is the cohort that supplies one). Probe accuracy on high- vs
low-confidence cells is the closest available estimate. Not yet run.

---

## 8. Measurement caveats — read before quoting any number

- **Cross-harness drift ±0.04.** `u_coh` under logistic regression reads 0.619, 0.628 and 0.659
  across the three scripts here, because they filter classes and split differently. **Trust the
  deltas within a single run; do not quote absolute values to three decimals.**
- **Danenberg's class count is unstable** between runs (13 vs 29) — the fold label space used
  changes which cells survive the label filter. Excluded from the encoder decomposition.
- **"raw markers" was mislabelled** in the first pass of this work. Sections 1, 4, 6 and 7 use
  `u_coh` (harmonised). Only section 2 uses the true `raw::` intensities.
- **Everything here is expression-only.** `area_px2`, `x_px`, `y_px` are present in every raw
  table and are used nowhere in this project. The neighbour sidecars feed the encoder only.
- **n = 7.** Every correlation in section 5 rests on seven points.

---

## 9. What is NOT yet answered

| question | experiment | status |
|---|---|---|
| decoder design vs the price of zero target labels | retrieval vs prototype, zero-shot | **DONE — decoder is not the lever, see 1a** |
| does the harmonized vocabulary itself cost signal | probe held-out embedding → harmonized vs → native | **DONE — it helps, +0.053, see 1b** |
| where do the 17 zero-scoring classes go | confusion matrices, from cached predictions | **not run — cheapest** |
| is the few-shot exchange rate real | re-run the curve against **harmonized** labels | not run |
| is panel size causal or a platform artifact | marker subsampling within CRC | not run |
| panel *quality* vs panel *size* | `s1b_labels.lineage_score()` per cohort vs separability | not run |
| what is the label-noise ceiling | probe on high- vs low-confidence UPMC cells | not run |
| does morphology help | add `area_px2` to the probe | not run |

---

## How to re-run

Scripts are in this directory; results in `results/`. All CPU, all read-only against
`work/ckpt/` and `work/values/`.

```bash
# work/ must resolve to the artifact tree (a junction to dropped_past_works/work is fine)
cd pipeline2
python ../diagnostics/probe.py        # linear probe: raw vs embedding, seen vs held-out
python ../diagnostics/fewshot.py      # few-shot curve
python ../diagnostics/ceiling.py      # MAPS ceiling, held-out embedding
python ../diagnostics/ceiling_seen.py # MAPS on the in-distribution embedding
python ../diagnostics/harmcost.py     # cost of ECDF harmonisation
python ../diagnostics/ari_probe.py    # ARI probe (see section 6 before using)
python ../diagnostics/ari_controls.py # raw and in-distribution ARI controls
```

`ceiling.py` imports the vendored MAPS source. Its path is `dropped_past_works/MAPS` in these
scripts; `s10_external.py` expects `pipeline2_resultas/MAPS`. Whichever location survives the
repo reorganisation, both need to point at it.

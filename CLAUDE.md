# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A research pipeline that learns cell-type representations from **spatial proteomics** data (CODEX, IMC, MIBI-TOF) across seven cohorts that disagree on panel, value scale, tissue, and label names. The headline claim is tested by **leave-one-cohort-out (LOCO)**: hide a whole cohort, train on the rest, score once.

`README.md` explains the science stage by stage. Read it before changing method logic.

## Environment

Python 3.12, torch, pandas, numpy, matplotlib, scikit-learn. No `requirements.txt` — the environment is set up by hand. The local machine has **CPU-only torch**, so training stages need `--quick` (smoke test) or `--cpu`, and full gates are expected to run elsewhere.

`Datasets/` and `work/` are gitignored. `Datasets/` must be populated before Stage 0 can build anything.

## Running stages

All code is in `celltype_transfer/` (renamed from `pipeline2/` on 2026-09-17; stage files were renamed to say what they do). Every entry script adds its own folder to `sys.path`, so **run everything from the repo root**:

```bash
python celltype_transfer/run.py                 # list every step, in order, with its exact command
python celltype_transfer/run.py cpu             # run the local CPU steps 1-8
python celltype_transfer/load_cohorts.py --build
python celltype_transfer/train_adversarial_encoder.py --lambda-sweep --quick
```

Every script's module docstring lists its exact invocations. Read the docstring first — the flags are not in `--help` (most scripts parse `sys.argv` directly, not argparse). `build_label_space.py` reads some flags at IMPORT time (the `RA` dict), so importing it from a script that was given those flags changes its behaviour.

Order, with the gate each step produces (old `pipeline2` name in brackets, for reading old reports):

| # | Step | Command | Gate |
|---|---|---|---|
| 1 | load cohorts [s0_audit] | `load_cohorts.py --build` | 0 |
| 2 | marker identity [s0b_markers] | `resolve_markers.py` (`--offline`: only the recorded answers in `work/api_cache.json`) | 0b |
| 3 | value harmonisation [s1_values] | `harmonise_values.py` then `--bakeoff` | 1 |
| 4 | label space [s1b_labels] | `build_label_space.py --expect gate1b_v4_expect.csv` (`--resign` rebuilds signatures; the script's default is the v1 cases file) | 1b |
| 5 | vocabulary + wide tables [s2_tokens --build] | `build_marker_vocabulary.py` | — |
| 6 | label confidence [s6_confidence] | `build_label_confidence.py` (after step 5: it keeps only the cells in the wide tables) | — |
| 7 | fold-local label spaces [s7_spaces --folds] | `build_fold_label_spaces.py --folds` | 1b-fold |
| 8 | neighbour graph [s4_spatial --build] | `build_neighbour_graph.py` | 4 (checks 6–7) |
| 9 | masked-marker pretraining [s2_tokens] | `pretrain_masked_markers.py --check` then `--loto` | 2 |
| 10 | classifier / losses [s6_train] | `train_prototype_classifier.py --ablate-losses` | 6 |
| 11 | encoder + adversary [s3_encoder] | `train_adversarial_encoder.py --lambda-sweep` | 3 |
| 12 | spatial context [s4_spatial] | `train_spatial_context.py --gate` (`--pilot` = 1 fold, a signal) | 4 |
| 13 | external baseline [s10_external] | `compare_external_baseline.py --gate` | 10 |
| 14 | labels vs clusters [s13_label_cluster_comparison] | `compare_labels_to_clusters.py` | — |

Steps 9–12 need a GPU and run on Kaggle: `celltype_transfer/gpu/README.md` (bundle → notebook → `import_gpu_results.py`). The old 5-cohort stages (s1b_control main, s1c, s7_eval, s7b, s8, s9, s12) are archived in `dropped_past_works/pipeline2_stale/`, not ported.

Shared flags on the model stages: `--quick` (tiny smoke run, scores nothing — use this to prove a code path), `--refit` (ignore the checkpoint cache), `--cpu` (force CPU), `--folds A,B` (hold out only these folds). There is no `--report` flag: re-running a stage without `--refit` loads every finished fit from `work/ckpt/` and re-writes the report.

`CT_WORK` and `CT_REPORTS` environment variables redirect `work/` and `reports/` (used by the golden test and for side-by-side runs).

**Tests.** `python celltype_transfer/tests/golden.py --check` is a characterisation test: tiny CPU fits of every model stage plus exact re-scores of reference checkpoints, compared by weight hash with `tests/golden.json`. Any refactor must leave it PASSING — it proves no weight or score changed. Beyond that, **a stage's gate report is its test**: after changing a stage, re-run it and read the numbers in its `reports/*.md`. There is no linter.

## Architecture

### One declarative registry, no cohort branches

`celltype_transfer/config.py` holds `SPECS` — one dict per cohort describing base table, joins, marker-column rule, pixel size, and arrival scale. `loaders.py` is a single generic reader driven by those specs.

**There is no `if cohort == "X"` anywhere else in the codebase, and new code must not add one.** Adding a dataset means adding a dict.

`config.py` also owns every path (`WORK`, `VALUES`, `PANEL` = `celltype_transfer/declared/`, `REPORTS`, `RAW`), `SEED = 20260810`, `full_table(cohort)` and `rng(*purpose)` (the one per-purpose RNG every stage uses). Import from it; never hard-code a path. `models/device.py` holds the one device rule (`DEV`).

### The chain of artifacts

Stages talk to each other mainly through files in `work/`:

- `work/raw/{cohort}.parquet` — standard table, all cells, nothing dropped (Stage 0)
- `work/marker_registry.csv` — every raw marker column resolved to a `(gene, epitope, modification)` **triple** (Stage 0b)
- `work/values/{cohort}.parquet` — per-cohort mid-rank ECDF values, 40,000-cell stratified subsample (Stage 1)
- `work/label_signatures.npz`, `label_map.csv`, `label_graph.json`, `prototypes.npy` — the derived shared label space (step 4)
- `work/label_conf/{cohort}.parquet` — per-cell label confidence (step 5)
- `work/panel.json`, `work/values/{cohort}_full.parquet`, `marker_dynamic_range.csv` — the **frozen** marker vocabulary, one stable index per triple, and the wide tables every model reads (step 6)
- `work/spaces/` — one label space per fold + `index.json` hashes (step 7)
- `work/values/{cohort}_neighbours.npz`, `neighbour_graph_summary.csv` — the kNN graph (step 8)
- `work/ckpt/{pretrain,encoder,classifier,spatial,external}_*.pt` — cached fits, bypassed with `--refit`

Later stages also import earlier ones as modules (`import pretrain_masked_markers as pretrain`, `import train_adversarial_encoder as adversarial`, `import train_prototype_classifier as classifier`) to reuse their builders. `train_spatial_context.py` reads and PATCHES `classifier`'s module globals (`N_TRAIN`, `SCORE_CELLS`, `EPOCHS`, ...). Changing a builder's signature or a global's name breaks downstream stages silently — grep before renaming, and run the golden test.

### Three ideas the whole design rests on

1. **Marker identity is a triple, not a name.** CD45, CD45RA and CD45RO are all `PTPRC`; resolving to the gene alone would merge antibodies marking opposite states. `declared/never_merge.csv` is an assertion that the resolver works.
2. **An unmeasured marker is not a zero.** A cohort that never measured a marker gets an explicit absent-marker embedding. A zero means "negative", which is a different and false claim. Any change that reintroduces zero-filling is a bug.
3. **The label space is derived from marker signatures, not from names.** Text names are for readability only; they never drive clustering or similarity.

### Two evaluation protocols — keep them apart

- **7-fold LOCO** (`dropped_past_works/benchmark_protocol.yaml`) is the headline protocol and the only one in `celltype_transfer/`.
- **Ferguson frozen holdout** (`s7_eval.py`, archived in `dropped_past_works/pipeline2_stale/`) was an older zero-shot demo trained on five cohorts.

The whole-roster label space (`work/label_map.csv`) **was built with every cohort present** and must never be described as blind to any of them. LOCO numbers use the fold-local spaces. Do not mix the two protocols in one report.

### Subsampling and the spatial exception

Stages 1 onward work on a 40,000-cell stratified subsample per cohort (`N_SUB` in `harmonise_values.py`). The spatial stage is the exception: nearest neighbours inside a 1.9% subsample are not neighbours at all, so `build_neighbour_graph.py` reads the **full** raw tables and writes neighbour sidecars. Never rebuild the graph from `work/values/`.

## Gates and the pre-registration rule

Each stage produces a gate. Thresholds and expected cases are written **before** the run into `celltype_transfer/declared/gate*_expect.csv` (the file names are kept as registered), and the stage scores itself against that file.

Rules that keep this honest:

- Do not edit a `gate*_expect.csv` threshold to make a run pass. If a rule is replaced, keep the old row marked `REPLACED` and record why.
- A waived failure stays visible: set `waived=1` and write the evidence in `waiver_reason` (see the stroma row in `gate1b_expect.csv`).
- Comments in the code record failed attempts on purpose. Do not clean them out.

Stages write their own markdown report to `reports/` by appending lines to a list (`A = L.append`) and dumping it at the end. Follow that pattern when adding output.

## Current state (branch `rebuild-7cohort`)

Authoritative documents, in this order:

1. `dropped_past_works/benchmark_protocol.yaml` (v2, pre-registered) — the protocol.
2. `dropped_past_works/docs/plan_two_tracks.md` — status table and decision log (D-numbers) up to 2026-09-13.
3. `README.md` — method and rationale. `dropped_past_works/docs/full view.md` is a longer narrative version.

The results below are from the `pipeline2` runs. Their reports and checkpoints carry the OLD names (`s6_train.md`, `s4_*.pt`, ...) and live in `work_reference/` (a junction to `dropped_past_works/work`; reports unpacked under `work_reference/reports/`). `work/` is the fresh rebuild.

Live decisions from the v2 protocol that change how code should be written:

- **The hand mapping is gone** as a method, a gate, and a validation source. Comparison now happens in **Cell Ontology** space via CellMarker 2.0. Do not reintroduce hand-written label dictionaries.
- Three label types must never be collapsed: `native` (raw input, never treated as correct), `harmonized` (this project's derived space — a hypothesis, never called "the correct space"), `reference` (external, published).
- `work/` is the live output directory; `work_reference/` is the previous run, the comparison baseline. `dropped_past_works/` holds `MAPS/`, `_dead/` and the archived code (moved, not deleted).
- Gate 2 was re-run on the 109-triple vocabulary (2026-09-11, Kaggle T4): PASS, `stage2_arm = absent`. Its checkpoints carry a vocabulary fingerprint and a stale one is refitted, not loaded. `pretrain_masked_markers.py` picks the GPU automatically (`--cpu` forces CPU); on Kaggle use `celltype_transfer/gpu/gpu_session1.ipynb`.
- **Within-cohort splits are by patient** (plan F3). Always get train/val/test cells through `splits.split_masks(cohort, v.image_id)`, passing the table's `image_id` column unchanged (it already carries the `cohort|` prefix; never add a second one). Every checkpoint records `split_fp`; a mismatch refits (`splits.split_stale`). The slide-split Gate 2 run is archived (`reports/s2_masking_slidesplit.md`, `work/ckpt/_slide_split_2026-09-11/`). Gate 2 was re-run on the patient split (2026-09-11, Kaggle): PASS, Arm B ships again; its 24 + 2 LOTO checkpoints are in `work/ckpt/` and every LOCO/LOTO fold has a leak-free warm start.
- **LOCO/LOTO folds use fold-local label spaces** (plan F4). Build them with `python celltype_transfer/build_fold_label_spaces.py --folds` (→ `work/spaces/`, rules `declared/gate1b_fold_v2_expect.csv`); the encoder, classifier and spatial stages load a fold's space through `train_adversarial_encoder.space_for(held, mode)` — `--space fold` is the default, `--space shipped` is only the comparison row. The whole-roster `work/label_map.csv` was built with every held-out cohort present and must never be a LOCO headline. Three LOCO folds (CRC, Keren, UPMC) use a post-hoc nearest-feasible cut; carry that flag into any number from them.
- **Gates 6 and 3 have run on this protocol** (2026-09-12, Kaggle T4, `kaggle/stage36.ipynb`). Gate 6: **FAIL by 0.0008** on check 4, headline LOCO macro-F1 **0.3151**; not waived. Gate 3: PASS, ships **lambda 0.01**, which overturns D-36 on this roster — Stage 6 trained without an adversary, so that gap is an open decision, not a settled one. Reports: `reports/s6_train.md`, `reports/s3_encoder.md`.
- **Gate 4 has run and PASSED** (2026-09-13, Kaggle T4, `kaggle/stage4.ipynb`, 315.6 min, 30 fits;
  report `reports/s4_spatial.md`, checkpoints `work/ckpt/s4_*.pt`). Headline LOCO macro-F1: `cell`
  0.2956, `neigh` 0.3063, `shuffle` 0.2932, `ctx` 0.3164. Checks 1/2/2b/5/6/7/8 all PASS: `neigh`
  beats `cell` (+0.0107) and beats the same-image `shuffle` control (+0.0130), and `shuffle` falls
  back toward `cell` as the leakage control requires. **Check 3's paired 95% CI on (neigh − cell)
  spans zero** ([-0.0239, +0.0453], exact sign-flip p = 0.547, n = 7) — per `gate4_expect.csv` that
  means the sentence "spatial context improves transfer" is NOT earned, whatever the PASS or the
  mean says. This is an open question, not a settled win. The optional 4th-loss arm `ctx` beats
  `neigh` (+0.0101, check 4, not required) and **ships**. UPMC's assumed `px_um` was rescaled ×1.3
  and ×0.77 and check 1's verdict did not move (check 5 PASS), so the distance features are not an
  artifact of the unpublished scale.
  **`nbr_same` (the homotypic fraction) must never become a feature** — it is built from the
  neighbours' native labels, which on the held-out cohort are the thing being predicted.
  Three design decisions in `train_spatial_context.py` that must not be quietly reverted:
  (a) the neighbourhood enters as a **residual on `z_cell` through a zero-initialised gate**, so
  arm `neigh` starts bit-identical to arm `cell` and must earn its margin — concatenation would
  widen the prototype space and break Stage 6's prototype initialisation;
  (b) the pooled mean is **not** the only neighbour feature. `het` (neighbour disagreement) and
  `d_self` (cell vs neighbourhood average, a boundary detector) exist because a mean over 15
  neighbours can express neither "is a vessel touching me" (max) nor "am I at a boundary"
  (variance) — without them a FAILED gate could not be distinguished from pooling that destroyed
  the signal. `d_self` is computed **after** the shuffle, or the control would not destroy it;
  (c) attention over neighbours is deliberately **not** used yet: 2–3× compute, and a failure
  would be unattributable (idea or architecture). Gate 4 passed, so attention is now the
  candidate next experiment — but check 3's interval spans zero, so that decision should weigh
  the n=7 caveat above, not just the PASS.
- **Stage 2 warm starts are fold-local.** Always get Stage 2 weights through `pretrain_masked_markers.warm_for(train_cohorts)`, never by loading `pretrain_armB.pt` directly: it returns only a checkpoint whose recorded training cohorts are a subset of the fit's, and raises when none exists. Loading `pretrain_armB.pt` for a LOCO fold leaks the held-out cohort (it was trained on all 7). `--no-warm` is the only deliberate cold start.
- **Fresh re-run COMPLETE (2026-09-18/19), `reports/fresh_run_summary.md`.** Every stage re-ran from `Datasets/` through `celltype_transfer/` and reproduced the old run EXACTLY: CPU artifacts byte-identical, 82/82 shared GPU fits bit-identical weights, all 30 Gate 4 and 21 MAPS fits identical scores. New: Gate 6 check 3b `proto2adv` 0.2964 (−0.0187, adversary not shipped); check 6-v2 confidence on−off +0.0001 (CI ±0.034, no effect); `ctx` vs `proto2` +0.0013 (CI [−0.043, +0.045], a tie). Gate 4's `cell` arm has the same weights as Gate 6 `proto3`. The 7-cohort Gate 1 bake-off picks `V2b` on mean R² but the protocol fixes `V3`; not acted on. Run steps with `run.py`: markers `--offline` (the API cache is an input), label space `--expect gate1b_v4_expect.csv`, confidence AFTER vocabulary. Open: Phase 5 candidate 1 (spatial on the 2-loss config), declared first.
- **Route C (learned prototypes) was tried and dropped, 2026-09-13 — negative result, code removed.** Full account in `docs/plan_two_tracks.md`'s Corrections section. One-line lesson: it correctly kept different native labels apart but never merged any of Gate 1b's declared "same" cross-cohort cases (0 of 7), because no loss term in that design ever compared cells from two different cohorts — removing batch identity (the adversary) is not the same as positively pulling matching biology together across cohorts, and cross-cohort merging does not emerge for free just because batch signal is erased. **Any future method aimed at recognizing that two cohorts' differently-named labels are the same cell type needs an explicit cross-cohort comparison term, or it will fail the same way regardless of architecture.**

## Writing about results

When reporting a run, use: **Result** (what happened) → **Meaning** (what it implies) → **Reason** (why) → **Next** (what to check). State limitations plainly instead of smoothing them over — much of this project's value is in the negative results staying visible.


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



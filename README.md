# A Transferable Cell-Type Representation Framework Across Heterogeneous Spatial Proteomics Datasets

Can a model label cell types in a dataset it has never seen, when that dataset uses a different machine, antibody panel, tissue and label names?

## Data

7 public cancer cohorts, about 6 million cells:

| Cohort | Platform | Tissue |
|---|---|---|
| CRC | CODEX | colorectal |
| UPMC | CODEX | head and neck |
| Phillips | CODEX | skin |
| Keren | MIBI-TOF | breast |
| Danenberg | IMC | breast |
| Sorin | IMC | lung |
| Ferguson | IMC | skin |

Raw data is not included. See `celltype_transfer/data_acquisition/` to download it.

## Method in short

1. **Match markers** by molecule (gene, epitope, modification), not by name.
2. **Put values on one scale** with a per-cohort rank transform.
3. **Build a shared label space** by grouping labels with similar marker profiles. It uses no text and no hand-made mapping.
4. **Encode each cell as a set of marker tokens.** A marker a cohort never measured gets an "absent" token, not a zero.
5. **Pretrain** by hiding markers and predicting them, then **train a prototype classifier**.

## How it is tested

Leave-one-cohort-out: hide one cohort, train on the other six, score the hidden one. Repeat for all 7.

## Main result

- Macro-F1 on unseen cohorts: **0.315**, compared with 0.059 for random guessing and 0.007 for always picking the most common class.
- Filling unmeasured markers with zeros destroys transfer (0.017).
- A published baseline (MAPS) scores 0.332. With only 7 cohorts, the two are a statistical tie.
- Spatial context and a batch adversary did not give a measurable gain.

Full details: [thesis_report/methodology.md](thesis_report/methodology.md) and [thesis_report/results.md](thesis_report/results.md).

## How to run

Python 3.12 with torch, pandas, numpy, scipy, scikit-learn, networkx and matplotlib. Run from the repo root:

```bash
python celltype_transfer/run.py        # list every step and its command
python celltype_transfer/run.py cpu    # run the CPU steps (1-8)
```

Training steps (9-13) need a GPU. See `celltype_transfer/gpu/README.md`.

## Repo layout

| Folder | What is in it |
|---|---|
| `celltype_transfer/` | the pipeline code |
| `celltype_transfer/declared/` | pass/fail rules, written before each run |
| `celltype_transfer/tests/` | reproducibility test (`golden.py --check`) |
| `diagnostics/` | extra analyses run after the main results |
| `thesis_report/` | methodology and results chapters |

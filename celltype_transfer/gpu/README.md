# Running the GPU stages on Kaggle

The model stages need a GPU. This folder holds everything for running them on Kaggle
(GPU T4 x2, 12 h per session, Internet off).

| File | What it does |
|---|---|
| `make_gpu_bundle.py` | Builds the folder you upload as a Kaggle Dataset: the code, the data tables, and `MANIFEST.json` (a hash for every file). |
| `notebooks.py` | Writes `gpu_session1.ipynb` and `gpu_session2.ipynb`. Edit this file, not the notebooks. |
| `import_gpu_results.py` | Checks a downloaded results zip, then copies it into `work/` and `reports/`. |

## The two sessions

| Session | Gates | Fits | Expected time on 2 × T4 |
|---|---|---|---|
| 1 | Gate 2 pretraining (+ LOTO), Gate 6 classifier, Gate 3 encoder | 26 + 34 + 35 | about 4.5–5 h |
| 2 | Gate 4 spatial context | 30 | about 3 h |

Session 2 needs session 1's pretraining checkpoints, so it can only start after session 1 has
been imported.

## Steps

All commands run from the repo root.

**Session 1**

1. Build the local artifacts first (see the main `README.md`, steps 0–7).
2. `python celltype_transfer/gpu/make_gpu_bundle.py --session1`
   - It must end with "nothing missing".
3. Upload `work/gpu_bundle_session1.zip` to Kaggle as a new private Dataset named `cta-session1`.
4. Create a notebook from `celltype_transfer/gpu/gpu_session1.ipynb` (File → Import Notebook).
   - Settings: Accelerator = GPU T4 x2, Internet = off.
   - Add the dataset (Add Input).
5. Click Save Version → Save & Run All.
6. When it finishes, download `session1_results.zip` from the Output tab.
7. `python celltype_transfer/gpu/import_gpu_results.py session1_results.zip`

**Session 2**

8. `python celltype_transfer/gpu/make_gpu_bundle.py --session2`
9. Upload it as `cta-session2`, import `gpu_session2.ipynb`, and run it the same way.
10. `python celltype_transfer/gpu/import_gpu_results.py session2_results.zip`

## What each notebook checks before it trains

- Every uploaded file matches its hash in `MANIFEST.json`.
- Vocabulary fingerprint `109:66147d20` and patient split `patient-v1:015916c8`.
- Every fold's label space matches the hash recorded when it was built.
- Every fold's warm start never saw its held-out cohort.
- A 2-epoch smoke test on the GPU passes. It scores nothing, and its files are deleted afterwards.

## How the two GPUs are used

The training scripts use one GPU each. For each gate the notebook:

1. starts two worker processes, one per GPU, each holding out a different set of folds (`--folds`);
2. waits for both (it prints progress every 10 minutes);
3. runs the script once more over all folds.
   - Every finished fit loads from the cache in `work/ckpt/`, and the gate report is written.
   - If a worker crashed, this run trains the missing fits itself, so nothing is skipped silently.

Each fit is the same computation it would be on one GPU. Only the wall time halves.

## If a session is cut off

- Results are zipped after every gate, before anything is displayed:
  - `session1_after_gate2.zip`, then `session1_after_gate6.zip`, then `session1_results.zip`;
  - `session2_results.zip` for session 2.
- Download the newest zip and import it. Then rebuild the bundle and run again.
- Finished fits are in the checkpoint cache, so a re-run only trains what is missing.
  - This is true within one Kaggle session.
  - Across sessions, the imported checkpoints must be in the new upload. Session 2's bundle carries
    the pretraining checkpoints, but session 1's bundle carries none. So if session 1 stops
    partway, the next run starts it from the beginning.

## Are GPU results reproducible?

- Measured 2026-09-18: yes, on the same hardware. Session 1 was re-run from scratch on Kaggle
  T4s, and all 82 fits that also exist in the previous run have **bit-identical weights**
  (26 pretraining, 21 classifier, 35 encoder). Every per-fold score matches to the last digit.
- This holds for the same GPU model and software. A different GPU (for example a P100) or
  another torch/CUDA version can change the last bits, and small drifts can then grow during
  training. So the declared ±0.015 rule in the fresh-run summary stays as the pass mark.
- The exact code check is still `tests/golden.py`, which runs on the CPU.

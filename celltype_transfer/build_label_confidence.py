"""
Build work/label_conf/{cohort}.parquet - per-cell label confidence, keyed on cell_id.

    python celltype_transfer/build_label_confidence.py

WHY A SIDECAR rather than a column in the wide tables. Adding it to work/values/{c}_full.parquet
would mean re-running Stage 2's --build, which regenerates the tables every downstream number was
measured on. A separate small file is additive: nothing already measured can change, and it costs
about 1 MB to carry to a GPU box instead of the 627 MB of raw tables.

WHAT IT ACTUALLY COVERS - 1 of the 7 cohorts. Only UPMC ships a real confidence (kNN.prob,
51 distinct values spanning 0.14 to 1.0; re-measured 2026-09-17). CRC, Keren, ferguson, Phillips,
Danenberg and Sorin all report a constant 1.0, where confidence weighting is exactly equal to
plain cross-entropy. (Written as "1 of 5" on the old 5+1 roster.) Gate 6 check 6-v2 compares
confidence on vs off on the 6 folds where UPMC trains, as a diagnostic, never as a roster-wide
effect. (Check 6 v1 ran only on the UPMC fold, where no training cohort has a confidence - D-45.)
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK, SPECS, raw_table

OUT = os.path.join(WORK, 'label_conf')


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for c in [c for c, s in SPECS.items() if os.path.exists(raw_table(c))]:
        d = pd.read_parquet(raw_table(c), columns=['cell_id', 'label_confidence'])
        v = pd.to_numeric(d.label_confidence, errors='coerce').fillna(1.0).clip(0.0, 1.0)
        informative = bool(v.nunique() > 1)

        # Only the cells the wide table actually holds - Stage 6 never sees the rest, and the
        # full column is 50x bigger for nothing.
        wt = os.path.join(WORK, 'values', f'{c}_full.parquet')
        keep = len(d)
        if os.path.exists(wt):
            ids = set(pd.read_parquet(wt, columns=['cell_id']).cell_id)
            m = d.cell_id.isin(ids)
            d, v, keep = d[m], v[m], int(m.sum())

        # NO FILE for a cohort whose confidence is constant. Weighting there is exactly plain
        # cross-entropy, so a column of 1.0s is pure carriage. Stage 6 defaults to 1.0 when the
        # file is absent, which also makes "a file exists" mean "there is real information here".
        kb = 0
        if informative:
            p = os.path.join(OUT, f'{c}.parquet')
            pd.DataFrame(dict(cell_id=d.cell_id.values,
                              conf=v.values.astype('float32'))).to_parquet(p, index=False)
            kb = round(os.path.getsize(p) / 1e3)
        rows.append(dict(cohort=c, cells=keep, unique=int(v.nunique()),
                         lo=round(float(v.min()), 4), hi=round(float(v.max()), 4),
                         informative=informative, kb=kb))
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    n = int(t.informative.sum())
    print(f"\n{n} of {len(t)} cohorts carry a real confidence. "
          f"On the other {len(t)-n}, weighting is identical to plain cross-entropy.")
    print(f"total {t.kb.sum():.0f} KB -> {OUT}")


if __name__ == "__main__":
    main()

"""The marker vocabulary and the wide value tables - the first half of what `s2_tokens.py --build` did.

    python celltype_transfer/build_marker_vocabulary.py    # work/panel.json, work/values/{c}_full.parquet,
                                                           # work/marker_dynamic_range.csv  (needs work/raw/)

THE VOCABULARY. One token per protein TRIPLE (gene_or_complex, epitope, modification) that any
built cohort measures - never a marker NAME (decision D-9, which keeps CD45 / CD45RA / CD45RO
apart). work/panel.json is written here once and is the single source of truth from then on
(D-39): every model stage reads it with `read_panel()`, never re-derives it from files on disk.

THE WIDE TABLES. For each cohort, the per-cohort mid-rank ECDF value (`u_coh`) and the raw value
of every panel marker, for exactly the cells Stage 1 already subsampled.

The dynamic-range score is built HERE. The plan said Stage 0b computed it; it does not, and that is
recorded as a disproven assumption in the decision log. The naive measure fails anyway: `u_coh` is a
rank, so its IQR is 0.5 by construction, and raw IQR is not comparable across cohorts on different
scales. Two scale-free measures replace it - `tie_mass` (share of cells on the single most common
raw value) and IQR / (p99 - p01). `rank_spread` (variance of u_coh / 1/12) decides the D-28
exclusion rule, applied in pretrain_masked_markers.py.
"""
import os
import sys
import json
import time
import zlib

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # runs from any directory
import panel_utils
from config import SPECS, WORK, raw_table, value_table, full_table

UNIFORM_VAR = 1.0 / 12.0                # variance of an untied per-cohort rank

META = ['cell_id', 'cohort', 'image_id', 'patient_id', 'native_label']


# ----------------------------------------------------------------------------- panel
def registry():
    return pd.read_csv(os.path.join(WORK, 'marker_registry.csv'), keep_default_na=False)


def built():
    return [c for c in SPECS if os.path.exists(raw_table(c))]


def panel_spec(cohorts):
    """The token vocabulary: every protein triple any built cohort measures.

    Identity is the resolved triple (gene_or_complex, epitope, modification) from Stage 0b, never
    a marker NAME - that is decision D-9, and it is what keeps CD45 / CD45RA / CD45RO apart.
    """
    r = registry()
    r = r[(r.kind != 'non_protein') & r.cohort.isin(cohorts)]
    triples = sorted(r.triple.unique())
    tri2idx = {t: i for i, t in enumerate(triples)}
    per = {c: sorted(r[r.cohort == c].triple.unique()) for c in cohorts}
    genes = r.drop_duplicates('triple').set_index('triple').gene.to_dict()
    ncoh = r.groupby('triple').cohort.nunique().to_dict()
    return triples, tri2idx, per, genes, ncoh


def read_panel(cohorts=None):
    """The CANONICAL token vocabulary, read from work/panel.json. Never recomputed.

    Why this exists (D-39). Stage 3 originally derived its vocabulary by scanning which
    {cohort}_full.parquet files were on disk. That makes the token index space depend on which
    FILES ARE PRESENT, not on the data: locally, with ferguson's table there, it resolved to 99
    triples; on the Kaggle box, where ferguson is deliberately not uploaded because it is the
    frozen holdout, it resolved to 88. Same code, same seed, different model.

    Three things broke quietly:
      - the two runs were not comparable, while both printed a confident vocabulary line;
      - `ident.weight` is [n_vocab, d_tok], so at 88 it no longer matched the Stage 2 checkpoint
        and load_stage2() dropped it on a shape test - the single most valuable pretrained tensor,
        skipped while the log still said "warm-starting the token layer";
      - work/prototypes.npy is [25, 99], so a Stage 6 built on 88 slots would have misaligned
        every prototype BY INDEX, with nothing to signal it.

    panel.json is written once by Stage 2 and is the single source of truth from then on.
    `cohorts` optionally restricts only the per-cohort marker lists, never the index space.
    """
    p = os.path.join(WORK, 'panel.json')
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing - run `python celltype_transfer/pretrain_masked_markers.py --build` first.")
    o = json.load(open(p))
    triples = [t for t, _ in sorted(o['index'].items(), key=lambda kv: kv[1])]
    per = {c: v for c, v in o['per_cohort'].items()
           if cohorts is None or c in cohorts}
    return triples, dict(o['index']), per, dict(o['gene']), \
        {t: int(n) for t, n in o['cohorts_measuring'].items()}


def write_panel(triples, per, genes, ncoh):
    """work/panel.json - the token vocabulary.

    The plan attributed this file to Stage 0b. It was never written there, the same way the
    dynamic-range score was never written there. Both are built here instead, and both corrections
    are recorded rather than quietly patched.
    """
    p = os.path.join(WORK, 'panel.json')
    obj = dict(n_vocab=len(triples),
               index={t: i for i, t in enumerate(triples)},
               gene={t: genes[t] for t in triples},
               cohorts_measuring={t: int(ncoh[t]) for t in triples},
               per_cohort={c: v for c, v in per.items()})
    # a --build must not silently drop the arm Gate 2 decided; Stage 3 reads it to pick which
    # checkpoint to warm-start from, and a wrong default is a silent, hard-to-spot error
    if os.path.exists(p):
        old = json.load(open(p))
        for k in ('stage2_arm', 'stage2_arm_note'):
            if k in old:
                obj[k] = old[k]
    json.dump(obj, open(p, 'w'), indent=1)
    return p


def panel_fp():
    """The vocabulary fingerprint of work/panel.json - same formula as main()'s VOCAB_FP."""
    triples = read_panel()[0]
    return f"{len(triples)}:{zlib.crc32('|'.join(triples).encode()):08x}"


# ----------------------------------------------------------------------------- build
def build(cohort, triples_c):
    """One pass over a cohort: the full-panel value table plus its dynamic-range statistics.

    References (the cohort ECDF) come from ALL cells; only the written table is subsampled. The
    subsample is not re-drawn - it reuses the exact cell ids Stage 1 already wrote, so the two
    stages stay aligned and the run reproduces.

    `method='average'` (mid-rank) again splits ties down the middle. Sorin arrives uint8, so ties
    are everywhere and a left- or right-rank would bias every quantised marker.
    """
    cols = panel_utils.cols_for(registry(), cohort, triples_c)
    df = pd.read_parquet(raw_table(cohort),
                         columns=META + panel_utils.read_cols(cols, triples_c))
    X = panel_utils.matrix(df, cols, triples_c)
    n = len(X)

    # --- dynamic range, measured on RAW values over EVERY cell of the cohort.
    #     tie_mass is the statistic that matters: it directly captures Sorin's uint8
    #     zero-inflation and UPMC's flat markers, and it is exactly the condition under which a
    #     constant predictor wins and the reconstruction loss stops measuring biology.
    u = X.rank(pct=True, method='average').astype('float32')

    rows = []
    for t in triples_c:
        s = X[t]
        vc = s.value_counts(dropna=False)
        tie = float(vc.iat[0] / n) if len(vc) else 1.0
        q = s.quantile([0.01, 0.25, 0.75, 0.99]).to_numpy()
        den = float(q[3] - q[0])
        var = float(u[t].var())
        rows.append(dict(cohort=cohort, triple=t,
                         tie_mass=round(tie, 4),
                         robust_dispersion=round(float(q[2] - q[1]) / den, 4) if den > 1e-12 else 0.0,
                         var_ucoh=round(var, 6),
                         rank_spread=round(var / UNIFORM_VAR, 4),
                         distinct_values=int(len(vc)), cells=n))

    keep = pd.read_parquet(value_table(cohort), columns=['cell_id']).cell_id.to_numpy()
    pos = pd.Index(df.cell_id).get_indexer(keep)
    assert (pos >= 0).all(), f'{cohort}: Stage 1 cell ids missing from the raw table'

    out = df.iloc[pos][META].reset_index(drop=True)
    for name, block in (('u_coh', u), ('raw', X)):
        b = block.iloc[pos].reset_index(drop=True)
        b.columns = [f'{name}::{t}' for t in triples_c]
        out = pd.concat([out, b], axis=1)
    out.to_parquet(full_table(cohort), index=False)

    return pd.DataFrame(rows), dict(cohort=cohort, markers=len(triples_c), cells_total=n,
                                    cells_kept=len(out), slides=int(df.image_id.nunique()))


def do_build(cohorts, per):
    dyn, info = [], []
    for c in cohorts:
        t0 = time.time()
        d, i = build(c, per[c])
        i['seconds'] = round(time.time() - t0, 1)
        dyn.append(d); info.append(i)
        print(f"  {c:9} {i['markers']:>3} markers  {i['cells_kept']:>7,} / {i['cells_total']:>9,}"
              f" cells  {i['slides']:>4} slides  {i['seconds']:>5.1f}s")
    D = pd.concat(dyn, ignore_index=True)
    D.to_csv(os.path.join(WORK, 'marker_dynamic_range.csv'), index=False)
    return D, pd.DataFrame(info)


def main():
    cohorts = built()
    if not cohorts:
        sys.exit("no work/raw/*.parquet - run load_cohorts.py --build first (this step needs raw)")
    triples, tri2idx, per, genes, ncoh = panel_spec(cohorts)
    fp = f"{len(triples)}:{zlib.crc32('|'.join(triples).encode()):08x}"
    print(f"vocabulary: {len(triples)} marker triples across {len(cohorts)} cohorts "
          f"(fingerprint {fp})")
    for c in cohorts:
        print(f"   {c:9} {len(per[c]):>3} markers  ({SPECS[c]['role']})")
    print("\nbuilding wide value tables + dynamic-range score...")
    do_build(cohorts, per)
    print(f"  wrote {write_panel(triples, per, genes, ncoh)}")


if __name__ == "__main__":
    main()

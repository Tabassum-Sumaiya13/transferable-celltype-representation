"""Shared marker-matrix helper - one column per triple, duplicate reagents averaged.

THE PROBLEM THIS FIXES. One triple can be carried by SEVERAL raw columns inside one cohort.
Danenberg ships two HER2 antibody clones - `HER2 (3B5)` and `HER2 (D8F12)` - and both resolve
to HGNC:3430. No earlier cohort produced a duplicate, so nothing was ever written to handle it.

What happened before: all three readers (harmonise_values.build, pretrain_masked_markers.build, build_label_space.
build_signatures) called `drop_duplicates('triple')`, which silently kept whichever clone came
first in registry order and discarded the other. Nothing logged it, and nothing declared it.
That is a policy - "keep the first" - that no one chose and no reader could see.

DECLARED POLICY (panel/gate0b_expect.csv, committed before the Stage 1 rebuild):
duplicate reagents for one triple are AVERAGED, and every duplicate is printed when it is used.

Why average rather than keep one. Both clones measure the same protein, so averaging uses both
channels the dataset actually shipped and cancels clone-specific noise. Keeping one discards a
real measurement - the same mistake as feeding a zero for a marker nobody looked at, which is
the failure Stage 2 exists to avoid.

Why not pick the higher-dynamic-range clone (the other option files/03 3.3 floated). It needs
the dynamic range before the value table exists, so it would reorder the pipeline; and it is a
data-dependent choice made after seeing the data, which is what declaring policies in advance
exists to prevent.
"""
import pandas as pd

_announced = set()


def cols_for(reg, cohort, triples, verbose=True):
    """{triple: [raw_column, ...]} for one cohort. A list of length > 1 is a duplicate reagent.

    `reg` is the full marker registry. Order inside each list follows registry order, so the
    result is deterministic and a re-run reproduces byte-identically.
    """
    g = reg[reg.cohort == cohort]
    out = {}
    for t in triples:
        cs = g.loc[g.triple == t, 'raw_column'].tolist()
        if not cs:
            raise KeyError(f"[{cohort}] no raw column for triple {t}")
        out[t] = cs
    dup = {t: cs for t, cs in out.items() if len(cs) > 1}
    if dup and verbose and cohort not in _announced:
        _announced.add(cohort)
        for t, cs in sorted(dup.items()):
            print(f"  [{cohort}] duplicate reagent for {t}: {cs} -> averaged (declared policy)")
    return out


def matrix(df, cols, triples):
    """DataFrame with exactly one float32 column per triple, in `triples` order.

    Duplicate reagents are averaged across their raw columns. `skipna=True` is deliberate: if
    one clone is missing on a cell the other still carries the measurement, which is the whole
    reason both are kept.
    """
    data = {}
    for t in triples:
        cs = cols[t]
        data[t] = (df[cs[0]].astype('float32') if len(cs) == 1
                   else df[cs].astype('float32').mean(axis=1, skipna=True).astype('float32'))
    return pd.DataFrame(data, columns=list(triples), index=df.index)


def read_cols(cols, triples):
    """The flat, de-duplicated raw-column list to hand to pd.read_parquet(columns=...)."""
    seen, out = set(), []
    for t in triples:
        for c in cols[t]:
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out

"""#2 - THE NAME-MATCHING BASELINE. Tests this project's founding claim.

README: "The label space is derived from marker signatures, not from names. Text names are for
readability only; they never drive clustering or similarity."

That claim has never been tested against the obvious alternative: just merge labels whose NAMES
are similar. This is the anti-strawman for harmonisation, the role MAPS plays for annotation.

Four name-similarity variants, each cut to the SAME number of clusters as the shipped space so
granularity is matched. All partitions then scored on the same three measures:

  declared cases      gate1b_v4_expect.csv, same/different (nested needs a graph, skipped)
  merge fidelity      does it merge pairs that are textbook-separable on the 9 universal markers
  ARI vs shipped      how much of the signature partition do names alone reproduce
"""
import os
import re
import sys
import itertools

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"D:\Desktop\FYDP\FYDP final works\cell type annotation"
sys.path.insert(0, os.path.join(ROOT, 'pipeline2'))
os.chdir(os.path.join(ROOT, 'pipeline2'))
sys.argv = [sys.argv[0]]

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score

import s2_tokens as s2
import s3_encoder as s3
from config import WORK

UNIV = ['COMPLEX:CD3|pan|none', 'HGNC:1678|pan|none', 'HGNC:1706|pan|none',
        'HGNC:6106|pan|none', 'HGNC:7315|pan|none', 'HGNC:8823|pan|none',
        'FAMILY:KRT_PAN|pan|none', 'HGNC:1693|pan|none', 'COMPLEX:HLA-DR|pan|none']

lm = pd.read_csv(os.path.join(WORK, 'label_map.csv'))
lm['key'] = lm.cohort + '|' + lm.label.astype(str)
K = lm.cluster.nunique()
print(f"shipped space: {len(lm)} labels, {K} clusters\n")


def norm(s):
    s = re.sub(r'[^a-z0-9 ]', ' ', str(s).lower())
    return re.sub(r'\s+', ' ', s).strip()


names = [norm(x) for x in lm.label]


def tok_jaccard():
    T = [set(n.split()) for n in names]
    return np.array([[len(a & b) / max(1, len(a | b)) for b in T] for a in T])


def char_jaccard(n=3):
    G = [set(x[i:i + n] for i in range(max(1, len(x) - n + 1))) for x in names]
    return np.array([[len(a & b) / max(1, len(a | b)) for b in G] for a in G])


def tfidf_char():
    X = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4)).fit_transform(names)
    X = X.toarray()
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
    return Xn @ Xn.T


def exact():
    return np.array([[1.0 if a == b else 0.0 for b in names] for a in names])


VARIANTS = {'token_jaccard': tok_jaccard(), 'char3_jaccard': char_jaccard(),
            'tfidf_char': tfidf_char(), 'exact_name': exact()}

parts = {'SHIPPED (signatures)': lm.cluster.to_numpy()}
for nm, S in VARIANTS.items():
    D = 1.0 - np.clip(S, 0, 1)
    np.fill_diagonal(D, 0.0)
    parts[nm] = AgglomerativeClustering(n_clusters=K, metric='precomputed',
                                        linkage='average').fit_predict(D)

# ---------- declared cases
exp = pd.read_csv(os.path.join(ROOT, 'pipeline2', 'panel', 'gate1b_v4_expect.csv'))
idx = {k: i for i, k in enumerate(lm.key)}


def score_cases(lab):
    ok = tot = 0
    detail = []
    for _, r in exp.iterrows():
        if r['relation'] not in ('same', 'different'):
            continue
        mem = [m for m in str(r['members']).split(';') if m in idx]
        if len(mem) < 2:
            continue
        cl = {lab[idx[m]] for m in mem}
        good = (len(cl) == 1) if r['relation'] == 'same' else (len(cl) == len(mem))
        ok += good
        tot += 1
        detail.append((r['case'], r['relation'], good))
    return ok, tot, detail


# ---------- merge fidelity on the universal 9
cohorts = s3.available()
prof = {}
for c in cohorts:
    v = pd.read_parquet(s2.full_table(c), columns=['native_label'] + [f'u_coh::{t}' for t in UNIV])
    prof[c] = v.groupby('native_label')[[f'u_coh::{t}' for t in UNIV]].mean()

gaps, pidx = [], []
for a, b in itertools.combinations(range(len(lm)), 2):
    ca, la = lm.cohort.iloc[a], lm.label.iloc[a]
    cb, lb = lm.cohort.iloc[b], lm.label.iloc[b]
    if ca != cb or la not in prof[ca].index or lb not in prof[cb].index:
        continue
    gaps.append(float(np.abs(prof[ca].loc[la].values - prof[cb].loc[lb].values).max()))
    pidx.append((a, b))
gaps = np.array(gaps)
sep = gaps > 0.5
print(f"{len(gaps)} within-cohort pairs, {sep.sum()} textbook-separable (gap > 0.5)\n")

rows = []
for nm, lab in parts.items():
    ok, tot, det = score_cases(lab)
    merged = np.array([lab[a] == lab[b] for a, b in pidx])
    rows.append(dict(method=nm, clusters=len(set(lab)),
                     cases=f"{ok}/{tot}",
                     bad_merges=f"{int(merged[sep].sum())}/{int(sep.sum())}",
                     bad_merge_rate=round(float(merged[sep].mean()), 4),
                     ari_vs_shipped=round(float(adjusted_rand_score(parts['SHIPPED (signatures)'], lab)), 3)))
d = pd.DataFrame(rows)
print(d.to_string(index=False))
d.to_csv(os.path.join(HERE, 'results', 'name_baseline.csv'), index=False)

print("\nper-case, shipped vs best name baseline:")
best = max((k for k in VARIANTS), key=lambda k: score_cases(parts[k])[0])
_, _, ds = score_cases(parts['SHIPPED (signatures)'])
_, _, dn = score_cases(parts[best])
print(f"  {'case':26} {'rel':10} {'signatures':11} {best}")
for (c, r, g1), (_, _, g2) in zip(ds, dn):
    print(f"  {c:26} {r:10} {'PASS' if g1 else 'FAIL':11} {'PASS' if g2 else 'FAIL'}")

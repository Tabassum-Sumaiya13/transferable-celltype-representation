"""CROSS-FOLD STABILITY - do seven independently built harmonisations agree?

work/spaces/ holds 7 fold-local label spaces, each clustered WITHOUT one cohort. If the
harmonisation is reproducible, the seven should merge the labels they share the same way.

Each pair of folds is compared only on labels from cohorts that were TRAINING in BOTH - the
held-out cohort's labels are placed post-hoc by a different rule and are not comparable.

  ARI          agreement between two folds' partitions on their commonly-trained labels
  pair vote    for every label pair, how often across the 7 folds is the merge decision the same
"""
import os
import sys
import itertools

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"D:\Desktop\FYDP\FYDP final works\cell type annotation"
sys.path.insert(0, os.path.join(ROOT, 'pipeline2'))
sys.argv = [sys.argv[0]]

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

SP = os.path.join(ROOT, 'work', 'spaces')

folds = {}
for f in sorted(os.listdir(SP)):
    if f.startswith('label_map_loco_') and f.endswith('.csv'):
        h = f.replace('label_map_loco_', '').replace('.csv', '')
        m = pd.read_csv(os.path.join(SP, f))
        m['key'] = m.cohort + '|' + m.label.astype(str)
        folds[h] = m.set_index('key')

print("granularity per fold (this is already a stability signal):")
for h, m in folds.items():
    ok = m[m.cluster >= 0]
    print(f"  loco_{h:10} {m.cluster.nunique():>3} clusters   "
          f"{len(ok):>3} labels placed   {int(m.unreliable.sum()):>3} unreliable")
cn = [m.cluster.nunique() for m in folds.values()]
print(f"  cluster count: min={min(cn)} max={max(cn)} ratio={max(cn)/min(cn):.1f}x\n")

# ---- pairwise ARI on commonly-trained labels only
names = list(folds)
M = pd.DataFrame(index=names, columns=names, dtype=float)
rows = []
for a, b in itertools.combinations(names, 2):
    A, B = folds[a], folds[b]
    shared = [k for k in A.index if k in B.index
              and k.split('|')[0] not in (a, b)          # trained in BOTH folds
              and A.loc[k, 'cluster'] >= 0 and B.loc[k, 'cluster'] >= 0]
    if len(shared) < 10:
        continue
    r = adjusted_rand_score(A.loc[shared, 'cluster'], B.loc[shared, 'cluster'])
    M.loc[a, b] = M.loc[b, a] = round(r, 3)
    rows.append(dict(a=a, b=b, n_labels=len(shared), ari=round(r, 4)))
np.fill_diagonal(M.values, 1.0)
print("pairwise ARI between fold partitions (commonly-trained labels only):")
print(M.to_string())
d = pd.DataFrame(rows)
print(f"\n  mean ARI = {d.ari.mean():.4f}   min = {d.ari.min():.4f}   max = {d.ari.max():.4f}")

# ---- per label-pair merge consistency across all 7 folds
allk = set.intersection(*[set(m.index) for m in folds.values()])
allk = sorted(allk)
votes = {}
for h, m in folds.items():
    held = h
    for x, y in itertools.combinations(allk, 2):
        if x.split('|')[0] in (held,) or y.split('|')[0] in (held,):
            continue                                       # skip the held-out cohort's labels
        cx, cy = m.loc[x, 'cluster'], m.loc[y, 'cluster']
        if cx < 0 or cy < 0:
            continue
        votes.setdefault((x, y), []).append(int(cx == cy))

v = {k: np.array(t) for k, t in votes.items() if len(t) >= 5}
cons = np.array([max(t.mean(), 1 - t.mean()) for t in v.values()])
always = np.array([t.mean() == 1 or t.mean() == 0 for t in v.values()])
merged_rate = np.array([t.mean() for t in v.values()])

print(f"\nMERGE CONSISTENCY across folds — {len(v)} label pairs seen in >=5 folds")
print(f"  unanimous (all folds agree)        : {always.mean():.1%}")
print(f"  mean consistency                   : {cons.mean():.3f}")
print(f"  pairs merged by SOME folds not all : {((merged_rate > 0) & (merged_rate < 1)).mean():.1%}")
for lo, hi in [(0, .01), (.01, .5), (.5, .99), (.99, 1.01)]:
    n = ((merged_rate >= lo) & (merged_rate < hi)).sum()
    print(f"    merged in {lo:.0%}-{hi:.0%} of folds: {n:>5} pairs")

pd.DataFrame(rows).to_csv(os.path.join(HERE, 'results', 'fold_stability.csv'), index=False)

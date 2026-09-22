"""FEW-SHOT CURVE - how many labelled cells in a NEW cohort buy how much macro-F1.

Honest setting: the embedding comes from the fold where this cohort was HELD OUT, so the
encoder never saw it. N cells per native type are drawn from the cohort's TRAIN patients
only; scoring is on its TEST patients. Zero-shot LOCO baseline for the same folds = 0.3151.
"""
import os
import sys

sys.argv = [sys.argv[0]]
HERE = os.path.dirname(os.path.abspath(__file__))
P2 = r"D:\Desktop\FYDP\FYDP final works\cell type annotation\pipeline2"
os.chdir(P2)
sys.path.insert(0, P2)

import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')

from config import WORK, SPECS
import s2_tokens as s2
import s3_encoder as s3
import s6_train as s6

CKPT = os.path.join(WORK, 'ckpt')
SEED = 20260810
SHOTS = [1, 2, 5, 10, 20, 50, 100, 250]
NSEED = 5

cohorts = s3.available()
triples, tri2idx, per, genes, ncoh = s2.read_panel(cohorts)
V = len(triples)
train = [c for c in cohorts if SPECS[c]['role'] == 'train']
excl = s6.excluded_pairs()

rows = []
for c in train:
    r = torch.load(os.path.join(CKPT, f's6_proto2_fold_{c}.pt'),
                   weights_only=False, map_location='cpu')
    Lh, _, _ = s6.space_for(c, 'fold')
    m = s6.Stage6(V, Lh['n'], head='proto')
    m.load_state_dict(r['state'])
    m.eval()

    d = s6.load_cohort(c, per[c], tri2idx, V, Lh, excl)
    if d is None:
        continue

    def enc(part):
        U, Z = d['U'][part], []
        with torch.no_grad():
            for i in range(0, len(U), 1024):
                z, _, _ = m(U[i:i + 1024], d['idx'], d['present'])
                Z.append(z.cpu())
        return torch.cat(Z).numpy()

    Xtr, Xte = enc('train'), enc('test')
    v = pd.read_parquet(s2.full_table(c), columns=['cell_id', 'native_label'])
    look = v.set_index('cell_id').native_label
    ytr = look.reindex(d['cid']['train']).to_numpy()
    yte = look.reindex(d['cid']['test']).to_numpy()
    classes = np.unique(ytr)
    idx_by_class = {k: np.flatnonzero(ytr == k) for k in classes}

    line = [f"  {c:10} k={len(classes):>2}"]
    for n in SHOTS:
        got = []
        for s in range(NSEED):
            rng = np.random.default_rng(SEED + s)
            pick = np.concatenate([rng.choice(v_, min(n, len(v_)), replace=False)
                                   for v_ in idx_by_class.values()])
            if len(np.unique(ytr[pick])) < 2:
                continue
            lr = LogisticRegression(max_iter=3000)
            lr.fit(Xtr[pick], ytr[pick])
            keep = np.isin(yte, np.unique(ytr[pick]))
            got.append(f1_score(yte[keep], lr.predict(Xte[keep]), average='macro'))
        if got:
            rows.append(dict(cohort=c, k=len(classes), shots=n,
                             f1=round(float(np.mean(got)), 4),
                             sd=round(float(np.std(got)), 4), cells=n * len(classes)))
            line.append(f"n={n}:{np.mean(got):.3f}")
    print("  ".join(line), flush=True)

d = pd.DataFrame(rows)
d.to_csv(os.path.join(HERE, 'fewshot.csv'), index=False)
piv = d.pivot_table(index='shots', columns='cohort', values='f1')
piv['MEAN'] = piv.mean(axis=1)
print("\nmacro-F1 by shots per type (embedding from the fold that HELD THIS COHORT OUT)")
print(piv.round(4).to_string())
print("\nzero-shot LOCO pipeline baseline = 0.3151")

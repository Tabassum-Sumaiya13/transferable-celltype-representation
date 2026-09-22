"""WHAT DOES STAGE 1's ECDF HARMONISATION COST?

Within a single cohort there is no cross-cohort scale problem for ECDF to solve, so any gap
between the two feature sets on the SAME cells, splits and targets is the price of harmonising.

  raw::    the original vendor intensity
  u_coh::  Stage 1's mid-rank ECDF inside (cohort, marker)

Logistic regression -> that cohort's own native labels, patient-split. Also log1p(raw) and
z-scored raw, because a bare intensity is heavy-tailed and LR would be unfairly punished.
"""
import os
import sys
import warnings

sys.argv = [sys.argv[0]]
HERE = os.path.dirname(os.path.abspath(__file__))
P2 = r"D:\Desktop\FYDP\FYDP final works\cell type annotation\pipeline2"
os.chdir(P2)
sys.path.insert(0, P2)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from config import SPECS
import s2_tokens as s2
import s3_encoder as s3

SEED, CAP = 20260810, 20000
rng = np.random.default_rng(SEED)

cohorts = s3.available()
triples, tri2idx, per, genes, ncoh = s2.read_panel(cohorts)
train = [c for c in cohorts if SPECS[c]['role'] == 'train']


def fit(Xtr, ytr, Xte, yte):
    if len(Xtr) > CAP:
        s = rng.choice(len(Xtr), CAP, replace=False)
        Xtr, ytr = Xtr[s], ytr[s]
    lr = LogisticRegression(max_iter=2000, class_weight='balanced')
    lr.fit(Xtr, ytr)
    return float(f1_score(yte, lr.predict(Xte), average='macro'))


rows = []
for c in train:
    tl = list(per[c])
    cols = (['cell_id', 'image_id', 'native_label']
            + [f'u_coh::{t}' for t in tl] + [f'raw::{t}' for t in tl])
    v = pd.read_parquet(s2.full_table(c), columns=cols)
    v = v[v.native_label.notna()].reset_index(drop=True)

    sm = s2.split_masks(c, v.image_id.to_numpy())
    tr, te = sm['train'], sm['test']
    if tr.sum() == 0 or te.sum() == 0:
        continue
    y = v.native_label.to_numpy()
    keep = np.isin(y, np.unique(y[tr]))
    te = te & keep

    U = v[[f'u_coh::{t}' for t in tl]].to_numpy('float32')
    R = v[[f'raw::{t}' for t in tl]].to_numpy('float32')
    Rl = np.log1p(np.clip(R, 0, None))
    mu, sd = Rl[tr].mean(0, keepdims=True), Rl[tr].std(0, keepdims=True).clip(1e-6)
    Rz = (Rl - mu) / sd

    f_u = fit(U[tr], y[tr], U[te], y[te])
    f_r = fit(Rz[tr], y[tr], Rz[te], y[te])
    both = np.hstack([U, Rz])
    f_b = fit(both[tr], y[tr], both[te], y[te])

    rows.append(dict(cohort=c, k=len(np.unique(y[tr])), markers=len(tl),
                     n_tr=int(tr.sum()), n_te=int(te.sum()),
                     u_coh=round(f_u, 4), raw_log_z=round(f_r, 4), both=round(f_b, 4),
                     delta=round(f_r - f_u, 4)))
    print(f"  {c:10} k={rows[-1]['k']:>2}  u_coh={f_u:.4f}  raw={f_r:.4f}  "
          f"both={f_b:.4f}   raw-u_coh={f_r - f_u:+.4f}", flush=True)

d = pd.DataFrame(rows)
d.to_csv(os.path.join(HERE, 'harmcost.csv'), index=False)
print("\n" + d.to_string(index=False))
print(f"\nMEAN  u_coh={d.u_coh.mean():.4f}  raw={d.raw_log_z.mean():.4f}  "
      f"both={d.both.mean():.4f}  delta={d.delta.mean():+.4f}")

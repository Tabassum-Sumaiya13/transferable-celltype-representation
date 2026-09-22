"""PAIRWISE TRANSFER - turns n=7 into n=42.

Every LOCO verdict in this project is a mean over 7 folds, and the fold-to-fold SD (0.118) is
3x the spread of every method ever tried (0.037). That is why no gate margin is provable.
This does not fix the method question - it changes the QUESTION to one the data can answer:

    what makes a cohort transferable?

For each TARGET cohort B, the encoder from fold B is used - it has never seen B. A decoder is
then fitted on ONE source cohort A at a time, in fold B's own label space, and scored on B's
test patients. 7 targets x 6 sources = 42 ordered pairs, plus two references per target:

  pooled   decoder fitted on all 6 sources at once   -> does pooling beat the best single source?
  loco     the shipped prototype number for that fold -> does a linear decoder on 6 match it?

Covariates recorded per pair so the 42 rows can be regressed: shared markers, same tissue,
same platform, panel sizes, and how much of the target's label space the source even covers.
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
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from config import SPECS
import s2_tokens as s2
import s3_encoder as s3
import s6_train as s6

CKPT = os.path.join(s6.WORK, 'ckpt')
SEED, CAP = 20260810, 15000
rng = np.random.default_rng(SEED)

cohorts = s3.available()
triples, tri2idx, per, genes, ncoh = s2.read_panel(cohorts)
V = len(triples)
train = [c for c in cohorts if SPECS[c]['role'] == 'train']


def meta(c, key, default='?'):
    s = SPECS[c]
    return s.get(key, s.get('tech', default))


def fit_score(Xtr, ytr, Xte, yte, n_class, drop):
    if len(Xtr) > CAP:
        s = rng.choice(len(Xtr), CAP, replace=False)
        Xtr, ytr = Xtr[s], ytr[s]
    if len(np.unique(ytr)) < 2:
        return float('nan'), 0
    lr = LogisticRegression(max_iter=2000, class_weight='balanced')
    lr.fit(Xtr, ytr)
    f, _ = s3.macro_f1(yte, lr.predict(Xte), n_class, drop=drop)
    return float(f), len(np.unique(ytr))


rows = []
for B in train:                                   # TARGET - the encoder never saw it
    r = torch.load(os.path.join(CKPT, f's6_proto2_fold_{B}.pt'),
                   weights_only=False, map_location='cpu')
    Lh, _, _ = s6.space_for(B, 'fold')
    m = s6.Stage6(V, Lh['n'], head='proto')
    m.load_state_dict(r['state'])
    m.eval()
    excl = s6.excluded_pairs()

    def enc(d, part):
        U, Z = d['U'][part], []
        with torch.no_grad():
            for i in range(0, len(U), 1024):
                z, _, _ = m(U[i:i + 1024], d['idx'], d['present'])
                Z.append(z.cpu())
        return torch.cat(Z).numpy()

    dB = s6.load_cohort(B, per[B], tri2idx, V, Lh, excl)
    if dB is None:
        continue
    XB, yB = enc(dB, 'test'), dB['y']['test'].cpu().numpy()

    src = {}
    for A in train:
        if A == B:
            continue
        dA = s6.load_cohort(A, per[A], tri2idx, V, Lh, excl)
        if dA is None:
            continue
        src[A] = (enc(dA, 'train'), dA['y']['train'].cpu().numpy())

    # reference: all six sources pooled
    Xp = np.vstack([v[0] for v in src.values()])
    yp = np.concatenate([v[1] for v in src.values()])
    f_pool, k_pool = fit_score(Xp, yp, XB, yB, Lh['n'], Lh['unreliable'])

    for A, (XA, yA) in src.items():
        f, kA = fit_score(XA, yA, XB, yB, Lh['n'], Lh['unreliable'])
        shared = len(set(per[A]) & set(per[B]))
        cov = len(set(np.unique(yA)) & set(np.unique(yB))) / max(1, len(np.unique(yB)))
        rows.append(dict(
            source=A, target=B, f1=round(f, 4),
            shared_markers=shared,
            jaccard=round(shared / len(set(per[A]) | set(per[B])), 3),
            n_src=len(per[A]), n_tgt=len(per[B]),
            same_tissue=int(meta(A, 'tissue') == meta(B, 'tissue')),
            same_platform=int(meta(A, 'platform') == meta(B, 'platform')),
            src_classes=kA, tgt_classes=len(np.unique(yB)),
            class_cover=round(cov, 3),
            pooled=round(f_pool, 4), loco=round(float(r['f1_core']), 4)))

    best = max(x['f1'] for x in rows if x['target'] == B)
    print(f"  target {B:10} best_single={best:.4f}  pooled6={f_pool:.4f}  "
          f"loco_proto={r['f1_core']:.4f}", flush=True)

d = pd.DataFrame(rows)
d.to_csv(os.path.join(HERE, 'results', 'pairwise.csv'), index=False)

print(f"\n{len(d)} ordered pairs\n")
print(d.pivot_table(index='source', columns='target', values='f1').round(3).to_string())
print(f"\nMEAN single-source f1 = {d.f1.mean():.4f}   "
      f"pooled = {d.groupby('target').pooled.first().mean():.4f}   "
      f"loco = {d.groupby('target').loco.first().mean():.4f}")
print("\n  correlations over the 42 pairs:")
for c in ['shared_markers', 'jaccard', 'same_tissue', 'same_platform',
          'n_src', 'n_tgt', 'class_cover', 'src_classes']:
    print(f"    corr(f1, {c:15}) = {d.f1.corr(d[c]):+.3f}")

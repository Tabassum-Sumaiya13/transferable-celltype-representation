"""CoRA CLAIM 1 - does source disagreement predict per-class failure, WITHOUT labels?

For each target cohort B (encoder from fold B, which never saw B):
  * fit ONE decoder per source cohort A (6 of them) on A's train cells, fold-B label space
  * every source votes on each of B's test cells
  * coverage(cell) = fraction of the 6 votes agreeing with the modal vote   <- no labels used
  * confidence(cell) = max softmax of the SHIPPED prototype model           <- the standard signal

Aggregate both per TRUE class, then correlate each against that class's actual F1.

  claim 1   coverage correlates with per-class F1
  the novelty claim   coverage correlates BETTER than confidence

If confidence wins, CoRA is a relabelled uncertainty-sampling baseline and should be dropped.
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
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

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

rows = []
for B in train:
    r = torch.load(os.path.join(CKPT, f's6_proto2_fold_{B}.pt'),
                   weights_only=False, map_location='cpu')
    Lh, _, _ = s6.space_for(B, 'fold')
    m = s6.Stage6(V, Lh['n'], head='proto')
    m.load_state_dict(r['state'])
    m.eval()
    excl = s6.excluded_pairs()

    def enc(d, part):
        U, Z, LG = d['U'][part], [], []
        with torch.no_grad():
            for i in range(0, len(U), 1024):
                z, lg, _ = m(U[i:i + 1024], d['idx'], d['present'])
                Z.append(z.cpu())
                LG.append(lg.cpu())
        return torch.cat(Z).numpy(), torch.cat(LG)

    dB = s6.load_cohort(B, per[B], tri2idx, V, Lh, excl)
    if dB is None:
        continue
    XB, lgB = enc(dB, 'test')
    yB = dB['y']['test'].cpu().numpy()

    # the SHIPPED model's own confidence, the signal CoRA claims is the wrong one
    conf = F.softmax(lgB, dim=1).max(1).values.numpy()

    # six independent source decoders vote
    votes = []
    for A in train:
        if A == B:
            continue
        dA = s6.load_cohort(A, per[A], tri2idx, V, Lh, excl)
        if dA is None:
            continue
        XA, _ = enc(dA, 'train')
        yA = dA['y']['train'].cpu().numpy()
        if len(XA) > CAP:
            s = rng.choice(len(XA), CAP, replace=False)
            XA, yA = XA[s], yA[s]
        if len(np.unique(yA)) < 2:
            continue
        lr = LogisticRegression(max_iter=2000, class_weight='balanced')
        lr.fit(XA, yA)
        votes.append(lr.predict(XB))
    Vt = np.vstack(votes)                                   # (6, n_cells)

    # coverage = fraction of sources agreeing with the modal vote, per cell
    cover = np.array([np.bincount(col).max() / len(col) for col in Vt.T])

    pc = r['per_cls'].set_index('cluster')
    for k in np.unique(yB):
        if k in Lh['unreliable'] or k not in pc.index:
            continue
        msk = yB == k
        if msk.sum() < 20:
            continue
        rows.append(dict(target=B, cluster=int(k), n=int(msk.sum()),
                         f1=float(pc.loc[k, 'f1']),
                         coverage=round(float(cover[msk].mean()), 4),
                         confidence=round(float(conf[msk].mean()), 4)))
    print(f"  {B:10} {len(np.unique(yB))} classes, mean coverage={cover.mean():.3f}", flush=True)

d = pd.DataFrame(rows)
d.to_csv(os.path.join(HERE, 'results', 'coverage_claim1.csv'), index=False)

print(f"\n{len(d)} scored classes across {d.target.nunique()} targets\n")
print(d.sort_values('f1').head(12).to_string(index=False))


def rep(x, y, lab):
    r_ = np.corrcoef(x, y)[0, 1]
    n = len(x)
    t = r_ * np.sqrt(n - 2) / np.sqrt(1 - r_ ** 2)
    print(f"  corr(F1, {lab:11}) = {r_:+.3f}   t={t:+.2f}  n={n}")
    return r_


print("\nCLAIM 1 — does each signal predict per-class F1?")
rc = rep(d.coverage, d.f1, 'coverage')
rf = rep(d.confidence, d.f1, 'confidence')
print(f"\n  coverage - confidence = {rc - rf:+.3f}  "
      f"({'COVERAGE WINS' if rc > rf else 'CONFIDENCE WINS - CoRA is dead'})")

z = d[d.f1 < 0.01]
nz = d[d.f1 >= 0.01]
print(f"\n  zero-F1 classes (n={len(z)}):     coverage={z.coverage.mean():.3f}  "
      f"confidence={z.confidence.mean():.3f}")
print(f"  non-zero classes (n={len(nz)}):   coverage={nz.coverage.mean():.3f}  "
      f"confidence={nz.confidence.mean():.3f}")

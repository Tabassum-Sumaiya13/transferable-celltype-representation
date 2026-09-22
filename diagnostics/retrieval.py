"""RETRIEVAL vs PROTOTYPE, zero-shot - splits the 0.255 decoding loss.

Same embedding, same cells, same fold-local label space, same metric, same unreliable-cluster
drop. ONLY the decoder changes. No training happens: the shipped s6_proto2_fold_*.pt checkpoints
are loaded and run forward, exactly as Stage 7b does.

  proto     the shipped decoder - one learned prototype per harmonised cluster  (= f1_core)
  knn       plain k-nearest-neighbour vote over the TRAINING cohorts' cells
  knn_bal   per CLASS, the mean cosine distance to that class's k nearest members, argmin.
            Balanced by construction like prototypes, but keeps the class MULTI-MODAL -
            this is the direct test of "one prototype per class destroys sub-structure".

If knn_bal beats proto, the loss is prototype compression and it is a fixable design choice.
If neither beats proto, the 0.255 is the price of having no target labels, and no decoder
redesign recovers it.
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

from config import SPECS
import s2_tokens as s2
import s3_encoder as s3
import s6_train as s6

CKPT = os.path.join(s6.WORK, 'ckpt')
KS = [1, 5, 25, 100]

cohorts = s3.available()
triples, tri2idx, per, genes, ncoh = s2.read_panel(cohorts)
V = len(triples)
train = [c for c in cohorts if SPECS[c]['role'] == 'train']
excl = s6.excluded_pairs()


def encode(m, d, part):
    U, Z = d['U'][part], []
    with torch.no_grad():
        for i in range(0, len(U), 1024):
            z, _, _ = m(U[i:i + 1024], d['idx'], d['present'])
            Z.append(z.cpu())
    return torch.cat(Z)


rows = []
for held in train:
    p = os.path.join(CKPT, f's6_proto2_fold_{held}.pt')
    r = torch.load(p, weights_only=False, map_location='cpu')
    Lh, _, _ = s6.space_for(held, 'fold')
    m = s6.Stage6(V, Lh['n'], head='proto')
    m.load_state_dict(r['state'])
    m.eval()

    rest = [c for c in train if c != held]
    data = {c: s6.load_cohort(c, per[c], tri2idx, V, Lh, excl) for c in train}
    data = {c: d for c, d in data.items() if d is not None}
    rest = [c for c in rest if c in data]

    # retrieval bank: every TRAINING cohort's train-split cells, in the fold's label space
    B = torch.cat([encode(m, data[c], 'train') for c in rest])
    by = torch.cat([data[c]['y']['train'].cpu() for c in rest]).numpy()
    Q = encode(m, data[held], 'test')
    yt = data[held]['y']['test'].cpu().numpy()

    Bn, Qn = F.normalize(B, dim=1), F.normalize(Q, dim=1)
    classes = np.unique(by)
    cls_idx = {int(k): np.flatnonzero(by == k) for k in classes}

    # cosine similarity in blocks
    S = torch.empty(len(Qn), len(Bn), dtype=torch.float32)
    for i in range(0, len(Qn), 512):
        S[i:i + 512] = Qn[i:i + 512] @ Bn.T

    out = dict(held=held, n_test=len(yt), n_bank=len(by), n_class=Lh['n'],
               proto=round(float(r['f1_core']), 4))

    for k in KS:
        # --- plain kNN vote
        nb = S.topk(k, dim=1).indices.numpy()
        vote = np.array([np.bincount(by[row], minlength=Lh['n']).argmax() for row in nb])
        f_knn, _ = s3.macro_f1(yt, vote, Lh['n'], drop=Lh['unreliable'])

        # --- class-balanced: mean similarity to each class's own k nearest members
        sc = torch.empty(len(Qn), len(classes))
        for j, c_ in enumerate(classes):
            col = S[:, cls_idx[int(c_)]]
            kk = min(k, col.shape[1])
            sc[:, j] = col.topk(kk, dim=1).values.mean(1)
        pred = classes[sc.argmax(1).numpy()]
        f_bal, _ = s3.macro_f1(yt, pred, Lh['n'], drop=Lh['unreliable'])

        out[f'knn{k}'] = round(float(f_knn), 4)
        out[f'bal{k}'] = round(float(f_bal), 4)

    rows.append(out)
    best = max(out[f'bal{k}'] for k in KS)
    print(f"  {held:10} proto={out['proto']:.4f}  "
          + "  ".join(f"bal{k}={out[f'bal{k}']:.4f}" for k in KS)
          + f"   best_bal-proto={best - out['proto']:+.4f}", flush=True)

d = pd.DataFrame(rows)
d.to_csv(os.path.join(HERE, 'results', 'retrieval.csv'), index=False)
print("\n" + d.to_string(index=False))
print("\nMEAN")
for c in d.columns:
    if c in ('held', 'n_test', 'n_bank', 'n_class'):
        continue
    print(f"  {c:8} {d[c].mean():.4f}   (vs proto {d.proto.mean():.4f}, "
          f"delta {d[c].mean() - d.proto.mean():+.4f})")

"""WHAT IS THE REAL WITHIN-COHORT CEILING? Logistic regression said 0.6277 - but that is a
LOWER bound, because a linear decoder is weak. This runs the VENDORED MAPS MLP (Shaban et al.
2024) on exactly the same cells, splits and targets.

  maps_raw   MAPS MLP on the cohort's own marker values -> its own native labels
  maps_emb   MAPS MLP on z_cell from the fold that HELD THIS COHORT OUT
  lr_raw     logistic regression, the number already measured (0.6277 mean)

Within-cohort, patient-split. No transfer. This is the easiest possible task, so it bounds
everything the project can ever score.
"""
import os
import sys

sys.argv = [sys.argv[0]]
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"D:\Desktop\FYDP\FYDP final works\cell type annotation"
P2 = os.path.join(ROOT, 'pipeline2')
os.chdir(P2)
sys.path.insert(0, P2)
sys.path.insert(0, os.path.join(ROOT, 'dropped_past_works', 'MAPS'))   # vendored MAPS

import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')

from config import WORK, SPECS
import s2_tokens as s2
import s3_encoder as s3
import s6_train as s6

from maps.cell_phenotyping.networks import MLP          # raises if the vendored source moved

CKPT = os.path.join(WORK, 'ckpt')
SEED = 20260810
HIDDEN, DROPOUT, LR, BATCH = 512, 0.10, 1e-3, 128        # MAPS published defaults (s10)
MAX_EPOCHS, PATIENCE = 60, 8
DEV = torch.device('cpu')


def logits_of(o):
    return o[0] if isinstance(o, (tuple, list)) else o


def fit_maps(Xtr, ytr, Xva, yva, Xte, yte, k):
    """MAPS MLP, class-balanced batches, early-stopped on validation macro-F1."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    m = MLP(input_dim=Xtr.shape[1], hidden_dim=HIDDEN, num_classes=k, dropout=DROPOUT).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True).clip(1e-6)
    A, B, C = [torch.from_numpy(((X - mu) / sd).astype('float32')) for X in (Xtr, Xva, Xte)]
    T = torch.from_numpy(ytr).long()
    pool = {c: np.flatnonzero(ytr == c) for c in np.unique(ytr)}
    per_k = max(1, len(ytr) // max(1, len(pool)))

    best, state, bad = -np.inf, None, 0
    for ep in range(MAX_EPOCHS):
        m.train()
        take = np.concatenate([rng.choice(v, per_k, replace=len(v) < per_k)
                               for v in pool.values()])
        take = take[rng.permutation(len(take))]
        for i in range(0, len(take), BATCH):
            b = torch.from_numpy(take[i:i + BATCH]).long()
            loss = lossf(logits_of(m(A[b])), T[b])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pv = logits_of(m(B)).argmax(1).numpy()
        f = f1_score(yva, pv, average='macro')
        if f > best:
            best, state, bad = f, {k_: v.clone() for k_, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    m.load_state_dict(state)
    m.eval()
    with torch.no_grad():
        return float(f1_score(yte, logits_of(m(C)).argmax(1).numpy(), average='macro'))


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
    mdl = s6.Stage6(V, Lh['n'], head='proto')
    mdl.load_state_dict(r['state'])
    mdl.eval()
    d = s6.load_cohort(c, per[c], tri2idx, V, Lh, excl)
    if d is None:
        continue

    def enc(part):
        U, Z = d['U'][part], []
        with torch.no_grad():
            for i in range(0, len(U), 1024):
                z, _, _ = mdl(U[i:i + 1024], d['idx'], d['present'])
                Z.append(z.cpu())
        return torch.cat(Z).numpy()

    v = pd.read_parquet(s2.full_table(c), columns=['cell_id', 'native_label'])
    look = v.set_index('cell_id').native_label
    lab = {p: look.reindex(d['cid'][p]).to_numpy() for p in ('train', 'val', 'test')}
    cls = np.unique(lab['train'])
    code = {s: i for i, s in enumerate(cls)}
    keep = {p: np.isin(lab[p], cls) for p in lab}
    y = {p: np.array([code[s] for s in lab[p][keep[p]]]) for p in lab}

    raw = {p: d['U'][p].cpu().numpy()[keep[p]] for p in lab}
    emb = {p: e[keep[p]] for p, e in ((q, enc(q)) for q in lab)}

    f_maps_raw = fit_maps(raw['train'], y['train'], raw['val'], y['val'],
                          raw['test'], y['test'], len(cls))
    f_maps_emb = fit_maps(emb['train'], y['train'], emb['val'], y['val'],
                          emb['test'], y['test'], len(cls))
    lr = LogisticRegression(max_iter=2000, class_weight='balanced')
    lr.fit(raw['train'][:20000], y['train'][:20000])
    f_lr = float(f1_score(y['test'], lr.predict(raw['test']), average='macro'))

    rows.append(dict(cohort=c, k=len(cls), maps_raw=round(f_maps_raw, 4),
                     maps_emb=round(f_maps_emb, 4), lr_raw=round(f_lr, 4)))
    print(f"  {c:10} k={len(cls):>2}  MAPS_raw={f_maps_raw:.4f}  "
          f"MAPS_emb={f_maps_emb:.4f}  LR_raw={f_lr:.4f}", flush=True)

d = pd.DataFrame(rows)
d.to_csv(os.path.join(HERE, 'ceiling.csv'), index=False)
print("\n" + d.to_string(index=False))
print(f"\nMEAN  MAPS_raw={d.maps_raw.mean():.4f}  MAPS_emb={d.maps_emb.mean():.4f}  "
      f"LR_raw={d.lr_raw.mean():.4f}")
print("zero-shot LOCO pipeline = 0.3151")

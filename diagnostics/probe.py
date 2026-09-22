"""LINEAR PROBE - how much native-label information exists, and how much survives encoding.

Within-cohort, patient-split (s2.split_masks), no transfer, no shared label space.
Logistic regression -> that cohort's OWN native labels. Macro-F1 on the held-out patients.

  raw        probe on the harmonised marker values              info in the DATA
  emb_seen   probe on z from a fold that TRAINED on this cohort info that SURVIVED encoding
  emb_held   probe on z from the fold that HELD OUT this cohort same, out of distribution

raw high + emb low  -> the encoder discards cell-type signal   (objective problem, fixable)
raw ~ emb, both low -> markers do not determine these labels   (information/label problem)
"""
import os
import sys

sys.argv = [sys.argv[0]]
HERE = os.path.dirname(os.path.abspath(__file__))
P2 = r"D:\Desktop\FYDP\FYDP final works\cell type annotation\pipeline2"
os.chdir(P2)
sys.path.insert(0, P2)

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from config import WORK, SPECS
import s2_tokens as s2
import s3_encoder as s3
import s6_train as s6

CKPT = os.path.join(WORK, 'ckpt')
SEED, CAP = 20260810, 20000

cohorts = s3.available()
triples, tri2idx, per, genes, ncoh = s2.read_panel(cohorts)
V = len(triples)
train = [c for c in cohorts if SPECS[c]['role'] == 'train']
excl = s6.excluded_pairs()
rng = np.random.default_rng(SEED)


def load_model(fold_held):
    p = os.path.join(CKPT, f's6_proto2_fold_{fold_held}.pt')
    r = torch.load(p, weights_only=False, map_location='cpu')
    Lh, _, _ = s6.space_for(fold_held, 'fold')
    m = s6.Stage6(V, Lh['n'], head='proto')
    m.load_state_dict(r['state'])
    m.eval()
    return m, Lh


def encode(m, d, part):
    U, Z = d['U'][part], []
    with torch.no_grad():
        for i in range(0, len(U), 1024):
            z, _, _ = m(U[i:i + 1024], d['idx'], d['present'])
            Z.append(z.cpu())
    return torch.cat(Z).numpy()


def probe(Xtr, ytr, Xte, yte):
    """macro-F1 of a balanced multinomial logistic regression, scored on unseen patients."""
    if len(Xtr) > CAP:
        s = rng.choice(len(Xtr), CAP, replace=False)
        Xtr, ytr = Xtr[s], ytr[s]
    keep = np.isin(yte, np.unique(ytr))
    if keep.sum() == 0:
        return float('nan')
    lr = LogisticRegression(max_iter=2000, class_weight='balanced', n_jobs=-1)
    lr.fit(Xtr, ytr)
    return float(f1_score(yte[keep], lr.predict(Xte[keep]), average='macro'))


rows = []
for c in train:
    other = [h for h in train if h != c][0]        # a fold that TRAINED on c
    m_seen, L_seen = load_model(other)
    m_held, L_held = load_model(c)                 # the fold that HELD OUT c

    d = s6.load_cohort(c, per[c], tri2idx, V, L_seen, excl)
    if d is None:
        continue
    v = pd.read_parquet(s2.full_table(c), columns=['cell_id', 'native_label'])
    look = v.set_index('cell_id').native_label
    ytr = look.reindex(d['cid']['train']).to_numpy()
    yte = look.reindex(d['cid']['test']).to_numpy()

    raw_tr, raw_te = d['U']['train'].cpu().numpy(), d['U']['test'].cpu().numpy()
    f_raw = probe(raw_tr, ytr, raw_te, yte)
    f_seen = probe(encode(m_seen, d, 'train'), ytr, encode(m_seen, d, 'test'), yte)

    d2 = s6.load_cohort(c, per[c], tri2idx, V, L_held, excl)
    y2tr = look.reindex(d2['cid']['train']).to_numpy()
    y2te = look.reindex(d2['cid']['test']).to_numpy()
    f_held = probe(encode(m_held, d2, 'train'), y2tr, encode(m_held, d2, 'test'), y2te)

    rows.append(dict(cohort=c, n_native=len(pd.unique(ytr)), n_train=len(ytr), n_test=len(yte),
                     raw=round(f_raw, 4), emb_seen=round(f_seen, 4), emb_held=round(f_held, 4),
                     markers=raw_tr.shape[1]))
    print(f"  {c:10} markers={raw_tr.shape[1]:>3} k={rows[-1]['n_native']:>2}  "
          f"raw={f_raw:.4f}  emb_seen={f_seen:.4f}  emb_held={f_held:.4f}", flush=True)

d = pd.DataFrame(rows)
d.to_csv(os.path.join(HERE, 'probe.csv'), index=False)
print("\n" + d.to_string(index=False))
print(f"\nMEAN  raw={d.raw.mean():.4f}  emb_seen={d.emb_seen.mean():.4f}  "
      f"emb_held={d.emb_held.mean():.4f}")

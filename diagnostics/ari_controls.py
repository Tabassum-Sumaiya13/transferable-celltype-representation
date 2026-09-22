"""Two controls that make ARI_embed = 0.19 interpretable.

  RAW    k-means on the harmonised marker values, NO MODEL AT ALL, vs native labels.
         If raw beats the embedding, the model is destroying structure.

  SEEN   the SAME fold checkpoint encoding a cohort it was TRAINED on, vs that cohort's
         native labels. This is the in-distribution ceiling for this architecture.
         held-out ARI vs seen ARI = the transfer gap.
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
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from config import WORK, SPECS
import s2_tokens as s2
import s3_encoder as s3
import s6_train as s6

CKPT = os.path.join(WORK, 'ckpt')
SEED = 20260810

cohorts = s3.available()
triples, tri2idx, per, genes, ncoh = s2.read_panel(cohorts)
V = len(triples)
train = [c for c in cohorts if SPECS[c]['role'] == 'train']
excl = s6.excluded_pairs()


def km_ari(X, native, k):
    p = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X)
    return float(adjusted_rand_score(native, p))


def native_for(cid, cohort):
    v = pd.read_parquet(s2.full_table(cohort), columns=['cell_id', 'native_label'])
    return v.set_index('cell_id').native_label.reindex(cid).to_numpy()


rows = []
for held in train:
    p = os.path.join(CKPT, f's6_proto2_fold_{held}.pt')
    if not os.path.exists(p):
        continue
    r = torch.load(p, weights_only=False, map_location='cpu')
    Lh, pt, sp = s6.space_for(held, 'fold')
    m = s6.Stage6(V, Lh['n'], head='proto')
    m.load_state_dict(r['state'])
    m.eval()

    def encode(d):
        U, Z = d['U']['test'], []
        with torch.no_grad():
            for i in range(0, len(U), 1024):
                z, _, _ = m(U[i:i + 1024], d['idx'], d['present'])
                Z.append(z.cpu())
        return torch.cat(Z).numpy()

    # ---- held-out cohort: raw markers vs embedding
    d = s6.load_cohort(held, per[held], tri2idx, V, Lh, excl)
    if d is None:
        continue
    nat = native_for(d['cid']['test'], held)
    k = len(pd.unique(nat))
    raw = d['U']['test'].cpu().numpy()
    ari_raw = km_ari(raw, nat, k)
    ari_emb = km_ari(encode(d), nat, k)

    # ---- a cohort this same model WAS trained on (in-distribution ceiling)
    seen = [c for c in train if c != held][0]
    ds = s6.load_cohort(seen, per[seen], tri2idx, V, Lh, excl)
    if ds is not None:
        nats = native_for(ds['cid']['test'], seen)
        ks = len(pd.unique(nats))
        ari_seen_raw = km_ari(ds['U']['test'].cpu().numpy(), nats, ks)
        ari_seen_emb = km_ari(encode(ds), nats, ks)
    else:
        ari_seen_raw = ari_seen_emb = float('nan')

    rows.append(dict(held=held, seen=seen,
                     raw_held=round(ari_raw, 4), emb_held=round(ari_emb, 4),
                     raw_seen=round(ari_seen_raw, 4), emb_seen=round(ari_seen_emb, 4)))
    print(f"  {held:10} held: raw={ari_raw:.4f} emb={ari_emb:.4f}   "
          f"seen({seen}): raw={ari_seen_raw:.4f} emb={ari_seen_emb:.4f}", flush=True)

d = pd.DataFrame(rows)
d.to_csv(os.path.join(HERE, 'ari_controls.csv'), index=False)
print("\n" + d.to_string(index=False))
print(f"\nMEAN  raw_held={d.raw_held.mean():.4f}  emb_held={d.emb_held.mean():.4f}  "
      f"raw_seen={d.raw_seen.mean():.4f}  emb_seen={d.emb_seen.mean():.4f}")

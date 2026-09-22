"""Does the representation transfer, independent of the shared label space?

Two numbers per LOCO fold, from the CACHED shipped checkpoints (s6_proto2_fold_*.pt):

  macro-F1   the headline metric - needs the shared label space
  ARI_pred   ARI(predicted harmonized label, the cohort's OWN native label)
  ARI_embed  ARI(kmeans on z_cell, the cohort's OWN native label)

ARI needs no shared vocabulary. If ARI_embed is decent while macro-F1 is 0.31, the
representation transfers and the label space is what destroys the score.
"""
import os
import sys

sys.argv = [sys.argv[0]]                      # stages parse sys.argv directly
HERE = os.path.dirname(os.path.abspath(__file__))
P2 = r"D:\Desktop\FYDP\FYDP final works\cell type annotation\pipeline2"
os.chdir(P2)
sys.path.insert(0, P2)

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from config import WORK, SPECS
import s2_tokens as s2
import s3_encoder as s3
import s6_train as s6

CKPT = os.path.join(WORK, 'ckpt')

cohorts = s3.available()
triples, tri2idx, per, genes, ncoh = s2.read_panel(cohorts)
V = len(triples)
train = [c for c in cohorts if SPECS[c]['role'] == 'train']
excl = s6.excluded_pairs()
print(f"vocab {V} triples | train roster {train}", flush=True)

rows = []
for held in train:
    p = os.path.join(CKPT, f's6_proto2_fold_{held}.pt')
    if not os.path.exists(p):
        print(f"  {held:10} NO CHECKPOINT", flush=True)
        continue
    r = torch.load(p, weights_only=False, map_location='cpu')

    Lh, pt, sp = s6.space_for(held, 'fold')
    data = s6.load_cohort(held, per[held], tri2idx, V, Lh, excl)
    if data is None:
        print(f"  {held:10} no mappable cells", flush=True)
        continue

    m = s6.Stage6(V, Lh['n'], head='proto')
    m.load_state_dict(r['state'])
    m.eval()

    U = data['U']['test']
    Z, LG = [], []
    with torch.no_grad():
        for i in range(0, len(U), 1024):
            z, lg, _ = m(U[i:i + 1024], data['idx'], data['present'])
            Z.append(z.cpu())
            LG.append(lg.argmax(1).cpu())
    Z = torch.cat(Z).numpy()
    y_pred = torch.cat(LG).numpy()
    y_true = data['y']['test'].cpu().numpy()

    # the cohort's OWN native labels for the same test cells
    cid = data['cid']['test']
    v = pd.read_parquet(s2.full_table(held), columns=['cell_id', 'native_label'])
    native = v.set_index('cell_id').native_label.reindex(cid).to_numpy()

    f1_core, _ = s3.macro_f1(y_true, y_pred, Lh['n'], drop=Lh['unreliable'])
    k = len(pd.unique(native))
    km = KMeans(n_clusters=k, n_init=10, random_state=20260810).fit_predict(Z)

    rows.append(dict(
        held=held, n_cells=len(y_true), n_native=k, n_clusters=Lh['n'],
        f1_core=round(float(f1_core), 4),
        ari_pred=round(float(adjusted_rand_score(native, y_pred)), 4),
        ari_embed=round(float(adjusted_rand_score(native, km)), 4),
        nmi_embed=round(float(normalized_mutual_info_score(native, km)), 4),
        ari_cached_f1=round(float(r['f1_core']), 4)))
    print(f"  {held:10} F1={rows[-1]['f1_core']:.4f} (cached {rows[-1]['ari_cached_f1']:.4f})  "
          f"ARI_pred={rows[-1]['ari_pred']:.4f}  ARI_embed={rows[-1]['ari_embed']:.4f}  "
          f"native_k={k}", flush=True)

d = pd.DataFrame(rows)
out = os.path.join(HERE, 'ari_probe.csv')
d.to_csv(out, index=False)
print("\n" + d.to_string(index=False))
print(f"\nMEAN  f1_core={d.f1_core.mean():.4f}  ari_pred={d.ari_pred.mean():.4f}  "
      f"ari_embed={d.ari_embed.mean():.4f}  nmi_embed={d.nmi_embed.mean():.4f}")
print(f"written {out}")

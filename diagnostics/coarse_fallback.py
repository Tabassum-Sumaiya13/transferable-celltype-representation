"""TASK 2 - how much of the LOCO failure is FINE confusion vs COARSE lineage error?

17 of 73 classes score exactly 0.000. A cell called "CD8 T cell" that was predicted "CD4 T cell"
scores the same zero as one predicted "tumour". Those are not the same mistake, and a biologist
would accept the first and reject the second.

This re-scores the SHIPPED predictions at a coarse lineage level. The taxonomy is NOT Stage 1b's
nesting graph (which missed all three cases declared for it and is untrusted). It is built from
the five textbook mutually-exclusive pairs in panel/gate1b_exclusions.csv, declared before any run:

    epithelial   FAMILY:KRT_PAN         T cell   COMPLEX:CD3
    B cell       HGNC:7315  (MS4A1)     endothelial  HGNC:8823 (PECAM1)
    immune       HGNC:9666  (PTPRC)

Each fold-local cluster is assigned a lineage from its own mean marker ranks. Cells are then
scored on whether the PREDICTED cluster shares a lineage with the TRUE cluster.
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

from config import SPECS
import s2_tokens as s2
import s3_encoder as s3
import s6_train as s6

CKPT = os.path.join(s6.WORK, 'ckpt')

# the five declared lineage markers, exactly as gate1b_exclusions.csv names them
LIN = {'epithelial': 'FAMILY:KRT_PAN|pan|none',
       'T cell':     'COMPLEX:CD3|pan|none',
       'B cell':     'HGNC:7315|pan|none',
       'endothelial': 'HGNC:8823|pan|none',
       'immune':     'HGNC:9666|pan|none'}

cohorts = s3.available()
triples, tri2idx, per, genes, ncoh = s2.read_panel(cohorts)
V = len(triples)
train = [c for c in cohorts if SPECS[c]['role'] == 'train']
excl = s6.excluded_pairs()

rows, per_cls = [], []
for held in train:
    r = torch.load(os.path.join(CKPT, f's6_proto2_fold_{held}.pt'),
                   weights_only=False, map_location='cpu')
    Lh, _, _ = s6.space_for(held, 'fold')
    m = s6.Stage6(V, Lh['n'], head='proto')
    m.load_state_dict(r['state'])
    m.eval()

    d = s6.load_cohort(held, per[held], tri2idx, V, Lh, excl)
    if d is None:
        continue
    U, Z, P = d['U']['test'], [], []
    with torch.no_grad():
        for i in range(0, len(U), 1024):
            z, lg, _ = m(U[i:i + 1024], d['idx'], d['present'])
            Z.append(z.cpu())
            P.append(lg.argmax(1).cpu())
    yp = torch.cat(P).numpy()
    yt = d['y']['test'].cpu().numpy()

    # per-cluster mean marker rank, from the cells themselves, on this cohort's own slots
    slot = {t: s_ for t, s_ in zip(d['triples'], d['slots'])}
    Uc = U.cpu().numpy()
    lin_of = {}
    for k in np.unique(np.concatenate([yt, yp])):
        msk = yt == k
        if msk.sum() < 10:
            continue
        best, bv = None, -np.inf
        for name, tri in LIN.items():
            if tri not in slot:
                continue
            v = Uc[msk, slot[tri]].mean()
            if v > bv:
                best, bv = name, v
        lin_of[int(k)] = best if bv > 0.55 else 'unassigned'   # 0.55 on the ECDF scale

    ok = np.array([lin_of.get(int(a)) is not None and lin_of.get(int(a)) == lin_of.get(int(b))
                   for a, b in zip(yp, yt)])
    fine = np.array([a == b for a, b in zip(yp, yt)])

    pc = r['per_cls'].set_index('cluster')
    for k in np.unique(yt):
        if k in Lh['unreliable'] or k not in pc.index:
            continue
        msk = yt == k
        if msk.sum() < 20:
            continue
        per_cls.append(dict(held=held, cluster=int(k), n=int(msk.sum()),
                            f1_fine=float(pc.loc[k, 'f1']),
                            acc_fine=round(float(fine[msk].mean()), 4),
                            acc_coarse=round(float(ok[msk].mean()), 4),
                            lineage=lin_of.get(int(k))))

    rows.append(dict(held=held, n=len(yt), fine_acc=round(float(fine.mean()), 4),
                     coarse_acc=round(float(ok.mean()), 4),
                     f1_fine=round(float(r['f1_core']), 4),
                     lineages=len({v for v in lin_of.values() if v})))
    print(f"  {held:10} fine_acc={fine.mean():.4f}  coarse_acc={ok.mean():.4f}  "
          f"gain={ok.mean() - fine.mean():+.4f}", flush=True)

d1 = pd.DataFrame(rows)
d2 = pd.DataFrame(per_cls)
d1.to_csv(os.path.join(HERE, 'results', 'coarse_fold.csv'), index=False)
d2.to_csv(os.path.join(HERE, 'results', 'coarse_perclass.csv'), index=False)

print("\n" + d1.to_string(index=False))
print(f"\nMEAN  fine_acc={d1.fine_acc.mean():.4f}  coarse_acc={d1.coarse_acc.mean():.4f}  "
      f"gain={d1.coarse_acc.mean() - d1.fine_acc.mean():+.4f}")

z = d2[d2.f1_fine < 0.01]
print(f"\nTHE ZERO-F1 CLASSES (n={len(z)}) - are they at least in the right lineage?")
print(z[['held', 'cluster', 'n', 'f1_fine', 'acc_fine', 'acc_coarse', 'lineage']].to_string(index=False))
print(f"\n  mean fine accuracy on these  = {z.acc_fine.mean():.4f}")
print(f"  mean COARSE accuracy on these = {z.acc_coarse.mean():.4f}")

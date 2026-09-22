"""DOES THE HARMONIZED VOCABULARY COST SIGNAL? - the last piece of the decomposition.

Section 1a showed the decoder is not the lever (four families span 0.0195). What remains is
~0.235, which bundles TWO things that have never been separated:

  (a) having no target labels at all
  (b) predicting HARMONIZED clusters instead of the cohort's own NATIVE labels

Same embedding, same cells, same patient split, same decoder. Only the TARGET changes:

  -> native       the cohort's own label vocabulary
  -> harmonized   the fold-local cluster the cohort's labels map into

Both are fit ON the held-out cohort, so supervision is held constant and the only difference is
the vocabulary. A large gap means the harmonisation itself destroys signal and the label space is
back as the primary lever. A small gap means the whole 0.235 is supervision, and few-shot is the
only real answer.
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
SEED, CAP = 20260810, 20000
rng = np.random.default_rng(SEED)

cohorts = s3.available()
triples, tri2idx, per, genes, ncoh = s2.read_panel(cohorts)
V = len(triples)
train = [c for c in cohorts if SPECS[c]['role'] == 'train']
excl = s6.excluded_pairs()


def probe(Xtr, ytr, Xte, yte):
    if len(Xtr) > CAP:
        s = rng.choice(len(Xtr), CAP, replace=False)
        Xtr, ytr = Xtr[s], ytr[s]
    keep = np.isin(yte, np.unique(ytr))
    lr = LogisticRegression(max_iter=2000, class_weight='balanced')
    lr.fit(Xtr, ytr)
    return float(f1_score(yte[keep], lr.predict(Xte[keep]), average='macro'))


rows = []
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

    def enc(part):
        U, Z = d['U'][part], []
        with torch.no_grad():
            for i in range(0, len(U), 1024):
                z, _, _ = m(U[i:i + 1024], d['idx'], d['present'])
                Z.append(z.cpu())
        return torch.cat(Z).numpy()

    Xtr, Xte = enc('train'), enc('test')

    # target A: the cohort's own native labels
    v = pd.read_parquet(s2.full_table(held), columns=['cell_id', 'native_label'])
    look = v.set_index('cell_id').native_label
    nat_tr = look.reindex(d['cid']['train']).to_numpy()
    nat_te = look.reindex(d['cid']['test']).to_numpy()

    # target B: the harmonized cluster those same cells carry (what LOCO predicts)
    har_tr = d['y']['train'].cpu().numpy()
    har_te = d['y']['test'].cpu().numpy()

    f_nat = probe(Xtr, nat_tr, Xte, nat_te)
    f_har = probe(Xtr, har_tr, Xte, har_te)

    rows.append(dict(held=held, k_native=len(np.unique(nat_tr)),
                     k_harm=len(np.unique(har_tr)),
                     to_native=round(f_nat, 4), to_harmonized=round(f_har, 4),
                     delta=round(f_har - f_nat, 4), loco=round(float(r['f1_core']), 4)))
    print(f"  {held:10} k_nat={rows[-1]['k_native']:>2} k_harm={rows[-1]['k_harm']:>2}  "
          f"->native={f_nat:.4f}  ->harmonized={f_har:.4f}  "
          f"delta={f_har - f_nat:+.4f}   (LOCO zero-shot {r['f1_core']:.4f})", flush=True)

d = pd.DataFrame(rows)
d.to_csv(os.path.join(HERE, 'results', 'vocabcost.csv'), index=False)
print("\n" + d.to_string(index=False))
print(f"\nMEAN  ->native={d.to_native.mean():.4f}  ->harmonized={d.to_harmonized.mean():.4f}  "
      f"delta={d.delta.mean():+.4f}  LOCO={d.loco.mean():.4f}")
print(f"\n  supervised in harmonized space : {d.to_harmonized.mean():.4f}")
print(f"  zero-shot  in harmonized space : {d.loco.mean():.4f}")
print(f"  => cost of having NO target labels = {d.to_harmonized.mean() - d.loco.mean():+.4f}")

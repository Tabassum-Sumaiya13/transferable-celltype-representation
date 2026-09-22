"""
Build the folder to upload to Kaggle as a private Dataset, one per GPU session.

    python celltype_transfer/gpu/make_gpu_bundle.py --session1   # pretraining + encoder + classifier
    python celltype_transfer/gpu/make_gpu_bundle.py --session2   # spatial context (needs session 1's results)
    python celltype_transfer/gpu/make_gpu_bundle.py --session2 --out D:/somewhere/else

Writes work/gpu_bundle_session{1,2}/ and a zip of it next to it. The run must end with
"nothing missing" - a missing file means the notebook would fail, or worse, fall back silently.

WHY CODE AND DATA GO IN ONE DATASET, not a git clone in the notebook. A clone needs Internet on the
notebook and lets the code drift from the numbers. One upload pins the exact code and the exact
tables together, and the notebook runs with Internet OFF. MANIFEST.json carries a hash per file and
the notebook re-checks every one before it trains anything.

WHAT IS DELIBERATELY LEFT OUT.
  work/raw/*.parquet   627 MB of acquisition output. No GPU step opens it; the neighbour graph is
                       built from it LOCALLY (build_neighbour_graph.py) and only its sidecars travel.
  Datasets/            the source data.
  work/label_signatures.npz   only the local label-space builders read it; the finished fold
                       spaces travel instead, hash-checked against work/spaces/index.json.

SESSION 1 carries no model checkpoints at all: it trains the pretraining models itself, and every
later fit in the session warm-starts from those (warm_for(), fold-local). SESSION 2 carries
session 1's pretraining checkpoints and the panel.json that records Gate 2's arm decision - both
come back into work/ through import_gpu_results.py.
"""
import os
import sys
import json
import shutil
import hashlib

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
from config import ROOT, WORK, SPECS                     # noqa: E402

TRAIN = [c for c, s in SPECS.items() if s['role'] == 'train']


def sha(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while True:
            b = f.read(n)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def w(*parts):
    return os.path.join(WORK, *parts)


def session1_files():
    """Everything pretraining (Gate 2), the encoder (Gate 3) and the classifier (Gate 6) read."""
    want = [(w('values', f'{c}_full.parquet'), f'work/values/{c}_full.parquet') for c in TRAIN]
    want += [(w(f), f'work/{f}') for f in ('panel.json', 'marker_dynamic_range.csv',
                                           'marker_registry.csv', 'label_map.csv',
                                           'prototypes.npy')]
    sd = w('spaces')
    want += [(os.path.join(sd, 'index.json'), 'work/spaces/index.json')]
    if os.path.isdir(sd):
        want += [(os.path.join(sd, f), f'work/spaces/{f}') for f in sorted(os.listdir(sd))
                 if f.endswith(('.csv', '.npy'))]
    lc = w('label_conf')
    if os.path.isdir(lc):
        want += [(os.path.join(lc, f), f'work/label_conf/{f}') for f in sorted(os.listdir(lc))]
    else:
        want += [(lc, 'work/label_conf/  (run build_label_confidence.py)')]
    return want


def session2_files():
    """Session 1's inputs, plus the neighbour sidecars and session 1's pretraining checkpoints."""
    want = session1_files()
    want += [(w('values', f'{c}_neighbours.npz'), f'work/values/{c}_neighbours.npz')
             for c in TRAIN]
    want += [(w('neighbour_graph_summary.csv'), 'work/neighbour_graph_summary.csv')]
    ck = w('ckpt')
    pre = sorted(f for f in os.listdir(ck) if f.startswith('pretrain_') and f.endswith('.pt')
                 and 'quick_' not in f) if os.path.isdir(ck) else []
    if not pre:
        want += [(w('ckpt', 'pretrain_*.pt'), 'work/ckpt/pretrain_*.pt  (import session 1 first)')]
    want += [(os.path.join(ck, f), f'work/ckpt/{f}') for f in pre]
    return want


def main():
    s1, s2 = '--session1' in sys.argv, '--session2' in sys.argv
    if s1 == s2:
        sys.exit(__doc__)
    session = '1' if s1 else '2'
    out = (sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv
           else w(f'gpu_bundle_session{session}'))
    if os.path.exists(out):
        shutil.rmtree(out)

    # the code, minus caches, tests and this folder's own bundles
    shutil.copytree(PKG, os.path.join(out, 'celltype_transfer'),
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'tests'))

    want = session1_files() if s1 else session2_files()
    manifest, missing, total = [], [], 0
    for src, rel in want:
        if not os.path.isfile(src):
            missing.append(rel)
            continue
        dst = os.path.join(out, rel.replace('/', os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        n = os.path.getsize(src)
        total += n
        manifest.append(dict(path=rel, bytes=n, sha256_16=sha(src)))
    # the Kaggle input mount is read-only; the notebook copies everything to /kaggle/working
    os.makedirs(os.path.join(out, 'work', 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(out, 'reports', 'figures'), exist_ok=True)

    panel = w('panel.json')
    arm = json.load(open(panel)).get('stage2_arm') if os.path.exists(panel) else None
    if s2 and arm is None:
        missing.append('work/panel.json has no stage2_arm - import session 1 results first')
    json.dump(dict(session=session, stage2_arm=arm, train_cohorts=TRAIN,
                   total_bytes=total, files=manifest),
              open(os.path.join(out, 'MANIFEST.json'), 'w'), indent=2)

    code = sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(os.path.join(out, 'celltype_transfer')) for f in fs)
    print(f"bundle: {out}")
    print(f"  code {code / 1e6:7.2f} MB")
    print(f"  data {total / 1e6:7.2f} MB   ({len(manifest)} files)")
    print(f"  stage2_arm = {arm}")
    if missing:
        print(f"\n  MISSING - the session {session} notebook will fail without these:")
        for m in missing:
            print(f"    {m}")
        sys.exit(1)
    z = shutil.make_archive(out, 'zip', out)
    print(f"\n  nothing missing. Upload {z} as a NEW private Kaggle Dataset "
          f"named cta-session{session}.")


if __name__ == '__main__':
    main()

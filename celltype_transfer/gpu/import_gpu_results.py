"""
Bring a Kaggle session's results zip back into work/ and reports/, after checking it belongs here.

    python celltype_transfer/gpu/import_gpu_results.py session1_results.zip
    python celltype_transfer/gpu/import_gpu_results.py session2_results.zip
    python celltype_transfer/gpu/import_gpu_results.py session1_results.zip --dry   # check only

Nothing is copied unless EVERY check passes. A fit that was trained on another split, another
vocabulary or another fold's label space would load here without complaint and then be refitted
(or worse, mis-read) by the next stage - so it is caught at the door instead.

  work/panel.json   must equal the local one except for `stage2_arm` / `stage2_arm_note`, the two
                    keys Gate 2 writes on Kaggle. The local file then takes those two keys.
  work/ckpt/*.pt    every fit's `split_fp` must equal the local patient split, its `vocab_fp` (when
                    it records one) the local vocabulary, and its `space` ("fold:<fold>:<hash>")
                    the hash in the local work/spaces/index.json. No `quick_` checkpoint may travel.
  reports/**        copied as they are.
  logs/**           copied to reports/gpu_logs/session<N>/.
"""
import os
import sys
import json
import shutil
import zipfile
import tempfile

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
from config import WORK, REPORTS                         # noqa: E402

ARM_KEYS = ('stage2_arm', 'stage2_arm_note')


def check_panel(src, problems):
    local = os.path.join(WORK, 'panel.json')
    if not os.path.exists(local):
        problems.append('no local work/panel.json - run build_marker_vocabulary.py first')
        return None
    a, b = json.load(open(local)), json.load(open(src))
    diff = sorted(k for k in set(a) | set(b) if k not in ARM_KEYS and a.get(k) != b.get(k))
    if diff:
        problems.append(f'work/panel.json differs from the local one outside the arm keys: {diff}')
    return b.get('stage2_arm')


def check_ckpts(paths, problems):
    import torch
    import splits
    import build_marker_vocabulary as vocab
    sfp, vfp = splits.split_fp(), vocab.panel_fp()
    index = json.load(open(os.path.join(WORK, 'spaces', 'index.json')))
    n = 0
    for p in paths:
        name = os.path.basename(p)
        if 'quick_' in name:
            problems.append(f'{name}: a smoke-test checkpoint travelled')
            continue
        r = torch.load(p, weights_only=False, map_location='cpu')
        if isinstance(r, list):                      # a sweep summary, not a fit
            continue
        n += 1
        if r.get('split_fp') != sfp:
            problems.append(f"{name}: split {r.get('split_fp')} != local {sfp}")
        if 'vocab_fp' in r and r['vocab_fp'] != vfp:
            problems.append(f"{name}: vocabulary {r['vocab_fp']} != local {vfp}")
        sp = str(r.get('space') or '')
        if sp.startswith('fold:'):
            _, fold, sha = sp.split(':')
            want = index.get(fold, {}).get('sha256_16')
            if sha != want:
                problems.append(f'{name}: label space {fold}:{sha} != local {want}')
        elif name.startswith(('classifier_', 'encoder_', 'spatial_')):
            problems.append(f"{name}: not trained in a fold-local label space ({sp or 'none'})")
    return n


def main():
    zips = [a for a in sys.argv[1:] if not a.startswith('--')]
    if not zips:
        sys.exit(__doc__)
    dry = '--dry' in sys.argv
    for zp in zips:
        session = ''.join(ch for ch in os.path.basename(zp).split('_')[0] if ch.isdigit()) or 'x'
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(zp) as z:
                z.extractall(tmp)
            problems, moves = [], []
            for r, _, fs in os.walk(tmp):
                for f in fs:
                    rel = os.path.relpath(os.path.join(r, f), tmp).replace(os.sep, '/')
                    if rel.startswith('work/ckpt/') and rel.endswith('.pt'):
                        dst = os.path.join(WORK, 'ckpt', f)
                    elif rel == 'work/panel.json':
                        dst = os.path.join(WORK, 'panel.json')
                    elif rel.startswith('reports/'):
                        dst = os.path.join(REPORTS, *rel.split('/')[1:])
                    elif rel.startswith('logs/'):
                        dst = os.path.join(REPORTS, 'gpu_logs', f'session{session}',
                                           *rel.split('/')[1:])
                    else:
                        problems.append(f'unexpected file in the zip: {rel}')
                        continue
                    moves.append((os.path.join(r, f), dst, rel))

            pj = [s for s, _, rel in moves if rel == 'work/panel.json']
            arm = check_panel(pj[0], problems) if pj else None
            n = check_ckpts([s for s, _, rel in moves if rel.startswith('work/ckpt/')], problems)

            print(f"{zp}")
            print(f"  {len(moves)} files, {n} fits checked, stage2_arm = {arm}")
            if problems:
                print("  NOT IMPORTED - problems:")
                for p in problems:
                    print(f"    {p}")
                sys.exit(1)
            if dry:
                print("  all checks pass (--dry: nothing copied)")
                continue
            replaced = 0
            for src, dst, _ in moves:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                replaced += os.path.exists(dst)
                shutil.copy2(src, dst)
            print(f"  imported into {WORK} and {REPORTS} ({replaced} existing files replaced)")


if __name__ == '__main__':
    main()

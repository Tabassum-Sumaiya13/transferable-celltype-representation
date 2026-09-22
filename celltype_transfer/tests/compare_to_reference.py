"""
Compare the locally built artifacts in work/ with the previous run in work_reference/.

    python celltype_transfer/tests/compare_to_reference.py
    python celltype_transfer/tests/compare_to_reference.py --ref D:/some/other/work

Every CPU step is deterministic, so IDENTICAL is the expected answer for every row. A file that
differs is a finding to explain before any GPU run, never something to wave through.

Bytes are compared first. A parquet or npz that differs in bytes is then compared by CONTENT
(the writer can stamp metadata), and reported as `same content` only if every column / array is
exactly equal. work_reference keeps the old pipeline2 file names; NAMES maps them.
"""
import os
import sys
import json
import hashlib

import numpy as np
import pandas as pd

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)
from config import ROOT, WORK                            # noqa: E402

# new name (in work/) -> old name (in work_reference/)
NAMES = {'label_signatures.npz': 's1b_signatures.npz',
         'marker_dynamic_range.csv': 's2_dynrange.csv',
         'neighbour_graph_summary.csv': 's4_graph.csv'}
ARM_KEYS = ('stage2_arm', 'stage2_arm_note')       # written later, by Gate 2 on the GPU
TIMING = ['seconds']                                 # wall-clock columns, never reproducible


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def old_name(rel):
    d, f = os.path.split(rel)
    if f.endswith('_neighbours.npz'):
        f = f.replace('_neighbours.npz', '_nbr.npz')
    return os.path.join(d, NAMES.get(f, f))


def same_content(a, b):
    if a.endswith('.parquet'):
        x, y = pd.read_parquet(a), pd.read_parquet(b)
        return list(x.columns) == list(y.columns) and x.equals(y)
    if a.endswith('.npz'):
        x, y = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
        return sorted(x.files) == sorted(y.files) and all(
            np.array_equal(x[k], y[k]) for k in x.files)
    if a.endswith('.npy'):
        return np.array_equal(np.load(a, allow_pickle=True), np.load(b, allow_pickle=True))
    if a.endswith('.csv'):
        # `seconds` is wall-clock time, the one column a deterministic build may not reproduce
        x, y = pd.read_csv(a), pd.read_csv(b)
        x, y = x.drop(columns=TIMING, errors='ignore'), y.drop(columns=TIMING, errors='ignore')
        return list(x.columns) == list(y.columns) and x.equals(y)
    if a.endswith('panel.json'):
        x, y = json.load(open(a)), json.load(open(b))
        return all(x.get(k) == y.get(k) for k in set(x) | set(y) if k not in ARM_KEYS)
    return False


def main():
    ref = (sys.argv[sys.argv.index('--ref') + 1] if '--ref' in sys.argv
           else os.path.join(ROOT, 'work_reference'))
    rows = []
    for root, dirs, files in os.walk(WORK):
        dirs[:] = [d for d in dirs if d not in ('ckpt', '_v1_2026-09-11')
                   and not d.startswith('gpu_bundle')]
        for f in sorted(files):
            if f.endswith('.zip') or f == 'api_cache.json':
                continue
            rel = os.path.relpath(os.path.join(root, f), WORK)
            a, b = os.path.join(WORK, rel), os.path.join(ref, old_name(rel))
            if not os.path.exists(b):
                rows.append((rel, 'NEW - no reference file'))
            elif sha(a) == sha(b):
                rows.append((rel, 'identical'))
            elif same_content(a, b):
                rows.append((rel, 'same content'))
            else:
                rows.append((rel, 'DIFFERENT'))
    w = max(len(r) for r, _ in rows)
    for rel, v in sorted(rows):
        print(f"  {rel:{w}}  {v}")
    bad = [r for r, v in rows if v == 'DIFFERENT']
    print(f"\n{len(rows)} files: {sum(v == 'identical' for _, v in rows)} identical, "
          f"{sum(v == 'same content' for _, v in rows)} same content, "
          f"{sum(v.startswith('NEW') for _, v in rows)} new, {len(bad)} DIFFERENT")
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()

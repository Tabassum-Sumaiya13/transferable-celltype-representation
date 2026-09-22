"""
Golden (characterisation) test - proves a refactor changed NOTHING the pipeline computes.

    python celltype_transfer/tests/golden.py --setup      # copy the inputs it needs into a scratch work dir
    python celltype_transfer/tests/golden.py --record     # run on the current code, write golden.json
    python celltype_transfer/tests/golden.py --check      # run again, compare to golden.json, exit 1 on any diff

WHY. The codebase is being renamed and cleaned. "Cleaner" must not mean "different": a moved
function that consumes one extra random number changes every weight downstream, and nothing
would error. This test pins three things on the CPU, where computation is deterministic:

  1. RE-SCORE   the published Stage 3 (lambda 0.01) and Stage 6 (proto2) checkpoints, all 7 folds,
                through the current loaders, label spaces, splits and metric. Must equal the
                f1_core stored in each checkpoint.
  2. TINY FITS  every training stage's own per-fold run function (Stage 2, 3, 6, 4, 10) on a tiny
                budget: 1 epoch, a few hundred cells. The SHA-256 of the trained weights and the
                scores are recorded. Any change to data order, RNG use, init or loss moves them.
  3. ARTIFACTS  split fingerprint, vocabulary fingerprint, per-cohort split masks, fold-space hashes.

It never writes into the real work/ or reports/: CT_WORK and CT_REPORTS point at a scratch copy.
Reference inputs are read from work_reference/ (the published run), never modified.
"""
import hashlib
import json
import os
import shutil
import sys
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
ROOT = os.path.dirname(PKG)
REF = os.path.join(ROOT, 'work_reference')
SCRATCH = os.environ.get('GOLDEN_SCRATCH', os.path.join(ROOT, '_golden_scratch'))
GOLDEN = os.path.join(HERE, 'golden.json')

# Module names. The rename edits ONLY this map; the recorded numbers must not move.
MOD = dict(
    s2='pretrain_masked_markers', s3='train_adversarial_encoder', s6='train_prototype_classifier',
    s4='train_spatial_context', s10='compare_external_baseline',
    spaces='build_fold_label_spaces', models_encoder='models.encoder',
    splits='splits', vocab='build_marker_vocabulary', metrics='metrics', config='config',
)
# work_reference keeps the PUBLISHED file names; the scratch copy uses the names the code reads now.
RENAME = {'s2_dynrange.csv': 'marker_dynamic_range.csv', 's4_graph.csv': 'neighbour_graph_summary.csv',
          's1b_signatures.npz': 'label_signatures.npz'}

COHORTS = ['CRC', 'UPMC', 'Keren', 'ferguson', 'Phillips', 'Danenberg', 'Sorin']


def setup():
    """Copy exactly the inputs the test reads. ~400 MB, all regenerable from work_reference."""
    w = os.path.join(SCRATCH, 'work')
    if os.path.exists(SCRATCH):
        shutil.rmtree(SCRATCH)
    for d in ('values', 'spaces', 'label_conf', 'ckpt'):
        os.makedirs(os.path.join(w, d), exist_ok=True)
    for f in ('panel.json', 'marker_registry.csv', 's2_dynrange.csv', 'label_map.csv',
              'label_graph.json', 'prototypes.npy', 's1b_signatures.npz', 's4_graph.csv'):
        shutil.copy2(os.path.join(REF, f), os.path.join(w, RENAME.get(f, f)))
    for c in COHORTS:
        shutil.copy2(os.path.join(REF, 'values', c + '_full.parquet'),
                     os.path.join(w, 'values', c + '_full.parquet'))
        shutil.copy2(os.path.join(REF, 'values', c + '_nbr.npz'),
                     os.path.join(w, 'values', c + '_neighbours.npz'))
    for f in glob.glob(os.path.join(REF, 'spaces', '*')):
        if os.path.isfile(f):
            shutil.copy2(f, os.path.join(w, 'spaces', os.path.basename(f)))
    for f in glob.glob(os.path.join(REF, 'label_conf', '*')):
        shutil.copy2(f, os.path.join(w, 'label_conf', os.path.basename(f)))
    for pat in ('s2_armB.pt', 's2_armb_loco_*.pt', 's2_armb_loto_*.pt'):
        for f in glob.glob(os.path.join(REF, 'ckpt', pat)):
            b = os.path.basename(f)
            shutil.copy2(f, os.path.join(w, 'ckpt', 'pretrain_' + b[len('s2_'):]))
    print(f"scratch ready: {SCRATCH}")


def _env():
    os.environ['CT_WORK'] = os.path.join(SCRATCH, 'work')
    os.environ['CT_REPORTS'] = os.path.join(SCRATCH, 'reports')
    sys.argv = [sys.argv[0], '--cpu']          # DEV and the import-time switches read argv
    sys.path.insert(0, PKG)


def _hash_state(state):
    h = hashlib.sha256()
    for k in sorted(state):
        v = state[k]
        if hasattr(v, 'detach'):
            h.update(k.encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v, dict):
            h.update(k.encode())
            h.update(_hash_state(v).encode())
    return h.hexdigest()[:16]


def _r(x):
    return None if x is None else round(float(x), 6)


def run():
    _env()
    import importlib
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=True)
    s2 = importlib.import_module(MOD['s2'])
    s3 = importlib.import_module(MOD['s3'])
    s6 = importlib.import_module(MOD['s6'])
    s4 = importlib.import_module(MOD['s4'])
    s10 = importlib.import_module(MOD['s10'])
    sp = importlib.import_module(MOD['spaces'])
    enc_mod = importlib.import_module(MOD['models_encoder'])
    splits = importlib.import_module(MOD['splits'])
    vocab = importlib.import_module(MOD['vocab'])
    metrics = importlib.import_module(MOD['metrics'])
    cfg = importlib.import_module(MOD['config'])
    from config import SPECS

    out = {}
    cohorts = s3.available()
    triples, tri2idx, per, genes, ncoh = vocab.read_panel(cohorts)
    V = len(triples)
    train = [c for c in cohorts if SPECS[c]['role'] == 'train']
    excl = s6.excluded_pairs()

    # ---------------------------------------------------------------- 3. artifacts
    out['artifacts'] = dict(
        split_fp=splits.split_fp(), panel_fp=vocab.panel_fp(), n_vocab=V, train=train,
        n_excluded=len(excl),
        split_masks={c: hashlib.sha256(np.concatenate(
            [m.astype('uint8') for m in splits.split_masks(
                c, __import__('pandas').read_parquet(cfg.full_table(c),
                                                     columns=['image_id']).image_id.to_numpy()
            ).values()]).tobytes()).hexdigest()[:16] for c in train},
        fold_spaces={h: sp.fold_space([h])[2]['sha256_16'] for h in train},
    )
    print('artifacts', out['artifacts']['split_fp'], out['artifacts']['panel_fp'])

    # ---------------------------------------------------------------- 1. re-score published fits
    ref_ckpt = os.path.join(REF, 'ckpt')
    rescore = {}
    for h in train:
        L, pt, s = s3.space_for(h, 'fold')
        # Stage 6 proto2
        r = torch.load(os.path.join(ref_ckpt, f'{CKPT_PREFIX_REF["s6"]}proto2_fold_{h}.pt'),
                       weights_only=False, map_location='cpu')
        m = s6.Stage6(V, L['n'], head='proto')
        m.load_state_dict(r['state'])
        m.eval()
        d = s6.load_cohort(h, per[h], tri2idx, V, L, excl)
        yp = s6.predict(m, d)
        f1, _ = metrics.macro_f1(d['y']['test'].cpu().numpy(), yp, L['n'], drop=L['unreliable'])
        rescore[f's6_proto2_{h}'] = dict(got=_r(f1), stored=_r(r['f1_core']))
        # Stage 3 lambda 0.01
        r = torch.load(os.path.join(ref_ckpt, f'{CKPT_PREFIX_REF["s3"]}fold_lam0.01_{h}.pt'),
                       weights_only=False, map_location='cpu')
        st = r['state']
        enc = enc_mod.CellEncoder(st['n_vocab'], d_tok=st['d_tok'], d_z=st['d_z'],
                                  blocks=st['blocks'], heads=st['heads_n'])
        enc.load_state_dict(st['enc'])
        heads = enc_mod.Heads(st['d_z'], st['n_class'], len(st['slide_index']),
                              len(st['cohorts']))
        heads.load_state_dict(st['heads'])
        enc.eval(); heads.eval()
        d3 = s3.load_cohort(h, per[h], tri2idx, V, 'absent', None, L)
        yp = s3.predict(enc, heads, d3)
        f1, _ = metrics.macro_f1(d3['y']['test'].cpu().numpy(), yp, L['n'], drop=L['unreliable'])
        rescore[f's3_lam0.01_{h}'] = dict(got=_r(f1), stored=_r(r['f1_core']))
        print('rescore', h, rescore[f's6_proto2_{h}'], rescore[f's3_lam0.01_{h}'])
    out['rescore'] = rescore
    out['rescore_mean'] = dict(
        s6_proto2=_r(np.mean([v['got'] for k, v in rescore.items() if k.startswith('s6')])),
        s3_lam001=_r(np.mean([v['got'] for k, v in rescore.items() if k.startswith('s3')])))

    # ---------------------------------------------------------------- 2. tiny fits
    s2.N_TRAIN, s2.VAL_CELLS, s2.SCORE_CELLS = 400, 200, 200
    s3.N_TRAIN, s3.SCORE_CELLS, s3.PROBE_HOLD = 400, 300, 150
    s6.N_TRAIN, s6.SCORE_CELLS = 400, 300
    s10.MAX_EPOCHS = 1
    fits = {}

    def rec(name, r, state=None):
        st = state if state is not None else r.get('state')
        row = {k: _r(r[k]) for k in ('f1_core', 'f1_all', 'f1_majority', 'f1_random',
                                     'fresh_bits', 'cotrained_bits', 'cohort_acc', 'derangement')
               if k in r and r[k] is not None}
        if 'r2' in r and hasattr(r['r2'], 'select_dtypes'):
            row['r2'] = hashlib.sha256(
                r['r2'].select_dtypes('number').round(8).to_numpy().tobytes()).hexdigest()[:16]
        row['state'] = _hash_state(st) if st is not None else None
        fits[name] = row
        print('fit', name, row)

    held = 'Sorin'
    L, pt, s = s3.space_for(held, 'fold')

    # Stage 2 - masked-marker pretraining (LOCO arm B)
    rec('s2_armb', s2.loco_run('golden_armb_Sorin', 'absent', held, train, per, tri2idx, V,
                               excl, True, 1))

    # Stage 3 - encoder, lambda 0 and 0.01
    for lam in (0.0, 0.01):
        rec(f's3_lam{lam}', s3.loco_run(f'golden_lam{lam}_{held}', lam, held, train, per,
                                        tri2idx, V, L, 'absent', 'fold-local', True, 1,
                                        space=s))

    # Stage 6 - the four arms (and the confidence-off arm, on a fold where UPMC trains)
    for name, head, vic, conf, lam in [('proto3', 'proto', True, True, 0.0),
                                       ('proto2', 'proto', False, True, 0.0),
                                       ('linear', 'linear', True, True, 0.0),
                                       ('proto2adv', 'proto', False, True, 0.01),
                                       ('noconf', 'proto', False, False, 0.0)]:
        rec(f's6_{name}', s6.loco_run(f'golden_{name}_{held}', held, train, per, tri2idx,
                                      triples, L, excl, head, vic, conf, True, 1,
                                      proto_path=pt, space=s, lam_adv=lam))

    # Stage 4 - spatial arms. gate_run stores no weights, so capture them from fit().
    captured = {}
    real_fit = s4.fit

    def spy(*a, **k):
        m, info = real_fit(*a, **k)
        captured['state'] = {kk: v.detach().cpu().clone() for kk, v in m.state_dict().items()}
        return m, info
    s4.fit = spy
    try:
        for name, nbr, ctx, shuf, px in [('cell', False, False, False, None),
                                         ('neigh', True, False, False, None),
                                         ('shuffle', True, False, True, None),
                                         ('ctx', True, True, False, None),
                                         ('px1.3_neigh', True, False, False, {'UPMC': 1.3})]:
            r = s4.gate_run(f'golden_{name}_{held}', held, train, per, tri2idx, triples, L,
                            excl, nbr, ctx, shuf, True, 1, space=s, px_scales=px,
                            proto_path=pt)
            rec(f's4_{name}', r, state=captured.pop('state'))
    finally:
        s4.fit = real_fit

    # Stage 10 - the external MAPS baseline, one arm
    real_maps = s10.fit_maps

    def spy_maps(*a, **k):
        m, mu, sd, info = real_maps(*a, **k)
        captured['state'] = {kk: v.detach().cpu().clone() for kk, v in m.state_dict().items()}
        return m, mu, sd, info
    s10.fit_maps = spy_maps
    try:
        r = s10.run_fold(f'golden_shared_{held}', 'shared', held, per, tri2idx, triples, L,
                         excl, True, s)
        rec('s10_shared', r, state=captured.pop('state'))
    finally:
        s10.fit_maps = real_maps
    out['fits'] = fits
    return out


# The PUBLISHED checkpoints keep their original names in work_reference, whatever the code is
# renamed to.
CKPT_PREFIX_REF = dict(s3='s3_', s6='s6_')


def main():
    if '--setup' in sys.argv:
        setup()
        return
    record, check = '--record' in sys.argv, '--check' in sys.argv
    if not (record or check):
        sys.exit(__doc__)
    got = run()
    if record:
        with open(GOLDEN, 'w') as f:
            json.dump(got, f, indent=1, sort_keys=True)
        print(f"recorded {GOLDEN}")
        return
    want = json.load(open(GOLDEN))
    got = json.loads(json.dumps(got, sort_keys=True))
    bad = []

    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                walk(a.get(k), b.get(k), f"{path}/{k}")
        elif a != b:
            bad.append(f"{path}: golden={a!r} now={b!r}")
    walk(want, got, '')
    if bad:
        print("GOLDEN CHECK FAILED - the refactor changed what the pipeline computes:")
        print("\n".join(bad))
        sys.exit(1)
    print("GOLDEN CHECK PASSED - identical weights, scores and artifacts")


if __name__ == '__main__':
    main()

"""
STAGE 3 - cell encoder + domain-adversarial training.  Produces GATE 3.

What it does. Turns a cell's marker tokens into `z_cell` (128-d) that carries biology but not
batch, using a gradient reversal layer against SLIDE identity. Stage 1b supplies the shared label
space, so for the first time in the pipeline a cross-cohort classifier can actually be scored.

    python train_adversarial_encoder.py --lambda-sweep          # GATE 3
    python train_adversarial_encoder.py --lambda-sweep --quick  # smoke test: proves the code path, scores nothing
    python train_adversarial_encoder.py --lambda-sweep --lambdas 0,0.03,0.3   # a narrower grid
    python train_adversarial_encoder.py --lambda-sweep --refit  # ignore the checkpoint cache
    python train_adversarial_encoder.py --lambda-sweep --cpu    # force CPU even where a GPU exists

Runs on CPU or CUDA from the SAME file - there is no separate GPU fork, because two copies of a
training script drift and then the local number and the Kaggle number stop being comparable. The
device is picked automatically; `--cpu` forces the fallback. Every cohort's tensors are moved onto
the device once at load time (about 40 MB for all five) so the training loop never transfers.

TWO CHANGES TO THE WRITTEN GATE, both settled before the run.

D-32. Gate 3 honours the stroma waiver (D-23) exactly as Gate 7 must. The 23 labels flagged
`unreliable=1` in work/label_map.csv - 3 clusters, 420,862 cells, 8.5% of the roster - are kept
out of the headline macro-F1 and reported in their own table. Without this, Gate 3's number would
not be comparable with Gate 7's, and a region already known to be unreliable would be moving the
lambda decision.

Scope note. The cell-type head here is a PLAIN LINEAR head with class-balanced sampling. The
prototype loss, the collapse guard and Kendall sigma weighting are Stage 6's job. Gate 3 asks one
question - does the adversary help cross-cohort transfer - and the fewest moving parts answer it
most fairly. Descendant-tolerant cross-entropy is also deliberately absent: section 7 gap 2
records that Stage 1b's nesting layer missed all three cases declared for it, so the parent/child
graph is not yet trustworthy enough to put inside a loss.
"""
import os, sys, json, time, zlib, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # runs from any directory
import config
from config import SPECS, WORK, VALUES, PANEL, REPORTS, FIGURES, SEED
from models.encoder import CellEncoder, Heads, SlideProbe
import splits
import build_marker_vocabulary as vocab
import metrics
import pretrain_masked_markers as pretrain
from metrics import macro_f1                                          # plan F6, D-F6: moved to a
                                                                       # stage-independent metric
                                                                       # library, byte-identical -
                                                                       # see metrics.py's docstring

CKPT = os.path.join(WORK, 'ckpt')
os.makedirs(CKPT, exist_ok=True)

from models.device import DEV   # one rule for all stages; --cpu forces the CPU

# --------------------------------------------------------------- declared constants
LAMBDAS      = [0.0, 0.01, 0.03, 0.1, 0.3]   # the plan's grid, "at least" these
COHORT_FRAC  = 0.1        # the cohort head gets a tenth of lambda - guard, not objective (D-16)
D_TOK, D_Z   = 64, 128
BLOCKS, HEADS = 2, 4
N_TRAIN      = 15_000
SCORE_CELLS  = 8_000
PROBE_HOLD   = 4_000      # D-35: cells held out of training, from the SAME slides, for the probe
BATCH, LR    = 512, 1e-3
EPOCHS, PATIENCE = 30, 4
RAMP_EPOCHS  = 5          # lambda ramps 0 -> cap over this many epochs
# TEST_FRAC / VAL_FRAC used to be repeated here and were never read - the split is s2's alone
# (splits.split_plan, by patient since F3). Removed so there is one copy to change.
PROBE_STEPS  = 600


from config import rng as _rng   # one definition, config.rng


# ----------------------------------------------------------------------------- labels
def label_space(path=None):
    """Stage 1b's shared label space, plus the stroma waiver flags.

    `label_map.csv` is AUTO-GENERATED and never hand-edited - that is the whole point of Stage 1b.
    It is read here, not reproduced.

    `path` reads another map in the same format - a fold-local space from build_fold_label_spaces.py --folds
    (plan F4). Cluster -1 there means NOVEL: the label is kept in `raw` and listed in `novel`, but
    it gets NO key, so its cells are dropped by load_cohort() exactly like an unmapped label -
    out of the closed-set score, never forced into a class. The shipped map has no -1 rows, so
    for it this returns what it always did.
    """
    lm = pd.read_csv(path or os.path.join(WORK, 'label_map.csv'), keep_default_na=False)
    ok = lm[lm.cluster >= 0]
    clusters = sorted(ok.cluster.unique())
    c2i = {c: i for i, c in enumerate(clusters)}
    key = {(r.cohort, r.label): c2i[r.cluster] for r in ok.itertuples()}
    names = (ok.drop_duplicates('cluster').set_index('cluster').cluster_name
             .reindex(clusters).fillna('').to_dict())
    unreliable = sorted({c2i[r.cluster] for r in ok.itertuples() if int(r.unreliable) == 1})
    novel = [(r.cohort, r.label, int(r.n_cells)) for r in lm.itertuples() if r.cluster < 0]
    return dict(n=len(clusters), key=key, names=[names[c] for c in clusters],
                unreliable=set(unreliable), raw=lm, novel=novel)


def space_for(held, mode):
    """The label space a fold is trained and scored in: (L, prototypes path, space record).

    mode 'fold' (default, plan F4 - benchmark_protocol.yaml label_space.primary_rule): the space
    built from this fold's TRAINING cohorts only, with the held-out cohort's labels placed by the
    admission rule or marked NOVEL (build_fold_label_spaces.py --folds). Its own prototypes: the training-cohort
    centroids. The map's hash is checked against the one the build recorded.

    mode 'shipped': the whole-roster work/label_map.csv - built WITH every held-out cohort
    present, so NOT blind to any of them. Kept as the comparison row, never the headline.

    The `space` string goes into every checkpoint, and a cached fit from another space refits.
    """
    if mode == 'fold':
        import build_fold_label_spaces as fold_spaces
        lm, pt, e = fold_spaces.fold_space([held])
        return label_space(lm), pt, dict(space=f"fold:{e['fold']}:{e['sha256_16']}",
                                         cut_rule=e['cut_rule'], tau=e['tau'])
    p = os.path.join(WORK, 'label_map.csv')
    sha = hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
    return label_space(), None, dict(space=f'shipped:{sha}', cut_rule='whole roster', tau=None)


def space_mode():
    """--space fold (default) | shipped, shared by Stages 3 and 6."""
    mode = sys.argv[sys.argv.index('--space') + 1] if '--space' in sys.argv else 'fold'
    if mode not in ('fold', 'shipped'):
        sys.exit(f"--space must be 'fold' or 'shipped', not {mode!r}")
    return mode


# ----------------------------------------------------------------------------- data
def load_cohort(c, triples_c, tri2idx, n_vocab, arm, excluded, L):
    """One cohort as tensors: values, slot layout, cluster label, slide index, optional L2.

    Cells whose (cohort, native_label) is not in the Stage 1b map are dropped - they carry no
    label in the shared space, so they cannot be scored and must not be trained on either.
    """
    tl = list(triples_c)
    v = pd.read_parquet(config.full_table(c),
                        columns=['image_id', 'patient_id', 'native_label'] +
                                [f'u_coh::{t}' for t in tl])
    y = np.array([L['key'].get((c, str(lab)), -1) for lab in v.native_label])
    ok = y >= 0
    if not ok.any():
        return None
    v, y = v[ok].reset_index(drop=True), y[ok]
    U = v[[f'u_coh::{t}' for t in tl]].to_numpy('float32')
    # the table's own ids, unchanged ('CRC|reg001_A') - the one format splits.split_masks accepts (F3)
    img = v.image_id.to_numpy()
    sm = splits.split_masks(c, img)

    def draw(mask, cap, why):
        w = np.flatnonzero(mask)
        if len(w) > cap:
            w = np.sort(_rng('draw', c, why).choice(w, cap, replace=False))
        return w

    # D-35. The slide probe needs cells the encoder did not train on, FROM SLIDES IT DID - so the
    # probe is asked to recognise slides it has examples of, which is the only question a slide
    # classifier can answer. The probe holdout is reserved FIRST so the training draw never has to
    # shrink for it. (Still SLIDES after F3: the probe asks about slide identity; the training
    # slides are now the slides of the training patients.)
    w_tr = np.flatnonzero(sm['train'])
    w_tr = w_tr[_rng('draw', c, 'trainpool').permutation(len(w_tr))]
    n_probe = min(PROBE_HOLD, len(w_tr) // 5)
    probe_ix, train_ix = np.sort(w_tr[:n_probe]), np.sort(w_tr[n_probe:n_probe + N_TRAIN])

    # A slide can be drawn into the probe holdout and miss the training draw. Such a slide has no
    # training example for either discriminator, which is the exact defect D-35 removes, so those
    # cells are dropped rather than scored.
    tr_slides = set(img[train_ix])
    probe_ix = probe_ix[np.array([s in tr_slides for s in img[probe_ix]], bool)] \
        if len(probe_ix) else probe_ix

    parts = dict(train=train_ix, sprobe=probe_ix, val=draw(sm['val'], N_TRAIN // 4, 'val'),
                 test=draw(sm['test'], SCORE_CELLS, 'test'))
    slots = np.array([tri2idx[t] for t in tl])

    if arm == 'absent':
        M = n_vocab
        full = np.zeros((len(U), M), 'float32'); full[:, slots] = U
        idx = np.arange(M)
        present = np.zeros(M, bool); present[slots] = True
        U = full
    else:
        idx, present = slots, None

    # Everything lands on the device once, here. All five cohorts together are ~40 MB, so the
    # training loop never pays a host-to-device transfer.
    pat = v.patient_id.to_numpy()          # already 'CRC|1' (loaders.py) - no second prefix
    return dict(cohort=c, pat={k: pat[w] for k, w in parts.items()},
                idx=torch.from_numpy(np.ascontiguousarray(idx)).long().to(DEV),
                present=None if present is None else torch.from_numpy(present).to(DEV),
                U={k: torch.from_numpy(U[w]).to(DEV) for k, w in parts.items()},
                y={k: torch.from_numpy(y[w]).long().to(DEV) for k, w in parts.items()},
                img={k: img[w] for k, w in parts.items()},
                n={k: len(w) for k, w in parts.items()})


def nested_slide_index(data, keys=('train',)):
    """D-38. Slide vocabulary restricted to slides that SHARE A PATIENT with another slide.

    Gate 3's adversary was told to erase slide identity, but on this roster a slide is often a
    whole patient: Keren has 1.00 slides per patient and Sorin 1.13, against CRC 4.00, UPMC 3.80
    and Phillips 4.93. Counting the actual classes, 51-73% of what the adversary was erasing in
    four of five folds was an individual PATIENT - that is tumour biology, not batch.

    Restricting the vocabulary to multi-slide patients makes every remaining distinction technical
    by construction: same patient, same tumour, same block, different acquisition. Cells from
    single-slide patients contribute nothing to the adversarial loss, which drops exactly the
    Keren and Sorin cases that were poisoning it - without any per-cohort special-casing.

    Returns (index, keep) where `keep` maps a cohort to a boolean mask per split, so the caller
    knows which cells may be used for the adversarial term.
    """
    from collections import defaultdict
    slides_of = defaultdict(set)
    for d in data.values():
        for k in keys:
            for s, p in zip(d['img'][k], d['pat'][k]):
                slides_of[p].add(s)
    usable = {s for p, ss in slides_of.items() if len(ss) >= 2 for s in ss}
    idx = {v: i for i, v in enumerate(sorted(usable))}
    keep = {c: {k: np.isin(d['img'][k], list(usable)) for k in d['img']}
            for c, d in data.items()}
    return idx, keep


def slide_index(data, keys=('train',)):
    """Slide vocabulary for the adversary. Ids are cohort-prefixed - they are only unique WITHIN a
    cohort, and Keren and Sorin both start numbering at 1.

    Built from the TRAIN split alone (D-35). It used to include the val slides, which the slide
    head never trains on: those classes were dead weight that inflated log2(N_slides) and made
    retained_bits look better than it was, for slides no discriminator could ever predict.
    """
    s = sorted({x for d in data.values() for k in keys for x in d['img'][k]})
    return {v: i for i, v in enumerate(s)}


# ----------------------------------------------------------------------------- scoring
# macro_f1 moved to metrics.py (plan F6, D-F6) - imported at the top of this file, byte-identical
# (verified: `python metrics.py --selftest`). Every existing `metrics.macro_f1` caller is unaffected.


def baselines(data, rest, held, n_class, drop):
    """Two reference predictors scored exactly like the model, on the same fold and same waiver.

    Gate 3 check 1 compares against these rather than a guessed threshold. MAJORITY predicts the
    commonest cluster in the TRAINING cohorts - it never looks at the held-out cohort. RANDOM is
    kept alongside because macro-F1 can favour a uniform guess over a majority one when classes
    are very unbalanced, so beating majority alone is not sufficient.
    """
    yt = data[held]['y']['test'].cpu().numpy()
    tr = np.concatenate([data[c]['y']['train'].cpu().numpy() for c in rest])
    maj = int(np.bincount(tr, minlength=n_class).argmax())
    f_maj, _ = macro_f1(yt, np.full(len(yt), maj), n_class, drop=drop)
    rnd = _rng('baseline', held).integers(0, n_class, len(yt))
    f_rnd, _ = macro_f1(yt, rnd, n_class, drop=drop)
    return float(f_maj), float(f_rnd), maj


def retained_bits(logits, y, n_slide):
    """retained_bits = log2(N_slides) - CE_bits.  Zero at chance, scale-free across lambda (D-15).

    Accuracy is unusable here: chance on ~900 slides is about 0.1%, no run reaches it, and the
    original gate gave no threshold - so it could have failed an adversary that was working.
    """
    ce = float(F.cross_entropy(logits, y)) / np.log(2.0)
    return float(np.log2(n_slide) - ce), ce


def embed(enc, data, cohorts, split, sidx):
    """Frozen-encoder embeddings for one split, with their slide labels."""
    Z, Y = [], []
    enc.eval()
    with torch.no_grad():
        for c in cohorts:
            d = data[c]
            U = d['U'][split]
            if len(U) == 0:
                continue
            for i in range(0, len(U), 1024):
                Z.append(enc(U[i:i + 1024], d['idx'], d['present']))
            Y.append(torch.tensor([sidx.get(s, -1) for s in d['img'][split]], device=DEV))
    if not Z:
        return None, None
    Z, Y = torch.cat(Z), torch.cat(Y)
    keep = Y >= 0                 # drop cells with no slide class (D-38 nested domain)
    return Z[keep], Y[keep]


def fresh_probe(enc, data, cohorts, sidx, seed=SEED, steps=PROBE_STEPS):
    """Freeze the encoder, train a NEW slide discriminator, score it on CELLS it never saw.

    D-35 - this is the second rewrite of this measurement, and the reason is worth keeping.

    The plan said "score on slides it never saw". That cannot work: the probe is an N-way SLIDE
    classifier, so holding out slide IDs asks it to name classes it has never seen one example
    of. It can never be right, and it was not - fresh_acc came out at exactly 0.000000 on 4 of 4
    smoke folds, with retained_bits at -28.5 against a ceiling of log2(751) = 9.55, a value the
    quantity cannot take. That is the same species of defect D-15 was written to remove.

    What the measurement is actually for is unchanged: a CO-TRAINED discriminator can be beaten
    without the information being gone, because the encoder only has to find a direction that one
    particular discriminator is not looking in. The protection against that is that the probe is
    freshly initialised and never adversarially trained - not that it sees unfamiliar slides. So
    the split moves to CELLS: train on the training draw, score on `sprobe`, which is held out of
    training and drawn from the same slides. If slide identity survives in z_cell, a fresh probe
    recovers it on cells it has never seen.
    """
    Ztr, Ytr = embed(enc, data, cohorts, 'train', sidx)
    Zte, Yte = embed(enc, data, cohorts, 'sprobe', sidx)
    if Ztr is None or Zte is None or len(Ztr) < 100 or len(Zte) < 100:
        return float('nan'), float('nan'), float('nan')

    torch.manual_seed(seed)
    probe = SlideProbe(Ztr.shape[1], len(sidx)).to(Ztr.device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    Ztr = Ztr.detach()
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        b = torch.randint(0, len(Ztr), (min(1024, len(Ztr)),), generator=g).to(Ztr.device)
        opt.zero_grad()
        F.cross_entropy(probe(Ztr[b]), Ytr[b]).backward()
        opt.step()
    probe.eval()
    with torch.no_grad():
        lg = probe(Zte.detach())
        bits, ce = retained_bits(lg, Yte, len(sidx))
        acc = float((lg.argmax(1) == Yte).float().mean())
    return bits, ce, acc


# ----------------------------------------------------------------------------- fit
def fit(data, train_cohorts, lam, n_vocab, n_class, sidx, warm=None, epochs=EPOCHS,
        patience=PATIENCE, seed=SEED, deep_slide=False):
    """Train encoder + cell-type head + two adversarial heads on `train_cohorts`.

    Class-balanced sampling, because imbalance runs to 1,253:1 in CRC and an unbalanced batch
    would let the model score well by never predicting a rare type at all.

    Lambda RAMPS from 0 over the first epochs. Starting at full strength makes the encoder fight a
    discriminator that has not learned anything yet, which is noise rather than signal.
    """
    torch.manual_seed(seed)
    enc = CellEncoder(n_vocab, d_tok=D_TOK, d_z=D_Z, blocks=BLOCKS, heads=HEADS)
    if warm is not None:
        enc.load_stage2(warm)
        # load_stage2() skips a shape mismatch silently (D-39) - check the key tensor explicitly
        if not torch.equal(enc.ident.weight.detach().cpu(), warm['ident.weight'].cpu()):
            raise RuntimeError("warm start: ident.weight did not load - vocabulary mismatch")
    n_coh = len(train_cohorts)
    heads = Heads(D_Z, n_class, len(sidx), n_coh, cohort_frac=COHORT_FRAC,
                  deep_slide=deep_slide)
    enc, heads = enc.to(DEV), heads.to(DEV)
    opt = torch.optim.Adam(list(enc.parameters()) + list(heads.parameters()), lr=LR)
    gen = torch.Generator().manual_seed(seed)
    ci = {c: i for i, c in enumerate(train_cohorts)}

    # class-balanced index per cohort, drawn once
    pools = {}
    for c in train_cohorts:
        y = data[c]['y']['train'].cpu().numpy()
        pools[c] = {int(k): np.flatnonzero(y == k) for k in np.unique(y)}

    best, best_state, bad, used = -np.inf, None, 0, epochs
    for ep in range(epochs):
        lam_ep = lam * min(1.0, (ep + 1) / max(1, RAMP_EPOCHS))
        enc.train(); heads.train()
        for c in [train_cohorts[i] for i in torch.randperm(n_coh, generator=gen).tolist()]:
            d, pool = data[c], pools[c]
            n = d['n']['train']
            ks = list(pool)
            per = max(1, n // max(1, len(ks)))
            take = np.concatenate([_rng('bal', c, ep).choice(pool[k], per,
                                                             replace=len(pool[k]) < per)
                                   for k in ks])
            take = take[_rng('shuf', c, ep).permutation(len(take))]
            # -1 marks a cell with no slide class. Under the nested domain (D-38) that is a cell
            # from a single-slide patient, whose "slide identity" is just patient identity and
            # must not be erased. Those cells still train the cell-type and cohort heads.
            sl = torch.tensor([sidx.get(s, -1) for s in d['img']['train']], device=DEV)
            for i in range(0, len(take), BATCH):
                b = torch.from_numpy(take[i:i + BATCH]).long().to(DEV)
                z = enc(d['U']['train'][b], d['idx'], d['present'])
                lc, ls, lo = heads(z, lam_ep)
                m = sl[b] >= 0
                l_slide = F.cross_entropy(ls[m], sl[b][m]) if bool(m.any()) else ls.sum() * 0.0
                loss = (F.cross_entropy(lc, d['y']['train'][b])
                        + l_slide
                        + F.cross_entropy(lo, torch.full((len(b),), ci[c], device=DEV)))
                opt.zero_grad(); loss.backward(); opt.step()

        # early stopping on held-out SLIDES of the training cohorts, by macro-F1 - the quantity
        # the gate decides on, not the loss, which the adversary makes non-comparable across lambda
        enc.eval(); heads.eval()
        yt, yp = [], []
        with torch.no_grad():
            for c in train_cohorts:
                d = data[c]
                U = d['U']['val']
                for i in range(0, len(U), 1024):
                    yp.append(heads.cls(enc(U[i:i + 1024], d['idx'], d['present'])).argmax(1))
                yt.append(d['y']['val'])
        f1, _ = macro_f1(torch.cat(yt).cpu().numpy(), torch.cat(yp).cpu().numpy(), n_class)
        if f1 > best + 1e-5:
            best, bad = f1, 0
            best_state = ({k: v.detach().clone() for k, v in enc.state_dict().items()},
                          {k: v.detach().clone() for k, v in heads.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                used = ep + 1
                break
    if best_state is not None:
        enc.load_state_dict(best_state[0]); heads.load_state_dict(best_state[1])
    enc.eval(); heads.eval()
    return enc, heads, dict(epochs_used=used, val_f1=round(best, 4))


def predict(enc, heads, d):
    out = []
    with torch.no_grad():
        U = d['U']['test']
        for i in range(0, len(U), 1024):
            out.append(heads.cls(enc(U[i:i + 1024], d['idx'], d['present'])).argmax(1))
    return torch.cat(out).cpu().numpy() if out else np.array([])


# ----------------------------------------------------------------------------- runs
def loco_run(tag, lam, held, train, per, tri2idx, V, L, arm, warm, refit, epochs,
             nested=False, deep_slide=False, space=None):
    """Train on the other training cohorts at this lambda, score on the held-out one."""
    p = os.path.join(CKPT, f'encoder_{tag}.pt')
    space = space or {}
    if os.path.exists(p) and not refit:
        r = torch.load(p, weights_only=False, map_location='cpu')
        # plan F3 / F4: a fold trained on another within-cohort split, or in another label space,
        # is refitted, never reused
        if not splits.split_stale(r) and r.get('space') == space.get('space'):
            print(f"    [cache] {tag}")
            return r
        print(f"    [STALE] {tag}: trained on split {r.get('split_fp')} / space "
              f"{r.get('space')} - refitting")

    t0 = time.time()
    rest = [c for c in train if c != held]
    data = {c: load_cohort(c, per[c], tri2idx, V, arm, None, L) for c in train}
    data = {c: d for c, d in data.items() if d is not None}
    rest = [c for c in rest if c in data]
    if not rest:
        raise ValueError(f"fold '{held}' has no training cohorts left. `train` must stay whole - "
                         f"--folds restricts which cohort is HELD OUT, not which are trained on.")
    if nested:
        sidx, _ = nested_slide_index({c: data[c] for c in rest})
        allsl = slide_index({c: data[c] for c in rest})
        print(f"    nested domain: {len(sidx)} of {len(allsl)} slides share a patient "
              f"({100*len(sidx)/max(1,len(allsl)):.0f}%)")
    else:
        sidx = slide_index({c: data[c] for c in rest})
    if not sidx:
        raise ValueError(f"fold '{held}': empty slide vocabulary - the train draw produced no "
                         f"cells for any of {rest}"
                         + (" that share a patient with another slide (nested domain)."
                            if nested else "."))

    wstate, wname = None, 'cold (--no-warm)'
    if warm is not None:
        wstate, wname, _ = pretrain.warm_for(rest, arm=arm)
    enc, heads, info = fit(data, rest, lam, V, L['n'], sidx, warm=wstate, epochs=epochs,
                           deep_slide=deep_slide)
    info['warm_start'] = wname

    yp = predict(enc, heads, data[held])
    yt = data[held]['y']['test'].cpu().numpy()
    f1_all, per_cls = macro_f1(yt, yp, L['n'])
    f1_core, _ = macro_f1(yt, yp, L['n'], drop=L['unreliable'])       # D-32: waiver honoured
    f1_waived, _ = macro_f1(yt, yp, L['n'],
                            drop=set(range(L['n'])) - L['unreliable'])
    f1_maj, f1_rnd, maj_cls = baselines(data, rest, held, L['n'], L['unreliable'])

    bits, ce, acc = fresh_probe(enc, data, rest, sidx)

    # Gate 3 column 3 - the cohort head, read as an INVERTED GUARD (D-16). Scored on the TRAINING
    # cohorts' held-out slides, because the head only has classes for cohorts it trained on and
    # the held-out cohort is by definition not one of them. If this falls toward chance, tissue
    # identity is being erased and that is a WARNING, not a success: cohort is confounded with
    # tissue on this roster, so a model that cannot tell the cohorts apart cannot tell colon from
    # lung either.
    ci = {c: i for i, c in enumerate(rest)}
    hit = tot = 0
    with torch.no_grad():
        for c in rest:
            d = data[c]
            U = d['U']['val']
            for i in range(0, len(U), 1024):
                pr = heads.cohort(enc(U[i:i + 1024], d['idx'], d['present'])).argmax(1)
                hit += int((pr == ci[c]).sum()); tot += len(pr)
    coh_acc = hit / tot if tot else float('nan')

    # The co-trained discriminator, for the side-by-side gap that shows HIDING from REMOVING.
    # Scored on `sprobe` - the same cells the fresh probe is scored on (D-35). It used to be
    # scored on the val split, whose slides are disjoint from the train slides by construction,
    # so the co-trained head was being asked for classes it had never trained on either. Both
    # columns now answer the same question on the same cells, which is the only way their gap
    # means anything.
    with torch.no_grad():
        zs, ys = [], []
        for c in rest:
            d = data[c]
            U = d['U']['sprobe']
            if len(U) == 0:
                continue
            for i in range(0, len(U), 1024):
                zs.append(heads.slide(enc(U[i:i + 1024], d['idx'], d['present'])))
            ys.append(torch.tensor([sidx.get(s, -1) for s in d['img']['sprobe']], device=DEV))
        if zs:
            zc, yc_ = torch.cat(zs), torch.cat(ys)
            k = yc_ >= 0
            co_bits, co_ce = (retained_bits(zc[k], yc_[k], len(sidx)) if bool(k.any())
                              else (float('nan'), float('nan')))
        else:
            co_bits, co_ce = float('nan'), float('nan')

    r = dict(lam=lam, held=held, f1_core=f1_core, f1_all=f1_all, f1_waived=f1_waived,
             f1_majority=f1_maj, f1_random=f1_rnd, majority_cluster=maj_cls,
             per_cls=per_cls, fresh_bits=bits, fresh_ce=ce, fresh_acc=acc,
             cotrained_bits=co_bits, cohort_acc=coh_acc, n_cohort=len(rest),
             n_slide=len(sidx), n_probe_cells=sum(data[c]['n']['sprobe'] for c in rest),
             info=info, seconds=round(time.time() - t0, 1), split_fp=splits.split_fp(),
             space=space.get('space'), cut_rule=space.get('cut_rule'), n_class=L['n'],
             # The fitted weights, so a fold can be reopened without a 40-minute retrain: to plot
             # its embedding, probe it differently, or check a suspicious number. Stored on CPU so
             # a checkpoint written on a GPU box still loads on this one. ~0.5 MB per fold.
             nested=nested, deep_slide=deep_slide,
             state=dict(enc={k: v.detach().cpu() for k, v in enc.state_dict().items()},
                        heads={k: v.detach().cpu() for k, v in heads.state_dict().items()},
                        slide_index=sidx, cohorts=rest, arm=arm, d_tok=D_TOK, d_z=D_Z,
                        blocks=BLOCKS, heads_n=HEADS, n_vocab=V, n_class=L['n']))
    torch.save(r, p)
    print(f"    [done ] {tag}  F1core={f1_core:.4f} (maj {f1_maj:.4f})  bits={bits:.2f}  "
          f"probe_acc={acc:.4f}  coh={coh_acc:.3f}  {r['seconds']:.0f}s  "
          f"epochs={info['epochs_used']}")
    return r


# ----------------------------------------------------------------------------- gate
def assemble(res, L, lams):
    """Score GATE 3 against panel/gate3_expect.csv. Returns the table and the shipped lambda."""
    d = pd.DataFrame([{k: r[k] for k in
                       ('lam', 'held', 'f1_core', 'f1_all', 'f1_waived', 'f1_majority',
                        'f1_random', 'fresh_bits', 'fresh_acc',
                        'cotrained_bits', 'cohort_acc', 'n_cohort', 'n_slide', 'seconds')}
                      for r in res])
    per_lam = d.groupby('lam').agg(f1_core=('f1_core', 'mean'), f1_maj=('f1_majority', 'mean'),
                                   f1_rnd=('f1_random', 'mean'), fresh=('fresh_bits', 'mean'),
                                   cotr=('cotrained_bits', 'mean'),
                                   coh=('cohort_acc', 'mean')).reset_index()

    # Selection: argmax mean F1, ties to the SMALLER lambda. Fallback to 0.0 is automatic - if no
    # lambda beats it, lambda=0 is itself the argmax.
    best = per_lam.sort_values(['f1_core', 'lam'], ascending=[False, True]).iloc[0]
    ship = float(best.lam)
    w = per_lam[per_lam.lam == ship].iloc[0]

    # check 4's guard is 1.5x chance, and chance is 1/n_cohort within each fold
    chance = float((1.0 / d.n_cohort).mean())
    c1a = float(w.f1_core - w.f1_maj)
    c1b = float(w.f1_core - w.f1_rnd)
    mono = bool((per_lam.sort_values('lam').fresh.diff().dropna() <= 1e-9).all())

    checks = [
        dict(check='1  beats majority-class by >= 0.05', result='PASS' if c1a >= 0.05 else 'FAIL',
             detail=f"shipped F1 {w.f1_core:.4f} vs majority {w.f1_maj:.4f} - margin {c1a:+.4f}"),
        dict(check='1  beats random-uniform', result='PASS' if c1b > 0 else 'FAIL',
             detail=f"shipped F1 {w.f1_core:.4f} vs random {w.f1_rnd:.4f} - margin {c1b:+.4f}"),
        dict(check='2  lambda selection', result='decided',
             detail=(f"ship lambda = {ship:g}" +
                     (" - the FALLBACK: no lambda beat the plain encoder" if ship == 0.0
                      else f" (lambda=0 scored {float(per_lam[per_lam.lam == 0].f1_core.iloc[0]):.4f})"
                      if (per_lam.lam == 0).any() else ""))),
        dict(check='3  retained bits fall with lambda', result='yes' if mono else 'no',
             detail=f"diagnostic only - {' > '.join(f'{v:.2f}' for v in per_lam.sort_values('lam').fresh)}"),
        dict(check='4  cohort guard (INVERTED)',
             result='OK' if w.coh >= 1.5 * chance else 'WARNING',
             detail=(f"cohort accuracy {w.coh:.3f} vs chance {chance:.3f}, guard {1.5*chance:.3f}"
                     + ('' if w.coh >= 1.5 * chance else
                        ' - tissue identity may be being erased, see D-16'))),
        # Sign matters here. cotrained < fresh is the HIDING signature: the co-trained head has
        # been beaten while a fresh probe still recovers the slide, so the information was never
        # removed. cotrained ~ fresh means the adversary removed what it appeared to remove.
        dict(check='5  hiding vs removing', result='diagnostic',
             detail=(f"co-trained {w.cotr:.2f} bits vs fresh {w.fresh:.2f} bits, "
                     f"gap {w.cotr-w.fresh:+.2f}" +
                     (" - lambda=0, so there is NO adversary and this gap is only the linear "
                      "co-trained head being weaker than the 3-layer probe. Not a hiding signal"
                      if ship == 0.0 else
                      " - fresh recovers MORE than the co-trained head, so the adversary is "
                      "hiding rather than removing" if w.cotr < w.fresh - 0.1 else
                      " - the two agree, so what the adversary beat is genuinely gone"))),
    ]
    hard = [c for c in checks if c['check'].startswith('1')]
    verdict = 'PASS' if all(c['result'] == 'PASS' for c in hard) else 'FAIL'
    return d, per_lam, pd.DataFrame(checks), ship, verdict


def write_report(path, d, per_lam, checks, ship, verdict, L, arm, dev, mins,
                 complete=True, n_folds=5, mode='shipped', spaces=None):
    md = lambda x: x.to_markdown(index=False)
    w = per_lam[per_lam.lam == ship].iloc[0]
    if mode == 'fold':
        space_txt = ("Label space: **fold-local** (plan F4) - each fold trains and scores in a space "
                     "built from its training cohorts only (`work/spaces/`, "
                     "`reports/build_fold_label_spaces.md`); NOVEL held-out labels are left out of the "
                     "closed-set macro-F1, and fold class sets differ.\n\n" +
                     md(pd.DataFrame([dict(fold=h, clusters=Lh['n'],
                                           unreliable=len(Lh['unreliable']),
                                           novel_held_out_labels=len(Lh['novel']),
                                           cut_rule=s['cut_rule'])
                                      for h, (Lh, _, s) in (spaces or {}).items()])))
    else:
        space_txt = (f"Label space: the SHIPPED whole-roster space, {L['n']} Stage 1b clusters, "
                     f"{len(L['unreliable'])} flagged unreliable and held out of every headline "
                     f"number (stroma waiver, D-23/D-32). Built WITH every held-out cohort present "
                     f"- the comparison row, not the protocol's headline (plan F4).")
    partial = '' if complete else (
        "\n> **PARTIAL RUN - NOT THE GATE.** This report covers "
        f"{d.lam.nunique()} lambda(s) and {d.held.nunique()} fold(s), not the declared "
        f"{len(LAMBDAS)} x {n_folds}. The verdict line below is arithmetic on what ran, not a gate "
        "result, and must not be recorded as one.\n")
    txt = f"""# Stage 3 - cell encoder + domain-adversarial training (GATE 3)
{partial}
**{verdict}** - shipped lambda = **{ship:g}** - {mins:.1f} min on {dev}.

Arm **{arm}** from Gate 2. {space_txt}

## The gate

{md(checks)}

Thresholds were declared in `celltype_transfer/declared/gate3_expect.csv` before this run. Two rows in that
file are marked REPLACED and kept alongside their replacements: the decision metric (D-31) and the
fresh-probe split (D-35).

## Per lambda

{md(per_lam.round(4))}

`f1_core` is macro-F1 over the reliable clusters, averaged over the {n_folds} LOCO folds. `f1_maj` and
`f1_rnd` are a majority-class and a random-uniform predictor scored on the same folds under the
same waiver - check 1 compares against them rather than a guessed threshold. `fresh` and `cotr`
are retained slide bits: `log2(N_slides) - CE_bits`, zero at chance.

## Every fold

{md(d.round(4))}

## How the slide probe is scored (D-35)

The declared design said "train a new discriminator from scratch, score on slides it never saw".
That is structurally impossible. The probe is an N-way SLIDE classifier, so holding out slide IDs
asks it to name classes it never saw one example of - it can never be right. The first smoke run
measured exactly that: `fresh_acc` 0.000000 on 4 of 4 folds and retained bits of -28.5 against a
ceiling of log2(751) = 9.55, a value the quantity cannot take. The co-trained column had the same
defect more gently, because the within-cohort split makes the train and val slides disjoint (then
`pretrain.slide_split`; since F3, `splits.split_masks` by patient, which keeps them disjoint too).

What the measurement is for is unchanged. A co-trained discriminator can be beaten without the
information being gone - the encoder only has to find a direction that one discriminator is not
looking in. The protection against that is that the probe is **freshly initialised and never
adversarially trained**, not that it meets unfamiliar slides. So the split moved to CELLS: the
probe trains on the training draw and is scored on `sprobe`, {int(d.n_slide.mean())} slides' worth
of cells held out of training and drawn from the same slides. The co-trained head is now scored on
the same cells, so the gap between the two columns compares like with like.

## The cohort guard reads backwards

Check 4 is the one number in this project where FAILING LOW is the bad outcome. Cohort is
confounded with tissue on this roster - colorectal, head and neck, breast, lung, skin twice - so
an embedding that cannot tell the cohorts apart cannot tell colon from lung either, and colon and
lung tumour cells genuinely differ. Driving cohort accuracy to chance would be destruction of
biology reported as success. That is why the cohort head carries only {COHORT_FRAC} of lambda and
why the domain being erased is SLIDE, not cohort (D-16).

Measured at the shipped lambda: **{w.coh:.3f}** against chance {float((1.0/d.n_cohort).mean()):.3f}.
"""
    open(path, 'w', encoding='utf-8').write(txt)
    return path


def available():
    """Cohorts Stage 3 can actually use - the ones with a Stage 2 wide value table.

    Deliberately NOT `vocab.built()`, which tests for work/raw/{c}.parquet. Those are acquisition
    outputs totalling 627 MB and Stage 3 never opens one; it reads {c}_full.parquet only (75 MB
    for all five). Testing raw availability would make a GPU box carry eight times the data to
    satisfy an existence check for files it does not use.
    """
    return [c for c in SPECS if os.path.exists(config.full_table(c))]


def main():
    cohorts = available()
    # D-39: the vocabulary comes from panel.json, never from whichever parquet files are on disk.
    # See vocab.read_panel() for what that silently broke.
    triples, tri2idx, per, genes, ncoh = vocab.read_panel(cohorts)
    V = len(triples)
    train = [c for c in cohorts if SPECS[c]['role'] == 'train']
    L = label_space()

    print(f"label space: {L['n']} Stage 1b clusters · {len(L['unreliable'])} flagged unreliable "
          f"(stroma waiver, D-23)")
    print(f"vocabulary: {V} triples · {len(train)} training cohorts")
    print(f"device: {DEV}" + (f" ({torch.cuda.get_device_name(0)})" if DEV.type == 'cuda' else ''))

    if '--lambda-sweep' not in sys.argv:
        print("\nnothing to do. run with --lambda-sweep for GATE 3.")
        return

    quick = '--quick' in sys.argv
    epochs = 2 if quick else EPOCHS
    q = 'quick_' if quick else ''
    lams = LAMBDAS
    if '--lambdas' in sys.argv:
        lams = [float(x) for x in sys.argv[sys.argv.index('--lambdas') + 1].split(',')]
    # --folds lets the 25 runs be split across sessions (a Kaggle timeout, or one fold timed on
    # CPU for a real budget). It restricts WHICH COHORT IS HELD OUT - `train` itself must stay
    # whole, or the remaining cohorts to train on vanish with it. The gate is assembled only from
    # the folds present, so a partial run reports a partial table rather than pretending to be
    # complete.
    # STAGE 3b (D-37, D-38). Both default OFF so the Gate 3 result reproduces byte for byte, and
    # both change the tag so a 3b fold can never be mistaken for, or cached over, a Gate 3 fold.
    nested = '--nested' in sys.argv
    deep_slide = '--deep-slide' in sys.argv
    tagpfx = ('n' if nested else '') + ('d' if deep_slide else '')
    tagpfx = f'{tagpfx}_' if tagpfx else ''
    if nested:
        print("  --nested: adversary domain is SLIDE WITHIN PATIENT (D-38). Cells from "
              "single-slide patients take no part in the adversarial loss.")
    if deep_slide:
        print("  --deep-slide: slide head matched to SlideProbe's capacity (D-37).")

    folds = list(train)
    if '--folds' in sys.argv:
        pick = set(sys.argv[sys.argv.index('--folds') + 1].split(','))
        folds = [c for c in train if c in pick]
        print(f"  --folds: holding out only {folds} (still training on every other cohort)")
    if quick:
        global N_TRAIN, SCORE_CELLS
        N_TRAIN, SCORE_CELLS = 1_500, 800
        print("\n*** --quick: 2 epochs on a tiny draw. Proves the code path; scores nothing. ***")

    arm = json.load(open(os.path.join(WORK, 'panel.json'))).get('stage2_arm', 'set')
    # FOLD-LOCAL WARM START (2026-09-11, plan F4). This loaded pretrain_armB.pt ONCE and handed it to
    # every fold - a model trained on all 7 cohorts, so the fold holding out X started from an
    # encoder that had already seen X. Each fold now asks pretrain.warm_for() inside loco_run() for the
    # Stage 2 model trained only on its own training cohorts, and a fold with none raises.
    warm = None if '--no-warm' in sys.argv else 'fold-local'
    print(f"warm start: {warm or 'cold (--no-warm)'}")

    # LABEL SPACE PER FOLD (plan F4) - same rule and flag as Stage 6 (see space_for)
    mode = space_mode()
    spaces = {h: space_for(h, mode) for h in folds}
    # `b3pfx` = the Stage 3b arm alone (d_/n_/nd_), which decides the REPORT name below. The fold
    # prefix goes into checkpoint tags only. First Kaggle run (2026-09-12) folded 'fold_' into the
    # one prefix, so the name logic read every fold-space Gate 3 as a Stage 3b arm and wrote
    # s3b_fold_encoder.md; the notebook's collect cell then found no train_adversarial_encoder.md and crashed
    # before zipping. Checkpoint tags are unchanged by the fix, so those fits load from cache.
    b3pfx = tagpfx
    if mode == 'fold':
        tagpfx = f'fold_{tagpfx}'
    print(f"label space: {mode}" + ''.join(
        f"\n    {h:9} {Lh['n']:>3} clusters, {len(Lh['novel'])} NOVEL held-out labels "
        f"({s['cut_rule']})" for h, (Lh, _, s) in spaces.items()))

    print(f"\nlambda sweep {lams} x {len(folds)} LOCO folds = {len(lams)*len(folds)} runs")
    t0 = time.time()
    res = []
    for lam in lams:
        print(f"\n  lambda = {lam}")
        for h in folds:
            Lh, _, s = spaces[h]
            res.append(loco_run(f'{q}{tagpfx}lam{lam}_{h}', lam, h, train, per, tri2idx, V, Lh,
                                arm, warm, '--refit' in sys.argv, epochs,
                                nested=nested, deep_slide=deep_slide, space=s))
    # The aggregate is a summary, so the per-fold weights are stripped out of it - they already
    # live in their own s3_lam*_*.pt files and would make this file 25x bigger for nothing.
    light = [{k: v for k, v in r.items() if k != 'state'} for r in res]
    torch.save(light, os.path.join(CKPT, f'encoder_{q}{tagpfx}sweep.pt'))
    print(f"\nwrote {os.path.join(CKPT, f'encoder_{q}{tagpfx}sweep.pt')}")

    d, per_lam, checks, ship, verdict = assemble(res, L, lams)
    print(f"\n--- GATE 3 ---\n{checks.to_string(index=False)}")
    print(f"\n{per_lam.round(4).to_string(index=False)}")
    if quick:
        print("\n--quick: these numbers score nothing. No report written.")
        return
    # Stage 3b is a different experiment, not a re-score of Gate 3, so it never overwrites the
    # gate report even when it runs the full grid.
    # Was `len(folds) == len(train) == 5` - the 5+1 roster's fold count, hard-coded. On the
    # 7-cohort roster no run could ever be complete, so every Gate 3 report would have been
    # labelled PARTIAL (found 2026-09-11, plan F4). Every training cohort is a fold.
    complete = (sorted(lams) == sorted(LAMBDAS) and len(folds) == len(train))
    # tagpfx is in the NAME, not just the body: the three Stage 3b arms (d_, n_, nd_) each write
    # their own report, or they would overwrite one another and the ablation would be unreadable.
    name = (f'train_adversarial_encoder_variant_{b3pfx}{"_PARTIAL" if not complete else ""}.md' if b3pfx else
            'train_adversarial_encoder.md' if complete else 'train_adversarial_encoder_PARTIAL.md')
    if mode == 'shipped':
        name = name.replace('.md', '_shipped_space.md')
    p = write_report(os.path.join(REPORTS, name), d, per_lam, checks, ship, verdict,
                     L, arm, DEV, (time.time() - t0) / 60, complete, n_folds=len(train),
                     mode=mode, spaces=spaces)
    print(f"\n{verdict if complete else 'PARTIAL (not the gate)'} · "
          f"ship lambda={ship:g} · wrote {p}")


if __name__ == "__main__":
    main()

"""
Spatial context (GATE 4) - does a cell's neighbourhood improve cross-cohort transfer?

    python train_spatial_context.py --pilot      # 1 fold x 3 arms, a signal, not a gate
    python train_spatial_context.py --gate       # GATE 4: 4 arms x 7 LOCO folds + 2 rescale fits
    python train_spatial_context.py --gate --quick            # smoke test: proves the code path, scores nothing
    python train_spatial_context.py --gate --refit            # ignore the checkpoint cache
    python train_spatial_context.py --gate --folds Keren      # one fold
    python train_spatial_context.py --gate --space shipped    # comparison row (default is fold-local, plan F4)

Needs the neighbour sidecars from build_neighbour_graph.py (see that file for why the graph is
built from the FULL raw tables and never from the 40,000-cell value tables).

THE FOUR ARMS, and why each one exists.

    cell      the cell's own markers only. NOT read from Gate 6's report - re-measured inside this
              run, because Gate 6's 0.3151 was produced by a different model (no fusion path) and
              comparing across runs would change two things at once (D-27's discipline).
    neigh     + the 15 nearest neighbours, pooled, through the same encoder.
    shuffle   + ANOTHER CELL'S neighbours, drawn from the same image. The control the design names
              by name: if shuffling does not hurt, the gain is not spatial.
    ctx       neigh + a 4th loss predicting the pooled neighbourhood from z_cell alone. This is
              the loss gate6_expect.csv recorded as unwritable because Stage 4 did not exist
              (D-40), so the declared "2 vs 4 losses" ablation becomes real here for the first
              time. Declared required=0: the first arm to drop if the week runs short.
"""
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # runs from any directory
import config
import panel_utils
from config import SPECS, WORK, VALUES, REPORTS, FIGURES, SEED, raw_table
from metrics import paired   # one definition of the paired CI / sign-flip test
from build_neighbour_graph import (K, MIN_D, DIST_LO, DIST_HI, TRAIN, assert_roster, nbr_path,
                                   registry)
from models.encoder import CellEncoder
from models.losses import (Prototypes, CollapseGuard, KendallSigma, vicreg, masked_marker_loss,
                       confidence_weighted_ce)
import splits
import build_marker_vocabulary as vocab
import metrics
import pretrain_masked_markers as pretrain
import train_adversarial_encoder as adversarial
import train_prototype_classifier as classifier

CKPT = os.path.join(WORK, 'ckpt')
os.makedirs(CKPT, exist_ok=True)
from models.device import DEV   # one rule for all stages; --cpu forces the CPU

# The four arms. Order matters: if a session dies the REQUIRED arms are already on disk.
# `ctx` is gate4_expect.csv check 4, required=0 - the first thing to drop if the week runs short.
ARMS = [('cell',    False, False),      # (name, uses the neighbourhood, uses the context loss)
        ('neigh',   True,  False),
        ('shuffle', True,  False),      # same inputs as `neigh`, correspondence destroyed
        ('ctx',     True,  True)]

N_GEO = 7                     # the per-cell context vector: 6 from nbr_feats, 1 from attach_nbr
DERANGE_MIN = 0.99            # check 8: the shuffle must actually move ~every cell
PX_SCALES = (1.3, 0.77)       # check 5: UPMC's px_um is ASSUMED (M4)


from config import rng as _rng   # one definition, config.rng


# --------------------------------------------------------------------------- features
def nbr_feats(c, tri2idx, n_vocab, px_scale=1.0):
    """The sidecar as two per-cell arrays: a pooled neighbourhood profile and a geometry vector.

    PROFILE - the mean u_coh of a cell's valid neighbours, scattered into the canonical vocabulary
    slots. It is fed through THE SAME encoder as the cell itself, as a synthetic 'average
    neighbour', so the unmeasured slots get the same [ABSENT] token and nothing is zero-filled
    (idea 2 in CLAUDE.md). Averaging over neighbours is the plainest pooling there is; anything
    smarter would put a second untested moving part inside the one comparison this gate makes.

    WHY THE MEAN IS NOT ENOUGH, and what is added beside it. An average over 15 neighbours
    destroys the two facts that make a neighbourhood informative in the first place:

      - "is there a VESSEL touching me" is a MAX question. One CD31-high neighbour out of 15
        barely moves the mean.
      - "am I at the TUMOUR BOUNDARY" is a VARIANCE question. A cell with 7 tumour and 8 immune
        neighbours has almost the same average as a cell in a uniformly mixed region, and the two
        are completely different biology.

    A model that only sees the mean can express neither, so a FAILED gate would not distinguish
    "the neighbourhood carries nothing" from "the pooling threw it away". Two scalars close that
    gap for almost no cost - `het` here and `d_self` in attach_nbr(). Both are averaged over the
    markers a cohort actually measured, so a 17-marker cohort and a 57-marker cohort produce
    comparable numbers. Attention over the 15 neighbours would be more expressive still and is
    deliberately NOT done: it would cost 2-3x the compute and make a failure unattributable,
    which is the one thing this gate exists to avoid.

    CONTEXT VECTOR - six scalars here, the seventh added by attach_nbr():
        0  log of the median edge length in MICRONS   <- the ONLY feature a wrong px_um moves,
                                                         which is what makes check 5 a measurement
                                                         rather than an assertion
        1  median edge length / the image's median nearest-neighbour distance (scale-free)
        2  anisotropy of the neighbour offsets, 0 = blob, 1 = collinear (vessels)
        3  1 where anisotropy is defined, 0 where the neighbourhood was too small
        4  fraction of the k slots that hold a real neighbour
        5  het - how much the neighbours DISAGREE with each other: the spread across neighbours,
           averaged over measured markers. Uniform nest vs mixed boundary zone.

    WHAT IS DELIBERATELY NOT A FEATURE: `nbr_same`, the homotypic fraction. It is built from the
    neighbours' NATIVE LABELS. On the held-out cohort those labels are the thing being predicted,
    so feeding them in would hand the model a smoothed copy of the answer. It is the sharpest
    single number in the sidecar and it is exactly the one that cannot be used. It stays in the
    npz as a descriptive statistic for the build report only.
    """
    z = np.load(nbr_path(c), allow_pickle=False)
    tri = [str(t) for t in z['triples']]
    keep = [(i, tri2idx[t]) for i, t in enumerate(tri) if t in tri2idx]
    src = np.array([i for i, _ in keep])
    dst = np.array([j for _, j in keep])

    valid = z['nbr_valid']                                   # (N, k)
    nv = np.maximum(valid.sum(1), 1).astype('float32')
    nu = z['nbr_u'][:, :, src].astype('float32')             # (N, k, M_kept)
    w = valid[:, :, None]
    mu = (nu * w).sum(1) / nv[:, None]                       # (N, M_kept)
    prof = np.zeros((len(valid), n_vocab), 'float32')
    prof[:, dst] = mu
    # E[x^2] - E[x]^2 rather than (nu - mu)**2, which would allocate a second copy of a tensor
    # that is already 134 MB on CRC. Clipped at 0: the identity is exact in real arithmetic and
    # slightly negative in float32 when a marker is constant across the neighbourhood.
    var = (nu ** 2 * w).sum(1) / nv[:, None] - mu ** 2
    het = np.sqrt(np.maximum(var, 0.0)).mean(1)

    d = z['nbr_d'] * float(px_scale)
    rel = z['nbr_rel']
    big = np.where(valid, d, np.nan)
    rbig = np.where(valid, rel, np.nan)
    with np.errstate(invalid='ignore'):
        med_d = np.nanmedian(big, axis=1)
        med_r = np.nanmedian(rbig, axis=1)
    aniso = z['aniso']
    geo = np.stack([np.log(np.maximum(np.nan_to_num(med_d, nan=MIN_D), MIN_D)),
                    np.nan_to_num(med_r, nan=0.0),
                    np.nan_to_num(aniso, nan=0.0),
                    np.isfinite(aniso).astype('float32'),
                    valid.mean(1).astype('float32'),
                    het.astype('float32')], axis=1).astype('float32')
    return z['cell_id'], prof, geo


def within_image_derangement(img, key):
    """Permute cells WITHIN each image, never across it.

    Across images the control would also destroy slide identity, and would then be testing two
    things at once (gate4_expect.csv check 2). A random cyclic shift by a non-zero offset is used
    rather than a free permutation because it is a derangement BY CONSTRUCTION for every image
    holding >= 2 drawn cells - a free permutation can return near-identity, which is D-45's
    failure (two arms that were secretly the same computation) wearing a different hat.
    """
    perm = np.arange(len(img))
    order = np.argsort(img, kind='stable')
    simg = img[order]
    bounds = np.flatnonzero(np.r_[True, simg[1:] != simg[:-1], True])
    rng = _rng(*key)
    for a, b in zip(bounds[:-1], bounds[1:]):
        g = order[a:b]
        if len(g) >= 2:
            perm[g] = np.roll(g, int(rng.integers(1, len(g))))
    return perm


def attach_nbr(data, tri2idx, n_vocab, shuffle, px_scales=None):
    """Hang the neighbourhood tensors on Stage 6's data dicts, aligned by cell_id.

    Returns the measured derangement fraction (check 8). The shuffle permutes the profile AND the
    geometry together: a neighbour list is one object, and permuting only the profile would leave
    each cell its own real distances, so the control would still carry part of the signal it
    exists to remove.

    The 7th context scalar is added HERE and not in nbr_feats() because it is the only one that
    pairs the cell with its neighbourhood rather than describing either alone:

        d_self - how far this cell's own marker profile sits from its neighbourhood's average,
                 over the markers this cohort measured. A BOUNDARY DETECTOR: near zero inside a
                 uniform nest, large for a cell sitting against a different tissue compartment.

    It is computed AFTER the shuffle on purpose. In the shuffled arm the cell keeps its own
    markers but gets someone else's neighbourhood, so d_self becomes meaningless - which is
    exactly what the control is for. Computing it before the permutation would smuggle a real
    spatial measurement into the arm that is supposed to have none.
    """
    px_scales = px_scales or {}
    deranged = []
    for c, d in data.items():
        cid, prof, geo = nbr_feats(c, tri2idx, n_vocab, px_scales.get(c, 1.0))
        pos = pd.Index(cid).get_indexer(np.concatenate([d['cid'][k] for k in ('train',
                                                                              'val', 'test')]))
        assert (pos >= 0).all(), f"{c}: drawn cells missing from the neighbour sidecar"
        d['prof'], d['geo'] = {}, {}
        off = 0
        for k in ('train', 'val', 'test'):
            n = d['n'][k]
            p = pos[off:off + n]
            off += n
            P, G = prof[p], geo[p]
            if shuffle:
                q = within_image_derangement(d['img'][k], ('shuffle', c, k))
                deranged.append(float((q != np.arange(len(q))).mean()))
                P, G = P[q], G[q]
            Pt = torch.from_numpy(np.ascontiguousarray(P)).to(DEV)
            Gt = torch.from_numpy(np.ascontiguousarray(G)).to(DEV)
            # restricted to the measured slots: everywhere else both sides are a structural zero,
            # so including them would just divide by the size of the panel and make a 17-marker
            # cohort's number incomparable with a 57-marker one's
            pres = d['present']
            ds = (Pt[:, pres] - d['U'][k][:, pres]).abs().mean(1, keepdim=True)
            d['prof'][k] = Pt
            d['geo'][k] = torch.cat([Gt, ds], dim=1)
    return float(np.mean(deranged)) if deranged else 1.0


# ------------------------------------------------------------------------------ model
class Stage4Net(nn.Module):
    """Stage 6's model with one addition: the neighbourhood enters as a residual on z_cell.

    WHY A RESIDUAL WITH A ZERO-INITIALISED GATE, and not a concatenation. Three reasons, all of
    them about keeping check 1 a fair comparison:

      - concatenating [z_cell, z_neigh] doubles the width the prototypes live in, so Stage 6's
        prototype initialisation (each cluster's Stage 1b signature pushed through the encoder)
        would no longer fit and the spatial arms would start from random prototypes while the
        control started from informed ones. That difference alone could produce the margin.
      - with `fuse`'s last layer initialised to zero, arm `neigh` at epoch 0 is BIT-IDENTICAL to
        arm `cell`. The neighbourhood has to earn every point of the margin; it cannot win by
        being a different model.
      - the neighbourhood profile goes through the SAME encoder as the cell, so no new marker
        pathway is introduced and the [ABSENT] token still covers unmeasured slots.
    """

    def __init__(self, n_vocab, n_class, use_nbr, use_ctx, proto_init=None):
        super().__init__()
        self.enc = CellEncoder(n_vocab, d_tok=classifier.D_TOK, d_z=classifier.D_Z,
                               blocks=classifier.BLOCKS, heads=classifier.HEADS)
        self.proto = Prototypes(n_class, classifier.D_Z, init=proto_init, temp=classifier.PROTO_TEMP)
        self.mask_head = nn.Linear(classifier.D_TOK, 1)
        self.use_nbr, self.use_ctx = use_nbr, use_ctx
        if use_nbr:
            self.geo = nn.Sequential(nn.Linear(N_GEO, 32), nn.GELU(), nn.Linear(32, classifier.D_Z))
            self.fuse = nn.Sequential(nn.Linear(2 * classifier.D_Z, classifier.D_Z), nn.GELU(),
                                      nn.Linear(classifier.D_Z, classifier.D_Z))
            nn.init.zeros_(self.fuse[-1].weight)
            nn.init.zeros_(self.fuse[-1].bias)
        # the context auxiliary loss predicts the POOLED NEIGHBOURHOOD PROFILE from z_cell alone,
        # so the neighbourhood has to be pushed into the cell's own vector instead of riding
        # alongside it. This is the 4th loss gate6_expect.csv recorded as unwritable (D-40).
        self.ctx_head = nn.Linear(classifier.D_Z, n_vocab) if use_ctx else None

    def forward(self, u, idx, present, prof=None, geo=None, hide=None):
        z_cell, tok = self.enc(u, idx, present, hide=hide, return_tokens=True)
        z = z_cell
        if self.use_nbr:
            z_n = self.enc(prof, idx, present)
            z = z_cell + self.fuse(torch.cat([z_n, self.geo(geo)], dim=-1))
        ctx = self.ctx_head(z_cell) if self.use_ctx else None
        return z, self.proto.logits(z), self.mask_head(tok).squeeze(-1), ctx


def ctx_loss(pred, target, present):
    """Mean squared error on the MEASURED slots only.

    An unmeasured slot's pooled profile is a structural zero, not a measurement of zero, so
    scoring it would train the model to predict the panel rather than the neighbourhood.
    """
    m = present.view(1, -1).expand_as(pred)
    return ((pred - target) ** 2 * m).sum() / m.sum().clamp(min=1)


# -------------------------------------------------------------------------------- fit
def fit(data, train_cohorts, L, V, use_nbr, use_ctx, epochs=None, patience=None, seed=SEED,
        proto_path=None):
    """Stage 6's training loop with the neighbourhood tensors threaded through.

    It is a near-copy of classifier.fit rather than a refactor of it ON PURPOSE. classifier.fit produced Gate 6's
    shipped 0.3151 and 23 cached checkpoints; changing its signature to carry two optional tensors
    would put every one of those numbers at risk to save forty lines. Everything that decides a
    result - the losses, the sigma weighting, the collapse guard, the warm start, the prototype
    initialisation, the early-stopping rule - is IMPORTED from s6, not re-implemented here.
    """
    epochs = classifier.EPOCHS if epochs is None else epochs
    patience = classifier.PATIENCE if patience is None else patience
    torch.manual_seed(seed)
    m = Stage4Net(V, L['n'], use_nbr, use_ctx).to(DEV)
    warm, wname = classifier.warm_state(train_cohorts)
    if warm is not None:
        m.enc.load_stage2(warm)
        if not torch.equal(m.enc.ident.weight.detach().cpu(), warm['ident.weight']):
            raise RuntimeError(f"warm start {wname}: ident.weight did not load")
    # THE FOLD'S OWN prototypes, not work/prototypes.npy. The whole-roster file has 21 clusters
    # and a fold-local space has its own count, so passing no path makes init_prototypes fall
    # back to random - which would start every arm here from random prototypes while Gate 6
    # started from Stage 1b's signatures, and would make check 1's margin unreadable. Caught by
    # the --quick smoke run printing "prototypes.npy is (21, 109), expected (20, 109)".
    pi = classifier.init_prototypes(m.enc, L['n'], V, path=proto_path)
    if pi is None:
        raise RuntimeError(f"no usable prototypes for a {L['n']}-cluster space "
                           f"(path={proto_path}) - refusing to start from random ones")
    with torch.no_grad():
        m.proto.p.copy_(pi)
        m.proto.p0.copy_(pi)

    aux = ['mask', 'vicreg'] + (['ctx'] if use_ctx else [])
    sig = KendallSigma(aux).to(DEV)
    guard = CollapseGuard(m.proto, classifier.COLLAPSE_FRAC)
    opt = torch.optim.Adam(list(m.parameters()) + list(sig.parameters()), lr=classifier.LR)
    gen = torch.Generator().manual_seed(seed)

    pools = {c: {int(k): np.flatnonzero(data[c]['y']['train'].cpu().numpy() == k)
                 for k in np.unique(data[c]['y']['train'].cpu().numpy())}
             for c in train_cohorts}

    best, best_state, bad, used = -np.inf, None, 0, epochs
    for ep in range(epochs):
        m.train()
        for c in [train_cohorts[i] for i in torch.randperm(len(train_cohorts),
                                                           generator=gen).tolist()]:
            d, pool = data[c], pools[c]
            ks = list(pool)
            per = max(1, d['n']['train'] // max(1, len(ks)))
            take = np.concatenate([_rng('bal', c, ep).choice(pool[k], per,
                                                             replace=len(pool[k]) < per)
                                   for k in ks])
            take = take[_rng('shuf', c, ep).permutation(len(take))]
            for i in range(0, len(take), classifier.BATCH):
                b = torch.from_numpy(take[i:i + classifier.BATCH]).long().to(DEV)
                u = d['U']['train'][b]
                pr = d['prof']['train'][b] if use_nbr else None
                gg = d['geo']['train'][b] if use_nbr else None
                hide = classifier.make_hide(len(b), d['elig'], gen)
                z, logits, pred, cx = m(u, d['idx'], d['present'], prof=pr, geo=gg, hide=hide)

                w = d['conf']['train'][b] if d['has_conf'] else None
                l_cls = confidence_weighted_ce(logits, d['y']['train'][b], w)
                losses = {'mask': masked_marker_loss(pred, u, hide), 'vicreg': vicreg(z)}
                if use_ctx:
                    losses['ctx'] = ctx_loss(cx, d['prof']['train'][b], d['present'])
                l_aux, _ = sig(losses)
                loss = l_cls + l_aux
                opt.zero_grad(); loss.backward(); opt.step()
        guard.step(ep)

        m.eval()
        yt, yp = [], []
        with torch.no_grad():
            for c in train_cohorts:
                d = data[c]
                U = d['U']['val']
                for i in range(0, len(U), 1024):
                    sl = slice(i, i + 1024)
                    _, lg, _, _ = m(U[sl], d['idx'], d['present'],
                                    prof=d['prof']['val'][sl] if use_nbr else None,
                                    geo=d['geo']['val'][sl] if use_nbr else None)
                    yp.append(lg.argmax(1))
                yt.append(d['y']['val'])
        f1, _ = metrics.macro_f1(torch.cat(yt).cpu().numpy(), torch.cat(yp).cpu().numpy(), L['n'])
        if f1 > best + 1e-5:
            best, bad = f1, 0
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                used = ep + 1
                break
    if best_state is not None:
        m.load_state_dict(best_state)
    m.eval()
    return m, dict(epochs_used=used, val_f1=round(best, 4),
                   warm_start=wname or 'cold (--no-warm)',
                   sigma_names=aux, guard_passed=guard.passed())


def predict(m, d, use_nbr):
    out = []
    with torch.no_grad():
        U = d['U']['test']
        for i in range(0, len(U), 1024):
            sl = slice(i, i + 1024)
            _, lg, _, _ = m(U[sl], d['idx'], d['present'],
                            prof=d['prof']['test'][sl] if use_nbr else None,
                            geo=d['geo']['test'][sl] if use_nbr else None)
            out.append(lg.argmax(1))
    return torch.cat(out).cpu().numpy() if out else np.array([])


def gate_run(tag, held, train, per, tri2idx, triples, L, excl, use_nbr, use_ctx, shuffle,
             refit, epochs, space=None, px_scales=None, proto_path=None):
    p = os.path.join(CKPT, f'spatial_{tag}.pt')
    space = space or {}
    if os.path.exists(p) and not refit:
        r = torch.load(p, weights_only=False, map_location='cpu')
        if not splits.split_stale(r) and r.get('space') == space.get('space'):
            print(f"    [cache] {tag}")
            return r
        print(f"    [STALE] {tag}: another split or label space - refitting")

    t0 = time.time()
    V = len(triples)
    rest = [c for c in train if c != held]
    data = {c: classifier.load_cohort(c, per[c], tri2idx, V, L, excl) for c in train}
    data = {c: d for c, d in data.items() if d is not None}
    rest = [c for c in rest if c in data]
    derange = attach_nbr(data, tri2idx, V, shuffle, px_scales) if use_nbr else 1.0

    m, info = fit(data, rest, L, V, use_nbr, use_ctx, epochs=epochs, proto_path=proto_path)
    yp = predict(m, data[held], use_nbr)
    yt = data[held]['y']['test'].cpu().numpy()
    f1_all, _ = metrics.macro_f1(yt, yp, L['n'])
    f1_core, _ = metrics.macro_f1(yt, yp, L['n'], drop=L['unreliable'])
    f1_maj, f1_rnd, _ = adversarial.baselines(data, rest, held, L['n'], L['unreliable'])

    r = dict(tag=tag, held=held, use_nbr=use_nbr, use_ctx=use_ctx, shuffle=shuffle,
             f1_core=f1_core, f1_all=f1_all, f1_majority=f1_maj, f1_random=f1_rnd,
             derangement=round(derange, 4), info=info, n_cohort=len(rest),
             split_fp=splits.split_fp(), space=space.get('space'), cut_rule=space.get('cut_rule'),
             n_class=L['n'], px_scales=px_scales or {}, seconds=round(time.time() - t0, 1),
             # The fitted weights (added 2026-09-17). The first Gate 4 run saved none, so its 30 fits
             # could never be re-scored or reused; additive - nothing that is scored changes.
             state={k: v.detach().cpu() for k, v in m.state_dict().items()})
    torch.save(r, p)
    print(f"    [done ] {tag}  F1core={f1_core:.4f} (maj {f1_maj:.4f})  "
          f"{r['seconds']:.0f}s  epochs={info['epochs_used']}")
    return r


# ------------------------------------------------------------------------------- gate


def assemble(res, folds):
    d = pd.DataFrame([{k: r[k] for k in ('tag', 'held', 'use_nbr', 'use_ctx', 'shuffle',
                                         'f1_core', 'f1_all', 'f1_majority', 'derangement',
                                         'seconds')} for r in res])
    d['arm'] = [arm_of(r) for r in res]
    base = d[~d.tag.str.startswith('px')]
    piv = base.pivot_table(index='held', values='f1_core', columns='arm', aggfunc='first')
    piv = piv.reindex([f for f in folds if f in piv.index])

    def mean(a):
        return float(piv[a].mean()) if a in piv else float('nan')

    m = {a: mean(a) for a, _, _ in ARMS}
    have = [a for a, _, _ in ARMS if a in piv and piv[a].notna().all()]

    checks, stats_rows = [], {}
    if 'neigh' in have and 'cell' in have:
        st = paired(piv['neigh'] - piv['cell'])
        stats_rows['neigh - cell'] = st
        checks.append(dict(check='1  neigh beats cell', result='PASS' if st['mean'] > 0 else 'FAIL',
                           detail=f"neigh {m['neigh']:.4f} vs cell {m['cell']:.4f} "
                                  f"- delta {st['mean']:+.4f}"))
    if 'neigh' in have and 'shuffle' in have:
        st = paired(piv['neigh'] - piv['shuffle'])
        stats_rows['neigh - shuffle'] = st
        checks.append(dict(check='2  neigh beats the shuffled control',
                           result='PASS' if st['mean'] > 0 else 'FAIL',
                           detail=f"neigh {m['neigh']:.4f} vs shuffle {m['shuffle']:.4f} "
                                  f"- delta {st['mean']:+.4f}"))
        sc, nc = m['shuffle'] - m['cell'], m['neigh'] - m['cell']
        checks.append(dict(check='2b shuffle falls back toward cell',
                           result='PASS' if abs(sc) < nc else 'FAIL',
                           # abs() written out rather than with pipe bars: the detail column is
                           # rendered inside a markdown table, and a bare | splits the cell.
                           detail=f"cell {m['cell']:.4f} <= shuffle {m['shuffle']:.4f} "
                                  f"< neigh {m['neigh']:.4f}; abs(shuffle-cell) {abs(sc):.4f} "
                                  f"vs (neigh-cell) {nc:+.4f}"))
    if 'neigh - cell' in stats_rows:
        st = stats_rows['neigh - cell']
        spans = st['lo'] <= 0 <= st['hi']
        checks.append(dict(check='3  paired interval on (neigh - cell)',
                           result='spans zero' if spans else 'excludes zero',
                           detail=f"mean {st['mean']:+.4f}, 95% CI "
                                  f"[{st['lo']:+.4f}, {st['hi']:+.4f}], "
                                  f"exact sign-flip p = {st['p']:.3f}, n = {st['n']}"))
    if 'ctx' in have and 'neigh' in have:
        st = paired(piv['ctx'] - piv['neigh'])
        stats_rows['ctx - neigh'] = st
        checks.append(dict(check='4  the context loss earns its place (not required)',
                           result='ship ctx' if st['mean'] > 0 else 'drop ctx',
                           detail=f"ctx {m['ctx']:.4f} vs neigh {m['neigh']:.4f} "
                                  f"- delta {st['mean']:+.4f}"))

    px = d[d.tag.str.startswith('px')]
    if len(px) and 'cell' in piv and 'UPMC' in piv.index:
        base_u = float(piv.loc['UPMC', 'cell'])
        rows, moved = [], []
        for r in res:
            if not r['tag'].startswith('px'):
                continue
            alt = piv['neigh'] - piv['cell']
            alt = alt.copy()
            alt.loc['UPMC'] = r['f1_core'] - base_u
            rows.append((r['px_scales'].get('UPMC'), r['f1_core'], float(alt.mean())))
            moved.append(float(alt.mean()) > 0)
        same = all(moved) == (stats_rows.get('neigh - cell', {}).get('mean', 0) > 0)
        checks.append(dict(check='5  verdict survives UPMC px_um x1.3 and x0.77',
                           result='PASS' if same else 'FAIL',
                           detail='; '.join(f"x{s}: UPMC neigh {f:.4f}, check 1 delta {a:+.4f}"
                                            for s, f, a in rows)))

    der = base[base.shuffle].derangement
    if len(der):
        checks.append(dict(check='8  the shuffle is a real derangement',
                           result='PASS' if der.min() >= DERANGE_MIN else 'FAIL',
                           detail=f"{der.min():.4f} of cells moved, floor {DERANGE_MIN}"))

    hard = [c for c in checks if c['check'][0] in '125' or c['check'][:2] == '2b'
            or c['check'][0] == '8']
    verdict = 'PASS' if hard and all(c['result'] == 'PASS' for c in hard) else 'FAIL'
    return d, piv, checks, verdict, m, stats_rows


def arm_of(r):
    if not r['use_nbr']:
        return 'cell'
    if r['shuffle']:
        return 'shuffle'
    return 'ctx' if r['use_ctx'] else 'neigh'


def figure(piv):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(FIGURES, exist_ok=True)
    if 'neigh' not in piv or 'cell' not in piv:
        return {}
    dl = (piv['neigh'] - piv['cell']).sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(dl.index, dl.values, color=['crimson' if v < 0 else 'steelblue' for v in dl.values])
    ax.axvline(0, c='k', lw=1)
    ax.set_xlabel('macro-F1(neigh) - macro-F1(cell), per LOCO fold')
    ax.set_title('Gate 4 check 1, fold by fold. A mean built from one fold is not a result.',
                 fontsize=9)
    fig.tight_layout()
    p = os.path.join(FIGURES, 'spatial_delta.png')
    fig.savefig(p, dpi=130); plt.close(fig)
    return {'delta': p}


def write_report(path, d, piv, checks, verdict, m, stats_rows, graph, mode, spaces, figs, mins):
    def md(x):
        return x.to_markdown(index=False)

    st = pd.DataFrame([dict(comparison=k, folds=v['n'], mean=round(v['mean'], 4),
                            ci_lo=round(v['lo'], 4), ci_hi=round(v['hi'], 4),
                            sign_flip_p=round(v['p'], 3)) for k, v in stats_rows.items()])
    sp = pd.DataFrame([dict(fold=h, clusters=L['n'], unreliable=len(L['unreliable']),
                            novel_held_out_labels=len(L['novel']), cut_rule=s['cut_rule'])
                       for h, (L, _, s) in (spaces or {}).items()])
    ci = stats_rows.get('neigh - cell')
    nfold = ci['n'] if ci else len(piv)
    honest = ''
    if ci is not None and ci['lo'] <= 0 <= ci['hi']:
        honest = ("\n> **The interval spans zero.** gate4_expect.csv check 3 says in as many words: "
                  "do NOT write 'spatial context improves transfer' if the interval spans zero, "
                  "whatever the mean says. It spans zero, so that sentence is not written here.\n")

    txt = f"""# Stage 4 - adaptive spatial context (GATE 4)

**{verdict}** - {mins:.1f} min on {DEV}, {len(d)} fits.

| arm | what it sees | LOCO macro-F1 |
|---|---|---|
| `cell` | the cell's own markers only | {m.get('cell', float('nan')):.4f} |
| `neigh` | + its 15 nearest neighbours, pooled | {m.get('neigh', float('nan')):.4f} |
| `shuffle` | + another cell's neighbours, same image | {m.get('shuffle', float('nan')):.4f} |
| `ctx` | `neigh` + the neighbourhood-context loss | {m.get('ctx', float('nan')):.4f} |
{honest}
## The gate

{md(pd.DataFrame(checks))}

Thresholds were declared in `celltype_transfer/declared/gate4_expect.csv` on 2026-09-06, before any of this
code existed. Checks 6 and 7 are asserted at BUILD time, not here - see the graph table below.

## Check 3 - the paired test

{md(st) if len(st) else '_not enough arms finished to pair._'}

Each row is a paired comparison over the LOCO folds: the same fold, the same cells, the same label
space, one thing changed. The interval is a t-interval on {nfold} folds; the p-value is an exact
sign-flip test over all {2 ** nfold} reassignments, so it assumes nothing about the shape of the
differences. **n = {nfold} is small.** An interval this wide cannot separate a small real effect
from none, and that is a statement about the roster, not about the method.

## Per fold

{md(piv.reset_index().round(4))}
{_img(figs, 'delta', 'per-fold difference between neigh and cell')}
## Every fit

{md(d.round(4))}

## The graph these arms read

{md(graph)}

Built from the FULL raw tables, never the 40,000-cell subsample - a cell's 15 nearest neighbours
inside a 1.9% sample are not its neighbours at all (UPMC's mean spacing would rise from 11.7 um to
about 79 um). Check 7 is that assertion: every cohort's median edge length lands inside
{DIST_LO}-{DIST_HI} um. Check 6 asserts no edge crosses an image boundary, on every edge of every
cohort.

## What the neighbourhood is allowed to carry

The pooled profile is the mean `u_coh` of a cell's valid neighbours, pushed through the SAME
encoder as the cell, so unmeasured slots get the same `[ABSENT]` token and nothing is zero-filled.

**The mean alone would not have been a fair test.** Averaging 15 neighbours cannot express "is
there a vessel touching me" (a max question - one bright neighbour barely moves the mean) or "am I
at the tumour boundary" (a variance question - 7 tumour + 8 immune neighbours average out to
roughly the same vector as a uniformly mixed region). A failure under mean-only pooling could not
be told apart from the pooling having thrown the signal away. Two scalars close that gap for
almost no compute:

| scalar | what it measures |
|---|---|
| `het` | how much the 15 neighbours disagree with each other - uniform nest vs mixed zone |
| `d_self` | how far the cell's own profile sits from its neighbourhood's average - a boundary detector |

Both are averaged over the markers the cohort actually measured, so a 17-marker cohort and a
57-marker one give comparable numbers. Measured before any fit: in Keren, tumour and
keratin-positive tumour have the *lowest* `d_self` (0.061-0.069, uniform nests) and Tregs,
neutrophils and NK cells the *highest* (0.099-0.116, scattered in unlike surroundings). CRC shows
the same ordering with a narrower spread.

`d_self` is computed AFTER the shuffle, so in the `shuffle` arm it is meaningless by construction -
the cell keeps its own markers and gets someone else's neighbourhood. Computing it first would
smuggle a real spatial measurement into the arm that is supposed to have none.

Five further scalars describe the neighbourhood's shape: log median edge length in microns, median
edge length relative to the image's own nearest-neighbour spacing, anisotropy of the neighbour
offsets, whether anisotropy is defined, and the fraction of the 15 slots that hold a real
neighbour.

**Attention over the 15 neighbours is deliberately not used.** It would be more expressive and it
would cost 2-3x the compute, but a failure would then be unattributable - idea or architecture?
That is the one thing this gate exists to rule out. If the neighbourhood earns its place here,
attention is the obvious next experiment.

**`nbr_same` - the homotypic fraction - is deliberately NOT a feature.** It is built from the
neighbours' native labels, and on the held-out cohort those labels are the thing being predicted.
It is the sharpest single number in the sidecar and it is exactly the one that cannot be used. It
stays in the npz as a build statistic.

The neighbourhood enters as a residual on `z_cell` through a zero-initialised gate, so at epoch 0
arm `neigh` is bit-identical to arm `cell`. The neighbourhood has to earn the margin; it cannot
win by being a larger model with better-placed prototypes.

## Label space

Label space: **{mode}**{' (plan F4 - each fold trained and scored in a space built from its training cohorts only)' if mode == 'fold' else ' - the whole-roster space, built WITH every held-out cohort present. Comparison row, not a headline.'}

{md(sp) if len(sp) else ''}

## Check 5 - the assumed pixel size

UPMC's `px_um` is recorded as "ASSUMED - not published anywhere". Stage 4 is the only stage that
ever uses physical distance, so an assumption that was harmless everywhere else becomes
load-bearing here. The gate re-fits the UPMC fold with that scale multiplied by {PX_SCALES[0]} and by
{PX_SCALES[1]} and asks whether check 1's verdict moves. The edge vector already carries a
scale-free distance; this measures that rather than asserting it.
"""
    open(path, 'w', encoding='utf-8').write(txt)
    return path


def _img(figs, k, alt):
    return f"\n![{alt}](figures/{os.path.basename(figs[k])})\n" if k in figs else ''


def run_gate():
    assert_roster()
    missing = [c for c in TRAIN if not os.path.exists(nbr_path(c))]
    if missing:
        sys.exit(f"no neighbour sidecar for {missing} - run `python train_spatial_context.py --build` first")

    cohorts = adversarial.available()
    triples, tri2idx, per, genes, ncoh = vocab.read_panel(cohorts)
    V = len(triples)
    train = [c for c in cohorts if SPECS[c]['role'] == 'train']
    excl = classifier.excluded_pairs()

    quick = '--quick' in sys.argv
    pilot = '--pilot' in sys.argv
    refit = '--refit' in sys.argv
    epochs = 2 if quick else classifier.EPOCHS
    q = 'quick_' if quick else ''
    if quick:
        classifier.N_TRAIN, classifier.SCORE_CELLS = 1_500, 800
        print("\n*** --quick: 2 epochs on a tiny draw. Proves the code path; scores nothing. ***")

    folds = train
    if '--folds' in sys.argv:
        pick = set(sys.argv[sys.argv.index('--folds') + 1].split(','))
        folds = [c for c in train if c in pick]
    elif pilot:
        folds = ['Keren']          # smallest panel-overlap fold, a signal, not a gate
    arms = [a for a in ARMS if not (pilot and a[0] == 'ctx')]

    mode = adversarial.space_mode()
    sp_tag = 'fold_' if mode == 'fold' else ''
    spaces = {h: adversarial.space_for(h, mode) for h in folds}
    print(f"\nvocabulary : {V} triples x {len(train)} cohorts")
    print(f"device     : {DEV}" +
          (f" ({torch.cuda.get_device_name(0)})" if DEV.type == 'cuda' else ''))
    print(f"label space: {mode}")
    for h, (Lh, _, s) in spaces.items():
        print(f"    {h:9} {Lh['n']:>3} clusters, {len(Lh['novel'])} NOVEL  ({s['cut_rule']})")

    t0, res = time.time(), []
    for name, use_nbr, use_ctx in arms:
        print(f"\n  arm {name}: neighbourhood={use_nbr} context_loss={use_ctx}"
              f"{' SHUFFLED' if name == 'shuffle' else ''}")
        for h in folds:
            Lh, pt, s = spaces[h]
            res.append(gate_run(f'{q}{name}_{sp_tag}{h}', h, train, per, tri2idx, triples, Lh,
                                excl, use_nbr, use_ctx, name == 'shuffle', refit, epochs,
                                space=s, proto_path=pt))

    # check 5 - 2 extra fits on the UPMC fold only
    if 'UPMC' in folds and not pilot:
        print("\n  check 5: UPMC px_um rescaled (M4 - the value is ASSUMED, not published)")
        Lh, pt, s = spaces['UPMC']
        for sc in PX_SCALES:
            res.append(gate_run(f'px{sc}_{q}neigh_{sp_tag}UPMC', 'UPMC', train, per, tri2idx,
                                triples, Lh, excl, True, False, False, refit, epochs,
                                space=s, px_scales={'UPMC': sc}, proto_path=pt))

    torch.save([{k: v for k, v in r.items() if k != 'state'} for r in res],
               os.path.join(CKPT, f'spatial_{q}{sp_tag}sweep.pt'))
    d, piv, checks, verdict, m, stats_rows = assemble(res, folds)
    print(f"\n--- GATE 4 ---\n{pd.DataFrame(checks).to_string(index=False)}")
    if quick or pilot:
        print(f"\n--{'quick' if quick else 'pilot'}: a signal, not a gate. No report written.")
        return
    graph = pd.read_csv(os.path.join(WORK, 'neighbour_graph_summary.csv'))
    p = write_report(os.path.join(REPORTS, 'train_spatial_context.md'), d, piv, checks, verdict, m,
                     stats_rows, graph, mode, spaces, figure(piv), (time.time() - t0) / 60)
    print(f"\n{verdict} - wrote {p}")


def main():
    if '--gate' in sys.argv or '--pilot' in sys.argv:
        run_gate()
        return
    print("nothing to do. run build_neighbour_graph.py first, then --pilot or --gate.")


if __name__ == '__main__':
    main()

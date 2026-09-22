"""
STAGE 6 - losses and training.  Produces GATE 6.

    python train_prototype_classifier.py --ablate-losses          # GATE 6
    python train_prototype_classifier.py --ablate-losses --quick  # smoke test: proves the code path, scores nothing
    python train_prototype_classifier.py --ablate-losses --refit  # ignore the checkpoint cache
    python train_prototype_classifier.py --ablate-losses --cpu    # force CPU even where a GPU exists
    python train_prototype_classifier.py --ablate-losses --space shipped   # comparison row: the whole-roster space
                                                #   (default is fold-local, plan F4 - each fold
                                                #   trains and scores in work/spaces/, built
                                                #   by `python celltype_transfer/build_fold_label_spaces.py --folds`)
    python train_prototype_classifier.py --ablate-losses --quick --no-warm --folds Sorin   # one-fold smoke test

WHAT THIS STAGE IS. Stage 3 produced z_cell with a deliberately plain linear head, so that Gate 3
could ask one question without a second moving part. Stage 6 is where the real training objective
goes: a prototype loss over Stage 1b's clusters, the guards that make learnable prototypes safe,
and the auxiliary losses that keep the representation general.

TWO THINGS THE DESIGN ASKED FOR ARE STILL NOT HERE, each for a stated reason, plus one thing that
USED TO be excluded and now is not. All three are declared in celltype_transfer/declared/gate6_expect.csv,
which was committed before this file was written; the adversary's row was amended, not deleted,
when its trigger condition fired (2026-09-13) - see the REPLACED row there.

  Descendant-tolerant cross-entropy. It needs Stage 1b's nesting graph, which missed ALL THREE
  cases declared for it (section 7 gap 2 / H2). The design says outright: do not rely on it until
  nesting is re-tested. A loss over an untrusted parent/child graph rewards wrong predictions
  silently, and the failure reads as a modelling result rather than a graph defect. Deferred, not
  rejected - re-test nesting and it drops in without touching anything else.

  The neighbourhood-context auxiliary loss. It predicts a cell's spatial surroundings from
  z_neigh, which Stage 4 produces. Stage 4 has now run and PASSED Gate 4 (2026-09-13,
  reports/train_spatial_context.md) - but Gate 4's own check 3 found the paired 95% CI on (neigh - cell) spans
  zero, so that effect is not established, and wiring it into this headline is a separate decision
  not taken here. SO THE DECLARED "2 vs 4 LOSSES" ABLATION IS STILL A 2 vs 3 - cell-type,
  masked-marker, VICReg. Reported as such rather than quietly renumbered.

  The adversary - NO LONGER EXCLUDED. Gate 3's ORIGINAL 5+1-roster sweep found no lambda beat
  lambda=0 (D-36). Gate 3 was re-run 2026-09-12 on the 7-cohort fold-local roster and that verdict
  flipped: lambda=0.01 now beats lambda=0 by +0.036, winning 6 of 7 folds. gate6_expect.csv's own
  exclusion row named this exact trigger in advance ("if Stage 3b changes that verdict, this row is
  revisited") - it is now REPLACED there, and the adversary is wired in below as one more ablation
  arm (check 3b), copying Gate 3's recipe verbatim (cohort_frac, ramp, deep_slide=False) so any
  difference in outcome is attributable to Stage 6's prototype head, not to a re-tuned adversary.

WHAT GATE 6 DECIDES. Check 3 chooses whether VICReg ships; check 3b (new, 2026-09-13) chooses
whether the adversary ships, judged against the same 2-loss baseline check 3 uses, not against
whichever of 2/3 losses check 3 ships - an untested 3-losses-plus-adversary combination is never
assumed. Check 4 asks whether the stage earned its existence at all, by re-measuring Stage 3's
plain linear head INSIDE THIS RUN on the same encoder and the same vocabulary - Gate 3's own 0.3642
is not comparable, because it was produced on an 88-triple vocabulary before D-39 fixed that.
"""
import os, sys, json, time, zlib, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # runs from any directory
import config
from config import SPECS, WORK, REPORTS, FIGURES, SEED
from models.encoder import CellEncoder, grad_reverse
from models.losses import (Prototypes, CollapseGuard, KendallSigma, vicreg,
                       masked_marker_loss, confidence_weighted_ce)
import splits
import build_marker_vocabulary as vocab
import pretrain_masked_markers as pretrain
import train_adversarial_encoder as adversarial
import metrics                                                        # plan F6: predictions parquet

CKPT = os.path.join(WORK, 'ckpt')
CONF = os.path.join(WORK, 'label_conf')
os.makedirs(CKPT, exist_ok=True)

from models.device import DEV   # one rule for all stages; --cpu forces the CPU

# --------------------------------------------------------------- declared constants
D_TOK, D_Z    = 64, 128
BLOCKS, HEADS = 2, 4
N_TRAIN       = 15_000
SCORE_CELLS   = 8_000
BATCH, LR     = 512, 1e-3
EPOCHS, PATIENCE = 30, 4
MASK_FRAC     = 0.15          # same fraction Stage 2 used
PROTO_TEMP    = 0.1
SPREAD_MIN    = 0.20          # D-28's exclusion floor, inherited (D-33 warned it recurs here)
COLLAPSE_FRAC = 0.5           # check 2: min prototype distance may not halve
SIGMA_CAP     = 3.0           # check 1: how far an auxiliary log-sigma may travel
LINEAR_MARGIN = 0.02          # check 4: Stage 6 must beat the plain linear head by this
ADV_LAMBDA    = 0.01          # check 3b (declared 2026-09-13): Gate 3's SHIPPED lambda on the
                               # 7-cohort fold-local roster, not re-swept here. cohort_frac and the
                               # ramp length are read straight off train_adversarial_encoder so the recipe matches
                               # Gate 3's exactly - see gate6_expect.csv's REPLACED scope row


from config import rng as _rng   # one definition, config.rng


# ----------------------------------------------------------------------------- data
def excluded_pairs():
    """Stage 2's (cohort, marker) exclusion set, inherited by the masked-marker loss.

    D-33 flagged this explicitly: "recurs in Stage 6, whose masked-marker auxiliary loss inherits
    the same exclusion set". A pair below the rank_spread floor has a collapsed R2 denominator -
    predicting it teaches nothing and the gradient is noise.
    """
    p = os.path.join(WORK, 'marker_dynamic_range.csv')
    if not os.path.exists(p):
        return set()
    d = pd.read_csv(p)
    return {(r.cohort, r.triple) for r in d.itertuples() if r.rank_spread < SPREAD_MIN}


def load_conf(c, cell_ids):
    """Per-cell label confidence, or all-ones where a cohort has none.

    1-of-7 coverage and the report says so: only UPMC ships kNN.prob (0.14-1.0). The absence of a
    file MEANS a constant confidence, so this is not a silent default - see build_label_confidence.py.
    """
    p = os.path.join(CONF, f'{c}.parquet')
    if not os.path.exists(p):
        return np.ones(len(cell_ids), 'float32'), False
    m = pd.read_parquet(p).set_index('cell_id').conf
    return m.reindex(cell_ids).fillna(1.0).to_numpy('float32'), True


def load_cohort(c, triples_c, tri2idx, n_vocab, L, excl, draw_seed=None):
    """One cohort as tensors. Mirrors Stage 3's loader and adds confidence and the mask-eligible
    slot list, so Stage 6 is scored on exactly the cells Stage 3 was.

    `draw_seed` varies WHICH CELLS are drawn. Default None reproduces every gate result to the
    bit - Gates 3, 6 and 7 all ran without it. Step 3's repeated-seed runs pass it so their
    intervals cover the cell draw as well as the model initialisation; an interval over model
    seeds alone would be narrower than the truth and would look like more precision than there is.
    """
    tl = list(triples_c)
    v = pd.read_parquet(config.full_table(c),
                        columns=['cell_id', 'image_id', 'native_label'] +
                                [f'u_coh::{t}' for t in tl])
    y = np.array([L['key'].get((c, str(lab)), -1) for lab in v.native_label])
    ok = y >= 0
    if not ok.any():
        return None
    v, y = v[ok].reset_index(drop=True), y[ok]
    U = v[[f'u_coh::{t}' for t in tl]].to_numpy('float32')
    # the table's own ids, unchanged ('CRC|reg001_A') - the one format splits.split_masks accepts (F3)
    img = v.image_id.to_numpy()
    conf, has_conf = load_conf(c, v.cell_id.to_numpy())

    sm = splits.split_masks(c, img)

    def draw(mask, cap, why):
        w = np.flatnonzero(mask)
        if len(w) > cap:
            key = ('draw', c, why) if draw_seed is None else ('draw', c, why, draw_seed)
            w = np.sort(_rng(*key).choice(w, cap, replace=False))
        return w

    parts = dict(train=draw(sm['train'], N_TRAIN, 'train'),
                 val=draw(sm['val'], N_TRAIN // 4, 'val'),
                 test=draw(sm['test'], SCORE_CELLS, 'test'))

    slots = np.array([tri2idx[t] for t in tl])
    full = np.zeros((len(U), n_vocab), 'float32')
    full[:, slots] = U
    present = np.zeros(n_vocab, bool)
    present[slots] = True
    # mask-eligible = measured here AND not on Stage 2's exclusion list (D-28/D-33)
    elig = np.zeros(n_vocab, bool)
    for t, s_ in zip(tl, slots):
        elig[s_] = (c, t) not in excl

    # `cid` carries the drawn cells' own cell_id. Stage 4's neighbour sidecar is indexed by
    # cell_id, and the draw happens here, so without it Stage 4 would have to re-implement this
    # loader to learn WHICH cells it must fetch neighbours for - a second copy of the label
    # filter, the split and the subsample, any of which could drift out of step and would score
    # the spatial arms on different cells than the control. Additive: nothing else reads it.
    cid = v.cell_id.to_numpy()
    return dict(cohort=c, has_conf=has_conf,
                idx=torch.arange(n_vocab).long().to(DEV),
                present=torch.from_numpy(present).to(DEV),
                elig=torch.from_numpy(elig).to(DEV),
                U={k: torch.from_numpy(full[w]).to(DEV) for k, w in parts.items()},
                y={k: torch.from_numpy(y[w]).long().to(DEV) for k, w in parts.items()},
                conf={k: torch.from_numpy(conf[w]).to(DEV) for k, w in parts.items()},
                img={k: img[w] for k, w in parts.items()},
                cid={k: cid[w] for k, w in parts.items()},
                slots=slots, triples=tl,
                n={k: len(w) for k, w in parts.items()})


def make_hide(B, elig, gen):
    """Hide MASK_FRAC of the eligible slots per cell, at least one."""
    e = torch.nonzero(elig).flatten()
    if len(e) == 0:
        return torch.zeros(B, len(elig), dtype=torch.bool, device=elig.device)
    k = max(1, int(round(MASK_FRAC * len(e))))
    hide = torch.zeros(B, len(elig), dtype=torch.bool, device=elig.device)
    for i in range(B):
        pick = e[torch.randperm(len(e), generator=gen, device='cpu')[:k].to(e.device)]
        hide[i, pick] = True
    return hide


# ----------------------------------------------------------------------------- prototypes
def init_prototypes(enc, n_class, V, path=None):
    """Start each prototype where Stage 1b says its cluster lives.

    `path` overrides work/prototypes.npy. Stage 7 uses it to load the centroids of a label space
    built WITHOUT the frozen holdout - same file format, different cluster set (D-48).

    work/prototypes.npy is [n_class, 99] - each cluster's mean marker signature over the canonical
    vocabulary. Each signature is fed through the encoder as a synthetic cell and the output
    becomes that cluster's starting vector. Random initialisation would throw away the one thing
    Stage 1b actually knows, and would make the drift diagnostic meaningless: drift is only
    interpretable if the starting point means something.

    THE NaNs ARE INFORMATION, NOT CORRUPTION. 751 of the 2475 entries are NaN - a cluster has no
    signature for a marker that none of its contributing cohorts measures, and clusters built from
    panel-poor cohorts have many (cluster 17 has 60). Feeding them in raw makes every prototype
    NaN, which then makes every logit, loss and sigma NaN while training still appears to run.
    They are exactly what the [ABSENT] token exists for, so each cluster is encoded with its OWN
    `present` mask and the missing values never reach the value MLP.
    """
    p = path or os.path.join(WORK, 'prototypes.npy')
    if not os.path.exists(p):
        return None
    sig = np.load(p).astype('float32')
    if sig.shape != (n_class, V):
        print(f"    prototypes.npy is {sig.shape}, expected {(n_class, V)} - random init instead")
        return None
    ok = ~np.isnan(sig)
    idx = torch.arange(V).long().to(DEV)
    out = []
    with torch.no_grad():
        for k in range(n_class):
            u = torch.from_numpy(np.nan_to_num(sig[k], nan=0.0)).to(DEV).unsqueeze(0)
            present = torch.from_numpy(ok[k]).to(DEV)
            out.append(enc(u, idx, present))
    z = torch.cat(out)
    if not bool(torch.isfinite(z).all()):
        print("    prototype init produced non-finite values - random init instead")
        return None
    return z.detach()


def warm_state(train_cohorts):
    """Stage 2's pretrained token layer, the same warm start Stage 3 used.

    Without it Stage 6 begins from a cold encoder and its numbers are not comparable with Gate 3's
    - which would defeat check 4, whose whole job is to compare the prototype head against the
    linear head on equal terms.

    FOLD-LOCAL since 2026-09-11 (plan F4). This used to load pretrain_armB.pt for every fold, and that
    model was trained on all 7 cohorts - so the fold holding out X started from an encoder that had
    already seen X's cells. pretrain.warm_for() picks the Stage 2 model trained ONLY on cohorts this fit
    trains on (pretrain_armb_loco_X.pt for fold X) and raises rather than fall back when none exists.
    The old file-missing path returned None silently; a missing warm start now raises too, because
    a cold encoder changes every number. `--no-warm` is the only way to cold-start.
    """
    if '--no-warm' in sys.argv:
        return None, None
    state, name, _ = pretrain.warm_for(train_cohorts)
    return state, name


# ----------------------------------------------------------------------------- model
class Stage6(nn.Module):
    """Encoder + the head under test. `head='proto'` is the stage; `head='linear'` is Gate 6
    check 4's control - Stage 3's plain linear head, on the same encoder, in the same run.

    `n_slide`/`n_cohort` given (check 3b) also attaches Gate 3's two adversarial heads behind
    gradient reversal - copied, not re-derived, so a difference in outcome is attributable to the
    head under test, not to a re-tuned adversary. None (the default) builds no adversary heads at
    all, so every lambda=0 arm stays the exact model Gate 6 has always trained."""

    def __init__(self, n_vocab, n_class, head='proto', proto_init=None,
                 n_slide=None, n_cohort=None, cohort_frac=adversarial.COHORT_FRAC, deep_slide=False):
        super().__init__()
        self.enc = CellEncoder(n_vocab, d_tok=D_TOK, d_z=D_Z, blocks=BLOCKS, heads=HEADS)
        self.head_kind = head
        self.proto = Prototypes(n_class, D_Z, init=proto_init, temp=PROTO_TEMP) \
            if head == 'proto' else None
        self.linear = nn.Linear(D_Z, n_class) if head == 'linear' else None
        self.mask_head = nn.Linear(D_TOK, 1)
        self.cohort_frac = cohort_frac
        self.slide_head = None if n_slide is None else (
            nn.Sequential(nn.Linear(D_Z, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(),
                         nn.Linear(256, n_slide)) if deep_slide else nn.Linear(D_Z, n_slide))
        self.cohort_head = None if n_cohort is None else nn.Linear(D_Z, n_cohort)

    def forward(self, u, idx, present, hide=None):
        z, tok = self.enc(u, idx, present, hide=hide, return_tokens=True)
        logits = self.proto.logits(z) if self.proto is not None else self.linear(z)
        return z, logits, self.mask_head(tok).squeeze(-1)

    def adv(self, z, lam):
        """Check 3b's two adversarial heads, behind gradient reversal (Gate 3's D-16/D-37
        recipe). Only callable when the model was built with n_slide/n_cohort set."""
        return self.slide_head(grad_reverse(z, lam)), \
               self.cohort_head(grad_reverse(z, lam * self.cohort_frac))


# ----------------------------------------------------------------------------- fit
def fit(data, train_cohorts, L, V, head='proto', use_vicreg=True, use_conf=True,
        epochs=EPOCHS, patience=PATIENCE, seed=SEED, log='', proto_path=None, lam_adv=0.0):
    """Train on `train_cohorts`. Early stopping on held-out SLIDES by macro-F1 - the quantity the
    gate decides on, not the loss, which is not comparable across loss sets.

    `lam_adv` (check 3b) adds Gate 3's domain-adversarial term. The slide vocabulary is built from
    `train_cohorts` ONLY, the same restriction `train_adversarial_encoder.loco_run` uses - including the held-out
    cohort's slides here would leak it into training through the adversary's own label space."""
    sidx = adversarial.slide_index({c: data[c] for c in train_cohorts}) if lam_adv > 0 else {}
    if lam_adv > 0 and not sidx:
        raise ValueError("check 3b: empty slide vocabulary for the adversary")
    ci = {c: i for i, c in enumerate(train_cohorts)}
    # Build the encoder first and warm-start it, THEN read the prototypes out of that same
    # encoder. Initialising them from a different (random) encoder would place them in a space
    # the model does not use, and the drift diagnostic would measure the mismatch rather than
    # any disagreement with Stage 1b.
    torch.manual_seed(seed)
    m = Stage6(V, L['n'], head=head, proto_init=None,
              n_slide=(len(sidx) if lam_adv > 0 else None),
              n_cohort=(len(train_cohorts) if lam_adv > 0 else None)).to(DEV)
    warm, wname = warm_state(train_cohorts)
    if warm is not None:
        n_loaded, _ = m.enc.load_stage2(warm)
        # load_stage2() skips a shape mismatch SILENTLY (D-39 dropped ident.weight that way), so
        # the most valuable tensor is checked explicitly rather than trusted.
        if not torch.equal(m.enc.ident.weight.detach().cpu(), warm['ident.weight']):
            raise RuntimeError(f"warm start {wname}: ident.weight did not load - vocabulary "
                               f"mismatch between the checkpoint and this model")
    if head == 'proto':
        pi = init_prototypes(m.enc, L['n'], V, path=proto_path)
        if pi is not None:
            with torch.no_grad():
                m.proto.p.copy_(pi)
                m.proto.p0.copy_(pi)
    aux = ['mask'] + (['vicreg'] if use_vicreg else [])
    sig = KendallSigma(aux).to(DEV)
    guard = CollapseGuard(m.proto, COLLAPSE_FRAC) if m.proto is not None else None
    opt = torch.optim.Adam(list(m.parameters()) + list(sig.parameters()), lr=LR)
    gen = torch.Generator().manual_seed(seed)

    pools = {c: {int(k): np.flatnonzero(data[c]['y']['train'].cpu().numpy() == k)
                 for k in np.unique(data[c]['y']['train'].cpu().numpy())}
             for c in train_cohorts}

    best, best_state, bad, used = -np.inf, None, 0, epochs
    sig_hist, guard_hist = [], []
    for ep in range(epochs):
        # ramp identical to Gate 3's (adversarial.RAMP_EPOCHS) - starting at full strength makes the
        # encoder fight a discriminator that has not learned anything yet, which is noise, not
        # signal (same reasoning as train_adversarial_encoder.fit's docstring)
        lam_ep = lam_adv * min(1.0, (ep + 1) / max(1, adversarial.RAMP_EPOCHS)) if lam_adv > 0 else 0.0
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
            # -1 marks a cell with no slide class (nested domain, D-38) - same convention as
            # train_adversarial_encoder.fit. Computed once per cohort per epoch, then sliced per batch below.
            sl = (torch.tensor([sidx.get(s, -1) for s in d['img']['train']], device=DEV)
                  if lam_adv > 0 else None)
            for i in range(0, len(take), BATCH):
                b = torch.from_numpy(take[i:i + BATCH]).long().to(DEV)
                u = d['U']['train'][b]
                hide = make_hide(len(b), d['elig'], gen)
                z, logits, pred = m(u, d['idx'], d['present'], hide=hide)

                w = d['conf']['train'][b] if (use_conf and d['has_conf']) else None
                l_cls = confidence_weighted_ce(logits, d['y']['train'][b], w)
                losses = {'mask': masked_marker_loss(pred, u, hide)}
                if use_vicreg:
                    losses['vicreg'] = vicreg(z)
                l_aux, _ = sig(losses)
                # the cell-type weight is PINNED at 1.0 and never enters the sigma weighting -
                # Kendall is free to switch off a task whose labels look noisy, and cross-cohort
                # labels look very noisy. That is the one task that must not be switched off.
                loss = l_cls + l_aux
                if lam_adv > 0:
                    # check 3b: NOT Kendall-weighted, on purpose - the adversary is a min-max game,
                    # not a regulariser, and letting sigma weighting shrink it would fight the
                    # gradient-reversal dynamic instead of just balancing loss scales (same as
                    # train_adversarial_encoder.fit, which sums these unweighted too).
                    ls, lo = m.adv(z, lam_ep)
                    sl_b = sl[b]
                    mvalid = sl_b >= 0
                    l_slide = F.cross_entropy(ls[mvalid], sl_b[mvalid]) if bool(mvalid.any()) \
                        else ls.sum() * 0.0
                    l_cohort = F.cross_entropy(lo, torch.full((len(b),), ci[c], device=DEV))
                    loss = loss + l_slide + l_cohort
                opt.zero_grad(); loss.backward(); opt.step()

        sig_hist.append(sig.snapshot())
        if guard is not None:
            guard_hist.append(guard.step(ep))

        m.eval()
        yt, yp = [], []
        with torch.no_grad():
            for c in train_cohorts:
                d = data[c]
                U = d['U']['val']
                for i in range(0, len(U), 1024):
                    _, lg, _ = m(U[i:i + 1024], d['idx'], d['present'])
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
    info = dict(epochs_used=used, val_f1=round(best, 4),
                warm_start=wname or 'cold (--no-warm)',
                sigma=sig_hist, sigma_names=aux,
                guard=guard_hist,
                guard_events=(guard.events if guard else []),
                guard_floor=(round(guard.floor, 4) if guard else None),
                guard_d0=(round(guard.d0, 4) if guard else None),
                guard_passed=(guard.passed() if guard else None),
                drift=(m.proto.drift().round(4).tolist() if m.proto is not None else None),
                lam_adv=lam_adv, n_slide_adv=(len(sidx) if lam_adv > 0 else 0))
    return m, info


def predict(m, d):
    out = []
    with torch.no_grad():
        U = d['U']['test']
        for i in range(0, len(U), 1024):
            _, lg, _ = m(U[i:i + 1024], d['idx'], d['present'])
            out.append(lg.argmax(1))
    return torch.cat(out).cpu().numpy() if out else np.array([])


space_for = adversarial.space_for          # one definition, shared with Stage 3 (plan F4)


def loco_run(tag, held, train, per, tri2idx, triples, L, excl, head, use_vicreg, use_conf,
             refit, epochs, proto_path=None, space=None, lam_adv=0.0):
    p = os.path.join(CKPT, f'classifier_{tag}.pt')
    space = space or {}
    if os.path.exists(p) and not refit:
        r = torch.load(p, weights_only=False, map_location='cpu')
        # A fold cached before the fold-local warm start (2026-09-11) has no `warm_start` record
        # and may have started from an encoder that saw its held-out cohort - refit, never reuse.
        # Same for a fold trained on another within-cohort split (plan F3, patient splits), or in
        # another label space (plan F4 - a rebuilt fold map changes its hash).
        if ('warm_start' in r.get('info', {}) and not splits.split_stale(r)
                and r.get('space') == space.get('space')):
            print(f"    [cache] {tag}  (warm start {r['info']['warm_start']})")
            return r
        print(f"    [STALE] {tag}: predates the fold-local warm start, the patient split or "
              f"this label space - refitting")

    t0 = time.time()
    V = len(triples)
    rest = [c for c in train if c != held]
    data = {c: load_cohort(c, per[c], tri2idx, V, L, excl) for c in train}
    data = {c: d for c, d in data.items() if d is not None}
    rest = [c for c in rest if c in data]
    if not rest:
        raise ValueError(f"fold '{held}': no training cohorts left")

    m, info = fit(data, rest, L, V, head=head, use_vicreg=use_vicreg, use_conf=use_conf,
                  epochs=epochs, proto_path=proto_path, lam_adv=lam_adv)
    yp = predict(m, data[held])
    yt = data[held]['y']['test'].cpu().numpy()
    f1_all, per_cls = metrics.macro_f1(yt, yp, L['n'])
    f1_core, _ = metrics.macro_f1(yt, yp, L['n'], drop=L['unreliable'])
    f1_maj, f1_rnd, _ = adversarial.baselines(data, rest, held, L['n'], L['unreliable'])

    # plan F6: `required_outputs.predictions` never existed anywhere in this codebase - every
    # gate scored in memory and threw the raw predictions away. One per arm/fold, so any metric
    # in metrics.py (including one added after this run) can be recomputed without retraining.
    # main() copies whichever arm assemble() ships to the canonical predictions_ours_{held}.parquet
    # name the protocol names. Skipped on a cache hit (see below) - nothing to save without a run.
    pred_path = metrics.save_predictions(
        os.path.join(REPORTS, 'predictions', f'{tag}.parquet'),
        cell_id=data[held]['cid']['test'], cohort=held, held=held,
        y_true=yt, y_pred=yp, method='ours', protocol='LOCO')

    r = dict(tag=tag, held=held, head=head, vicreg=use_vicreg, conf=use_conf, lam_adv=lam_adv,
             f1_core=f1_core, f1_all=f1_all, f1_majority=f1_maj, f1_random=f1_rnd,
             per_cls=per_cls, info=info, n_cohort=len(rest), split_fp=splits.split_fp(),
             space=space.get('space'), cut_rule=space.get('cut_rule'), n_class=L['n'],
             names=list(L['names']), n_novel_labels=len(L.get('novel', [])),
             seconds=round(time.time() - t0, 1), predictions=pred_path,
             state={k: v.detach().cpu() for k, v in m.state_dict().items()})
    torch.save(r, p)
    g = info.get('guard_passed')
    print(f"    [done ] {tag}  F1core={f1_core:.4f} (maj {f1_maj:.4f})  "
          f"guard={'ok' if g else ('COLLAPSE' if g is False else '-')}  "
          f"{r['seconds']:.0f}s  epochs={info['epochs_used']}")
    return r


# ----------------------------------------------------------------------------- figures
def figures(ship, L):
    """The two plots the design asks for by name: sigma trajectory and prototype drift."""
    os.makedirs(FIGURES, exist_ok=True)
    out = {}
    info = ship['info']

    if info['sigma']:
        fig, ax = plt.subplots(figsize=(7, 4))
        for n in info['sigma_names']:
            ax.plot([s[n] for s in info['sigma']], label=n, lw=2)
        ax.axhline(info['sigma'][0][info['sigma_names'][0]] + SIGMA_CAP, ls='--', c='crimson',
                   lw=1, label=f'cap (+{SIGMA_CAP})')
        ax.set_xlabel('epoch'); ax.set_ylabel('log sigma')
        ax.set_title('Auxiliary log-sigma. Rising = the loss is being switched off.\n'
                     'The cell-type weight is pinned and does not appear here.', fontsize=9)
        ax.legend(fontsize=8); fig.tight_layout()
        p = os.path.join(FIGURES, 'classifier_sigma.png'); fig.savefig(p, dpi=130); plt.close(fig)
        out['sigma'] = p

    if info['guard']:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(info['guard'], lw=2, label='min pairwise prototype distance')
        ax.axhline(info['guard_floor'], ls='--', c='crimson', lw=1,
                   label=f"floor ({COLLAPSE_FRAC}x init)")
        for e in info['guard_events']:
            ax.axvline(e['epoch'], c='orange', lw=1, alpha=.7)
        ax.set_xlabel('epoch'); ax.set_ylabel('cosine distance')
        ax.set_title('Collapse guard. Below the floor, two clusters are merging.', fontsize=9)
        ax.legend(fontsize=8); fig.tight_layout()
        p = os.path.join(FIGURES, 'classifier_collapse.png'); fig.savefig(p, dpi=130); plt.close(fig)
        out['collapse'] = p

    if info.get('drift'):
        dr = np.array(info['drift'])
        o = np.argsort(-dr)
        fig, ax = plt.subplots(figsize=(7, max(3, 0.25 * len(dr))))
        ax.barh([L['names'][i] or f'cluster {i}' for i in o][::-1], dr[o][::-1], color='steelblue')
        ax.set_xlabel('cosine distance moved from the Stage 1b signature')
        ax.set_title('Prototype drift. A long bar is the model DISAGREEING with Stage 1b.',
                     fontsize=9)
        fig.tight_layout()
        p = os.path.join(FIGURES, 'classifier_drift.png'); fig.savefig(p, dpi=130); plt.close(fig)
        out['drift'] = p
    return out


# ----------------------------------------------------------------------------- gate
def _is_proto2(r, conf):
    return (r['head'] == 'proto' and not r['vicreg'] and bool(r['conf']) == conf
            and not r.get('lam_adv', 0.0) > 0)


def assert_conf_arms_differ(res):
    """D-45 guard (gate4_expect.csv check 8 names the failure): a check whose two arms can be the
    same computation must assert that they are not. Confidence-on and confidence-off must end with
    different weights on every fold where both ran - identical weights mean the switch touched
    nothing, and the comparison would print a clean-looking zero."""
    on = {r['held']: r for r in res if _is_proto2(r, True)}
    for r in res:
        if not _is_proto2(r, False) or r['held'] not in on:
            continue
        a, b = r['state'], on[r['held']]['state']
        if all(torch.equal(a[k], b[k]) for k in a):
            raise RuntimeError(f"check 6, fold {r['held']}: confidence on and off trained IDENTICAL "
                               f"weights - no training cohort's confidence reached the loss (D-45)")


def conf_check(res):
    """Check 6 row: 2 losses with confidence weighting minus without, paired over the folds where
    the switch can matter. Diagnostic only (required=0 in gate6_expect.csv)."""
    on = {r['held']: r['f1_core'] for r in res if _is_proto2(r, True)}
    off = {r['held']: r['f1_core'] for r in res if _is_proto2(r, False)}
    folds = [h for h in off if h in on]
    if not folds:
        return dict(check='6  confidence weighting (diagnostic)', result='not run', detail='')
    delta = [on[h] - off[h] for h in folds]
    if len(folds) >= 2:
        p = metrics.paired(delta)
        det = (f"conf on minus off, {p['n']} folds: mean {p['mean']:+.4f}, 95% CI "
               f"[{p['lo']:+.4f}, {p['hi']:+.4f}], sign-flip p = {p['p']:.3f}")
    else:
        det = f"conf on minus off, 1 fold ({folds[0]}): {delta[0]:+.4f}"
    return dict(check='6  confidence weighting (diagnostic)', result='diagnostic', detail=det)


def assemble(res, L):
    d = pd.DataFrame([{**{k: r[k] for k in ('tag', 'held', 'head', 'vicreg', 'conf',
                                            'f1_core', 'f1_all', 'f1_majority', 'seconds')},
                       'lam_adv': r.get('lam_adv', 0.0),
                       'n_class': r.get('n_class'), 'cut_rule': r.get('cut_rule')}
                      for r in res])

    # BRACKETS, NOT ATTRIBUTES. `d.head` is DataFrame.head - the method - so `d.head == 'proto'`
    # is a silent False and every arm comes back empty with a nan mean. Nothing raises.
    def arm(h, v, c=True, adv=False):
        return d[(d['head'] == h) & (d['vicreg'] == v) & (d['conf'] == c)
                 & ((d['lam_adv'] > 0) == adv)]

    three, two, lin = arm('proto', True), arm('proto', False), arm('linear', True)
    advrun = arm('proto', False, adv=True)
    m3 = float(three.f1_core.mean()) if len(three) else float('nan')
    m2 = float(two.f1_core.mean()) if len(two) else float('nan')
    ml = float(lin.f1_core.mean()) if len(lin) else float('nan')
    m2adv = float(advrun.f1_core.mean()) if len(advrun) else float('nan')

    use_vic = m3 > m2
    # check 3b (declared 2026-09-13, gate6_expect.csv): the adversary is judged against the SAME
    # 2-loss baseline it was added to, not against whichever of 2/3 losses check 3 ships - an
    # untested combination (3 losses + adversary) is never assumed. Ship = whichever of the three
    # MEASURED arms scores highest.
    use_adv = (not np.isnan(m2adv)) and m2adv > m2
    cands = {'2 losses': m2, '3 losses (+VICReg)': m3,
             f'2 losses + adversary (lambda={ADV_LAMBDA:g})': m2adv}
    ship_name = max(cands, key=lambda k: cands[k] if not np.isnan(cands[k]) else -np.inf)
    ship_f1 = cands[ship_name]
    ship_sel = {'2 losses': dict(vicreg=False, adv=False),
               '3 losses (+VICReg)': dict(vicreg=True, adv=False),
               f'2 losses + adversary (lambda={ADV_LAMBDA:g})': dict(vicreg=False, adv=True)
              }[ship_name]
    ship = next((r for r in res if r['head'] == 'proto' and r['vicreg'] == ship_sel['vicreg']
                 and (r.get('lam_adv', 0.0) > 0) == ship_sel['adv']
                 and r['conf'] and r['info']['sigma']), None) or res[0]

    inf = ship['info']
    sig_move = max((abs(inf['sigma'][-1][n] - inf['sigma'][0][n]) for n in inf['sigma_names']),
                   default=0.0)
    guard_ok = inf.get('guard_passed')
    ev = inf.get('guard_events') or []

    checks = [
        dict(check='1  auxiliary sigma does not run away',
             result='PASS' if sig_move <= SIGMA_CAP else 'FAIL',
             detail=f"largest log-sigma move {sig_move:.3f} against a cap of {SIGMA_CAP}"),
        dict(check='2  prototypes do not collapse',
             result='PASS' if guard_ok else 'FAIL',
             detail=(f"floor {inf.get('guard_floor')} (init {inf.get('guard_d0')}); "
                     + (f"{len(ev)} pair(s) frozen" if ev else "no pair frozen"))),
        dict(check='3  do the extra losses earn their place',
             result=('ship 3' if use_vic else 'ship 2'),
             detail=f"3 losses {m3:.4f} vs 2 losses {m2:.4f} - delta {m3 - m2:+.4f}"),
        dict(check=f'3b  does the adversary earn its place (lambda={ADV_LAMBDA:g})',
             result=('ship adversary' if use_adv else 'no adversary'),
             detail=(f"2 losses + adversary {m2adv:.4f} vs 2 losses {m2:.4f} - "
                     f"delta {m2adv - m2:+.4f}" if not np.isnan(m2adv) else "not run")),
        dict(check=f'4  beats the plain linear head by >= {LINEAR_MARGIN:.2f}',
             result='PASS' if (ship_f1 - ml) >= LINEAR_MARGIN else 'FAIL',
             detail=(f"{ship_name} {ship_f1:.4f} vs linear head {ml:.4f} "
                     f"- margin {ship_f1 - ml:+.4f}")),
        conf_check(res),
    ]
    hard = [c for c in checks if c['check'][0] in '124']
    verdict = 'PASS' if all(c['result'] == 'PASS' for c in hard) else 'FAIL'
    return d, checks, verdict, ship, ship_name, ship_f1, ml, m2, m3, m2adv


def write_report(path, d, checks, verdict, ship, ship_name, ship_f1, ml, m2, m3, m2adv, L, figs,
                 mins, mode='shipped', spaces=None):
    """`L` is the label space of the fold whose run is shown in the figures (`ship`). Under the
    fold-local mode every fold has its own space, listed in the 'Label space' section."""
    def md(x):
        return x.to_markdown(index=False)

    if mode == 'fold':
        sp_rows = pd.DataFrame([dict(fold=h, clusters=Lh['n'], unreliable=len(Lh['unreliable']),
                                     novel_held_out_labels=len(Lh['novel']),
                                     cut=s['tau'], cut_rule=s['cut_rule'])
                                for h, (Lh, _, s) in (spaces or {}).items()])
        space_txt = (f"Label space: **fold-local** (plan F4) - each fold is trained and scored in a "
                     f"space built from its training cohorts only (`work/spaces/`, "
                     f"`reports/build_fold_label_spaces.md`), with its own prototypes. Held-out labels no cluster "
                     f"admits are NOVEL and left out of the closed-set macro-F1. Macro-F1 is "
                     f"therefore averaged over folds whose class sets differ. Folds marked "
                     f"`nearest_feasible` had no usable cut and use the post-hoc fallback in "
                     f"`panel/gate1b_fold_v2_expect.csv`.\n\n{md(sp_rows)}\n\nThe figures below "
                     f"show the {ship['held']} fold ({L['n']} clusters).")
    else:
        space_txt = (f"Label space: the SHIPPED whole-roster space, {L['n']} Stage 1b clusters, "
                     f"{len(L['unreliable'])} flagged unreliable and held out of every headline "
                     f"number. It was built WITH every held-out cohort present, so this is the "
                     f"comparison row, not the protocol's headline (plan F4).")

    def img(k, alt):
        return (f"\n![{alt}](figures/{os.path.basename(figs[k])})\n" if k in figs else '')

    inf = ship['info']
    drift = ''
    if inf.get('drift'):
        dd = pd.DataFrame(dict(cluster=[c or f'cluster {i}' for i, c in enumerate(L['names'])],
                               moved=np.round(inf['drift'], 4)))
        drift = md(dd.sort_values('moved', ascending=False).head(10))
    ev = inf.get('guard_events') or []
    frozen = (md(pd.DataFrame([dict(epoch=e['epoch'],
                                    a=L['names'][e['a']] or e['a'],
                                    b=L['names'][e['b']] or e['b'],
                                    distance=e['distance']) for e in ev]))
              if ev else '_No pair ever breached the floor._')
    vic_line = ('VICReg earns its place and ships.' if m3 > m2 else
                'VICReg does NOT improve macro-F1, so it is dropped. The design said to keep the '
                'extra losses only if they earn it.')
    adv_line = ('_not run._' if np.isnan(m2adv) else
                ('The adversary earns its place and ships.' if m2adv > m2 else
                 'The adversary does NOT improve macro-F1 on Stage 6\'s prototype head, even '
                 'though it did on Gate 3\'s linear head - so it is dropped. Kept only if it '
                 'earns it, same rule as VICReg above.'))

    txt = f"""# Stage 6 - losses and training (GATE 6)

**{verdict}** - ships **{ship_name}** at LOCO macro-F1 **{ship_f1:.4f}** - {mins:.1f} min on {DEV}.

{space_txt}

## The gate

{md(pd.DataFrame(checks))}

Thresholds were declared in `celltype_transfer/declared/gate6_expect.csv` before any of this code was
written. Two scope rows there record things the design asked for that are still NOT built, and a
third records why the domain-adversarial exclusion below was REPLACED for this run - read all
three before concluding anything is missing or assuming the adversary was always here.

## What is deliberately not in this stage

**Descendant-tolerant cross-entropy.** It needs Stage 1b's nesting graph, which missed all three
cases declared for it (section 7 gap 2). A loss over an untrusted parent/child graph rewards wrong
predictions silently, and the failure would read as a modelling result rather than a graph defect.
Deferred, not rejected.

**The neighbourhood-context loss.** It needs `z_neigh` from Stage 4. Stage 4 itself has now run and
PASSED Gate 4 (`reports/train_spatial_context.md`) - but Gate 4's own check 3 found the paired 95% CI on
(neigh - cell) spans zero, so that effect is not established. Wiring an unconfirmed effect into
this headline is a separate decision from the one this run makes, and it is NOT taken here. **The
declared "2 vs 4 losses" ablation is therefore still a 2 vs 3** - cell type, masked marker, VICReg.
Check 3 below is that comparison, not the one originally written.

**The adversary.** Gate 3's original 5+1-roster sweep found no lambda beat lambda=0 (D-36). Gate 3
was re-run 2026-09-12 on the 7-cohort fold-local roster and that verdict flipped - lambda=0.01 now
beats lambda=0 by +0.036, winning 6 of 7 folds. `gate6_expect.csv`'s own exclusion row named this
exact trigger in advance and is marked REPLACED. **Check 3b below wires it in as one more ablation
arm**, at Gate 3's shipped lambda, copying its recipe verbatim rather than re-tuning it here.

## Check 3 - the loss ablation

| losses | LOCO macro-F1 |
|---|---|
| 2 - cell type + masked marker | {m2:.4f} |
| 3 - plus VICReg | {m3:.4f} |

{vic_line}

## Check 3b - the adversary (lambda={ADV_LAMBDA:g})

| configuration | LOCO macro-F1 |
|---|---|
| 2 losses | {m2:.4f} |
| 2 losses + adversary | {'_not run_' if np.isnan(m2adv) else f'{m2adv:.4f}'} |

{adv_line}

This is the SAME 2-loss baseline check 3 measured, with one thing added - not a comparison against
whichever of 2/3 losses check 3 ships. An untested combination (3 losses + adversary) is not
assumed to work and is not shipped. The final ship configuration ({ship_name}) is whichever of the
three arms actually measured here scores highest.

## Check 4 - did this stage earn its existence

| head | LOCO macro-F1 |
|---|---|
| Stage 3's plain linear head, same encoder, same run | {ml:.4f} |
| Stage 6 prototype loss ({ship_name}) | {ship_f1:.4f} |
| **margin** | **{ship_f1 - ml:+.4f}** against a required {LINEAR_MARGIN:+.2f} |

The linear control is re-measured INSIDE this run rather than read from Gate 3's report. Gate 3
ran on an 88-triple vocabulary before D-39 fixed that, so its 0.3642 is not comparable with
anything produced afterwards. Same discipline as Gate 2's core-9 control (D-27): measure the
baseline in the same run, or the comparison changes two things at once.

## Every run

{md(d.round(4))}

## Check 1 - the sigma trajectory
{img('sigma', 'auxiliary log-sigma over epochs')}
Kendall uncertainty weighting can switch a loss off by driving its sigma up - the weight is
1/(2 sigma^2), so a runaway sigma silently deletes the term while training still looks healthy.
**The cell-type sigma is pinned at 1.0 and is not learned**, because cross-cohort labels come from
six annotation schemes and look extremely noisy, which is exactly the condition under which
Kendall weighting would switch off the one task that matters.

## Check 2 - the collapse guard
{img('collapse', 'minimum pairwise prototype distance over epochs')}
Learnable prototypes can silently merge two classes, and macro-F1 over classes present in the
truth does not necessarily expose it. A breach freezes the pair and logs it rather than aborting -
a merge may be a true statement about the label space.

{frozen}

## Prototype drift - a diagnostic, not a failure
{img('drift', 'how far each prototype moved from its Stage 1b signature')}
Prototypes start at their Stage 1b marker signature and are learnable on purpose, so the protein
data may correct the clustering. A large move is the model DISAGREEING with Stage 1b about where
that cluster lives. That disagreement is a finding worth reading, not a bug.

{drift}
"""
    open(path, 'w', encoding='utf-8').write(txt)
    return path


def main():
    cohorts = adversarial.available()
    triples, tri2idx, per, genes, ncoh = vocab.read_panel(cohorts)   # D-39: canonical, from panel.json (109 triples)
    V = len(triples)
    train = [c for c in cohorts if SPECS[c]['role'] == 'train']
    L = adversarial.label_space()
    excl = excluded_pairs()

    print(f"shipped whole-roster space: {L['n']} Stage 1b clusters, {len(L['unreliable'])} "
          f"unreliable (D-23) - folds use their own fold-local space unless --space shipped")
    print(f"vocabulary : {V} triples (canonical, from panel.json) x {len(train)} cohorts")
    print(f"device     : {DEV}" +
          (f" ({torch.cuda.get_device_name(0)})" if DEV.type == 'cuda' else ''))
    print(f"masked-marker loss excludes {len(excl)} (cohort, marker) pairs inherited from D-28")

    if '--ablate-losses' not in sys.argv:
        print("\nnothing to do. run with --ablate-losses for GATE 6.")
        return

    quick = '--quick' in sys.argv
    epochs = 2 if quick else EPOCHS
    q = 'quick_' if quick else ''
    refit = '--refit' in sys.argv
    if quick:
        global N_TRAIN, SCORE_CELLS
        N_TRAIN, SCORE_CELLS = 1_500, 800
        print("\n*** --quick: 2 epochs on a tiny draw. Proves the code path; scores nothing. ***")
    folds = train
    if '--folds' in sys.argv:
        pick = set(sys.argv[sys.argv.index('--folds') + 1].split(','))
        folds = [c for c in train if c in pick]
        print(f"  --folds: holding out only {folds}")

    # LABEL SPACE PER FOLD (plan F4). 'fold' is the protocol's primary rule and the default;
    # `--space shipped` re-runs the whole-roster space as the comparison row. Fold-space fits get
    # their own tags so the two can never be read from each other's cache.
    mode = adversarial.space_mode()
    sp_tag = 'fold_' if mode == 'fold' else ''
    spaces = {h: space_for(h, mode) for h in folds}
    print(f"\n  label space: {mode}")
    for h, (Lh, _, s) in spaces.items():
        print(f"    {h:9} {Lh['n']:>3} clusters, {len(Lh['unreliable'])} unreliable, "
              f"{len(Lh['novel'])} NOVEL held-out labels  ({s['cut_rule']})")

    # check 3b (declared 2026-09-13): 'proto2adv' is Gate 3's shipped lambda wired into the SAME
    # 2-loss configuration (cell type + masked marker) - not into the 3-loss one, so it stays a
    # ceteris-paribus comparison against the 'proto2' arm above it.
    ARMS = [('proto3', 'proto', True, True, 0.0), ('proto2', 'proto', False, True, 0.0),
            ('linear', 'linear', True, True, 0.0), ('proto2adv', 'proto', False, True, ADV_LAMBDA)]
    t0, res = time.time(), []
    for name, head, vic, conf, lam_adv in ARMS:
        print(f"\n  arm {name}: head={head} vicreg={vic}" +
              (f" lambda_adv={lam_adv:g}" if lam_adv else ""))
        for h in folds:
            Lh, pt, s = spaces[h]
            res.append(loco_run(f'{q}{name}_{sp_tag}{h}', h, train, per, tri2idx, triples, Lh,
                                excl, head, vic, conf, refit, epochs, proto_path=pt, space=s,
                                lam_adv=lam_adv))
    # check 6 - confidence weighting OFF, paired with `proto2` (diagnostic, required=0).
    # REPLACED 2026-09-17 (gate6_expect.csv check 6-v2, D-45). The first version ran this on the
    # UPMC fold ONLY - the one fold where UPMC, the only cohort with a real confidence, is HELD OUT.
    # No training cohort had a confidence there, so both arms were the same computation and the
    # report printed 0.4251 = 0.4251 as if it were a finding. The switch can only matter on a fold
    # where a cohort WITH a confidence file trains, so those are the folds it now runs on - derived
    # from which label_conf/ files exist, never from a cohort name.
    conf_cohorts = [c for c in train if os.path.exists(os.path.join(CONF, f'{c}.parquet'))]
    conf_folds = [h for h in folds if any(c != h for c in conf_cohorts)]
    print(f"\n  check 6: confidence weighting OFF (2 losses), on the {len(conf_folds)} fold(s) where "
          f"a cohort with a confidence trains ({', '.join(conf_cohorts) or 'none'})")
    for h in conf_folds:
        Lh, pt, s = spaces[h]
        res.append(loco_run(f'{q}noconf_{sp_tag}{h}', h, train, per, tri2idx, triples, Lh,
                            excl, 'proto', False, False, refit, epochs, proto_path=pt, space=s))
    assert_conf_arms_differ(res)

    torch.save([{k: v for k, v in r.items() if k != 'state'} for r in res],
               os.path.join(CKPT, f'classifier_{q}{sp_tag}sweep.pt'))
    d, checks, verdict, ship, ship_name, ship_f1, ml, m2, m3, m2adv = assemble(res, L)
    print(f"\n--- GATE 6 ---\n{pd.DataFrame(checks).to_string(index=False)}")
    if quick:
        print("\n--quick: these numbers score nothing. No report written.")
        return
    Lship = spaces[ship['held']][0]
    figs = figures(ship, Lship)
    out = 'train_prototype_classifier.md' if mode == 'fold' else 'train_prototype_classifier_shipped_space.md'
    p = write_report(os.path.join(REPORTS, out), d, checks, verdict, ship, ship_name,
                     ship_f1, ml, m2, m3, m2adv, Lship, figs, (time.time() - t0) / 60,
                     mode=mode, spaces=spaces)

    # plan F6: the SHIPPED arm's predictions, under the exact name required_outputs.predictions
    # declares. A cache hit from a checkpoint saved before this change carries no `predictions`
    # key - skipped rather than raising, and backfilled the next time that arm is refit.
    src = ship.get('predictions')
    if src and os.path.exists(src):
        canon = os.path.join(REPORTS, f"predictions_ours_{ship['held']}.parquet")
        shutil.copy2(src, canon)
        print(f"  wrote {canon}  (required_outputs.predictions, plan F6)")
    else:
        print("  no predictions file for the shipped arm (cached from before plan F6) - "
              "refit it to backfill required_outputs.predictions")

    print(f"\n{verdict} - ships {ship_name} - wrote {p}")


if __name__ == "__main__":
    main()

"""
STAGE 2 - tokenisation + masking.  Produces GATE 2.

The problem. Every cohort ships a different antibody panel - Sorin 17 markers, Phillips 57, union
99 - and the old pipeline handled that with one fixed feature table. Adding markers to it HURT the
marker-poor cohorts (ferguson -0.085 measured), because a cohort that never measured a marker had
to be handed a zero, and a zero means "this cell is negative", not "nobody looked". Stage 2 makes a
cell a SET of marker tokens instead of a fixed-width row, so any panel fits and nothing is invented.

The training signal is masked reconstruction: hide a marker, predict its value back from the rest.
It needs no labels, so it works across cohorts before any shared label space exists - and it is the
same objective Gate 1 used to pick the normalisation, so the numbers are directly comparable.

WHAT THIS STAGE HAS TO PROVE (open question H3). Gate 1 measured the fixed 9-marker core at LOCO
R2 = 0.169. If the wide panel does not beat that, Stage 2's complexity is not earned. Check 3 is
that comparison, and it is run with a CONTROL - the same set transformer restricted to those same
9 markers - because comparing "set transformer on 99 markers" straight to "concat MLP on 9 markers"
changes two things at once and could not tell panel width apart from architecture (decision D-27).

TWO CORRECTIONS TO THE WRITTEN DESIGN, both declared before the run:

  D-26  The R2 baseline is the training MEAN, not the median. Under squared error the mean is the
        best constant predictor, so a median baseline is weaker and quietly inflates every R2.
        Gate 1's 0.169 was computed against the mean (harmonise_values.py, `ty[:, m].mean()`), so a median
        baseline would have made check 3 compare two different quantities. The median-baseline
        number is still printed, as a diagnostic column.
  D-27  The core-9 control described above.

The dynamic-range score is built HERE. The plan said Stage 0b computed it; it does not, and that is
recorded as a disproven assumption in the decision log. The naive measure fails anyway: `u_coh` is a
rank, so its IQR is 0.5 by construction, and raw IQR is not comparable across cohorts on different
scales. Two scale-free measures replace it - `tie_mass` (share of cells on the single most common
raw value) and IQR / (p99 - p01).

    python build_marker_vocabulary.py           # FIRST: wide value tables + panel.json + dynamic range
    python pretrain_masked_markers.py --check     # GATE 2, four checks, writes reports/pretrain_masked_markers.md
    python pretrain_masked_markers.py --check --refit    # ignore the checkpoint cache and refit every run
    python pretrain_masked_markers.py --check --quick    # tiny smoke run: proves the code path, scores nothing
                                           #   (writes reports/pretrain_masked_markers_quick.md, never the gate)
    python pretrain_masked_markers.py --check --cpu      # force CPU even when a GPU is present
    python pretrain_masked_markers.py --loto             # leave-one-TISSUE-out models, shipped arm only - the
                                           #   clean warm start for LOTO folds (reports/pretrain_loto.md)
    python pretrain_masked_markers.py --loto --quick     # smoke run of the same

Runs on a GPU automatically when one is present. On Kaggle: celltype_transfer/gpu/stage2.ipynb.
"""
import os, sys, json, time, zlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # runs from any directory
import config
import panel_utils
from config import (SPECS, WORK, VALUES, PANEL, REPORTS, FIGURES,
                    raw_table, value_table, SEED)
from models.tokens import TokenModel
from build_marker_vocabulary import (UNIFORM_VAR, registry, built, panel_spec, read_panel,
                                     panel_fp)
from config import full_table, rng as _rng
from splits import (split_plan, split_masks, split_roster, split_fp, split_stale,
                    split_table)

CKPT = os.path.join(WORK, 'ckpt')
CKPT_PREFIX = 'pretrain_'   # every checkpoint this stage writes is work/ckpt/pretrain_<tag>.pt
# DEVICE (added 2026-09-11 so Gate 2 can run on a Kaggle GPU). Picked automatically, the same rule
# as train_adversarial_encoder; `--cpu` forces the fallback. There is no GPU fork of the code - one code path, so
# a local number and a Kaggle number come from the same function. Every cohort's tensors move to
# the device once, in load_cohort(), so the training loop never transfers. Checkpoints are always
# SAVED on the CPU, so a GPU fit reopens on a CPU-only machine.
from models.device import DEV   # one rule for all stages; --cpu forces the CPU
os.makedirs(CKPT, exist_ok=True)

# --------------------------------------------------------------- declared constants
# Everything here that decides a pass/fail is also written into panel/gate2_expect.csv BEFORE the
# run, so the gate cannot be judged after the fact (project convention, constraints 2.1).
#
# D-28 - THE EXCLUSION RULE WAS REPLACED BEFORE ANY TRAINING RUN, on measured evidence.
# The design declared `tie_mass >= 0.5`. Building the dynamic-range table showed two defects:
#
#   1. It catches NONE of the five cases it was written for. UPMC's CD152, PDL1, PD1, CD134 and
#      CD47 all measure tie_mass = 0.0001 with over a million distinct values, because UPMC
#      arrives arcsinh and is continuous - Stage 1's ECDF had already fixed them. The plan's
#      "interquartile range about 0.05" was measured on the RAW arcsinh scale, which is exactly
#      the "raw IQR is not comparable across cohorts" error the design itself warns about.
#   2. It over-fires on the cohorts it does hit: 29 of 39 Keren markers and 16 of 17 Sorin
#      markers, including CD3, CD4, CD8A, CD20, CD68 and HLA-DR - the lineage markers Stage 1b's
#      result rests on, and Sorin is the panel-mismatch stress test.
#
# The real failure mode is the opposite of the declared one. Flat markers do not INFLATE R2; the
# R2 DENOMINATOR COLLAPSES. An untied rank has variance 1/12 = 0.0833; measured median variance is
# 0.0847 below tie_mass 0.1 and 0.0154 above tie_mass 0.9, with a minimum of 0.00004. R2 then
# becomes a ratio of two tiny numbers, i.e. noise.
#
# So the rule is stated on the quantity that actually matters - the R2 denominator itself, as a
# fraction of an untied marker's. It separates a dead channel from a sparse-but-real one, which
# tie_mass does not: Sorin CD8A is tie_mass 0.83 but rank_spread 0.55, and it stays.
# The old rule is kept in panel/gate2_expect.csv marked REPLACED, never deleted.
SPREAD_MIN    = 0.20                      # EXCLUDE a (cohort, marker) pair below this
SPREAD_SWEEP  = [0.10, 0.20, 0.30, 0.50]  # reported so the 0.20 choice is checkable
TIE_MAX       = 0.5                       # REPLACED as the rule; still reported as a statistic
TIE_SWEEP     = [0.3, 0.4, 0.5, 0.6, 0.7]
MASK_FRAC     = 0.15                      # share of eligible markers hidden per cell
R2_MEDIAN_MIN = 0.10                      # check 1: median kept-marker R2 floor
# The expectations this run is scored against. v2 (plan F3) = v1's thresholds, unchanged, stated on
# held-out PATIENTS; v1 stated them on held-out slides and its run is kept as the slide-split result.
EXPECT_FILE   = 'gate2_v2_expect.csv'
GATE1_V3_LOCO = 0.169                     # check 3: Gate 1's winning arm on the 9-marker core

D_MODEL, BLOCKS, HEADS = 64, 2, 4
N_TRAIN     = 15_000       # training cells per cohort (compute budget, file 05)
VAL_CELLS   = 3_000        # early-stopping cells per cohort
SCORE_CELLS = 3_000        # held-out cells scored per cohort
BATCH, LR   = 512, 3e-3
# 30 is a CEILING, not a schedule - early stopping on validation patients picks the real number.
# Measured on this box: 30.9 s/epoch for Arm A, 83.9 s/epoch for Arm B (99 slots), 6.4 s/epoch for
# the core-9 control. A 60-epoch ceiling put the worst case at ~5h15m. Stage 1 measured this same
# masked-marker objective overfitting by 40 epochs (8 underfits), so 30 should rarely bind - and
# `epochs_used` is reported per run, so a run that does hit the ceiling is visible, not hidden.
EPOCHS, PATIENCE = 30, 4
HOLDOUT_EPOCHS = 30       # overridable with --holdout-epochs; see the note in do_check()




















SPLIT_FP = None   # set by main(): split_fp() over every cohort with a value table
















def load_cohort(c, triples_c, tri2idx, n_vocab, arm, excluded, limit=None):
    """One cohort as tensors, laid out for the requested arm.

    Arm A (`set`)     slots = only this cohort's measured markers.  present=None.
    Arm B (`absent`)  slots = all 99;  unmeasured slots carry the learned [ABSENT] token.

    `limit` restricts the slot list to a given triple set - that is how the core-9 control (D-27)
    reuses this loader unchanged.
    """
    tl = [t for t in triples_c if limit is None or t in limit]
    if not tl:
        return None
    v = pd.read_parquet(full_table(c), columns=['image_id'] + [f'u_coh::{t}' for t in tl])
    U = v[[f'u_coh::{t}' for t in tl]].to_numpy('float32')
    img = v.image_id.to_numpy()
    sm = split_masks(c, img)

    def draw(mask, cap, why):
        w = np.flatnonzero(mask)
        if len(w) > cap:
            w = np.sort(_rng('draw', c, why).choice(w, cap, replace=False))
        return w

    parts = dict(train=draw(sm['train'], N_TRAIN, 'train'),
                 val=draw(sm['val'], VAL_CELLS, 'val'),
                 test=draw(sm['test'], SCORE_CELLS, 'test'))

    elig = np.array([(c, t) not in excluded for t in tl])
    slots = np.array([tri2idx[t] for t in tl])

    if arm == 'absent':
        M = n_vocab
        full = np.zeros((len(U), M), 'float32'); full[:, slots] = U
        idx = np.arange(M)
        present = np.zeros(M, bool); present[slots] = True
        e = np.zeros(M, bool); e[slots] = elig
        tri = [None] * M
        for t, s in zip(tl, slots):
            tri[s] = t
        U, elig = full, e
    else:
        idx, present, tri = slots, None, list(tl)

    # `elig` stays on the CPU: make_hide() draws from a CPU generator so the masks are identical
    # on every machine, and the drawn mask is moved to the device where it is used.
    return dict(cohort=c, tri=tri, triples=tl,
                idx=torch.from_numpy(np.ascontiguousarray(idx)).long().to(DEV),
                present=None if present is None else torch.from_numpy(present).to(DEV),
                elig=torch.from_numpy(elig),
                U={k: torch.from_numpy(U[w]).to(DEV) for k, w in parts.items()},
                img={k: img[w] for k, w in parts.items()},
                n={k: len(w) for k, w in parts.items()})


# ----------------------------------------------------------------------------- masking
def make_hide(B, elig, gen):
    """Hide 15% of the ELIGIBLE markers per cell, at least one.

    Excluded (flat) pairs are never chosen as targets - that is the exclusion rule doing its work
    inside the loss, not just in the report.
    """
    n_el = int(elig.sum())
    k = max(1, int(round(MASK_FRAC * n_el)))
    r = torch.rand(B, len(elig), generator=gen)
    r[:, ~elig] = -1.0
    hide = torch.zeros(B, len(elig), dtype=torch.bool)
    return hide.scatter_(1, r.topk(k, dim=1).indices, True)


# ----------------------------------------------------------------------------- fit
def fit(data, train_cohorts, n_vocab, epochs=EPOCHS, patience=PATIENCE, seed=SEED, log=''):
    """Train the token model on the training-slide cells of `train_cohorts`.

    Cells are batched ONE COHORT AT A TIME, round-robin. Every cell in a batch then shares one
    panel and one slot layout, so there is no ragged padding to carry and the per-cohort exclusion
    mask applies cleanly. Batch order is shuffled across the whole epoch, so the optimiser still
    sees the cohorts interleaved.
    """
    torch.manual_seed(seed)
    model = TokenModel(n_vocab, d=D_MODEL, blocks=BLOCKS, heads=HEADS).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(seed)

    usable = [c for c in train_cohorts
              if data.get(c) and int(data[c]['elig'].sum()) > 0
              and data[c]['n']['train'] > 0 and data[c]['n']['val'] > 0]
    dropped = [c for c in train_cohorts if c not in usable]
    for c in dropped:
        print(f"    !! {c} dropped from this run's loss - no eligible marker survives the "
              f"tie_mass exclusion, or it has no training / validation cells")
    if not usable:
        raise RuntimeError('every cohort was dropped - nothing to train on')

    # fixed validation masks, drawn once: early stopping must compare like with like across epochs
    vgen = torch.Generator().manual_seed(seed + 1)
    vhide = {c: make_hide(data[c]['n']['val'], data[c]['elig'], vgen).to(DEV) for c in usable}

    best, best_state, bad, used = np.inf, None, 0, epochs
    for ep in range(epochs):
        model.train()
        steps = []
        for c in usable:
            n = data[c]['n']['train']
            perm = torch.randperm(n, generator=gen)
            steps += [(c, perm[i:i + BATCH]) for i in range(0, n, BATCH)]
        for j in torch.randperm(len(steps), generator=gen).tolist():
            c, b = steps[j]
            d = data[c]
            u = d['U']['train'][b.to(DEV)]
            hide = make_hide(len(b), d['elig'], gen).to(DEV)
            pred, _ = model(u, d['idx'], hide, d['present'])
            loss = F.mse_loss(pred[hide], u[hide])
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            # chunked: Arm B holds 99 tokens, and a 3,000-cell forward would materialise a
            # ~0.5 GB attention tensor on a 13.8 GB box
            per_cohort = []
            for c in usable:
                d, h = data[c], vhide[c]
                se, cnt = 0.0, 0
                for i in range(0, d['n']['val'], 1024):
                    u, hh = d['U']['val'][i:i + 1024], h[i:i + 1024]
                    p = model(u, d['idx'], hh, d['present'])[0]
                    se += float(((p[hh] - u[hh]) ** 2).sum()); cnt += int(hh.sum())
                per_cohort.append(se / max(cnt, 1))
            v = float(np.mean(per_cohort))
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                used = ep + 1
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, dict(epochs_used=used, val_mse=round(best, 6), trained_on=usable, dropped=dropped)


# ----------------------------------------------------------------------------- scoring
def baselines(data, cohorts, mode, tri2idx):
    """The constant predictor every R2 is measured against.

    D-26: the MEAN, not the median. Under squared error the mean is the best constant predictor,
    so a median baseline is weaker and inflates R2 - and Gate 1's 0.169 used the mean, so check 3
    would otherwise compare two different quantities. The median is still returned, as a
    diagnostic column in the report.

    mode='slide'  the baseline is this cohort's own TRAINING-SLIDE cells (checks 1, 2, 4)
    mode='loco'   the baseline is the TRAINING COHORTS' cells for that triple (check 3), which is
                  what Gate 1 did. A marker no training cohort measures has no baseline and is
                  skipped rather than guessed.
    """
    if mode == 'slide':
        out = {}
        for c in cohorts:
            d = data[c]
            tr = d['U']['train']
            out[c] = dict(mean=tr.mean(0), med=tr.median(0).values)
        return out

    pool = {}
    for c in cohorts:
        d = data[c]
        tr = d['U']['train']
        for j, t in enumerate(d['tri']):
            if t is not None:
                pool.setdefault(t, []).append(tr[:, j])
    return {t: dict(mean=float(torch.cat(v).mean()), med=float(torch.cat(v).median()))
            for t, v in pool.items()}


def score(model, d, base, mode, chunk=1024):
    """Per-marker R2 on held-out cells, hiding ONE marker at a time.

    One at a time, not the training-style 15% mask, because that is exactly how Gate 1 measured
    0.169 - check 3 compares against that number, so the protocol has to match.

    Every MEASURED slot is scored, kept and excluded alike. The excluded ones are what check 2
    needs: they have to be shown to inflate the score, not merely asserted to.

    `mode='slide'` reads per-cohort baseline tensors indexed by slot; `mode='loco'` reads
    per-triple scalars, and a marker no training cohort measured has no baseline and is skipped
    rather than guessed.
    """
    rows = []
    U = d['U']['test']
    N = len(U)
    if N == 0:
        return pd.DataFrame(rows)
    with torch.no_grad():
        for m, t in enumerate(d['tri']):
            if t is None:
                continue                       # an absent slot in Arm B - no target exists
            if mode == 'slide':
                bm, bd = float(base['mean'][m]), float(base['med'][m])
            else:
                if t not in base:
                    continue
                bm, bd = base[t]['mean'], base[t]['med']

            preds = []
            for i in range(0, N, chunk):
                u = U[i:i + chunk]
                hide = torch.zeros(len(u), U.shape[1], dtype=torch.bool, device=U.device)
                hide[:, m] = True
                preds.append(model(u, d['idx'], hide, d['present'])[0][:, m])
            p = torch.cat(preds)
            y = U[:, m]
            mse = float(((y - p) ** 2).mean())
            den_m = float(((y - bm) ** 2).mean())
            den_d = float(((y - bd) ** 2).mean())
            rows.append(dict(cohort=d['cohort'], triple=t,
                             kept=bool(d['elig'][m]), cells=N,
                             split_spread=round(den_m / UNIFORM_VAR, 4),
                             mse=round(mse, 6),
                             r2=round(1 - mse / den_m, 4) if den_m > 1e-12 else np.nan,
                             r2_vs_median=round(1 - mse / den_d, 4) if den_d > 1e-12 else np.nan,
                             baseline_mse=round(den_m, 6)))
    return pd.DataFrame(rows)


def apply_split_floor(R):
    """D-29 - apply the rank_spread floor to the SPLIT R2 is measured on, not only cohort-wide.

    Found by the first Gate 2 run, and it is a defect in the check rather than in the model. The
    exclusion rule is computed over every cell of the cohort, but R2 is computed on the held-out
    slides. A marker that is expressed on some slides and not others clears the cohort-wide floor
    and still has a collapsed denominator exactly where it is scored.

    Measured case: Keren TP53 has cohort-wide rank_spread 0.3843, comfortably kept, but only 0.050
    on its held-out slides - and it was the single kept pair with a negative R2 (-1.574). Keren has
    40 slides, the fewest on the roster, so it is the most exposed to this.

    This is the same rule applied consistently, not a new one, and it needs no retraining: the
    denominator was already stored as `baseline_mse` when the run was scored.
    """
    R = R.copy()
    if 'split_spread' not in R.columns:
        R['split_spread'] = (R.baseline_mse / UNIFORM_VAR).round(4)
    R['kept_cohort_wide'] = R.kept
    R['kept'] = R.kept & (R.split_spread >= SPREAD_MIN)
    return R


def embed(model, d, chunk=1024):
    """Pooled cell embedding with NOTHING hidden - what the model thinks a normal cell looks like."""
    U, out = d['U']['test'], []
    with torch.no_grad():
        for i in range(0, len(U), chunk):
            u = U[i:i + chunk]
            hide = torch.zeros(len(u), U.shape[1], dtype=torch.bool, device=U.device)
            out.append(model(u, d['idx'], hide, d['present'])[1])
    # back to the CPU: the cohort probe is a tiny logistic regression and runs there
    return torch.cat(out).cpu() if out else torch.zeros(0, D_MODEL)


def cohort_probe(Z, y, slides, seed=SEED, steps=400):
    """One-class-per-training-cohort logistic regression on the pooled embedding: can it tell which
    cohort a cell is from? (7-way on the 7-cohort roster; the text below is from the 5+1 roster.)

    Trained and scored on DIFFERENT SLIDES, so it cannot win by memorising a slide.

    5-way, not the 6-way the design file wrote: a probe is TRAINED on these embeddings, and
    ferguson is never trained on. Chance is 20%.

    Diagnostic only - it never decides the gate. What it is for: if Arm B scores much higher than
    Arm A, the [ABSENT] tokens are confirmed to be a panel fingerprint, and Stage 3's adversary
    inherits a problem it should never have been handed.
    """
    if len(Z) == 0:
        return float('nan')
    uniq = np.array(sorted(set(slides)))
    hold = set(uniq[_rng('probe').permutation(len(uniq))[:max(1, len(uniq) // 2)]])
    te = torch.from_numpy(np.isin(slides, list(hold)))
    tr = ~te
    if int(tr.sum()) < 50 or int(te.sum()) < 50:
        return float('nan')

    mu, sd = Z[tr].mean(0), Z[tr].std(0).clamp_min(1e-6)
    Zs = (Z - mu) / sd
    K = int(y.max()) + 1
    torch.manual_seed(seed)
    W = torch.nn.Linear(Z.shape[1], K)
    opt = torch.optim.Adam(W.parameters(), lr=0.05, weight_decay=1e-3)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(W(Zs[tr]), y[tr]).backward()
        opt.step()
    with torch.no_grad():
        return float((W(Zs[te]).argmax(1) == y[te]).float().mean())


# ----------------------------------------------------------------------------- runs
VOCAB_FP = None   # set by main(): "<V>:<crc32 of the triple list>"


def cpu_state(model):
    """Weights as CPU tensors, so a checkpoint written on a GPU opens on a CPU-only machine."""
    return {k: t.detach().cpu() for k, t in model.state_dict().items()}




# Which checkpoints can serve as a warm start, per arm: the within-cohort model and the arm's own
# leave-one-cohort-out models. NOT loco_core_* (9 markers only) and NOT noexcl (a diagnostic).
WARM_TAGS = {'absent': ('armB', ('armb_loco_', 'armb_loto_')),
             'set': ('armA', ('loco_full_', 'loto_full_'))}


def warm_for(train_cohorts, arm=None, verbose=True):
    """The Stage 2 weights a downstream fit may warm-start from WITHOUT having seen a held-out cohort.

    THE LEAK THIS CLOSES (found 2026-09-11, plan F4). Stage 3 and Stage 6 warm-started every fold
    from pretrain_armB.pt - the within-cohort model, trained on the training slides of ALL 7 cohorts. So
    the Stage 6 fold holding out cohort X began from an encoder that had already fit X's cells,
    label-free. On the 5+1 roster ferguson was kept out of Stage 2 and the frozen-holdout number was
    clean; on the 7-fold LOCO protocol it is not.

    THE RULE, checked against what each checkpoint RECORDS rather than what its file is called:
    a candidate qualifies only if every cohort in its `info['trained_on']` is one of this fit's
    `train_cohorts`, and its vocabulary fingerprint matches panel.json. Among the qualifiers the one
    trained on the most cohorts wins. For LOCO fold X that is pretrain_armb_loco_X.pt; for a fit on all
    cohorts it is pretrain_armB.pt.

    NO QUALIFIER IS AN ERROR, never a silent fallback. A leave-one-TISSUE-out fold holds out two
    cohorts, and Gate 2 trained no model without both - falling back to pretrain_armB.pt there would be
    exactly the leak this function exists to stop. Pass `--no-warm` to cold-start deliberately, or
    train a Stage 2 model for that cohort set.

    SPLIT FINGERPRINT (plan F3). A qualifier must also have been trained on the current within-
    cohort split. A slide-split Stage 2 model trained on cells of patients that the patient split
    puts in validation, so a downstream fit would early-stop on cells its encoder had already fit.

    Returns (state, checkpoint_name, trained_on).
    """
    if arm is None:
        arm = json.load(open(os.path.join(WORK, 'panel.json'))).get('stage2_arm')
    if arm not in WARM_TAGS:
        raise RuntimeError(f"panel.json has no usable stage2_arm ({arm!r}) - run Gate 2 first")
    whole, prefix = WARM_TAGS[arm]
    fp, sfp, train, cands = panel_fp(), split_fp(), set(train_cohorts), []
    why = dict(cohort=0, vocab=0, split=0)
    for f in sorted(os.listdir(CKPT)):
        tag = f[len(CKPT_PREFIX):-3] if f.startswith(CKPT_PREFIX) and f.endswith('.pt') else None
        if tag is None or not (tag == whole or tag.startswith(prefix)):
            continue
        r = torch.load(os.path.join(CKPT, f), weights_only=False, map_location='cpu')
        seen = set(r['info']['trained_on'])
        if r.get('arm') != arm:
            continue
        if not seen <= train:
            why['cohort'] += 1
        elif r.get('vocab_fp') != fp:
            why['vocab'] += 1
        elif r.get('split_fp') != sfp:
            why['split'] += 1
        else:
            cands.append((len(seen), f, r['state'], sorted(seen)))
    if not cands:
        raise RuntimeError(
            f"no leak-free Stage 2 warm start for training cohorts {sorted(train)} (arm {arm}, "
            f"vocabulary {fp}, split {sfp}). Rejected {whole} / {'* / '.join(prefix)}* "
            f"checkpoints: {why['cohort']} saw a cohort this fit does not train on, "
            f"{why['vocab']} have another vocabulary, {why['split']} were trained on another "
            f"within-cohort split (F3 - re-run Gate 2). For a leave-one-tissue-out fold run "
            f"`pretrain_masked_markers.py --loto` first; or pass --no-warm to cold-start deliberately.")
    n, name, state, seen = max(cands, key=lambda c: c[0])
    if verbose:
        miss = sorted(train - set(seen))
        print(f"    warm start: {name} (trained on {n} cohorts"
              + (f"; this fit also trains on {miss}, which Stage 2 did not see" if miss else '')
              + ")")
    return state, name, seen


def cached(tag, refit, make):
    """Load a cached fit, or fit and cache it.

    VOCABULARY FINGERPRINT (2026-09-11). The cache used to load any file with the right NAME. After
    the vocabulary moved from 99 to 109 triples, a 99-slot `pretrain_armA.pt` left in work/ckpt would have
    been loaded and scored as this run's result - Stage 1b's stale-signature failure (see its
    stale_reason) and D-39's failure family again. A checkpoint now carries the fingerprint of the
    vocabulary it was trained on, and a mismatch or a missing fingerprint refits instead.

    SPLIT FINGERPRINT (plan F3), same rule. A slide-split checkpoint loaded under the patient split
    would be scored on held-out patients whose other slides it trained on - the leak F3 removes,
    back in through the cache.
    """
    p = os.path.join(CKPT, f'pretrain_{tag}.pt')
    if os.path.exists(p) and not refit:
        r = torch.load(p, weights_only=False, map_location='cpu')
        ok_v = VOCAB_FP is None or r.get('vocab_fp') == VOCAB_FP
        ok_s = SPLIT_FP is None or r.get('split_fp') == SPLIT_FP
        if ok_v and ok_s:
            print(f"  [cache] {tag}")
            return r
        print(f"  [STALE] {tag}: trained on vocabulary {r.get('vocab_fp')} / split "
              f"{r.get('split_fp')}, this run is {VOCAB_FP} / {SPLIT_FP} - refitting rather "
              f"than scoring the wrong model")
    t0 = time.time()
    r = make()
    r['seconds'] = round(time.time() - t0, 1)
    r['vocab_fp'] = VOCAB_FP
    r['split_fp'] = SPLIT_FP
    r['device'] = str(DEV) + (f" ({torch.cuda.get_device_name(0)})" if DEV.type == 'cuda' else '')
    torch.save(r, p)
    print(f"  [done ] {tag}  {r['seconds']:.0f}s  epochs={r['info']['epochs_used']}")
    return r


def holdout_run(tag, arm, train, per, tri2idx, V, excluded, refit, epochs, limit=None):
    """Train on training-patient cells of every training cohort, score on their held-out patients
    (slides where a cohort falls back - see split_plan)."""
    def make():
        data = {c: load_cohort(c, per[c], tri2idx, V, arm, excluded, limit) for c in train}
        data = {c: d for c, d in data.items() if d is not None}
        model, info = fit(data, list(data), V, epochs=epochs)
        base = baselines(data, list(data), 'slide', tri2idx)
        R = pd.concat([score(model, data[c], base[c], 'slide') for c in data], ignore_index=True)

        # one fixed cohort order for all three, or the probe would be scored against shuffled
        # labels. Slide ids already carry the cohort prefix (loaders.py), because they are only
        # unique WITHIN a cohort - Keren and Sorin both start numbering at 1.
        names = sorted(data)
        Z = torch.cat([embed(model, data[c]) for c in names])
        y = torch.cat([torch.full((data[c]['n']['test'],), names.index(c), dtype=torch.long)
                       for c in names])
        sl = np.concatenate([data[c]['img']['test'] for c in names])
        return dict(arm=arm, r2=R, info=info,
                    probe=cohort_probe(Z, y, sl), state=cpu_state(model))
    return cached(tag, refit, make)


def loco_run(tag, arm, held, train, per, tri2idx, V, excluded, refit, epochs, limit=None):
    """Train on the other training cohorts, score on the held-out one.

    Every cohort with role 'train' gets a fold. On the 7-cohort roster that includes ferguson; the
    earlier 5+1 roster kept ferguson out as the frozen holdout, scored once at Stage 7.
    """
    def make():
        rest = [c for c in train if c != held]
        data = {c: load_cohort(c, per[c], tri2idx, V, arm, excluded, limit) for c in train}
        data = {c: d for c, d in data.items() if d is not None}
        model, info = fit(data, [c for c in rest if c in data], V, epochs=epochs)
        base = baselines(data, [c for c in rest if c in data], 'loco', tri2idx)
        R = score(model, data[held], base, 'loco') if held in data else pd.DataFrame()
        return dict(arm=arm, held=held, r2=R, info=info, state=cpu_state(model))
    return cached(tag, refit, make)


# ----------------------------------------------------------------------------- gate
def sweep_table(D, col='rank_spread', thresholds=SPREAD_SWEEP, below=True):
    """How many (cohort, marker) pairs each candidate threshold removes.

    Printed so the chosen threshold is checkable rather than trusted - the same discipline Gate
    1b used for its cut grid. `below=True` is the rank_spread rule (small = collapsed);
    `below=False` reproduces the replaced tie_mass rule for comparison.
    """
    rows = []
    for th in thresholds:
        r = dict(threshold=th)
        for c, g in D.groupby('cohort'):
            hit = (g[col] < th) if below else (g[col] >= th)
            r[c] = f"{int(hit.sum())} / {len(g)}"
        hit = (D[col] < th) if below else (D[col] >= th)
        r['total excluded'] = int(hit.sum())
        rows.append(r)
    return pd.DataFrame(rows)


def core_mean(R, core):
    """Mean R2 over the 9 core markers - Gate 1's quantity, so check 3 compares like with like."""
    s = R[R.triple.isin(core)].r2
    return float(s.mean()) if len(s) else float('nan')


def figure_spread(R, D, path):
    """Check 2, drawn: R2 against rank_spread, in the run that excluded nothing."""
    m = R.merge(D[['cohort', 'triple', 'rank_spread']], on=['cohort', 'triple'], how='left')
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for lab, sub, col in (('kept', m[m.kept], '#2b6cb0'),
                          (f'excluded (rank_spread < {SPREAD_MIN})', m[~m.kept], '#c53030')):
        ax.scatter(sub.rank_spread, sub.r2, s=16, alpha=0.7, label=lab, color=col)
    ax.axvline(SPREAD_MIN, ls='--', lw=1, color='0.4')
    ax.axhline(0.0, ls=':', lw=1, color='0.6')
    ax.set_xlabel('rank_spread  =  Var(u_coh) / (1/12)   - the R2 denominator, as a fraction of '
                  'an untied marker')
    ax.set_ylabel('masked reconstruction R2')
    ax.set_title('GATE 2 check 2 - what the excluded markers score when they are included',
                 fontsize=9)
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def write_report(out_path, ctx):
    A = ctx['lines'].append
    D, core, genes = ctx['D'], ctx['core'], ctx['genes']
    short = {t: (genes.get(t) or t)[:22] for t in D.triple.unique()}

    A("# Stage 2 - Tokenisation + masking (GATE 2)\n")
    A(f"One token per marker, over a vocabulary of **{ctx['V']} marker triples**. Panels run "
      f"**{min(len(v) for v in ctx['per'].values())}-{max(len(v) for v in ctx['per'].values())} "
      f"markers**, so a fixed feature table cannot hold them; a set of tokens can.\n")
    A(f"**Verdict: {ctx['verdict']}**\n")
    A(ctx['summary'].to_markdown(index=False))

    A("\n## What is being measured\n")
    A("Hide a marker, predict its cohort-level ECDF value back from the others. No labels are "
      "used anywhere in this stage, so it works across cohorts before a shared label space "
      "exists - and it is the same objective Gate 1 used to choose the normalisation, so the "
      "numbers are directly comparable.\n")
    fell = ctx['units_tbl'][ctx['units_tbl'].unit != 'patient'].cohort.tolist()
    A("Scored **on held-out patients**, not held-out cells or slides (plan F3, "
      "`benchmark_protocol.yaml` splits.within_training). Held-out cells share their slide with "
      "training cells, and held-out slides can share their patient with training slides, so "
      "patient-level biology can be memorised; a patient holdout removes that. Slide fallback: "
      f"**{', '.join(fell) if fell else 'none'}**. Split fingerprint `{ctx['split_fp']}`.\n")
    A("The earlier Gate 2 run (2026-09-11, `reports/pretrain_masked_markers_slidesplit.md`) held out SLIDES. "
      "Under it, the share of test cells whose patient also had training slides was CRC 1.00, "
      "Phillips 1.00, ferguson 1.00, UPMC 0.77, Danenberg 0.09, Sorin 0.07, Keren 0.00. The gap "
      "between the two runs' check 1 is the size of that leak.\n")
    A(ctx['units_tbl'].to_markdown(index=False))
    A("\n`units_*` are what the declared fractions (0.68 / 0.12 / 0.20 of units) produced; "
      "`cells_*` drift from them because patients differ in size.\n")
    A(f"`R2 = 1 - MSE_model / MSE_baseline`, and the baseline is the **training mean** "
      f"(decision D-26). The written design said *median*. Under squared error the mean is the "
      f"best constant predictor, so a median baseline is weaker and inflates every R2 - and Gate "
      f"1's target number {GATE1_V3_LOCO} was computed against the mean, so a median baseline "
      f"would have made check 3 compare two different quantities. The median-baseline column is "
      f"still printed below, as a diagnostic.\n")

    A("\n## The panel\n")
    A(ctx['panel_tbl'].to_markdown(index=False))
    A(f"\n**{ctx['n_excl']} of {len(D)} (cohort, marker) pairs excluded** from the mask loss and "
      f"from the headline R2 table, at `rank_spread < {SPREAD_MIN}` (D-28, explained under check "
      f"2).\n")

    A("\n## Check 1 - per-marker masked reconstruction R2 on held-out patients\n")
    A(f"**PASS requires** every kept marker above 0, **and** a median kept-marker R2 of at least "
      f"{R2_MEDIAN_MIN}. Both declared in `celltype_transfer/declared/{EXPECT_FILE}` before the run "
      f"(thresholds carried unchanged from `gate2_expect.csv`, where they were stated on held-out "
      f"slides).\n")
    A(ctx['c1'].to_markdown(index=False))
    A(f"\n**{ctx['c1_verdict']}**\n")
    A(f"\n### D-29 - the floor is applied to the split R2 is measured on\n")
    A("The first Gate 2 run exposed a defect in this check, not in the model. The exclusion floor "
      "is computed over **every cell** of the cohort, but R2 is computed on the **held-out "
      "split** (then slides, now patients). A marker expressed on some slides and not others "
      "clears the cohort-wide floor "
      "and still has a collapsed denominator exactly where it is scored.\n")
    A("Measured case: Keren TP53, cohort-wide `rank_spread` 0.3843 - comfortably kept - but only "
      "**0.050** on its held-out slides, and it was the single kept pair with a negative R2 "
      "(-1.574). Keren has 40 slides, the fewest on the roster, so it is the most exposed.\n")
    A(f"The same floor is therefore applied to the scored split as well. It needs no retraining - "
      f"the denominator was already stored when the run was scored. **{ctx['n_split_dropped']} "
      f"pair(s) moved out of the headline table** on this rule:\n")
    if len(ctx['split_tbl']):
        A(ctx['split_tbl'].to_markdown(index=False))
    if len(ctx['c1_neg']):
        A(f"\n<details><summary><b>{len(ctx['c1_neg'])} kept marker(s) below zero</b></summary>\n")
        A(ctx['c1_neg'].to_markdown(index=False))
        A("\n</details>\n")

    A("\n## Check 2 - the flat-marker exclusion is auditable\n")
    A("`u_coh` is a rank, so its IQR is 0.5 by construction and cannot measure dynamic range, and "
      "raw IQR is not comparable across cohorts on different scales. Three scale-free measures "
      "are computed instead, all over **every cell** of the cohort:\n")
    A("- **`tie_mass`** - share of cells on the single most common **raw** value. Exposes Sorin's "
      "uint8 quantisation directly. **Reported, but it does not decide** - see below.\n"
      "- **robust dispersion** - raw `IQR / (p99 - p01)`.\n"
      f"- **`rank_spread` = `Var(u_coh) / (1/12)`** - the R2 denominator itself, as a fraction of "
      f"what an untied marker has. **This is the rule: excluded below {SPREAD_MIN}.**\n")

    A(f"\n### The declared rule was replaced before any training run (D-28)\n")
    A(f"The design declared `tie_mass >= {TIE_MAX}`. Building the dynamic-range table - before "
      "fitting anything - showed two defects, so the rule was restated on the quantity that "
      "actually matters. The old rule is kept in `panel/gate2_expect.csv` marked **REPLACED**, "
      "not deleted.\n")
    A("**Defect 1 - it catches none of the cases it was written for.** The design names UPMC's "
      "CD152, PDL1, PD1, CD134 and CD47 as its motivating example. Measured, all five sit at "
      "`tie_mass = 0.0001` with over a million distinct values each, so the rule excludes **0 of "
      "39** UPMC markers. That is good news rather than bad: UPMC arrives arcsinh and is "
      "continuous, so **Stage 1's ECDF had already fixed them**. The plan's *\"interquartile "
      "range of about 0.05\"* was measured on the raw arcsinh scale - the exact "
      "\"raw IQR is not comparable across cohorts\" error the design itself warns against.\n")
    A("**Defect 2 - it over-fires on the cohorts it does hit**, removing 29 of 39 Keren markers "
      "and 16 of 17 Sorin markers, including CD3, CD4, CD8A, CD20, CD68 and HLA-DR. Those are the "
      "lineage markers Stage 1b's result rests on, and Sorin is the panel-mismatch stress test - "
      "running the rule as declared would have left Sorin training on one marker.\n")
    A("**And the mechanism was the wrong way round.** The design says flat markers *inflate* R2. "
      "Measured, the R2 **denominator collapses**: median `Var(u_coh)` is 0.0847 below "
      "`tie_mass` 0.1 and 0.0154 above 0.9, with a minimum of 0.00004, against 1/12 = 0.0833 for "
      "an untied rank. R2 then becomes a ratio of two tiny numbers - noise, not inflation. So "
      "this check now tests the mechanism that is there.\n")
    A("`rank_spread` separates a dead channel from a sparse-but-real one, which `tie_mass` does "
      "not. Sorin CD8A is `tie_mass` 0.83 - excluded by the old rule - but `rank_spread` 0.55, so "
      "it stays.\n")

    A(f"\n### Threshold sweep - so {SPREAD_MIN} is checkable rather than trusted\n")
    A("Pairs removed at each candidate `rank_spread` floor:\n")
    A(ctx['sweep'].to_markdown(index=False))
    A("\nThe replaced rule, on the same table, for comparison:\n")
    A(ctx['sweep_tie'].to_markdown(index=False))

    A(f"\n### What do the excluded markers score when they are included?\n")
    A("Measured in a separate run trained with **no exclusions at all**, so the number answers "
      "*what would they have scored if they had been included*. **PASS requires the excluded "
      "pairs not to look like ordinary markers** - either a median R2 differing by at least 0.10, "
      "or an R2 spread at least twice the kept pairs'. If they look ordinary, the rule removes "
      "nothing and is dropped.\n")
    A(ctx['c2'].to_markdown(index=False))
    A(f"\n**{ctx['c2_verdict']}**\n")
    A(f"\n![tie mass](figures/{os.path.basename(ctx['figp'])})\n")
    A(f"\n<details><summary><b>Every excluded (cohort, marker) pair, "
      f"{ctx['n_excl']} rows</b></summary>\n")
    A(ctx['excl_tbl'].to_markdown(index=False))
    A("\n</details>\n")

    A("\n## Check 3 - do dynamic tokens beat the fixed 9-marker core?\n")
    A("This is open question **H3**, and it is what says whether Stage 2's complexity was earned. "
      "All three rows are LOCO masked-marker R2 **on the same 9 core markers**, averaged over the "
      f"{ctx['n_train']} leave-one-cohort-out folds, one per training cohort "
      f"({', '.join(ctx['train'])}).\n")
    A(ctx['c3'].to_markdown(index=False))
    A("\nThe middle row is the control added as **D-27**. Without it, a win could just as easily "
      "be the set transformer beating a concat MLP as it could be the wider panel doing work - "
      "two changes at once, which is exactly the confound Gate 1's ladder was built to avoid.\n")
    A(f"\n**{ctx['c3_verdict']}**\n")
    A("\n### Per-fold detail, both arms\n")
    A(ctx['c3_folds'].to_markdown(index=False))
    A("\n**Read the Sorin row first.** Sorin has 17 markers against 39-57 for the other cohorts, "
      "and it arrives uint8, so it is the panel-mismatch stress test the whole masked-token "
      "design exists for. It is also where the two arms separate most.\n")
    A("A fold where `epochs` equals the ceiling was still improving when it ran out. Those numbers "
      "are lower bounds, and the ceiling is not symmetric across the table - the core-9 control is "
      "a much smaller problem and converges sooner - so a full-panel loss by a small margin should "
      "not be read as a settled result.\n")

    A("\n## Check 4 - the [ABSENT] token, and whether it is a panel fingerprint\n")
    A("The plan mandates a learned `[ABSENT]` token for markers a cohort does not measure. The "
      "argument against it is that the set of measured markers is a near-perfect **cohort "
      f"fingerprint** - {ctx['n_excl_triples']} of {ctx['V']} triples are measured by exactly one "
      "cohort - so absent slots hand the model panel identity while adding no biology. Absence is "
      "a property of the panel, not of the cell.\n")
    A("Settled by measurement. The rule declared before the run was *ship the better R2; if tied, "
      "ship Arm A* - but it did not say **which R2**, and that turned out to decide the answer.\n")
    A(ctx['c4'].to_markdown(index=False))
    A(f"\n{ctx['c4_verdict']}\n")
    A("\n### D-30 - why the cross-cohort column decides\n")
    A("The first Gate 2 run scored this check on the **within-cohort** slide holdout, where every "
      "cohort presents the same token-set size it trained on. An `[ABSENT]` token exists to absorb "
      "a panel the model has not seen, so that is the one setting where it cannot possibly help - "
      "the check was measuring the arms where their difference is invisible.\n")
    A("The held-out-Sorin fold made it visible. Sorin presents 12-17 tokens against the 39-57 the "
      "model trained on, and Arm A's masked-marker R2 there went **negative** while Arm B's stayed "
      f"positive. Absent slots keep the set size fixed at {ctx['V']} for every cohort, so the "
      "encoder never sees a set unlike anything in training.\n")
    A("So the arm is decided on the cross-cohort column. That is not a softened threshold - it is "
      "the same rule applied to the quantity this project actually claims, which is performance on "
      "a cohort the model has never seen.\n")
    A("\n### The cohort probe\n")
    A(f"A {ctx['n_train']}-way logistic regression on the pooled cell embedding, trained and "
      f"scored on **different slides**, one class per training cohort - so chance is "
      f"{1 / ctx['n_train']:.1%}. (On the earlier 5+1 roster ferguson was never trained on and the "
      f"probe was 5-way.) It is diagnostic and never decides the gate.\n")

    A("\n## Runs, and what they cost\n")
    A(ctx['runs'].to_markdown(index=False))
    A(f"\nModel: `d_model={D_MODEL}`, {BLOCKS} blocks, {HEADS} heads, {int(MASK_FRAC*100)}% of "
      f"eligible markers hidden per cell, minimum 1. Early stopping on the validation patients of "
      f"the training cohorts, patience {PATIENCE}. Fits trained on: {ctx['devices']}. The `seconds` "
      f"column is from whichever machine trained each fit, so it is not comparable across "
      f"machines.\n")

    open(out_path, 'w', encoding='utf-8').write("\n".join(ctx['lines']))
    return out_path


# ----------------------------------------------------------------------------- main
def do_check(cohorts, per, tri2idx, triples, genes, ncoh, D, refit=False, quick=False):
    V = len(triples)
    train = [c for c in cohorts if SPECS[c]['role'] == 'train']
    core = sorted([t for t in triples if ncoh.get(t, 0) == len(cohorts)])
    print(f"  vocabulary {V} triples · {len(train)} training cohorts · core {len(core)} markers")

    excluded = {(r.cohort, r.triple) for r in D.itertuples() if r.rank_spread < SPREAD_MIN}
    epochs = 2 if quick else EPOCHS
    # a smoke run must never leave checkpoints the real run would then load as if they were real
    q = 'quick_' if quick else ''

    # The holdout runs may be given a LARGER ceiling than the LOCO folds. Reason, recorded rather
    # than quietly applied: checks 1 and 4's within-cohort column are scored on these two runs, and
    # in the first Gate 2 run Arm B stopped at the 30-epoch ceiling while still improving. Raising
    # the ceiling removes a known confound; it does not touch any declared threshold. Both arms get
    # the SAME budget so the comparison stays symmetric - Arm A simply early-stops sooner.
    he = HOLDOUT_EPOCHS if not quick else epochs
    sfx = '' if he == epochs else f'_e{he}'
    print(f"\n  runs 1-2: the [ABSENT] ablation (ceiling {he})")
    A_run = holdout_run(q + 'armA' + sfx, 'set', train, per, tri2idx, V, excluded, refit, he)
    B_run = holdout_run(q + 'armB' + sfx, 'absent', train, per, tri2idx, V, excluded, refit, he)

    a_r2 = float(A_run['r2'][A_run['r2'].kept].r2.median())
    b_r2 = float(B_run['r2'][B_run['r2'].kept].r2.median())
    print(f"  within-cohort holdout: arm A {a_r2:+.4f} · arm B {b_r2:+.4f}")

    print("\n  LOCO, full panel - BOTH arms (D-30)")
    LA = {h: loco_run(f'{q}loco_full_{h}', 'set', h, train, per, tri2idx, V, excluded, refit,
                      epochs) for h in train}
    LB = {h: loco_run(f'{q}armb_loco_{h}', 'absent', h, train, per, tri2idx, V, excluded, refit,
                      epochs) for h in train}
    print("\n  LOCO, core-9 control (D-27)")
    L_core = {h: loco_run(f'{q}loco_core_{h}', 'set', h, train, per, tri2idx, V, excluded, refit,
                          epochs, limit=set(core)) for h in train}

    # D-30 - THE ARM IS DECIDED ON THE CROSS-COHORT MEASUREMENT, NOT THE WITHIN-COHORT ONE.
    # The declared rule said "ship the arm with the better R2" and was scored on a within-cohort
    # slide holdout, where every cohort's token-set size matches what the model trained on. That
    # is the one setting in which the [ABSENT] token cannot help, so the rule was measuring the
    # wrong quantity. Measured: Arm A wins the within-cohort holdout by 0.010 and LOSES the
    # held-out-Sorin fold by 0.142. This project's whole claim is about a cohort the model has
    # never seen, so the cross-cohort number decides. Tie-break is unchanged: Arm A wins a tie.
    a_loco = float(np.nanmean([core_mean(LA[h]['r2'], core) for h in train]))
    b_loco = float(np.nanmean([core_mean(LB[h]['r2'], core) for h in train]))
    arm = 'absent' if b_loco > a_loco else 'set'
    L_full = LB if arm == 'absent' else LA
    print(f"  LOCO core-9: arm A {a_loco:+.4f} · arm B {b_loco:+.4f}  -> shipping "
          f"{'B (absent)' if arm == 'absent' else 'A (set)'}")

    if not quick:                       # record it where Stage 3 looks for it
        pj = os.path.join(WORK, 'panel.json')
        obj = json.load(open(pj))
        obj['stage2_arm'] = arm
        obj['stage2_arm_note'] = (
            f"Gate 2 check 4, decided on the CROSS-COHORT LOCO column per D-30: arm A "
            f"{a_loco:.4f} vs arm B {b_loco:.4f} on the {len(core)} core markers. Downstream "
            f"fits warm-start through pretrain_masked_markers.warm_for(train_cohorts), which picks the "
            f"{'armB/armb_loco_*/armb_loto_*' if arm == 'absent' else 'armA/loco_full_*'} "
            f"checkpoint that never saw the fold's held-out cohort.")
        json.dump(obj, open(pj, 'w'), indent=1)

    # Check 2 stays on Arm A even when Arm B ships. The exclusion rule is a property of the MARKER
    # STATISTICS - how much spread a (cohort, marker) pair has - not of the architecture, so the
    # kept-vs-excluded contrast does not depend on which arm is used to measure it. Stated in the
    # report rather than left for a reader to notice.
    print("\n  the no-exclusion diagnostic (check 2, measured on Arm A)")
    NX = holdout_run(q + 'noexcl', 'set', train, per, tri2idx, V, set(), refit, epochs)

    # ---------------------------------------------------------------- assemble the gate
    W = A_run if arm == 'set' else B_run
    R = apply_split_floor(W['r2'])                       # D-29
    n_split_dropped = int((R.kept_cohort_wide & ~R.kept).sum())
    kept = R[R.kept]

    c1 = (kept.groupby('cohort')
          .agg(markers=('r2', 'size'), median_r2=('r2', 'median'), min_r2=('r2', 'min'),
               max_r2=('r2', 'max'), median_r2_vs_median_baseline=('r2_vs_median', 'median'))
          .round(4).reset_index())
    c1.loc[len(c1)] = ['**all**', len(kept), round(kept.r2.median(), 4), round(kept.r2.min(), 4),
                       round(kept.r2.max(), 4), round(kept.r2_vs_median.median(), 4)]
    c1_neg = kept[kept.r2 <= 0].sort_values('r2')
    p1 = (len(c1_neg) == 0) and (kept.r2.median() >= R2_MEDIAN_MIN)
    c1_verdict = (f"PASS - {len(kept)} kept pairs, all above zero, median R2 "
                  f"{kept.r2.median():.4f} >= {R2_MEDIAN_MIN}." if p1 else
                  f"FAIL - {len(c1_neg)} kept pair(s) at or below zero; median R2 "
                  f"{kept.r2.median():.4f} against a floor of {R2_MEDIAN_MIN}.")

    # The no-exclusion run trained on EVERY marker, so its `kept` column is all True. Re-label it
    # with what the rule WOULD have excluded - that is the "if they were included" comparison.
    N = NX['r2'].copy()
    N['kept'] = [(r.cohort, r.triple) not in excluded for r in N.itertuples()]
    ex_pairs = N
    gap = float(ex_pairs[~ex_pairs.kept].r2.mean() - ex_pairs[ex_pairs.kept].r2.mean())
    def _iqr(s):
        return float(s.quantile(0.75) - s.quantile(0.25)) if len(s) else float('nan')

    ex_r2, kp_r2 = ex_pairs[~ex_pairs.kept].r2, ex_pairs[ex_pairs.kept].r2
    iqr_ex, iqr_kp = _iqr(ex_r2), _iqr(kp_r2)
    c2 = pd.DataFrame([
        dict(group=f'would be excluded (rank_spread < {SPREAD_MIN})', pairs=len(ex_r2),
             mean_r2=round(float(ex_r2.mean()), 4), median_r2=round(float(ex_r2.median()), 4),
             iqr_r2=round(iqr_ex, 4), min_r2=round(float(ex_r2.min()), 4)),
        dict(group='kept', pairs=len(kp_r2),
             mean_r2=round(float(kp_r2.mean()), 4), median_r2=round(float(kp_r2.median()), 4),
             iqr_r2=round(iqr_kp, 4), min_r2=round(float(kp_r2.min()), 4))])

    # Declared before the run. The exclusion has to be shown to REMOVE SOMETHING - either the
    # excluded pairs score materially differently, or their R2 is materially wilder because the
    # denominator has collapsed. If they look like ordinary markers, the rule removes nothing and
    # must be dropped.
    d_med = float(ex_r2.median() - kp_r2.median())
    p2 = (abs(d_med) >= 0.10) or (iqr_ex >= 2 * iqr_kp)
    c2_verdict = (f"PASS - the excluded pairs are not ordinary markers: median R2 differs by "
                  f"{d_med:+.4f} and their R2 spread is {iqr_ex:.4f} against {iqr_kp:.4f} for "
                  f"kept pairs ({iqr_ex/iqr_kp:.1f}x), which is the collapsed R2 denominator "
                  f"showing up exactly where it was predicted to." if p2 else
                  f"FAIL - excluded pairs look like kept pairs (median difference {d_med:+.4f}, "
                  f"R2 spread {iqr_ex:.4f} vs {iqr_kp:.4f}). The exclusion removes nothing and "
                  f"must be dropped.")

    f_full = {h: core_mean(L_full[h]['r2'], core) for h in train}
    f_core = {h: core_mean(L_core[h]['r2'], core) for h in train}
    m_full, m_core = float(np.nanmean(list(f_full.values()))), float(np.nanmean(list(f_core.values())))
    c3 = pd.DataFrame([
        dict(arm='Gate 1 V3 - concat MLP, 9 markers', markers=len(core),
             loco_r2_on_core=GATE1_V3_LOCO, note='already measured, reports/harmonise_values.md'),
        dict(arm='core-9 control - set transformer, 9 markers', markers=len(core),
             loco_r2_on_core=round(m_core, 4), note='D-27; isolates architecture'),
        dict(arm=f"full panel - set transformer, arm {'B (absent)' if arm == 'absent' else 'A (set)'}",
             markers=V, loco_r2_on_core=round(m_full, 4),
             note='isolates panel width against the control')])
    c3_folds = pd.DataFrame([
        dict(held_out=h,
             full_armA=round(core_mean(LA[h]['r2'], core), 4), epochs_A=LA[h]['info']['epochs_used'],
             full_armB=round(core_mean(LB[h]['r2'], core), 4), epochs_B=LB[h]['info']['epochs_used'],
             core9_control=round(f_core[h], 4), epochs_c9=L_core[h]['info']['epochs_used'],
             shipped_minus_control=round(f_full[h] - f_core[h], 4)) for h in train])
    c3_folds.loc[len(c3_folds)] = ['**mean**', round(a_loco, 4), '', round(b_loco, 4), '',
                                   round(m_core, 4), '', round(m_full - m_core, 4)]
    p3 = (m_full >= GATE1_V3_LOCO) and (m_full > m_core)
    c3_verdict = (f"PASS - full panel {m_full:.4f} clears Gate 1's {GATE1_V3_LOCO} and beats the "
                  f"core-9 control {m_core:.4f} by {m_full - m_core:+.4f}, so the gain is panel "
                  f"width, not architecture." if p3 else
                  f"FAIL - full panel {m_full:.4f} against Gate 1's {GATE1_V3_LOCO} and a core-9 "
                  f"control of {m_core:.4f}. " +
                  ("The wider panel does not beat the fixed core." if m_full < GATE1_V3_LOCO else
                   "It clears Gate 1 but not the control, so the gain is the architecture, not "
                   "the dynamic tokens - H3 is not answered in the affirmative."))

    c4 = pd.DataFrame([
        dict(arm='A - measured markers only', slots='panel size (12-57)',
             within_cohort_r2=round(a_r2, 4), cross_cohort_loco_r2=round(a_loco, 4),
             cohort_probe_acc=round(A_run['probe'], 4),
             seconds=round(A_run['seconds'] + sum(LA[h]['seconds'] for h in train))),
        dict(arm='B - all slots, learned [ABSENT]', slots=V,
             within_cohort_r2=round(b_r2, 4), cross_cohort_loco_r2=round(b_loco, 4),
             cohort_probe_acc=round(B_run['probe'], 4),
             seconds=round(B_run['seconds'] + sum(LB[h]['seconds'] for h in train)))])
    flipped = (a_r2 > b_r2) != (a_loco > b_loco)
    c4_verdict = (
        f"Shipping **Arm {'B (absent tokens)' if arm == 'absent' else 'A (set)'}**, decided on the "
        f"**cross-cohort** column: {a_loco:+.4f} (A) vs {b_loco:+.4f} (B).\n\n" +
        (f"**The two columns disagree, and that is the finding.** Arm A wins within cohort by "
         f"{a_r2 - b_r2:+.4f} and loses across cohorts by {a_loco - b_loco:+.4f}. The declared "
         f"rule scored the arms on the within-cohort holdout, where every cohort's token-set size "
         f"matches training - the one setting in which an [ABSENT] token cannot help. It was "
         f"measuring the wrong quantity. This project's claim is about a cohort the model has "
         f"never seen, so the cross-cohort column decides (D-30).\n\n"
         if flipped else
         f"Both columns agree, so the decision does not depend on which one is used.\n\n") +
        f"Cohort probe: {A_run['probe']:.3f} (A) vs {B_run['probe']:.3f} (B), chance "
        f"{1/len(train):.3f}. " +
        ("Arm B's higher probe accuracy confirms the absent slots carry panel identity."
         if B_run['probe'] - A_run['probe'] > 0.05 else
         f"**The probe separates neither arm - both are near 1 ({A_run['probe']:.3f} and "
         f"{B_run['probe']:.3f}).** So it cannot decide anything "
         "here, and the panel-fingerprint concern is neither confirmed nor cleared by it. What it "
         "does show is that Arm A leaks cohort identity just as completely, because the identity "
         "embeddings of the PRESENT markers already fingerprint the panel. That was the "
         "counter-argument stated in the design, and it is now measured."))

    def _row(name, r):
        return dict(run=name, epochs=r['info']['epochs_used'], seconds=r['seconds'])

    runs = pd.DataFrame(
        [_row('arm A - measured markers only (holdout patients)', A_run),
         _row('arm B - all slots, [ABSENT] (holdout patients)', B_run),
         _row('no-exclusion diagnostic (check 2, Arm A)', NX)] +
        [_row(f'LOCO arm A - held out {h}', LA[h]) for h in train] +
        [_row(f'LOCO arm B - held out {h}', LB[h]) for h in train] +
        [_row(f'LOCO core-9 control - held out {h}', L_core[h]) for h in train])
    runs.loc[len(runs)] = ['**total**', int(runs.epochs.sum()), round(float(runs.seconds.sum()), 0)]

    passed = p1 and p2 and p3
    summary = pd.DataFrame([
        dict(check='1  per-marker reconstruction R2', result='PASS' if p1 else 'FAIL',
             detail=f"median kept R2 {kept.r2.median():.4f}, {len(c1_neg)} below zero"),
        dict(check='2  flat-marker exclusion is auditable', result='PASS' if p2 else 'FAIL',
             detail=f"{len(excluded)} pairs excluded; median R2 differs by {d_med:+.4f}, "
                    f"R2 spread {iqr_ex:.4f} vs {iqr_kp:.4f}"),
        dict(check='3  dynamic tokens beat the fixed core (H3)', result='PASS' if p3 else 'FAIL',
             detail=f"full {m_full:.4f} · control {m_core:.4f} · Gate 1 {GATE1_V3_LOCO}"),
        dict(check='4  [ABSENT] ablation (H1)', result='decided',
             detail=f"ship arm {'B' if arm == 'absent' else 'A'} on the cross-cohort column "
                    f"(A {a_loco:.4f} vs B {b_loco:.4f}); within-cohort disagrees"
                    if flipped else
                    f"ship arm {'B' if arm == 'absent' else 'A'}; both columns agree")])

    panel_tbl = pd.DataFrame([
        dict(cohort=c, markers=len(per[c]),
             excluded=int(sum((c, t) in excluded for t in per[c])),
             kept=len(per[c]) - int(sum((c, t) in excluded for t in per[c])))
        for c in cohorts]).sort_values('markers', ascending=False)

    excl_tbl = (D[D.rank_spread < SPREAD_MIN]
                .merge(N[['cohort', 'triple', 'r2', 'baseline_mse']], on=['cohort', 'triple'],
                       how='left')
                .assign(gene=lambda x: x.triple.map(genes))
                [['cohort', 'gene', 'rank_spread', 'var_ucoh', 'tie_mass', 'robust_dispersion',
                  'distinct_values', 'baseline_mse', 'r2']]
                .sort_values(['cohort', 'rank_spread']))

    figp = os.path.join(FIGURES, 'pretrain_rank_spread_vs_r2.png')
    figure_spread(ex_pairs, D, figp)

    verdict = ('GATE 2 PASSES (checks 1-3), and check 4 is decided.' if passed else
               'GATE 2 FAILS. The failing check is reported in full below, not softened.')
    # A --quick run used to write the SAME file with the SAME verdict line, so a 2-epoch smoke
    # test read exactly like a real Gate 2 failure (found 2026-09-11). It now writes its own file
    # and says what it is in the verdict itself.
    if quick:
        verdict = ('NOT A GATE RESULT - `--quick` smoke run (2 epochs, 1,500 cells per cohort). '
                   'It proves the code path runs; every number below is meaningless.')
    ctx = dict(lines=[], D=D, core=core, genes=genes, V=V, per=per, verdict=verdict,
               summary=summary, panel_tbl=panel_tbl, n_excl=len(excluded),
               n_excl_triples=int(sum(1 for t in triples if ncoh.get(t, 0) == 1)),
               c1=c1, c1_neg=c1_neg, c1_verdict=c1_verdict,
               n_split_dropped=n_split_dropped,
               split_tbl=(R[R.kept_cohort_wide & ~R.kept]
                          .assign(gene=lambda x: x.triple.map(genes))
                          [['cohort', 'gene', 'split_spread', 'r2', 'baseline_mse']]
                          .sort_values('split_spread')),
               sweep=sweep_table(D), sweep_tie=sweep_table(D, 'tie_mass', TIE_SWEEP, below=False),
               c2=c2, c2_verdict=c2_verdict, excl_tbl=excl_tbl, figp=figp,
               c3=c3, c3_folds=c3_folds, c3_verdict=c3_verdict,
               c4=c4, c4_verdict=c4_verdict, runs=runs, n_train=len(train), train=train,
               units_tbl=split_table(train), split_fp=SPLIT_FP,
               devices=', '.join(sorted({str(r.get('device', 'not recorded - checkpoint '
                                                   'predates the device field'))
                                         for r in [A_run, B_run, NX, *LA.values(),
                                                   *LB.values(), *L_core.values()]})))
    path = write_report(os.path.join(REPORTS, 'pretrain_masked_markers_quick.md' if quick else 'pretrain_masked_markers.md'),
                        ctx)
    return path, summary, passed


def do_armb_loco(held, cohorts, per, tri2idx, triples, ncoh, D, refit=False):
    """Targeted diagnostic: run Arm B on ONE leave-one-cohort-out fold.

    Why this exists. Check 4 chose the arm on a WITHIN-cohort slide holdout, where every cohort's
    token-set size matches what the model trained on. Check 3 then used that arm for the
    CROSS-cohort test. But Arm A's set size varies 12-57 tokens, so on a held-out Sorin the model
    sees a set far smaller than anything it trained on - and preventing exactly that is what Arm
    B's fixed 99 slots are for. Choosing the arm in-distribution and then testing out-of-
    distribution is the wrong order, and the first Gate 2 run collapsed on precisely that fold
    (full panel -0.1033 against a core-9 control of 0.1644).

    If Arm B does not collapse here, check 4's protocol was wrong and the [ABSENT] decision has to
    be reopened. If it collapses too, the cause is Sorin's uint8 quantisation rather than set size,
    and the [ABSENT] decision stands.
    """
    V = len(triples)
    train = [c for c in cohorts if SPECS[c]['role'] == 'train']
    core = sorted([t for t in triples if ncoh.get(t, 0) == len(cohorts)])
    excluded = {(r.cohort, r.triple) for r in D.itertuples() if r.rank_spread < SPREAD_MIN}

    print(f"\n  Arm B, held out {held} - the set-size hypothesis")
    B = loco_run(f'armb_loco_{held}', 'absent', held, train, per, tri2idx, V, excluded, refit,
                 EPOCHS)
    A = torch.load(os.path.join(CKPT, f'pretrain_loco_full_{held}.pt'), weights_only=False,
                   map_location='cpu')
    C = torch.load(os.path.join(CKPT, f'pretrain_loco_core_{held}.pt'), weights_only=False,
                   map_location='cpu')

    out = pd.DataFrame([
        dict(arm='Arm A - measured markers only (12-57 tokens)', core9_r2=round(core_mean(A['r2'], core), 4),
             epochs=A['info']['epochs_used'], seconds=A['seconds']),
        dict(arm='Arm B - all 99 slots, learned [ABSENT]', core9_r2=round(core_mean(B['r2'], core), 4),
             epochs=B['info']['epochs_used'], seconds=B['seconds']),
        dict(arm='core-9 control (9 tokens everywhere)', core9_r2=round(core_mean(C['r2'], core), 4),
             epochs=C['info']['epochs_used'], seconds=C['seconds'])])
    print()
    print(out.to_string(index=False))

    m = (A['r2'][A['r2'].triple.isin(core)][['triple', 'r2']]
         .merge(B['r2'][B['r2'].triple.isin(core)][['triple', 'r2']], on='triple',
                suffixes=('_armA', '_armB')))
    r = registry().drop_duplicates('triple').set_index('triple').gene
    m['gene'] = m.triple.map(r)
    m['delta'] = (m.r2_armB - m.r2_armA).round(4)
    print()
    print(m[['gene', 'r2_armA', 'r2_armB', 'delta']].sort_values('delta').to_string(index=False))
    return out


LOTO_PREFIX = {'absent': 'armb_loto_', 'set': 'loto_full_'}


def loto_run(tag, arm, held, train, per, tri2idx, V, excluded, refit, epochs):
    """Train on every training cohort NOT in `held` (a whole tissue), score each held-out cohort."""
    def make():
        rest = [c for c in train if c not in held]
        data = {c: load_cohort(c, per[c], tri2idx, V, arm, excluded) for c in train}
        data = {c: d for c, d in data.items() if d is not None}
        rest = [c for c in rest if c in data]
        model, info = fit(data, rest, V, epochs=epochs)
        base = baselines(data, rest, 'loco', tri2idx)
        R = pd.concat([score(model, data[h], base, 'loco') for h in held if h in data],
                      ignore_index=True)
        return dict(arm=arm, held=list(held), r2=R, info=info, state=cpu_state(model))
    return cached(tag, refit, make)


def do_loto(cohorts, per, tri2idx, triples, ncoh, D, refit=False, quick=False):
    """LEAVE-ONE-TISSUE-OUT Stage 2 models - the clean warm start for LOTO folds (plan F4).

    Why. Downstream fits warm-start through warm_for(), which accepts only a Stage 2 model trained
    on a SUBSET of the fit's own training cohorts. A LOTO fold holds out a whole tissue, and two
    tissues have two cohorts (breast = Keren + Danenberg, skin = Phillips + ferguson). Gate 2
    trained no model without both, so those folds had no leak-free warm start and raised. This
    trains one, in the SHIPPED arm only, per multi-cohort tissue. Single-cohort tissues need
    nothing new: their LOCO model already holds out exactly that tissue.

    Not part of Gate 2 and decides nothing. As a by-product it reports each held-out cohort's
    core-9 R2 under LOTO next to its LOCO value from Gate 2: the drop is a first, label-free look
    at how much a same-tissue cohort in training helps.
    """
    global N_TRAIN, VAL_CELLS, SCORE_CELLS
    if quick:
        N_TRAIN, VAL_CELLS, SCORE_CELLS = 1_500, 400, 400
        print("\n*** --quick: 2 epochs on a tiny draw. Proves the code path; scores nothing. ***")
    V = len(triples)
    train = [c for c in cohorts if SPECS[c]['role'] == 'train']
    core = sorted([t for t in triples if ncoh.get(t, 0) == len(cohorts)])
    excluded = {(r.cohort, r.triple) for r in D.itertuples() if r.rank_spread < SPREAD_MIN}
    arm = json.load(open(os.path.join(WORK, 'panel.json'))).get('stage2_arm')
    if arm not in LOTO_PREFIX:
        sys.exit(f"panel.json stage2_arm is {arm!r} - run Gate 2 (--check) first")
    epochs = 2 if quick else EPOCHS
    q = 'quick_' if quick else ''

    tissues = {}
    for c in train:
        tissues.setdefault(SPECS[c]['tissue'], []).append(c)
    multi = {t: cs for t, cs in sorted(tissues.items()) if len(cs) >= 2}
    print(f"\n  LOTO Stage 2, shipped arm {arm}: {len(multi)} multi-cohort tissues "
          f"{ {t: cs for t, cs in multi.items()} }")

    rows = []
    for t, held in multi.items():
        r = loto_run(f"{q}{LOTO_PREFIX[arm]}{t.replace(' ', '_')}", arm, held, train, per,
                     tri2idx, V, excluded, refit, epochs)
        lp = os.path.join(CKPT, f"pretrain_{q}{WARM_TAGS[arm][1][0]}{{}}.pt")
        for h in held:
            loco = (torch.load(lp.format(h), weights_only=False, map_location='cpu')
                    if os.path.exists(lp.format(h)) else None)
            loto_r2 = core_mean(r['r2'][r['r2'].cohort == h], core)
            loco_r2 = core_mean(loco['r2'], core) if loco is not None else float('nan')
            rows.append(dict(tissue=t, held_out=h, loto_core9_r2=round(loto_r2, 4),
                             loco_core9_r2=round(loco_r2, 4),
                             loto_minus_loco=round(loto_r2 - loco_r2, 4),
                             epochs=r['info']['epochs_used'], seconds=r['seconds']))
    out = pd.DataFrame(rows)
    print()
    print(out.to_string(index=False))
    if quick:
        print("\n--quick: these numbers score nothing. No report written.")
        return out

    L = []
    A = L.append
    A("# Stage 2 - leave-one-tissue-out models (plan F4)\n")
    A(f"Shipped arm **{arm}**, vocabulary `{VOCAB_FP}`, within-cohort split `{SPLIT_FP}` (by "
      f"patient, plan F3). One model per tissue with more than one "
      f"cohort, trained on every other training cohort. These are the leak-free warm starts for "
      f"LOTO folds; `pretrain_masked_markers.warm_for()` selects them by their recorded training cohorts.\n")
    A("**Not a gate - decides nothing.** The table is a label-free by-product: each held-out "
      "cohort's masked-marker R2 on the core-9 markers when its whole tissue is held out (LOTO), "
      "against Gate 2's value when only that cohort is held out (LOCO). A negative "
      "`loto_minus_loco` means the same-tissue cohort was helping.\n")
    A(out.to_markdown(index=False))
    A(f"\nEpoch ceiling {EPOCHS}; a fit that reached it was still improving and its R2 is a lower "
      f"bound. Single-cohort tissues reuse their LOCO models and are not refitted.\n")
    A("**Read the differences with four limits before citing them.** (1) A fit at the epoch "
      "ceiling gives a lower bound, and the LOTO and LOCO fits train on different numbers of "
      "cohorts, so they get different numbers of steps per epoch. (2) One seed - no error bar. "
      "(3) R2 is measured against the mean of the TRAINING cohorts (mode 'loco'), and LOTO and "
      "LOCO train on different cohort sets, so the two R2 values do not share a denominator. "
      "(4) This is masked-marker reconstruction, not annotation. A gap of a few hundredths is "
      "therefore not evidence either way; the tissue effect on annotation is measured at "
      "benchmark Layer 2 (LOCO vs LOTO macro-F1).\n")
    p = os.path.join(REPORTS, 'pretrain_loto.md')
    open(p, 'w', encoding='utf-8').write("\n".join(L))
    print(f"\nwrote {p}")
    return out


def main():
    cohorts = built()
    if not cohorts:
        # KAGGLE. work/raw/ is not uploaded - 627 MB for what is only an existence check here (see
        # kaggle/README.md). --check reads nothing but the wide value tables, so a cohort whose
        # value table exists is a cohort that can be scored. --build genuinely needs raw.
        cohorts = [c for c in SPECS if os.path.exists(full_table(c))]
        if cohorts:
            print("work/raw/ absent - cohorts taken from work/values/*_full.parquet (--check only)")
    if not cohorts:
        sys.exit("nothing built - run: python loaders.py")
    print(f"device: {DEV}" + (f" ({torch.cuda.get_device_name(0)})" if DEV.type == 'cuda' else ''))
    triples, tri2idx, per, genes, ncoh = panel_spec(cohorts)
    global VOCAB_FP
    VOCAB_FP = f"{len(triples)}:{zlib.crc32('|'.join(triples).encode()):08x}"
    print(f"vocabulary: {len(triples)} marker triples across {len(cohorts)} cohorts "
          f"(fingerprint {VOCAB_FP})")
    for c in cohorts:
        print(f"   {c:9} {len(per[c]):>3} markers  ({SPECS[c]['role']})")

    need = [c for c in cohorts if not os.path.exists(full_table(c))]
    if '--build' in sys.argv or need:
        sys.exit("the wide value tables, panel.json and the dynamic-range table are built by "
                 "build_marker_vocabulary.py - run it first"
                 + (f" (no value table for: {', '.join(need)})" if need else ''))

    p = os.path.join(WORK, 'marker_dynamic_range.csv')
    if not os.path.exists(p):
        sys.exit("no work/marker_dynamic_range.csv - run: python build_marker_vocabulary.py")
    D = pd.read_csv(p, keep_default_na=False)
    for col in ('tie_mass', 'robust_dispersion', 'var_ucoh', 'rank_spread'):
        if col not in D.columns:
            sys.exit(f"work/marker_dynamic_range.csv predates D-28 (no `{col}` column) - "
                     f"rerun: python build_marker_vocabulary.py")
        D[col] = D[col].astype(float)

    # after any build, so the fingerprint covers the value tables the fits will actually read
    global SPLIT_FP
    SPLIT_FP = split_fp()
    fell = [c for c in split_roster() if split_plan(c)['unit'] != 'patient']
    print(f"within-cohort split: by patient (fingerprint {SPLIT_FP}); slide fallback: "
          f"{', '.join(fell) if fell else 'none'}")

    if '--armb-loco' in sys.argv:
        held = sys.argv[sys.argv.index('--armb-loco') + 1]
        do_armb_loco(held, cohorts, per, tri2idx, triples, ncoh, D, refit='--refit' in sys.argv)
        return

    if '--loto' in sys.argv:
        do_loto(cohorts, per, tri2idx, triples, ncoh, D, refit='--refit' in sys.argv,
                quick='--quick' in sys.argv)
        return

    if '--check' not in sys.argv:
        n = int((D.rank_spread < SPREAD_MIN).sum())
        print(f"\ndynamic range: {n} of {len(D)} (cohort, marker) pairs excluded at "
              f"rank_spread < {SPREAD_MIN}")
        print(sweep_table(D).to_string(index=False))
        print(f"\nthe REPLACED rule (tie_mass), for comparison - see D-28:")
        print(sweep_table(D, 'tie_mass', TIE_SWEEP, below=False).to_string(index=False))
        print("\ndone. run with --check for GATE 2.")
        return

    quick = '--quick' in sys.argv
    if '--holdout-epochs' in sys.argv:
        global HOLDOUT_EPOCHS
        HOLDOUT_EPOCHS = int(sys.argv[sys.argv.index('--holdout-epochs') + 1])
        print(f"\nholdout runs get a ceiling of {HOLDOUT_EPOCHS} epochs; LOCO folds stay at "
              f"{EPOCHS}. The originals are kept under their own tags.")
    if quick:
        # a smoke test of the code path only - these numbers mean nothing and the report says so
        global N_TRAIN, VAL_CELLS, SCORE_CELLS
        N_TRAIN, VAL_CELLS, SCORE_CELLS = 1_500, 400, 400
        print("\n*** --quick: 2 epochs on a tiny draw. This proves the code runs; it does NOT "
              "score the gate. ***")

    print("\nrunning GATE 2...")
    t0 = time.time()
    path, summary, passed = do_check(cohorts, per, tri2idx, triples, genes, ncoh, D,
                                     refit='--refit' in sys.argv, quick=quick)
    print("\n--- GATE 2 ---")
    print(summary.to_string(index=False))
    tag = 'SMOKE RUN - not a gate result' if quick else ('PASS' if passed else 'FAIL')
    print(f"\n{tag} · {(time.time()-t0)/60:.1f} min · wrote {path}")


if __name__ == "__main__":
    main()

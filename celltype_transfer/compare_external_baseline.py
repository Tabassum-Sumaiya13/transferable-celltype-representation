"""
STAGE 10 - the external baseline.  Produces GATE 10.

    python compare_external_baseline.py --gate            # 3 MAPS arms x 7 LOCO folds, CPU is fine
    python compare_external_baseline.py --gate --quick    # smoke test: proves the code path, scores nothing
    python compare_external_baseline.py --gate --refit    # ignore the checkpoint cache
    python compare_external_baseline.py --gate --folds Keren,Sorin

WHY THIS STAGE EXISTS. Gate 6 produced a headline LOCO macro-F1 of 0.3151 with NO PUBLISHED METHOD
BESIDE IT. A reader cannot tell whether that is good, bad or trivial, and "no baseline" is the
single largest hole in the write-up. This stage puts a real, published, independently written
model on exactly the same folds, the same cells, the same patient splits, the same fold-local
label spaces and the same metric.

THE BASELINE IS NOT REIMPLEMENTED. `maps.cell_phenotyping.networks.MLP` is imported from the
vendored MAPS source in dropped_past_works/MAPS (Shaban et al., "MAPS: pathologist-level cell type
annotation from tissue images through machine learning", Nat Commun 2024). A baseline written by
the same person who wrote the method under test proves nothing. If the import fails this stage
RAISES - it never silently substitutes a lookalike MLP, because a fake baseline that loses is
worse than no baseline at all.

THE STRUCTURAL POINT THIS STAGE MEASURES. MAPS takes a FIXED `input_dim`. It has no way to say
"this cohort never measured that marker". So under 7-cohort LOCO it can only use markers that
EVERY cohort measures, and on this roster that is NINE triples - Sorin's 17-marker panel is the
floor. This project uses all 109 because of the [ABSENT] token (design idea 2). That gap is not a
handicap imposed on the baseline; it is the argument.

    arm            input                                         why it is here
    core9          the 9 triples all 7 cohorts measure           what MAPS can structurally do
    shared         the 6 TRAINING cohorts' shared panel, with    anti-strawman, check 2b
                   whatever the held-out cohort lacks zeroed
    zerofill       all 109 slots, unmeasured filled with 0.0     what a practitioner actually does

The first version of the `shared` arm intersected the training panels WITH the held-out cohort's,
and is kept in gate10_expect.csv as row `2-REPLACED`. That is the 7-cohort intersection by
definition - training plus held-out ARE all 7 - so it produced the same 9 markers on all 7 folds
and could not have said anything. Caught before any fit ran; nothing was re-scored afterwards.

TWO DELIBERATE DEVIATIONS FROM MAPS's PUBLISHED DEFAULTS, both declared before the run and both
made IN THE BASELINE'S FAVOUR:

  Class-balanced sampling. MAPS trains on plain cross-entropy. The metric here is macro-F1, and a
  plain-CE model on labels this imbalanced collapses onto the common classes and scores near zero.
  That would measure "MAPS was not tuned for macro-F1", not "MAPS cannot transfer across panels",
  and it would make check 1 a win over a broken opponent. MAPS therefore gets the SAME balanced
  sampler Stage 6 uses.

  Epoch budget. MAPS defaults to min_epochs=250, max_epochs=500, patience=100 - written for
  single-dataset training. Here it gets early stopping on validation macro-F1 with max 60 epochs
  and patience 8, more than double Stage 6's own 30 / 4. `epochs_used` is reported per fold: where
  a fold stops before 60 the budget was not binding, and undertraining cannot explain the result.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # runs from any directory
import config
from config import SPECS, WORK, REPORTS, SEED
import splits
import build_marker_vocabulary as vocab
import metrics
import pretrain_masked_markers as pretrain
import train_adversarial_encoder as adversarial
import train_prototype_classifier as classifier
from metrics import paired          # one definition of the paired CI / sign-flip test

CKPT = os.path.join(WORK, 'ckpt')
from models.device import DEV   # one rule for all stages; --cpu forces the CPU

# Same literal-pinned roster guard as Stage 4 check 0 (H14/D-39).
TRAIN = ['CRC', 'UPMC', 'Keren', 'Phillips', 'Sorin', 'Danenberg', 'ferguson']

ARMS = ['core9', 'shared', 'zerofill']
HIDDEN, DROPOUT, LR, BATCH = 512, 0.10, 1e-3, 128      # MAPS's published defaults
MAX_EPOCHS, PATIENCE = 60, 8                            # declared deviation, see the docstring
MAJORITY_FOLDS_MIN = 5                                  # check 5


def assert_roster():
    got = [c for c, s in SPECS.items() if s['role'] == 'train']
    assert set(got) == set(TRAIN), (
        f"roster drift: config says {sorted(got)}, expected {sorted(TRAIN)}")
    held = [c for c, s in SPECS.items() if s['role'] == 'holdout']
    assert not held, f"a frozen holdout still exists ({held}); the protocol is 7-fold LOCO"
    print(f"check 0 PASS - {len(TRAIN)} cohorts, no holdout: {', '.join(TRAIN)}")
    return True


def maps_mlp(input_dim, num_classes):
    """The VENDORED MAPS network. Raises rather than fall back to a lookalike."""
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'dropped_past_works', 'MAPS')   # was dropped_past_works/, renamed
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from maps.cell_phenotyping.networks import MLP
    except Exception as e:
        raise RuntimeError(
            f"could not import the vendored MAPS source from {root}: {e}\n"
            "This stage must not substitute its own MLP - a baseline written by the author of "
            "the method under test is not a baseline.") from e
    return MLP(input_dim=input_dim, hidden_dim=HIDDEN, num_classes=num_classes, dropout=DROPOUT)


def arm_slots(arm, held, per, tri2idx):
    """Which vocabulary slots this arm may feed to MAPS, and one line saying why.

    `shared` trains on every marker the 6 TRAINING cohorts share and zero-fills whatever the
    held-out cohort lacks at test time. That is what a careful practitioner actually does: use as
    much of the panel as training allows and accept imputation at test. It is strictly more
    training signal than core-9, which is what makes it the anti-strawman arm.
    """
    rest = [c for c in TRAIN if c != held]
    if arm == 'zerofill':
        return np.arange(len(tri2idx)), f'all {len(tri2idx)} slots, unmeasured filled with 0.0'
    if arm == 'core9':
        keep = set(per[TRAIN[0]])
        for c in TRAIN[1:]:
            keep &= set(per[c])
        why = f'the {len(keep)} triples all 7 cohorts measure'
    else:
        # THE TRAINING-PANEL INTERSECTION, deliberately NOT intersected with the held-out cohort.
        # gate10_expect.csv carries the first version of this arm as row `2-REPLACED`: it did
        # intersect with the held-out panel, which is mathematically the 7-cohort intersection
        # (training + held-out ARE all 7) and gave 9 markers on all 7 folds - the same 9 as
        # core9. The arm could not have said anything. Found before any fit ran, so nothing was
        # re-scored after the fact.
        keep = set(per[rest[0]])
        for c in rest[1:]:
            keep &= set(per[c])
        miss = len(keep - set(per[held]))
        why = (f'{len(keep)} triples shared by the 6 training cohorts; '
               f'{miss} of them unmeasured in {held} and zero-filled at test')
    return np.array(sorted(tri2idx[t] for t in keep)), why


def logits_of(out):
    """MAPS's forward returns (logits, probs); tolerate a bare tensor too."""
    return out[0] if isinstance(out, (tuple, list)) else out


def fit_maps(data, rest, L, slots, seed=SEED):
    """Train the vendored MAPS MLP, class-balanced, early-stopped on validation macro-F1."""
    torch.manual_seed(seed)
    m = maps_mlp(len(slots), L['n']).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)
    sl = torch.from_numpy(slots).long().to(DEV)

    # z-score on the TRAINING cells only, exactly as MAPS's CellExpressionCSV does
    tr = torch.cat([data[c]['U']['train'][:, sl] for c in rest])
    mu, sd = tr.mean(0, keepdim=True), tr.std(0, keepdim=True).clamp(min=1e-6)

    def feats(c, part):
        return (data[c]['U'][part][:, sl] - mu) / sd

    pools = {c: {int(k): np.flatnonzero(data[c]['y']['train'].cpu().numpy() == k)
                 for k in np.unique(data[c]['y']['train'].cpu().numpy())} for c in rest}

    best, best_state, bad, used = -np.inf, None, 0, MAX_EPOCHS
    for ep in range(MAX_EPOCHS):
        m.train()
        for c in [rest[i] for i in torch.randperm(len(rest), generator=gen).tolist()]:
            pool = pools[c]
            ks = list(pool)
            per_k = max(1, data[c]['n']['train'] // max(1, len(ks)))
            take = np.concatenate([classifier._rng('bal', c, ep).choice(pool[k], per_k,
                                                                replace=len(pool[k]) < per_k)
                                   for k in ks])
            take = take[classifier._rng('shuf', c, ep).permutation(len(take))]
            X, Y = feats(c, 'train'), data[c]['y']['train']
            for i in range(0, len(take), BATCH):
                b = torch.from_numpy(take[i:i + BATCH]).long().to(DEV)
                loss = lossf(logits_of(m(X[b])), Y[b])
                opt.zero_grad(); loss.backward(); opt.step()

        m.eval()
        yt, yp = [], []
        with torch.no_grad():
            for c in rest:
                X = feats(c, 'val')
                for i in range(0, len(X), 4096):
                    yp.append(logits_of(m(X[i:i + 4096])).argmax(1))
                yt.append(data[c]['y']['val'])
        f1, _ = metrics.macro_f1(torch.cat(yt).cpu().numpy(), torch.cat(yp).cpu().numpy(), L['n'])
        if f1 > best + 1e-5:
            best, bad = f1, 0
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                used = ep + 1
                break
    if best_state is not None:
        m.load_state_dict(best_state)
    m.eval()
    return m, mu, sd, dict(epochs_used=used, val_f1=round(best, 4))


def run_fold(tag, arm, held, per, tri2idx, triples, L, excl, refit, space):
    p = os.path.join(CKPT, f'external_{tag}.pt')
    if os.path.exists(p) and not refit:
        r = torch.load(p, weights_only=False, map_location='cpu')
        if not splits.split_stale(r) and r.get('space') == space.get('space'):
            print(f"    [cache] {tag}  F1core={r['f1_core']:.4f}")
            return r
        print(f"    [STALE] {tag}: another split or label space - refitting")

    t0 = time.time()
    V = len(triples)
    rest = [c for c in TRAIN if c != held]
    data = {c: classifier.load_cohort(c, per[c], tri2idx, V, L, excl) for c in TRAIN}
    data = {c: d for c, d in data.items() if d is not None}
    rest = [c for c in rest if c in data]
    slots, why = arm_slots(arm, held, per, tri2idx)

    m, mu, sd, info = fit_maps(data, rest, L, slots)
    sl = torch.from_numpy(slots).long().to(DEV)
    d = data[held]
    yp = []
    with torch.no_grad():
        X = (d['U']['test'][:, sl] - mu) / sd
        for i in range(0, len(X), 4096):
            yp.append(logits_of(m(X[i:i + 4096])).argmax(1))
    yp = torch.cat(yp).cpu().numpy()
    yt = d['y']['test'].cpu().numpy()
    f1_all, _ = metrics.macro_f1(yt, yp, L['n'])
    f1_core, _ = metrics.macro_f1(yt, yp, L['n'], drop=L['unreliable'])
    f1_maj, f1_rnd, _ = adversarial.baselines(data, rest, held, L['n'], L['unreliable'])

    r = dict(tag=tag, arm=arm, held=held, n_markers=len(slots), panel=why,
             f1_core=f1_core, f1_all=f1_all, f1_majority=f1_maj, f1_random=f1_rnd,
             n_test=int(len(yt)), n_class=L['n'], info=info,
             split_fp=splits.split_fp(), space=space.get('space'),
             seconds=round(time.time() - t0, 1))
    torch.save(r, p)
    print(f"    [done ] {tag}  F1core={f1_core:.4f} (maj {f1_maj:.4f})  "
          f"{len(slots)} markers  {r['seconds']:.0f}s  epochs={info['epochs_used']}")
    return r


def ours(folds):
    """Our own per-fold numbers, read from the CACHED Gate 6 fits - never re-measured here.

    THE ARM MUST BE THE ONE GATE 6 ACTUALLY SHIPPED, not a hardcoded checkpoint name. An earlier
    version of this function hardcoded `classifier_proto3_fold_*.pt` (then s6_proto3_fold_*.pt) (3 losses, +VICReg) - but Gate 6's
    OWN check 3 (classifier.assemble, 'does the extra loss earn its place') found VICReg HURT macro-F1 on
    this roster and shipped **2 losses** instead: mean 0.3151, the project's real headline. The
    hardcoded version compared MAPS against 0.2956 - a configuration Gate 6 itself rejected -
    and reported a FAIL that used the wrong number on our side. Caught after the first full run;
    the run was repeated, nothing was re-scored by hand.

    `classifier.assemble()` is called on the cached sweep so the two decisions - which loss set ships,
    and what MAPS is compared against - can never drift apart even if a future Gate 6 rerun
    changes the winner.

    Check 6 asserts these came from the same split and the same fold-local space MAPS was just
    scored in. A comparison across two different label spaces would be meaningless.
    """
    p = os.path.join(CKPT, 'classifier_fold_sweep.pt')
    if not os.path.exists(p):
        sys.exit(f"{p} not found - Gate 6 must be run before Gate 10 can compare against it")
    res = torch.load(p, weights_only=False, map_location='cpu')
    _, _, _, _, ship_name, _, _, _, _, _ = classifier.assemble(res, {})   # L is unused inside assemble()
    # check 3b (2026-09-13) added a third candidate arm - map every ship_name assemble() can
    # produce to its checkpoint tag explicitly, rather than a startswith() guess that only knew
    # about two of them.
    tag = {'2 losses': 'proto2', '3 losses (+VICReg)': 'proto3'}.get(ship_name, 'proto2adv')
    print(f"    Gate 6 shipped: {ship_name} - Gate 10 compares against that arm, per fold")
    out = {}
    for h in folds:
        p = os.path.join(CKPT, f'classifier_{tag}_fold_{h}.pt')
        if os.path.exists(p):
            out[h] = torch.load(p, weights_only=False, map_location='cpu')
    return out


# ------------------------------------------------------------------------------- gate
def assemble(res, folds, mine):
    d = pd.DataFrame([{k: r[k] for k in ('tag', 'arm', 'held', 'n_markers', 'f1_core', 'f1_all',
                                         'f1_majority', 'n_test', 'n_class', 'seconds')}
                      for r in res])
    d['epochs'] = [r['info']['epochs_used'] for r in res]
    piv = d.pivot_table(index='held', values='f1_core', columns='arm', aggfunc='first')
    piv = piv.reindex([f for f in folds if f in piv.index])
    piv.insert(0, 'ours', [mine[h]['f1_core'] if h in mine else np.nan for h in piv.index])

    checks, stats_rows = [], {}

    # check 6 FIRST: if the two sides are not on the same footing nothing below means anything
    bad = []
    for r in res:
        h = r['held']
        if h not in mine:
            bad.append(f"{h}: no cached Gate 6 fit")
            continue
        o = mine[h]
        if o.get('space') != r.get('space'):
            bad.append(f"{h}: space {o.get('space')} vs {r.get('space')}")
        if o.get('n_class') != r.get('n_class'):
            bad.append(f"{h}: n_class {o.get('n_class')} vs {r.get('n_class')}")
        if o.get('split_fp') != r.get('split_fp'):
            bad.append(f"{h}: split_fp differs")
    checks.append(dict(check='6  same cells, split and label space as ours',
                       result='PASS' if not bad else 'FAIL',
                       detail=('every fold matches the cached Gate 6 fit on space hash, class '
                               'count and split fingerprint' if not bad
                               else '; '.join(sorted(set(bad))[:4]))))

    have = [a for a in ARMS if a in piv and piv[a].notna().all()]
    ok_ours = piv['ours'].notna().all()

    for n, arm, lbl in (("1", "core9", "MAPS on the core-9 panel"),
                        ("2b", "shared", "MAPS on the training-panel intersection")):
        if arm in have and ok_ours:
            st = paired(piv['ours'] - piv[arm])
            stats_rows[f'ours - {arm}'] = st
            checks.append(dict(check=f'{n}  this project beats {lbl}',
                               result='PASS' if st['mean'] > 0 else 'FAIL',
                               detail=f"ours {piv['ours'].mean():.4f} vs {arm} "
                                      f"{piv[arm].mean():.4f} - delta {st['mean']:+.4f}"))

    if 'zerofill' in have and 'core9' in have:
        st = paired(piv['zerofill'] - piv['core9'])
        stats_rows['zerofill - core9'] = st
        better = st['mean'] > 0
        checks.append(dict(check='3  does zero-filling rescue the baseline',
                           result='zero-fill helps MAPS' if better else 'zero-fill does not help',
                           detail=f"zerofill {piv['zerofill'].mean():.4f} vs core9 "
                                  f"{piv['core9'].mean():.4f} - delta {st['mean']:+.4f} "
                                  f"(confounded: two MAPS arms, not a clean test of [ABSENT])"))

    if 'ours - core9' in stats_rows:
        st = stats_rows['ours - core9']
        spans = st['lo'] <= 0 <= st['hi']
        checks.append(dict(check='4  paired interval on (ours - core9)',
                           result='spans zero' if spans else 'excludes zero',
                           detail=f"mean {st['mean']:+.4f}, 95% CI [{st['lo']:+.4f}, "
                                  f"{st['hi']:+.4f}], exact sign-flip p = {st['p']:.3f}, "
                                  f"n = {st['n']}"))

    if 'core9' in have:
        sub = d[d.arm == 'core9']
        won = int((sub.f1_core > sub.f1_majority).sum())
        checks.append(dict(check=f'5  MAPS core-9 beats majority on >= {MAJORITY_FOLDS_MIN}/7 folds',
                           result='PASS' if won >= MAJORITY_FOLDS_MIN else 'FAIL',
                           detail=f"{won}/{len(sub)} folds. If this fails the baseline is broken "
                                  f"and checks 1-2 are not wins"))

    hard = [c for c in checks if c['check'][0] in '1256']
    verdict = 'PASS' if hard and all(c['result'] == 'PASS' for c in hard) else 'FAIL'
    return d, piv, checks, verdict, stats_rows


def write_report(path, d, piv, checks, verdict, stats_rows, panels, mins):
    def md(x):
        return x.to_markdown(index=False)

    st = pd.DataFrame([dict(comparison=k, folds=v['n'], mean=round(v['mean'], 4),
                            ci_lo=round(v['lo'], 4), ci_hi=round(v['hi'], 4),
                            sign_flip_p=round(v['p'], 3)) for k, v in stats_rows.items()])
    ci = stats_rows.get('ours - core9')
    honest = ''
    if ci is not None and ci['lo'] <= 0 <= ci['hi']:
        honest = ("\n> **The interval spans zero.** `gate10_expect.csv` check 4 says not to write "
                  "\"this project beats the published baseline\" if it does, whatever the mean "
                  "says. It does, so that sentence is not written here.\n")

    means = {c: piv[c].mean() for c in piv.columns if piv[c].notna().any()}
    summary = pd.DataFrame([dict(method=k, loco_macro_f1=round(v, 4)) for k, v in means.items()])

    txt = f"""# Stage 10 - the external baseline (GATE 10)

**{verdict}** - {mins:.1f} min on {DEV}, {len(d)} fits.

{md(summary)}
{honest}
## What this stage is for

Gate 6 produced a headline LOCO macro-F1 of **0.3151** with no published method beside it. A
reader could not tell whether that is good, bad or trivial. This stage puts **MAPS** (Shaban et
al., *Nat Commun* 2024) on exactly the same 7 LOCO folds, the same cells, the same patient splits,
the same fold-local label spaces and the same metric.

**The baseline is not reimplemented.** `maps.cell_phenotyping.networks.MLP` is imported from the
vendored MAPS source. If that import fails the stage raises - it never substitutes a lookalike,
because a fake baseline that loses is worse than no baseline.

## The gate

{md(pd.DataFrame(checks))}

Thresholds were declared in `celltype_transfer/declared/gate10_expect.csv` before this code was written.

## The structural point

MAPS takes a **fixed `input_dim`**. It has no way to say *this cohort never measured that marker*.
So under 7-cohort LOCO it can only use markers every cohort measures, and on this roster that is
**nine triples** - Sorin's 17-marker panel is the floor. This project uses all 109 because of the
`[ABSENT]` token (design idea 2). That gap is not a handicap imposed on the baseline; it is the
argument the project makes.

{md(panels)}

## Per fold

{md(piv.reset_index().round(4))}

## Check 4 - the paired test

{md(st) if len(st) else '_not enough arms finished to pair._'}

The interval is a t-interval on 7 folds; the p-value is an exact sign-flip test over all 128
reassignments, so it assumes nothing about the shape of the differences. **n = 7 is small** and an
interval this wide cannot separate a small real effect from none.

## Where the baseline was given the advantage

Two deliberate deviations from MAPS's published defaults, both declared before the run and both in
the baseline's favour:

- **Class-balanced sampling.** MAPS trains on plain cross-entropy. The metric here is macro-F1,
  and a plain-CE model on labels this imbalanced collapses onto the common classes. That would
  measure "MAPS was not tuned for macro-F1", not "MAPS cannot transfer across panels". MAPS gets
  the same balanced sampler Stage 6 uses.
- **Epoch budget.** MAPS defaults to 250-500 epochs with patience 100, written for single-dataset
  training. Here it gets max 60 epochs with patience 8 - more than double Stage 6's own 30 / 4.
  `epochs` is reported per fold below: where a fold stops before 60 the budget was not binding, so
  undertraining cannot explain the result.

The `shared` arm additionally chooses its markers using the **held-out cohort's panel
composition**. That is metadata known before any label is seen, so it is not leakage - but it is a
generosity to the baseline and is stated rather than buried.

## Every fit

{md(d.round(4))}
"""
    open(path, 'w', encoding='utf-8').write(txt)
    return path


def main():
    if '--gate' not in sys.argv:
        print(__doc__.strip().split('\n\n')[1])
        print("\nnothing to do. run with --gate for GATE 10.")
        return
    assert_roster()

    cohorts = adversarial.available()
    triples, tri2idx, per, genes, ncoh = vocab.read_panel(cohorts)
    excl = classifier.excluded_pairs()
    quick = '--quick' in sys.argv
    refit = '--refit' in sys.argv
    if quick:
        classifier.N_TRAIN, classifier.SCORE_CELLS = 1_500, 800
        global MAX_EPOCHS, PATIENCE
        MAX_EPOCHS, PATIENCE = 2, 1
        print("\n*** --quick: 2 epochs on a tiny draw. Proves the code path; scores nothing. ***")

    folds = [c for c in TRAIN if c in cohorts]
    if '--folds' in sys.argv:
        pick = set(sys.argv[sys.argv.index('--folds') + 1].split(','))
        folds = [c for c in folds if c in pick]

    mode = adversarial.space_mode()
    spaces = {h: adversarial.space_for(h, mode) for h in folds}
    q = 'quick_' if quick else ''
    print(f"vocabulary : {len(triples)} triples x {len(TRAIN)} cohorts")
    print(f"device     : {DEV}   label space: {mode}")

    prow = []
    for arm in ARMS:
        sl, why = arm_slots(arm, folds[0], per, tri2idx)
        prow.append(dict(arm=arm, markers_on_fold_1=len(sl), rule=why))
    print('\n' + pd.DataFrame(prow).to_string(index=False))

    t0, res = time.time(), []
    for arm in ARMS:
        print(f"\n  arm {arm}")
        for h in folds:
            Lh, _, s = spaces[h]
            res.append(run_fold(f'{q}{arm}_{h}', arm, h, per, tri2idx, triples, Lh, excl,
                                refit, s))

    mine = ours(folds)
    d, piv, checks, verdict, stats_rows = assemble(res, folds, mine)
    print(f"\n--- GATE 10 ---\n{pd.DataFrame(checks).to_string(index=False)}")
    if quick:
        print("\n--quick: these numbers score nothing. No report written.")
        return
    panels = pd.DataFrame([dict(fold=h, **{f'{a}_markers': len(arm_slots(a, h, per, tri2idx)[0])
                                           for a in ARMS}) for h in folds])
    p = write_report(os.path.join(REPORTS, 'compare_external_baseline.md'), d, piv, checks, verdict,
                     stats_rows, panels, (time.time() - t0) / 60)
    print(f"\n{verdict} - wrote {p}")


if __name__ == '__main__':
    main()

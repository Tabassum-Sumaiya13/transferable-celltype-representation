"""Fold-local label spaces - one per LOCO / LOTO fold, built from that fold's TRAINING cohorts only.

    python celltype_transfer/build_fold_label_spaces.py --folds   # -> work/spaces/, reports/build_fold_label_spaces.md
    python celltype_transfer/build_fold_label_spaces.py           # print what exists, build nothing

benchmark_protocol.yaml label_space.primary_rule (plan F4, 2026-09-11): the label space a fold is
trained and scored in is built from its TRAINING cohorts only, inside every LOCO and LOTO fold.
Rules declared in declared/gate1b_fold_expect.csv (v1) and gate1b_fold_v2_expect.csv (v2) before
each run. Stages 3, 4 and 6 read these spaces through train_adversarial_encoder.space_for().

THE CHAIN. Exactly the whole-roster Stage 1b sequence (build_label_space.py), run on a subset:
rescale -> containment -> choose_cut -> cluster -> refine -> nesting -> name_clusters. The two
functions that do this, `subset` and `run`, used to live in s1b_control.py, the old 5-cohort H10
control (D-46). That control was built for the retired ferguson frozen holdout and is archived
(dropped_past_works/pipeline2_stale/s1b_control.py); the two functions moved here verbatim. The
old B1/B2 frozen-holdout spaces this file also built are archived with it
(dropped_past_works/pipeline2_stale/s7_spaces_full.py).

THE ADMISSION RULE. The held-out cohort's labels are placed into the frozen partition from their
marker signatures alone: a label joins the cluster whose members are closest on average, admitted
only within the fold's own cut `tau`. A label no cluster admits is NOVEL - never forced into the
nearest bin. This is what average linkage would have done had the label arrived last.

MEASURED 2026-09-11, FIRST RUN - THE SHIPPED CUT RULE DOES NOT SURVIVE REMOVING ONE COHORT.
Checks 0 and 1 held in all 9 folds. But 3 of 7 LOCO folds (CRC, Keren, UPMC) have NO usable cut:
the two guards (>= 95% of CELLS in cross-cohort clusters; biggest cluster <= 25% of labels) are
never met at the same cut. The cross-cohort share reaches 0.95 only at 0.775-0.825, where the
biggest cluster is already 25.2% (CRC fold), 31.6% (Keren), 25.4% (UPMC). With all 7 cohorts the
window exists at 0.800 - by a margin one cohort can remove. The declared fallback (most stable cut
over the whole sweep) then picks a DEGENERATE partition: near-singletons at 0.300 (stability
0.9995; Keren and UPMC folds, ~106 clusters, 40% / 84.5% of held-out cells NOVEL) or one cluster
holding 92.5% of labels at 1.100 (stability 1.000; CRC fold, 5 clusters). So LOCO stability does
NOT punish degenerate answers here, contrary to what Stage 1b's report says. The 6 other folds
are sensible (18-41 clusters). Declared predictions: 5a confirmed (8 of 9 folds move off 0.800);
5b wrong (Sorin 0% NOVEL, ferguson 9%; the highest were the broken folds and Danenberg 42.5%);
5c wrong in 3 folds (matched-cut weighted ARI Keren 0.782, breast 0.843, Sorin 0.846 < 0.85).
v2 replaced ONLY that fallback with the nearest-feasible cut (see run(fallback='nearest')).
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_label_space as labels                                   # noqa: E402
from config import WORK, REPORTS, SPECS                              # noqa: E402

FOLD_DIR = os.path.join(WORK, 'spaces')
V1_DIR = os.path.join(FOLD_DIR, '_v1_2026-09-11')   # the first run, kept (degenerate fallback)
FOLD_EXPECT = 'gate1b_v4_expect.csv'        # the shipped Gate 1b cases (gate1b_fold_expect check 4)


def fold_paths(fold):
    """work/spaces/label_map_{fold}.csv and prototypes_{fold}.npy - benchmark_protocol.yaml
    required_outputs.label_space. Stages 3 and 6 read these per fold."""
    return (os.path.join(FOLD_DIR, f'label_map_{fold}.csv'),
            os.path.join(FOLD_DIR, f'prototypes_{fold}.npy'))


# ------------------------------------------------------------------ the chain, run on a subset
def subset(S, cohorts):
    """A signature cache holding only `cohorts`, in the original node order."""
    m = np.isin(S['cohort'], list(cohorts))
    out = {k: (v[m] if isinstance(v, np.ndarray) and v.shape[:1] == S['cohort'].shape else v)
           for k, v in S.items() if k not in ('nodes', 'dropped', 'rng')}
    rm = np.isin(S['rng_cohort'], list(cohorts))
    out['rng_cohort'], out['RNG'] = S['rng_cohort'][rm], S['RNG'][rm]
    out['nodes'] = pd.DataFrame(dict(cohort=out['cohort'], label=out['label'],
                                     prev=out['PREV'], n_cells=out['NCELL']))
    out['rng'] = {c: out['RNG'][i] for i, c in enumerate(out['rng_cohort'])}
    return out, np.flatnonzero(m)


def run(S, tag, force_tau=None, fallback='stable'):
    """Exactly the sequence in build_label_space.main(), nothing added and nothing skipped.

    `force_tau` is used ONLY by the matched-cut diagnostic (checks 7/8), never by checks 1-4.

    `fallback` decides the cut when NO cut is usable. 'stable' (default, and what Stage 1b and
    this control do) = the most stable cut over the whole sweep. 'nearest' = the cut with the
    smallest labels.guard_violation, ties to the most stable - the fold-local spaces use it
    (panel/gate1b_fold_v2_expect.csv), because 'stable' returned degenerate partitions there.
    """
    nodes = S['nodes']
    Z, P, Wm = labels.rescale(S)
    SIM, C, EV, CX = labels.containment(S, (Z, P, Wm))
    sweep = labels.choose_cut(S, SIM, EV)
    viol = labels.guard_violation(sweep)          # a separate series: the sweep table is unchanged
    ok = sweep[sweep.usable]
    usable = len(ok) > 0
    if not usable:
        print(f'  [{tag}] NO usable cut - no granularity satisfies all three guards'
              + (' -> nearest-feasible fallback' if fallback == 'nearest' else ''))
        ok = sweep[viol == viol.min()] if fallback == 'nearest' else sweep
    tau = float(ok.cut[ok.stability.idxmax()]) if force_tau is None else float(force_tau)
    memb0 = labels.cluster(np.arange(len(nodes)), SIM, EV, tau)
    memb1, splits = labels.refine(S, memb0, SIM, EV)
    memb, D, gi = labels.nesting(S, memb1, C, EV, SIM=SIM, cut=tau)
    ncoh = np.array([nodes.cohort[np.isfinite(P[:, t])].nunique() for t in range(P.shape[1])])
    names, cen, zc = labels.name_clusters(S, memb, P, ncoh)
    row = sweep[sweep.cut == tau].iloc[0]
    print(f'  [{tag}] {len(nodes)} labels | cut {tau:.3f} | {memb0.max()+1} at the cut -> '
          f'{memb1.max()+1} refined -> {memb.max()+1} after SCC | usable_window={usable}')
    # cnames / cen / graph added 2026-09-11 for the fold-local spaces (build_fold_label_spaces.build_fold): the
    # per-cluster names, the centroids that become prototypes, and the nesting graph that scores
    # nested cases. Added keys only - nothing this file reads changed.
    return dict(tag=tag, nodes=nodes, SIM=SIM, EV=EV, tau=tau, memb=memb,
                cnames=list(names), cen=cen, graph=D,
                names=[names[c] for c in memb], n_cut=int(memb0.max() + 1),
                n_refined=int(memb1.max() + 1), n_final=int(memb.max() + 1),
                splits=splits, sweep=sweep, usable=usable,
                violation=float(viol[sweep.cut == tau].iloc[0]),
                biggest_share=float(row.biggest_share), cohort_ari=float(row.cohort_ari),
                stability=float(row.stability))


def admit(D, rows, members, tau):
    """THE admission rule, in one place (gate1b_fold_expect.csv check 3).

    Each row joins the cluster whose members are closest ON AVERAGE, and is admitted only if that
    average is within the cut `tau` the partition was built at and below FAR - exactly what average
    linkage would have done had the label arrived last. Otherwise -1: NOVEL, never forced into the
    nearest cluster. Ties go to the lowest cluster id.

    `members` maps cluster id -> row indices of D. Returns (cluster, avg_distance, admitted) arrays.
    Used by build_fold(). (The archived frozen-holdout placement and s9_newcohort.place() used
    it too - there were two copies of this loop before, and plan F4 would have made a third.)
    """
    out, dist, adm = [], [], []
    for r in rows:
        best_k, best_d = -1, np.inf
        for k in sorted(members):
            mem = members[k]
            d = float(np.mean(D[r, mem])) if len(mem) else np.inf
            if d < best_d:
                best_k, best_d = k, d
        ok = bool(np.isfinite(best_d) and best_d <= tau and best_d < labels.FAR)
        out.append(best_k if ok else -1)
        dist.append(best_d)
        adm.append(ok)
    return np.array(out, int), np.array(dist), np.array(adm, bool)


# ----------------------------------------------------------------------------- fold-local (F4)
def folds(cohorts):
    """Every fold the protocol scores, DERIVED from config.SPECS - never written by hand.

    7 LOCO folds, one per training cohort, plus one LOTO fold per tissue with >= 2 training
    cohorts. A single-cohort tissue's LOTO fold IS its LOCO fold (protocol splits.secondary) and
    is not built twice.
    """
    train = [c for c in cohorts if SPECS[c]['role'] == 'train']
    out = {f'loco_{c}': [c] for c in train}
    by_t = {}
    for c in train:
        by_t.setdefault(SPECS[c]['tissue'], []).append(c)
    for t, cs in sorted(by_t.items()):
        if len(cs) >= 2:
            out[f"loto_{t.replace(' ', '_')}"] = sorted(cs)
    return out


def fold_of(held):
    """The fold name for a held-out cohort set, e.g. ['Sorin'] -> 'loco_Sorin'."""
    held = sorted(held)
    for name, h in folds(list(SPECS)).items():
        if sorted(h) == held:
            return name
    raise KeyError(f'no declared fold holds out exactly {held}')


def build_fold(name, held, S, SIM, EV, Dfull, shipped, ex, tau_ship):
    """One fold-local label space. Every rule is in panel/gate1b_fold_expect.csv.

    Returns a summary dict; writes work/spaces/label_map_{name}.csv + prototypes_{name}.npy.
    """
    nodes = S['nodes'].reset_index(drop=True)
    train = sorted(c for c in nodes.cohort.unique() if c not in held)
    Sf, keep = subset(S, train)

    # ---- check 0: the held-out cohort is not in the table the space is built from
    fc = set(Sf['nodes'].cohort)
    assert not (fc & set(held)), f'{name}: held-out rows reached the fold table - STOP'
    assert fc == set(train), f'{name}: fold table holds {sorted(fc)}, expected {train}'

    # v2 (gate1b_fold_v2_expect.csv check 2): nearest-feasible cut when the window is empty
    r = run(Sf, name, fallback='nearest')
    memb, tau, cnames = r['memb'], r['tau'], r['cnames']
    n_cl = int(memb.max()) + 1

    # ---- check 1: the training block is bit-identical to the full build's
    blk = np.ix_(keep, keep)
    assert np.array_equal(r['SIM'], SIM[blk], equal_nan=True) and \
        np.array_equal(r['EV'], EV[blk], equal_nan=True), \
        f'{name}: check 1 FAILED - dropping {held} moved the training geometry. STOP.'

    # ---- check 3: place the held-out labels by the admission rule
    h_rows = np.flatnonzero(nodes.cohort.isin(held).to_numpy())
    members = {k: keep[memb == k] for k in range(n_cl)}
    h_cl, h_d, h_ok = admit(Dfull, h_rows, members, tau)

    # ---- check 4: unreliable flags, recomputed inside the fold
    nf = Sf['nodes'].reset_index(drop=True)
    hc = labels.score_cases(nf, memb, cnames, ex, r['graph'], cohorts=set(train))
    flagged, exempt = labels.unreliable_clusters(nf, memb, cnames, ex, hc, cohorts=set(train))

    # ---- the map, Stage 1b format; NOVEL rows kept with cluster -1
    why = f'Gate 1b waived failure inside fold {name} - see panel/gate1b_fold_expect.csv check 4'
    rows = [dict(cohort=nf.cohort[i], label=nf.label[i], prev=nf.prev[i],
                 n_cells=int(nf.n_cells[i]), cluster=int(memb[i]), cluster_name=cnames[memb[i]],
                 unreliable=int(memb[i] in flagged),
                 unreliable_reason=why if memb[i] in flagged else '')
            for i in range(len(nf))]
    for j, i in enumerate(h_rows):
        k = int(h_cl[j])
        rows.append(dict(cohort=nodes.cohort[i], label=nodes.label[i], prev=nodes.prev[i],
                         n_cells=int(nodes.n_cells[i]), cluster=k,
                         cluster_name=cnames[k] if k >= 0 else 'NOVEL - no cluster admits it',
                         unreliable=int(k in flagged) if k >= 0 else 1,
                         unreliable_reason=(why if k in flagged else '') if k >= 0 else
                         f'NOVEL: nearest cluster at average distance {h_d[j]:.3f} > tau '
                         f'{tau:.3f}'))
    out = pd.DataFrame(rows)
    os.makedirs(FOLD_DIR, exist_ok=True)
    lm_path, pt_path = fold_paths(name)
    out.to_csv(lm_path, index=False)
    np.save(pt_path, r['cen'])                  # training-cohort centroids only
    # v2 changed ONLY the no-usable-cut fallback, so a fold with a usable cut must reproduce its
    # first-run map byte for byte (gate1b_fold_v2_expect.csv check 2 note)
    v1 = os.path.join(V1_DIR, os.path.basename(lm_path))
    if r['usable'] and os.path.exists(v1):
        assert open(v1, 'rb').read() == open(lm_path, 'rb').read(), \
            f'{name}: has a usable cut but its map differs from the v1 run - v2 changed more than ' \
            f'the fallback. STOP.'

    # ---- 5c: agreement with the shipped space (report only), and the matched-cut diagnostic
    ship = {(a, b): k for a, b, k in zip(shipped.cohort, shipped.label, shipped.cluster)}
    tr_ship = np.array([ship[(a, b)] for a, b in zip(nf.cohort, nf.label)])
    w_tr = nf.n_cells.to_numpy(float)
    adm = out[(out.cluster >= 0)]
    all_ship = np.array([ship[(a, b)] for a, b in zip(adm.cohort, adm.label)])
    rm = run(Sf, f'{name}@{tau_ship:.3f}', force_tau=tau_ship)

    h = out[out.cohort.isin(held)]
    return dict(
        fold=name, held=', '.join(held), tau=round(tau, 3), usable_cut=bool(r['usable']),
        cut_rule='shipped rule' if r['usable'] else 'nearest_feasible',
        violation=round(r['violation'], 4), clusters=n_cl, cohort_ari=round(r['cohort_ari'], 3),
        biggest_share=round(r['biggest_share'], 3),
        unreliable_clusters=len(flagged),
        unreliable_cell_share=round(float(w_tr[np.isin(memb, list(flagged))].sum() / w_tr.sum()),
                                    3),
        held_labels=len(h), held_novel=int((h.cluster < 0).sum()),
        held_novel_cells=round(float(h[h.cluster < 0].n_cells.sum() / max(1, h.n_cells.sum())),
                               3),
        ari_train_w=round(labels.ari(memb, tr_ship, w_tr), 3),
        ari_train_u=round(labels.ari(memb, tr_ship), 3),
        ari_all_w=round(labels.ari(adm.cluster.to_numpy(), all_ship, adm.n_cells.to_numpy(float)),
                        3),
        ari_matched_w=round(labels.ari(rm['memb'], tr_ship, w_tr), 3),
        ari_matched_u=round(labels.ari(rm['memb'], tr_ship), 3),
        cases=hc, novel=h[h.cluster < 0][['label', 'n_cells', 'unreliable_reason']],
        exempt=sorted(exempt),
        sha256=hashlib.sha256(open(lm_path, 'rb').read()).hexdigest()[:16])


def build_folds():
    S = labels.load_signatures()
    why = labels.stale_reason(S, labels.built())
    assert not why, f'signature cache is stale ({why}) - run build_label_space.py first'
    nodes = S['nodes']
    Z, P, Wm = labels.rescale(S)
    SIM, C, EV, CX = labels.containment(S, (Z, P, Wm))
    Dfull = labels.distance(np.arange(len(nodes)), SIM, EV)
    shipped = pd.read_csv(os.path.join(WORK, 'label_map.csv'), keep_default_na=False)
    tau_ship = float(json.load(open(os.path.join(WORK, 'label_graph.json')))['tau'])
    ex = labels.read_expect(FOLD_EXPECT)
    F = folds(sorted(nodes.cohort.unique()))
    print(f'{len(F)} folds: {F}')
    res = [build_fold(n, h, S, SIM, EV, Dfull, shipped, ex, tau_ship) for n, h in F.items()]
    write_fold_report(res, tau_ship)
    # what Stages 3 / 6 read to find a fold's space and carry its flags into their reports
    idx = {r['fold']: dict(held=F[r['fold']], tau=r['tau'], cut_rule=r['cut_rule'],
                           violation=r['violation'], clusters=r['clusters'],
                           sha256_16=r['sha256'], expect='gate1b_fold_v2_expect.csv')
           for r in res}
    json.dump(idx, open(os.path.join(FOLD_DIR, 'index.json'), 'w'), indent=1)
    return res


def fold_space(held):
    """(label_map path, prototypes path, index entry) for the fold holding out `held`.

    Raises if the fold was never built, or if its map no longer matches the hash the build
    recorded - a hand-edited or stale map must never be trained on silently.
    """
    name = fold_of(held)
    lm, pt = fold_paths(name)
    ip = os.path.join(FOLD_DIR, 'index.json')
    if not (os.path.exists(lm) and os.path.exists(ip)):
        raise FileNotFoundError(f'fold space {name} not built - run: python celltype_transfer/build_fold_label_spaces.py '
                                f'--folds')
    entry = json.load(open(ip))[name]
    got = hashlib.sha256(open(lm, 'rb').read()).hexdigest()[:16]
    if got != entry['sha256_16']:
        raise RuntimeError(f'{lm} hash {got} != recorded {entry["sha256_16"]} - rebuild with '
                           f'--folds rather than training on a changed map')
    return lm, pt, dict(entry, fold=name)


def write_fold_report(res, tau_ship):
    L = []
    A = L.append
    A('# Stage 1b - fold-local label spaces (plan F4)\n')
    A('Each fold\'s label space is built from its **training cohorts only**, by the shipped Stage '
      '1b chain with the shipped v4 switches and the shipped cut rule. The held-out cohort\'s '
      'labels are then placed into the frozen partition by the admission rule, or marked NOVEL. '
      'Rules in `celltype_transfer/declared/gate1b_fold_v2_expect.csv`.\n')
    A('**v2 - one POST-HOC change.** The first run (`reports/build_fold_label_spaces_v1.md`, maps in '
      '`work/spaces/_v1_2026-09-11/`) found no usable cut in 3 of 7 LOCO folds, and the declared '
      'fallback (most stable cut over the whole sweep) returned degenerate spaces. v2 replaces '
      'that fallback, and only that, with the NEAREST-FEASIBLE cut (smallest relative guard '
      'violation), chosen by the user after seeing the failure. Folds with a usable cut are '
      'asserted byte-identical to v1.\n')
    A('**Checks 0 and 1 are asserted, not reported** - this file exists only if both held in every '
      'fold: no held-out row reached any fold table, and every fold\'s training-to-training '
      'geometry is bit-identical to the full build\'s.\n')
    cols = ['fold', 'held', 'tau', 'cut_rule', 'violation', 'clusters', 'cohort_ari',
            'biggest_share',
            'unreliable_clusters', 'unreliable_cell_share', 'held_labels', 'held_novel',
            'held_novel_cells']
    T = pd.DataFrame([{k: r[k] for k in cols} for r in res])
    A('## Per fold (5a, 5b)\n')
    A(labels.md_table(T))
    A(f'\n`tau` is the fold\'s OWN cut; the shipped whole-roster space uses {tau_ship:.3f}. '
      f'`held_novel_cells` is the share of the held-out cohort\'s (label-QC\'d) cells in NOVEL '
      f'labels - dropped from the closed-set macro-F1, counted for open-set metrics.\n')
    bad = T[T.cut_rule != 'shipped rule']
    if len(bad):
        A(f'**{len(bad)} fold(s) had NO usable cut** ({", ".join(bad.fold)}): the guards admit no '
          f'granularity, so the cut is the nearest-feasible one (v2, post-hoc); `violation` is how '
          f'far it misses the worst guard, relative to that guard\'s threshold. Every number from '
          f'these folds carries the flag `nearest_feasible`.\n')
    A('`biggest_share` is measured at the cut, before per-branch refinement.\n')
    A('## Agreement with the shipped whole-roster space (5c, report only)\n')
    cols = ['fold', 'tau', 'ari_train_w', 'ari_train_u', 'ari_all_w', 'ari_matched_w',
            'ari_matched_u']
    A(labels.md_table(pd.DataFrame([{k: r[k] for k in cols} for r in res])))
    A(f'\n`train` = over the fold\'s training labels at the fold\'s own cut; `all` = training plus '
      f'admitted held-out labels; `matched` = the same training labels re-cut at the shipped '
      f'{tau_ship:.3f} (a diagnostic, never shipped). Weighted = each label counts by its cells.\n')
    A('## NOVEL held-out labels\n')
    for r in res:
        if len(r['novel']):
            A(f'**{r["fold"]}**\n')
            A(labels.md_table(r['novel']))
            A('')
    A('## Declared cases inside each fold (check 4)\n')
    A('Scored on the fold partition with members from the fold\'s training cohorts only; a case '
      'left with < 2 members is n/a. The waiver list is inherited from '
      f'`{FOLD_EXPECT}` - decided after seeing the full build, so NOT blind to any held-out '
      'cohort (stated in the expect file).\n')
    for r in res:
        c = r['cases']
        req = c[c.required == 1]
        A(f'- **{r["fold"]}**: required {int((req.result == "PASS").sum())} pass, '
          f'{int(((req.result == "FAIL") & (req.waived == 1)).sum())} waived-and-failed, '
          f'{int(((req.result == "FAIL") & (req.waived == 0)).sum())} unwaived FAIL, '
          f'{int((req.result == "n/a").sum())} n/a; exempted clusters {r["exempt"]}')
    A('\n## Outputs (check 6)\n')
    A(labels.md_table(pd.DataFrame([dict(fold=r['fold'], file=f'work/spaces/label_map_{r["fold"]}'
                                      f'.csv', sha256_16=r['sha256']) for r in res])))
    p = os.path.join(REPORTS, 'build_fold_label_spaces.md')
    open(p, 'w', encoding='utf-8').write('\n'.join(L))
    print(f'\nwrote {p}')


def status():
    ip = os.path.join(FOLD_DIR, 'index.json')
    if not os.path.exists(ip):
        print('no fold-local label spaces yet - run with --folds')
        return
    for name, e in json.load(open(ip)).items():
        print(f"  {name:16} held {', '.join(e['held']):20} tau {e['tau']:.3f}  "
              f"{e['clusters']:>3} clusters  {e['cut_rule']:17} sha {e['sha256_16']}")


def main():
    if '--folds' in sys.argv:
        return build_folds()
    status()


if __name__ == '__main__':
    main()

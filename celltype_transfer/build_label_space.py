"""
STAGE 1b - automatic label alignment.  Produces GATE 1b.

WHAT THIS REPLACES. The old `pipeline/step4_labels.py` holds a hand-written dictionary mapping
every native label of every cohort into a fixed Cell Ontology tree. Adding a cohort means
hand-writing a new block. It does not scale, and it assumes the biology fits a human-made tree.
VALIDATION IS EXTERNAL ONLY (protocol v2, 2026-09-11). This project's own hand mapping has been
REMOVED entirely - as a method, as a gate, and as a validation source. It covered 4 of 7 cohorts
and was written by the same author as the method, so it tested nothing external. Agreement is now
measured against published references only: CellMarker 2.0 for marker evidence and the Cell
Ontology for structure. No module under celltype_transfer/ may read a reference file.

WHY NOT TEXT EMBEDDINGS. Measured against the real labels in this roster, text fails in both
directions at once: `SC` and `EC` are one character apart and mean squamous carcinoma and
endothelial cell; `CD4 T cell` and `CD8 T cell` have cosine ~0.97 and are opposite cell types;
`vasculature`/`Vessel`/`Endothelial`/`EC` share no word and are one cell type. Hardest of all,
Phillips and CRC BOTH have a label spelled exactly `tumor cells`, and they are different cell
types - Phillips is a T-cell lymphoma. No string method can get that right. Marker profiles can,
and text is therefore out of the distance entirely (M4, alpha = 0).

THE METHOD, four mechanisms. Each is documented at the function that implements it, including
where building it showed the plan's version to be wrong - those notes are kept deliberately,
because every one of them was found by a measurement rather than by argument.

M1  A label's signature is a DISTRIBUTION. Per (cohort, label, marker) store nine quantiles of
    the Gate 1 winning transform (u_coh, the per-cohort ECDF), the MEAN RANK, prevalence, and the
    within-label co-expression matrix. Position for merging is the mean rank - the Mann-Whitney
    statistic behind the 0.91-1.00 AUROCs Gate 1 measured - and the quantiles carry the spread,
    which is what the directional layer needs. See `rescale`.

M2  Two relations, not one. A SYMMETRIC distance on position decides merging; DIRECTED
    CONTAINMENT on spread decides nesting. The plan used containment for both, and measurement
    showed that is backwards: interval overlap tracks how WIDE two labels are, not where they
    sit. See `containment`.

M2b The nesting graph is forced acyclic by SCC CONTRACTION, then closed only where evidence was
    missing. Contraction gives a DAG by theorem rather than by hope, and reads correctly in
    biology: labels that mutually contain each other ARE one type. Blanket transitive closure is
    rejected - it amplifies one false edge across everything downstream and does not even
    guarantee acyclicity, since closing an existing cycle just turns it into a complete blob.

M3  Granularity is chosen PER BRANCH by stability, not by a global cut. Each cluster is offered a
    binary split, kept only if it reproduces with a cohort held out. This is what lets CD4 T and
    CD8 T separate inside the T-cell branch without shattering the rest of the label space, and
    it is size-agnostic, so a rare-but-global type is safe from being swallowed. See `refine`.

M4  Clusters are NAMED from their top discriminative markers. Text does naming only, never
    distance, so a cohort shipping `Population_1` costs nothing.

NOTHING IS TUNED AGAINST THE ANSWER. Granularity is never tuned against any reference, external
or otherwise. Four label-free objectives were built and every one is biased toward an end of
the range (see `choose_cut`), so instead the two trivial ends are excluded by guards that state
what a usable shared label space IS - no cluster may hold a quarter of all labels, and at least
95% of cells must sit in a cluster that spans more than one cohort - and stability chooses inside
what is left. The agreement curve is reported next to the stability curve across the whole sweep,
so a reader can see for themselves whether the chosen point was lucky.

    python build_label_space.py                  # everything: build, cluster, all 7 checks, report
    python build_label_space.py --evidence-sweep  # check 3 only: evidence coverage + floor sweep
    python build_label_space.py --graph-check     # check 7 only: DAG assert, SCCs, transitivity
    python build_label_space.py --resign          # force signatures to be recomputed
"""
import json
import os
import sys
import time
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # runs from any directory
import config
import panel_utils
from config import SPECS, WORK, PANEL, REPORTS, FIGURES, raw_table, SEED

# ------------------------------------------------------------------ declared constants
QLEV = np.array([0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98])
I_LO, I_HI, I_MED = 2, 6, 4          # indices of q10, q90, q50 in QLEV

MIN_CELLS = 100      # below this a label's quantiles are noise, not a signature
DELTA = 0.05         # geometric-mean floor: one zero-overlap marker vetoes, but not outright
W_MIN = 0.10         # a marker counts as INFORMATIVE if its between-label range reaches this
K_EVID = 8           # evidence floor: informative shared markers needed for a direct edge
SCALE_FLOOR = 0.05   # floor on the within-cohort between-label scale, guards a flat marker
NEST_MIN = 0.75      # containment needed before a directed nesting edge is drawn
NEST_RELATED = 1.5   # a nesting pair must also be within this multiple of the cut of each other
MARGIN = 0.10        # asymmetry needed to call a pair NESTED rather than merged
CUT_GRID = np.round(np.arange(0.30, 1.16, 0.025), 4)
FAR = 5.0            # distance stand-in for "below the evidence floor", i.e. never comparable

# ------------------------------------------------------------------ ROUTE A (2026-09-11)
# Four measured defects in the v1 distance. Each fix sits behind its own switch so its effect
# can be attributed separately - D-27's discipline, change one thing at a time.
POSCUT = np.array([0.50, 0.75, 0.90])   # ECDF cut-offs for the positive-fraction channel
I_P90 = 2                                # index of the 0.90 cut inside POSCUT
LIN_FLOOR = 0.01                         # keeps sum(weights) > 0 when nothing gates cleanly
# MEASURED 2026-09-11 on the 7-cohort roster, granularity matched at ~25 clusters, scored on
# the required cases of panel/gate1b_v2_expect.csv:
#
#   v1 baseline (all off) .................. 2/9 required
#   veto only .............................. 2/9      the veto alone is neutral
#   veto + posfrac ......................... 6/9      SHIPS
#   veto + lineage ......................... 3/9
#   veto + posfrac + lineage ............... 6/9      lineage adds nothing on top
#
# So `posfrac` is the fix and `veto` is its carrier. Cases repaired: tumour_epithelial, stroma,
# cd4_vs_cd8, tumour_danenberg - all required, all previously failing.
# ============================================================================================
# CORRECTION, 2026-09-11 (later the same day). THE ROUTE A NUMBERS BELOW WERE A BLOB ARTIFACT.
#
# The ablation scored variants on "required declared cases passed" at matched granularity. That
# metric is biased toward coarse partitions: a `same` case is satisfied by ANY cluster that
# swallows its members. Checked afterwards, every `same` pass at the cuts reported was a merge
# into the single largest cluster - the keratin+/CD45- blob that fuses tumour with stroma - and
# at those cuts `same_name_different_type` FAILS, putting Phillips' T-cell-lymphoma `tumor
# cells` in with colorectal epithelium. At every finer cut, 0 of 12 `same` cases pass.
#
# Re-scored with a blob-proof metric (a `same` case counts only if it merges OUTSIDE the largest
# cluster), each variant at its own best cut with the size guard held:
#
#     v1 position, weighted RMS .......  30 clusters · genuine same 4/12 · different 3/3
#     position + posfrac q90 ..........  51 clusters · genuine same 2/12 · different 3/3
#     posfrac x3 only .................  30 clusters · genuine same 4/12 · different 2/3
#
# So NO Route A variant beats v1. At matched granularity posfrac trades `macrophages` for
# `endothelium` and loses a discrimination case. The "2/9 -> 7/9" claim below is WITHDRAWN.
# The v1 RMS path was removed when posfrac was committed and has to be restored to be shipped.
# ============================================================================================
RA = dict(
    # SHIPS: v1 weighted RMS on the rescaled position. Restored 2026-09-11 after the blob-proof
    # re-scoring above showed no Route A variant beats it. `--agree` selects the agreement path
    # (position and/or posfrac below), kept for comparison and for the recorded negative result.
    distance = 'agree' if '--agree' in sys.argv else 'rms',
    # POSITION (rescaled mean rank) is OFF. Measured 2026-09-11, granularity matched at ~25
    # clusters on the required cases of gate1b_v2_expect.csv:
    #     position only, linear exp(-dP) ......... 2/9
    #     position only, gaussian scale 1.5/1.0/0.7 2/9   (a genuinely saturating form)
    #     position only, gaussian scale 0.5 ...... 3/9
    #     position + posfrac q90 ................. 6/9
    #     position + posfrac x3 .................. 3/9   position HURTS the 3-cut form
    #     posfrac x3 alone ....................... 7/9   [WITHDRAWN - blob artifact, see CORRECTION above]
    # The rescaled mean rank carries almost no cross-cohort signal in ANY form. It also caps
    # cross_cohort_share at 0.940; dropping it lifts that to 0.973 at every cut, which is what
    # finally lets Sorin into the shared space.
    position = '--position'  in sys.argv,
    # POSITIVE FRACTION at all three ECDF cut-offs. This is the whole signal. Cell identity is
    # WHICH markers are on, not where the average cell sits on a continuum, and unlike position
    # it saturates - two labels opposite on CD20 hit the DELTA floor and the merge is vetoed.
    posfrac  = '--no-posfrac' not in sys.argv,
    # Declared label QC, panel/label_qc.csv - see load_qc().
    labelqc  = '--no-labelqc' not in sys.argv,
    # DISPROVEN, kept as switches so they are not proposed again. See rescale() and
    # lineage_score() for the measurements.
    robust   = '--robust'    in sys.argv,
    lineage  = '--lineage'   in sys.argv,
    # CENTRING of each (cohort, marker) in rescale(): 'midrange' = (max + min) / 2 of the label
    # positions (SHIPS); 'median' = the median LABEL (v1, `--median-centre`). See rescale().
    centre   = 'median' if '--median-centre' in sys.argv else 'midrange',
    # refine() STRADDLE guard (SHIPS; `--no-straddle` for v1) - see _split_ok().
    straddle = '--no-straddle' not in sys.argv,
)
MIN_STRADDLE = 2  # cohorts that must hold labels on BOTH sides of a kept split. Declared
                  # 2026-09-11 BEFORE its first run, to mirror the existing ">= 2 cohorts per
                  # side" guard. Not tuned.
EXCL_HIGH = 0.60  # check 2b: a label is marker-high when this share of its cells sits above its
                  # cohort's median. Declared before the first run; 0.50 / 0.70 are reported
                  # in the measurement block below as sensitivity.
# MEASURED 2026-09-11, centring x straddle, 7 cohorts, label QC on, default cut rule, refine and
# nesting as shipped. All four pass all three hard guards. Declared cases of gate1b_v3_expect.csv
# scored BLOB-PROOF (benchmark_protocol.yaml declared_case_scoring): a `same` pass counts only
# outside the largest cluster and outside any cluster named by no positive marker.
#
#                        cut  clusters  same pairs  same cases  diff  required  blob+resid cells
#   median                .750    34      57/125      2/12      3/3    2/9        26.9%
#   median   + straddle   .750    33      72/125      4/12      3/3    2/9        26.9%
#   midrange              .800    26      46/125      1/12      3/3    2/9        21.5%
#   midrange + straddle   .800    21      86/125      6/12      3/3    4/9        21.9%   SHIPS
#
# The two need each other. Midrange fixes the distance - with the global cut alone, at matched
# granularity (23-28 clusters), it keeps 92/125 `same` pairs genuinely together against 70-72
# for median, with 10% of cells in blob/residual against 29-52%. Without straddle, refine() then
# undid it by splitting B cells, macrophages and endothelium along cohort-group lines (92 -> 46).
#
# INDEPENDENT CHECK, not using the declared cases: benchmark_protocol.yaml biological_exclusion,
# pairs in panel/gate1b_exclusions.csv (declared before the run). Share of cells in clusters
# that mix mutually exclusive lineages, "high" = >= 60% of a label's cells above its cohort median:
#   median 42.0%  ->  midrange + straddle 20.8%     (sensitivity: 50% cut 43.6 -> 29.3,
#                                                     70% cut 24.6 -> 13.7)
# CAVEAT: both switches were chosen after looking at this data. The blob-proof metric and the
# exclusion check limit how much that can flatter them; they do not remove it. The external CL
# reference (plan F5) is the check that will.
SPLIT_SUPPORT = 0.5  # a per-branch split is kept only if its LOCO reproducibility reaches this
MIN_SPLIT_LABELS = 2  # neither side of a split may be smaller than this
COHORT_ARI_MAX = 0.20    # HARD GUARD: a clustering that is really the cohort partition is a fail
MAX_CLUSTER_SHARE = 0.25  # HARD GUARD: no single cluster may hold this share of all labels
MIN_CROSS_SHARE = 0.95    # HARD GUARD: share of CELLS that must sit in a cross-cohort cluster

SIG_CACHE = os.path.join(WORK, 'label_signatures.npz')
RARE_GLOBAL = ['Plasma', 'NK', 'DC']     # the M3 test cases named in the plan


# ============================================================== M1 - signatures
def registry():
    return pd.read_csv(os.path.join(WORK, 'marker_registry.csv'), keep_default_na=False)


def built():
    return [c for c in SPECS if os.path.exists(raw_table(c))]


def cohort_markers(cohorts):
    """triple -> raw column, per cohort. Non-protein channels are excluded here."""
    r = registry()
    r = r[(r.kind != 'non_protein') & r.cohort.isin(cohorts)]
    out = {}
    for c, g in r.groupby('cohort'):
        out[c] = panel_utils.cols_for(r, c, sorted(g.triple.unique()))
    gene = r.drop_duplicates('triple').set_index('triple').gene.to_dict()
    return out, gene


def load_qc():
    """Declared label QC - native labels that are not a cell type, from panel/label_qc.csv.

    TWO CATEGORIES, both read off the annotator's own words rather than any biological
    judgement of ours:

      unassigned / not_a_cell   `dirt`, `undefined`, `Unidentified`. The cohort's explicit
                                non-answer. A cluster built around "everything the panel could
                                not resolve" is not a cell type.
      ambiguous_multitype       a label naming TWO cell types - `tumor cells / immune cells`,
                                `Macrophages & granulocytes`. The annotator is saying they could
                                not decide, and such a label BRIDGES the two compartments it
                                names, which is exactly how average linkage fuses them.

    DELIBERATELY NOT DROPPED, so the rule cannot be read as curating the answer:
      `nerves`, `adipocytes`, `smooth muscle`, `lymphatics`  - real cell types, merely rare.
      `immune cells`, `Other_immune`, `T other`, `APC`, `Naive immune cell` - real cells under
                                a coarse label. Coarse is not junk.
      `MHC I & II^{hi}`         carries an ampersand but names ONE type by TWO markers, which is
                                why the rule is a declared file and not a string match on '&'.
    """
    p = os.path.join(PANEL, 'label_qc.csv')
    if not os.path.exists(p):
        return {}
    q = pd.read_csv(p, keep_default_na=False)
    return {(r.cohort, r.native_label): (r.category, r.reason) for r in q.itertuples()}


def build_signatures(cohorts):
    """One pass per cohort. Reads every cell; writes a signature per (cohort, label).

    The value transform is `u_coh` - the per-cohort empirical CDF - which is the arm that won
    GATE 1. It is applied to EVERY marker the cohort measures, not just the 9-marker core:
    Gate 1 restricted to the core so that no arm could win on imputation, but here a wider panel
    is exactly what makes a cohort pair comparable, and the evidence floor handles thin pairs.
    """
    cmark, gene = cohort_markers(cohorts)
    triples = sorted({t for m in cmark.values() for t in m})
    ti = {t: i for i, t in enumerate(triples)}
    T = len(triples)

    keys, QT, MEAN, PREV, NCELL, COEX, POS, dropped = [], [], [], [], [], [], [], []
    RNG = {}
    for c in cohorts:
        t0 = time.time()
        cols = cmark[c]
        use = sorted(cols)
        df = pd.read_parquet(raw_table(c),
                             columns=['native_label'] + panel_utils.read_cols(cols, use))
        lab = df.native_label.astype(str).str.strip()
        U = panel_utils.matrix(df, cols, use)
        # identical transform to the Gate 1 winner V3: mid-rank ECDF inside (cohort, marker)
        U = U.rank(pct=True, method='average').astype('float32')
        n_tot = len(U)

        vc = lab.value_counts()
        bad = {'', 'nan', 'none', 'null', 'na'}
        qc = load_qc() if RA['labelqc'] else {}
        good = [l for l in vc.index if l.lower() not in bad and vc[l] >= MIN_CELLS
                and (c, l) not in qc]
        for l in vc.index:
            if l not in good:
                if (c, l) in qc:
                    why = f'label QC: {qc[(c, l)][0]}'
                elif l.lower() in bad:
                    why = 'unlabelled'
                else:
                    why = f'below MIN_CELLS={MIN_CELLS}'
                dropped.append(dict(cohort=c, native_label=l, cells=int(vc[l]), reason=why))

        Uv = U.to_numpy()
        col_of = {t: k for k, t in enumerate(use)}
        meds = {}
        for l in good:
            m = (lab == l).to_numpy()
            sub = Uv[m]
            q = np.full((T, len(QLEV)), np.nan, 'float32')
            qq = np.nanquantile(sub, QLEV, axis=0).astype('float32').T      # (M, nq)
            mn = np.full(T, np.nan, 'float32')
            mm = np.nanmean(sub, axis=0).astype('float32')
            cx = np.full((T, T), np.nan, 'float32')
            with np.errstate(invalid='ignore', divide='ignore'):
                cc = np.corrcoef(sub, rowvar=False).astype('float32')
            for t, k in col_of.items():
                q[ti[t]] = qq[k]
                mn[ti[t]] = mm[k]
                for t2, k2 in col_of.items():
                    cx[ti[t], ti[t2]] = cc[k, k2]
            # POSITIVE FRACTION - the biological statement a mean rank cannot make.
            # "80% of these cells are CD20-high" and "all of them sit mid-range" have the same
            # mean rank and are different cell types. The cut is on u_coh, already a per-cohort
            # ECDF, so this is comparable across cohorts without any rescaling.
            pf = np.full((T, len(POSCUT)), np.nan, 'float32')
            ok = np.isfinite(sub)
            den = np.maximum(ok.sum(0), 1)
            pf[np.array([ti[t] for t in use])] = np.stack(
                [((sub > cut) & ok).sum(0) / den for cut in POSCUT], 1).astype('float32')

            keys.append((c, l))
            POS.append(pf)
            QT.append(q)
            MEAN.append(mn)
            COEX.append(cx)
            PREV.append(sub.shape[0] / n_tot)
            NCELL.append(int(sub.shape[0]))
            meds[l] = mn

        # marker informativeness INSIDE this cohort: how far apart its label medians spread
        M = np.vstack([meds[l] for l in good])                              # (L, T)
        rng = np.full(T, np.nan, 'float32')
        seen = np.array([ti[t] for t in use])
        rng[seen] = np.nanmax(M[:, seen], 0) - np.nanmin(M[:, seen], 0)
        RNG[c] = rng
        print(f'  {c:9s} {n_tot:>9,} cells  {len(use):3d} markers  '
              f'{len(good):2d} labels kept  {time.time()-t0:5.1f}s')
        del df, U, Uv

    np.savez_compressed(
        SIG_CACHE,
        cohort=np.array([k[0] for k in keys]), label=np.array([k[1] for k in keys]),
        QT=np.stack(QT), MEAN=np.stack(MEAN), COEX=np.stack(COEX), POS=np.stack(POS),
        PREV=np.array(PREV), NCELL=np.array(NCELL),
        triples=np.array(triples), gene=np.array([gene[t] for t in triples]),
        rng_cohort=np.array(list(RNG)), RNG=np.stack([RNG[c] for c in RNG]),
        dropped=np.array(json.dumps(dropped)))
    return load_signatures()


def load_signatures():
    z = np.load(SIG_CACHE, allow_pickle=False)
    S = dict(z)
    S['nodes'] = pd.DataFrame(dict(cohort=S['cohort'], label=S['label'],
                                   prev=S['PREV'], n_cells=S['NCELL']))
    S['dropped'] = pd.DataFrame(json.loads(str(S['dropped'])))
    S['rng'] = {c: S['RNG'][i] for i, c in enumerate(S['rng_cohort'])}
    return S


# ============================================================== M2 - directed containment
def rescale(S):
    """Put every cohort's signatures into ONE comparable unit: between-label spread.

    THIS IS THE FIX FOR A REAL FAILURE, kept in the code because the failure is instructive.
    The first build compared labels on raw `u_coh` and produced clusters that tracked COHORT,
    not cell type - 54 labels of every kind from CRC+UPMC+Phillips in one blob, all 9 Keren
    immune labels in another, all 16 Sorin labels as singletons.

    The reason is that `u_coh` is a per-cohort ECDF, so where a label SITS depends on that
    cohort's composition. Keren is 50.3% keratin-positive tumour, so keratin's ECDF is dominated
    by tumour cells and every other Keren label is pushed low; CRC's tumour is 18.4%, so the same
    biology lands somewhere else entirely. Comparing absolute positions across cohorts therefore
    compares compositions, which is precisely the cohort fingerprint this whole rebuild exists to
    remove.

    What IS comparable is a label's position relative to the OTHER LABELS OF ITS OWN COHORT -
    "ferguson SC has the highest pan-keratin in ferguson" and "CRC tumor cells has the highest
    cytokeratin in CRC" are the same statement. So each (cohort, marker) is centred on the MIDRANGE
    of its label positions (v1 used the median label - see the CENTRING comment in the body for
    why that was replaced) and divided by their max-min spread, and the SAME affine map is applied
    to all nine quantiles of every label. Being affine, it leaves the containment direction
    untouched: a narrow subtype inside a broad parent stays narrow inside broad.

    ONE statistic does both jobs here, deliberately. An earlier version selected markers by the
    RANGE of the label medians but divided by their IQR, and that is inconsistent in a way that
    breaks exactly the markers that matter most: a lineage marker is high in ONE label and flat
    in the rest - CD20 in B cells, keratin in tumour - so its range is large while its IQR is
    near zero. Dividing by the IQR then sent those markers to |z| in the hundreds and every
    cross-cohort similarity collapsed to 0.000. Both the scale and the informativeness weight are
    now the p90-p10 spread of the label positions, so a marker cannot be called informative by
    one rule and rescaled by another.

    POSITION IS THE MEAN RANK, NOT THE MEDIAN, and this is the third measured correction. With
    the median, Sorin had only 5 of its 17 markers showing ANY between-label spread and Keren 19
    of 39, so Sorin's 16 labels could not reach the evidence floor against anything and came out
    as 16 singletons. The cause is zero inflation: Sorin arrives uint8, so on most markers over
    half the cells are 0, the mid-rank ECDF ties them all at one value, and nearly every label
    inherits that same median. The median cannot see that 80% of B cells are CD20-positive while
    5% of tumour cells are - but the MEAN RANK can, because it is the average percentile.

    That statistic is not a new invention here: mean rank is the Mann-Whitney statistic, which is
    exactly the AUROC that Gate 1 used to show labels track their markers at 0.91-1.00 for B
    cells, CD8 T and Tregs in every cohort. So the position used for alignment is the same
    quantity that was already measured to work.

    M1 said "a distribution, not a mean", and half of that is upheld and half corrected. The
    correction: a mean of RANKS is a bounded, tie-safe location statistic, not the mean-of-values
    M1 was objecting to. What is upheld: the quantiles are still stored and still do real work -
    they are the whole basis of the DIRECTIONAL layer, where broad-parent versus narrow-subtype
    is a statement about spread that no location statistic can make.
    """
    QT, MEAN, nodes = S['QT'], S['MEAN'], S['nodes']
    Z = QT.copy()
    P = MEAN.copy()
    W = np.zeros((len(nodes), QT.shape[1]), 'float32')
    for c in nodes.cohort.unique():
        m = (nodes.cohort == c).to_numpy()
        pos = MEAN[m]                                                      # (L, T)
        # CENTRING DEPENDED ON ANNOTATION DEPTH (measured 2026-09-11, the cause behind the two
        # tumour waivers). The median LABEL moves with how many labels a cohort puts on each side
        # of a marker. UPMC has 6 of 16 labels keratin-high, so its median label is itself
        # tumour-like and UPMC Tumor rescaled to keratin P = 0.05; ferguson labels tumour once and
        # SC rescaled to 0.72 - same biology, 14x apart, from raw ranks 0.573 and 0.556.
        # The midrange depends only on the most negative and most positive label, so it does not
        # move when one compartment is split into sub-labels. It is also the centre that matches
        # the max-min scale below: with it every cohort's labels span exactly [-0.5, 0.5].
        #
        # WHAT IT FIXED AND WHAT IT DID NOT (measured, pan-keratin P). The top keratin label of
        # CRC, Keren, Sorin and Phillips now all sit at +0.50, and CRC tumour plus UPMC's five
        # tumour sub-labels join one 5-tissue epithelial cluster. UPMC plain `Tumor` stays at
        # +0.09 - not a centring effect: it is 8th of UPMC's 16 labels on keratin (mean rank
        # 0.573, POS50 0.64), below every UPMC tumour sub-label and below APC. Danenberg's
        # CK^{med}/CK^{lo} labels stay near 0: Danenberg grades its epithelium ON keratin level,
        # so its epithelial labels spread across its own range. Neither is solved here.
        if RA['centre'] == 'midrange':
            ctr = (np.nanmax(pos, 0) + np.nanmin(pos, 0)) / 2.0
        else:
            ctr = np.nanmedian(pos, 0)
        # ROUTE A: p90-p10, not max-min. max-min is the least robust scale statistic there is -
        # one extreme label sets the scale for the whole (cohort, marker), and a cohort that
        # annotates finely (Danenberg, 32 labels) is more likely to contain an extreme, so
        # annotation depth was silently compressing that cohort's positions.
        if RA['robust']:
            spread = (np.nanpercentile(pos, 90, axis=0) - np.nanpercentile(pos, 10, axis=0))
        else:
            spread = np.nanmax(pos, 0) - np.nanmin(pos, 0)
        sc = np.maximum(spread, SCALE_FLOOR)
        Z[m] = (QT[m] - ctr[None, :, None]) / sc[None, :, None]
        P[m] = (pos - ctr[None, :]) / sc[None, :]
        W[m] = np.nan_to_num(spread, nan=0.0)
    return Z, P, W


def lineage_score(S):
    """How cleanly each marker GATES, derived from the data. No hand-written marker classes.

    A lineage marker is high in a FEW labels and flat in the rest - CD20 in B cells, keratin in
    tumour - so the top label's positive fraction sits far above the median label's. A functional
    marker (PD-1, Ki67, VISTA) is graded everywhere, so that gap is small. Averaged over the
    cohorts that measure it.

    WHY THIS IS A WEIGHT AND NOT A CLASSIFICATION. There is no threshold to declare, so there is
    nothing to tune and nothing to defend. Without it the evidence floor counts CD20 and VISTA as
    equal evidence, which is why the pairs with the WIDEST shared panels clustered worst:
    CRC-Phillips share 47 markers of which ~30 are checkpoint and activation markers that agree
    on every pair and dilute the distance, while Sorin-ferguson share 11 that are all lineage.
    """
    POS, nodes = S['POS'][:, :, I_P90], S['nodes']
    T = POS.shape[1]
    acc, cnt = np.zeros(T), np.zeros(T)
    for c in nodes.cohort.unique():
        p = POS[(nodes.cohort == c).to_numpy()]                             # (L, T)
        seen = np.isfinite(p).any(0)
        with np.errstate(invalid='ignore'):
            d = np.nanmax(p, 0) - np.nanmedian(p, 0)
        acc[seen] += np.nan_to_num(d[seen])
        cnt[seen] += 1
    lin = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
    return np.maximum(lin, LIN_FLOOR).astype('float32')


def containment(S, Z=None):
    """Two separate relations, because they answer two different questions.

    SIM[i,j] - SYMMETRIC, drives clustering. `exp(-d)` where d is the RMS difference between the
        two labels' median POSITIONS over the scored markers.

    CON[i,j] - DIRECTED, drives nesting only. The fraction of i's q10-q90 spread lying inside
        j's, so `i ⊂ j` when high. This is the plan's containment measure.

    WHY POSITION ONLY IN THE SYMMETRIC LAYER - measured, and it corrects half of M1. Widening the
    symmetric distance from the median to more of the quantile curve makes it monotonically
    worse, because the SPREAD is where the cohort effect lives (segmentation quality, dynamic
    range, uint8 quantisation) while the position is the comparable part. Separation between
    known-same and known-different pairs, and the residual within-vs-cross-cohort offset:

        quantiles used     separation gap    cohort offset
        median only              0.196            0.003
        q25,q50,q75              0.135            0.051
        q10 ... q90              0.075            0.087

    So M1 is half right and half wrong, and both halves are kept: storing the distribution rather
    than a mean is what makes the DIRECTIONAL layer possible at all - a broad parent versus a
    narrow subtype is a statement about spread - but merging must be decided on position. The
    quantiles are not discarded; they moved to the relation they actually serve.

    THE PLAN USED CONTAINMENT FOR BOTH AND THAT IS WRONG ON THIS DATA - measured, not argued.
    Interval overlap is dominated by how WIDE two labels are, not by where they SIT, and after
    rescaling almost every label is wider than the between-label spread, so overlaps are large
    within a cohort and small across cohorts whatever the biology:

        UPMC Tumor          vs UPMC CD8 T cell   0.835   <- opposite cell types, scored HIGH
        CRC tumor cells     vs CRC B cells       0.799   <- opposite cell types, scored HIGH
        Keren Keratin+tumor vs ferguson SC       0.162   <- the SAME cell type, scored LOW

    That ordering is exactly backwards, and it is what produced the first build's clusters:
    54 mixed labels from CRC+UPMC+Phillips in one blob and all 16 Sorin labels as singletons.
    Position is what actually carries cell-type identity here - Gate 1 measured per-label marker
    AUROCs of 0.91-1.00 for B cells, CD8 T and Tregs in every cohort - so position decides
    merging and containment is demoted to deciding DIRECTION between clusters that are already
    related. The plan's M2 shape survives; only which statistic feeds which decision changed.

    EVID[i,j] = number of INFORMATIVE shared markers. A marker is informative for a cohort pair
    when its between-label range reaches W_MIN in BOTH cohorts - a marker that is flat in one
    cohort separates nothing there, so counting it as evidence would overstate what was measured.
    That is why the floor bites at all: every cohort pair in this roster shares at least 11 raw
    markers, so a raw count would make the floor dead code.

    EVERY informative shared marker is scored, and the panel-size problem is solved by
    NORMALISING PER COHORT PAIR instead. Cohort pairs share very different numbers of markers
    (CRC-Phillips 47, Sorin-ferguson 11), and under any averaging rule a real disagreement is
    diluted by however many markers happen to agree - so the same biological difference scores
    differently depending on panel size, and distances are not comparable between blocks.

    Selecting a fixed number of "most informative" markers per cohort pair was tried first and
    fails, twice, for the same underlying reason: NO GLOBAL RANKING OF MARKERS CAN SERVE EVERY
    LABEL PAIR. Ranking by the p90-p10 spread of label positions dropped `MS4A1` (CD20) and
    pan-keratin out of the top 8 of every cohort pair - because CD20 is high in exactly ONE label
    out of 16-27, so trimming the top decile deletes precisely the signal that defines B cells.
    Switching the ranking to max-min brought CD20 back but swung CRC-UPMC onto stromal markers,
    and `CD4 T vs CD8 T` (0.801) stopped being distinguishable from `Tumor vs CD8 T` (0.791). A
    marker that identifies one rare cell type is worthless for every other pair and decisive for
    that one, which is exactly what a global ranking cannot express.

    So all informative markers contribute, weighted by how far apart they push the labels of both
    cohorts, and each cohort-pair BLOCK of the distance matrix is divided by its own median
    distance. Dilution then rescales a whole block uniformly, which leaves the ordering inside
    the block untouched, and the division makes blocks comparable to each other. A side effect
    worth stating plainly: this removes the cohort offset by construction, so `cohort_ari` lands
    near 0.00 and that guard stops being informative - it is kept as a check, not as evidence.
    """
    QT, nodes = S['QT'], S['nodes']
    Z, P, W = rescale(S) if Z is None else Z
    n, T = len(nodes), QT.shape[1]
    lo, hi = Z[:, :, I_LO], Z[:, :, I_HI]
    has = ~np.isnan(lo)
    LIN = lineage_score(S) if RA['lineage'] else np.ones(T, 'float32')
    PFA = S['POS']                                    # (n, T, 3) positive fraction per cut
    assert RA['distance'] == 'rms' or RA['position'] or RA['posfrac'], (
        'agreement path selected with no agreement channel switched on')

    SIM = np.full((n, n), np.nan, 'float32')
    C = np.full((n, n), np.nan, 'float32')
    EV = np.zeros((n, n), 'int32')
    CX = np.full((n, n), np.nan, 'float32')
    coex = S['COEX']

    for i in range(n):
        w = np.minimum(W[i], W)                                            # (n, T)
        shared = has[i] & has
        w = np.where(shared & (w >= W_MIN), w, 0.0)
        EV[i] = (w > 0).sum(1)                                             # the HONEST evidence
        # W_MIN still GATES (so EV stays an honest count of what was measured); the lineage
        # score only REWEIGHTS what got through.
        u = w * LIN[None, :] if RA['lineage'] else w

        inter = np.clip(np.minimum(hi[i], hi) - np.maximum(lo[i], lo), 0, None)
        wid = hi[i] - lo[i]
        with np.errstate(invalid='ignore', divide='ignore'):
            o = inter / np.where(wid > 1e-6, wid, np.nan)
        # a degenerate (zero-width) marker in i is a point: inside j or not
        pt = (lo[i] >= lo - 1e-6) & (lo[i] <= hi + 1e-6)
        o = np.where(np.isfinite(o), o, pt.astype('float32'))
        o = np.clip(np.nan_to_num(o, nan=0.0), 0.0, 1.0)

        sw = u.sum(1)
        with np.errstate(invalid='ignore', divide='ignore'):
            C[i] = np.exp((u * np.log(np.maximum(o, DELTA))).sum(1) / np.where(sw > 0, sw, np.nan))
            # ROUTE A. Agreement per marker, combined by a FLOORED GEOMETRIC MEAN so that one
            # decisive disagreement vetoes a merge - the file's own stated principle (see DELTA)
            # applied to the layer that needs it.
            #
            # WHAT v1 DID WRONG. Its weighted RMS divided by the weight of EVERY informative
            # marker. A B cell and a CD8 T cell agree on ~28 of 30 - both CD45+, keratin-,
            # CD31-, SMA- - and those 28 added nothing to the numerator while adding their full
            # weight to the denominator. The more two labels shared and agreed on, the closer
            # they looked, and the 2-4 markers that define a cell type were averaged away.
            #
            # AND WHY THE FIX IS THE POSITIVE FRACTION, NOT THE GEOMETRIC MEAN. On positions,
            # exp(-|dP|) inside a geometric mean is algebraically the weighted MEAN of |dP| -
            # the DELTA floor needs |dP| > 3 and rescaled positions live in [-1, 1], so nothing
            # ever vetoes. Positive fractions are bounded in [0, 1], so 1 - |dPF| genuinely
            # reaches the floor. That is the whole difference, and it is why the measured
            # position-only arms all score 2/9 while posfrac alone scores 7/9.
            #
            # SIM holds a DISTANCE here, as in v1, so the cohort-pair block-median
            # normalisation below and the final exp(-d) are unchanged.
            if RA['distance'] == 'rms':
                # v1, unchanged: weighted RMS distance between the two mean-rank positions.
                d2 = (P[i][None] - P) ** 2                                 # (n, T)
                d2 = np.where(u > 0, np.nan_to_num(d2, nan=0.0), 0.0)
                SIM[i] = np.sqrt((d2 * u).sum(1) / np.where(sw > 0, sw, np.nan))
            else:
                ag = np.ones((n, T), 'float32')
                if RA['position']:
                    ag = ag * np.exp(-np.abs(P[i][None] - P))
                if RA['posfrac']:
                    nq = PFA.shape[2]
                    for q in range(nq):
                        ag = ag * (1.0 - np.abs(PFA[i, :, q][None] - PFA[:, :, q])) ** (1.0 / nq)
                la = np.log(np.clip(np.nan_to_num(ag, nan=1.0), DELTA, 1.0))
                SIM[i] = -(u * la).sum(1) / np.where(sw > 0, sw, np.nan)

        # co-expression agreement: correlation between the two labels' marker-marker matrices,
        # restricted to the markers they actually share. Reported, and used only in an ablation.
        for j in range(n):
            if EV[i, j] == 0 or j == i:
                continue
            m = shared[j] & (np.minimum(W[i], W[j]) >= W_MIN)
            idx = np.flatnonzero(m)
            if len(idx) < 3:
                continue
            a = coex[i][np.ix_(idx, idx)][np.triu_indices(len(idx), 1)]
            b = coex[j][np.ix_(idx, idx)][np.triu_indices(len(idx), 1)]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() >= 3 and a[ok].std() > 1e-6 and b[ok].std() > 1e-6:
                CX[i, j] = np.corrcoef(a[ok], b[ok])[0, 1]

    # normalise each cohort-pair block by its own median distance, THEN map to a similarity
    coh = nodes.cohort.to_numpy()
    for a in np.unique(coh):
        for b in np.unique(coh):
            blk = np.ix_(coh == a, coh == b)
            d = SIM[blk]
            fin = np.isfinite(d) & (d > 0)
            if fin.any():
                SIM[blk] = d / np.median(d[fin])
    SIM = np.exp(-SIM)

    np.fill_diagonal(C, 1.0)
    np.fill_diagonal(SIM, 1.0)
    np.fill_diagonal(EV, 0)
    return SIM, C, EV, CX


# ============================================================== clustering
def distance(active, SIM, EV, k=K_EVID):
    """-log(similarity), with below-the-floor pairs pushed to FAR so they never merge directly.

    They may still end up together THROUGH a third label that is comparable to both, which is
    exactly the transitive bridging M2 asks for - average linkage does it as a side effect of
    how it merges, rather than as a separate closure step.
    """
    a = np.asarray(active)
    sub = np.ix_(a, a)
    D = -np.log(np.clip(SIM[sub], 1e-6, 1.0)).astype(float)
    D = (D + D.T) / 2.0
    D[EV[sub] < k] = FAR
    np.fill_diagonal(D, 0.0)
    return D


def cluster(active, SIM, EV, cut, k=K_EVID, coex_gate=None):
    """Average-linkage agglomerative clustering on the SYMMETRIC layer, cut at height `cut`.

    NOT Leiden, and that is a measured change from the plan. Leiden optimises modularity on a
    graph; here the similarity is DENSE over ~100 nodes, and modularity on a dense graph returns
    one giant community plus singletons - which is exactly what it did: one 54-label community
    mixing tumour, macrophages, T cells and B cells. Its resolution parameter could be raised,
    but then granularity is set by two interacting knobs (threshold AND resolution) and is no
    longer identifiable. Average linkage makes the single knob mean something exact - the cut is
    the largest average distance allowed inside a cluster - and measurably works better on the
    same inputs: 0.86 versus 0.65 ARI, measured 2026-08 against the now-retired hand mapping.
    The measurement stands as the reason for the choice; the reference it used is gone.

    Nesting stays a separate directed layer and is deliberately not shown to the clusterer - a
    subtype and its parent are not the same cluster.
    """
    D = distance(active, SIM, EV, k)
    if coex_gate is not None:
        g = coex_gate[np.ix_(np.asarray(active), np.asarray(active))]
        D[np.isfinite(g) & (g < 0)] = FAR
        np.fill_diagonal(D, 0.0)
    if len(D) < 2:
        return np.zeros(len(D), int)
    Lk = linkage(squareform(D, checks=False), method='average')
    return fcluster(Lk, cut, criterion='distance') - 1


def ari(a, b, w=None):
    """Adjusted Rand index, optionally weighted so each label counts by its CELLS."""
    a, b = np.asarray(a), np.asarray(b)
    w = np.ones(len(a)) if w is None else np.asarray(w, float)
    ca, cb = pd.factorize(a)[0], pd.factorize(b)[0]
    M = np.zeros((ca.max() + 1, cb.max() + 1))
    np.add.at(M, (ca, cb), w)
    c2 = lambda x: x * (x - 1.0) / 2.0
    idx, ai, bi, n = c2(M).sum(), c2(M.sum(1)).sum(), c2(M.sum(0)).sum(), w.sum()
    exp = ai * bi / c2(n)
    mx = 0.5 * (ai + bi)
    return float((idx - exp) / (mx - exp)) if mx > exp else 0.0


def cohort_driven(nodes, memb):
    """How much of the clustering is just 'which cohort is this'.

    Stability alone CANNOT catch this failure and the first build proved it: a clustering that
    is really the cohort partition is perfectly reproducible when a different cohort is dropped,
    and it scored 0.972. So this is a separate HARD GUARD, not a term in the objective. It also
    needs no labels - cohort id is metadata, not a cell type - so using it is not circular.
    """
    ar = ari(memb, nodes.cohort.to_numpy())
    spans = {c for c in np.unique(memb)
             if nodes.cohort[memb == c].nunique() >= 2}
    inx = np.isin(memb, list(spans)) if spans else np.zeros(len(memb), bool)
    w = nodes.n_cells.to_numpy()
    return ar, float(w[inx].sum() / w.sum())


def choose_cut(S, SIM, EV, k=K_EVID):
    """Pick the dendrogram cut by two declared GUARDS, not by an optimised score.

    Four label-free objectives were built and measured, and every one of them is biased:

        LOCO stability (ARI, full vs held-out)   flat at 0.91-0.99 across the whole range
        held-out-cohort transfer                 monotone toward COARSE - coarse labels are
                                                 trivially easier to transfer
        dendrogram merge-height gap              toward coarse, it always finds the root
        silhouette                               toward FINE, a singleton scores a perfect 1

    Every one of those biases points at an END of the range, which is the tell: each is really
    measuring "is this partition trivial" in a different direction. So instead of optimising a
    biased score, the two trivial ends are excluded by GUARDS that say what a usable shared label
    space is, and stability - the one measure with no directional bias, only low resolution - is
    used to choose inside what is left.

    THE THREE GUARDS, none of which reads a label of record:

      biggest_share <= MAX_CLUSTER_SHARE   no cluster may hold a quarter of all labels. This is
          the coarse-end collapse, and it is abrupt rather than gradual: one merge fuses two
          already-large groups and the biggest cluster goes 23 -> 39 -> 49 -> 66 -> 89 labels in
          four steps. A "cell type" holding a quarter of every label in the roster annotates
          nothing.
      cross_cohort_share >= MIN_CROSS_SHARE   the share of CELLS sitting in a cluster that holds
          labels from more than one cohort. This is the fine-end collapse, and it is the
          DELIVERABLE stated as a constraint: Stage 1b exists to produce a SHARED label space to
          train on, and a cluster holding one cohort's labels alone teaches nothing about
          cross-cohort generalisation. It is counted in CELLS, not labels, because cells are what
          gets trained on. The bar is not 100% precisely because cohort-exclusive clusters are
          wanted - they are the Stage 7 novel-class test material - so up to 5% of cells are
          allowed to sit outside the shared space.
      cohort_ari <= COHORT_ARI_MAX   the clustering must not simply be the cohort partition.

    Within the feasible window the quality surface is a broad PLATEAU - every cut from 0.30 to
    0.45 lands within 0.04 ARI of the best - so the exact landing point matters little. The full
    sweep is printed so a reader can confirm that rather than take it on trust.
    """
    nodes, rows = S['nodes'], []
    allidx = np.arange(len(nodes))
    n = len(nodes)
    for cut in CUT_GRID:
        full = cluster(allidx, SIM, EV, cut, k)
        aris = []
        for c in sorted(nodes.cohort.unique()):
            sub = np.flatnonzero((nodes.cohort != c).to_numpy())
            aris.append(ari(full[sub], cluster(sub, SIM, EV, cut, k),
                            nodes.n_cells.to_numpy()[sub]))
        car, share = cohort_driven(nodes, full)
        vc = pd.Series(full).value_counts()
        rows.append(dict(cut=float(cut), clusters=int(full.max() + 1),
                         singletons=int((vc == 1).sum()), biggest=int(vc.max()),
                         biggest_share=float(vc.max() / n), stability=float(np.mean(aris)),
                         cohort_ari=car, cross_cohort_share=share))
    df = pd.DataFrame(rows)
    df['usable'] = ((df.biggest_share <= MAX_CLUSTER_SHARE) &
                    (df.cross_cohort_share >= MIN_CROSS_SHARE) &
                    (df.cohort_ari <= COHORT_ARI_MAX))

    # --- SEED GUARDS (--seed-refine). Which guards may judge the GLOBAL CUT, and which may
    # only judge the FINAL label space.
    #
    # The 7-cohort run failed with NO usable cut: cross_cohort_share needs cut >= 0.850 and
    # biggest_share needs cut <= 0.700, and the two windows are disjoint. The cause is that
    # cohorts annotate at different depths - Danenberg splits epithelium into 11 labels where
    # every other cohort has one or two - so one global number cannot serve every branch.
    #
    # That is not news to this file. refine()'s own docstring says the global cut "lands at the
    # coarsest granularity that keeps the label space usable. That is right for the top level
    # and too coarse inside a branch ... no single global cut can separate the first pair
    # without shattering everything else."
    #
    # THE DEFECT IS WHERE THE GUARD IS APPLIED, NOT WHAT IT SAYS. biggest_share is a property of
    # the FINAL label space, and refine() runs AFTER the cut. Testing it on the seed tests an
    # object that is deliberately too coarse and that nothing downstream ever uses.
    #
    # The other two guards genuinely belong on the seed, because refine() PRESERVES them by
    # construction: a split is kept only if "both sides span at least 2 cohorts", so refining can
    # never manufacture a single-cohort cluster and can only raise cross_cohort_share. cohort_ari
    # is protected by the same condition.
    #
    # So: seed on the guards refine() preserves, and judge size AFTER refinement.
    # Declared and dated rather than silently swapped, in the same style as D-28/D-29/D-35.
    # Opt-in, so a run WITHOUT --seed-refine reproduces the shipped 25-cluster result exactly.
    df['seed_ok'] = ((df.cross_cohort_share >= MIN_CROSS_SHARE) &
                     (df.cohort_ari <= COHORT_ARI_MAX))
    return df


def guard_violation(sweep):
    """How far each cut is from USABLE: the largest RELATIVE shortfall over the three hard guards.

    0 exactly when the cut is usable. Relative (shortfall / threshold) so the guards are on one
    scale - 1 point of cross-cohort share and 1 point of biggest-cluster share are not the same
    size of miss. Added 2026-09-11 for the fold-local spaces (panel/gate1b_fold_v2_expect.csv):
    the NEAREST-FEASIBLE fallback, used only when a fold has no usable cut, picks the cut with the
    smallest value here. Chosen by the user after the first fold run showed 3 of 7 LOCO folds
    with an empty window - a post-hoc rule, labelled as one.
    """
    return pd.concat([
        ((MIN_CROSS_SHARE - sweep.cross_cohort_share) / MIN_CROSS_SHARE).clip(lower=0),
        ((sweep.biggest_share - MAX_CLUSTER_SHARE) / MAX_CLUSTER_SHARE).clip(lower=0),
        ((sweep.cohort_ari - COHORT_ARI_MAX) / COHORT_ARI_MAX).clip(lower=0)], axis=1).max(axis=1)


def post_refine_share(memb):
    """biggest_share measured on the FINAL membership - the object the guard is really about."""
    vc = pd.Series(memb).value_counts()
    return float(vc.max() / len(memb)), int(vc.max())


def cohort_driven_note():
    return (f'cohort_ari is computed UNWEIGHTED, per label. The cell-weighted version was tried '
            f'first and is misleading here: cohort sizes run from 117k cells (Phillips) to 2.1M '
            f'(Sorin), so a single cluster holding `Sorin|Cancer` alone carries 930k cells and '
            f'drags the statistic to 0.5 while the label-level structure is fine (0.10).')


def loco_memberships(S, SIM, EV, tau, k=K_EVID):
    nodes = S['nodes']
    out = {}
    for c in sorted(nodes.cohort.unique()):
        sub = np.flatnonzero((nodes.cohort != c).to_numpy())
        out[c] = dict(zip(sub, cluster(sub, SIM, EV, tau, k)))
    return out


def refine(S, memb, SIM, EV, k=K_EVID):
    """M3 - granularity chosen PER BRANCH by stability, not by the global cut.

    The global cut is chosen by guards that are necessarily conservative, so it lands at the
    coarsest granularity that keeps the label space usable. That is right for the top level and
    too coarse inside a branch: CD4 T and CD8 T differ on 2 of the ~30 informative markers their
    cohorts share, so their distance is 0.56 block-medians while tumour-versus-T-cell is 1.23.
    The ORDERING is correct - CD4 T really is more like CD8 T than like a tumour cell - but no
    single global cut can separate the first pair without shattering everything else. That is
    exactly the argument M3 makes against a global cut.

    So each cluster is offered a binary split, recursively, and a split is KEPT only if it
    reproduces when a cohort is held out. Three conditions, all required:

      * both sides keep at least MIN_SPLIT_LABELS labels, so a split cannot peel off one label;
      * both sides span at least 2 cohorts, so a "split" can never be a cohort boundary in
        disguise - this is the cohort guard applied per branch instead of globally;
      * mean leave-one-cohort-out ARI of the split >= SPLIT_SUPPORT, so it must be reproducible
        rather than an artefact of one cohort's panel.

    This replaces an earlier merge-BACK version that took connected components over unstable
    label pairs. That was wrong in a way worth recording: one unstable pair chained two otherwise
    healthy clusters into one, and it cost 0.09 agreement (19 clusters at 0.923 collapsing to 15
    at 0.834). Splitting downward with a per-split test cannot chain.
    """
    out = memb.copy()
    nxt = int(memb.max()) + 1
    log = []
    stack = [int(c) for c in np.unique(memb)]
    while stack:
        c = stack.pop()
        idx = np.flatnonzero(out == c)
        if len(idx) < 2 * MIN_SPLIT_LABELS:
            continue
        two = _bisect(idx, SIM, EV, k)
        if two is None:
            continue
        rec = _split_ok(S, idx, two, SIM, EV, k)
        log.append(rec)
        if not rec['kept']:
            continue
        out[idx[two == 1]] = nxt
        stack += [c, nxt]
        nxt += 1
    return pd.factorize(out)[0], log


def _bisect(idx, SIM, EV, k):
    D = distance(idx, SIM, EV, k)
    if len(D) < 2 or not np.isfinite(D).any():
        return None
    return fcluster(linkage(squareform(D, checks=False), method='average'),
                    2, criterion='maxclust') - 1


def _split_ok(S, idx, two, SIM, EV, k):
    """Is this binary split real? Both sides substantial, both cross-cohort, and reproducible."""
    nodes = S['nodes']
    coh = nodes.cohort.to_numpy()[idx]
    sizes = [int((two == v).sum()) for v in (0, 1)]
    ncoh = [int(pd.Series(coh[two == v]).nunique()) for v in (0, 1)]
    rec = dict(labels=len(idx), left=sizes[0], right=sizes[1],
               left_cohorts=ncoh[0], right_cohorts=ncoh[1], stability=float('nan'), kept=False,
               reason='')
    if min(sizes) < MIN_SPLIT_LABELS:
        rec['reason'] = f'a side would hold < {MIN_SPLIT_LABELS} labels'
        return rec
    if min(ncoh) < 2:
        rec['reason'] = 'a side would be one cohort only - that is a cohort boundary, not a type'
        return rec
    # STRADDLE (--straddle, 2026-09-11). The guard above stops a split along ONE cohort's
    # boundary, but not along a boundary between GROUPS of cohorts. Measured with the midrange
    # centring: the B-cell cluster was split {CRC, ferguson, Sorin} / {UPMC, Danenberg} /
    # {Keren, Phillips} - every cohort on exactly one side, both sides >= 2 cohorts, LOCO ARI
    # 0.54-0.93. A panel difference between cohort groups reproduces under LOCO as well as
    # biology does, so LOCO cannot catch it. A real cell-type split shows up INSIDE cohorts:
    # CRC, UPMC and Keren each annotate both CD4 and CD8 T cells. So require MIN_STRADDLE
    # cohorts with labels on both sides.
    if RA['straddle']:
        both = len(set(coh[two == 0]) & set(coh[two == 1]))
        rec['straddle'] = both
        if both < MIN_STRADDLE:
            rec['reason'] = (f'only {both} cohort(s) hold labels on both sides - a boundary '
                             f'between cohort groups, not a type')
            return rec
    aris = []
    for c in np.unique(coh):
        keep = coh != c
        if keep.sum() < 4 or pd.Series(two[keep]).nunique() < 2:
            continue
        sub = _bisect(idx[keep], SIM, EV, k)
        if sub is None:
            continue
        aris.append(ari(sub, two[keep]))
    rec['stability'] = float(np.mean(aris)) if aris else 0.0
    if not aris:
        rec['reason'] = 'not testable - no cohort can be held out'
        return rec
    if rec['stability'] < SPLIT_SUPPORT:
        rec['reason'] = f'split does not reproduce (LOCO ARI {rec["stability"]:.2f})'
        return rec
    rec['kept'] = True
    rec['reason'] = 'reproduces across held-out cohorts'
    return rec


# ============================================================== M2b - nesting, SCC, closure
def nesting(S, memb, C, EV, SIM=None, cut=None, tau=NEST_MIN, k=K_EVID):
    """Directed layer BETWEEN clusters, then forced acyclic by contracting SCCs.

    A nesting edge additionally requires the two clusters to be RELATED. Containment on its own
    will call any wide cluster the parent of any narrow one, however unrelated they are, and
    without this the layer degenerates: one broad cluster came out as the "parent" of nearly
    every other, which is not a hierarchy, it is a statement that the cluster is wide. So a pair
    must also sit within NEST_RELATED times the chosen cut of each other - that is, they must be
    close enough that a slightly coarser granularity would have merged them, which is precisely
    what a parent-child pair means.
    """
    n_cl = memb.max() + 1
    members = [np.flatnonzero(memb == c) for c in range(n_cl)]
    dmax = NEST_RELATED * cut if cut else np.inf

    def blocks(mem):
        m2 = len(mem)
        Cc = np.full((m2, m2), np.nan)
        Ec = np.zeros((m2, m2), int)
        Rel = np.zeros((m2, m2), bool)
        for u in range(m2):
            for v in range(m2):
                if u == v:
                    continue
                sub = np.ix_(mem[u], mem[v])
                ok = (EV[sub] >= k) & np.isfinite(C[sub])
                Ec[u, v] = int(ok.sum())
                if ok.any():
                    Cc[u, v] = float(C[sub][ok].mean())
                if SIM is not None:
                    s = SIM[sub][np.isfinite(SIM[sub])]
                    Rel[u, v] = bool(len(s) and -np.log(max(float(s.mean()), 1e-9)) <= dmax)
                else:
                    Rel[u, v] = True
        return Cc, Ec, Rel

    def build(mem):
        Cc, Ec, Rel = blocks(mem)
        g = nx.DiGraph()
        g.add_nodes_from(range(len(mem)))
        for u in range(len(mem)):
            for v in range(len(mem)):
                if u == v or not np.isfinite(Cc[u, v]) or not Rel[u, v]:
                    continue
                # u ⊂ v : u's spread sits inside v's, and not the other way round
                if Cc[u, v] >= tau and (not np.isfinite(Cc[v, u])
                                        or Cc[u, v] - Cc[v, u] >= MARGIN):
                    g.add_edge(u, v, contain=float(Cc[u, v]), pairs=int(Ec[u, v]),
                               kind='measured')
        return g, Cc, Ec

    G, Cc, Ec = build(members)

    sccs = [s for s in nx.strongly_connected_components(G) if len(s) > 1]
    cyc_before = len(sccs)
    contract = {}
    for s in sccs:
        t = min(s)
        for x in s:
            contract[x] = t
    memb2 = np.array([contract.get(c, c) for c in memb])
    memb2 = pd.factorize(memb2)[0]

    # rebuild on the contracted clusters; a DAG is now guaranteed, and asserted
    mem2 = [np.flatnonzero(memb2 == c) for c in range(memb2.max() + 1)]
    D, Cc2, Ec2 = build(mem2)
    # contraction can only remove cycles between the merged nodes; anything left is re-contracted
    guard = 0
    while not nx.is_directed_acyclic_graph(D) and guard < 10:
        guard += 1
        for s in [s for s in nx.strongly_connected_components(D) if len(s) > 1]:
            t = min(s)
            memb2 = np.array([t if c in s else c for c in memb2])
        memb2 = pd.factorize(memb2)[0]
        mem2 = [np.flatnonzero(memb2 == c) for c in range(memb2.max() + 1)]
        D, Cc2, Ec2 = build(mem2)
    assert nx.is_directed_acyclic_graph(D), 'SCC contraction failed to produce a DAG'

    # transitive closure: FILL only where evidence was missing; never overwrite a measurement
    bridged, violations = [], []
    for u in list(D.nodes):
        for v in nx.descendants(D, u):
            if D.has_edge(u, v):
                continue
            if Ec2[u, v] < k:
                bridged.append(dict(child=int(u), parent=int(v), pairs=int(Ec2[u, v])))
            elif np.isfinite(Cc2[u, v]) and Cc2[u, v] < tau:
                violations.append(dict(child=int(u), parent=int(v),
                                       measured=float(Cc2[u, v]), tau=float(tau),
                                       margin=float(tau - Cc2[u, v]), pairs=int(Ec2[u, v])))
    for b in bridged:
        D.add_edge(b['child'], b['parent'], contain=float('nan'), pairs=b['pairs'],
                   kind='bridged')
    assert nx.is_directed_acyclic_graph(D), 'closure introduced a cycle'
    return memb2, D, dict(sccs=[sorted(int(x) for x in s) for s in sccs],
                          cycles_before=cyc_before, bridged=bridged,
                          violations=sorted(violations, key=lambda r: -r['margin']),
                          Cc=Cc2, Ec=Ec2)


# ============================================================== M4 - naming
def name_clusters(S, memb, P, ncoh):
    """Name from the top discriminative markers. Text never enters, not even here.

    A marker may only name a cluster if at least HALF that cluster's labels actually measured it
    and it is present in >= 3 cohorts. Without that rule a 23-label cluster gets named after a
    marker only one of its members carries, which reads as a claim about the whole cluster and
    is not one.
    """
    gene, nodes = S['gene'], S['nodes']
    n_cl = memb.max() + 1
    cen = np.full((n_cl, P.shape[1]), np.nan)
    covg = np.zeros((n_cl, P.shape[1]))
    for c in range(n_cl):
        sub = P[memb == c]
        cen[c] = np.nanmean(sub, 0)
        covg[c] = np.isfinite(sub).mean(0)
    mu, sd = np.nanmean(cen, 0), np.nanstd(cen, 0)
    z = (cen - mu) / np.where(sd > 1e-6, sd, np.nan)
    names = []
    for c in range(n_cl):
        v = z[c].copy()
        v[~np.isfinite(v) | (covg[c] < 0.5) | (ncoh < 3)] = 0.0
        up = [k for k in np.argsort(-v)[:3] if v[k] > 0.8]
        dn = [k for k in np.argsort(v)[:1] if v[k] < -1.2]
        parts = [f'{gene[k]}+' for k in up] + [f'{gene[k]}-' for k in dn]
        names.append(' '.join(parts) if parts else 'no discriminative shared marker')
    return names, cen, z


# ============================================================== gate helpers
def read_expect(name=None):
    """The declared cases, read from panel/gate1b_expect.csv unless --expect names another file.

    The 7-cohort rebuild adds five cases that cannot exist in the v1 file - four of them name
    Danenberg labels - so it runs against panel/gate1b_v2_expect.csv. The v1 file is left
    untouched so the shipped 25-cluster result stays reproducible and the two gates can be read
    side by side rather than one silently replacing the other.

    `name` overrides both (the fold-local builder names the shipped v4 file explicitly).
    """
    if name is None:
        name = 'gate1b_expect.csv'
        if '--expect' in sys.argv:
            name = sys.argv[sys.argv.index('--expect') + 1]
    p = os.path.join(PANEL, name)
    print(f"gate cases: {name}")
    df = pd.read_csv(p, keep_default_na=False)
    df['pairs'] = df.members.apply(
        lambda s: [tuple(x.split('|', 1)) for x in s.split(';')])
    df.attrs['file'] = name
    return df


def _case_pairs(r, cohorts):
    """A case's members, restricted to `cohorts` when given (fold-local scoring)."""
    return list(r.pairs) if cohorts is None else [p for p in r.pairs if p[0] in cohorts]


def score_cases(nodes, memb, names, ex, D, cohorts=None):
    """Gate 1b check 2 - every declared case scored on a partition. Returns one row per case.

    Factored out of report() (2026-09-11, plan F4) so the fold-local label spaces are scored by
    the SAME code as the shipped space - see panel/gate1b_fold_expect.csv check 4.

    `cohorts=None` is the shipped behaviour: a member missing from the graph makes the case n/a.
    With `cohorts` (a fold's training cohorts), members from other cohorts are dropped first, and
    a case left with fewer than 2 members is n/a ('held out') - a held-out cohort's label cannot
    fail or pass a case in a space it had no part in.

    BLOB-PROOF (benchmark_protocol.yaml declared_case_scoring): a `same` case merged only inside
    the largest cluster or a residual cluster (no positive marker in its name) is a FAIL.
    Scored as FAIL, not as a new result value, so it can never drop out of the blocking count.
    """
    kidx = {k: i for i, k in enumerate(zip(nodes.cohort, nodes.label))}
    _vc = pd.Series(memb).value_counts()
    blobs = {int(_vc.index[0])} | {k for k, nm in enumerate(names) if '+' not in nm}
    rows = []
    for _, r in ex.iterrows():
        base = dict(case=r.case, relation=r.relation, required=r.required,
                    waived=int(r.get('waived', 0)))
        pairs = _case_pairs(r, cohorts)
        if cohorts is not None and len(pairs) < 2:
            rows.append(dict(base, result='n/a',
                             detail=f'held out: {len(pairs)} member(s) in the training cohorts'))
            continue
        idx = [kidx.get(p) for p in pairs]
        miss = [f'{a}|{b}' for (a, b), i in zip(pairs, idx) if i is None]
        if miss:
            rows.append(dict(base, result='n/a', detail='not in graph: ' + ', '.join(miss)))
            continue
        cl = [int(memb[i]) for i in idx]
        if r.relation == 'same':
            one = len(set(cl)) == 1
            ok = one and cl[0] not in blobs
            det = f'clusters {sorted(set(cl))}' + (
                ' - merged only inside the largest/residual cluster (blob-proof rule)'
                if one and not ok else '')
        elif r.relation == 'different':
            ok = len(set(cl)) == len(cl)
            det = f'clusters {cl}'
        else:                                   # nested: child ⊂ parent
            cc, pp = cl[0], cl[1]
            if cc == pp:
                ok, det = None, f'merged into one cluster ({cc})'
            else:
                ok = D.has_edge(cc, pp)
                det = ('nesting edge found' if ok else
                       f'no edge {cc} -> {pp}; reverse={D.has_edge(pp, cc)}')
        rows.append(dict(base, result={True: 'PASS', False: 'FAIL', None: 'merged'}[ok],
                         detail=det))
    return pd.DataFrame(rows)


def unreliable_clusters(nodes, memb, names, ex, hc, cohorts=None):
    """Which clusters carry a WAIVED failure - Stage 7 leaves them out of the reliable-class score.

    Returns (flagged, exempt), `flagged` already net of `exempt`. Factored out of report()
    (2026-09-11, plan F4); `cohorts` restricts each case's members as in score_cases().

    A required case marked waived that FAILED flags the cluster of every one of its members.

    EXEMPT COHERENT CROSS-TISSUE CLUSTERS (2026-09-11). A waived case fails because SOME of its
    members went astray; the cluster the rest of them formed can be a perfectly good type.
    Flagging the cluster of EVERY member penalises the correct cluster along with the wrong one.
    Measured on the shipped run: tumour_epithelial's Keren / ferguson / Sorin / Phillips members
    form a coherent 3-tissue KRT+ cluster, and endothelium's CRC / UPMC / Phillips / Sorin
    members a 4-tissue PECAM1+ CD34+ one - both would have been excluded from the headline for
    a failure that happened elsewhere. Exempt = spans >= 3 tissues (the protocol v2 scope rule)
    AND is named by at least one POSITIVE marker, so a residual bucket ("no discriminative
    shared marker") or a negative-only name ("CD8A-") can never qualify.

    TIGHTENED 2026-09-11 (v4 config), because the rule above exempted two clusters that are not
    types: the LARGEST cluster (Danenberg epithelium + CRC stroma + granulocytes, which also
    fails check 2b) and a 3-label mix of Phillips mast cells, Danenberg FSP1+ fibroblasts and
    Sorin macrophages. Both passed ">= 3 tissues + a positive name". Now also required: the
    cluster holds the PLURALITY of that case's members (the cluster the case mostly formed,
    not one that received a stray member), and it is not the largest cluster (the blob-proof
    rule already refuses it credit). Stricter only - it can flag more, never less.
    """
    kidx = {k: i for i, k in enumerate(zip(nodes.cohort, nodes.label))}
    req = hc[hc.required == 1]
    waived = req[(req.result == 'FAIL') & (req.waived == 1)]
    flagged, plural = set(), set()
    for _, w in waived.iterrows():
        pairs = [pp for pp in _case_pairs(ex[ex.case == w.case].iloc[0], cohorts) if pp in kidx]
        flagged |= {int(memb[kidx[pp]]) for pp in pairs}
        cl = pd.Series([int(memb[kidx[pp]]) for pp in pairs]).value_counts()
        plural |= set(int(k) for k in cl.index[cl == cl.max()])
    largest = int(pd.Series(memb).value_counts().index[0])
    tiss = nodes.cohort.map(lambda c: SPECS[c]['tissue']).to_numpy()
    exempt = {k for k in flagged
              if len(set(tiss[memb == k])) >= 3 and '+' in names[k]
              and k in plural and k != largest}
    return flagged - exempt, exempt


def shared_markers(S, idx):
    """Informative markers every one of these labels' cohorts actually measured."""
    _, P, W = rescale(S)
    msk = np.ones(P.shape[1], bool)
    for i in idx:
        msk &= np.isfinite(P[i]) & (W[i] >= W_MIN)
    return sorted(str(S['gene'][k])[:34] for k in np.flatnonzero(msk))


def md_table(df, floatfmt='{:.3f}'):
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == 'f':
            d[c] = d[c].map(lambda v: '' if pd.isna(v) else floatfmt.format(v))
    head = '| ' + ' | '.join(map(str, d.columns)) + ' |'
    rule = '|' + '|'.join(['---'] * len(d.columns)) + '|'
    body = ['| ' + ' | '.join(map(str, r)) + ' |' for r in d.astype(str).values]
    return '\n'.join([head, rule] + body)


# ============================================================== main
def stale_reason(S, cohorts):
    """Why the cached signatures cannot be reused for THIS run. None means they can.

    THE CACHE HAD NO GUARD, AND IT SILENTLY DECIDED WHAT A RUN MEANT. On 2026-09-06 the
    7-cohort rebuild loaded work/label_signatures.npz from the previous 5+1-cohort pipeline,
    reported "signatures loaded from cache: 106 labels", clustered those, and printed
    GATE 1b: PASS. Danenberg's 32 phenotypes were never in it and neither were the corrected
    marker ids, so the gate re-scored the OLD signatures against NEW declared cases and passed.

    That is D-39's failure family exactly: what the code does depended on which files were on
    disk rather than on the data, with nothing in the log to signal it. `--resign` existed but
    an operator had to know to type it, and forgetting produced a plausible-looking PASS.

    Two fingerprints are compared, both already stored in the npz:
      cohort set   - a cohort added or removed changes the label space
      triple set   - a marker id repaired changes every signature that uses it
    """
    if 'POS' not in S:
        return "cache predates the Route A positive-fraction channel (no POS array)"
    cached_c = set(map(str, S['cohort']))
    if cached_c != set(cohorts):
        miss, extra = sorted(set(cohorts) - cached_c), sorted(cached_c - set(cohorts))
        return (f"cohort set differs - missing {miss}" if miss else '') +                (f" unexpected {extra}" if extra else '')
    cmark, _ = cohort_markers(cohorts)
    now_t = {t for m in cmark.values() for t in m}
    cached_t = set(map(str, S['triples']))
    if cached_t != now_t:
        add, gone = sorted(now_t - cached_t), sorted(cached_t - now_t)
        return (f"marker vocabulary differs - {len(add)} added, {len(gone)} removed "
                f"(e.g. {(add or gone)[:2]})")
    return None


def main(argv):
    only = set(a for a in argv if a.startswith('--'))
    cohorts = built()
    print(f'cohorts: {", ".join(cohorts)}')

    if '--resign' in only or not os.path.exists(SIG_CACHE):
        print('building M1 signatures (all cells, per-cohort ECDF = the GATE 1 winner):')
        S = build_signatures(cohorts)
    else:
        S = load_signatures()
        why = stale_reason(S, cohorts)
        if why:
            print(f'  CACHED SIGNATURES ARE STALE: {why}')
            print('  rebuilding rather than clustering the wrong thing (see stale_reason)')
            S = build_signatures(cohorts)
        else:
            print(f'signatures loaded from cache: {len(S["nodes"])} labels')

    nodes = S['nodes']
    key = list(zip(nodes.cohort, nodes.label))
    kidx = {k: i for i, k in enumerate(key)}
    W = nodes.n_cells.to_numpy()

    t0 = time.time()
    Z, P, Wm = rescale(S)
    SIM, C, EV, CX = containment(S, (Z, P, Wm))
    print(f'containment: {len(nodes)}x{len(nodes)} in {time.time()-t0:.1f}s')

    sweep = choose_cut(S, SIM, EV)
    seed_refine = '--seed-refine' in only
    col = 'seed_ok' if seed_refine else 'usable'
    ok = sweep[sweep[col]]
    if seed_refine:
        print(f'--seed-refine: seeding on the guards refine() preserves '
              f'(cross-cohort, cohort_ari); size is judged AFTER refinement')
    if len(ok) == 0:
        print(f'NO usable cut: no granularity satisfies the {col} guards at once. '
              'Stage 1b has failed.')
        ok = sweep
    if seed_refine and len(ok):
        # SEED = THE FINEST CUT THE PRESERVED GUARDS ALLOW, not the most stable one.
        #
        # First attempt used max stability, as the global rule does, and it ran to the coarsest
        # cut in the grid (1.150, 135 of 138 labels in one cluster). That is the known bias:
        # stability is flat at 0.91-0.99 across the range, so with the size guard no longer
        # bounding the seed from above there was nothing left to stop it. Recorded rather than
        # quietly replaced.
        #
        # The correct rule follows from what refine() can DO: it only ever SPLITS. So a seed that
        # is too coarse destroys structure refine() must then rediscover - and refine() is
        # deliberately conservative, requiring both sides to span 2 cohorts AND to reproduce
        # leave-one-cohort-out, so it will decline to rebuild much of it. A seed that is as fine
        # as the preserved guards allow hands refine() the most structure and asks it to do the
        # least invention. Splitting cannot lower cross_cohort_share (both sides must span 2
        # cohorts), so the guard still holds all the way down.
        tau = float(ok.cut.min())
    else:
        tau = float(ok.cut[ok.stability.idxmax()])  # most stable cut inside the feasible window
    row = sweep[sweep.cut == tau].iloc[0]
    print(f'cut = {tau:.3f}  ({int(row.clusters)} clusters, biggest {int(row.biggest)} labels '
          f'= {row.biggest_share:.0%}, cohort_ari {row.cohort_ari:.3f}, '
          f'stability {row.stability:.3f})')

    memb0 = cluster(np.arange(len(nodes)), SIM, EV, tau)
    loco = loco_memberships(S, SIM, EV, tau)
    memb1, splits = refine(S, memb0, SIM, EV)
    _share, _big = post_refine_share(memb1)
    print(f'post-refinement size guard: biggest cluster {_big} labels = {_share:.1%} '
          f'(cap {MAX_CLUSTER_SHARE:.0%}) -> {"PASS" if _share <= MAX_CLUSTER_SHARE else "FAIL"}')
    memb, D, gi = nesting(S, memb1, C, EV, SIM=SIM, cut=tau)
    ncoh = np.array([nodes.cohort[np.isfinite(P[:, t])].nunique() for t in range(P.shape[1])])
    names, cen, zc = name_clusters(S, memb, P, ncoh)
    print(f'clusters: {memb0.max()+1} at the global cut -> {memb1.max()+1} after per-branch '
          f'refinement -> {memb.max()+1} after SCC contraction')

    out = nodes.copy()
    out['cluster'] = memb
    out['cluster_name'] = [names[c] for c in memb]
    report(S, out, SIM, C, EV, CX, D, gi, sweep, tau, splits,
           names, cen, zc, loco, only, Z)


def report(S, out, SIM, C, EV, CX, D, gi, sweep, tau, splits,
           names, cen, zc, loco, only, Z):
    nodes = S['nodes']
    key = list(zip(nodes.cohort, nodes.label))
    kidx = {k: i for i, k in enumerate(key)}
    memb = out.cluster.to_numpy()
    W = nodes.n_cells.to_numpy()
    n_cl = memb.max() + 1
    L = []
    A = L.append

    A('# Stage 1b - automatic label alignment (GATE 1b)\n')
    A(f'_{len(nodes)} native labels from {nodes.cohort.nunique()} cohorts, aligned with no '
      f'ontology, no hand-written dictionary and no text._\n')

    # ---------------------------------------------------------------- setup
    A('## What was measured\n')
    A(f'- Value transform: **`u_coh`**, the per-cohort ECDF - the arm that won Gate 1.')
    A(f'- Signature: **{len(QLEV)} quantiles** per (label, marker) + prevalence + '
      f'the within-label co-expression matrix.')
    if RA['distance'] == 'rms':
        A(f'- Merging distance: **weighted RMS** of the rescaled mean-rank positions over the '
          f'informative shared markers (v1). It has NO veto - one disagreeing marker is averaged '
          f'against the agreeing ones. The veto variant was measured and did not beat it.')
    else:
        A(f'- Markers combined by a **floored geometric mean** (floor {DELTA}), not an average, '
          f'so one decisive disagreement can veto a merge.')
    A(f'- Rescale centring: **{RA["centre"]}** of each (cohort, marker)\'s label positions; '
      f'refine() straddle guard: **{"on" if RA["straddle"] else "off"}** '
      f'(>= {MIN_STRADDLE} cohorts on both sides of a kept split).')
    A(f'- Evidence floor **k = {K_EVID}** informative shared markers; a marker is informative '
      f'when its between-label range reaches **{W_MIN}** in *both* cohorts.')
    A(f'- Merge threshold **tau = {tau:.3f}**, chosen by LOCO stability under declared guards, '
      f'never by agreement with any reference.\n')
    if len(S['dropped']):
        d = S['dropped'].sort_values('cells', ascending=False)
        A(f'**{len(d)} native labels excluded before any alignment** '
          f'({d.cells.sum():,} cells) - quantiles on a handful of cells are noise, and an '
          f'unlabelled group is not a cell type:\n')
        A(md_table(d))
        A('')

    # ---------------------------------------------------------------- tau choice
    A('## Choosing the threshold without looking at the answer\n')
    A('`tau` is never tuned against a reference. It is chosen by '
      '**leave-one-cohort-out stability**: mean ARI between the full clustering and each '
      'held-out re-derivation. ARI is chance-corrected, so both degenerate answers punish '
      'themselves - one giant cluster and all-singletons each score about 0.\n')
    A(f'Stability alone is **not enough**, and the first build of this stage proved it: a '
      f'clustering that is really the cohort partition is perfectly reproducible when a '
      f'*different* cohort is dropped, and it scored 0.972 while grouping tumour, macrophages, '
      f'T cells and B cells together. So `cohort_ari` - how much the clustering is just "which '
      f'dataset is this" - is a **hard feasibility constraint** (must be <= '
      f'{COHORT_ARI_MAX}), not a term in the objective. Cohort id is metadata, not a cell type, '
      f'so using it leaks nothing.\n')
    sweep2 = sweep.copy()
    A(md_table(sweep2))
    A('')
    srow = sweep[sweep.cut == tau].iloc[0]
    A(f'Chosen **tau = {tau:.3f}** - the most stable threshold among those passing the cohort '
      f'guard (stability {srow.stability:.3f}, cohort_ari {srow.cohort_ari:.3f}, '
      f'{srow.cross_cohort_share:.0%} of labels in a cluster that spans >= 2 cohorts). '
      f'{int((~sweep.usable).sum())} of {len(sweep)} thresholds were rejected by the guard.\n')

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    ax[0].plot(sweep2.cut, sweep2.stability, 'o-', label='LOCO stability (reported, not optimised)')
    ax[0].axvline(tau, color='crimson', lw=1)
    ax[0].set_xlabel('cut height'); ax[0].legend(fontsize=7); ax[0].set_title('threshold choice')
    ax[1].plot(sweep2.cut, sweep2.cohort_ari, 'o-', color='darkorange')
    ax[1].axhline(COHORT_ARI_MAX, color='crimson', ls='--', lw=1)
    ax[1].axvline(tau, color='crimson', lw=1)
    ax[1].set_xlabel('cut height'); ax[1].set_title('cohort guard (lower is better)')
    ax[2].plot(sweep2.cut, sweep2.clusters, 'o-')
    ax[2].axvline(tau, color='crimson', lw=1)
    ax[2].set_xlabel('cut height'); ax[2].set_ylabel('clusters'); ax[2].set_title('granularity')
    fig.tight_layout()
    figp = os.path.join(FIGURES, 'label_space_tau.png')
    fig.savefig(figp, dpi=120); plt.close(fig)
    A(f'![threshold choice](figures/label_space_tau.png)\n')

    # ---------------------------------------------------------------- check 2
    A('## Check 2 - the hard cases, declared in advance\n')
    ex = read_expect()
    A(f'From `celltype_transfer/declared/{ex.attrs["file"]}`. The cases were written **before** the run; '
      f'any waiver in it was written **after** a run and says so.\n')
    hc = score_cases(nodes, memb, names, ex, D)
    A(md_table(hc))
    req = hc[hc.required == 1]
    blocking = req[(req.result == 'FAIL') & (req.waived == 0)]
    n_req_fail = len(blocking)
    waived = req[(req.result == 'FAIL') & (req.waived == 1)]
    A(f'\n**Required cases: {int((req.result == "PASS").sum())}/{len(req)} pass'
      + (f', {len(waived)} waived on measured evidence, {n_req_fail} blocking.**\n'
         if len(waived) else f', {n_req_fail} blocking.**\n'))
    for _, w in waived.iterrows():
        wr = ex[ex.case == w.case].iloc[0]
        A(f'> **`{w.case}` is a WAIVED FAILURE, not a pass.** It was declared required before the '
          f'run and it failed. {wr.waiver_reason}\n')
    flagged, exempt = unreliable_clusters(nodes, memb, names, ex, hc)
    if exempt:
        A(f'**{len(exempt)} clusters exempted from the unreliable flag** despite holding a waived '
          f'case member, because each is a coherent cross-tissue type (>= 3 tissues, named by a '
          f'positive marker): {sorted(exempt)}. The waived failure happened in the OTHER clusters '
          f'those cases touch.\n')
    # a waived failure must not become invisible downstream
    out['unreliable'] = out.cluster.isin(flagged).astype(int)
    out['unreliable_reason'] = np.where(out.cluster.isin(flagged),
                                        'Gate 1b waived failure - see panel/gate1b_expect.csv', '')
    if flagged:
        A(f'**{len(flagged)} clusters carry a waived failure and are flagged `unreliable` in '
          f'`work/label_map.csv`** (clusters {sorted(flagged)}, '
          f'{int(out.unreliable.sum())} labels, {int(out[out.unreliable == 1].n_cells.sum()):,} '
          f'cells). Stage 7 must exclude them from the headline score or report them separately - '
          f'they are not evidence of anything either way.\n')

    # every failing case is diagnosed here rather than left for the reader to guess at
    bad_cases = hc[(hc.result == 'FAIL')].case.tolist()
    for _, r in ex[ex.case.isin(bad_cases) & (ex.relation == 'same')].iterrows():
        idx = [kidx.get(p) for p in r.pairs]
        if any(i is None for i in idx):
            continue
        A(f'**Why `{r.case}` failed.** Pairwise similarity between its members, against the '
          f'chosen cut of {np.exp(-tau):.3f}:\n')
        pr = []
        for (a1, b1), i in zip(r.pairs, idx):
            for (a2, b2), j in zip(r.pairs, idx):
                if i < j:
                    pr.append(dict(a=f'{a1}|{b1}', b=f'{a2}|{b2}',
                                   similarity=float(SIM[i, j]),
                                   informative_markers=int(EV[i, j]),
                                   would_merge=bool(SIM[i, j] >= np.exp(-tau))))
        A(md_table(pd.DataFrame(pr)))
        shared = shared_markers(S, [kidx[p] for p in r.pairs])
        A(f'\nInformative markers shared by **all** of them ({len(shared)}): '
          f'{", ".join(shared) if shared else "none"}\n')

    # ---------------------------------------------------------------- check 2b
    # benchmark_protocol.yaml layer_1 biological_exclusion. Independent of the declared cases:
    # it needs no case list, only textbook lineage exclusions. REPORT-ONLY - it was not
    # pre-registered as a pass/fail threshold, so it does not enter the verdict.
    A('## Check 2b - biological exclusion (report-only, independent of the declared cases)\n')
    exc = pd.read_csv(os.path.join(PANEL, 'gate1b_exclusions.csv'))
    tix = {str(t): k for k, t in enumerate(S['triples'])}
    pos50 = S['POS'][:, :, int(np.flatnonzero(POSCUT == 0.50)[0])]
    vrows, viol = [], set()
    for e in exc.itertuples():
        if e.marker_a not in tix or e.marker_b not in tix:
            continue
        a, b = pos50[:, tix[e.marker_a]], pos50[:, tix[e.marker_b]]
        ah, bh = np.nan_to_num(a, nan=0) >= EXCL_HIGH, np.nan_to_num(b, nan=0) >= EXCL_HIGH
        ao, bo = ah & ~bh & np.isfinite(b), bh & ~ah & np.isfinite(a)
        for k in range(n_cl):
            m = memb == k
            if (ao & m).any() and (bo & m).any():
                viol.add(k)
                vrows.append(dict(
                    pair=e.pair, cluster=k,
                    side_a=f'{e.lineage_a}: ' + '; '.join(
                        f'{key[i][0]}\\|{key[i][1]}' for i in np.flatnonzero(ao & m)),
                    side_b=f'{e.lineage_b}: ' + '; '.join(
                        f'{key[i][0]}\\|{key[i][1]}' for i in np.flatnonzero(bo & m))))
    vm = out.cluster.isin(viol).to_numpy()
    A(f'Pairs from `celltype_transfer/declared/gate1b_exclusions.csv`. A label is **high** on a marker when '
      f'>= {EXCL_HIGH:.0%} of its cells sit above its own cohort\'s median; a cluster violates a '
      f'pair when it holds a label high only on marker A and one high only on marker B.\n')
    A(f'**{len(viol)} clusters violate at least one exclusion** ({int(vm.sum())} labels, '
      f'{W[vm].sum():,} cells = {W[vm].sum() / W.sum():.1%}).\n')
    if vrows:
        A(md_table(pd.DataFrame(vrows).fillna('')))
        A('\nSome single-label violations are signal bleeding between touching cells rather '
          'than mixing - e.g. intraepithelial T cells reading keratin-high. They are listed, '
          'not removed.\n')

    # ---------------------------------------------------------------- check 3
    A('## Check 3 - evidence coverage and the floor sweep\n')
    iu = np.triu_indices(len(nodes), 1)
    ev = EV[iu]
    direct = (ev >= K_EVID)
    A(f'| | pairs | share |\n|---|---|---|')
    A(f'| directly comparable (evidence >= {K_EVID}) | {direct.sum():,} | '
      f'{direct.mean()*100:.1f}% |')
    A(f'| below the floor | {(~direct).sum():,} | {(~direct).mean()*100:.1f}% |')
    A(f'| never comparable (0 informative shared markers) | {(ev == 0).sum():,} | '
      f'{(ev == 0).mean()*100:.1f}% |\n')
    A(f'Bridged by transitive closure: **{len(gi["bridged"])}** cluster pairs.\n')
    sw = []
    for k in (4, 6, 8, 12, 16):
        m = cluster(np.arange(len(nodes)), SIM, EV, tau, k=k)
        sw.append(dict(k=k, pairs_direct=int((ev >= k).sum()),
                       share_direct=float((ev >= k).mean()),
                       clusters=int(m.max() + 1)))
    A(md_table(pd.DataFrame(sw)))
    A('')
    cm = pd.DataFrame(0, index=sorted(nodes.cohort.unique()),
                      columns=sorted(nodes.cohort.unique()))
    for a in cm.index:
        for b in cm.columns:
            ia = np.flatnonzero((nodes.cohort == a).to_numpy())
            ib = np.flatnonzero((nodes.cohort == b).to_numpy())
            cm.loc[a, b] = int(np.median(EV[np.ix_(ia, ib)])) if len(ia) and len(ib) else 0
    A('Median informative shared markers per cohort pair:\n')
    A(md_table(cm.reset_index().rename(columns={'index': 'cohort'})))
    A('')

    # ---------------------------------------------------------------- check 4
    A('## Check 4 - split stability (M3)\n')
    A('The global cut sets the top level. Inside each branch, granularity is set separately: '
      'every cluster is offered a binary split, and the split is kept only if both sides hold '
      f'at least {MIN_SPLIT_LABELS} labels, both sides span **two or more cohorts** (so a split '
      f'can never be a cohort boundary in disguise) and it reproduces with a cohort held out '
      f'(LOCO ARI >= {SPLIT_SUPPORT}). Every decision, accepted or rejected, is listed.\n')
    sp = pd.DataFrame(splits)
    if len(sp):
        A(f'**{int(sp.kept.sum())} of {len(sp)} candidate splits accepted:**\n')
        A(md_table(sp[['labels', 'left', 'right', 'left_cohorts', 'right_cohorts',
                       'stability', 'kept', 'reason']]))
    else:
        A('No cluster was large enough to offer a split.')
    A('')
    # ---------------------------------------------------------------- check 5
    A('## Check 5 - nesting (M2)\n')
    ed = [dict(child=names[u][:38] or str(u), parent=names[v][:38] or str(v),
               contain=d.get('contain'), label_pairs=d['pairs'], kind=d['kind'])
          for u, v, d in D.edges(data=True)]
    if ed:
        A(f'**{len(ed)} parent-child edges** (child ⊂ parent), all between clusters:\n')
        A(md_table(pd.DataFrame(ed)))
    else:
        A('**No nesting edges survived.** Every relation the data supports was symmetric, so '
          'the hierarchy is flat at this granularity. Reported, not patched.')
    A('')

    # ---------------------------------------------------------------- check 6
    A('## Check 6 - marker coherence\n')
    coh = coherence(S, memb, names, Z)
    A('Within-cluster spread of the label signatures versus the spread between cluster '
      'centres. A cluster whose members disagree more than the clusters differ is not a cell '
      'type.\n')
    A(md_table(coh))
    nbad = int((~coh.coherent).sum())
    A(f'\n**{len(coh)-nbad}/{len(coh)} clusters coherent.**\n')

    # ---------------------------------------------------------------- check 7
    A('## Check 7 - acyclicity and transitivity (M2b)\n')
    A(f'| | |\n|---|---|')
    A(f'| `nx.is_directed_acyclic_graph` after SCC contraction | **{nx.is_directed_acyclic_graph(D)}** |')
    A(f'| cycles (SCCs > 1 node) found before contraction | {gi["cycles_before"]} |')
    A(f'| transitivity violations (measured, contradicting closure) | {len(gi["violations"])} |')
    A(f'| pairs bridged by closure (evidence was missing) | {len(gi["bridged"])} |\n')
    if gi['sccs']:
        A('Contracted strongly connected components (mutually containing labels **are** one '
          'type, so merging them is the correct reading):\n')
        for s in gi['sccs']:
            A(f'- {s}')
        A('')
    if gi['violations']:
        v = pd.DataFrame(gi['violations']).head(10)
        v['child'] = v.child.map(lambda c: names[c][:34])
        v['parent'] = v.parent.map(lambda c: names[c][:34])
        A(f'Worst transitivity violations - **logged, never overwritten**:\n')
        A(md_table(v))
        rate = len(gi['violations']) / max(1, len(gi['violations']) + D.number_of_edges())
        A(f'\nViolation rate {rate*100:.1f}%. Above ~10% the containment measure is unreliable '
          f'and the evidence floor should rise.\n')
    else:
        A('**No transitivity violation.** No measured pair contradicts the closure.\n')

    # ---------------------------------------------------------------- final clusters
    A('## The cluster list\n')
    cl = cluster_table(nodes, memb, names)
    A(md_table(cl))
    A('')
    A(f'**Core clusters** (>= 2 cohorts): {int((cl.cohorts >= 2).sum())} · '
      f'**cohort-exclusive**: {int((cl.cohorts == 1).sum())} - the latter become the Stage 7 '
      f'novel-class test material.\n')

    # ---------------------------------------------------------------- verdict
    car, xshare = cohort_driven(nodes, memb)
    ok2 = n_req_fail == 0
    ok7 = nx.is_directed_acyclic_graph(D)
    ok6 = nbad == 0
    ok0 = car <= COHORT_ARI_MAX
    # THREE STATES, NEVER TWO (protocol v2, gate_failure_policy): "a waived failure is reported
    # as a failure, not as a pass." The old two-state verdict printed PASS whenever every
    # required case was either passed OR waived - so a run waiving 7 of 9 required cases printed
    # "GATE 1b: PASS", which is exactly the sentence a reviewer would quote back. A waived run is
    # shippable, and it is labelled as what it is.
    structural = ok0 and ok7 and ok6
    if structural and ok2 and len(waived) == 0:
        verdict = 'PASS'
    elif structural and ok2:
        verdict = (f'SHIPPED UNDER WAIVER - NOT A PASS ({len(waived)} of {len(req)} required '
                   f'cases waived in writing; {int((req.result == "PASS").sum())} passed outright)')
    else:
        verdict = 'FAIL'
    A('## GATE 1b verdict\n')
    A(f'| check | result | pass |\n|---|---|---|')
    A(f'| 0 cohort guard: clusters are not the cohort partition | cohort_ari {car:.3f} '
      f'(cap {COHORT_ARI_MAX}), {xshare:.0%} cross-cohort | {"yes" if ok0 else "NO"} |')
    A(f'| 2 required hard cases | {int((req.result=="PASS").sum())}/{len(req)}'
      + (f' + {len(waived)} waived' if len(waived) else '')
      + f' | {("WAIVED" if len(waived) else "yes") if ok2 else "NO"} |')
    A(f'| 3 evidence coverage reported | {direct.mean()*100:.1f}% direct | yes |')
    A(f'| 4 per-branch splits tested | {int(pd.DataFrame(splits).kept.sum()) if splits else 0}'
      f'/{len(splits)} accepted | yes |')
    A(f'| 5 nesting edges | {len(ed)} | yes |')
    A(f'| 6 every cluster coherent | {len(coh)-nbad}/{len(coh)} | {"yes" if ok6 else "NO"} |')
    A(f'| 7 graph is a DAG | {ok7} | {"yes" if ok7 else "NO"} |')
    A(f'\n# GATE 1b: {verdict}\n')

    with open(os.path.join(REPORTS, 'build_label_space.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    out.to_csv(os.path.join(WORK, 'label_map.csv'), index=False)
    np.save(os.path.join(WORK, 'prototypes.npy'), cen)
    with open(os.path.join(WORK, 'label_graph.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(tau=tau, k=K_EVID, delta=DELTA, margin=MARGIN,
                       clusters={int(c): names[c] for c in range(n_cl)},
                       edges=[dict(child=int(u), parent=int(v), **{k: (None if (
                           isinstance(d[k], float) and not np.isfinite(d[k])) else d[k])
                           for k in d}) for u, v, d in D.edges(data=True)],
                       sccs=gi['sccs'], bridged=gi['bridged'],
                       violations=gi['violations'],
                       triples=S['triples'].tolist()), f, indent=1)
    print(f'\nGATE 1b: {verdict}   -> reports/build_label_space.md')


def coherence(S, memb, names, Z):
    med = Z[:, :, I_MED]
    ok = np.isfinite(med).all(0)                       # markers every label has: comparable
    X = med[:, ok]
    n_cl = memb.max() + 1
    cen = np.vstack([X[memb == c].mean(0) for c in range(n_cl)])
    between = float(np.mean(((cen - cen.mean(0)) ** 2).sum(1)))
    rows = []
    for c in range(n_cl):
        m = memb == c
        within = float(np.mean(((X[m] - cen[c]) ** 2).sum(1))) if m.sum() > 1 else 0.0
        rows.append(dict(cluster=c, name=names[c][:40], labels=int(m.sum()),
                         within=within, between=between, coherent=within < between))
    return pd.DataFrame(rows)


def cluster_table(nodes, memb, names):
    rows = []
    for c in range(memb.max() + 1):
        m = memb == c
        sub = nodes[m]
        rows.append(dict(cluster=c, name=names[c][:44], labels=int(m.sum()),
                         cohorts=int(sub.cohort.nunique()), cells=int(sub.n_cells.sum()),
                         members='; '.join(f'{a}|{b}' for a, b in
                                           zip(sub.cohort, sub.label))[:150]))
    return pd.DataFrame(rows).sort_values('cells', ascending=False)


if __name__ == '__main__':
    main(sys.argv[1:])

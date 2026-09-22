"""
STAGE 1 - continuous value harmonisation.  Produces GATE 1.

The problem. CRC ships raw fluorescence, UPMC ships arcsinh, Keren ships z-scores, Sorin ships
uint8. Those are four different CURVE SHAPES into one encoder. A LayerNorm only shifts and
scales - it cannot undo a nonlinearity. An empirical-CDF (rank) transform can: it is monotone,
continuous, has no bins, and puts every cohort on the same footing in one step.

What is NOT decided in advance is the GROUP the rank is taken inside. That choice has a real
failure mode. On a slide that is 90% tumour, ranking inside the slide forces half those cells
below the median on keratin - inventing a negative population that does not exist. The old
pipeline hit exactly this and patched it with a GMM fallback returning a flat 0.85/0.15, which
is the cliff this whole rebuild exists to remove. So the group is MEASURED, not assumed.

Six arms, built as a ladder so that each comparison isolates exactly one decision:

    V1       u_img                       ECDF inside (image, marker) - the old behaviour
    V3       u_coh                       ECDF inside (cohort, marker), no correction
    V2a      FiLM_2(u_coh)               learned slide correction, 4 parameters per marker
    V2b      FiLM_mlp(u_coh)             learned slide correction, full capacity
    V1+V3    u_img, u_coh                both groupings, nothing learned
    V2a+V1   FiLM_2(u_coh), u_img        does FiLM still pay once the image rank is present?

    V1  vs V3       is per-image grouping harmful on its own?
    V3  vs V2a/V2b  is a learned slide correction worth anything at all?
    V2a vs V2b      is the EXTRA CAPACITY in that correction worth anything?
    V3  vs V1+V3    does the image rank ADD to the cohort rank, rather than replace it?
    V1+V3 vs V2a+V1 does FiLM beat simply handing over the image rank?

WHAT THE PLAN GOT WRONG HERE, corrected after measuring. The plan called for a second channel
holding "the value relative to a cohort-level reference" (tanh of a robust z-score) so that
ranking would not destroy prevalence. Built and measured, it saturates - 5.5-17.5% of cells at
|lvl| > 0.99, worst on Sorin, which arrives uint8 so most markers have median 0 and a
near-zero IQR. And it is redundant by construction: any per-cell function of the raw value
computed from cohort statistics is a monotone transform of that value, so it carries what
`u_coh` already carries (measured correlation 0.89-0.97). It would also have broken the
bake-off, because an arm holding {u_img, lvl} strictly contains an arm holding {u_coh, lvl} -
V1 could not have lost. The second channel is therefore the OTHER GROUPING, which is genuinely
independent information, and prevalence is kept by `u_coh` itself.

    python harmonise_values.py              # build work/values/*.parquet + slide statistics
    python harmonise_values.py --bakeoff    # the five GATE 1 checks + reports/harmonise_values.md
"""
import os, sys, json, time
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # runs from any directory
import config
import panel_utils
from config import (SPECS, WORK, VALUES, PANEL, REPORTS, FIGURES,
                    raw_table, value_table, SEED)
from models.marker_encoder import MaskedMarkerProbe, EPS_FILM

META = ['cell_id', 'cohort', 'image_id', 'patient_id', 'native_label']

# channels are ECDF sources; `film` names the learned slide correction applied to channel 1
ARMS = {
    'V1':     dict(ch=['u_img'],          film=None),
    'V3':     dict(ch=['u_coh'],          film=None),
    'V2a':    dict(ch=['u_coh'],          film='scalar'),
    'V2b':    dict(ch=['u_coh'],          film='mlp'),
    'V1+V3':  dict(ch=['u_img', 'u_coh'], film=None),
    'V2a+V1': dict(ch=['u_coh', 'u_img'], film='scalar'),
}
N_SUB = 40_000            # cells per cohort for local development (plan's compute note)
EPOCHS = 60               # a ceiling; early stopping on held-out slides picks the real number
PATIENCE = 6
D_MODEL = 32
AUROC_MIN = 0.75          # a declared assertion must separate its label this well


# ----------------------------------------------------------------------------- shared panel
def registry():
    return pd.read_csv(os.path.join(WORK, 'marker_registry.csv'), keep_default_na=False)


def built():
    return [c for c in SPECS if os.path.exists(raw_table(c))]


def core_triples(cohorts):
    """Triples measured by EVERY built cohort.

    Gate 1 compares arms against each other, so the feature space must be identical in every
    fold - otherwise an arm could win on imputation rather than on normalisation. Restricting
    to the fully shared core removes that confound: no missing values, nothing filled in. It
    costs breadth, not fairness, and the wider panel returns at Stage 2 where the [ABSENT]
    token handles it properly.
    """
    r = registry()
    r = r[(r.kind != 'non_protein') & r.cohort.isin(cohorts)]
    n = r.groupby('triple').cohort.nunique()
    keep = sorted(n[n == len(cohorts)].index)
    gene = r.drop_duplicates('triple').set_index('triple').gene.to_dict()
    return keep, {t: gene[t] for t in keep}


def raw_cols_for(cohort, triples):
    """{triple: [raw_column, ...]}. A list longer than one is a DUPLICATE REAGENT - Danenberg
    ships two HER2 clones that resolve to the same gene. Averaged, by the policy declared in
    panel/gate0b_expect.csv. See panel_utils for why averaging and not 'keep the first'."""
    return panel_utils.cols_for(registry(), cohort, triples)


# ----------------------------------------------------------------------------- build
def build(cohort, triples, n_sub=N_SUB, rng=None):
    """One pass over a cohort: references from ALL its cells, table written for a subsample.

    The references (cohort ECDF, per-image ECDF, slide statistics) are computed on every cell,
    so they are honest. Only the transformed TABLE is subsampled, because the full matrix does
    not fit in local RAM - the Kaggle run sets n_sub=None and takes the identical code path.
    """
    rng = rng or np.random.default_rng(SEED)
    cols = raw_cols_for(cohort, triples)
    df = pd.read_parquet(raw_table(cohort),
                         columns=META + panel_utils.read_cols(cols, triples))
    X = panel_utils.matrix(df, cols, triples)
    img = df.image_id.to_numpy()

    # `method='average'` splits ties down the middle. That matters here: Sorin arrives uint8,
    # so ties are everywhere and a left- or right-rank would bias every quantised marker.
    u_img = X.groupby(img, sort=False).rank(pct=True, method='average').astype('float32')
    u_coh = X.rank(pct=True, method='average').astype('float32')

    # FiLM input: per (slide, marker) median and IQR of the cohort-level rank, computed over
    # ALL cells of the slide, then standardised WITHIN COHORT. That standardisation is the
    # thing that stops FiLM from reading its own input as a cohort fingerprint - see the
    # module docstring in models/marker_encoder.py.
    g = u_coh.groupby(img, sort=False)
    smed, siqr = g.median(), g.quantile(0.75) - g.quantile(0.25)
    smed = (smed - smed.mean()) / smed.std().replace(0.0, np.nan).fillna(1.0)
    siqr = (siqr - siqr.mean()) / siqr.std().replace(0.0, np.nan).fillna(1.0)
    smed.columns = [f'med::{t}' for t in triples]
    siqr.columns = [f'iqr::{t}' for t in triples]
    st = pd.concat([smed, siqr], axis=1).fillna(0.0).astype('float32')
    st.index.name = 'image_id'

    # stratified subsample: round-robin over (patient x native label), so a rare label is kept
    # whole and a huge stratum cannot swamp the draw. Cell ids are saved, so runs reproduce.
    idx = np.arange(len(df))
    if n_sub and n_sub < len(df):
        key = df.patient_id.astype(str) + '||' + df.native_label.astype(str)
        codes = pd.factorize(key)[0]
        order = rng.permutation(len(df))
        turn = pd.Series(codes[order]).groupby(codes[order]).cumcount().to_numpy()
        idx = np.sort(order[np.argsort(turn, kind='stable')[:n_sub]])

    def assemble(rows):
        o = df.iloc[rows][META].reset_index(drop=True)
        for name, block in (('u_img', u_img), ('u_coh', u_coh), ('raw', X)):
            b = block.iloc[rows].reset_index(drop=True)
            b.columns = [f'{name}::{t}' for t in triples]
            o = pd.concat([o, b], axis=1)
        return o

    out = assemble(idx)
    out.to_parquet(value_table(cohort), index=False)

    # A second, UNSTRATIFIED draw. The stratified table above is right for the probe, which
    # needs rare labels represented - but it is wrong for any check on the marginal
    # DISTRIBUTION, because class-balancing reweights it away from the cohort's real mixture.
    # Measured: |mean(u_coh) - 0.5| is 0.0009 on a random draw and 0.0475 on the stratified one
    # for Keren. Distribution checks use this table instead.
    ridx = np.sort(rng.choice(len(df), min(n_sub or len(df), len(df)), replace=False))
    assemble(ridx).to_parquet(os.path.join(VALUES, f'{cohort}_rand.parquet'), index=False)
    st.reset_index().to_parquet(os.path.join(VALUES, f'{cohort}_slidestats.parquet'), index=False)

    return dict(cohort=cohort, cells_total=len(df), cells_kept=len(out),
                slides=int(df.image_id.nunique()),
                labels_total=int(df.native_label.nunique()),
                labels_kept=int(out.native_label.nunique()))


# ----------------------------------------------------------------------------- load
def load_arm(cohort, triples, arm, rand=False):
    """(channels, target, slide_stats, meta) for one arm.

    The TARGET is always the cohort-level ECDF, identical for every arm - only the INPUT
    representation differs. Without that the R2 numbers would not be comparable.

    `rand=True` reads the UNSTRATIFIED draw. The probe wants the stratified table so rare
    labels are represented; every check on a distribution or a slide's composition wants this
    one, because class-balancing reweights both.
    """
    v = pd.read_parquet(os.path.join(VALUES, f'{cohort}_rand.parquet') if rand
                        else value_table(cohort))
    ch = np.stack([v[[f'{src}::{t}' for t in triples]].to_numpy('float32')
                   for src in ARMS[arm]['ch']], axis=-1)                 # (N, M, n_ch)
    y = v[[f'u_coh::{t}' for t in triples]].to_numpy('float32')

    s = pd.read_parquet(os.path.join(VALUES, f'{cohort}_slidestats.parquet')).set_index('image_id')
    S = np.stack([s[[f'med::{t}' for t in triples]].to_numpy('float32'),
                  s[[f'iqr::{t}' for t in triples]].to_numpy('float32')], axis=-1)
    pos = {k: i for i, k in enumerate(s.index)}
    return ch, y, S[v.image_id.map(pos).to_numpy()], v[META]


def raw_values(cohort, triples, rand=True):
    v = pd.read_parquet(os.path.join(VALUES, f'{cohort}_rand.parquet') if rand
                        else value_table(cohort))
    return v[[f'raw::{t}' for t in triples]].to_numpy('float32')


def T(*a):
    return [torch.from_numpy(np.ascontiguousarray(x)) for x in a]


# ----------------------------------------------------------------------------- probe
def run_fold(arm, triples, train_cohorts, test_cohort, epochs=EPOCHS, patience=PATIENCE,
             seed=SEED):
    """Train the masked-marker probe on `train_cohorts`, score it on the held-out cohort.

    Why a masked-marker probe rather than a classifier: Gate 1 must be scored leave-one-cohort-
    out, because fitting to slide statistics is invisible on held-out CELLS (same slides, same
    statistics) and only shows up on a held-out COHORT. A cross-cohort classifier would need a
    shared label space, which does not exist until Stage 1b. This task needs no labels at all
    and is the exact objective Stage 2 trains on.
    """
    torch.manual_seed(seed)
    M = len(triples)
    spec = ARMS[arm]

    def pack(cohorts):
        p = [load_arm(c, triples, arm) for c in cohorts]
        arrs = [np.concatenate([x[i] for x in p]) for i in range(3)]
        meta = pd.concat([x[3] for x in p], ignore_index=True)
        return T(*arrs) + [meta]

    tc, ty, ts, tmeta = pack(train_cohorts)
    ec, ey, es, _ = pack([test_cohort])

    # Early stopping on HELD-OUT SLIDES of the training cohorts - never the test cohort. A
    # fixed epoch budget would be arbitrary and could favour whichever arm happens to converge
    # at that speed; measured here, 8 epochs underfits and 40 overfits. The stop is chosen by
    # data, and the test cohort is never looked at until scoring.
    slides = tmeta.image_id.to_numpy()
    uniq = np.unique(slides)
    val_slides = set(np.random.default_rng(seed).choice(uniq, max(1, len(uniq) // 7), False))
    is_val = torch.from_numpy(np.isin(slides, list(val_slides)))
    tr_idx = torch.nonzero(~is_val).squeeze(1)
    va_idx = torch.nonzero(is_val).squeeze(1)
    if len(va_idx) > 15_000:      # the per-epoch check runs all M masks; cap it for speed
        va_idx = va_idx[torch.randperm(len(va_idx), generator=torch.Generator()
                                       .manual_seed(seed))[:15_000]]

    model = MaskedMarkerProbe(M, film=spec['film'], n_ch=len(spec['ch']), d=D_MODEL)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad, stopped = np.inf, None, 0, epochs

    for ep in range(epochs):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx), generator=gen)]
        for i in range(0, len(perm), 4096):
            b = perm[i:i + 4096]
            mi = torch.randint(0, M, (len(b),), generator=gen)
            pred, _ = model(tc[b], ts[b], mi)
            loss = torch.nn.functional.mse_loss(pred, ty[b, mi])
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            v = np.mean([float(torch.nn.functional.mse_loss(
                model(tc[va_idx], ts[va_idx],
                      torch.full((len(va_idx),), m, dtype=torch.long))[0], ty[va_idx, m]))
                for m in range(M)])
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                stopped = ep + 1
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # score every marker in turn, so none can be quietly skipped. The R2 baseline is the
    # TRAINING mean - predicting it is what "learned nothing transferable" looks like.
    model.eval()
    per = {}
    with torch.no_grad():
        for m in range(M):
            mi = torch.full((len(ec),), m, dtype=torch.long)
            pred, _ = model(ec, es, mi)
            res = ((ey[:, m] - pred) ** 2).sum()
            tot = ((ey[:, m] - ty[:, m].mean()) ** 2).sum()
            per[triples[m]] = float(1 - res / tot)

    gstat = None
    if model.film is not None:
        with torch.no_grad():
            _, g, b = model.film(ec[..., 0], es)
            gstat = dict(gamma_dev_mean=float((g - 1).abs().mean()),
                         beta_abs_mean=float(b.abs().mean()),
                         gamma_dev_max=float((g - 1).abs().max()),
                         beta_abs_max=float(b.abs().max()))
    return dict(r2=float(np.mean(list(per.values()))), per=per, film=gstat, model=model,
                epochs_used=stopped)


# ----------------------------------------------------------------------------- gate checks
def ks_mean(cohorts, triples, values):
    """Mean pairwise Kolmogorov-Smirnov distance between cohorts, per marker.

    KS is the right statistic here because it is invariant to any monotone rescaling: it asks
    whether the SHAPES line up, which is exactly what arrival-state differences break.
    """
    rows = []
    for j, t in enumerate(triples):
        d = []
        for a, b in combinations(cohorts, 2):
            xa = np.sort(values[a][:, j]); xb = np.sort(values[b][:, j])
            grid = np.concatenate([xa, xb])
            fa = np.searchsorted(xa, grid, 'right') / len(xa)
            fb = np.searchsorted(xb, grid, 'right') / len(xb)
            d.append(float(np.abs(fa - fb).max()))
        rows.append(float(np.mean(d)))
    return rows


def skew_check(cohorts, triples, expect, values, metas):
    """The check that decides V1 - as a WITHIN-LABEL contrast, not an absolute level.

    Take the label that dominates the most composition-skewed slides, and compare its median
    marker value on those skewed slides against its median on the most BALANCED slides. Same
    label, same cohort, same marker - only the slide composition changes, so prevalence is
    controlled by construction.

    A negative delta is the "invented negative population": ranking inside a 90%-tumour slide
    forces half those tumour cells below the median on keratin, so the identical cell type
    reads lower purely because of what it was sitting next to. That is the defect this stage
    exists to remove, and it is what an absolute median cannot separate from base rate.
    """
    tpos = {t: i for i, t in enumerate(triples)}
    rows = []
    for c in cohorts:
        meta, V = metas[c], values[c]
        exp = expect[expect.cohort == c].set_index('native_label').triple.to_dict()
        p = meta.groupby('image_id').native_label.value_counts(normalize=True)
        ent = p.groupby(level=0).apply(lambda s: float(-(s * np.log(s + 1e-12)).sum()))
        dom = p.groupby(level=0).idxmax().map(lambda k: k[1])
        order = ent.sort_values()
        skewed = set(order.index[:max(1, len(order) // 10)])
        balanced = set(order.index[len(order) // 2:])
        d, n = [], 0
        for im in skewed:
            t = exp.get(dom[im])
            if t is None:
                continue
            lab = dom[im]
            a = ((meta.image_id == im) & (meta.native_label == lab)).to_numpy()
            b = (meta.image_id.isin(balanced) & (meta.native_label == lab)).to_numpy()
            if a.sum() >= 20 and b.sum() >= 20:
                d.append(float(np.median(V[a, tpos[t]]) - np.median(V[b, tpos[t]])))
                n += 1
        if d:
            rows.append(dict(cohort=c, pairs=n, delta=round(float(np.mean(d)), 3)))
    return pd.DataFrame(rows)


def _auroc(pos, neg):
    """Rank-based AUROC. Ties get mid-ranks, so quantised markers score honestly."""
    a = np.concatenate([pos, neg])
    r = pd.Series(a).rank(method='average').to_numpy()
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def consistency_check(cohorts, triples, expect, values, metas):
    """Declared-in-advance assertions from panel/gate1_expect.csv, scored by AUROC.

    A cell whose native label names a marker must read HIGHER on that marker than the rest of
    its own cohort. This file SCORES the gate and never feeds the pipeline - same status as
    never_merge.csv and must_merge.csv at Stage 0b.

    Scored by AUROC, NOT by "median rank in the top quartile", which was the first attempt and
    is mathematically broken. A label of prevalence p sitting at the top of the distribution
    has a best-possible median rank of 1 - p/2, so the fixed 0.75 threshold is UNREACHABLE for
    any label above 50% prevalence. Measured: ferguson `SC` is 56.0% of its cohort, ceiling
    0.720; Keren `Keratin_positive_tumor` is 50.3%, ceiling 0.748 - and it "passed" at 0.751,
    i.e. the threshold was scoring base rate rather than biology. AUROC is invariant to class
    prevalence and asks the question actually intended: does this marker separate this label
    from everything else?
    """
    tpos = {t: i for i, t in enumerate(triples)}
    rows = []
    for _, r in expect.iterrows():
        if r.cohort not in cohorts or r.triple not in tpos:
            continue
        meta = metas[r.cohort]
        sel = (meta.native_label == r.native_label).to_numpy()
        if sel.sum() < 20 or (~sel).sum() < 20:
            continue
        col = values[r.cohort][:, tpos[r.triple]]
        auc = _auroc(col[sel], col[~sel])
        rows.append(dict(cohort=r.cohort, native_label=r.native_label, marker=r.marker,
                         cells=int(sel.sum()), auroc=round(auc, 3),
                         passes=bool(auc >= AUROC_MIN), note=getattr(r, 'note', '')))
    return pd.DataFrame(rows)


def cliff_figure(models, triples, path):
    """No cliffs: embedding norm against u must be smooth. A step means bins survived."""
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    grid = torch.linspace(0, 1, 501)
    for arm, mdl in models.items():
        with torch.no_grad():
            u = grid.view(-1, 1, 1).repeat(1, len(triples), mdl.enc.n_ch)
            tok = mdl.enc(u)
            ax.plot(grid.numpy(), tok.norm(dim=-1).mean(1).numpy(), lw=1.4, label=arm)
    ax.set_xlabel('u  (ECDF position)'); ax.set_ylabel('mean token norm')
    ax.set_title('GATE 1 - no cliffs: token norm must vary smoothly with u', fontsize=9)
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ----------------------------------------------------------------------------- main
def do_build(cohorts, triples):
    rows = []
    for c in cohorts:
        t0 = time.time()
        r = build(c, triples)
        r['seconds'] = round(time.time() - t0, 1)
        rows.append(r)
        print(f"  {c:9} {r['cells_kept']:>7,} / {r['cells_total']:>9,} cells  "
              f"{r['slides']:>4} slides  {r['seconds']:>5.1f}s")
    return pd.DataFrame(rows)


def do_bakeoff(cohorts, triples, genes, refit=False):
    train = [c for c in cohorts if SPECS[c]['role'] == 'train']
    expect = pd.read_csv(os.path.join(PANEL, 'gate1_expect.csv'), keep_default_na=False)

    metas = {c: load_arm(c, triples, 'V1', rand=True)[3] for c in cohorts}
    values, folds, films, models = {}, {}, {}, {}
    M, cache = len(triples), os.path.join(WORK, 'harmonise_folds.pt')

    def blank(arm):
        return MaskedMarkerProbe(M, film=ARMS[arm]['film'], n_ch=len(ARMS[arm]['ch']), d=D_MODEL)

    # --- LOCO probe, one fold per training cohort. 30 folds is ~45 minutes, so the results and
    #     the fitted weights are cached: iterating on the REPORT must not mean refitting.
    if os.path.exists(cache) and not refit:
        blob = torch.load(cache, weights_only=False)
        for arm, per_fold in blob.items():
            for held, r in per_fold.items():
                m = blank(arm); m.load_state_dict(r.pop('state')); m.eval()
                r['model'] = m
        folds = blob
        print(f"  loaded {sum(len(v) for v in folds.values())} cached folds from {cache}")
    else:
        for arm in ARMS:
            per_fold = {}
            for held in train:
                rest = [c for c in train if c != held]
                t0 = time.time()
                r = run_fold(arm, triples, rest, held)
                per_fold[held] = r
                print(f"  {arm:7} hold-out {held:9} R2={r['r2']:+.4f}  ({time.time()-t0:.0f}s)")
            folds[arm] = per_fold
        torch.save({a: {h: dict(r2=r['r2'], per=r['per'], film=r['film'],
                                epochs_used=r['epochs_used'],
                                state=r['model'].state_dict())
                        for h, r in pf.items()} for a, pf in folds.items()}, cache)
        print(f"  cached {cache}")

    for arm in ARMS:
        models[arm] = folds[arm][train[0]]['model']
        if folds[arm][train[0]]['film'] is not None:
            films[arm] = pd.DataFrame([dict(fold=k, **v['film']) for k, v in folds[arm].items()])

    # --- the channel-1 value each arm actually presents, for KS / skew / consistency.
    #     For a FiLM arm the correction comes from the fold where that cohort was HELD OUT, so
    #     no cohort is ever scored by a FiLM that trained on it. ferguson is not in any fold,
    #     so it borrows the first fold's model - it is a holdout and never scores the gate.
    for arm in ARMS:
        V = {}
        for c in cohorts:
            ch, y, s, meta = load_arm(c, triples, arm, rand=True)
            if ARMS[arm]['film']:
                mdl = folds[arm].get(c, folds[arm][train[0]])['model']
                with torch.no_grad():
                    V[c] = mdl.film(*T(ch[..., 0], s))[0].numpy()
            else:
                V[c] = ch[..., 0]
        values[arm] = V

    raw = {c: raw_values(c, triples) for c in cohorts}
    return dict(train=train, expect=expect, metas=metas, values=values, folds=folds,
                films=films, models=models, raw=raw)


def write_report(cohorts, triples, genes, B, build_rows):
    train, expect, metas, values = B['train'], B['expect'], B['metas'], B['values']
    short = {t: genes[t] if len(genes[t]) < 18 else t.split('|')[0] for t in triples}

    ks_before = ks_mean(cohorts, triples, B['raw'])
    ks = {a: ks_mean(cohorts, triples, values[a]) for a in ARMS}
    K = pd.DataFrame({'marker': [short[t] for t in triples],
                      'raw (before)': np.round(ks_before, 3),
                      **{a: np.round(ks[a], 3) for a in ARMS}})

    R2 = pd.DataFrame({a: {h: round(B['folds'][a][h]['r2'], 4) for h in train} for a in ARMS})
    R2.loc['**mean**'] = R2.mean().round(4)
    winner = R2.loc['**mean**'].idxmax()

    SK = {a: skew_check(cohorts, triples, expect, values[a], metas) for a in ARMS}
    first = next(iter(ARMS))
    skew = SK[first][['cohort', 'pairs']].copy()
    for a in ARMS:
        skew[a] = SK[a].delta.values

    CO = {a: consistency_check(cohorts, triples, expect, values[a], metas) for a in ARMS}
    cons = pd.DataFrame({a: [f"{int(CO[a].passes.sum())} / {len(CO[a])}"] for a in ARMS},
                        index=['assertions passing'])

    figp = os.path.join(FIGURES, 'harmonise_no_cliffs.png')
    cliff_figure(B['models'], triples, figp)

    out, A = [], None
    A = out.append
    A("# Stage 1 - Continuous value harmonisation (GATE 1)\n")
    A(f"**{len(ARMS)} arms** raced on **{len(triples)} markers measured by all {len(cohorts)} "
      f"cohorts**, scored **leave-one-cohort-out** over the {len(train)} training cohorts.\n")
    A(f"**Winner on mean LOCO R2: `{winner}`.**\n")

    A("## The arms, and what each comparison isolates\n")
    A(pd.DataFrame([dict(arm=a, channels=' , '.join(s['ch']), film=s['film'] or '-')
                    for a, s in ARMS.items()]).to_markdown(index=False))
    A("\n| comparison | question it answers |")
    A("|---|---|")
    A("| V1 vs V3 | is per-image grouping harmful on its own? |")
    A("| V3 vs V2a / V2b | is a learned slide correction worth anything at all? |")
    A("| V2a vs V2b | is the **extra capacity** in that correction worth anything? |")
    A("| V3 vs V1+V3 | does the image rank **add** to the cohort rank rather than replace it? |")
    A("| V1+V3 vs V2a+V1 | does FiLM beat simply handing over the image rank? |")

    A("\n### A channel the plan asked for, dropped on measurement\n")
    A("The plan specified a second channel holding *the value relative to a cohort-level "
      "reference* (tanh of a robust z-score), so that ranking would not destroy prevalence. "
      "Built and measured, it fails twice. It **saturates** - 5.5-17.5% of cells sit at "
      "`|lvl| > 0.99`, worst on Sorin, which arrives uint8 so most markers have median 0 and a "
      "near-zero IQR. And it is **redundant by construction**: any per-cell function of the raw "
      "value computed from cohort statistics is a monotone transform of that value, so it "
      "carries what `u_coh` already carries (measured correlation 0.89-0.97).\n")
    A("It would also have broken this gate. An arm holding `{u_img, lvl}` strictly contains an "
      "arm holding `{u_coh, lvl}`, so **V1 could not have lost**. The second channel is "
      "therefore the *other grouping*, which is genuinely independent information.\n")

    A("## The shared core\n")
    A("Gate 1 compares arms against each other, so every fold must see an identical feature "
      "space - otherwise an arm could win on imputation rather than on normalisation. Only "
      "triples present in every cohort are used, so nothing is filled in.\n")
    A(pd.DataFrame({'triple': triples, 'gene / members': [genes[t] for t in triples]})
      .to_markdown(index=False))

    A("\n## Subsample actually built\n")
    A("Stratified round-robin over (patient x native label), so a rare label is kept whole and "
      "a huge stratum cannot swamp the draw. References (cohort ECDF, per-image ECDF, slide "
      "statistics) come from **all** cells; only the written table is subsampled.\n")
    A(build_rows.to_markdown(index=False))

    A("\n## Check 1 - cross-cohort distribution overlap (mean pairwise KS)\n")
    A("Lower is better. `raw (before)` is the values exactly as each cohort ships them. All "
      "distribution checks below use the **unstratified** draw - see the note at the end of "
      "this section.\n")
    A(K.to_markdown(index=False))
    A(f"\n**Mean over markers** - raw **{np.mean(ks_before):.3f}** -> " +
      " · ".join(f"{a} **{np.mean(ks[a]):.3f}**" for a in ARMS) + "\n")
    A("\n**This check confirms the arrival-state problem is fixed, and it CANNOT rank the "
      "arms.** Raw values sit at 0.705 mean KS - four different curve shapes, exactly the "
      "defect Stage 1 exists to remove - and every arm lands near 0.2-0.3. But a per-group "
      "ECDF forces each group's marginal to uniform *by construction*, so between-cohort KS is "
      "near zero for any rank-based arm on a representative sample, and whatever spread remains "
      "reflects the sample, not the method. Reading an arm ranking off this table would be "
      "reading noise. Checks 2, 3 and 4 decide.\n")
    A("*(First attempt at this check ran on the class-balanced table and appeared to rank V1 "
      "best and V3 worst. That was an artefact of the balancing: measured "
      "`|mean(u_coh) - 0.5|` is 0.0009 on a random draw of Keren and 0.0475 on the stratified "
      "one. Distribution checks now use the unstratified draw.)*\n")

    A("\n## Check 2 - composition-skew robustness (the check that decides V1)\n")
    A("Take the label that dominates the most composition-skewed 10% of slides, and compare "
      "its median marker value **on those slides** against its median **on the most balanced "
      "slides**. Same label, same cohort, same marker - only the neighbours change.\n")
    A("A negative delta is the *invented negative population*: ranking inside a 90%-tumour "
      "slide pushes half those tumour cells below the median on keratin, so an identical cell "
      "reads lower purely because of what it was sitting next to. Reported as a contrast rather "
      "than an absolute level, because an absolute median cannot be separated from base rate.\n")
    A(skew.to_markdown(index=False))

    A("\n## Check 3 - label / marker consistency (AUROC)\n")
    A("Assertions declared in advance in `celltype_transfer/declared/gate1_expect.csv`: a cell whose "
      "native label names a marker must read **higher on that marker than the rest of its own "
      f"cohort**, at AUROC >= {AUROC_MIN}. That file scores the gate and never feeds the "
      "pipeline - the same status `never_merge.csv` has at Stage 0b.\n")
    A(cons.to_markdown())
    A("\n*Scored by AUROC, not by \"median rank in the top quartile\", which was the first "
      "attempt and is mathematically broken. A label of prevalence `p` sitting at the top of "
      "the distribution has a best-possible median rank of `1 - p/2`, so a fixed 0.75 threshold "
      "is **unreachable** above 50% prevalence. Measured: ferguson `SC` is 56.0% of its cohort "
      "(ceiling 0.720) and Keren `Keratin_positive_tumor` is 50.3% (ceiling 0.748) - the latter "
      "\"passed\" at 0.751, i.e. the threshold was scoring base rate, not biology. AUROC is "
      "invariant to prevalence and asks the intended question.*\n")
    fails = CO[winner][~CO[winner].passes]
    if len(fails):
        A(f"\n<details><summary><b>{len(fails)} assertion(s) failing under the winner "
          f"{winner}</b></summary>\n")
        A(fails.to_markdown(index=False))
        A("\n</details>\n")

    A("\n## Check 4 - LOCO masked-marker transfer R2 (the decision)\n")
    A("Hide one marker, predict its cohort-level ECDF value from the other "
      f"{len(triples)-1}; train on {len(train)-1} cohorts, score on the held-out one. The "
      "target is the same quantity for every arm, so the numbers are comparable.\n")
    A("Scored **leave-one-cohort-out, not on held-out cells** - on purpose. Fitting to slide "
      "statistics is invisible on held-out cells, which share the very slides the statistics "
      "came from, and only shows up on a cohort the model has never seen.\n")
    A(R2.to_markdown())

    A("\n## Check 5 - FiLM capacity\n")
    if B['films']:
        A(f"How hard FiLM is actually pushing. The output is bounded at "
          f"`eps = {EPS_FILM}`, so `|gamma-1|` near {EPS_FILM} means it is straining against "
          "the bound - which usually means it is fitting cohort identity, not slide drift.\n")
        for a, f in B['films'].items():
            A(f"\n**{a}**\n")
            A(f.round(4).to_markdown(index=False))
    else:
        A("No FiLM arm produced statistics.\n")

    A("\n## No cliffs\n")
    A(f"![no cliffs](figures/{os.path.basename(figp)})\n")
    A("Token norm against `u`. The line must be smooth: any step function means a bin survived "
      "somewhere, which is the exact defect this stage replaces.\n")

    path = os.path.join(REPORTS, 'harmonise_values.md')
    open(path, 'w', encoding='utf-8').write("\n".join(out))
    return path, winner, R2, K, skew, cons, CO


def main():
    cohorts = built()
    if not cohorts:
        sys.exit("nothing built - run: python loaders.py")
    triples, genes = core_triples(cohorts)
    print(f"shared core: {len(triples)} markers across {len(cohorts)} cohorts")
    for t in triples:
        print(f"   {genes[t][:40]:42} {t}")

    need = [c for c in cohorts if not os.path.exists(value_table(c))]
    rows = None
    if '--rebuild' in sys.argv or need:
        print("\nbuilding value tables...")
        rows = do_build(cohorts if '--rebuild' in sys.argv else need, triples)

    if '--bakeoff' not in sys.argv:
        print("\ndone. run with --bakeoff for GATE 1.")
        return

    if rows is None or len(rows) < len(cohorts):
        rows = pd.DataFrame([dict(cohort=c,
                                  cells_kept=len(pd.read_parquet(value_table(c), columns=['cell_id'])),
                                  cells_total=json.load(open(os.path.join(
                                      config.RAW, f'{c}_meta.json')))['n_cells'],
                                  slides=json.load(open(os.path.join(
                                      config.RAW, f'{c}_meta.json')))['n_images'])
                             for c in cohorts])

    print("\nrunning the bake-off...")
    B = do_bakeoff(cohorts, triples, genes, refit='--refit' in sys.argv)
    path, winner, R2, K, skew, cons, CO = write_report(cohorts, triples, genes, B, rows)

    print("\n--- LOCO masked-marker R2 ---")
    print(R2.to_string())
    print(f"\nwinner: {winner}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

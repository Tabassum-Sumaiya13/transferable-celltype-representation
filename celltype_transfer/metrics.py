"""
Shared metric library (plan F6) - every Layer-2 annotation metric computed ONE way, from a saved
predictions table.

    python metrics.py --selftest    # cross-checks every metric against sklearn / hand cases

THE PROBLEM THIS FIXES. `macro_f1` was written once, inside train_adversarial_encoder.py - a TRAINING script -
and every later stage reached into it as `s3.macro_f1` (s4, s6, s7, s7b, s8, s9, s10). That
worked by accident: nothing ever changed train_adversarial_encoder's copy. But it means the metric library IS a
training stage, and a future arm that never trains anything (MAPS, ASTIR, a Track B baseline)
would have to import a whole encoder-training module just to get a score.
`benchmark_protocol.yaml`'s evaluation section says the underlying rule in as many words: no arm
may be scored by its own reporting code. This file is that one piece of code.

WHAT MOVED HERE, BYTE-IDENTICAL. `macro_f1` is the exact computation train_adversarial_encoder.py shipped,
verified by `--selftest` below - not a rewrite. train_adversarial_encoder.py now does
`from metrics import macro_f1` instead of defining its own, so `s3.macro_f1` keeps working for
every existing caller and no gate's number moves by a single digit. `train_adversarial_encoder.baselines` is
LEFT WHERE IT IS, unchanged - it reads Stage 3/6's own `data` dict shape, and coupling this file
to that shape would defeat the point of a stage-independent library. `metrics.baselines` below is
a new, decoupled version (plain label arrays in, no dict) for callers that are not shaped like a
Stage 3/6 training loop.

WHAT IS NEW - none of this existed anywhere in the codebase before this file:
  balanced_accuracy, cohen_kappa, weighted_f1   (`evaluation.layer_2_annotation_performance.
                                                  secondary_aggregate`)
  confusion_pairs                                (`...confusion`: top confused pairs, not just
                                                  the aggregate)
  support_stratified_f1                          (`...support_stratified_f1`: rare types are the
                                                  hard part and an aggregate can hide them)
  per_platform                                   (`...per_platform_breakdown`)
  save_predictions / load_predictions            (`required_outputs.predictions`: a
                                                  `reports/predictions_{method}_{fold}.parquet`
                                                  file did not exist anywhere before this)

Every new metric that needs a class SET (balanced_accuracy, cohen_kappa, confusion_pairs) uses the
SAME convention `macro_f1` already uses - classes PRESENT IN THE TRUTH minus `drop` - so a stroma
waiver (D-32) or an unreliable-cluster exclusion applies identically everywhere a fold reports
more than one metric, instead of each metric silently picking its own class set.

WHAT IS DELIBERATELY NOT HERE YET. Layer 1 (label-space quality - Stage 1b's own scoring code
already covers this and reads no gate through this file), Layer 2 open-set / calibration (Gate 7b
already computes msp/proto abstention on the frozen holdout), and Layer 2 robustness (panel
masking, the LOCO-LOTO generalization gap). Writing the whole of `benchmark_protocol.yaml`'s
evaluation section into one file before Track A/B exist to consume most of it would be scope
creep pretending to be progress - this first cut is the metric set every gate already needs
today. Extend this file, do not start a second one, when those are actually needed.
"""
import os
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # runs from any directory
import config
from config import WORK, REPORTS, SPECS

PRED_COLUMNS = ('cell_id', 'cohort', 'held', 'method', 'protocol', 'y_true', 'y_pred')


# ----------------------------------------------------------------------------- paired fold comparison
def paired(delta):
    """(Moved here from the spatial stage, verbatim.) Mean, 95% CI and an EXACT sign-flip p-value on a paired per-fold difference.

    gate4_expect.csv check 3, and the first gate in this project to be scored with an interval
    (H9). The one time the tests were run retrospectively they did not survive: Gate 6 check 4
    passed its declared +0.02 with p = 0.460 and a CI four times the margin, all of it from a
    single fold (D-44). With n = 7 the sign-flip test is exact - all 2^7 = 128 reassignments are
    enumerated - so no normality is assumed for the p-value, only for the interval.
    """
    from scipy import stats
    x = np.asarray(delta, float)
    n = len(x)
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else float('nan')
    h = float(stats.t.ppf(0.975, n - 1) * se) if n > 1 else float('nan')
    signs = np.array([[1 if (i >> b) & 1 else -1 for b in range(n)] for i in range(2 ** n)])
    means = (signs * x).mean(1)
    p = float((np.abs(means) >= abs(mean) - 1e-12).mean())
    return dict(n=n, mean=mean, lo=mean - h, hi=mean + h, p=p)


# ----------------------------------------------------------------------------- the primary metric
def macro_f1(y_true, y_pred, n_class, drop=frozenset()):
    """Macro-F1 over classes PRESENT IN THE TRUTH, excluding `drop`.

    Averaging over classes a cohort does not contain would score absent classes as 0 and turn the
    metric into a measure of label coverage rather than accuracy. Classes present in the
    prediction but not the truth still cost precision, so this does not let a model win by
    over-predicting.

    Moved verbatim from train_adversarial_encoder.py (D-F6) - see this module's docstring. Every stage that used
    to call `s3.macro_f1` still can; train_adversarial_encoder.py now imports this same function under that name.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    present = sorted(set(int(v) for v in np.unique(y_true)) - set(drop))
    if not present:
        return float('nan'), pd.DataFrame()
    rows = []
    for c in present:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        rows.append(dict(cluster=c, support=int((y_true == c).sum()),
                         precision=round(p, 4), recall=round(r, 4),
                         f1=round(2 * p * r / (p + r), 4) if p + r else 0.0))
    d = pd.DataFrame(rows)
    return float(d.f1.mean()), d


def baselines(y_train, y_test, n_class, drop=frozenset(), seed=0):
    """MAJORITY and RANDOM reference predictors, scored with the same `macro_f1`.

    Decoupled from any one stage's tensor layout on purpose - callers pass plain label arrays, so
    this works for a future arm's saved predictions too, not only a Stage 3/6 training loop.
    `train_adversarial_encoder.baselines` / `train_prototype_classifier`'s inline majority+random are UNCHANGED and still read
    the `data` dict directly; this is the version for everything else.
    """
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    maj = int(np.bincount(y_train, minlength=n_class).argmax())
    f_maj, _ = macro_f1(y_test, np.full(len(y_test), maj), n_class, drop=drop)
    rnd = np.random.default_rng(seed).integers(0, n_class, len(y_test))
    f_rnd, _ = macro_f1(y_test, rnd, n_class, drop=drop)
    return float(f_maj), float(f_rnd), maj


# ----------------------------------------------------------------------------- confusion (shared by cohen_kappa and confusion_pairs)
def _confusion(y_true, y_pred, classes):
    """Square confusion matrix over `classes`, rows = truth, columns = predictions.

    Restricted to rows whose TRUTH is in `classes` - the same "present in the truth minus drop"
    filter `macro_f1` applies - so a dropped/unreliable truth label never enters. A PREDICTED
    value outside `classes` is not given its own row or column (the protocol never declares an
    'other' class): that row simply scores no true positive for anything, which is the correct
    cost for a wrong prediction, not a silently invented category.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    idx = {c: i for i, c in enumerate(classes)}
    keep = np.isin(y_true, classes)
    yt = np.array([idx[c] for c in y_true[keep]], dtype=int)
    yp_raw = y_pred[keep]
    in_class = np.isin(yp_raw, classes)
    m = np.zeros((len(classes), len(classes)), dtype=np.int64)
    if in_class.any():
        yp = np.array([idx[c] for c in yp_raw[in_class]], dtype=int)
        np.add.at(m, (yt[in_class], yp), 1)
    return m


def cohen_kappa(y_true, y_pred, n_class, drop=frozenset()):
    """Chance-corrected agreement: (po - pe) / (1 - pe).

    Restricted to classes present in the truth minus `drop`, same convention as `macro_f1`, so
    this is comparable to the primary metric on the same rows rather than picking its own class
    set the way a default library call would.
    """
    y_true = np.asarray(y_true)
    classes = sorted(set(int(v) for v in np.unique(y_true)) - set(drop))
    if len(classes) < 2:
        return float('nan')
    m = _confusion(y_true, y_pred, classes)
    n = m.sum()
    if n == 0:
        return float('nan')
    po = np.trace(m) / n
    row, col = m.sum(1) / n, m.sum(0) / n
    pe = float((row * col).sum())
    return float('nan') if pe >= 1.0 else float((po - pe) / (1 - pe))


def confusion_pairs(y_true, y_pred, names=None, drop=frozenset(), top_k=10):
    """Top confused (true, predicted) pairs, OFF-DIAGONAL only.

    `evaluation.layer_2_annotation_performance.confusion`: "report the top confused class pairs
    per fold, not only the aggregate" - a macro-F1 number alone cannot say WHICH two types a
    model mixes up.
    """
    y_true = np.asarray(y_true)
    classes = sorted(set(int(v) for v in np.unique(y_true)) - set(drop))
    if not classes:
        return pd.DataFrame(columns=['true', 'predicted', 'count'])
    m = _confusion(y_true, y_pred, classes)
    rows = [dict(true=names[t] if names else t, predicted=names[p] if names else p,
                count=int(m[i, j]))
            for i, t in enumerate(classes) for j, p in enumerate(classes)
            if i != j and m[i, j] > 0]
    d = pd.DataFrame(rows)
    return (d.sort_values('count', ascending=False).head(top_k).reset_index(drop=True)
            if len(d) else d)


# ----------------------------------------------------------------------------- derived from the per-class table (never a second pass over raw arrays)
def balanced_accuracy(y_true, y_pred, n_class, drop=frozenset()):
    """Unweighted mean of per-class recall - the standard definition (sklearn's included).

    Reuses `macro_f1`'s own per-class table rather than re-deriving recall in a second pass, so
    this can never disagree with `macro_f1` about which classes are in scope.
    """
    _, per_cls = macro_f1(y_true, y_pred, n_class, drop=drop)
    return float(per_cls.recall.mean()) if len(per_cls) else float('nan')


def weighted_f1(per_cls):
    """Support-weighted mean F1 (`secondary_aggregate.weighted_f1`).

    Takes the per-class table `macro_f1()` already returns - `weighted_f1(macro_f1(...)[1])` -
    so it is never computed from a second, possibly-different pass over the raw predictions.
    """
    if not len(per_cls):
        return float('nan')
    w = per_cls.support.to_numpy(dtype=float)
    return float((per_cls.f1.to_numpy() * w).sum() / w.sum()) if w.sum() else float('nan')


def support_stratified_f1(per_cls, edges=(0, 50, 200, 1000, np.inf),
                          labels=('rare (<50)', 'small (50-200)', 'medium (200-1000)',
                                  'large (>=1000)')):
    """Bin classes by TEST support and report mean F1 per bin.

    `support_stratified_f1`: "rare types are the hard part" - a single macro-F1 number cannot
    say whether a model works everywhere or only on the common classes it sees the most of.
    Bins are on TEST support (the `support` column `macro_f1` returns for the SCORED fold), not
    training support - it is the number of held-out cells actually available to be scored.
    """
    if not len(per_cls):
        return pd.DataFrame(columns=['bin', 'n_classes', 'mean_f1'])
    b = pd.cut(per_cls.support, bins=edges, labels=labels, right=False)
    b.name = 'bin'
    g = per_cls.groupby(b, observed=False).agg(n_classes=('f1', 'size'), mean_f1=('f1', 'mean'))
    return g.reset_index()


def per_platform(y_true, y_pred, cohorts, n_class, drop=frozenset()):
    """`per_platform_breakdown`: [CODEX, MIBI-TOF, IMC].

    The platform for each cohort comes from `config.SPECS['tech']`, never hand-typed here - the
    same "cohort-specific knowledge lives only in config.py" rule the rest of the pipeline keeps.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    plat = np.array([SPECS[c]['tech'] for c in cohorts])
    rows = []
    for p in sorted(set(plat)):
        m = plat == p
        f1, _ = macro_f1(y_true[m], y_pred[m], n_class, drop=drop)
        rows.append(dict(platform=p, n_cells=int(m.sum()), macro_f1=f1))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- the saved-predictions contract (required_outputs.predictions)
def save_predictions(path, cell_id, cohort, held, y_true, y_pred, method='ours', protocol='LOCO'):
    """Write `reports/predictions_{method}_{fold}.parquet` (`required_outputs.predictions`).

    This file did not exist anywhere in the codebase before plan F6: every gate scored its
    predictions in memory and threw them away, which is exactly what makes "no arm can use a
    different implementation" unverifiable after the fact. Saving the raw arrays lets any
    metric in this file - including one added after a run finished - be recomputed from the
    same predictions without retraining anything.
    """
    df = pd.DataFrame({
        'cell_id': np.asarray(cell_id),
        'cohort': cohort,      # scalar (one held-out cohort) or an array-like of per-row cohorts -
        'held': held,          # pandas broadcasts either correctly; np.asarray would not
        'method': method,
        'protocol': protocol,
        'y_true': np.asarray(y_true, dtype='int64'),
        'y_pred': np.asarray(y_pred, dtype='int64'),
    })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_predictions(path):
    return pd.read_parquet(path)


def score_from_predictions(df, n_class, drop=frozenset(), names=None, top_k_confusion=10):
    """The one function every arm should call to turn a saved predictions table into every
    Layer-2 annotation metric this file knows - `evaluation.layer_2_annotation_performance`.

    Takes a DataFrame shaped like `save_predictions` writes (or several folds concatenated); a
    `cohort` column, if present, also gets the per-platform breakdown for free.
    """
    yt, yp = df.y_true.to_numpy(), df.y_pred.to_numpy()
    f1_core, per_cls = macro_f1(yt, yp, n_class, drop=drop)
    f1_all, _ = macro_f1(yt, yp, n_class)
    out = dict(
        macro_f1_reliable=f1_core,
        macro_f1_all_classes=f1_all,
        balanced_accuracy=balanced_accuracy(yt, yp, n_class, drop=drop),
        cohen_kappa=cohen_kappa(yt, yp, n_class, drop=drop),
        weighted_f1=weighted_f1(per_cls),
        per_class=per_cls,
        support_stratified=support_stratified_f1(per_cls),
        confusion_top=confusion_pairs(yt, yp, names=names, drop=drop, top_k=top_k_confusion),
    )
    if 'cohort' in df.columns:
        out['per_platform'] = per_platform(yt, yp, df.cohort.to_numpy(), n_class, drop=drop)
    return out


# ----------------------------------------------------------------------------- selftest
def _selftest():
    """Cross-checks every metric against an independent reference (sklearn where the two
    definitions coincide - no `drop`, `classes` = the full label set - plus hand-checkable
    tiny cases) so this file is verified against something other than its own math."""
    from sklearn.metrics import (f1_score, precision_score, recall_score, balanced_accuracy_score,
                                 cohen_kappa_score, confusion_matrix as sk_confusion)

    rng = np.random.default_rng(20260810)
    n_class = 6
    yt = rng.integers(0, n_class, 4000)
    yp = np.where(rng.random(4000) < 0.7, yt, rng.integers(0, n_class, 4000))
    classes = list(range(n_class))   # every class appears in yt with this many draws - checked below
    assert set(np.unique(yt)) == set(classes), "selftest draw did not cover every class - rerun"

    f1, per_cls = macro_f1(yt, yp, n_class)
    assert list(per_cls.cluster) == classes
    sk_p = precision_score(yt, yp, labels=classes, average=None, zero_division=0)
    sk_r = recall_score(yt, yp, labels=classes, average=None, zero_division=0)
    sk_f = f1_score(yt, yp, labels=classes, average=None, zero_division=0)
    assert np.allclose(per_cls.precision, sk_p, atol=1e-4), "precision disagrees with sklearn"
    assert np.allclose(per_cls.recall, sk_r, atol=1e-4), "recall disagrees with sklearn"
    assert np.allclose(per_cls.f1, sk_f, atol=1e-4), "f1 disagrees with sklearn"
    assert abs(f1 - sk_f.mean()) < 1e-4, "macro_f1 mean disagrees with sklearn's macro average"
    print(f"  macro_f1 / per-class precision-recall-f1 match sklearn  (f1={f1:.4f})")

    ba = balanced_accuracy(yt, yp, n_class)
    assert abs(ba - balanced_accuracy_score(yt, yp)) < 1e-4, "balanced_accuracy disagrees"
    print(f"  balanced_accuracy matches sklearn  ({ba:.4f})")

    kappa = cohen_kappa(yt, yp, n_class)
    assert abs(kappa - cohen_kappa_score(yt, yp)) < 1e-4, "cohen_kappa disagrees with sklearn"
    print(f"  cohen_kappa matches sklearn  ({kappa:.4f})")

    wf1 = weighted_f1(per_cls)
    assert abs(wf1 - f1_score(yt, yp, labels=classes, average='weighted', zero_division=0)) < 1e-4
    print(f"  weighted_f1 matches sklearn  ({wf1:.4f})")

    m_ours = _confusion(yt, yp, classes)
    m_sklearn = sk_confusion(yt, yp, labels=classes)
    assert np.array_equal(m_ours, m_sklearn), "confusion matrix disagrees with sklearn"
    print("  confusion matrix matches sklearn")

    cp = confusion_pairs(yt, yp, top_k=3)
    assert len(cp) == 3 and (cp['count'].diff().dropna() <= 0).all(), "confusion_pairs not sorted"
    print(f"  confusion_pairs: top pair {cp.iloc[0].true}->{cp.iloc[0].predicted} "
          f"({cp.iloc[0]['count']} cells)")

    # drop: dropping a class must remove it from macro_f1, balanced_accuracy and cohen_kappa
    # ALL THREE, in the same way - the whole point of sharing one class-selection rule.
    drop = {0}
    f1d, per_cls_d = macro_f1(yt, yp, n_class, drop=drop)
    assert 0 not in set(per_cls_d.cluster), "drop did not remove the class from macro_f1"
    bad = balanced_accuracy(yt, yp, n_class, drop=drop)
    kappad = cohen_kappa(yt, yp, n_class, drop=drop)
    assert abs(bad - per_cls_d.recall.mean()) < 1e-9
    assert not np.isnan(kappad) and kappad != kappa
    print(f"  drop={{0}} changes macro_f1 ({f1:.4f}->{f1d:.4f}), balanced_accuracy and "
          f"cohen_kappa ({kappa:.4f}->{kappad:.4f}) consistently")

    # support_stratified_f1: bins must partition every class exactly once
    strat = support_stratified_f1(per_cls)
    assert strat.n_classes.sum() == len(per_cls), "support_stratified_f1 dropped or duplicated a class"
    print(f"  support_stratified_f1 bins partition all {len(per_cls)} classes")

    # hand case for baselines(): majority is whichever training class is commonest
    ytr = np.array([0, 0, 0, 1, 1, 2])
    maj_f1, rnd_f1, maj = baselines(ytr, yt[:200], n_class)
    assert maj == 0, f"expected majority class 0, got {maj}"
    print(f"  baselines(): majority class {maj} correct, maj_f1={maj_f1:.4f} rnd_f1={rnd_f1:.4f}")

    # per_platform: needs real cohort names - use two real ones from config with different tech
    cohorts_two = rng.choice(['CRC', 'Danenberg'], size=len(yt))   # CODEX vs MIBI-TOF
    pp = per_platform(yt, yp, cohorts_two, n_class)
    assert set(pp.platform) == {SPECS['CRC']['tech'], SPECS['Danenberg']['tech']}
    assert pp.n_cells.sum() == len(yt)
    print(f"  per_platform: {dict(zip(pp.platform, pp.n_cells))}")

    # save_predictions / load_predictions round-trip, scored two ways, must agree exactly
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = save_predictions(os.path.join(td, 'predictions_ours_test.parquet'),
                             cell_id=np.arange(len(yt)), cohort=cohorts_two, held='test',
                             y_true=yt, y_pred=yp)
        back = load_predictions(p)
        scored = score_from_predictions(back, n_class)
        assert abs(scored['macro_f1_reliable'] - f1) < 1e-9, "round-trip changed macro_f1"
        assert 'per_platform' in scored
        print(f"  save_predictions/load_predictions round-trip: macro_f1 unchanged "
              f"({scored['macro_f1_reliable']:.4f}), wrote {os.path.getsize(p)} bytes")

        # the real call pattern (train_prototype_classifier.loco_run): ONE held-out cohort, a scalar string, not
        # a per-row array - must broadcast, not raise or silently write one row.
        p2 = save_predictions(os.path.join(td, 'predictions_ours_CRC.parquet'),
                              cell_id=np.arange(len(yt)), cohort='CRC', held='CRC',
                              y_true=yt, y_pred=yp)
        back2 = load_predictions(p2)
        assert (back2.cohort == 'CRC').all() and len(back2) == len(yt), \
            "scalar cohort did not broadcast to every row"
        print(f"  save_predictions with a scalar cohort broadcasts correctly ({len(back2)} rows)")

    print("\n--selftest: all checks passed")


if __name__ == '__main__':
    import sys
    if '--selftest' in sys.argv:
        _selftest()
    else:
        print("nothing to do. run with --selftest.")

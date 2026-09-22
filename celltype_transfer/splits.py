"""The within-cohort train / val / test split, by PATIENT (plan F3) - one definition for every stage.

Every stage gets its cells through `split_masks(cohort, image_ids)`, passing the value table's own
`image_id` column unchanged. Every checkpoint records `split_fp()`; a cached fit whose fingerprint
differs is refitted (`split_stale`). Moved verbatim out of pretrain_masked_markers.py.
"""
import os
import zlib

import numpy as np
import pandas as pd

from config import SPECS, full_table, rng as _rng

# Fractions of SPLIT UNITS - patients, or slides where a cohort falls back - never of cells. See
# split_plan() for the rule and for what the slide split this replaced got wrong.
TEST_FRAC, VAL_FRAC = 0.20, 0.12


# ----------------------------------------------------------------------------- data
_PLANS = {}


def split_plan(cohort):
    """The within-cohort train / val / test split, by PATIENT (plan F3).

    THE RULE is benchmark_protocol.yaml `splits.within_training`, declared before this code:
    split unit = patient; fall back to slide only where patient_id is missing or constant, and say
    which cohorts fell back; fractions 0.68 / 0.12 / 0.20 of units; never the same patient in
    training and validation/test; every caller passes the same id format.

    WHAT THE SLIDE SPLIT IT REPLACES GOT WRONG (measured 2026-09-11 on the 7-cohort roster). It kept
    cells of one SLIDE together but let one PATIENT's slides fall on both sides. Share of test cells
    whose patient also had slides in training: CRC 1.00, Phillips 1.00, ferguson 1.00, UPMC 0.77,
    Danenberg 0.09, Sorin 0.07, Keren 0.00 (one slide per patient). CRC has 4 slides per patient,
    Phillips up to 10, so a 'held-out' CRC test set was 21 patients the model had already trained on.
    It inflated Gate 2 check 1 (within-cohort R2) and every early-stopping decision; it did NOT touch
    a LOCO score, which is scored on a cohort no fit trained on.

    BUILT FROM THE FULL VALUE TABLE, not from the caller's cells. Stages 3 and 6 drop cells whose
    label is not in the Stage 1b map before they split; if that ever emptied a slide, the old
    splitter shuffled a shorter list and every stage got a DIFFERENT split. Today it did not happen
    (no slide lost all its cells) - it was latent, and a label-map change would have made it live.

    Slides must nest inside units (asserted), so a slide id alone names its part - which is what
    lets every caller keep its cells-by-slide bookkeeping. The RNG key is unchanged ('split',
    cohort): where patient and slide coincide (Keren) the split is bit-identical to the old one.
    """
    if cohort in _PLANS:
        return _PLANS[cohort]
    v = pd.read_parquet(full_table(cohort), columns=['image_id', 'patient_id'])
    pre = f'{cohort}|'
    img, pid = v.image_id.astype(str), v.patient_id.astype(str)
    if not (img.str.startswith(pre).all() and pid.str.startswith(pre).all()):
        raise RuntimeError(f"{cohort}: image_id / patient_id do not carry the '{pre}' prefix - "
                           f"loaders.py writes it; the value table is from another build")
    tail = pid.str[len(pre):]
    missing = v.patient_id.isna() | tail.isin(['', 'unknown', 'nan', 'None'])
    if missing.any():
        unit, why = img, f'patient_id missing for {missing.mean():.1%} of cells'
    elif pid.nunique() == 1:
        unit, why = img, 'patient_id is constant'
    else:
        unit, why = pid, ''
    pairs = pd.DataFrame({'img': img, 'unit': unit}).drop_duplicates()
    if pairs.img.duplicated().any():
        bad = pairs[pairs.img.duplicated(keep=False)].img.iloc[0]
        raise RuntimeError(f"{cohort}: slide {bad} spans more than one patient - the split cannot "
                           f"be expressed per slide")
    units = np.array(sorted(pairs.unit.unique()))
    if len(units) < 3:
        raise RuntimeError(f"{cohort}: {len(units)} split units - train, val and test need 3. "
                           f"No rule for this is declared; add one to the protocol first")
    r = _rng('split', cohort).permutation(len(units))
    nt = max(1, int(round(TEST_FRAC * len(units))))
    nv = max(1, int(round(VAL_FRAC * len(units))))
    of_unit = {**{u: 'train' for u in units[r[nt + nv:]]},
               **{u: 'val' for u in units[r[nt:nt + nv]]},
               **{u: 'test' for u in units[r[:nt]]}}
    part = {i: of_unit[u] for i, u in zip(pairs.img, pairs.unit)}
    cells = img.map(part).value_counts(normalize=True)
    _PLANS[cohort] = dict(
        cohort=cohort, unit='slide' if why else 'patient', fallback=why, part=part,
        units=len(units), slides=len(pairs),
        n_units={k: sum(p == k for p in of_unit.values()) for k in ('train', 'val', 'test')},
        cell_frac={k: round(float(cells.get(k, 0.0)), 3) for k in ('train', 'val', 'test')})
    return _PLANS[cohort]


def split_masks(cohort, image_ids):
    """Train / val / test boolean masks over a caller's cells, from `split_plan`.

    ONE ID FORMAT (plan F3): pass the value table's own `image_id` column UNCHANGED - it already
    reads 'CRC|reg001_A'. Stage 2 passed it that way; Stages 3 and 6 re-prefixed it into
    'CRC|CRC|reg001_A'. The two orders sorted alike, so the old splits happened to agree - an
    accident, not a guarantee. An id the plan does not know is now an error, never a silent 'train'.
    """
    plan = split_plan(cohort)
    img = pd.Series(np.asarray(image_ids).astype(str))
    part = img.map(plan['part'])
    if part.isna().any():
        bad = img[part.isna()].unique()[:3].tolist()
        raise KeyError(f"{cohort}: {int(part.isna().sum())} cells carry slide ids the split plan "
                       f"does not know, e.g. {bad} - pass the value table's image_id column "
                       f"unchanged (one id format, plan F3)")
    part = part.to_numpy()
    return {k: part == k for k in ('train', 'val', 'test')}


def split_roster():
    """The cohorts a split fingerprint covers: every SPECS cohort with a value table, in SPECS order.
    One definition, used by main() and by warm_for(), so the two can never disagree."""
    return [c for c in SPECS if os.path.exists(full_table(c))]


def split_fp(cohorts=None):
    """Fingerprint of the within-cohort split: the unit rule plus every (slide -> part) assignment.

    A checkpoint records it, the way it records the vocabulary, so a model trained on one split is
    refitted - not loaded and scored - under another. Without it, the slide-split Gate 2 checkpoints
    would load under the patient split and score test patients they had trained on.
    """
    cohorts = tuple(sorted(split_roster() if cohorts is None else cohorts))
    if cohorts not in _FPS:
        s = ';'.join(f"{c}:{split_plan(c)['unit']}:" +
                     ','.join(f'{i}={p}' for i, p in sorted(split_plan(c)['part'].items()))
                     for c in cohorts)
        _FPS[cohorts] = f"patient-v1:{zlib.crc32(s.encode()):08x}"
    return _FPS[cohorts]


_FPS = {}


def split_stale(r):
    """True when a cached fit (any stage) was trained on another within-cohort split, or predates
    the split fingerprint. Stages 3, 6, 7 and 8 check this before reusing a checkpoint and write
    `split_fp=s2.split_fp()` into every new one - the same rule as Stage 2's own cache."""
    return r.get('split_fp') != split_fp()


def split_table(cohorts):
    """Per-cohort split summary for the reports - which unit, which cohorts fell back, and the CELL
    shares that the unit fractions actually produced (patients differ in size, so these drift)."""
    rows = []
    for c in cohorts:
        p = split_plan(c)
        rows.append(dict(cohort=c, unit=p['unit'], fallback_reason=p['fallback'] or '-',
                         units=p['units'], slides=p['slides'],
                         **{f'units_{k}': p['n_units'][k] for k in ('train', 'val', 'test')},
                         **{f'cells_{k}': p['cell_frac'][k] for k in ('train', 'val', 'test')}))
    return pd.DataFrame(rows)

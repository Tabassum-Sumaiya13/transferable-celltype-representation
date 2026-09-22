"""
One generic loader, driven entirely by the specs in config.py.

There is deliberately no cohort branching in this file. If a new dataset needs code changes
here rather than a new dict in config.py, the spec format is missing something - extend the
format, do not add an `if`.

Output per cohort: work/raw/{cohort}.parquet
    cell_id  cohort  image_id  patient_id  x_px  y_px  area_px2  native_label  label_confidence
    + every raw marker column, under its ORIGINAL vendor name (Stage 0b resolves those names
      against HGNC/UniProt, so they must arrive unmangled)

image_id and patient_id are prefixed with the cohort name so they are globally unique - Stage 3
uses slide id as the adversarial domain across all cohorts at once.
"""
import os, re, json
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # runs from any directory
import config
from config import SPECS, STANDARD, path, raw_table


# --------------------------------------------------------------------------- reading helpers
def _columns_of(file, fmt):
    """Column names without loading the data."""
    p = path(file)
    if fmt == 'parquet':
        import pyarrow.parquet as pq
        return list(pq.ParquetFile(p).schema_arrow.names)
    return list(pd.read_csv(p, nrows=0).columns)


def _read(file, fmt, usecols=None):
    p = path(file)
    if fmt == 'parquet':
        return pd.read_parquet(p, columns=usecols)
    return pd.read_csv(p, usecols=usecols)


def _resolve_markers(rule, available):
    """Turn a marker RULE into a concrete column list. Never a hand-written list."""
    kind = rule['kind']
    if kind == 'pattern':
        rx = re.compile(rule['regex'])
        return [c for c in available if rx.search(c)]
    if kind == 'exclude':
        drop = set(rule['cols'])
        return [c for c in available if c not in drop]
    if kind == 'range':
        # a contiguous block of marker columns, named by its first and last member.
        # Needed when markers sit between metadata and one-hot label columns and no
        # pattern separates them (Nolan-lab CODEX exports do exactly this).
        try:
            i, j = available.index(rule['start']), available.index(rule['end'])
        except ValueError as e:
            raise KeyError(f"marker range endpoint not found: {e}")
        if i > j:
            raise ValueError(f"marker range start '{rule['start']}' comes after end '{rule['end']}'")
        return available[i:j + 1]
    if kind == 'file':
        # A bare one-name-per-line list has NO header row - reading it with the default
        # header='infer' silently eats the first marker. Spec must say so explicitly.
        if 'header' in rule and rule['header'] is None:
            df = pd.read_csv(path(rule['path']), header=None, names=['marker'])
            col = 'marker'
        else:
            df = pd.read_csv(path(rule['path']))
            col = rule.get('column', df.columns[0])
        names = [str(x) for x in df[col].dropna().tolist()]
        missing = [n for n in names if n not in available]
        if missing:
            raise KeyError(f"marker file lists columns absent from the data: {missing[:10]}")
        return names
    raise ValueError(f"unknown marker rule kind: {kind}")


def _area(df, rule):
    if rule is None:
        return np.full(len(df), np.nan, 'float64')
    if rule['kind'] == 'column':
        return df[rule['column']].astype('float64').to_numpy()
    if rule['kind'] == 'divide':
        return (df[rule['numerator']] / df[rule['denominator']]).astype('float64').to_numpy()
    raise ValueError(f"unknown area rule: {rule['kind']}")


# --------------------------------------------------------------------------- the loader
def load(cohort, verbose=True):
    spec = SPECS[cohort]
    base = spec['base']
    joins = spec.get('joins', [])
    expr = spec.get('expr', {})

    def say(*a):
        if verbose:
            print(*a, flush=True)

    # ---- which columns does the base table need to carry?
    base_all = _columns_of(base['file'], base['format'])
    need = set(base['cols'].values())
    need |= set(base.get('keep', []))
    for r in (base.get('area'),):
        if r:
            need |= {v for k, v in r.items() if k != 'kind'}
    for j in joins:
        need |= set(j['left_on'])
    if 'left_on' in expr:
        need |= set(expr['left_on'])

    # markers living in the base file itself (no separate expression table)
    markers_in_base = 'file' not in expr
    if markers_in_base:
        markers = _resolve_markers(expr['markers'], base_all)
        need |= set(markers)
    else:
        markers = None

    missing = need - set(base_all)
    if missing:
        raise KeyError(f"[{cohort}] base table is missing columns: {sorted(missing)}")

    say(f"[{cohort}] reading base ({len(need)} cols) ...")
    df = _read(base['file'], base['format'], usecols=sorted(need))

    # ---- joins (patient ids, cell sizes, anything keyed)
    for j in joins:
        want = sorted(set(j['right_on']) | set(j['cols'].values()))
        aux = _read(j['file'], j['format'], usecols=want)
        aux = aux.rename(columns={v: k for k, v in j['cols'].items()})
        before = len(df)
        df = df.merge(aux, left_on=j['left_on'], right_on=j['right_on'], how='left')
        assert len(df) == before, (
            f"[{cohort}] join on {j['left_on']} duplicated rows "
            f"({before} -> {len(df)}); the right table's key is not unique")
        say(f"[{cohort}]   joined {os.path.basename(j['file'])}")

    # ---- geometry / area (a rule may reference a column that arrived via a join)
    area = _area(df, spec.get('base_area') or base.get('area'))

    # ---- expression table, if it lives in a separate file
    if not markers_in_base:
        expr_all = _columns_of(expr['file'], expr['format'])
        markers = _resolve_markers(expr['markers'], expr_all)
        want = sorted(set(expr['right_on']) | set(markers))
        say(f"[{cohort}] reading expression ({len(markers)} markers) ...")
        ex = _read(expr['file'], expr['format'], usecols=want)
        before = len(df)
        df = df.merge(ex, left_on=expr['left_on'], right_on=expr['right_on'], how='left')
        assert len(df) == before, (
            f"[{cohort}] expression join duplicated rows ({before} -> {len(df)})")
        n_null = int(df[markers].isna().all(axis=1).sum())
        if n_null:
            say(f"[{cohort}]   WARNING {n_null:,} cells got no expression row")

    # ---- assemble the standard table
    n = len(df)
    out = pd.DataFrame({'cell_id': np.arange(n, dtype='int64')})
    out['cohort'] = cohort
    out['image_id'] = cohort + '|' + df[base['cols']['image_id']].astype(str)
    if 'patient_id' in base['cols']:
        pid = df[base['cols']['patient_id']]
    elif 'patient_id' in df.columns:
        pid = df['patient_id']
    else:
        pid = pd.Series(['unknown'] * n)
    out['patient_id'] = cohort + '|' + pid.astype(str)
    out['x_px'] = df[base['cols']['x_px']].astype('float64').to_numpy()
    out['y_px'] = df[base['cols']['y_px']].astype('float64').to_numpy()
    out['area_px2'] = area
    out['native_label'] = df[base['cols']['native_label']].astype(str).to_numpy()
    conf = base['cols'].get('label_confidence')
    out['label_confidence'] = (df[conf].astype('float32').to_numpy() if conf
                               else np.ones(n, dtype='float32'))

    # ---- markers keep their original vendor names; guard against collisions
    clash = set(markers) & set(STANDARD)
    if clash:
        raise KeyError(f"[{cohort}] marker names collide with standard columns: {clash}")
    M = df[markers].astype('float32')
    M.columns = markers
    out = pd.concat([out.reset_index(drop=True), M.reset_index(drop=True)], axis=1)

    say(f"[{cohort}] {n:,} cells x {len(markers)} markers | "
        f"{out.image_id.nunique()} images | {out.patient_id.nunique()} patients")
    return out, markers


def build(cohort, verbose=True):
    """Load, write work/raw/{cohort}.parquet and its marker sidecar. Returns the table."""
    out, markers = load(cohort, verbose=verbose)
    out.to_parquet(raw_table(cohort), index=False)
    meta = dict(cohort=cohort, n_cells=int(len(out)), markers=markers,
                n_markers=len(markers),
                n_images=int(out.image_id.nunique()),
                n_patients=int(out.patient_id.nunique()),
                **{k: SPECS[cohort][k] for k in
                   ('tech', 'tissue', 'disease', 'role', 'px_um', 'px_um_source',
                    'arrival', 'citation')})
    with open(os.path.join(config.RAW, f"{cohort}_meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)
    return out


if __name__ == "__main__":
    import sys
    todo = sys.argv[1:] or list(SPECS)
    for c in todo:
        build(c)

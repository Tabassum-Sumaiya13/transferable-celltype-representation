"""
STAGE 0b - marker identity resolution.  Produces GATE 0b.

Every marker column of every cohort is resolved to a STABLE DATABASE IDENTIFIER, not a string.
The canonical key is a TRIPLE:

    (gene_or_complex_id, epitope, modification)

Why a triple and not a gene: CD45, CD45RA and CD45RO are all PTPRC. Resolving to the gene alone
would merge three antibodies that mark opposite cell states. Likewise phospho-S6 and total S6 are
both RPS6. With the triple they stay apart BY CONSTRUCTION, which is why panel/never_merge.csv
becomes an assertion that the resolver works rather than a hand-maintained blacklist.

Resolution order, every step field-scoped (never free text - measured: a free-text HGNC search
for "PD-1" ranks PSMA6 and PSMB6 above the correct PDCD1, so top-hit-wins is banned):

    1. complexes.csv   things that are genuinely not one gene: HLA-DR, pan-keratin, collagen IV,
                       DNA dyes, MIBI elemental channels. Checked first because HGNC would
                       happily return a gene for "CA" or "P".
    2. HGNC  /fetch/symbol  ->  /search/alias_symbol  ->  /search/prev_symbol
       Each tried with and without punctuation; ALL spellings that return something must agree.
    3. UniProt  reviewed + human, exact protein-name match
    4. manual_overrides.csv  whatever a human settled once
    5. review queue          never guessed, never silently dropped

Every HTTP answer is cached in work/api_cache.json, so the second run needs no network and the
output is byte-identical. That matters because Kaggle notebooks have internet off by default.

    python resolve_markers.py             # resolve (uses cache, calls API for misses)
    python resolve_markers.py --offline   # fail rather than call the network; proves reproducibility
"""
import os, re, sys, json, time
import numpy as np
import pandas as pd
import urllib.request, urllib.parse, urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # runs from any directory
import config
from config import RAW, WORK, REPORTS, SPECS

HERE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(HERE, 'declared')
CACHE = os.path.join(WORK, 'api_cache.json')
REGISTRY = os.path.join(WORK, 'marker_registry.csv')

OFFLINE = '--offline' in sys.argv

# ------------------------------------------------------------------ tiny cached HTTP layer
_cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
_stats = dict(hit=0, miss=0)


def _get(url):
    if url in _cache:
        _stats['hit'] += 1
        return _cache[url]
    if OFFLINE:
        raise RuntimeError(f"--offline but {url} is not cached")
    _stats['miss'] += 1
    req = urllib.request.Request(url, headers={'Accept': 'application/json',
                                               'User-Agent': 'cell-annotation-pipeline'})
    for attempt in range(3):
        try:
            body = urllib.request.urlopen(req, timeout=30).read().decode('utf-8')
            _cache[url] = body
            return body
        except urllib.error.HTTPError as e:
            if e.code == 404:
                _cache[url] = ''
                return ''
            time.sleep(1 + attempt)
        except Exception:
            time.sleep(1 + attempt)
    _cache[url] = ''
    return ''


def save_cache():
    with open(CACHE, 'w') as f:
        json.dump(_cache, f, indent=0, sort_keys=True)


# ------------------------------------------------------------------ name normalisation
CYC = re.compile(r':\s*Cyc[_ ]?\d+[_ ]?ch[_ ]?\d+\s*$', re.I)   # CRC / Phillips column suffix
DESC = re.compile(r'\s+-\s+.*$')                                # ' - cytotoxic T cells'


def strip_decoration(raw):
    """'CD8 - cytotoxic T cells:Cyc_3_ch_2' -> 'CD8'.  Cohort-agnostic: both rules are about
    how vendors decorate a column, not about which dataset it came from."""
    s = CYC.sub('', str(raw)).strip()
    s = s.split(':')[0].strip()      # any remaining ':<anything>' tail
    s = DESC.sub('', s).strip()
    return s


# epitope / modification splitting -------------------------------------------------
ISOFORM = re.compile(r'^(CD45)\s*[-_ ]?(RA|RO|RB|RC)$', re.I)
PHOSPHO = re.compile(r'^(?:p|phospho)[-_ ]?(?=[A-Z])(.+)$')
HISTONE = re.compile(r'^(?:H)?H3\s*[-_ ]?(K\d+(?:ac|me\d?))$', re.I)


def split_key(name):
    """Return (core_name, epitope, modification). Epitope and modification are what keep
    same-gene antibodies apart."""
    n = name.strip()

    m = HISTONE.match(n)                       # H3K9ac -> histone H3, mark K9ac
    if m:
        return 'H3', m.group(1), 'none'

    m = ISOFORM.match(n)                       # CD45RA -> CD45, epitope RA
    if m:
        return m.group(1), m.group(2).upper(), 'none'

    m = PHOSPHO.match(n)                       # pSTAT3 / phospho-S6 -> phospho modification
    if m:
        core = m.group(1)
        core = {'S6': 'RPS6'}.get(core, core)  # ribosomal S6 is RPS6; the antibody name is short
        return core, 'pan', 'phospho'

    return n, 'pan', 'none'


def simplify(s):
    """Aggressive key for table lookups: letters and digits only, upper-cased."""
    return re.sub(r'[^A-Z0-9]', '', str(s).upper())


# ------------------------------------------------------------------ local tables
def load_tables():
    # keep_default_na=False is NOT optional here: pandas otherwise reads the literal string
    # "NA" (sodium, a MIBI elemental channel) as a missing value, the row loses its key, and
    # sodium then resolves through HGNC to the gene XK - whose PREVIOUS SYMBOL is "NA".
    # A silently wrong marker id is the worst failure this stage can produce.
    cx = pd.read_csv(os.path.join(PANEL, 'complexes.csv'), keep_default_na=False)
    complexes = {}
    for _, r in cx.iterrows():
        entry = dict(kind=r['kind'], members=r['members'], note=r['note'], name=r['name'])
        keys = [r['name']] + [a for a in str(r.get('aliases', '')).split('|') if a]
        for k in keys:
            complexes[simplify(k)] = entry
    ov_path = os.path.join(PANEL, 'manual_overrides.csv')
    overrides = {}
    if os.path.exists(ov_path):
        ov = pd.read_csv(ov_path, keep_default_na=False)
        for _, r in ov.iterrows():
            overrides[simplify(r['raw_or_core'])] = dict(gene=r['gene'], hgnc=r.get('hgnc_id', ''),
                                                         note=r.get('note', ''))
    return complexes, overrides


# ------------------------------------------------------------------ HGNC / UniProt
HGNC = 'https://rest.genenames.org'
UNIPROT = 'https://rest.uniprot.org'


def _hgnc_docs(body):
    if not body:
        return []
    try:
        d = json.loads(body)
    except Exception:
        return []
    return d.get('response', {}).get('docs', [])


def hgnc_lookup(core):
    """Field-scoped only. Returns (symbol, hgnc_id, how) or (None, None, None).

    Several spellings are tried; every spelling that returns EXACTLY ONE hit must agree, and a
    spelling returning several hits is treated as no answer rather than as a vote. This is what
    stops the measured PD1/PD-1 failure (alias_symbol/PD1 gives 3 candidates)."""
    variants = {core, core.replace('-', ''), core.replace('-', ' '), core.replace(' ', '-'),
                core.replace('_', '-'), core.upper()}
    variants = [v for v in variants if v]
    for field, how in (('fetch/symbol', 'hgnc_symbol'),
                       ('search/alias_symbol', 'hgnc_alias'),
                       ('search/prev_symbol', 'hgnc_prev')):
        agree = {}
        for v in variants:
            docs = _hgnc_docs(_get(f"{HGNC}/{field}/{urllib.parse.quote(v)}"))
            if len(docs) == 1:
                agree[docs[0]['symbol']] = docs[0].get('hgnc_id', '')
        if len(agree) == 1:
            sym, hid = next(iter(agree.items()))
            return sym, hid, how
        if len(agree) > 1:
            return None, None, f'{how}_conflict:{"|".join(sorted(agree))}'
    return None, None, None


def uniprot_lookup(core):
    """Reviewed human entries, exact protein-name query. Accept only an unambiguous winner."""
    q = (f'protein_name:"{core}" AND organism_id:9606 AND reviewed:true')
    url = (f"{UNIPROT}/uniprotkb/search?query={urllib.parse.quote(q)}"
           f"&fields=accession,gene_primary,protein_name&format=json&size=5")
    body = _get(url)
    if not body:
        return None, None, None
    try:
        results = json.loads(body).get('results', [])
    except Exception:
        return None, None, None
    if not results:
        return None, None, None
    genes = []
    for r in results:
        g = r.get('genes') or []
        if g and g[0].get('geneName', {}).get('value'):
            genes.append((g[0]['geneName']['value'], r['primaryAccession']))
    if not genes:
        return None, None, None
    # accept only if the top hit is not tied with a different gene
    if len(genes) > 1 and genes[1][0] != genes[0][0]:
        top, second = genes[0], genes[1]
        # a single clear winner is fine only when the query matched its name exactly
        names = [ (r.get('proteinDescription',{}).get('recommendedName',{})
                    .get('fullName',{}).get('value','') or '') for r in results ]
        if simplify(names[0]) != simplify(core):
            return None, None, f'uniprot_ambiguous:{top[0]}|{second[0]}'
    return genes[0][0], genes[0][1], 'uniprot'


# ------------------------------------------------------------------ the resolver
def resolve(core, complexes, overrides):
    """core -> dict(id, gene, source, kind, note). Never guesses."""
    key = simplify(core)

    if key in complexes:
        c = complexes[key]
        kind = c['kind']
        if kind in ('COMPLEX', 'FAMILY'):
            return dict(id=f"{kind}:{c['name']}", gene=c['members'], source='complex_table',
                        kind='complex', note=c['note'])
        return dict(id=f"{kind}:{c['name']}", gene='', source='complex_table',
                    kind='non_protein', note=c['note'])

    if key in overrides:
        o = overrides[key]
        return dict(id=o['hgnc'] or f"GENE:{o['gene']}", gene=o['gene'],
                    source='manual', kind='protein', note=o['note'])

    sym, hid, how = hgnc_lookup(core)
    if sym:
        return dict(id=hid or f"GENE:{sym}", gene=sym, source=how, kind='protein', note='')

    gene, acc, how2 = uniprot_lookup(core)
    if gene:
        return dict(id=f"UniProt:{acc}", gene=gene, source='uniprot', kind='protein', note='')

    reason = how or how2 or 'no_hit'
    return dict(id='', gene='', source='unresolved', kind='unknown', note=reason)


# ------------------------------------------------------------------ driver
def build():
    complexes, overrides = load_tables()

    rows = []
    for cohort in sorted(SPECS):
        mp = os.path.join(RAW, f"{cohort}_meta.json")
        if not os.path.exists(mp):
            continue
        meta = json.load(open(mp))
        for raw in meta['markers']:
            core0 = strip_decoration(raw)
            core, epitope, mod = split_key(core0)
            r = resolve(core, complexes, overrides)
            triple = (f"{r['id']}|{epitope}|{mod}") if r['id'] else ''
            rows.append(dict(cohort=cohort, raw_column=raw, stripped=core0, core=core,
                             epitope=epitope, modification=mod,
                             resolved_id=r['id'], gene=r['gene'], kind=r['kind'],
                             source=r['source'], note=r['note'], triple=triple))
    R = pd.DataFrame(rows)
    save_cache()
    R.to_csv(REGISTRY, index=False)
    return R


# ------------------------------------------------------------------ gate checks
def gate(R):
    out, A = [], lambda s: out.append(s)
    n = len(R)
    prot = R[R.kind == 'protein']
    cplx = R[R.kind == 'complex']
    nonp = R[R.kind == 'non_protein']
    unres = R[R.kind == 'unknown']
    auto = n - len(unres)

    A("# Stage 0b - Marker identity resolution (GATE 0b)\n")
    A(f"**{n} marker columns** across {R.cohort.nunique()} cohorts, "
      f"**{R.raw_column.nunique()} distinct raw names**.\n")
    A(f"Cache: {_stats['hit']} hits, {_stats['miss']} network calls.\n")

    # --- check 1: coverage
    A("## 1. Resolution coverage\n")
    cov = pd.DataFrame({
        'outcome': ['protein (gene id)', 'complex / family', 'non-protein (dye, element, artefact)',
                    'UNRESOLVED'],
        'n_columns': [len(prot), len(cplx), len(nonp), len(unres)]})
    cov['share_%'] = (100 * cov['n_columns'] / n).round(1)
    A(cov.to_markdown(index=False))
    A(f"\n**Auto-resolved: {auto}/{n} = {100*auto/n:.1f}%** (target >= 90%).\n")
    A("\nBy source:\n")
    A(R.source.value_counts().rename_axis('source').reset_index(name='columns')
      .to_markdown(index=False))

    # --- check 2: beat the verbatim baseline
    A("\n## 2. Does resolution beat naive string matching?\n")
    per_raw = R.groupby('raw_column').cohort.nunique()
    verbatim_shared = int((per_raw >= 2).sum())
    verbatim_total = int(R.raw_column.nunique())
    real = R[R.triple != '']
    per_triple = real.groupby('triple').cohort.nunique()
    resolved_shared = int((per_triple >= 2).sum())
    resolved_total = int(real.triple.nunique())
    A(pd.DataFrame({
        'method': ['verbatim raw name', 'resolved triple'],
        'distinct markers': [verbatim_total, resolved_total],
        'shared by >=2 cohorts': [verbatim_shared, resolved_shared],
    }).to_markdown(index=False))
    A(f"\n**Shared markers: {verbatim_shared} -> {resolved_shared} "
      f"({resolved_shared - verbatim_shared:+d}).**\n")

    # --- check 3: never_merge is an assertion
    A("\n## 3. `never_merge` - must resolve to DIFFERENT triples\n")
    nm = pd.read_csv(os.path.join(PANEL, 'never_merge.csv'))
    look = {}
    for _, r in R.iterrows():
        look.setdefault(simplify(r.raw_column), r.triple)
        look.setdefault(simplify(r.stripped), r.triple)
    res = []
    for _, r in nm.iterrows():
        ta, tb = look.get(simplify(r.a)), look.get(simplify(r.b))
        if ta is None or tb is None:
            st = 'NOT IN DATA'
        elif ta == '' or tb == '':
            st = 'UNRESOLVED'
        else:
            st = 'PASS' if ta != tb else '**FAIL**'
        res.append(dict(a=r.a, b=r.b, triple_a=ta or '-', triple_b=tb or '-', status=st))
    NM = pd.DataFrame(res)
    A(NM.to_markdown(index=False))
    nm_fail = int((NM.status == '**FAIL**').sum())
    A(f"\n**{int((NM.status=='PASS').sum())}/{len(NM)} pass, {nm_fail} fail.**\n")

    # --- check 3b: must_merge - the positive test
    A("\n## 3b. `must_merge` - different spellings that must resolve to the SAME triple\n")
    mm = pd.read_csv(os.path.join(PANEL, 'must_merge.csv'))
    res = []
    for _, r in mm.iterrows():
        members = [m.strip() for m in str(r.members).split('|')]
        ts = {m: look.get(simplify(m)) for m in members}
        present = {m: t for m, t in ts.items() if t is not None}
        uniq = set(present.values())
        # An unresolved member is a FAILURE, not a free pass: the point of this check is that
        # the spellings actually unify, and an empty triple unifies nothing.
        if not present:
            st = 'NOT IN DATA'
        elif '' in uniq:
            st = '**FAIL (unresolved)**'
        elif len(uniq) == 1:
            st = 'PASS'
        else:
            st = '**FAIL**'
        res.append(dict(group=r.group, in_data=len(present), of=len(members),
                        distinct_triples=len(uniq), status=st,
                        triples=' ;; '.join(sorted(uniq)) if uniq else '-'))
    MM = pd.DataFrame(res)
    A(MM.to_markdown(index=False))
    mm_fail = int(MM.status.str.startswith('**FAIL').sum())

    # --- check 3c: a non-protein channel must never come back as a gene
    A("\n## 3c. Non-protein channels must not resolve to genes\n")
    A("A regression guard. `Na` (sodium) once resolved to the gene **XK**, whose *previous* HGNC "
      "symbol is literally `NA` — reached because pandas had parsed the string \"NA\" in the "
      "table as a missing value. A silently wrong marker id is the worst failure this stage can "
      "produce, so it is now asserted rather than eyeballed.\n")
    cx = pd.read_csv(os.path.join(PANEL, 'complexes.csv'), keep_default_na=False)
    nonprot_names = set()
    for _, r in cx[cx.kind.isin(['ELEMENT', 'DYE', 'ARTEFACT'])].iterrows():
        nonprot_names.add(simplify(r['name']))
        nonprot_names |= {simplify(a) for a in str(r['aliases']).split('|') if a}
    leaked = R[R.core.map(simplify).isin(nonprot_names) & (R.kind == 'protein')]
    if len(leaked):
        A(leaked[['cohort', 'raw_column', 'core', 'gene', 'source']].to_markdown(index=False))
    else:
        A(f"_{len(nonprot_names)} declared non-protein names checked; none resolved to a gene._")
    leak_fail = len(leaked)

    # --- check 4: availability matrix
    A("\n## 4. Marker x cohort availability\n")
    piv = (real[real.kind != 'non_protein']
           .assign(one=1).pivot_table(index='triple', columns='cohort', values='one',
                                      aggfunc='max', fill_value=0))
    ncoh = piv.sum(axis=1)
    dist = pd.DataFrame({'present in >= N cohorts': [6, 5, 4, 3, 2],
                         'markers': [int((ncoh >= k).sum()) for k in (6, 5, 4, 3, 2)]})
    A(dist.to_markdown(index=False))
    A(f"\nThe old 5-cohort pipeline's backbone was **19** markers in >=4 cohorts. "
      f"Here: **{int((ncoh>=4).sum())}**.\n")
    A("\nMarkers present in every cohort:\n")
    allc = sorted(real[real.triple.isin(ncoh[ncoh == piv.shape[1]].index)]
                  .drop_duplicates('triple').apply(
        lambda r: r.gene if r.kind == 'protein' else r.resolved_id, axis=1).tolist())
    A(', '.join(f"`{x}`" for x in allc) if allc else '_none_')

    # --- check 5: the review queue
    A("\n## 5. Review queue - unresolved, never guessed\n")
    if len(unres):
        U = (unres.groupby(['stripped', 'core', 'note']).cohort
             .agg(lambda s: ', '.join(sorted(set(s)))).reset_index()
             .rename(columns={'cohort': 'cohorts', 'note': 'why'}))
        A(U.to_markdown(index=False))
    else:
        A("_empty - everything resolved._")

    A("\n## 6. Non-proteins - flagged, not forced\n")
    A("These stay in the table with a `non_protein` kind so Stage 1b can exclude them on "
      "evidence rather than by a hidden hard-coded list.\n")
    if len(nonp):
        A(nonp.groupby(['resolved_id']).raw_column
          .agg(lambda s: ', '.join(sorted(set(s)))).reset_index()
          .rename(columns={'raw_column': 'raw columns'}).to_markdown(index=False))

    A("\n## Verdict\n")
    checks = [
        (f"coverage >= 90%", 100 * auto / n >= 90, f"{100*auto/n:.1f}%"),
        (f"resolution beats verbatim ({verbatim_shared})", resolved_shared > verbatim_shared,
         f"{verbatim_shared} -> {resolved_shared}"),
        ("never_merge: all pairs distinct", nm_fail == 0, f"{nm_fail} failures"),
        ("must_merge: all groups unified", mm_fail == 0, f"{mm_fail} failures"),
        ("no non-protein channel resolved to a gene", leak_fail == 0, f"{leak_fail} leaks"),
    ]
    A(pd.DataFrame([{'check': c, 'result': 'PASS' if ok else 'FAIL', 'value': v}
                    for c, ok, v in checks]).to_markdown(index=False))
    passed = all(ok for _, ok, _ in checks)
    A(f"\n**GATE 0b: {'PASS' if passed else 'FAIL'}**\n")

    path = os.path.join(REPORTS, 'resolve_markers.md')
    open(path, 'w', encoding='utf-8').write("\n".join(out))
    return path, checks, dict(NM=NM, MM=MM, unres=unres, ncoh=ncoh,
                              verbatim=verbatim_shared, resolved=resolved_shared)


if __name__ == "__main__":
    R = build()
    path, checks, extra = gate(R)
    print(f"\n{len(R)} marker columns | cache {_stats['hit']} hits / {_stats['miss']} calls")
    for c, ok, v in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {c}: {v}")
    print(f"\nwrote {REGISTRY}")
    print(f"wrote {path}")

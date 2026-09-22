"""
STAGE 0 - acquire + audit.  Produces GATE 0.

Reads every work/raw/{cohort}.parquet built by loaders.py and answers, per cohort:
  how many cells / images / patients / markers, what the micron-per-pixel is AND where that
  number came from, what state the values arrived in, how many native labels there are, and
  how much is missing.

Then draws one X/Y scatter per cohort from a real slide. That plot is the point of the gate:
a coordinate column read wrongly produces a grid, a line, or a blob - not tissue.

Deliberately does NOT drop anything. Whether `dirt` is a cell and whether `Au` is a protein
are decided later on evidence (Stage 0b's gene resolver, Stage 1b's coherence check), not by
a hand-written blacklist here.

    python load_cohorts.py            # audit whatever is built
    python load_cohorts.py --build    # (re)build every cohort first
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # runs from any directory
import config
from config import SPECS, REJECTED, RAW, REPORTS, FIGURES, raw_table

META = ['cell_id', 'cohort', 'image_id', 'patient_id', 'x_px', 'y_px',
        'area_px2', 'native_label', 'label_confidence']


def built():
    return [c for c in SPECS if os.path.exists(raw_table(c))]


def summarise(cohort):
    meta = json.load(open(os.path.join(RAW, f"{cohort}_meta.json")))
    df = pd.read_parquet(raw_table(cohort), columns=META)
    mk = meta['markers']
    px = meta['px_um']

    x, y = df.x_px.to_numpy(), df.y_px.to_numpy()
    # image extent in microns, averaged over slides - a units sanity check
    ext = (df.groupby('image_id')[['x_px', 'y_px']]
             .agg(lambda s: s.max() - s.min()).mean() * px)

    return dict(
        cohort=cohort, tech=meta['tech'], tissue=meta['tissue'], role=meta['role'],
        cells=len(df), images=meta['n_images'], patients=meta['n_patients'],
        markers=meta['n_markers'],
        px_um=px, px_um_source=meta['px_um_source'], arrival=meta['arrival'],
        labels=int(df.native_label.nunique()),
        median_cells_per_image=int(df.groupby('image_id').size().median()),
        slide_um_x=round(float(ext.x_px)), slide_um_y=round(float(ext.y_px)),
        pct_xy_missing=round(100 * float(np.mean(~np.isfinite(x) | ~np.isfinite(y))), 3),
        pct_area_missing=round(100 * float(df.area_px2.isna().mean()), 2),
        pct_patient_unknown=round(100 * float(df.patient_id.str.endswith(
            ('|nan', '|unknown', '|<NA>', '|-1')).mean()), 2),
        has_confidence=bool(df.label_confidence.nunique() > 1),
        citation=meta['citation'],
    )


def label_table(cohort):
    df = pd.read_parquet(raw_table(cohort), columns=['native_label'])
    vc = df.native_label.value_counts()
    return pd.DataFrame({'native_label': vc.index, 'cells': vc.values,
                         'share_%': (100 * vc.values / len(df)).round(2)})


def tissue_plot(cohort, ax):
    """Scatter one representative slide - the median-sized one, so it is not a lucky pick."""
    df = pd.read_parquet(raw_table(cohort), columns=['image_id', 'x_px', 'y_px', 'native_label'])
    sizes = df.groupby('image_id').size().sort_values()
    img = sizes.index[len(sizes) // 2]
    s = df[df.image_id == img]
    px = SPECS[cohort]['px_um']
    codes = pd.factorize(s.native_label)[0]
    ax.scatter(s.x_px * px, s.y_px * px, c=codes, s=0.6, cmap='tab20', linewidths=0)
    ax.set_aspect('equal')
    ax.set_title(f"{cohort}\n{img.split('|', 1)[1][:28]}  n={len(s):,}", fontsize=8)
    ax.set_xlabel('µm', fontsize=7)
    ax.tick_params(labelsize=6)
    return img, len(s)


def main(do_build=False):
    if do_build:
        import loaders
        for c in SPECS:
            loaders.build(c)

    cohorts = built()
    if not cohorts:
        sys.exit("nothing built yet - run: python loaders.py")

    rows = [summarise(c) for c in cohorts]
    S = pd.DataFrame(rows)
    S = S.sort_values(['role', 'cohort'], ascending=[False, True]).reset_index(drop=True)

    # ---- figures
    n = len(cohorts)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.4 * nrow))
    axes = np.atleast_1d(axes).ravel()
    picked = {}
    for ax, c in zip(axes, S.cohort):
        picked[c] = tissue_plot(c, ax)
    for ax in axes[n:]:
        ax.axis('off')
    fig.suptitle('GATE 0 - one representative slide per cohort (coloured by native label)',
                 fontsize=10)
    fig.tight_layout()
    figp = os.path.join(FIGURES, 'load_cohorts_tissue_check.png')
    fig.savefig(figp, dpi=140)
    plt.close(fig)

    # ---- report
    train = S[S.role == 'train']
    out = []
    A = out.append
    A("# Stage 0 - Acquire + Audit (GATE 0)\n")
    A(f"**{len(S)} cohorts built** - {len(train)} for training, "
      f"{len(S) - len(train)} frozen as held-out test.\n")
    A(f"Totals across training cohorts: **{train.cells.sum():,} cells**, "
      f"**{train.images.sum():,} slides**, **{train.patients.sum():,} patients**.\n")

    A("## The gate table\n")
    show = S[['cohort', 'tech', 'tissue', 'role', 'cells', 'images', 'patients', 'markers',
              'labels', 'px_um', 'arrival', 'median_cells_per_image',
              'slide_um_x', 'slide_um_y']].copy()
    show['cells'] = show.cells.map('{:,}'.format)
    A(show.to_markdown(index=False))

    A("\n## Where each micron-per-pixel comes from\n")
    A("This is the number every distance in the pipeline depends on. An assumed value is a "
      "stated risk, not a fact.\n")
    A(S[['cohort', 'px_um', 'px_um_source']].to_markdown(index=False))

    A("\n## Missing data\n")
    A(S[['cohort', 'pct_xy_missing', 'pct_area_missing', 'pct_patient_unknown',
         'has_confidence']].to_markdown(index=False))

    A("\n## Tissue check\n")
    A(f"![tissue](figures/{os.path.basename(figp)})\n")
    A("Each panel is the **median-sized slide** of that cohort, in microns, coloured by native "
      "label. Read them for shape: real tissue is irregular and clustered. A regular grid, a "
      "straight line, or a uniform blob means the coordinate columns were read wrongly.\n")
    A("| cohort | slide shown | cells |")
    A("|---|---|---|")
    for c, (img, k) in picked.items():
        A(f"| {c} | `{img.split('|', 1)[1]}` | {k:,} |")

    A("\n## Native labels per cohort\n")
    A("Every label is kept. Nothing is dropped here - whether `dirt` is a cell is decided at "
      "Stage 1b on marker evidence, not by a hand-written list at load time.\n")
    for c in S.cohort:
        lt = label_table(c)
        A(f"<details><summary><b>{c}</b> - {len(lt)} native labels</summary>\n")
        A(lt.to_markdown(index=False))
        A("\n</details>\n")

    if REJECTED:
        A("\n## Rejected during Stage 0\n")
        for name, r in REJECTED.items():
            A(f"### {name} - NOT USED\n")
            A(f"*{r['citation']}*\n")
            A(f"**Why:** {r['reason']}\n")
            if r.get('note'):
                A(f"**Note:** {r['note']}\n")

    A("\n## Sources\n")
    A(S[['cohort', 'citation']].to_markdown(index=False))

    path = os.path.join(REPORTS, 'load_cohorts.md')
    open(path, 'w', encoding='utf-8').write("\n".join(out))
    print("\n" + show.to_string(index=False))
    print(f"\nwrote {path}")
    print(f"wrote {figp}")


if __name__ == "__main__":
    main(do_build='--build' in sys.argv)

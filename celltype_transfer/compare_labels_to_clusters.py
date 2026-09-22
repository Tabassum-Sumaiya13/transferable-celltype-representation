"""Plot native cohort labels beside the derived cross-cohort clusters.

Run from the repository root:
    python celltype_transfer/compare_labels_to_clusters.py

The input is the shipped Stage 1b label map. The figure is descriptive: native label names are
not a ground-truth ontology, and the agreement scores should not be read as biological accuracy.
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from sklearn.metrics import normalized_mutual_info_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FIGURES, REPORTS, WORK
from build_label_space import ari


OUTPUT = os.path.join(FIGURES, "compare_labels_to_clusters.png")
REPORT = os.path.join(REPORTS, "compare_labels_to_clusters.md")


def weighted_nmi(left, right, weights):
    """NMI from a weighted contingency table, where each row represents many cells."""
    table = pd.crosstab(left, right, values=weights, aggfunc="sum", dropna=False).fillna(0)
    p = table.to_numpy(float)
    p /= p.sum()
    px, py = p.sum(1), p.sum(0)
    nz = p > 0
    ix, iy = np.nonzero(nz)
    mi = float(np.sum(p[ix, iy] * np.log(p[ix, iy] / (px[ix] * py[iy]))))
    hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = float(-np.sum(py[py > 0] * np.log(py[py > 0])))
    return 2 * mi / max(1e-12, hx + hy)


def load_map(path):
    lm = pd.read_csv(path, keep_default_na=False)
    needed = {"cohort", "label", "cluster", "cluster_name", "n_cells"}
    missing = needed - set(lm.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    lm = lm[lm.cluster >= 0].copy()
    lm["cluster"] = lm.cluster.astype(int)
    lm["n_cells"] = lm.n_cells.astype(float)
    lm["display_label"] = lm.cohort + " | " + lm.label.astype(str)
    return lm.sort_values(["cohort", "cluster", "label"]).reset_index(drop=True)


def plot(lm, output):
    cohorts = list(dict.fromkeys(lm.cohort))
    clusters = sorted(lm.cluster.unique())
    cluster_index = {c: i for i, c in enumerate(clusters)}
    names = (lm.drop_duplicates("cluster").set_index("cluster").cluster_name
             .reindex(clusters).fillna("").tolist())
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(20, len(clusters))))
    cmap = ListedColormap(colors[:len(clusters)])
    norm = BoundaryNorm(np.arange(-0.5, len(clusters) + 0.5), len(clusters))

    fig = plt.figure(figsize=(16, max(8, 0.23 * len(lm) + 3)), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.18, 1], wspace=0.08)
    ax_labels = fig.add_subplot(grid[0, 0])
    ax_clusters = fig.add_subplot(grid[0, 1])

    # Every native label is shown once. Color is the derived cluster assignment.
    y = np.arange(len(lm))
    ax_labels.scatter(np.zeros(len(lm)), y, c=[cluster_index[c] for c in lm.cluster],
                      cmap=cmap, norm=norm, s=42 + 18 * np.log10(lm.n_cells.clip(lower=1)),
                      edgecolors="white", linewidths=0.45, zorder=3)
    ax_labels.set_xlim(-0.45, 0.45)
    ax_labels.set_xticks([0])
    ax_labels.set_xticklabels(["derived cluster"], rotation=35, ha="right")
    ax_labels.set_yticks(y)
    ax_labels.set_yticklabels(lm.display_label, fontsize=7)
    ax_labels.invert_yaxis()
    ax_labels.set_title("All cohort-native labels", loc="left", weight="bold")
    ax_labels.set_xlabel("Marker-derived assignment; point size scales with cell count")
    ax_labels.grid(axis="y", color="#dddddd", linewidth=0.35)
    for i in range(1, len(lm)):
        if lm.cohort.iloc[i] != lm.cohort.iloc[i - 1]:
            ax_labels.axhline(i - 0.5, color="#333333", linewidth=0.8)

    # Cells represented by each cohort-native label, aggregated into the derived clusters.
    table = (lm.pivot_table(index="cohort", columns="cluster", values="n_cells",
                            aggfunc="sum", fill_value=0)
             .reindex(index=cohorts, columns=clusters, fill_value=0))
    image = np.log10(table.to_numpy(float) + 1)
    im = ax_clusters.imshow(image, aspect="auto", cmap="YlGnBu")
    ax_clusters.set_xticks(np.arange(len(clusters)))
    ax_clusters.set_xticklabels([f"C{c}" for c in clusters], rotation=70, ha="right", fontsize=8)
    ax_clusters.set_yticks(np.arange(len(cohorts)))
    ax_clusters.set_yticklabels(cohorts)
    ax_clusters.set_title("Derived clusters across cohorts", loc="left", weight="bold")
    ax_clusters.set_xlabel("Derived cluster; color is log10(cell count + 1)")
    ax_clusters.set_ylabel("Cohort")
    for r in range(len(cohorts)):
        for c in range(len(clusters)):
            value = table.iloc[r, c]
            if value > 0:
                ax_clusters.text(c, r, f"{value / 1e3:.0f}k" if value < 1e6 else f"{value / 1e6:.1f}M",
                                 ha="center", va="center", fontsize=6,
                                 color="white" if image[r, c] > image.max() * 0.55 else "black")
    fig.colorbar(im, ax=ax_clusters, fraction=0.046, pad=0.03, label="log10 cells + 1")

    handles = [plt.Line2D([0], [0], marker="o", color="w", label=f"C{c}: {names[i][:28]}",
                          markerfacecolor=colors[i], markersize=7)
               for i, c in enumerate(clusters)]
    fig.legend(handles=handles, title="Cluster names", loc="outside right center", fontsize=7,
               title_fontsize=8, frameon=False)
    fig.suptitle("Native cohort labels versus the derived cross-cohort label space", weight="bold")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(lm, output, report):
    cell_weights = lm.n_cells.to_numpy(float)
    weighted_ari = ari(lm.label.astype(str), lm.cluster, cell_weights)
    weighted_nmi_score = weighted_nmi(lm.label.astype(str), lm.cluster, cell_weights)
    row_nmi = normalized_mutual_info_score(lm.label.astype(str), lm.cluster)
    cluster_counts = lm.groupby("cluster").agg(labels=("label", "size"), cohorts=("cohort", "nunique"),
                                                cells=("n_cells", "sum"))
    lines = [
        "# Native labels versus derived clusters\n",
        "This is a descriptive comparison of the shipped `work/label_map.csv`. Native names are "
        "cohort-specific annotations, not an external ground truth.\n",
        f"- Cohorts: **{lm.cohort.nunique()}**\n",
        f"- Native labels shown: **{len(lm)}**\n",
        f"- Derived clusters: **{lm.cluster.nunique()}**\n",
        f"- Cell-weighted ARI (native label text vs cluster): **{weighted_ari:.3f}**\n",
        f"- Cell-weighted NMI (native label text vs cluster): **{weighted_nmi_score:.3f}**\n",
        f"- Unweighted row NMI: **{row_nmi:.3f}**\n",
        "\nThe left panel shows every cohort-native label together; color identifies its derived "
        "cluster and point size is proportional to cell count. The right panel shows how each "
        "derived cluster is distributed across cohorts.\n",
        f"![comparison]({os.path.relpath(output, REPORTS).replace(os.sep, '/')})\n",
        "## Cluster summary\n",
        cluster_counts.to_markdown(),
        "\n",
    ]
    with open(report, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default=os.path.join(WORK, "label_map.csv"),
                        help="Stage 1b label map CSV")
    parser.add_argument("--output", default=OUTPUT, help="PNG output path")
    parser.add_argument("--report", default=REPORT, help="Markdown summary path")
    args = parser.parse_args()
    lm = load_map(args.map)
    plot(lm, args.output)
    write_report(lm, args.output, args.report)
    print(f"wrote {args.output}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()

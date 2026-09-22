"""
The whole pipeline, in order, from one place.

    python celltype_transfer/run.py                  # list every step
    python celltype_transfer/run.py cpu              # run the local CPU steps 1-8, in order
    python celltype_transfer/run.py after-gpu        # steps 13-14, once the GPU results are imported
    python celltype_transfer/run.py label-space      # run one step by name
    python celltype_transfer/run.py pretrain --here  # run a GPU step on THIS machine (slow on CPU)

This file only runs the scripts. Each script keeps its own flags and its own checks, and its
docstring lists every other way to call it. Nothing here changes what a step computes.

Steps 9-12 need a GPU and normally run on Kaggle - see gpu/README.md. Without `--here`, asking for
one of them prints how to run it instead of starting a many-hour CPU job by accident.
"""
import os
import sys
import time
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))

# (name, where, [commands], what it writes, gate)
STEPS = [
    ('cohorts', 'cpu', [['load_cohorts.py', '--build']],
     'work/raw/{cohort}.parquet - every cell of every cohort, nothing dropped', '0'),
    # --offline: resolve ONLY from the recorded HGNC/UniProt answers in work/api_cache.json, so a
    # rebuild cannot drift with the live databases. A first-ever build (no cache) runs
    # `resolve_markers.py` without it once, by hand.
    ('markers', 'cpu', [['resolve_markers.py', '--offline']],
     'work/marker_registry.csv - each marker column as a (gene, epitope, modification) triple', '0b'),
    ('values', 'cpu', [['harmonise_values.py'], ['harmonise_values.py', '--bakeoff']],
     'work/values/{cohort}.parquet - per-cohort rank values, 40,000-cell sample', '1'),
    # --expect: the SHIPPED 7-cohort label space is scored against the v4 cases, and its waived
    # failures flag clusters `unreliable` (reports/build_label_space.md names the file). The
    # script's own default is still the v1 file, kept so the old 25-cluster result reproduces.
    ('label-space', 'cpu', [['build_label_space.py', '--expect', 'gate1b_v4_expect.csv']],
     'work/label_map.csv, prototypes.npy, label_graph.json, label_signatures.npz', '1b'),
    ('vocabulary', 'cpu', [['build_marker_vocabulary.py']],
     'work/panel.json, work/values/{cohort}_full.parquet, marker_dynamic_range.csv', '-'),
    # AFTER vocabulary: it keeps only the cells the {cohort}_full.parquet tables hold
    ('confidence', 'cpu', [['build_label_confidence.py']],
     'work/label_conf/{cohort}.parquet - per-cell label confidence (only UPMC has a real one)', '-'),
    ('fold-spaces', 'cpu', [['build_fold_label_spaces.py', '--folds']],
     'work/spaces/ - one label space per LOCO / LOTO fold, built without the held-out cohort', '1b-fold'),
    ('neighbours', 'cpu', [['build_neighbour_graph.py']],
     'work/values/{cohort}_neighbours.npz - 15 nearest neighbours, from the FULL raw tables', '4 (checks 6-7)'),
    ('pretrain', 'gpu', [['pretrain_masked_markers.py', '--check'], ['pretrain_masked_markers.py', '--loto']],
     'work/ckpt/pretrain_*.pt, stage2_arm in panel.json - masked-marker pretraining', '2'),
    ('classifier', 'gpu', [['train_prototype_classifier.py', '--ablate-losses']],
     'work/ckpt/classifier_*.pt - the headline LOCO classifier and its loss ablation', '6'),
    ('encoder', 'gpu', [['train_adversarial_encoder.py', '--lambda-sweep']],
     'work/ckpt/encoder_*.pt - the batch adversary, lambda sweep', '3'),
    ('spatial', 'gpu', [['train_spatial_context.py', '--gate']],
     'work/ckpt/spatial_*.pt - does the spatial neighbourhood help?', '4'),
    ('external', 'cpu', [['compare_external_baseline.py', '--gate']],
     'work/ckpt/external_*.pt - the published MAPS method on the same folds', '10'),
    ('clusters', 'cpu', [['compare_labels_to_clusters.py']],
     'reports/compare_labels_to_clusters.md - native labels vs unsupervised clusters', '-'),
]
GROUPS = {'cpu': [s[0] for s in STEPS[:8]], 'after-gpu': ['external', 'clusters']}


def listing():
    print(__doc__.strip().split('\n\n')[0], '\n')
    print(f"{'#':>2}  {'step':12} {'where':5} {'gate':14} writes")
    for i, (name, where, cmds, out, gate) in enumerate(STEPS, 1):
        print(f"{i:>2}  {name:12} {where:5} {gate:14} {out}")
        for c in cmds:
            print(f"{'':>4}{'':12}   python celltype_transfer/{' '.join(c)}")


def run_step(name, here):
    step = next((s for s in STEPS if s[0] == name), None)
    if step is None:
        sys.exit(f"no step {name!r}. Steps: {', '.join(s[0] for s in STEPS)}, or: "
                 f"{', '.join(GROUPS)}")
    _, where, cmds, out, gate = step
    if where == 'gpu' and not here:
        print(f"step {name} needs a GPU and runs on Kaggle - see celltype_transfer/gpu/README.md.\n"
              f"To run it on this machine anyway: python celltype_transfer/run.py {name} --here")
        return
    for c in cmds:
        print(f"\n{'=' * 78}\n[{name}] python celltype_transfer/{' '.join(c)}\n{'=' * 78}", flush=True)
        t0 = time.time()
        rc = subprocess.call([sys.executable, '-u', os.path.join(HERE, c[0])] + c[1:], cwd=HERE)
        print(f"[{name}] exit {rc} after {(time.time() - t0) / 60:.1f} min", flush=True)
        if rc != 0:
            sys.exit(f"step {name} failed - later steps depend on it, so the run stops here")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    here = '--here' in sys.argv
    if not args:
        listing()
        return
    for a in args:
        for name in GROUPS.get(a, [a]):
            run_step(name, here)


if __name__ == '__main__':
    main()

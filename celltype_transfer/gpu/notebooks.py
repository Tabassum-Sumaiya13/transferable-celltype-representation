"""
Write the two Kaggle notebooks for the fresh GPU run.

    python celltype_transfer/gpu/notebooks.py            # writes gpu_session1.ipynb + gpu_session2.ipynb
    python celltype_transfer/gpu/notebooks.py --print    # prints the cells instead

Kaggle settings for both: Accelerator = GPU T4 x2, Internet = OFF. Run as a saved version
(Save Version -> Save & Run All), so the run survives closing the browser.

SESSION 1  (upload: make_gpu_bundle.py --session1, dataset cta-session1)
    Gate 2  pretrain_masked_markers --check     24 fits   one GPU    old run: 54 min on a T4
            pretrain_masked_markers --loto       2 fits   one GPU    ~10 min
    Gate 6  train_prototype_classifier           34 fits   two GPUs   old run: 172 min for 22 fits
    Gate 3  train_adversarial_encoder            35 fits   two GPUs   old run: 105 min on one GPU
SESSION 2  (upload: make_gpu_bundle.py --session2, dataset cta-session2 - needs session 1 imported)
    Gate 4  train_spatial_context --gate         30 fits   two GPUs   old run: 316 min on one GPU

WHY TWO SESSIONS, not one. Spatial training warm-starts from session 1's pretraining models, and a
Kaggle commit that runs past 12 h can lose its output. Two sessions of about 5 h and 3 h each stay
well inside the limit.

HOW TWO GPUS ARE USED. The training scripts use one GPU each. So each gate starts two worker
processes, one per GPU (CUDA_VISIBLE_DEVICES), each holding out a different set of folds (--folds).
Every fit is exactly the computation it would be on one GPU - only the wall time halves. Each fit
caches to work/ckpt/ when it finishes. Then ONE normal run over all folds loads every fit from that
cache and writes the gate report. If a worker died, that normal run trains the missing fits itself,
so nothing is silently skipped - it only takes longer. With one GPU the workers are skipped.

ZIP BEFORE DISPLAY. After every gate the results are zipped FIRST, and only then shown. An earlier
run crashed in a display cell before zipping and 105 minutes of finished fits never left Kaggle.
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))

# fold groups for the two workers, balanced by fit count (UPMC has no confidence-off fit and
# carries spatial's two pixel-rescale fits, so it sits in the smaller group where that helps)
GROUPS = {
    'classifier': ('UPMC,CRC,Keren,ferguson', 'Phillips,Danenberg,Sorin'),    # 19 / 15 fits
    'encoder':    ('UPMC,CRC,Keren,ferguson', 'Phillips,Danenberg,Sorin'),    # 20 / 15 fits
    'spatial':    ('UPMC,CRC,Keren', 'ferguson,Phillips,Danenberg,Sorin'),    # 14 / 16 fits
}


def setup_cell(session):
    return rf'''
# SETUP - copy the upload to a writable folder, check the GPUs.
import os, shutil, json
SRC = "/kaggle/input/cta-session{session}"   # <-- your dataset's slug; change here if it differs
DST = "/kaggle/working/proj"

assert os.path.isdir(SRC), f"dataset not attached at {{SRC}} - check the Add Input panel"
# Kaggle sometimes nests an uploaded folder one level deeper. Find the folder holding MANIFEST.json.
if not os.path.exists(f"{{SRC}}/MANIFEST.json"):
    sub = [d for d in os.listdir(SRC) if os.path.exists(f"{{SRC}}/{{d}}/MANIFEST.json")]
    assert sub, f"no MANIFEST.json under {{SRC}} - was it built with make_gpu_bundle.py --session{session}?"
    SRC = f"{{SRC}}/{{sub[0]}}"
if os.path.isdir(DST):
    shutil.rmtree(DST)
shutil.copytree(SRC, DST)
for d in ("work/ckpt", "reports/figures", "logs"):
    os.makedirs(f"{{DST}}/{{d}}", exist_ok=True)

man = json.load(open(f"{{DST}}/MANIFEST.json"))
assert man["session"] == "{session}", f"this upload is for session {{man['session']}}"
print("session", man["session"], "| stage2_arm", man["stage2_arm"], "| cohorts",
      man["train_cohorts"], "| data", round(man["total_bytes"] / 1e6, 1), "MB")

import torch
N_GPU = torch.cuda.device_count()
print("torch", torch.__version__, "|", N_GPU, "GPU(s):",
      [torch.cuda.get_device_name(i) for i in range(N_GPU)])
assert N_GPU >= 1, "no GPU - set Accelerator to GPU T4 x2 in the notebook settings"
if N_GPU < 2:
    print("ONE GPU ONLY - every gate runs as a single process. Expect about twice the time.")
'''


INTEGRITY = r'''
# INTEGRITY - the upload arrived intact, and the fingerprints match the local build.
import hashlib, json, os, sys
DST = "/kaggle/working/proj"
EXPECT_VOCAB = "109:66147d20"          # marker vocabulary, measured locally
EXPECT_SPLIT = "patient-v1:015916c8"   # the patient split (plan F3)

def sha(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(n)
            if not b: break
            h.update(b)
    return h.hexdigest()[:16]

man = json.load(open(f"{DST}/MANIFEST.json"))
bad = [f["path"] for f in man["files"]
       if not os.path.exists(os.path.join(DST, f["path"]))
       or os.path.getsize(os.path.join(DST, f["path"])) != f["bytes"]
       or sha(os.path.join(DST, f["path"])) != f["sha256_16"]]
print("checked", len(man["files"]), "files ->", "ALL OK" if not bad else f"PROBLEMS {bad}")
assert not bad

sys.path.insert(0, f"{DST}/celltype_transfer")
import build_marker_vocabulary as vocab, splits, train_adversarial_encoder as adversarial
cohorts = adversarial.available()
print("cohorts:", cohorts)
assert len(cohorts) == 7, f"expected 7 cohorts, got {cohorts}"
assert vocab.panel_fp() == EXPECT_VOCAB, f"vocabulary {vocab.panel_fp()} != {EXPECT_VOCAB}"
assert splits.split_fp() == EXPECT_SPLIT, f"split {splits.split_fp()} != {EXPECT_SPLIT}"
# every fold's label space was built WITHOUT its held-out cohort, and is hash-checked on load
for h in cohorts:
    L, pt, s = adversarial.space_for(h, "fold")
    print(f"  {h:9} {L['n']:>2} clusters, {len(L['novel'])} NOVEL held-out labels | {s['cut_rule']}")
print("all 7 fold-local label spaces OK")
'''


HELPERS = r'''
# HELPERS used by every gate below.
import os, glob, shutil, subprocess, time, zipfile
import torch
DST = "/kaggle/working/proj"
CODE = f"{DST}/celltype_transfer"
N_GPU = torch.cuda.device_count()
NOISE = ("duplicate reagent",)

def stream(args, log, gpu=None):
    """Run one script to the end, printing its output live and keeping a copy in logs/."""
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    t0 = time.time()
    with open(f"{DST}/logs/{log}.log", "w") as f:
        p = subprocess.Popen(["python", "-u"] + args, cwd=CODE, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            f.write(line)
            if not any(n in line for n in NOISE):
                print(line, end="")
        p.wait()
    print(f"\n[{log}] exit {p.returncode} after {(time.time() - t0) / 60:.1f} min")
    return p.returncode

def two_workers(args, groups, log):
    """One process per GPU, each on its own folds. Output goes to logs/ and a short progress line
    is printed every 10 minutes. Worker exit codes are REPORTED, not asserted: the normal run that
    follows re-checks every fit and trains any that are missing."""
    if N_GPU < 2:
        print("one GPU - skipping the workers; the normal run below does every fit")
        return
    procs, t0 = [], time.time()
    for gpu, folds in enumerate(groups):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        f = open(f"{DST}/logs/{log}_gpu{gpu}.log", "w")
        procs.append((gpu, folds, f, subprocess.Popen(
            ["python", "-u"] + args + ["--folds", folds], cwd=CODE, env=env,
            stdout=f, stderr=subprocess.STDOUT, text=True)))
        print(f"worker gpu{gpu}: folds {folds}")
    while any(p.poll() is None for *_, p in procs):
        time.sleep(600)
        for gpu, folds, f, p in procs:
            f.flush()
            txt = open(f.name).read()
            print(f"  {(time.time() - t0) / 60:6.1f} min  gpu{gpu}  {txt.count('[done ')} fits done  "
                  f"{'running' if p.poll() is None else 'exit ' + str(p.returncode)}")
    for gpu, folds, f, p in procs:
        f.close()
        txt = open(f.name).read()
        print(f"\n===== worker gpu{gpu} ({folds}) exit {p.returncode} - last lines:")
        print("\n".join(l for l in txt.splitlines()[-25:] if not any(n in l for n in NOISE)))
    print(f"\nworkers finished after {(time.time() - t0) / 60:.1f} min")

def fits(prefix):
    return sorted(p for p in glob.glob(f"{DST}/work/ckpt/{prefix}*.pt")
                  if "quick_" not in os.path.basename(p) and not p.endswith("sweep.pt"))

def drop_quick():
    """Smoke-test output must never travel home."""
    for p in (glob.glob(f"{DST}/work/ckpt/*quick_*.pt") + glob.glob(f"{DST}/reports/*_quick.md")
              + glob.glob(f"{DST}/reports/predictions/quick_*")):
        os.remove(p)

def collect(name):
    """Zip everything worth taking home into /kaggle/working/{name}.zip. Runs BEFORE any display,
    and never raises before the zip is written."""
    drop_quick()
    out = f"/kaggle/working/{name}.zip"
    tmp = out + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(glob.glob(f"{DST}/work/ckpt/*.pt")):
            z.write(p, os.path.relpath(p, DST))
        z.write(f"{DST}/work/panel.json", "work/panel.json")
        for root in ("reports", "logs"):
            for r, _, fs in os.walk(f"{DST}/{root}"):
                for fn in fs:
                    z.write(os.path.join(r, fn), os.path.relpath(os.path.join(r, fn), DST))
    os.replace(tmp, out)
    print(f"wrote {out}  {os.path.getsize(out) / 1e6:.1f} MB")

def show(pattern):
    from IPython.display import Markdown, display
    reps = sorted(glob.glob(f"{DST}/reports/{pattern}"))
    print("reports:", [os.path.basename(r) for r in reps] or "NONE")
    for r in reps:
        display(Markdown(open(r, encoding="utf-8").read()))
'''


# ------------------------------------------------------------------------------------ session 1
S1_SMOKE2 = r'''
# SMOKE TEST, pretraining - a few minutes, 2 epochs on a tiny draw. Proves the GPU path; scores
# nothing and does NOT write stage2_arm. Its quick_ checkpoints and report are deleted.
t0 = time.time()
r = subprocess.run(["python", "-u", "pretrain_masked_markers.py", "--check", "--quick"],
                   cwd=CODE, capture_output=True, text=True)
print(r.stdout[-5000:]); print(r.stderr[-2000:])
assert r.returncode == 0, "smoke test failed - do not start the real run"
assert "device: cuda" in r.stdout, "ran on CPU"
assert "SMOKE RUN - not a gate result" in r.stdout
drop_quick()
print(f"\nsmoke test OK in {(time.time() - t0) / 60:.1f} min")
'''

S1_GATE2 = r'''
# GATE 2 - masked-marker pretraining, 24 fits, then the 2 leave-one-tissue-out fits. One GPU: the
# old run took 54 min, not worth splitting. --check decides the arm (D-30) and writes it into
# work/panel.json; --loto reads that arm, so the order matters. Re-running this cell resumes.
assert stream(["pretrain_masked_markers.py", "--check"], "gate2") == 0, "Gate 2 did not finish - re-run to resume"
assert stream(["pretrain_masked_markers.py", "--loto"], "gate2_loto") == 0, "LOTO did not finish - re-run to resume"
collect("session1_after_gate2")
import json
pj = json.load(open(f"{DST}/work/panel.json"))
print("\nstage2_arm:", pj.get("stage2_arm"), "\n", pj.get("stage2_arm_note", ""))
if pj.get("stage2_arm") != "absent":
    print("NOTE: the old run shipped 'absent' (Arm B). A different arm is a real result of this run, "
          "not an error - but report it.")
n = len(fits("pretrain_"))
print(f"{n} pretraining checkpoints (expected 26)")
assert n == 26, f"expected 26 pretraining checkpoints, found {n}"
show("pretrain_*.md")
'''

S1_WARM = r'''
# LEAK GUARD - every LOCO fold now has a warm start that never saw its held-out cohort.
import importlib, sys
sys.path.insert(0, CODE)
import pretrain_masked_markers as pretrain, splits
importlib.reload(pretrain)
for h in cohorts:
    rest = [c for c in cohorts if c != h]
    _, ck, seen = pretrain.warm_for(rest, verbose=False)
    assert h not in seen, f"fold {h}: warm start {ck} saw the held-out cohort - STOP"
    print(f"  {h:9} warm start {ck}")
bad = [os.path.basename(p) for p in fits("pretrain_")
       if torch.load(p, weights_only=False, map_location="cpu").get("split_fp") != splits.split_fp()]
assert not bad, f"pretraining checkpoints not on the patient split: {bad}"
print("all 7 folds: leak-free warm start, patient split")
'''

S1_SMOKE36 = r'''
# SMOKE TEST, classifier + encoder - 2 epochs on a tiny draw, scores nothing. Sorin is a
# shipped-rule fold, UPMC a nearest-feasible one; Keren at lambda 0.3 exercises the adversary's
# gradient-reversal layer. quick_ output is deleted afterwards.
for name, cmd, dev in [
        ("classifier", ["train_prototype_classifier.py", "--ablate-losses", "--quick",
                        "--folds", "Sorin,UPMC"], "device     : cuda"),
        ("encoder", ["train_adversarial_encoder.py", "--lambda-sweep", "--quick",
                     "--lambdas", "0.0,0.3", "--folds", "Keren"], "device: cuda")]:
    t0 = time.time()
    r = subprocess.run(["python", "-u"] + cmd, cwd=CODE, capture_output=True, text=True)
    out = "\n".join(l for l in r.stdout.splitlines() if not any(n in l for n in NOISE))
    print(f"===== {name} smoke ({(time.time() - t0) / 60:.1f} min)\n{out[-4000:]}\n{r.stderr[-2000:]}")
    assert r.returncode == 0, f"{name} smoke test failed - do not start the real run"
    assert dev in r.stdout, f"{name} ran on CPU"
    assert "warm start: pretrain_" in r.stdout or "warm start: fold-local" in r.stdout, \
        f"{name}: no fold-local warm start in the log"
    assert "guard=COLLAPSE" not in r.stdout, "prototypes collapsed at init - check D-42"
drop_quick()
print("\nsmoke tests OK")
'''

S1_GATE6 = rf'''
# GATE 6 - the headline. 34 fits: proto3 / proto2 / linear / proto2adv x 7 folds, plus the
# confidence-off arm on the 6 folds where UPMC (the one cohort with confidences) trains
# (check 6-v2, declared 2026-09-17). Runs BEFORE Gate 3 so the headline is safe if time runs out.
two_workers(["train_prototype_classifier.py", "--ablate-losses"], {GROUPS['classifier']!r}, "gate6")
rc = stream(["train_prototype_classifier.py", "--ablate-losses"], "gate6_report")   # cached: report only
collect("session1_after_gate6")
if os.path.exists(f"/kaggle/working/session1_after_gate2.zip"):
    os.remove(f"/kaggle/working/session1_after_gate2.zip")
assert rc == 0, "Gate 6 report run failed - re-run this cell; finished fits load from cache"
n = len(fits("classifier_"))
print(f"{{n}} classifier fits (expected 34)")
assert n == 34, f"expected 34 classifier fits, found {{n}}"
show("train_prototype_classifier*.md")
'''

S1_GATE3 = rf'''
# GATE 3 - the adversary lambda sweep, 5 lambdas x 7 folds = 35 fits.
two_workers(["train_adversarial_encoder.py", "--lambda-sweep"], {GROUPS['encoder']!r}, "gate3")
rc = stream(["train_adversarial_encoder.py", "--lambda-sweep"], "gate3_report")     # cached: report only
collect("session1_results")
if os.path.exists(f"/kaggle/working/session1_after_gate6.zip"):
    os.remove(f"/kaggle/working/session1_after_gate6.zip")
assert rc == 0, "Gate 3 report run failed - re-run this cell; finished fits load from cache"
n = len(fits("encoder_"))
print(f"{{n}} encoder fits (expected 35)")
assert n == 35, f"expected 35 encoder fits, found {{n}}"
show("train_adversarial_encoder*.md")
print("\nDONE. Download /kaggle/working/session1_results.zip, then locally:\n"
      "  python celltype_transfer/gpu/import_gpu_results.py session1_results.zip")
'''

# ------------------------------------------------------------------------------------ session 2
S2_CHECK = r'''
# SPATIAL INPUTS - session 1's warm starts, the pinned roster, and the neighbour graph.
import sys, pandas as pd
sys.path.insert(0, CODE)
import pretrain_masked_markers as pretrain, splits
import build_neighbour_graph as graph
import train_spatial_context as spatial
spatial.assert_roster()                     # gate 4 check 0
for h in cohorts:
    rest = [c for c in cohorts if c != h]
    _, ck, seen = pretrain.warm_for(rest, verbose=False)
    assert h not in seen, f"fold {h}: warm start {ck} saw the held-out cohort - STOP"
    print(f"  {h:9} warm start {ck}")
# Checks 6 and 7 were asserted at BUILD time, on the full raw tables that only exist locally.
# They are re-PRINTED here from the summary that travelled with the sidecars.
g = pd.read_csv(f"{DST}/work/neighbour_graph_summary.csv")
miss = [c for c in cohorts if not os.path.exists(f"{DST}/work/values/{c}_neighbours.npz")]
assert not miss, f"no neighbour sidecar for {miss}"
out = g[(g.median_um < graph.DIST_LO) | (g.median_um > graph.DIST_HI)]
print(g[["cohort", "cells", "markers", "images", "median_um", "homotypic", "no_nbr"]].to_string(index=False))
print(f"\ncheck 7 {'PASS' if out.empty else 'FAIL'} - median edge inside "
      f"{graph.DIST_LO}-{graph.DIST_HI} um for {len(g) - len(out)}/{len(g)} cohorts")
assert out.empty, f"median edge out of range - graph built on the subsample? {out.cohort.tolist()}"
'''

S2_SMOKE = r'''
# SMOKE TEST - 2 epochs, all four arms, Keren (narrowest panel overlap, so it exercises [ABSENT]
# on the neighbourhood profile as well). Scores nothing; quick_ output is deleted.
t0 = time.time()
r = subprocess.run(["python", "-u", "train_spatial_context.py", "--gate", "--quick", "--folds", "Keren"],
                   cwd=CODE, capture_output=True, text=True)
out = "\n".join(l for l in r.stdout.splitlines() if not any(n in l for n in NOISE))
print(f"===== spatial smoke ({(time.time() - t0) / 60:.1f} min)\n{out[-6000:]}\n{r.stderr[-2000:]}")
assert r.returncode == 0, "smoke test failed - do not start the real run"
assert "device     : cuda" in r.stdout, "ran on CPU"
assert "warm start: pretrain_" in r.stdout, "no fold-local warm start in the log"
# a prototype fallback would start every arm from random prototypes and make check 1 unreadable
assert "random init instead" not in r.stdout, "prototypes fell back to random - fold space not found"
for a in ("arm cell", "arm neigh", "arm shuffle", "arm ctx"):
    assert a in r.stdout, f"{a} never ran"
drop_quick()
print("\nsmoke test OK - all four arms ran on GPU")
'''

S2_GATE4 = rf'''
# GATE 4 - 4 arms x 7 folds + 2 UPMC pixel-rescale fits = 30 fits. Each worker runs its arms in
# the declared drop order (ctx last). Checkpoints now carry their weights, so they can be
# re-scored later on CPU.
two_workers(["train_spatial_context.py", "--gate"], {GROUPS['spatial']!r}, "gate4")
rc = stream(["train_spatial_context.py", "--gate"], "gate4_report")                  # cached: report only
collect("session2_results")
assert rc == 0, "Gate 4 report run failed - re-run this cell; finished fits load from cache"
n = len(fits("spatial_"))                   # px-rescale fits are spatial_px*.pt
print(f"{{n}} spatial fits (expected 30)")
assert n == 30, f"expected 30 spatial fits, found {{n}}"
show("train_spatial_context*.md")
print("\nDONE. Download /kaggle/working/session2_results.zip, then locally:\n"
      "  python celltype_transfer/gpu/import_gpu_results.py session2_results.zip")
'''

NOTEBOOKS = {
    'gpu_session1.ipynb': [setup_cell(1), INTEGRITY, HELPERS, S1_SMOKE2, S1_GATE2, S1_WARM,
                           S1_SMOKE36, S1_GATE6, S1_GATE3],
    'gpu_session2.ipynb': [setup_cell(2), INTEGRITY, HELPERS, S2_CHECK, S2_SMOKE, S2_GATE4],
}


def notebook(cells):
    return dict(cells=[dict(cell_type="code", metadata={}, source=c.strip().splitlines(True),
                            outputs=[], execution_count=None) for c in cells],
                metadata=dict(kernelspec=dict(name="python3", display_name="Python 3",
                                              language="python"),
                              language_info=dict(name="python")),
                nbformat=4, nbformat_minor=5)


def main():
    for name, cells in NOTEBOOKS.items():
        for i, c in enumerate(cells, 1):
            compile(c, f'{name}:cell{i}', 'exec')          # a syntax error fails HERE, not on Kaggle
        if '--print' in sys.argv:
            for i, c in enumerate(cells, 1):
                print(f"\n{'=' * 78}\n{name}  CELL {i}\n{'=' * 78}\n{c.strip()}")
            continue
        p = os.path.join(HERE, name)
        json.dump(notebook(cells), open(p, 'w', encoding='utf-8'), indent=1)
        print(f"wrote {p}  ({len(cells)} cells)")


if __name__ == '__main__':
    main()

"""v2 end of Stage 1: inclusion counts, minimal width h*, Section 3.4 check, and the freeze
(v2_rules.md, Sections 0 and 3). Uses TRAINING ERROR ONLY. Written by Claude, 19 Oct 2026.

python src/stage1.py
Writes results_v2/stage1.csv and results_v2/stage1_frozen.json (SHA-256 of every model file + h*).
analyze_v2.py refuses to run if a model file changed after the freeze.
"""
import argparse, csv, hashlib, json, pathlib, pickle, datetime

ap = argparse.ArgumentParser()
ap.add_argument("--in-dir", default="results_v2")
ap.add_argument("--min-included", type=int, default=10)       # Section 3.3
ap.add_argument("--max-train-err", type=float, default=0.02)  # inclusion rule (change only for made-up smoke tests)
a = ap.parse_args()
d = pathlib.Path(a.in_dir)

rows, frozen = [], {"frozen_at": datetime.datetime.now().isoformat(timespec="seconds"), "tasks": {}}
for f in sorted(d.glob("models_T*.pkl")):
    D = pickle.load(open(f, "rb"))
    task, runs = D["task"], D["runs"]
    widths = sorted({r["width"] for r in runs})
    hstar = None
    for h in widths:
        e = [r["train_err"] for r in runs if r["width"] == h]
        inc = {t: sum(x <= t for x in e) for t in (0.01, 0.02, 0.05)}
        rows.append(dict(task=task, width=h, trained=len(e), inc_1pct=inc[0.01], inc_2pct=inc[0.02], inc_5pct=inc[0.05],
                         steps=D["config"]["steps"]))
        if hstar is None and sum(x <= a.max_train_err for x in e) >= a.min_included:
            hstar = h
    frozen["tasks"][task] = dict(file=f.name, sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                                 hstar=hstar, steps=D["config"]["steps"])
    msg = f"h* = {hstar}" if hstar is not None else \
        ("NO WIDTH REACHED 10 INCLUDED -> Section 3.4: retrain this task once with --steps 10000"
         if D["config"]["steps"] == 3000 else "no minimal width found in range (Section 3.4 already used)")
    print(f"{task}: {msg}")

with open(d / "stage1.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
for r in rows:
    print(f"  {r['task']} width {r['width']}: {r['inc_2pct']}/{r['trained']} included at 2%")
json.dump(frozen, open(d / "stage1_frozen.json", "w"), indent=1)
print(f"wrote {d}/stage1.csv and stage1_frozen.json  (commit both before Stage 2)")

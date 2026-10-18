"""v2 Stage 2 for one task (v2_rules.md, Sections 4, 5, 7). Written by Claude, 19 Oct 2026.

python src/analyze_v2.py --task T3                # after stage1.py and the freeze commit
python src/analyze_v2.py --task T3 --all-runs     # adds the exploratory all-runs headline (Section 7)

Writes results_v2/<task>/: pairs.csv, mechanism.csv, excluded.csv, excl_vs_incl.csv, above_hstar.csv
(and allruns.csv with --all-runs). Barriers are fractions; x 100 = points.
"""
import argparse, csv, hashlib, json, pathlib, pickle, time
import numpy as np
from core import weight_match, lines, match_lines
from b1 import irreducibility
from extras import count_flips
from exact_search import exact_search
from measures import all_measures, rel_distance
from mechanism import mechanism
from excluded import classify
from polygon import edges_raw, coffee_edges_raw, to_standardized

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--in-dir", default="results_v2")
ap.add_argument("--max-train-err", type=float, default=0.02)
ap.add_argument("--exact-max", type=int, default=6)
ap.add_argument("--all-runs", action="store_true")
a = ap.parse_args()

d = pathlib.Path(a.in_dir)
frozen = json.load(open(d / "stage1_frozen.json"))["tasks"][a.task]
f = d / frozen["file"]
if hashlib.sha256(f.read_bytes()).hexdigest() != frozen["sha256"]:
    raise SystemExit(f"{f} changed after the Stage 1 freeze; refusing to run (Section 0).")
hstar = frozen["hstar"]
out = d / a.task
if a.max_train_err != 0.02:      # pre-registered inclusion sensitivity (1%, 5%): h* re-derived at that threshold
    _D = pickle.load(open(f, "rb"))   # from TRAINING ERROR only, same rule as stage1.py (DEVIATIONS.md, D7)
    hstar = next((h for h in sorted({r["width"] for r in _D["runs"]})
                  if sum(r["train_err"] <= a.max_train_err for r in _D["runs"] if r["width"] == h) >= 10), None)
    out = d / f"{a.task}_inc{round(a.max_train_err * 100)}"
out.mkdir(parents=True, exist_ok=True)
json.dump({"hstar": hstar, "max_train_err": a.max_train_err}, open(out / "hstar.json", "w"))
if hstar is None:
    (out / "NO_HSTAR.txt").write_text("No width reached 10 included networks; no barrier analysis (Section 3.4).\n")
    raise SystemExit(f"{a.task}: no h*, nothing to analyze.")

D = pickle.load(open(f, "rb"))
Xtr, ytr, Xte, yte = (D["data"][k] for k in ("Xtr", "ytr", "Xte", "yte"))
edges = to_standardized(*(coffee_edges_raw() if a.task == "T0" else edges_raw(np.asarray(D["vertices"]))), D["mu"], D["sd"])


def write(name, rows):
    if not rows:
        (out / name).write_text("none\n")      # e.g. no excluded runs at h*
        return
    with open(out / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def by_width(h):
    return sorted([r for r in D["runs"] if r["width"] == h], key=lambda r: r["seed"])


t0 = time.time()
runs = by_width(hstar)
ok = [r for r in runs if r["train_err"] <= a.max_train_err]
pair_rows, mech_rows = [], []
for i in range(0, len(ok) - 1, 2):
    rA, rB = ok[i], ok[i + 1]
    A, B = rA["params"], rB["params"]
    irrA, irrB = irreducibility(A, Xtr), irreducibility(B, Xtr)
    flagged = irrA["nearly_reducible"] or irrB["nearly_reducible"]
    ang, off = match_lines(lines(A), lines(B))
    h1 = bool(ang.max() <= 10 and off.max() <= 0.25)
    aligned, info = {"none": B, "wm": weight_match(A, B), "wm_sign": weight_match(A, B, True)}, {}
    if hstar <= a.exact_max:
        aligned["exact_perm"], info["exact_perm"] = exact_search(A, B, Xtr, ytr, False)
        aligned["exact_sign"], info["exact_sign"] = exact_search(A, B, Xtr, ytr, True)
    for mode, Bm in aligned.items():
        pair_rows.append(dict(task=a.task, width=hstar, pair=i // 2, seed_A=rA["seed"], seed_B=rB["seed"], mode=mode,
                              flagged=flagged, h1_pass=h1, h1_max_angle=float(ang.max()), h1_max_offset=float(off.max()),
                              n_flips=count_flips(B, Bm), rel_distance=rel_distance(A, Bm),
                              **all_measures(A, Bm, Xtr, ytr, Xte, yte)))
    if "exact_sign" in info:
        mech_rows.append(dict(task=a.task, pair=i // 2, flagged=flagged,
                              **mechanism(A, B, info["exact_sign"]["signs"], info["exact_sign"]["perm"], Xtr, ytr)))
    print(f"{a.task} h*={hstar} pair {i // 2}: done ({time.time() - t0:.0f}s)", flush=True)
write("pairs.csv", pair_rows); write("mechanism.csv", mech_rows)

# widths above h*: descriptive only, no claim (Section 4)
above = []
for h in sorted({r["width"] for r in D["runs"]}):
    if h <= hstar:
        continue
    okh = [r for r in by_width(h) if r["train_err"] <= a.max_train_err]
    for i in range(0, len(okh) - 1, 2):
        A, B = okh[i]["params"], okh[i + 1]["params"]
        for mode, Bm in {"none": B, "wm": weight_match(A, B), "wm_sign": weight_match(A, B, True)}.items():
            above.append(dict(task=a.task, width=h, pair=i // 2, mode=mode, n_flips=count_flips(B, Bm),
                              **all_measures(A, Bm, Xtr, ytr, Xte, yte, dense=())))
write("above_hstar.csv", above)

# excluded runs at h* (Section 7), and exploratory barriers to an included run
exc, evi = [], []
for r in runs:
    if r["train_err"] <= a.max_train_err:
        continue
    exc.append(dict(task=a.task, width=hstar, seed=r["seed"], train_err=r["train_err"],
                    train_err_ckpt=r["train_err_ckpt"], **classify(r, Xtr, edges)))
    if ok:
        A = ok[0]["params"]
        for mode, Bm in {"none": r["params"], "wm_sign": weight_match(A, r["params"], True)}.items():
            evi.append(dict(task=a.task, seed_excluded=r["seed"], seed_included=ok[0]["seed"], mode=mode,
                            **all_measures(A, Bm, Xtr, ytr, Xte, yte, dense=())))
write("excluded.csv", exc); write("excl_vs_incl.csv", evi)

# exploratory: headline on ALL runs at h*, ignoring inclusion (Section 7)
if a.all_runs:
    rows = []
    for i in range(0, len(runs) - 1, 2):
        A, B = runs[i]["params"], runs[i + 1]["params"]
        modes = {"none": B, "wm": weight_match(A, B), "wm_sign": weight_match(A, B, True)}
        if hstar <= a.exact_max:
            modes["exact_sign"] = exact_search(A, B, Xtr, ytr, True)[0]
        for mode, Bm in modes.items():
            rows.append(dict(task=a.task, width=hstar, pair=i // 2, mode=mode, exploratory=True,
                             **all_measures(A, Bm, Xtr, ytr, Xte, yte, dense=())))
    write("allruns.csv", rows)
print(f"{a.task}: wrote {out} ({time.time() - t0:.0f}s)")

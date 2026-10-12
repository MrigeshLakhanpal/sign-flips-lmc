"""Figures. python src/plot.py  ->  figures/*.png"""
import argparse, csv, pickle, collections, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core import lines_original, path, weight_match

ap = argparse.ArgumentParser()
ap.add_argument("--max-train-err", type=float, default=0.02)  # must match analyze.py
args = ap.parse_args()
pathlib.Path("figures").mkdir(exist_ok=True)
D = pickle.load(open("results/models.pkl", "rb"))
rows = list(csv.DictReader(open("results/barriers.csv")))
inc = {int(r["width"]): int(r["included"]) for r in csv.DictReader(open("results/inclusion.csv"))}

# Figure 1: every included seed's learned lines at the smallest width that has pairs
w0 = min(int(r["width"]) for r in rows)
runs = [r for r in D["runs"] if r["width"] == w0 and r["train_err"] <= args.max_train_err]
Xtr, ytr = D["data"]["Xtr"] * D["sd"] + D["mu"], D["data"]["ytr"]
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr, cmap="coolwarm", s=4, alpha=0.4)
t = np.linspace(150, 285, 2)
for r in runs:
    W, b = lines_original(r["params"], D["mu"], D["sd"])
    for wi, bi in zip(W, b):
        if abs(wi[1]) > 1e-9:
            ax.plot(t, -(wi[0] * t + bi) / wi[1], color="k", lw=0.6, alpha=0.35)
        else:
            ax.axvline(-bi / wi[0], color="k", lw=0.6, alpha=0.35)
ax.set(xlim=(150, 285), ylim=(11.5, 15.5), xlabel="Temperature (C)", ylabel="Duration (min)",
       title=f"Hidden-unit lines, width {w0}, {len(runs)} seeds")
fig.tight_layout(); fig.savefig("figures/fig1_lines.png", dpi=200)

# Figure 2: error barrier vs width, median with interquartile whiskers, per mode
by = collections.defaultdict(list)
for r in rows:
    by[(r["mode"], int(r["width"]))].append(float(r["err_barrier"]) * 100)
fig, ax = plt.subplots(figsize=(6.5, 4))
modes = ["none", "wm", "wm_sign", "bf_perm", "bf_sign"]
for k, m in enumerate(modes):
    ws = sorted(w for (mm, w) in by if mm == m)
    if not ws:
        continue
    med = [np.median(by[(m, w)]) for w in ws]
    lo = [med[i] - np.percentile(by[(m, w)], 25) for i, w in enumerate(ws)]
    hi = [np.percentile(by[(m, w)], 75) - med[i] for i, w in enumerate(ws)]
    x = np.array(ws) * (1 + 0.04 * (k - 2))          # small offset so whiskers don't overlap
    ax.errorbar(x, med, yerr=[lo, hi], marker="o", ms=4, capsize=3, label=m)
ax.set_xscale("log", base=2)
ax.set(xlabel="Hidden width", ylabel="Train error barrier (percentage points)",
       title="Barrier vs width (median, IQR over independent pairs)")
ax.axhline(0, color="grey", lw=0.6); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("figures/fig2_barrier_vs_width.png", dpi=200)

# Figure 3: interpolation curves for the first pair at the smallest width
A, B = runs[0]["params"], runs[1]["params"]
Xs, ys = D["data"]["Xtr"], D["data"]["ytr"]
al = np.linspace(0, 1, 25)
fig, ax = plt.subplots(figsize=(5.5, 3.8))
for name, Bm in [("no alignment", B), ("weight matching", weight_match(A, B)),
                 ("weight matching + signs", weight_match(A, B, True))]:
    ax.plot(al, path(A, Bm, Xs, ys)[:, 1] * 100, label=name)
ax.set(xlabel="alpha", ylabel="Train error (%)", title=f"Straight path between two width-{w0} models")
ax.legend(frameon=False); fig.tight_layout(); fig.savefig("figures/fig3_paths.png", dpi=200)
print("figures written; included per width:", inc)

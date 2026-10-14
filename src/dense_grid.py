"""Dense-grid check of the width-3 complete-group search (post-hoc, added 15 Oct 2026; written by Claude).

The published search chooses each alignment by its error barrier on 25 values of alpha
(k/24). This script re-runs that same choice, then re-measures the chosen alignment on
finer grids that CONTAIN the 25 original points (97 points = k/96, 385 points = k/384),
so a dense-grid barrier can only be equal to or larger than the 25-point one.

python src/dense_grid.py                      # 2% inclusion, width 3
python src/dense_grid.py --max-train-err 0.05
"""
import argparse
import numpy as np
import pickle
from core import path, barrier
from b1 import brute_force_exact

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="results/models.pkl")
ap.add_argument("--max-train-err", type=float, default=0.02)
ap.add_argument("--width", type=int, default=3)
a = ap.parse_args()

D = pickle.load(open(a.models, "rb"))
Xtr, ytr = D["data"]["Xtr"], D["data"]["ytr"]
ok = [r for r in D["runs"] if r["width"] == a.width and r["train_err"] <= a.max_train_err]
grids = [25, 97, 385]
res = {m: {g: [] for g in grids} for m in ("exact_perm", "exact_sign")}

for i in range(0, len(ok) - 1, 2):
    A, B = ok[i]["params"], ok[i + 1]["params"]
    for mode, sign in (("exact_perm", False), ("exact_sign", True)):
        Bm = brute_force_exact(A, B, Xtr, ytr, sign, 25)       # same choice as the paper
        for g in grids:
            res[mode][g].append(100 * barrier(path(A, Bm, Xtr, ytr, g))[1])   # error barrier, points

print(f"width {a.width}, {len(ok) // 2} pairs, error barrier in points (median / max)")
for mode in res:
    for g in grids:
        v = np.array(res[mode][g])
        print(f"  {mode:10s} {g:3d}-point grid: median {np.median(v):6.2f}   max {v.max():6.2f}")

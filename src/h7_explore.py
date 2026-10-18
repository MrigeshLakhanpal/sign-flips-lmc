"""EXPLORATORY (post hoc, decided 19 Oct 2026 after seeing H7 fail): H7 on measures without the error ceiling.
Written by Claude. The pre-registered H7 verdict (error barrier) stays "not supported"; this is labelled exploratory.

For every analyzed pair at h* (main 2% analysis), re-derive the exact_sign alignment (same rule, so the same
alignment as analyze_v2.py), then with the same permutation but WITHOUT the flips measure:
  B_F_loss      raw loss barrier
  B_F_cal_loss  calibrated loss barrier (81 scales, as in measures.py)
  B_F_err       error barrier (the pre-registered H7 outcome, recomputed as a check)
Predictor: M = sum over flipped units of |s_i^A| / 2.

python src/h7_explore.py      ->  results_v2/h7_explore.csv and printed Spearman rho with 95% bootstrap CI
"""
import json, pathlib, pickle
import numpy as np
import pandas as pd
from core import permute, path, barrier
from exact_search import exact_search
from measures import calibrated_curve
from mechanism import spearman_ci

d = pathlib.Path("results_v2")
frozen = json.load(open(d / "stage1_frozen.json"))["tasks"]
rows = []
for task, fr in sorted(frozen.items()):
    if fr["hstar"] is None or not (d / task / "pairs.csv").exists():
        continue
    D = pickle.load(open(d / fr["file"], "rb"))
    X, y = D["data"]["Xtr"], D["data"]["ytr"]
    P = pd.read_csv(d / task / "pairs.csv").drop_duplicates("pair")
    runs = {r["seed"]: r for r in D["runs"] if r["width"] == fr["hstar"]}
    for _, p in P.iterrows():
        A, B = runs[p["seed_A"]]["params"], runs[p["seed_B"]]["params"]
        _, info = exact_search(A, B, X, y, True)
        signs, perm = info["signs"], info["perm"]
        F = [i for i in range(len(perm)) if signs[perm[i]] < 0]
        if not F:
            continue
        Bp = permute(B, perm)
        lb, eb = barrier(path(A, Bp, X, y))
        cal, _ = calibrated_curve(A, Bp, X, y)
        a = np.linspace(0, 1, len(cal))
        rows.append(dict(task=task, pair=int(p["pair"]), flagged=bool(p["flagged"]), h1_pass=bool(p["h1_pass"]),
                         n_flipped=len(F), M=float(np.sum(np.abs(A["w2"])[F]) / 2),
                         B_F_err=float(eb), B_F_loss=float(lb),
                         B_F_cal_loss=float((cal - ((1 - a) * cal[0] + a * cal[-1])).max())))
    print(task, "done", flush=True)

R = pd.DataFrame(rows)
R.to_csv(d / "h7_explore.csv", index=False)
U = R[~R["flagged"]]
print(f"\nEXPLORATORY H7, {len(U)} unflagged pairs with >= 1 flip (pooled over tasks)")
for col in ("B_F_err", "B_F_loss", "B_F_cal_loss"):
    rho, lo, hi = spearman_ci(U["M"], U[col])
    print(f"  M vs {col:13s}: rho = {rho:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
print("\nPer task (M vs B_F_loss):")
for t, g in U.groupby("task"):
    if len(g) >= 5:
        rho, lo, hi = spearman_ci(g["M"], g["B_F_loss"], n_boot=2000)
        print(f"  {t}: n = {len(g):3d}  rho = {rho:+.3f}  [{lo:+.3f}, {hi:+.3f}]")

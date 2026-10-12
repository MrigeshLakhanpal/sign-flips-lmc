"""Run B1 on the saved models (written by Claude; plumbing only).

python src/run_b1.py                                   # main: 2% inclusion, widths 3 4 6 8
python src/run_b1.py --max-train-err 0.01 --widths 3 --out-dir results_b1_inc1
python src/run_b1.py --max-train-err 0.05 --widths 3 --out-dir results_b1_inc5

Writes into --out-dir:
  b1_barriers.csv        one row per (width, pair, mode): loss and error barrier, peak alpha, flips
  b1_irreducibility.csv  one row per trained network at those widths (included or not)
Pairs are formed exactly as in analyze.py, so pair numbers match barriers.csv.
Rules dated in the research notebook before this run.
"""
import argparse, csv, pathlib, pickle, time
import numpy as np
from core import brute_force, path, barrier
from extras import count_flips
from b1 import brute_force_exact, irreducibility, peak_alpha

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="results/models.pkl")
ap.add_argument("--max-train-err", type=float, default=0.02)
ap.add_argument("--widths", type=int, nargs="+", default=[3, 4, 6, 8])
ap.add_argument("--exact-perm-max", type=int, default=8)   # 8! = 40,320 versions
ap.add_argument("--exact-sign-max", type=int, default=6)   # 2^6 x 6! = 46,080 versions
ap.add_argument("--n-alpha", type=int, default=25)
ap.add_argument("--out-dir", default="results_b1")
a = ap.parse_args()

D = pickle.load(open(a.models, "rb"))
Xtr, ytr = D["data"]["Xtr"], D["data"]["ytr"]
out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
bar_rows, irr_rows = [], []
t0 = time.time()

for h in a.widths:
    runs = [r for r in D["runs"] if r["width"] == h]
    for r in runs:
        irr_rows.append(dict(width=h, seed=r["seed"], train_err=r["train_err"],
                             included=r["train_err"] <= a.max_train_err,
                             **irreducibility(r["params"], Xtr)))
    ok = [r for r in runs if r["train_err"] <= a.max_train_err]
    for i in range(0, len(ok) - 1, 2):
        A, B = ok[i]["params"], ok[i + 1]["params"]
        modes = {}
        if h <= a.exact_perm_max:
            modes["bf_perm"] = brute_force(A, B, Xtr, ytr)
            modes["exact_perm"] = brute_force_exact(A, B, Xtr, ytr, False, a.n_alpha)
        if h <= a.exact_sign_max:
            modes["bf_sign"] = brute_force(A, B, Xtr, ytr, allow_sign=True)
            modes["exact_sign"] = brute_force_exact(A, B, Xtr, ytr, True, a.n_alpha)
        for mode, Bm in modes.items():
            curve = path(A, Bm, Xtr, ytr, a.n_alpha)
            lb, eb = barrier(curve)
            bar_rows.append(dict(width=h, pair=i // 2, mode=mode, loss_barrier=lb,
                                 err_barrier=eb, peak_alpha=peak_alpha(curve) if eb > 1e-9 else np.nan,  # no bump, no peak
                                 n_flips=count_flips(B, Bm)))
        print(f"width {h}: pair {i // 2 + 1}/{len(ok) // 2} done ({time.time() - t0:.0f}s)", flush=True)
        # save after every pair, so a stopped run keeps what it finished
        for name, rows in [("b1_barriers", bar_rows), ("b1_irreducibility", irr_rows)]:
            with open(out / f"{name}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
print(f"wrote {out}/b1_barriers.csv and b1_irreducibility.csv")

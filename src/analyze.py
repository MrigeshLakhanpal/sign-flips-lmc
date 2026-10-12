"""Barriers and feature comparison for every width.

python src/analyze.py --max-train-err 0.02
Writes barriers.csv, lines.csv, inclusion.csv into --out-dir (default results/).
barriers.csv also records, per pair and mode, how many units the alignment flipped
(n_flips) and which calibration scales were chosen (both fixed before the main run).
Thresholds (max-train-err etc.) were dated in the research notebook before the first real run.
"""
import argparse, csv, pathlib, pickle
import numpy as np
from core import (weight_match, brute_force, path, barrier, lines, match_lines, SCALES)
from extras import count_flips, chosen_scales

ap = argparse.ArgumentParser()
ap.add_argument("--models", default="results/models.pkl")
ap.add_argument("--max-train-err", type=float, default=0.02)  # inclusion rule
ap.add_argument("--bf-perm-max", type=int, default=8)         # brute force perms up to this width (8! = 40320)
ap.add_argument("--bf-sign-max", type=int, default=6)         # perms x signs up to this width (2^6 * 6! = 46080)
ap.add_argument("--n-alpha", type=int, default=25)
ap.add_argument("--out-dir", default="results")            # e.g. results_inc1 for the 1% sensitivity run
a = ap.parse_args()

D = pickle.load(open(a.models, "rb"))
Xtr, ytr, Xte, yte = (D["data"][k] for k in ("Xtr", "ytr", "Xte", "yte"))
widths = sorted({r["width"] for r in D["runs"]})

bar_rows, line_rows, inc_rows = [], [], []
for h in widths:
    runs = [r for r in D["runs"] if r["width"] == h]
    ok = [r for r in runs if r["train_err"] <= a.max_train_err]
    inc_rows.append(dict(width=h, trained=len(runs), included=len(ok)))
    # disjoint pairs (0,1), (2,3), ... so pairs are independent samples
    for i in range(0, len(ok) - 1, 2):
        A, B = ok[i]["params"], ok[i + 1]["params"]
        aligned = {"none": B, "wm": weight_match(A, B), "wm_sign": weight_match(A, B, True)}
        if h <= a.bf_perm_max:
            aligned["bf_perm"] = brute_force(A, B, Xtr, ytr)
        if h <= a.bf_sign_max:
            aligned["bf_sign"] = brute_force(A, B, Xtr, ytr, allow_sign=True)
        for mode, Bm in aligned.items():
            lb, eb = barrier(path(A, Bm, Xtr, ytr, a.n_alpha))
            clb, _ = barrier(path(A, Bm, Xtr, ytr, a.n_alpha, calibrate=True))
            _, teb = barrier(path(A, Bm, Xte, yte, a.n_alpha))
            sc = chosen_scales(A, Bm, Xtr, ytr, a.n_alpha)
            bar_rows.append(dict(width=h, pair=i // 2, mode=mode, loss_barrier=lb,
                                 err_barrier=eb, cal_loss_barrier=clb, test_err_barrier=teb,
                                 n_flips=count_flips(B, Bm),
                                 cal_scale_min=sc.min(), cal_scale_max=sc.max(),
                                 cal_edge_hits=int(np.sum((sc == SCALES[0]) | (sc == SCALES[-1])))))
        ang, off = match_lines(lines(A), lines(B))
        line_rows.append(dict(width=h, pair=i // 2, max_angle_deg=ang.max(),
                              median_angle_deg=np.median(ang), max_offset=off.max()))
    print(f"width {h:>3}: {len(ok)}/{len(runs)} included, {len(ok) // 2} pairs")

out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
for name, rows in [("barriers", bar_rows), ("lines", line_rows), ("inclusion", inc_rows)]:
    with open(out / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
print(f"wrote {out}/barriers.csv, lines.csv, inclusion.csv")

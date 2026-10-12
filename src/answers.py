"""Read Q1-Q4 and H1-H3 off the main-run CSV files, using the rules dated in
the research notebook (2026-10-08). Prints numbers only; the verdicts are written by hand.
Run from the repo's top folder:  python src/answers.py   (optional: --results results_inc1)
"""
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
a = ap.parse_args()
EPS, ANG, OFF = 0.5, 10.0, 0.25            # pre-registered (2026-10-08)
pd.set_option("display.width", 140)

b = pd.read_csv(f"{a.results}/barriers.csv")
L = pd.read_csv(f"{a.results}/lines.csv")
b["err_pts"] = 100 * b["err_barrier"]       # training-set error barrier, percentage points
p = b.pivot_table(index=["width", "pair"], columns="mode", values="err_pts")


def q(s, x):
    return s.quantile(x)


print("== Error barrier (training set, points): median [Q1, Q3] per width and mode ==")
t = b.groupby(["width", "mode"])["err_pts"].agg(
    median="median", q1=lambda s: q(s, .25), q3=lambda s: q(s, .75), n="size").round(3)
print(t.unstack("mode")["median"].to_string(), "\n")
print("Upper quartile (Q3):"); print(t.unstack("mode")["q3"].to_string(), "\n")

print("== H1 / Q1: pairs whose every matched line is within 10 deg and 0.25 ==")
L["same"] = (L.max_angle_deg <= ANG) & (L.max_offset <= OFF)
print(L.groupby("width")["same"].agg(passing="sum", pairs="size").to_string(), "\n")

print("== H2: median wm error barrier at width 3 (band 0.5 to 3) ==")
print(round(p.loc[3, "wm"].median(), 3), "\n")

print("== H3: median wm error barrier per width ==")
print(p["wm"].groupby(level=0).median().round(3).to_string(), "\n")

print("== Q2: is the aligned upper quartile <= eps (barrier removed)? ==")
for m in ["wm", "wm_sign"]:
    print(m, (p[m].groupby(level=0).quantile(.75) <= EPS).to_dict())
print()

print("== Q3: weight matching minus brute force (points), per width ==")
for wm, bf in [("wm", "bf_perm"), ("wm_sign", "bf_sign")]:
    d = (p[wm] - p[bf]).dropna()
    print(f"{wm} - {bf}:")
    print(d.groupby(level=0).agg(gap_over_eps=lambda s: int((s > EPS).sum()), pairs="size",
                                 median="median", min="min", max="max").round(3).to_string())
print()

print("== Q4: bf_perm minus bf_sign (points) ==")
d = (p["bf_perm"] - p["bf_sign"]).dropna()
print(d.groupby(level=0).agg(lower_by_over_eps=lambda s: int((s > EPS).sum()),
                             within_eps=lambda s: int((s.abs() < EPS).sum()),
                             pairs="size", median="median").round(3).to_string(), "\n")

print("== Sign flips used (pairs with at least one flip) ==")
f = b.pivot_table(index=["width", "pair"], columns="mode", values="n_flips")
for m in ["wm_sign", "bf_sign"]:
    s = f[m].dropna()
    print(m, s.groupby(level=0).agg(lambda v: f"{int((v > 0).sum())}/{len(v)}").to_dict())
print()

print("== Calibration edge hits (pairs with any blend point at scale 0.25 or 8) ==")
print(b.groupby("mode")["cal_edge_hits"].agg(pairs_hit=lambda s: int((s > 0).sum()), pairs="size").to_string())

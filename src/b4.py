"""B4: training-length pitfall (weight norms) and the weight-decay side-check at width 64.
Rules dated in the research notebook before the run.
Check with:  python tests/test_b4.py   (made-up numbers, no coffee data)

Real use (after the extra training run and its analysis):
  python src/b4.py --base results --wd results_wd_all --models results/models.pkl \
                   --models-wd results/models_wd.pkl --exploratory
--exploratory: the decay run had no networks meeting the 2% inclusion rule, so the dated
rule gives no verdict; the comparison on all 40 networks is exploratory (post-hoc).
"""
import argparse, pickle
import numpy as np
import pandas as pd

EPS = 0.5   # points, pre-registered


def weight_norm(p):
    """L2 norm of all weights (W1 and w2; biases excluded, as weight decay excludes them)."""
    return float(np.sqrt(np.sum(p["W1"] ** 2) + np.sum(p["w2"] ** 2)))


def median_norms(runs, max_err=0.02):
    """Median weight norm per width over included networks."""
    rows = [(r["width"], weight_norm(r["params"])) for r in runs if r["train_err"] <= max_err]
    return pd.DataFrame(rows, columns=["width", "norm"]).groupby("width")["norm"].median()


def bootstrap_median_diff(a, b, paired, n_boot=10000, seed=0):
    """95% interval for median(b) - median(a), in the same units as a and b.
    paired=True: a[k] and b[k] are the same pair of seeds, so resample pair indices together.
    paired=False: resample a and b separately."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for k in range(n_boot):
        if paired:
            idx = rng.integers(0, len(a), len(a))
            d[k] = np.median(b[idx]) - np.median(a[idx])
        else:
            ia, ib = rng.integers(0, len(a), len(a)), rng.integers(0, len(b), len(b))
            d[k] = np.median(b[ib]) - np.median(a[ia])
    return np.percentile(d, [2.5, 97.5])


def verdict(drop, lo, hi):
    """Dated rule (research notebook, before the run): 'lowers' if the median drops by more than EPS and the interval
    for median(wd) - median(base) lies entirely below 0; 'raises' for the mirror case;
    otherwise 'no clear change'. drop = median(base) - median(wd)."""
    if drop > EPS and hi < 0:
        return "lowers"
    if drop < -EPS and lo > 0:
        return "raises"
    return "no clear change"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="results")
    ap.add_argument("--wd", default="results_wd_all")
    ap.add_argument("--models", default="results/models.pkl")
    ap.add_argument("--models-wd", default="results/models_wd.pkl")
    ap.add_argument("--exploratory", action="store_true")
    a = ap.parse_args()
    print("Median weight norm per width (included networks, no weight decay):")
    print(median_norms(pickle.load(open(a.models, "rb"))["runs"]).round(2).to_string(), "\n")
    try:
        runs_wd = [r for r in pickle.load(open(a.models_wd, "rb"))["runs"] if r["width"] == 64]
        e = [100 * r["train_err"] for r in runs_wd]
        print(f"Weight-decay run (width 64): {sum(x <= 2 for x in e)}/{len(e)} meet the 2% rule; "
              f"training error {min(e):.2f}-{max(e):.2f}%; median weight norm (all networks) "
              f"{np.median([weight_norm(r['params']) for r in runs_wd]):.2f}\n")
    except FileNotFoundError:
        print(f"({a.models_wd} not found: skipping the weight-decay network summary)\n")
    B = pd.read_csv(f"{a.base}/barriers.csv"); W = pd.read_csv(f"{a.wd}/barriers.csv")
    B, W = B[B.width == 64], W[W.width == 64]
    inc_b = pd.read_csv(f"{a.base}/inclusion.csv").set_index("width").loc[64, "included"]
    inc_w = pd.read_csv(f"{a.wd}/inclusion.csv").set_index("width").loc[64, "included"]
    paired = inc_b == inc_w == 40      # all seeds included in both runs: pair k is the same seeds
    note = " (inclusion ignored: exploratory, post-hoc)" if a.exploratory else ""
    print(f"Width 64 networks compared: base {inc_b}/40, wd {inc_w}/40{note} -> {'paired' if paired else 'unpaired'} comparison")
    for mode in ["none", "wm", "wm_sign"]:
        for col in ["err_barrier", "cal_loss_barrier"]:
            scale = 100 if col == "err_barrier" else 1        # error in points; loss as is
            x = B[B["mode"] == mode].sort_values("pair")[col].to_numpy() * scale
            z = W[W["mode"] == mode].sort_values("pair")[col].to_numpy() * scale
            lo, hi = bootstrap_median_diff(x, z, paired)
            drop = np.median(x) - np.median(z)
            if mode == "none" and col == "err_barrier":
                v = "exploratory, no dated verdict" if a.exploratory else verdict(drop, lo, hi)
            else:
                v = "reported only"
            print(f"{mode:8s} {col:17s} base {np.median(x):7.3f}  wd {np.median(z):7.3f}  "
                  f"wd-base 95% [{lo:7.3f}, {hi:7.3f}]  -> {v}")


if __name__ == "__main__":
    main()

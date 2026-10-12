"""H1b: do seeds with higher training error have lines further from the others?

Check with:  python tests/test_h1b.py   (made-up networks, no coffee data)
Real run, after src/train.py:  python src/h1b.py

Rules dated in the research notebook (2026-10-08):
- distance between two seeds = worst matched line, in threshold units:
  max over the matched lines of max(angle / 10, offset / 0.25)
- all width-3 seeds are used, excluded ones included
- verdict: Spearman rank correlation; H1b is wrong if rho <= 0
"""
import argparse, csv, pickle
import numpy as np
from scipy.stats import spearmanr
from core import lines, match_lines

ANGLE_T = 10.0    # degrees, pre-registered
OFFSET_T = 0.25   # standardized units, pre-registered


def pair_distance(pA, pB):
    """One number for how different two networks' lines are.

    match_lines(lines(pA), lines(pB)) returns two arrays, one entry per
    matched line: the angle (degrees) and the offset difference.
    Turn each into threshold units (divide by ANGLE_T and OFFSET_T),
    take the larger of the two for each line, then the largest over all lines.
    Return a plain float.
    """
    angles, offsets = match_lines(lines(pA), lines(pB))
    angle_units = angles / ANGLE_T
    offset_units = offsets / OFFSET_T
    max_per_line = np.maximum(angle_units, offset_units)
    return float(np.max(max_per_line))


def median_distances(params):
    """params is a list of networks (one per seed).
    Return a NumPy array with one entry per seed: the median of its
    pair_distance to every OTHER seed (never to itself).
    """
    result = []                              # one median per seed goes here
    for i in range(len(params)):             # "for each seed"
        dists = []                           # a fresh, empty collection for this seed
        for j in range(len(params)):         # "every ... seed"
            if i == j:                       # "... other": when should we skip?
                continue                     # 'continue' = skip to the next j
            dists.append(pair_distance(params[i], params[j]))             # "collect its distance": distance between which two?
        result.append(np.median(dists))                # "take the median" of the collection
    return np.array(result)


def bootstrap_spearman(x, y, n_boot=10_000, seed=0):
    """Spearman rho of x and y, plus a 95% bootstrap interval.

    One bootstrap sample = pick len(x) seed indices WITH replacement
    (rng.integers), compute rho on x[idx], y[idx]. Do this n_boot times
    with rng = np.random.default_rng(seed). The interval is the 2.5th and
    97.5th percentiles of those rhos (np.percentile). Some samples can give
    nan (when every picked value is equal); leave them out with np.nanpercentile.
    Return (rho, low, high) as floats.  spearmanr(a, b) returns an object whose
    first item [0] is rho.
    """
    x, y = np.asarray(x), np.asarray(y)          # make sure both are NumPy arrays
    rho = spearmanr(x, y)[0]                      # rho on the real data
    rng = np.random.default_rng(seed)             # fixed seed, so the result repeats exactly
    rhos = []
    for _ in range(n_boot):                       # '_' = a loop counter we never use
        idx = rng.integers(0, len(x), len(x))     # (1) how many seeds per draw?
        rhos.append(spearmanr(x[idx], y[idx])[0]) # (2) which values go in?
    low, high = np.nanpercentile(rhos, [2.5, 97.5])   # (3) the middle 95%
    return float(rho), float(low), float(high)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="results/models.pkl")
    ap.add_argument("--width", type=int, default=3)
    ap.add_argument("--max-train-err", type=float, default=0.02)  # only to mark who was excluded
    a = ap.parse_args()

    D = pickle.load(open(a.models, "rb"))
    runs = sorted((r for r in D["runs"] if r["width"] == a.width), key=lambda r: r["seed"])
    err = np.array([r["train_err"] for r in runs])
    dist = median_distances([r["params"] for r in runs])
    rho, lo, hi = bootstrap_spearman(err, dist)

    with open("results/h1b.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "train_err", "included", "median_distance"])
        for r, d in zip(runs, dist):
            w.writerow([r["seed"], r["train_err"], r["train_err"] <= a.max_train_err, d])

    verdict = "WRONG (rho <= 0)" if rho <= 0 else ("supported, weak (interval includes 0)"
                                                   if lo <= 0 else "supported")
    print(f"width {a.width}: {len(runs)} seeds, {int((err > a.max_train_err).sum())} excluded by inclusion rule")
    print(f"Spearman rho = {rho:.3f}, 95% bootstrap interval [{lo:.3f}, {hi:.3f}]")
    print("H1b:", verdict)
    print("wrote results/h1b.csv")


if __name__ == "__main__":
    main()

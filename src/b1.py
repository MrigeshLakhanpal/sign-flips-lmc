"""B1: alignment over the complete symmetry group (rules dated in the research notebook before the run).

Check with:  python tests/test_b1.py   (made-up networks and data, no coffee data)
Full run (several hours):  python src/run_b1.py
"""
from itertools import permutations, product
import numpy as np
from core import flip, permute, path, barrier, lines, sigmoid

ANGLE_T, OFFSET_T = 10.0, 0.25   # H1's same-line test, pre-registered
W2_RATIO_T = 0.05                # output weight "near zero": below 5% of the largest (dated rule)
ACT_STD_T = 0.01                 # unit "nearly constant": activation std on training data (dated rule)


# ------------------------------------------------------------ irreducibility checker
def output_weight_ratio(p):
    """Condition 1. Smallest |w2| divided by largest |w2|. Near 0 = a unit that barely
    reaches the output, so its input weights could be almost anything."""
    a = np.abs(p["w2"])
    return float(a.min() / a.max())


def closest_unit_pair(p):
    """Conditions 2 and 3. Of all pairs of units in ONE network, how close is the
    closest pair to being equal or opposite? Same units as H1b: threshold units,
    max(angle / 10, offset / 0.25). At most 1 means the pair passes H1's same-line test."""
    n, c = lines(p)                  # unit normals (h, 2) and offsets (h,)
    h = len(c)
    best = np.inf
    for i in range(h):
        for j in range(i + 1, h):
            dot = float(n[i] @ n[j])
            ang = np.degrees(np.arccos(min(1.0, abs(dot))))     # abs: opposite lines count as the same line
            off = abs(c[i] - np.sign(dot) * c[j])
            best = min(best, max(ang / ANGLE_T, off / OFFSET_T))
    return float(best)


def min_activation_std(p, X):
    """Condition 4. For each unit, how much its output varies over the training inputs;
    return the smallest. Near 0 = a unit that acts like a constant (an extra bias)."""
    act = sigmoid(X @ p["W1"].T + p["b1"])               # (n_points, h): one column per unit
    return float(act.std(axis=0).min())


def irreducibility(p, X):
    r, d, s = output_weight_ratio(p), closest_unit_pair(p), min_activation_std(p, X)
    nearly = r < W2_RATIO_T or d <= 1 or s < ACT_STD_T
    return dict(w2_ratio=r, pair_dist=d, act_std=s, nearly_reducible=bool(nearly))


# ------------------------------------------------------------ exact selection
def peak_alpha(curve):
    """Where the error bump peaks. curve is path(...) output: column 0 loss, column 1 error."""
    a = np.linspace(0, 1, len(curve))
    line = (1 - a) * curve[0, 1] + a * curve[-1, 1]
    return float(a[np.argmax(curve[:, 1] - line)])


def brute_force_exact(pA, pB, X, y, allow_sign=False, n_alpha=25):
    """Same search as core.brute_force (every flip pattern x every permutation),
    but keep the version with the lowest FULL error barrier over n_alpha points.
    Ties in error barrier (common: error moves in steps of 1/n) go to the lower loss barrier."""
    h = len(pB["b1"])
    best, best_p = (np.inf, np.inf), None
    sign_sets = product([1, -1], repeat=h) if allow_sign else [np.ones(h)]
    for s in list(sign_sets):
        q = flip(pB, s)
        for perm in permutations(range(h)):
            c = permute(q, perm)
            lb, eb = barrier(path(pA, c, X, y, n_alpha))
            eb = round(float(eb), 9)                     # removes float noise like 4e-19 (error moves in steps of 1/n)
            key = (eb, lb)
            if key < best:
                best, best_p = key, c
    return best_p

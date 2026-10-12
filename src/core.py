"""Core pieces for the coffee-roasting LMC probe (NumPy only).

Model: x (2) -> h sigmoid units -> 1 sigmoid output.
Params are a dict: W1 (h,2), b1 (h,), w2 (h,), b2 (float).
"""
from itertools import permutations, product

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------- data
def make_coffee(n, seed):
    """Coffee-roasting data, with the labeling rule of the Course 2 lab's
    load_coffee_data (checked against a public copy; see the paper, Section 3).
    Temperature 150-285 C, duration 11.5-15.5 min. Good roast = inside the
    triangle t > 175, d > 12, d <= -3/85 * t + 21."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(150, 285, n)
    d = rng.uniform(11.5, 15.5, n)
    y = ((t > 175) & (t < 260) & (d > 12) & (d < 15) & (d <= -3 / (260 - 175) * t + 21))
    return np.stack([t, d], 1), y.astype(float)


def standardize(X, mu=None, sd=None):
    if mu is None:
        mu, sd = X.mean(0), X.std(0)
    return (X - mu) / sd, mu, sd


# ---------------------------------------------------------------- model
def sigmoid(z):
    return 0.5 * (1 + np.tanh(0.5 * z))  # numerically stable


def init_params(h, rng):
    """Glorot-uniform weights, zero biases (the Keras default used in Course 2)."""
    l1, l2 = np.sqrt(6 / (2 + h)), np.sqrt(6 / (h + 1))
    return {"W1": rng.uniform(-l1, l1, (h, 2)), "b1": np.zeros(h),
            "w2": rng.uniform(-l2, l2, h), "b2": 0.0}


def logits(p, X):
    return sigmoid(X @ p["W1"].T + p["b1"]) @ p["w2"] + p["b2"]


def bce_from_logits(z, y):
    return np.mean(np.logaddexp(0, z) - y * z)


def metrics(p, X, y, scale=1.0):
    z = logits(p, X) * scale
    return bce_from_logits(z, y), np.mean((z > 0) != (y > 0.5))


def train(p, X, y, steps=3000, lr=0.01, wd=0.0):
    """Full-batch Adam on BCE. Full batch means the only randomness is the
    initialization, which isolates the effect of the seed."""
    keys = ["W1", "b1", "w2", "b2"]
    m = {k: np.zeros_like(np.asarray(p[k], float)) for k in keys}
    v = {k: np.zeros_like(np.asarray(p[k], float)) for k in keys}
    b1_, b2_, eps, n = 0.9, 0.999, 1e-8, len(y)
    for t in range(1, steps + 1):
        a = sigmoid(X @ p["W1"].T + p["b1"])          # (n,h)
        z = a @ p["w2"] + p["b2"]
        dz = (sigmoid(z) - y) / n                      # dL/dz
        g = {"w2": a.T @ dz, "b2": dz.sum()}
        da = np.outer(dz, p["w2"]) * a * (1 - a)       # (n,h)
        g["W1"], g["b1"] = da.T @ X, da.sum(0)
        for k in keys:
            gk = g[k] + wd * p[k] if k in ("W1", "w2") else g[k]
            m[k] = b1_ * m[k] + (1 - b1_) * gk
            v[k] = b2_ * v[k] + (1 - b2_) * gk ** 2
            step = lr * (m[k] / (1 - b1_ ** t)) / (np.sqrt(v[k] / (1 - b2_ ** t)) + eps)
            p[k] = p[k] - step
    p["b2"] = float(p["b2"])
    return p


# ---------------------------------------------------------------- symmetries
def permute(p, perm):
    perm = np.asarray(perm)
    return {"W1": p["W1"][perm], "b1": p["b1"][perm], "w2": p["w2"][perm], "b2": p["b2"]}


def flip(p, signs):
    """signs[i] = -1 flips unit i: sigma(-z) = 1 - sigma(z), so negate the
    unit's input weights, bias and output weight, and add w2[i] to b2.
    The function is unchanged."""
    s = np.asarray(signs, float)
    neg = s < 0
    return {"W1": p["W1"] * s[:, None], "b1": p["b1"] * s, "w2": p["w2"] * s,
            "b2": p["b2"] + p["w2"][neg].sum()}


def unit_vectors(p):
    return np.column_stack([p["W1"], p["b1"], p["w2"]])  # (h,4)


def weight_match(pA, pB, allow_sign=False):
    """Exact weight matching for one hidden layer via the Hungarian algorithm.
    With allow_sign, a B unit may also be flipped (u -> -u) before matching.
    The b2 shift caused by flips is not part of the matching objective."""
    uA, uB = unit_vectors(pA), unit_vectors(pB)
    S = uA @ uB.T
    rows, cols = linear_sum_assignment(-(np.abs(S) if allow_sign else S))
    perm = cols[np.argsort(rows)]                 # perm[i] = B unit matched to A unit i
    signs_B = np.ones(len(perm))
    if allow_sign:
        signs_B[perm] = np.where(S[np.arange(len(perm)), perm] < 0, -1.0, 1.0)
    return permute(flip(pB, signs_B), perm)


# ---------------------------------------------------------------- interpolation and barriers
def lerp(pA, pB, a):
    return {k: (1 - a) * np.asarray(pA[k]) + a * np.asarray(pB[k]) for k in pA}


SCALES = np.exp(np.linspace(np.log(0.25), np.log(8), 41))


def path(pA, pB, X, y, n_alpha=25, calibrate=False):
    """Loss and error along the straight path. With calibrate, each point's
    logits are rescaled by the best scalar on (X, y) (temperature scaling)."""
    out = []
    for a in np.linspace(0, 1, n_alpha):
        p = lerp(pA, pB, a)
        if calibrate:
            out.append(min((metrics(p, X, y, s) for s in SCALES), key=lambda r: r[0]))
        else:
            out.append(metrics(p, X, y))
    return np.array(out)  # (n_alpha, 2): loss, error


def barrier(curve):
    """Entezari et al. (2022), Eq. 1: max over alpha of curve minus the line
    joining its endpoints. Returns (loss barrier, error barrier)."""
    a = np.linspace(0, 1, len(curve))[:, None]
    line = (1 - a) * curve[0] + a * curve[-1]
    return (curve - line).max(0)


def midpoint_loss(pA, pB, X, y):
    return metrics(lerp(pA, pB, 0.5), X, y)[0]


def brute_force(pA, pB, X, y, allow_sign=False):
    """Try every permutation (and sign pattern) of B; keep the one with the
    lowest midpoint loss (the alpha = 0.5 surrogate of Entezari et al.)."""
    h = len(pB["b1"])
    best, best_p = np.inf, None
    sign_sets = product([1, -1], repeat=h) if allow_sign else [np.ones(h)]
    for s in list(sign_sets):
        q = flip(pB, s)
        for perm in permutations(range(h)):
            c = permute(q, perm)
            v = midpoint_loss(pA, c, X, y)
            if v < best:
                best, best_p = v, c
    return best_p


# ---------------------------------------------------------------- features as lines
def lines(p):
    """Each hidden unit's decision line in STANDARDIZED input space, as unit
    normal n and offset c with n.x + c = 0 and |n| = 1. Comparing in
    standardized space makes angles meaningful (raw temperature and duration
    have very different scales)."""
    norm = np.linalg.norm(p["W1"], axis=1, keepdims=True)
    return p["W1"] / norm, p["b1"] / norm[:, 0]


def lines_original(p, mu, sd):
    """Same lines in original units (temperature C, duration min), for plotting:
    returns (W, b) with W.x + b = 0."""
    return p["W1"] / sd, p["b1"] - (p["W1"] * mu / sd).sum(1)


def match_lines(LA, LB):
    """Sign-invariant matching of B's lines to A's. Returns per-line angle
    (degrees) and offset difference (standardized units)."""
    nA, cA = LA
    nB, cB = LB
    dot = nA @ nB.T
    ang = np.degrees(np.arccos(np.clip(np.abs(dot), 0, 1)))
    off = np.abs(cA[:, None] - np.sign(dot) * cB[None, :])
    r, c = linear_sum_assignment(ang / 10 + off)  # 10 degrees ~ 1 std of offset
    return ang[r, c], off[r, c]

"""v2 H7: does the midpoint-erasure bound predict the barrier? (v2_rules.md, Section 6). Written by Claude, 19 Oct 2026.

Given the exact_sign alignment (signs indexed by B's units, then permutation perm: aligned unit i = B unit perm[i]):
  F   = A's units i whose matched B unit is flipped
  M   = sum over i in F of |s_i^A| / 2          (s = output weight; bound on the midpoint logit shift)
  B_F = training-error barrier with the same permutation but WITHOUT the flips
"""
import numpy as np
from core import permute, path, barrier


def mechanism(pA, pB, signs, perm, X, y, n_alpha=25):
    signs, perm = np.asarray(signs), np.asarray(perm)
    F = [i for i in range(len(perm)) if signs[perm[i]] < 0]
    M = float(np.sum(np.abs(np.asarray(pA["w2"])[F])) / 2) if F else 0.0
    B_F = float(barrier(path(pA, permute(pB, perm), X, y, n_alpha))[1])
    return dict(n_flipped=len(F), M=M, B_F=B_F)


def spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).correlation)


def spearman_ci(x, y, n_boot=10000, seed=0):
    """Spearman rho with a 95% percentile bootstrap interval over pairs (resampling pairs)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng, rhos = np.random.default_rng(seed), []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
            rhos.append(spearman(x[i], y[i]))
    return spearman(x, y), float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))

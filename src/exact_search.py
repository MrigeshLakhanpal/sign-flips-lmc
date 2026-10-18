"""v2 exhaustive search over permutations (and sign flips), vectorized (v2_rules.md, Section 4).
Written by Claude, 19 Oct 2026.

Same rule as v1's b1.brute_force_exact: every flip pattern x every permutation, in the same order
(flip patterns outer, permutations inner); keep the one with the lowest 25-point training-error
barrier (rounded to 1e-9), ties going to the lower loss barrier, then to the earlier candidate.
tests/test_v2.py checks that it picks the same alignment as b1.brute_force_exact.
"""
from itertools import permutations, product
import numpy as np
from core import flip, permute


def symmetries(h, allow_sign):
    signs = list(product([1, -1], repeat=h)) if allow_sign else [tuple([1] * h)]
    perms = list(permutations(range(h)))
    S = np.array([s for s in signs for _ in perms], dtype=float)       # (G, h), indexed by B's units
    P = np.array([p for _ in signs for p in perms], dtype=int)         # (G, h): aligned unit i = B unit P[g, i]
    return S, P


def transform_stack(pB, S, P):
    """All transformed copies of B: flip(pB, S[g]) then permute(., P[g]), as stacked arrays."""
    W1, b1, w2, b2 = (np.asarray(pB[k], float) for k in ("W1", "b1", "w2", "b2"))
    W1s, b1s, w2s = W1[None] * S[:, :, None], b1[None] * S, w2[None] * S
    b2s = b2 + (w2[None] * (S < 0)).sum(1)
    g = np.arange(len(S))[:, None]
    return W1s[g, P], b1s[g, P], w2s[g, P], b2s


def stack_curves(pA, stack, X, y, n_alpha=25, chunk=256):
    """Reference implementation (slow, direct): loss and error along A -> each stacked copy. (G, n_alpha)."""
    W1g, b1g, w2g, b2g = stack
    W1A, b1A, w2A, b2A = (np.asarray(pA[k], float) for k in ("W1", "b1", "w2", "b2"))
    G = len(b2g)
    L, E = np.empty((G, n_alpha)), np.empty((G, n_alpha))
    for j, a in enumerate(np.linspace(0, 1, n_alpha)):
        for s in range(0, G, chunk):
            sl = slice(s, s + chunk)
            W1 = (1 - a) * W1A[None] + a * W1g[sl]
            b1 = (1 - a) * b1A[None] + a * b1g[sl]
            w2 = (1 - a) * w2A[None] + a * w2g[sl]
            b2 = (1 - a) * b2A + a * b2g[sl]
            pre = np.einsum("nd,ghd->gnh", X, W1) + b1[:, None, :]
            z = (0.5 * (1 + np.tanh(0.5 * pre)) * w2[:, None, :]).sum(2) + b2[:, None]
            L[sl, j] = np.mean(np.logaddexp(0, z) - y[None] * z, axis=1)
            E[sl, j] = np.mean((z > 0) != (y[None] > 0.5), axis=1)
    return L, E


def fast_curves(pA, pB, S, P, X, y, n_alpha=25, chunk=512):
    """Same numbers as stack_curves, much faster. At blend alpha, aligned unit i of a candidate is
    A's unit i blended with B's unit j = P[g, i], flipped or not (s = S[g, j]). Its contribution to the
    logit depends only on (i, j, s): h*h*2 possibilities. Compute those once per alpha, then each
    candidate's logit is a sum of h of them plus its output bias."""
    W1A, b1A, w2A, b2A = (np.asarray(pA[k], float) for k in ("W1", "b1", "w2", "b2"))
    W1B, b1B, w2B, b2B = (np.asarray(pB[k], float) for k in ("W1", "b1", "w2", "b2"))
    preA, preB = X @ W1A.T + b1A, X @ W1B.T + b1B              # (n, h); a flip negates a unit's pre-activation
    G, h = P.shape
    rows = np.arange(G)[:, None]
    J = P                                                       # (G, h)
    F = (S[rows, P] < 0).astype(int)                            # (G, h): is B unit P[g, i] flipped
    b2g = b2B + (w2B[None] * (S < 0)).sum(1)                    # (G,)
    I = np.arange(h)[None, :]
    L, E = np.empty((G, n_alpha)), np.empty((G, n_alpha))
    for jj, a in enumerate(np.linspace(0, 1, n_alpha)):
        T = np.empty((h, h, 2, len(y)))
        for f, sg in ((0, 1.0), (1, -1.0)):
            pre = (1 - a) * preA.T[:, None, :] + a * sg * preB.T[None, :, :]          # (h, h, n)
            w = (1 - a) * w2A[:, None] + a * sg * w2B[None, :]                       # (h, h)
            T[:, :, f, :] = 0.5 * (1 + np.tanh(0.5 * pre)) * w[:, :, None]
        b2 = (1 - a) * b2A + a * b2g
        for s0 in range(0, G, chunk):
            sl = slice(s0, s0 + chunk)
            z = T[I, J[sl], F[sl]].sum(1) + b2[sl, None]                               # (chunk, n)
            L[sl, jj] = np.mean(np.logaddexp(0, z) - y[None] * z, axis=1)
            E[sl, jj] = np.mean((z > 0) != (y[None] > 0.5), axis=1)
    return L, E


def _barrier_rows(C):
    a = np.linspace(0, 1, C.shape[1])[None]
    return (C - ((1 - a) * C[:, :1] + a * C[:, -1:])).max(1)


def exact_search(pA, pB, X, y, allow_sign=False, n_alpha=25, chunk=256):
    """Returns (aligned B, info) with info = dict(signs, perm, err_barrier, loss_barrier, n_candidates)."""
    h = len(pB["b1"])
    S, P = symmetries(h, allow_sign)
    L, E = fast_curves(pA, pB, S, P, X, y, n_alpha, chunk)
    eb, lb = np.round(_barrier_rows(E), 9), _barrier_rows(L)
    g = int(np.lexsort((np.arange(len(eb)), lb, eb))[0])
    Bm = permute(flip(pB, S[g]), P[g])
    return Bm, dict(signs=S[g].astype(int), perm=P[g], err_barrier=float(eb[g]),
                    loss_barrier=float(lb[g]), n_candidates=len(eb))

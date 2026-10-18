"""Tests for the v2 code, on MADE-UP networks and data only (no coffee or polygon training data).
Written by Claude, 19 Oct 2026.   python tests/test_v2.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from core import init_params, train, flip, permute, path, barrier, metrics
from b1 import brute_force_exact
from polygon import (make_polygon, label, edges_raw, accepted, area, coffee_edges_raw, to_standardized)
from train_v2 import train_ckpt
from exact_search import exact_search, symmetries
from measures import all_measures, rel_distance, SCALES_V2
from mechanism import mechanism, spearman_ci
from excluded import classify, edge_coverage

rng = np.random.default_rng(123)
X = rng.normal(size=(300, 2)); y = ((X[:, 0] > 0.2) & (X[:, 1] > -0.4) & (X.sum(1) < 1.3)).astype(float)
ok = 0


def check(name, cond):
    global ok
    assert cond, name
    ok += 1
    print("ok", name)


# polygon
for k in (3, 4, 5, 6):
    V, _ = make_polygon(k)
    check(f"polygon k={k} accepted, area 0.84", accepted(V) and abs(area(V) - 0.84) < 1e-9)
    P = np.random.default_rng(0).uniform(-1, 1, (50000, 2))
    check(f"polygon k={k} positive rate ~21%", abs(label(P, V).mean() - 0.21) < 0.01)
    N, c = edges_raw(V)
    check(f"polygon k={k} vertices lie on their edges", np.allclose((N * V).sum(1) + c, 0))
mu, sd = np.array([0.3, -0.2]), np.array([2.0, 0.5])
N, c = coffee_edges_raw(); Ns, cs = to_standardized(N, c, mu, sd)
pt_raw = np.array([175.0, 13.0]); pt_std = (pt_raw - mu) / sd
check("standardized edge passes through the same point", abs(Ns[0] @ pt_std + cs[0]) < 1e-12)

# train_ckpt is exactly core.train
p1 = train(init_params(3, np.random.default_rng(5)), X, y, steps=400)
p2, e = train_ckpt(init_params(3, np.random.default_rng(5)), X, y, steps=400, ckpt=100)
check("train_ckpt == core.train (bitwise)", all(np.array_equal(np.asarray(p1[k]), np.asarray(p2[k])) for k in p1))
check("checkpoint error is a valid error", 0 <= e <= 1)

# exact search == v1 brute_force_exact
A = train(init_params(3, np.random.default_rng(1)), X, y, steps=300)
B = train(init_params(3, np.random.default_rng(2)), X, y, steps=300)
for sign in (False, True):
    v1 = brute_force_exact(A, B, X, y, allow_sign=sign)
    v2, info = exact_search(A, B, X, y, allow_sign=sign)
    check(f"exact_search picks v1's alignment (sign={sign})",
          all(np.allclose(np.asarray(v1[k]), np.asarray(v2[k])) for k in v1))
from exact_search import transform_stack, stack_curves, fast_curves
S_, P_ = symmetries(3, True)
L1, E1 = stack_curves(A, transform_stack(B, S_, P_), X, y)
L2, E2 = fast_curves(A, B, S_, P_, X, y)
check("fast_curves == direct computation (all 48 candidates, 25 alphas)",
      np.allclose(L1, L2, atol=1e-12) and np.array_equal(E1, E2))
check("candidate count 2^h h! = 48", info["n_candidates"] == 48 and len(symmetries(3, True)[0]) == 48)
A4 = train(init_params(4, np.random.default_rng(3)), X, y, steps=200)
B4 = train(init_params(4, np.random.default_rng(4)), X, y, steps=200)
check("exact_search picks v1's alignment at width 4 (384 candidates)",
      all(np.allclose(np.asarray(brute_force_exact(A4, B4, X, y, True)[k]),
                      np.asarray(exact_search(A4, B4, X, y, True)[0][k])) for k in A4))

# a planted symmetry copy: exact_sign must find barrier 0, the flips, and H7's pieces
signs, perm = np.array([1, -1, -1]), np.array([2, 0, 1])
Bcopy = permute(flip(A, signs), perm)                      # same function as A
Bm, info = exact_search(A, Bcopy, X, y, True)
check("planted copy: exact_sign barrier 0", info["err_barrier"] == 0)
check("planted copy: aligned copy equals A", all(np.allclose(np.asarray(A[k]), np.asarray(Bm[k])) for k in A))
mech = mechanism(A, Bcopy, info["signs"], info["perm"], X, y)
check("planted copy: two units flipped", mech["n_flipped"] == 2)
check("planted copy: M = sum |s|/2 over flipped A units",
      np.isclose(mech["M"], np.sum(np.abs(A["w2"])[[1, 2]]) / 2))
Bperm = permute(A, perm)                                    # a copy with NO flips
_, info0 = exact_search(A, Bperm, X, y, True)
m0 = mechanism(A, Bperm, info0["signs"], info0["perm"], X, y)
check("permutation-only copy: no flips, M = 0, B_F = 0", m0["n_flipped"] == 0 and m0["M"] == 0 and abs(m0["B_F"]) < 1e-12)

# measures
mm = all_measures(A, B, X, y, X, y)
check("dense grids contain the 25 points, so barrier can only grow",
      mm["err_barrier_97"] >= mm["err_barrier"] - 1e-12 and mm["err_barrier_385"] >= mm["err_barrier_97"] - 1e-12)
check("81 calibration scales from 0.01 to 16", len(SCALES_V2) == 81 and np.isclose(SCALES_V2[0], 0.01) and np.isclose(SCALES_V2[-1], 16))
check("relative distance of a network to itself is 0", rel_distance(A, A) == 0)
check("identical endpoints: all barriers 0", all(abs(v) < 1e-12 for k, v in all_measures(A, A, X, y, X, y).items()))   # v1's barrier has ~1e-17 float noise

# excluded classifier
Nl = np.array([[1.0, 0.0], [0.0, 1.0], [-1 / np.sqrt(2), -1 / np.sqrt(2)]]); cl = np.array([-0.2, 0.4, 1.3 / np.sqrt(2)])
cov = edge_coverage(A, (Nl, cl))
check("edge coverage returns one distance per edge", cov.shape == (3,))
run = dict(params=A, train_err=0.10, train_err_ckpt=0.12)
check("changing error at checkpoint -> not converged (if not degenerate)",
      classify(run, X, (Nl, cl))["cls"] in ("a_not_converged", "c_degenerate"))
dead = {k: np.array(v, float) if k != "b2" else v for k, v in A.items()}; dead["w2"] = dead["w2"].copy(); dead["w2"][0] = 0.0
check("zero output weight -> degenerate", classify(dict(params=dead, train_err=0.1, train_err_ckpt=0.1), X, (Nl, cl))["cls"] == "c_degenerate")

# Spearman bootstrap
xs = np.arange(30.0); ys = xs + np.random.default_rng(0).normal(0, 3, 30)
rho, lo, hi = spearman_ci(xs, ys, n_boot=2000)
check("Spearman CI brackets rho and excludes 0 for a clear trend", lo <= rho <= hi and lo > 0)
print(f"\nall {ok} checks passed")

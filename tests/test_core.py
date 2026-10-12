"""Sanity tests. Run: python tests/test_core.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from core import (make_coffee, standardize, init_params, logits, bce_from_logits,
                  train, permute, flip, weight_match, brute_force, path, barrier,
                  lines, match_lines)

rng = np.random.default_rng(0)
X, y = make_coffee(500, 1)
X, mu, sd = standardize(X)


def test_permutation_and_flip_preserve_function():
    p = init_params(5, rng)
    p["b1"] = rng.normal(size=5); p["b2"] = 0.3
    q = flip(permute(p, [3, 0, 4, 1, 2]), [1, -1, -1, 1, -1])
    assert np.allclose(logits(p, X), logits(q, X))


def test_weight_match_recovers_scrambled_copy():
    p = init_params(6, rng); p["b1"] = rng.normal(size=6)
    q = permute(p, rng.permutation(6))
    assert np.allclose(weight_match(p, q)["W1"], p["W1"])
    r = flip(q, [-1, 1, -1, 1, 1, -1])
    m = weight_match(p, r, allow_sign=True)
    for k in ("W1", "b1", "w2"):
        assert np.allclose(m[k], p[k])
    assert np.isclose(m["b2"], p["b2"])


def test_barrier_zero_for_identical_models():
    p = init_params(3, rng)
    assert np.allclose(barrier(path(p, p, X, y, n_alpha=5)), 0)


def test_gradient_matches_finite_difference():
    p = init_params(3, rng); p["b1"] = rng.normal(size=3)
    # one tiny Adam-free check: numerical gradient of BCE w.r.t. w2[0]
    eps = 1e-6
    def L(q): return bce_from_logits(logits(q, X), y)
    q1 = {k: np.array(v, float) for k, v in p.items()}; q1["w2"][0] += eps
    q2 = {k: np.array(v, float) for k, v in p.items()}; q2["w2"][0] -= eps
    num = (L(q1) - L(q2)) / (2 * eps)
    from core import sigmoid
    a = sigmoid(X @ p["W1"].T + p["b1"]); dz = (sigmoid(logits(p, X)) - y) / len(y)
    assert np.isclose(num, (a.T @ dz)[0], rtol=1e-4)


def test_brute_force_finds_true_alignment():
    p = init_params(3, rng); p["b1"] = rng.normal(size=3)
    q = flip(permute(p, [2, 0, 1]), [-1, 1, -1])
    best = brute_force(p, q, X, y, allow_sign=True)
    # the search keeps the function and never does worse than the exact copy
    # (it can do better: a blend may beat both endpoints, so the midpoint can sit below the endpoints)
    assert np.allclose(logits(best, X), logits(p, X))
    from core import midpoint_loss
    assert midpoint_loss(p, best, X, y) <= midpoint_loss(p, p, X, y) + 1e-12


def test_line_matching_is_sign_invariant():
    p = init_params(3, rng); p["b1"] = rng.normal(size=3)
    ang, off = match_lines(lines(p), lines(flip(permute(p, [1, 2, 0]), [-1, -1, 1])))
    assert np.allclose(ang, 0, atol=1e-5) and np.allclose(off, 0, atol=1e-8)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("ok", name)

"""Checks for b1.py on made-up networks and made-up data (no coffee data).

python tests/test_b1.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from core import init_params, permute, flip, path, barrier, brute_force, train
from b1 import (output_weight_ratio, closest_unit_pair, min_activation_std,
                peak_alpha, brute_force_exact)

rng = np.random.default_rng(0)
X = rng.normal(size=(300, 2))
y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(float)


def test_output_weight_ratio():
    p = init_params(3, rng)
    p["w2"] = np.array([4.0, -0.2, 2.0])
    assert abs(output_weight_ratio(p) - 0.05) < 1e-12        # 0.2 / 4.0


def test_closest_unit_pair():
    p = init_params(3, rng)
    p["W1"][2], p["b1"][2] = p["W1"][0], p["b1"][0]          # unit 2 = unit 0
    assert closest_unit_pair(p) < 1e-6
    q = init_params(3, rng)
    q["W1"][2], q["b1"][2] = -q["W1"][0], -q["b1"][0]        # unit 2 = opposite of unit 0
    assert closest_unit_pair(q) < 1e-6


def test_min_activation_std():
    p = init_params(3, rng)
    p["W1"][1] = 0.0                                          # unit 1 ignores the input
    assert min_activation_std(p, X) < 1e-12


def test_peak_alpha():
    curve = np.zeros((5, 2))                                  # columns: loss, error
    curve[:, 1] = [0.0, 0.3, 0.1, 0.0, 0.0]                   # bump peaks at alpha = 0.25
    assert peak_alpha(curve) == 0.25


def test_exact_finds_hidden_copy():
    A = train(init_params(3, rng), X, y, steps=1500, lr=0.05)  # a trained net, so scrambled blends have real bumps
    B = permute(flip(A, np.array([1, -1, 1])), (2, 0, 1))     # same function, scrambled
    Bx = brute_force_exact(A, B, X, y, allow_sign=True)
    lb, eb = barrier(path(A, Bx, X, y))
    assert eb < 1e-9 and lb < 1e-9      # the hidden copy has zero barrier, so exact must reach zero
    # (it may return another version that also has zero barrier, so we don't compare weights)


def test_exact_never_worse_than_midpoint():
    for _ in range(3):
        A, B = init_params(3, rng), init_params(3, rng)
        for sign in (False, True):
            e_mid = barrier(path(A, brute_force(A, B, X, y, sign), X, y))[1]
            e_ex = barrier(path(A, brute_force_exact(A, B, X, y, sign), X, y))[1]
            assert e_ex <= e_mid + 1e-12


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t(); print("ok", t.__name__)
    print(f"{len(tests)}/{len(tests)} passed")

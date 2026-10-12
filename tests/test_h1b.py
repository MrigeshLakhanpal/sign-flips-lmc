"""Tests for src/h1b.py on made-up networks (not the coffee data).
Run: python tests/test_h1b.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from core import init_params, permute, flip, lines, match_lines
from h1b import pair_distance, median_distances, bootstrap_spearman

rng = np.random.default_rng(5)


def net(h=3):
    p = init_params(h, rng); p["b1"] = rng.normal(size=h); return p


def test_distance_of_a_network_to_its_own_copy_is_zero():
    p = net()
    q = flip(permute(p, [2, 0, 1]), [-1, 1, -1])   # same lines, shuffled and flipped
    assert np.isclose(pair_distance(p, q), 0, atol=1e-6)


def test_distance_uses_threshold_units_and_the_worst_line():
    for _ in range(20):
        p, q = net(), net()
        ang, off = match_lines(lines(p), lines(q))
        want = max(ang.max() / 10.0, off.max() / 0.25)
        got = pair_distance(p, q)
        assert isinstance(got, float), "return a plain float"
        assert np.isclose(got, want)
        assert np.isclose(pair_distance(q, p), got), "distance should not depend on the order"


def test_median_distances_skips_self_and_takes_the_median():
    ps = [net() for _ in range(5)]
    got = median_distances(ps)
    assert got.shape == (5,)
    for i in range(5):
        others = [pair_distance(ps[i], ps[j]) for j in range(5) if j != i]
        assert np.isclose(got[i], np.median(others))
    copies = [ps[0]] + [flip(ps[0], [1, -1, 1])] * 3 + [ps[1]]
    assert np.isclose(median_distances(copies)[0], 0, atol=1e-6), "a seed with 3 copies of itself among 4 others has median 0"


def test_bootstrap_spearman():
    x = np.arange(40.0)
    rho, lo, hi = bootstrap_spearman(x, x ** 2, n_boot=500)
    assert np.isclose(rho, 1) and lo <= rho <= hi
    rho, lo, hi = bootstrap_spearman(x, -x, n_boot=500)
    assert np.isclose(rho, -1)
    y = np.random.default_rng(1).normal(size=40)
    r1 = bootstrap_spearman(x, y, n_boot=500, seed=3)
    r2 = bootstrap_spearman(x, y, n_boot=500, seed=3)
    assert r1 == r2, "same seed must give the same interval"
    assert r1[1] < r1[0] < r1[2]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)

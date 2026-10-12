"""Tests for src/b4.py on made-up numbers. Run: python tests/test_b4.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from b4 import weight_norm, bootstrap_median_diff, verdict


def test_weight_norm():
    p = {"W1": np.array([[3.0, 0.0], [0.0, 0.0]]), "b1": np.array([100.0, 100.0]),
         "w2": np.array([0.0, 4.0]), "b2": 100.0}
    assert weight_norm(p) == 5.0                       # sqrt(3^2 + 4^2); biases ignored


def test_bootstrap_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(10, 1, 20); b = a - 3               # every pair drops by exactly 3
    lo, hi = bootstrap_median_diff(a, b, paired=True)
    assert abs(lo + 3) < 1e-9 and abs(hi + 3) < 1e-9
    lo, hi = bootstrap_median_diff(a, b, paired=False)
    assert lo < -3 < hi or np.isclose(lo, -3) or np.isclose(hi, -3)
    assert hi < 0


def test_verdict():
    assert verdict(2.0, -3.0, -1.0) == "lowers"
    assert verdict(-2.0, 1.0, 3.0) == "raises"
    assert verdict(0.3, -1.0, -0.1) == "no clear change"   # drop below EPS
    assert verdict(2.0, -3.0, 0.5) == "no clear change"    # interval includes 0


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t(); print("ok", t.__name__)
    print(f"{len(tests)}/{len(tests)} passed")

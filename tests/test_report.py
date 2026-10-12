"""Tests for src/report.py on made-up numbers. Run: python tests/test_report.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from report import median_ci, paired_diff_ci


def test_constant():
    lo, hi = median_ci(np.full(11, 3.0))
    assert lo == 3.0 and hi == 3.0                     # no spread, no uncertainty


def test_interval_contains_median_and_is_ordered():
    x = np.random.default_rng(1).normal(5, 2, 20)
    lo, hi = median_ci(x)
    assert lo <= np.median(x) <= hi and lo < hi


def test_more_data_narrower():
    rng = np.random.default_rng(2)
    small, big = rng.normal(0, 1, 10), rng.normal(0, 1, 200)
    w = lambda x: np.subtract(*median_ci(x)[::-1])
    assert w(big) < w(small)


def test_paired_shift():
    a = np.random.default_rng(3).normal(10, 3, 15)
    m, (lo, hi) = paired_diff_ci(a, a - 2)              # every pair drops by exactly 2
    assert m == -2 and np.isclose(lo, -2) and np.isclose(hi, -2)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t(); print("ok", t.__name__)
    print(f"{len(tests)}/{len(tests)} passed")

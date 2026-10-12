"""Tests for src/extras.py on made-up networks (not the coffee data).
Run: python tests/test_extras.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from core import init_params, permute, flip, weight_match, path, metrics, lerp, SCALES
from extras import count_flips, chosen_scales

rng = np.random.default_rng(9)
X = rng.normal(size=(300, 2))
y = (X[:, 0] - X[:, 1] > 0.3).astype(float)


def net(h):
    p = init_params(h, rng); p["b1"] = rng.normal(size=h); p["b2"] = float(rng.normal()); return p


def test_count_flips_on_known_alignments():
    p = net(5)
    assert count_flips(p, p) == 0
    assert count_flips(p, permute(p, [4, 2, 0, 1, 3])) == 0, "re-ordering alone is not a flip"
    assert count_flips(p, flip(p, [-1, 1, 1, 1, 1])) == 1
    assert count_flips(p, permute(flip(p, [-1, 1, -1, 1, -1]), [1, 0, 3, 4, 2])) == 3


def test_count_flips_on_real_matching():
    a, b = net(6), net(6)
    assert count_flips(b, weight_match(a, b)) == 0, "permutation-only matching never flips"
    m = weight_match(a, b, allow_sign=True)
    k = count_flips(b, m)
    assert 0 <= k <= 6


def test_chosen_scales_match_calibrated_path():
    a, b = net(3), net(3)
    s = chosen_scales(a, b, X, y, n_alpha=7)
    assert s.shape == (7,) and np.all(np.isin(s, SCALES))
    cal = path(a, b, X, y, n_alpha=7, calibrate=True)        # loss after calibration
    for k, al in enumerate(np.linspace(0, 1, 7)):
        assert np.isclose(metrics(lerp(a, b, al), X, y, s[k])[0], cal[k, 0]), "not the lowest-loss scale"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)

"""Tests for src/b2_figure.py on made-up networks (not the coffee data).
Run: python tests/test_b2.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from core import init_params, logits, sigmoid
from b2_figure import add_spare, slide, flip_one, spare_path, straight_path

rng = np.random.default_rng(7)
X = rng.normal(size=(300, 2))
y = (X[:, 0] + X[:, 1] > 0).astype(float)


def net():
    p = {k: 3 * np.asarray(v, float) for k, v in init_params(3, rng).items()}
    p["b1"] = rng.normal(size=3); p["b2"] = 0.4
    return p


def test_spare_changes_nothing():
    p = net()
    assert np.allclose(logits(add_spare(p, 1), X), logits(p, X))


def test_slide_is_flat_for_every_t():
    p = net(); i = 1; s, c = p["w2"][i], p["b2"]
    q = add_spare(p, i)
    for t in np.linspace(-2, 2, 9) * s:                     # even beyond [0, s]
        assert np.allclose(logits(slide(q, i, s, c, t), X), logits(p, X))


def test_end_is_flipped_copy():
    p = net(); i = 2; s, c = p["w2"][i], p["b2"]
    r = slide(add_spare(p, i), i, s, c, s)
    assert abs(r["w2"][i]) < 1e-12                          # unit i is dead
    f = flip_one(p, i)                                       # spare must equal the flipped unit
    assert np.allclose(r["W1"][-1], f["W1"][i]) and np.allclose(r["b1"][-1], f["b1"][i])
    assert np.isclose(r["w2"][-1], f["w2"][i]) and np.isclose(r["b2"], f["b2"])


def test_paths():
    p = net(); i = 0
    assert spare_path(p, i, X, y)[:, 0].max() < 1e-9
    mid = straight_path(p, i, X, y)[12, 0]                   # alpha = 0.5
    z = X @ p["W1"][i] + p["b1"][i]
    expected = np.abs(p["w2"][i] * (sigmoid(z) - 0.5)).max() # Step 4: |s (sigma(z) - 1/2)|
    assert np.isclose(mid, expected)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t(); print("ok", t.__name__)
    print(f"{len(tests)}/{len(tests)} passed")

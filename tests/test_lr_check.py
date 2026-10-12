"""Tests for src/lr_check.py. Run: python tests/test_lr_check.py
Uses a made-up task (not the coffee data), so it reveals nothing about the experiment."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from core import init_params, train, logits, bce_from_logits
from lr_check import train_with_history, same_params, curve_stats, passes

rng = np.random.default_rng(7)
X = rng.normal(size=(300, 2))
y = ((X[:, 0] + 0.5 * X[:, 1] > 0.2) & (X[:, 1] < 1.0)).astype(float)


def test_history_training_equals_core_train():
    for h, lr, wd in [(3, 0.01, 0.0), (8, 0.05, 0.0), (5, 0.01, 0.003)]:
        p, hist = train_with_history(init_params(h, np.random.default_rng(h)), X, y, 200, lr, wd)
        ref = train(init_params(h, np.random.default_rng(h)), X, y, 200, lr, wd)
        assert same_params(p, ref)
        assert len(hist) == 201


def test_history_endpoints_are_the_real_losses():
    p0 = init_params(4, np.random.default_rng(1))
    start = bce_from_logits(logits(p0, X), y)
    p, hist = train_with_history(p0, X, y, 150)
    assert np.isclose(hist[0], start)
    assert np.isclose(hist[-1], bce_from_logits(logits(p, X), y))


def test_rule_accepts_smooth_curve_and_rejects_spike_and_nan():
    smooth = np.exp(-np.linspace(0, 5, 1000)) + 0.01
    assert passes(curve_stats(smooth))
    spiky = smooth.copy(); spiky[600] *= 1.5
    s = curve_stats(spiky)
    expected = spiky[600] / smooth[599] - 1          # rise above the lowest loss seen before the spike
    assert not passes(s) and s["rebound_step"] == 600 and np.isclose(s["rebound"], expected)
    rising = smooth.copy(); rising[800:] = rising[799] * np.linspace(1, 1.2, 200)   # climbs 20% at the end
    assert not passes(curve_stats(rising))
    broken = smooth.copy(); broken[10] = np.nan
    assert not passes(curve_stats(broken))


def test_rule_ignores_the_burn_in():
    c = np.exp(-np.linspace(0, 5, 1000)) + 0.01
    c[50] *= 3.0
    assert passes(curve_stats(c))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
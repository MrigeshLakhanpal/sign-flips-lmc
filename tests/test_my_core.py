"""Checks src/my_core.py against src/core.py on made-up networks and data
(not the coffee data). Run: python tests/test_my_core.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import core, my_core
from core import init_params, logits, permute

rng = np.random.default_rng(11)
X = rng.normal(size=(200, 2))


def rand_net(h):
    p = init_params(h, rng)
    p["b1"] = rng.normal(size=h); p["b2"] = float(rng.normal())
    return p


def same(p, q):
    return all(np.allclose(np.asarray(p[k]), np.asarray(q[k])) for k in ("W1", "b1", "w2", "b2"))


def test_flip_keeps_the_function_and_matches_core():
    for h in (2, 3, 6):
        p = rand_net(h); before = {k: np.array(v, float) for k, v in p.items()}
        s = rng.choice([-1, 1], size=h); s[0] = -1
        q = my_core.flip(p, s)
        assert np.allclose(logits(q, X), logits(p, X)), "flip changed the function"
        assert same(q, core.flip(p, s)), "flip differs from core.flip"
        assert same(p, before), "flip changed its input network"
        assert same(my_core.flip(q, s), p), "flipping twice should give the original back"


def test_weight_match_recovers_a_scrambled_copy():
    for h in (3, 6, 16):
        p = rand_net(h)
        q = permute(p, rng.permutation(h))
        assert same(my_core.weight_match(p, q), p), "permutation-only matching failed"
        r = core.flip(q, rng.choice([-1, 1], size=h))
        assert same(my_core.weight_match(p, r, allow_sign=True), p), "sign-aware matching failed"


def test_weight_match_agrees_with_core_on_independent_networks():
    for h in (3, 8):
        for sign in (False, True):
            a, b = rand_net(h), rand_net(h)
            m = my_core.weight_match(a, b, allow_sign=sign)
            assert np.allclose(logits(m, X), logits(b, X)), "matching changed B's function"
            assert same(m, core.weight_match(a, b, allow_sign=sign)), "differs from core.weight_match"


def test_barrier_matches_core_and_known_cases():
    a = np.linspace(0, 1, 25)
    slope = np.column_stack([0.1 + 0.2 * a, 0.3 * a])            # straight slope: no barrier
    assert np.allclose(my_core.barrier(slope), 0)
    bump = slope + np.column_stack([0.5 * np.sin(np.pi * a), 0.2 * np.sin(np.pi * a)])
    assert np.allclose(my_core.barrier(bump), [0.5, 0.2])        # bump height above the slope
    dip = slope - 0.3 * np.sin(np.pi * a)[:, None]
    assert np.allclose(my_core.barrier(dip), 0)                  # a dip gives 0 here (endpoints)
    c = rng.random((25, 2))
    assert np.allclose(my_core.barrier(c), core.barrier(c))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
"""Two small measurements added before the main run (both fixed before the main run).
Check with:  python tests/test_extras.py   (made-up networks, no coffee data)
"""
import numpy as np
from core import unit_vectors, lerp, metrics, SCALES


def count_flips(pB, pB_aligned):
    """How many hidden units of B were sign-flipped by an alignment.

    Every unit of the aligned network is one of B's units, either as it was
    (same row [W1, b1, w2]) or flipped (the same row with every sign reversed).
    Re-ordering doesn't matter: we only ask, for each aligned unit, whether it
    equals the NEGATIVE of some unit of B.
    Works for any alignment mode (wm, wm_sign, bf_perm, bf_sign).
    """
    uB, uM = unit_vectors(pB), unit_vectors(pB_aligned)
    n = 0
    for row in uM:                                   # each unit of the aligned network
        if any(np.allclose(row, -u) for u in uB):
            n += 1
    return n


def chosen_scales(pA, pB, X, y, n_alpha=25):
    """For each blend point, the output scale that calibration picks.

    Calibration tries every scale in SCALES (41 values from 0.25 to 8) and keeps
    the one with the LOWEST loss. metrics(p, X, y, s) returns (loss, error).
    Returns an array of n_alpha scales, one per blend point.
    """
    out = []
    for a in np.linspace(0, 1, n_alpha):
        p = lerp(pA, pB, a)
        losses = [metrics(p, X, y, s)[0] for s in SCALES]
        out.append(SCALES[np.argmin(losses)])
    return np.array(out)

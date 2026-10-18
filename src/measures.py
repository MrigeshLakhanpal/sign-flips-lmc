"""v2 measures for one pair and one alignment (v2_rules.md, Section 5). Written by Claude, 19 Oct 2026.

All barriers are Entezari Eq. 1 (core.barrier), returned as fractions (x 100 = points).
Calibration: at each alpha, the best of 81 log-spaced output scales from 0.01 to 16 (v1 used 41, 0.25-8).
"""
import numpy as np
from core import path, barrier, lerp, logits, bce_from_logits

SCALES_V2 = np.exp(np.linspace(np.log(0.01), np.log(16), 81))


def calibrated_curve(pA, pB, X, y, n_alpha=25, scales=SCALES_V2):
    """Per alpha: the lowest loss over the output scales, and whether that scale sits at a grid edge."""
    losses, edge = [], 0
    for a in np.linspace(0, 1, n_alpha):
        z = logits(lerp(pA, pB, a), X)
        Ls = np.array([bce_from_logits(z * s, y) for s in scales])
        i = int(np.argmin(Ls))
        losses.append(Ls[i]); edge += int(i in (0, len(scales) - 1))
    return np.array(losses), edge


def _barrier_1d(c):
    a = np.linspace(0, 1, len(c))
    return float((c - ((1 - a) * c[0] + a * c[-1])).max())


def all_measures(pA, pB, Xtr, ytr, Xte, yte, n_alpha=25, dense=(97, 385)):
    lb, eb = barrier(path(pA, pB, Xtr, ytr, n_alpha))
    _, teb = barrier(path(pA, pB, Xte, yte, n_alpha))
    cal, edge = calibrated_curve(pA, pB, Xtr, ytr, n_alpha)
    out = dict(err_barrier=float(eb), test_err_barrier=float(teb), loss_barrier=float(lb),
               cal_loss_barrier=_barrier_1d(cal), cal_edge_hits=edge)
    for g in dense:
        out[f"err_barrier_{g}"] = float(barrier(path(pA, pB, Xtr, ytr, g))[1])
    return out


def flat(p):
    return np.concatenate([np.ravel(p["W1"]), np.ravel(p["b1"]), np.ravel(p["w2"]), [float(p["b2"])]])


def rel_distance(pA, pB):
    """||theta_A - theta_B|| / ||theta_A|| (audit F14), all parameters including biases."""
    return float(np.linalg.norm(flat(pA) - flat(pB)) / np.linalg.norm(flat(pA)))

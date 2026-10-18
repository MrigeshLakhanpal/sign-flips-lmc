"""v2 classification of networks that fail inclusion (v2_rules.md, Section 7). Written by Claude, 19 Oct 2026.

Order (first match wins):
  (c) degenerate      any nearly-reducible flag (b1.irreducibility, v1 thresholds)
  (a) not converged   training error 300 steps before the end differs from the final training error
  (b) wrong minimum   some ground-truth edge has no unit line within H1's tolerance (10 deg, 0.25)
  other
"""
import numpy as np
from core import lines
from b1 import irreducibility, ANGLE_T, OFFSET_T


def edge_coverage(p, edges_std):
    """For each ground-truth edge, the distance (threshold units) to the closest unit line. <= 1 = covered."""
    n, c = lines(p)
    En, Ec = edges_std
    dot = En @ n.T                                            # (edges, units)
    ang = np.degrees(np.arccos(np.clip(np.abs(dot), 0, 1)))
    off = np.abs(Ec[:, None] - np.sign(dot) * c[None, :])
    return np.maximum(ang / ANGLE_T, off / OFFSET_T).min(1)


def classify(run, X, edges_std):
    irr = irreducibility(run["params"], X)
    cov = edge_coverage(run["params"], edges_std)
    if irr["nearly_reducible"]:
        cls = "c_degenerate"
    elif run["train_err_ckpt"] != run["train_err"]:
        cls = "a_not_converged"
    elif np.any(cov > 1):
        cls = "b_wrong_minimum"
    else:
        cls = "other"
    return dict(cls=cls, edges_uncovered=int(np.sum(cov > 1)), **irr)

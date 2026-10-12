"""Day 2 exercise: flip, weight_match and barrier written by hand, as a check on core.py.
Not used by the pipeline. Check with:
    python tests/test_my_core.py

Nothing here touches the coffee data or the experiment. It is practice, written to
understand every line of the three functions the results depend on.

A network is a dict:  W1 (h,2)  b1 (h,)  w2 (h,)  b2 (float)
Output logit:         z(x) = sum_i w2[i] * sigmoid(W1[i] . x + b1[i]) + b2
"""
import numpy as np
from scipy.optimize import linear_sum_assignment   # needed for weight_match
from core import permute, unit_vectors              # shared helpers from core.py


def flip(p, signs):
    """Return a new network in which every unit i with signs[i] == -1 is flipped,
    and the function z(x) is unchanged.

    signs: list or array of +1 / -1, one per hidden unit.
    Fact to use: sigmoid(-z) = 1 - sigmoid(z).
    Work out on paper what must happen to W1[i], b1[i], w2[i] and b2.
    Do not change p itself; return a new dict.
    """
    W1 = p["W1"].copy()
    b1 = p["b1"].copy()
    w2 = p["w2"].copy()
    b2 = p["b2"]
    for i, s in enumerate(signs):
        if s == -1:
            W1[i] *= -1
            b1[i] *= -1
            b2 += w2[i]
            w2[i] *= -1
    return {"W1": W1, "b1": b1, "w2": w2, "b2": b2}

def weight_match(pA, pB, allow_sign=False):
    """Return B re-ordered (and, if allow_sign, flipped) to line up with A.

    Steps:
    1. uA = unit_vectors(pA), uB = unit_vectors(pB): one row [W1, b1, w2] per unit.
    2. S = similarity matrix, S[i, j] = dot product of A's unit i and B's unit j.
    3. Find the one-to-one pairing with the largest total similarity.
       linear_sum_assignment(cost) MINIMISES total cost, so think about the sign.
       With allow_sign, a pair's score is |S[i, j]| (a flipped unit has -u).
    4. perm[i] = the B unit paired with A's unit i.
    5. With allow_sign, flip each paired B unit whose S[i, perm[i]] is negative.
    6. Return the flipped, then permuted, B. Use your own flip and core's permute.
    """
    uA = unit_vectors(pA)
    uB = unit_vectors(pB)
    S = uA @ uB.T
    rows, cols = linear_sum_assignment(-(np.abs(S) if allow_sign else S))
    perm = cols[np.argsort(rows)]
    signs_B = np.ones(len(perm))
    if allow_sign:
        for i, j in enumerate(perm):
            if S[i, j] < 0:
                signs_B[j] = -1
    return permute(flip(pB, signs_B), perm)


def barrier(curve):
    """Entezari et al., Eq. 1. curve has one row per blend point, alpha going
    evenly from 0 (network A) to 1 (network B), and two columns: loss, error.

    Return an array of two numbers (loss barrier, error barrier): the largest
    height of the curve above the straight line joining its first and last row.
    """
    start, end = curve[0], curve[-1]
    slope = np.linspace(start, end, len(curve))
    return np.max(curve - slope, axis=0)
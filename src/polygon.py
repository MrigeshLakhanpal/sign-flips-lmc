"""v2 tasks: convex k-gon classification, plus the coffee triangle's edges (v2_rules.md, Section 2).
Written by Claude, 19 Oct 2026, after the pre-registration commit 349aa3a.

Polygon P_k (deterministic):
  angles theta_j = 2*pi*j/k + phi_j, phi_j ~ U(-0.2*2pi/k, 0.2*2pi/k); radii r_j ~ U(0.85, 1.15);
  centre (0.10, -0.05); radii scaled so the area is 0.84 (21% of the square [-1, 1]^2).
  Accepted only if convex, inside [-0.95, 0.95]^2, every interior angle <= 165 deg and every edge >= 0.25.
  Otherwise redraw with seed + 100. First seed: 7000 + k.
Label: 1 strictly inside P_k. Inputs x ~ U([-1, 1]^2); train seed 100*k, test seed 100*k + 1 (DEVIATIONS.md, D1).
"""
import numpy as np

AREA_TARGET = 0.84
CENTER = np.array([0.10, -0.05])
MAX_ANGLE_DEG, MIN_EDGE, BOX = 165.0, 0.25, 0.95


def area(V):
    x, y = V[:, 0], V[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))   # > 0 for counter-clockwise


def interior_angles(V):
    k, out = len(V), []
    for j in range(k):
        a, b = V[j - 1] - V[j], V[(j + 1) % k] - V[j]
        out.append(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))
    return np.array(out)


def edge_lengths(V):
    return np.linalg.norm(np.roll(V, -1, axis=0) - V, axis=1)


def is_convex_ccw(V):
    E = np.roll(V, -1, axis=0) - V
    cross = E[:, 0] * np.roll(E, -1, axis=0)[:, 1] - E[:, 1] * np.roll(E, -1, axis=0)[:, 0]
    return bool(np.all(cross > 0))


def accepted(V):
    return (is_convex_ccw(V) and np.all(np.abs(V) <= BOX)
            and np.all(interior_angles(V) <= MAX_ANGLE_DEG) and np.all(edge_lengths(V) >= MIN_EDGE))


def make_polygon(k, max_tries=1000):
    """Returns (vertices (k, 2) counter-clockwise, seed that was accepted)."""
    seed = 7000 + k
    for _ in range(max_tries):
        rng = np.random.default_rng(seed)
        phi = rng.uniform(-0.2 * 2 * np.pi / k, 0.2 * 2 * np.pi / k, k)
        r = rng.uniform(0.85, 1.15, k)
        th = 2 * np.pi * np.arange(k) / k + phi
        U = np.column_stack([r * np.cos(th), r * np.sin(th)])
        V = CENTER + U * np.sqrt(AREA_TARGET / area(U))
        if accepted(V):
            return V, seed
        seed += 100
    raise RuntimeError(f"no accepted polygon for k={k}")


def edges_raw(V):
    """Each edge as (n, c) with n.x + c > 0 on the inside (n points inward, raw input space)."""
    D = np.roll(V, -1, axis=0) - V
    N = np.column_stack([-D[:, 1], D[:, 0]])          # inward normal for a counter-clockwise polygon
    return N, -(N * V).sum(1)


def label(X, V):
    N, c = edges_raw(V)
    return (X @ N.T + c > 0).all(1).astype(float)


def make_polygon_data(k, n, seed, V):
    X = np.random.default_rng(seed).uniform(-1, 1, (n, 2))
    return X, label(X, V)


def coffee_edges_raw():
    """Coffee triangle (v1, make_coffee): t > 175, d > 12, d <= -3/85 t + 21 (the other two conditions are redundant)."""
    N = np.array([[1.0, 0.0], [0.0, 1.0], [-3 / 85, -1.0]])
    c = np.array([-175.0, -12.0, 21.0])
    return N, c


def to_standardized(N, c, mu, sd):
    """Edge lines in standardized space, as unit normal n and offset c (same convention as core.lines):
    x_raw = x_std * sd + mu, so n.(x_std*sd + mu) + c = (n*sd).x_std + (n.mu + c)."""
    Ns, cs = N * sd, N @ mu + c
    norm = np.linalg.norm(Ns, axis=1)
    return Ns / norm[:, None], cs / norm

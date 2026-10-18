"""v2 Stage 1: train every network (v2_rules.md, Sections 1-3). Written by Claude, 19 Oct 2026.

python src/train_v2.py --tasks T0 T3 T4 T5 T6            # main
python src/train_v2.py --tasks T5 --steps 10000           # ONLY if Section 3.4 is triggered (stage1.py says so)

Training is identical to v1 (core.train): full-batch Adam, lr 0.01, BCE, Glorot init, zero biases.
The only addition is recording the training error 300 steps before the end (Section 7, class (a)).
T0 retrains seeds 0-39 too (deterministic) so their step-2700 error exists; their final parameters are
compared with v1's results/models.pkl when that file is present (DEVIATIONS.md, D2).
Saves results_v2/models_<task>.pkl.
"""
import argparse, pathlib, pickle, time
import numpy as np
from core import make_coffee, standardize, init_params, sigmoid, metrics
from polygon import make_polygon, make_polygon_data

TASK_WIDTHS = {"T0": [3], "T3": [2, 3, 4, 5], "T4": [2, 3, 4, 5, 6], "T5": [3, 4, 5, 6, 7], "T6": [4, 5, 6, 7, 8]}


def train_ckpt(p, X, y, steps=3000, lr=0.01, ckpt=2700):
    """Exactly core.train (wd = 0), plus the training error after `ckpt` steps."""
    keys = ["W1", "b1", "w2", "b2"]
    m = {k: np.zeros_like(np.asarray(p[k], float)) for k in keys}
    v = {k: np.zeros_like(np.asarray(p[k], float)) for k in keys}
    b1_, b2_, eps, n = 0.9, 0.999, 1e-8, len(y)
    err_ckpt = None
    for t in range(1, steps + 1):
        a = sigmoid(X @ p["W1"].T + p["b1"])
        z = a @ p["w2"] + p["b2"]
        dz = (sigmoid(z) - y) / n
        g = {"w2": a.T @ dz, "b2": dz.sum()}
        da = np.outer(dz, p["w2"]) * a * (1 - a)
        g["W1"], g["b1"] = da.T @ X, da.sum(0)
        for k in keys:
            gk = g[k]
            m[k] = b1_ * m[k] + (1 - b1_) * gk
            v[k] = b2_ * v[k] + (1 - b2_) * gk ** 2
            step = lr * (m[k] / (1 - b1_ ** t)) / (np.sqrt(v[k] / (1 - b2_ ** t)) + eps)
            p[k] = p[k] - step
        if t == ckpt:
            err_ckpt = float(metrics(p, X, y)[1])
    p["b2"] = float(p["b2"])
    return p, err_ckpt


def task_data(task, ntrain=2000, ntest=2000):
    if task == "T0":
        Xtr, ytr = make_coffee(ntrain, seed=0)
        Xte, yte = make_coffee(ntest, seed=1)
        V = None
    else:
        k = int(task[1:])
        V, _ = make_polygon(k)
        Xtr, ytr = make_polygon_data(k, ntrain, 100 * k, V)
        Xte, yte = make_polygon_data(k, ntest, 100 * k + 1, V)
    Xtr, mu, sd = standardize(Xtr)
    Xte, _, _ = standardize(Xte, mu, sd)
    return Xtr, ytr, Xte, yte, mu, sd, V


def init_seed(task, h, s):
    return 1000 * h + s if task == "T0" else 100000 * int(task[1:]) + 1000 * h + s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["T0", "T3", "T4", "T5", "T6"])
    ap.add_argument("--seeds", type=int, default=None, help="default: 120 for T0, 100 for polygons")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--out-dir", default="results_v2")
    ap.add_argument("--v1-models", default="results/models.pkl")
    a = ap.parse_args()
    out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    for task in a.tasks:
        Xtr, ytr, Xte, yte, mu, sd, V = task_data(task)
        n_seeds = a.seeds or (120 if task == "T0" else 100)
        runs, t0 = [], time.time()
        for h in TASK_WIDTHS[task]:
            for s in range(n_seeds):
                p, e_ck = train_ckpt(init_params(h, np.random.default_rng(init_seed(task, h, s))),
                                     Xtr, ytr, steps=a.steps, ckpt=a.steps - 300)
                (ltr, etr), (lte, ete) = metrics(p, Xtr, ytr), metrics(p, Xte, yte)
                runs.append(dict(width=h, seed=s, params=p, train_loss=ltr, train_err=etr,
                                 test_loss=lte, test_err=ete, train_err_ckpt=e_ck))
            print(f"{task} width {h}: {n_seeds} seeds ({time.time() - t0:.0f}s)", flush=True)

        if task == "T0" and pathlib.Path(a.v1_models).exists():          # D2: reproduction check
            D1 = pickle.load(open(a.v1_models, "rb"))
            old = {r["seed"]: r["params"] for r in D1["runs"] if r["width"] == 3}
            diff = max(max(np.max(np.abs(np.asarray(r["params"][k]) - np.asarray(old[r["seed"]][k])))
                           for k in ("W1", "b1", "w2", "b2"))
                       for r in runs if r["seed"] in old)
            print(f"T0 seeds 0-39 vs v1 models.pkl: max |difference| = {diff:.3e}"
                  f"  ({'IDENTICAL' if diff == 0 else 'DIFFERENT - report this'})")

        with open(out / f"models_{task}.pkl", "wb") as f:
            pickle.dump(dict(task=task, config=vars(a), mu=mu, sd=sd, vertices=V, runs=runs,
                             data=dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)), f)
        print("saved", out / f"models_{task}.pkl")

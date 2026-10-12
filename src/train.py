"""Train many seeds at many widths. Saves every model to results/models.pkl.

python src/train.py --widths 2 3 4 6 8 16 32 64 --seeds 40
"""
import argparse, pickle, time, pathlib
import numpy as np
from core import make_coffee, standardize, init_params, train, metrics

ap = argparse.ArgumentParser()
ap.add_argument("--widths", type=int, nargs="+", default=[2, 3, 4, 6, 8, 16, 32, 64])
ap.add_argument("--seeds", type=int, default=40)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--lr", type=float, default=0.01)
ap.add_argument("--wd", type=float, default=0.0)
ap.add_argument("--ntrain", type=int, default=2000)
ap.add_argument("--ntest", type=int, default=2000)
ap.add_argument("--out", default="results/models.pkl")
a = ap.parse_args()

Xtr, ytr = make_coffee(a.ntrain, seed=0)        # data are fixed across all runs
Xte, yte = make_coffee(a.ntest, seed=1)
Xtr, mu, sd = standardize(Xtr)
Xte, _, _ = standardize(Xte, mu, sd)

runs = []
t0 = time.time()
for h in a.widths:
    for s in range(a.seeds):
        p = train(init_params(h, np.random.default_rng(1000 * h + s)), Xtr, ytr,
                  steps=a.steps, lr=a.lr, wd=a.wd)
        (ltr, etr), (lte, ete) = metrics(p, Xtr, ytr), metrics(p, Xte, yte)
        runs.append(dict(width=h, seed=s, params=p, train_loss=ltr, train_err=etr,
                         test_loss=lte, test_err=ete))
    print(f"width {h:>3}: {a.seeds} seeds done ({time.time() - t0:.0f}s)")

pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "wb") as f:
    pickle.dump(dict(config=vars(a), mu=mu, sd=sd, runs=runs,
                     data=dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)), f)
print("saved", a.out)

"""B2 figure: the spare-neuron path (flat) next to the straight blend (bump).
Derivation: Section 5 of the paper.
Check with:  python tests/test_b2.py   (made-up networks, no coffee data)
Then run:    python src/b2_figure.py   (one trained width-3 network from results/models.pkl)
Writes figures/b2_spare_path.png and prints the numbers for the caption.
"""
import pickle, pathlib
import numpy as np
from core import logits, metrics, flip, lerp

N = 25   # points along each path, same as the barriers


def add_spare(p, i):
    """Width r -> r + 1: append a spare unit that is the opposite of unit i, output weight 0."""
    return {"W1": np.vstack([p["W1"], -p["W1"][i]]),
            "b1": np.append(p["b1"], -p["b1"][i]),
            "w2": np.append(p["w2"], 0.0),
            "b2": p["b2"]}


def slide(q, i, s, c, t):
    """Step 2 of the lemma at position t (q is the network with the spare as its last unit)."""
    r = {k: np.array(v, dtype=float, copy=True) for k, v in q.items()}
    r["w2"][i] = s - t
    r["w2"][-1] = -t
    r["b2"] = c + t
    return r


def flip_one(p, i):
    signs = np.ones(len(p["b1"])); signs[i] = -1
    return flip(p, signs)


def spare_path(p, i, X, y):
    """Max |logit change| and training error (%) along the spare-neuron path, t from 0 to s."""
    s, c, z0 = float(p["w2"][i]), float(p["b2"]), logits(p, X)
    q = add_spare(p, i)
    out = []
    for t in np.linspace(0, s, N):
        r = slide(q, i, s, c, t)
        out.append((np.abs(logits(r, X) - z0).max(), 100 * metrics(r, X, y)[1]))
    return np.array(out)


def straight_path(p, i, X, y):
    """Same two numbers along the straight blend from p to its copy with unit i flipped."""
    B, z0 = flip_one(p, i), logits(p, X)
    out = []
    for a in np.linspace(0, 1, N):
        m = lerp(p, B, a)
        out.append((np.abs(logits(m, X) - z0).max(), 100 * metrics(m, X, y)[1]))
    return np.array(out)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = pickle.load(open("results/models.pkl", "rb"))
    X, y = D["data"]["Xtr"], D["data"]["ytr"]
    run = next(r for r in D["runs"] if r["width"] == 3 and r["train_err"] <= 0.02)  # first included width-3 seed
    p = run["params"]
    i = int(np.argmax(np.abs(p["w2"])))   # the unit with the largest output weight: the clearest example
    sp, st = spare_path(p, i, X, y), straight_path(p, i, X, y)
    x = np.linspace(0, 1, N)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for k, lab in enumerate(["max |logit change| over training inputs", "training error (%)"]):
        ax[k].plot(x, st[:, k], "o-", ms=3, label="straight blend to flipped copy (width 3)")
        ax[k].plot(x, sp[:, k], "s-", ms=3, label="spare-neuron path (width 4)")
        ax[k].set_xlabel("position along path"); ax[k].set_ylabel(lab)
    ax[0].legend(fontsize=8)
    fig.suptitle(f"Width-3 seed {run['seed']}, unit {i} flipped (output weight s = {p['w2'][i]:.2f})")
    fig.tight_layout()
    pathlib.Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/b2_spare_path.png", dpi=150)
    print(f"seed {run['seed']}, unit {i}, s = {p['w2'][i]:.3f}, |s|/2 = {abs(p['w2'][i]) / 2:.3f}")
    print(f"spare path:    max logit change {sp[:, 0].max():.2e}, error {sp[0, 1]:.2f}% to {sp[:, 1].max():.2f}%")
    print(f"straight path: max logit change {st[:, 0].max():.3f} (midpoint {st[N // 2, 0]:.3f}), "
          f"peak error {st[:, 1].max():.2f}% (ends {st[0, 1]:.2f}%)")
    print("wrote figures/b2_spare_path.png")


if __name__ == "__main__":
    main()

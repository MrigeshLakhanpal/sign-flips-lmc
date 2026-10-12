"""Day 0 pre-run check: are the Adam training curves smooth at lr = 0.01?

Run from the repo's top folder:
    python src/lr_check.py

Writes figures/lr_check.png and results/lr_check.csv, and prints PASS or REVIEW.

What it looks at: the training LOSS curve of 5 seeds at widths 3 and 64, using
exactly the data, seeds and settings of src/train.py. It computes no barriers,
no alignments and no training error, so it cannot leak anything about Q1 to Q4.

The decision rule is fixed in this file BEFORE the check is run (see RULE below).
The rule and its outcome were dated in the research notebook.
"""
import argparse, csv, pathlib
import numpy as np
from core import make_coffee, standardize, init_params, train, sigmoid, bce_from_logits

# ---------------------------------------------------------------- the rule
BURN_IN = 200        # steps ignored at the start, while the loss is still falling fast
MAX_REBOUND = 0.10   # largest allowed rise above the lowest loss seen so far (10%)
MAX_FINAL = 1.05     # final loss must be within 5% of the lowest loss of the run
RULE = (f"Keep lr if, for every run: the loss is always finite; after step {BURN_IN} "
        f"the loss never rises more than {MAX_REBOUND:.0%} above the lowest loss seen "
        f"so far; and the final loss is within {MAX_FINAL - 1:.0%} of the lowest loss.")


def train_with_history(p, X, y, steps=3000, lr=0.01, wd=0.0):
    """Same update as core.train, line for line, plus the loss at every step.
    history[t] is the loss BEFORE update t+1; the last entry is the final loss."""
    keys = ["W1", "b1", "w2", "b2"]
    m = {k: np.zeros_like(np.asarray(p[k], float)) for k in keys}
    v = {k: np.zeros_like(np.asarray(p[k], float)) for k in keys}
    b1_, b2_, eps, n = 0.9, 0.999, 1e-8, len(y)
    history = np.empty(steps + 1)
    for t in range(1, steps + 1):
        a = sigmoid(X @ p["W1"].T + p["b1"])
        z = a @ p["w2"] + p["b2"]
        history[t - 1] = bce_from_logits(z, y)
        dz = (sigmoid(z) - y) / n
        g = {"w2": a.T @ dz, "b2": dz.sum()}
        da = np.outer(dz, p["w2"]) * a * (1 - a)
        g["W1"], g["b1"] = da.T @ X, da.sum(0)
        for k in keys:
            gk = g[k] + wd * p[k] if k in ("W1", "w2") else g[k]
            m[k] = b1_ * m[k] + (1 - b1_) * gk
            v[k] = b2_ * v[k] + (1 - b2_) * gk ** 2
            step = lr * (m[k] / (1 - b1_ ** t)) / (np.sqrt(v[k] / (1 - b2_ ** t)) + eps)
            p[k] = p[k] - step
    p["b2"] = float(p["b2"])
    z = sigmoid(X @ p["W1"].T + p["b1"]) @ p["w2"] + p["b2"]
    history[steps] = bce_from_logits(z, y)
    return p, history


def same_params(p, q):
    return all(np.array_equal(np.asarray(p[k]), np.asarray(q[k])) for k in p)


def curve_stats(history, burn_in=BURN_IN):
    """Numbers the rule is applied to. rebound = largest rise of the loss above
    the lowest loss seen so far, as a fraction of that lowest loss."""
    finite = bool(np.all(np.isfinite(history)))
    if not finite:
        return dict(finite=False, min_loss=np.nan, final_loss=np.nan,
                    rebound=np.inf, rebound_step=-1, final_over_min=np.inf)
    running_min = np.minimum.accumulate(history)
    rel = (history - running_min) / running_min
    rel[:burn_in] = 0.0
    k = int(np.argmax(rel))
    return dict(finite=True, min_loss=float(history.min()), final_loss=float(history[-1]),
                rebound=float(rel[k]), rebound_step=k,
                final_over_min=float(history[-1] / history.min()))


def passes(s):
    return bool(s["finite"] and s["rebound"] <= MAX_REBOUND and s["final_over_min"] <= MAX_FINAL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="+", default=[3, 64])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _lr0.001")
    a = ap.parse_args()

    Xtr, ytr = make_coffee(2000, seed=0)          # identical to src/train.py
    Xtr, _, _ = standardize(Xtr)

    rows, curves = [], {}
    for h in a.widths:
        for s in range(a.seeds):
            seed = 1000 * h + s                    # identical to src/train.py
            p, hist = train_with_history(init_params(h, np.random.default_rng(seed)),
                                         Xtr, ytr, a.steps, a.lr, a.wd)
            ref = train(init_params(h, np.random.default_rng(seed)), Xtr, ytr, a.steps, a.lr, a.wd)
            if not same_params(p, ref):
                raise SystemExit(f"STOP: lr_check's training differs from core.train "
                                 f"(width {h}, seed {s}). Do not trust this check.")
            st = curve_stats(hist)
            rows.append(dict(width=h, seed=s, lr=a.lr, wd=a.wd, steps=a.steps,
                             ok=passes(st), **st))
            curves[(h, s)] = hist
        print(f"width {h:>3}: {a.seeds} seeds trained")

    pathlib.Path("results").mkdir(exist_ok=True)
    pathlib.Path("figures").mkdir(exist_ok=True)
    out_csv = f"results/lr_check{a.tag}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, grid, line, bad = "#1f2933", "#52606d", "#d9dee5", "#2f6fde", "#b4233c"
    fig, axes = plt.subplots(1, len(a.widths), figsize=(5.2 * len(a.widths), 3.8),
                             sharey=True, squeeze=False)
    for ax, h in zip(axes[0], a.widths):
        for s in range(a.seeds):
            r = next(r for r in rows if r["width"] == h and r["seed"] == s)
            ax.plot(curves[(h, s)], color=line if r["ok"] else bad, lw=1.4, alpha=0.75)
        ax.axvline(BURN_IN, color=muted, lw=0.8, ls=(0, (3, 3)))
        ax.set_yscale("log")
        n_ok = sum(r["ok"] for r in rows if r["width"] == h)
        ax.set_title(f"Width {h}: {n_ok} of {a.seeds} seeds pass", loc="left",
                     fontsize=11, color=ink)
        ax.set_xlabel("Adam step", color=muted)
        ax.grid(True, which="major", color=grid, lw=0.6)
        ax.tick_params(colors=muted, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
    axes[0][0].set_ylabel("Training loss (log scale)", color=muted)
    axes[0][0].annotate(f"rule applies after step {BURN_IN}", xy=(BURN_IN, 1), xycoords=("data", "axes fraction"),
                        xytext=(6, -12), textcoords="offset points", fontsize=8, color=muted)
    fig.suptitle(f"Training loss per seed, Adam lr = {a.lr}, weight decay = {a.wd}"
                 f" (blue = passes the rule, red = fails)", x=0.01, ha="left", fontsize=10, color=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png = f"figures/lr_check{a.tag}.png"
    fig.savefig(out_png, dpi=160)

    print("\nRULE:", RULE, "\n")
    print(f"{'width':>5} {'seed':>4} {'min loss':>10} {'final loss':>10} {'rebound':>8} {'at step':>7}  ok")
    for r in rows:
        print(f"{r['width']:>5} {r['seed']:>4} {r['min_loss']:>10.5f} {r['final_loss']:>10.5f} "
              f"{r['rebound']:>8.1%} {r['rebound_step']:>7}  {'yes' if r['ok'] else 'NO'}")
    n_bad = sum(not r["ok"] for r in rows)
    print(f"\nwrote {out_csv} and {out_png}")
    print("RESULT:", "PASS, keep this learning rate" if n_bad == 0
          else f"REVIEW, {n_bad} of {len(rows)} runs fail the rule; look at the figure before deciding")


if __name__ == "__main__":
    main()
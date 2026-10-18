"""v2 Section 2: write the accepted polygons and their figure BEFORE any training. Written by Claude, 19 Oct 2026.

python src/polygons_figure.py   ->  results_v2/polygons.json, figures/v2_polygons.png
"""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from polygon import make_polygon, make_polygon_data, area, interior_angles, edge_lengths

out = pathlib.Path("results_v2"); out.mkdir(exist_ok=True)
pathlib.Path("figures").mkdir(exist_ok=True)
info, fig = {}, plt.figure(figsize=(12, 3.2))
for j, k in enumerate([3, 4, 5, 6]):
    V, seed = make_polygon(k)
    X, y = make_polygon_data(k, 2000, 100 * k, V)
    info[f"T{k}"] = dict(seed_accepted=seed, vertices=V.round(6).tolist(), area=round(area(V), 6),
                         interior_angles_deg=interior_angles(V).round(2).tolist(),
                         edge_lengths=edge_lengths(V).round(4).tolist(), train_positive_rate=float(y.mean()))
    ax = fig.add_subplot(1, 4, j + 1)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=2, cmap="coolwarm")
    P = np.vstack([V, V[:1]]); ax.plot(P[:, 0], P[:, 1], "k-", lw=1)
    ax.set_title(f"T{k}: positives {y.mean():.1%}"); ax.set_aspect("equal"); ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
fig.tight_layout(); fig.savefig("figures/v2_polygons.png", dpi=150)
json.dump(info, open(out / "polygons.json", "w"), indent=1)
for t, v in info.items():
    print(t, "seed", v["seed_accepted"], "positives", f"{v['train_positive_rate']:.2%}", "angles", v["interior_angles_deg"])
print("wrote results_v2/polygons.json and figures/v2_polygons.png")

"""Generate the v2 LaTeX tables directly from results_v2/ (written by Claude, 19 Oct 2026)."""
import json, sys, pathlib
import numpy as np, pandas as pd
R = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2])
fr = json.load(open(R / "stage1_frozen.json"))["tasks"]
names = {"T0": "Coffee (triangle)", "T3": "Triangle", "T4": "Quadrilateral", "T5": "Pentagon", "T6": "Hexagon"}
pts = lambda x: f"{100*x:.2f}"
def piv(P):
    U = P[~P.flagged]; return U.pivot(index="pair", columns="mode")
rows = []
for t in ["T0", "T3", "T4", "T5", "T6"]:
    P = pd.read_csv(R / t / "pairs.csv"); W = piv(P)
    es, ep, ws = W.err_barrier.exact_sign, W.err_barrier.exact_perm, W.err_barrier.wm_sign
    fl = W.n_flips.exact_sign >= 1; h1 = P[~P.flagged].drop_duplicates("pair").h1_pass
    h5 = min(np.mean(W[m].exact_perm[fl] > W[m].exact_sign[fl]) for m in ["err_barrier","test_err_barrier","loss_barrier","cal_loss_barrier"])
    rows.append(f"{names[t]} & {fr[t]['hstar']} & {len(W)} & {int(h1.sum())}/{len(h1)} & {pts(es.median())} / {pts(es.quantile(.75))} / {pts(es.max())} & {pts(ep[fl].median())} & {int(fl.sum())} & {100*np.mean(abs(ws-es)<=0.005):.0f}\\% & {100*h5:.0f}\\% \\\\")
(out/"table_v2_main.tex").write_text("\n".join(rows)+"\n")
# sensitivity
rows = []
for inc, suf in [(1, "_inc1"), (2, ""), (5, "_inc5")]:
    for t in ["T0", "T3", "T4", "T5", "T6"]:
        d = R / f"{t}{suf}"; P = pd.read_csv(d / "pairs.csv")
        hs = json.load(open(d / "hstar.json"))["hstar"] if (d/"hstar.json").exists() and suf else fr[t]["hstar"]
        W = piv(P); es = W.err_barrier.exact_sign; h1 = P[~P.flagged].drop_duplicates("pair").h1_pass
        ok = "yes" if es.quantile(.75) <= 0.005 else "\\textbf{no}"
        rows.append(f"{inc}\\% & {names[t]} & {hs} & {len(W)} & {int(h1.sum())}/{len(h1)} & {pts(es.quantile(.75))} & {ok} \\\\")
    rows.append("\\midrule")
(out/"table_v2_sens.tex").write_text("\n".join(rows[:-1])+"\n")
# same lines vs different lines (main + sensitivity)
rows = []
for inc, suf in [(2, ""), (1, "_inc1"), (5, "_inc5")]:
    S = pd.concat([pd.read_csv(R / f"{t}{suf}" / "pairs.csv") for t in ["T0","T3","T4","T5","T6"]])
    E = S[(S["mode"] == "exact_sign") & (~S.flagged)]
    a, b = E[E.h1_pass].err_barrier, E[~E.h1_pass].err_barrier
    rb = f"{len(b)} & {pts(b.median())} & {pts(b.min())}--{pts(b.max())}" if len(b) else "0 & -- & --"
    rows.append(f"{inc}\\% & {len(a)} & {pts(a.median())} & {pts(a.max())} & {int((a>0.005).sum())} & {rb} \\\\")
(out/"table_v2_lines.tex").write_text("\n".join(rows)+"\n")
# excluded
rows = []
for t in ["T0","T3","T4","T5","T6"]:
    E = pd.read_csv(R / t / "excluded.csv"); c = E.cls.value_counts()
    rows.append(f"{names[t]} & {len(E)} & {c.get('a_not_converged',0)} & {c.get('b_wrong_minimum',0)} & {c.get('c_degenerate',0)} & {c.get('other',0)} \\\\")
(out/"table_v2_excl.tex").write_text("\n".join(rows)+"\n")
print(open(out/"table_v2_main.tex").read()); print(open(out/"table_v2_lines.tex").read()); print(open(out/"table_v2_excl.tex").read())

"""Post-hoc (D12): sensitivity of the same-line split to the line tolerance.

The pre-registered "same lines" test (H1) passes a pair when every matched line
differs by at most 10 degrees in angle and 0.25 standardized units in offset.
The exploratory split of the exact_sign residual by this test (D9; paper Section 5.2,
Table 6) depends on that tolerance. This script repeats the split with the
tolerance halved (5 deg, 0.125) and doubled (20 deg, 0.5).

No network is retrained and no alignment is recomputed: the line matching does not
depend on the tolerance, and pairs.csv already stores, for each pair, the largest
angle and offset difference over its matched lines (h1_max_angle, h1_max_offset).
Changing the tolerance only changes which pairs pass.

Usage:  python src/tolerance_check.py results_v2  [out.csv]
Written by Claude (26 Oct 2026); tested on made-up data in tests/test_tolerance_check.py.
"""
import sys
import pathlib
import pandas as pd

TASKS = ["T0", "T3", "T4", "T5", "T6"]
RUNS = [("2%", ""), ("1%", "_inc1"), ("5%", "_inc5")]
TOLERANCES = [("halved", 5.0, 0.125), ("pre-registered", 10.0, 0.25), ("doubled", 20.0, 0.5)]
EPS = 0.005  # 0.5 points, as a fraction


def load(root):
    """exact_sign rows of unflagged pairs, for every task and inclusion threshold."""
    frames = []
    for inc, suf in RUNS:
        for t in TASKS:
            f = pathlib.Path(root) / f"{t}{suf}" / "pairs.csv"
            if not f.exists():
                continue
            P = pd.read_csv(f)
            P = P[(P["mode"] == "exact_sign") & (~P["flagged"])].copy()
            P["inclusion"] = inc
            frames.append(P)
    if not frames:
        raise SystemExit(f"no pairs.csv files found under {root}")
    return pd.concat(frames, ignore_index=True)


def split(E, max_angle, max_offset):
    """Per-threshold rows and the distinct-pair totals, for one tolerance."""
    E = E.assign(same=(E["h1_max_angle"] <= max_angle) & (E["h1_max_offset"] <= max_offset))
    rows = []
    for inc, _ in RUNS:
        X = E[E["inclusion"] == inc]
        if X.empty:
            continue
        s, d = X[X.same].err_barrier, X[~X.same].err_barrier
        rows.append(dict(inclusion=inc,
                         same_pairs=len(s), same_below_eps=int((s <= EPS).sum()),
                         same_max_points=round(100 * s.max(), 2) if len(s) else None,
                         diff_pairs=len(d), diff_below_eps=int((d <= EPS).sum()),
                         diff_median_points=round(100 * d.median(), 2) if len(d) else None))
    # A pair is "distinct" by task, width and seeds; it can appear at several thresholds.
    key = ["task", "width", "seed_A", "seed_B"]
    D = E.drop_duplicates(key + ["same"])
    s, d = D[D.same], D[~D.same]
    total = dict(distinct_same=len(s), distinct_same_below_eps=int((s.err_barrier <= EPS).sum()),
                 distinct_diff=len(d), distinct_diff_below_eps=int((d.err_barrier <= EPS).sum()))
    return rows, total


def main(root, out=None):
    E = load(root)
    records = []
    for name, ang, off in TOLERANCES:
        rows, total = split(E, ang, off)
        print(f"\n== tolerance {name}: {ang:g} deg, {off:g} units ==")
        for r in rows:
            print(f"  {r['inclusion']}: same lines {r['same_below_eps']}/{r['same_pairs']} below eps "
                  f"(max {r['same_max_points']} pts); different lines {r['diff_below_eps']}/{r['diff_pairs']} below eps")
            records.append(dict(tolerance=name, max_angle=ang, max_offset=off, **r))
        print(f"  distinct over thresholds: same lines {total['distinct_same_below_eps']}/{total['distinct_same']} "
              f"below eps; different lines {total['distinct_diff_below_eps']}/{total['distinct_diff']} below eps")
        records.append(dict(tolerance=name, max_angle=ang, max_offset=off, inclusion="distinct", **total))
    if out:
        pd.DataFrame(records).to_csv(out, index=False)
        print(f"\nwrote {out}")
    return records


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_v2",
         sys.argv[2] if len(sys.argv) > 2 else "results_v2/tolerance_check.csv")

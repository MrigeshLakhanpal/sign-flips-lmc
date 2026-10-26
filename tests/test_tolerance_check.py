"""Made-up data test for src/tolerance_check.py (no real results used)."""
import pathlib
import sys
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import tolerance_check as tc


def _write(root, task, suf, rows):
    d = root / f"{task}{suf}"
    d.mkdir(parents=True)
    cols = ["task", "width", "pair", "seed_A", "seed_B", "mode", "flagged", "h1_max_angle", "h1_max_offset", "err_barrier"]
    pd.DataFrame(rows, columns=cols).to_csv(d / "pairs.csv", index=False)


def test_split_counts(tmp_path):
    # pair 0: angle 3, offset 0.1 -> same at every tolerance; residual 0.001
    # pair 1: angle 8, offset 0.1 -> same at 10 and 20, different at 5; residual 0.002
    # pair 2: angle 15, offset 0.4 -> same only at 20; residual 0.03
    # pair 3: flagged -> ignored; a wm row -> ignored
    rows = [
        ["T0", 3, 0, 1, 2, "exact_sign", False, 3, 0.1, 0.001],
        ["T0", 3, 1, 3, 4, "exact_sign", False, 8, 0.1, 0.002],
        ["T0", 3, 2, 5, 6, "exact_sign", False, 15, 0.4, 0.03],
        ["T0", 3, 3, 7, 8, "exact_sign", True, 1, 0.01, 0.5],
        ["T0", 3, 0, 1, 2, "wm", False, 3, 0.1, 0.2],
    ]
    _write(tmp_path, "T0", "", rows)
    _write(tmp_path, "T0", "_inc1", rows[:1])  # same pair 0 again at another threshold
    E = tc.load(tmp_path)
    assert len(E) == 4  # 3 unflagged exact_sign rows at 2% + 1 at 1%

    rows10, tot10 = tc.split(E, 10, 0.25)
    r2 = [r for r in rows10 if r["inclusion"] == "2%"][0]
    assert (r2["same_pairs"], r2["same_below_eps"], r2["diff_pairs"], r2["diff_below_eps"]) == (2, 2, 1, 0)
    assert tot10 == dict(distinct_same=2, distinct_same_below_eps=2, distinct_diff=1, distinct_diff_below_eps=0)

    _, tot5 = tc.split(E, 5, 0.125)
    assert tot5 == dict(distinct_same=1, distinct_same_below_eps=1, distinct_diff=2, distinct_diff_below_eps=1)

    _, tot20 = tc.split(E, 20, 0.5)
    assert tot20 == dict(distinct_same=3, distinct_same_below_eps=2, distinct_diff=0, distinct_diff_below_eps=0)

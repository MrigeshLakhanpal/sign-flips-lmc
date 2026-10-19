#!/usr/bin/env bash
# Study 2, Stage 2 (run AFTER the Stage 1 freeze is committed), from the project root.
set -euo pipefail
PY="${PYTHON:-$(command -v python3 || command -v python)}"
for t in T0 T3 T4 T5 T6; do "$PY" src/analyze_v2.py --task $t --all-runs; done
"$PY" src/hypotheses_v2.py
for t in T0 T3 T4 T5 T6; do "$PY" src/analyze_v2.py --task $t --max-train-err 0.01; "$PY" src/analyze_v2.py --task $t --max-train-err 0.05; done
"$PY" src/hypotheses_v2.py --inc 1
"$PY" src/hypotheses_v2.py --inc 5
"$PY" src/h7_explore.py
"$PY" src/dense_grid.py > results/dense_grid_2pct.txt
"$PY" src/dense_grid.py --max-train-err 0.05 > results/dense_grid_5pct.txt
"$PY" src/make_v2_tables.py results_v2 paper

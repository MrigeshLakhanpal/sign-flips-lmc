#!/usr/bin/env bash
# Full reproduction of every result except the multi-hour complete-group search (B1, bottom).
# CPU only. Run from the project root in a FRESH COPY of the project: it overwrites results/ and figures/.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-$(command -v python3 || command -v python)}"   # override with PYTHON=...
python() { "$PY" "$@"; }

# 0. Tests (made-up data only)
for t in test_core test_my_core test_lr_check test_extras test_h1b test_b1 test_b2 test_b4 test_report; do
  python tests/$t.py
done

# 1. Pre-run learning-rate check (rule dated 2026-10-07)
python src/lr_check.py

# 2. Main run (settings dated 2026-10-08): train, analyze at 2%, answer Q1-Q4, H1b, figures, intervals
python src/train.py --widths 2 3 4 6 8 16 32 64 --seeds 40 --steps 3000
python src/analyze.py --max-train-err 0.02
python src/h1b.py
python src/answers.py
python src/plot.py --max-train-err 0.02
python src/report.py

# 3. Inclusion sensitivity (1% and 5%)
python src/analyze.py --max-train-err 0.01 --out-dir results_inc1
python src/analyze.py --max-train-err 0.05 --out-dir results_inc5
python src/answers.py --results results_inc1
python src/answers.py --results results_inc5

# 4. B2 figure (one trained width-3 network)
python src/b2_figure.py

# 5. B4: weight norms and the weight-decay side-check (L2 0.003 inside Adam, width 64)
python src/train.py --widths 64 --seeds 40 --steps 3000 --wd 0.003 --out results/models_wd.pkl
python -c "import pickle; D=pickle.load(open('results/models_wd.pkl','rb')); e=[r['train_err'] for r in D['runs']]; print('wd run: included at 2%:', sum(x<=0.02 for x in e), 'of', len(e))"
python src/analyze.py --models results/models_wd.pkl --max-train-err 1.0 --out-dir results_wd_all   # exploratory, post-hoc
python src/b4.py --wd results_wd_all --models-wd results/models_wd.pkl --exploratory

echo "Done. Optional, several hours: python src/run_b1.py  (then width 3 at 1% and 5%, see src/run_b1.py)"

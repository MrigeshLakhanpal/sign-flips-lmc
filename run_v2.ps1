# v2 runs, Windows PowerShell. Run from the project root. Rules: preregistration/v2_rules.md (commit 349aa3a).
# PHASE A tonight (about 1-2 hours): tests, polygons, Stage 1 training, freeze.
# PHASE B after you commit the freeze (several hours; the width-6 exhaustive search is the long part).

# ---------- PHASE A ----------
python tests/test_v2.py
python src/polygons_figure.py                      # polygons BEFORE training (Section 2)
python src/train_v2.py --tasks T0 T3 T4 T5 T6      # Stage 1
python src/stage1.py                               # h*, Section 3.4 check, freeze file
# If stage1.py says "Section 3.4" for a task Tk:  python src/train_v2.py --tasks Tk --steps 10000 ; python src/stage1.py
# Then commit (results_v2/*.pkl stay out of git; their SHA-256 is in the freeze file):
#   git add src tests preregistration run_v2.ps1 run_v2.sh results_v2/polygons.json results_v2/stage1.csv results_v2/stage1_frozen.json figures/v2_polygons.png
#   git commit -m "v2 Stage 1 frozen"

# ---------- PHASE B ----------
# python src/analyze_v2.py --task T0 --all-runs
# python src/analyze_v2.py --task T3 --all-runs
# python src/analyze_v2.py --task T4 --all-runs
# python src/analyze_v2.py --task T5 --all-runs
# python src/analyze_v2.py --task T6 --all-runs
# python src/hypotheses_v2.py
# Pre-registered inclusion sensitivity (D7): repeat at 1% and 5%
# foreach ($t in "T0","T3","T4","T5","T6") { python src/analyze_v2.py --task $t --max-train-err 0.01; python src/analyze_v2.py --task $t --max-train-err 0.05 }
# python src/hypotheses_v2.py --inc 1
# python src/hypotheses_v2.py --inc 5
# python src/dense_grid.py *> results\dense_grid_2pct.txt
# python src/dense_grid.py --max-train-err 0.05 *> results\dense_grid_5pct.txt

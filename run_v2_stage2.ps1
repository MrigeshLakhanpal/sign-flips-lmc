# Study 2, Stage 2 (run AFTER the Stage 1 freeze is committed). Windows PowerShell, from the project root.
foreach ($t in "T0","T3","T4","T5","T6") { python src/analyze_v2.py --task $t --all-runs }
python src/hypotheses_v2.py
# Pre-registered inclusion sensitivity (D7)
foreach ($t in "T0","T3","T4","T5","T6") { python src/analyze_v2.py --task $t --max-train-err 0.01; python src/analyze_v2.py --task $t --max-train-err 0.05 }
python src/hypotheses_v2.py --inc 1
python src/hypotheses_v2.py --inc 5
# Exploratory (D8, D9) and the Study 1 dense-grid check
python src/h7_explore.py
python src/dense_grid.py *> results\dense_grid_2pct.txt
python src/dense_grid.py --max-train-err 0.05 *> results\dense_grid_5pct.txt
# Tables for the paper
python src/make_v2_tables.py results_v2 paper

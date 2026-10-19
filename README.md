# A Missing Sign Symmetry Accounts for the Linear-Mode-Connectivity Barrier in Minimal-Width Sigmoid Networks

[![DOI](https://zenodo.org/badge/1415245694.svg)](https://doi.org/10.5281/zenodo.23314245)

**Mrigesh Lakhanpal** · ORCID [0009-0007-9539-9091](https://orcid.org/0009-0007-9539-9091) · Preprint v2, October 2026

📄 **Paper:** [`paper/Lakhanpal_2026_sign_flips_LMC.pdf`](paper/Lakhanpal_2026_sign_flips_LMC.pdf) · DOI: [10.5281/zenodo.23314245](https://doi.org/10.5281/zenodo.23314245)

## Summary

For irreducible one-hidden-layer sigmoid networks, the complete group of exact symmetries is small and known
(permutations × sign flips, 2^h·h! elements), so at minimal width it can be searched exhaustively.

- **v1** (coffee task, width 3, 11 pairs): permutation matching leaves a median training-error barrier of **17.95 points**; with sign flips, **0.00**; an exhaustive search over all 48 symmetries leaves at most **0.11**.
- **v2** (publicly pre-registered, commit `349aa3a`; sign-aware alignment itself is prior work: Pittorino et al. 2022, Boufalis et al. 2025; 5 tasks, convex polygons with 3–6 edges, 100–120 seeds each, measured minimal widths 3–4): all pre-registered tests of the main claim pass under four barrier measures; the full search leaves an upper-quartile residual of at most **0.13 points** (165 pairs), against 9.95–21.65 points for the best permutation alone.
- **Learning the same lines suffices** (exploratory): all 157 pairs that learned the same decision lines end at ≤ 0.21 points, while the 8 that learned clearly different lines keep 2.38–9.58. Pairs whose lines differ only slightly can still be nearly connected.
- **Negative results:** the weights do not predict the size of the *error* barrier (pre-registered H7, ρ = 0.01; the error barrier saturates at the trivial classifier). On *loss* barriers they do, in an exploratory re-test decided afterwards (ρ = 0.86; calibrated 0.65). The claim fails for one task at each sensitivity threshold (1% and 5%).
- ReLU networks have no sign flip; this answers only the sigmoid analogue of the thin-model question.

## Reproduce

```bash
pip install -r requirements.txt     # Python 3.12.10; numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, matplotlib 3.11.2
bash run.sh                         # Linux / macOS / Git Bash
powershell -ExecutionPolicy Bypass -File run.ps1     # Windows
python src/run_b1.py                # optional: exhaustive complete-group search (several hours)
```

`run.sh` / `run.ps1` run all unit tests, training, every analysis, the bootstrap intervals and the figures
(about 87 minutes on a laptop CPU). A run from a clean copy on the same machine reproduced `results/barriers.csv`
and `results/h1b.csv` byte for byte; on another platform, loss values may differ in the last digits while every
error barrier is unchanged. All randomness is seeded. `run.sh` uses `python3` (or `python`); set `PYTHON=` to override.

### v2

```powershell
powershell -ExecutionPolicy Bypass -File run_v2.ps1          # Stage 1: tests, polygons, training, freeze
# commit the freeze, then:
powershell -ExecutionPolicy Bypass -File run_v2_stage2.ps1   # Stage 2, sensitivity, exploratory re-test, tables
```

Rules: `preregistration/v2_rules.md` (committed before any v2 training) and `preregistration/DEVIATIONS.md` (D1–D9, dated).
Stage 1 is frozen by SHA-256 fingerprints in `results_v2/stage1_frozen.json`; the analysis refuses to run on changed model files.
Model files (`results_v2/*.pkl`) are not in git; they are rebuilt by `src/train_v2.py`.

Timing on a Windows laptop (CPU only): Study 1 `run.ps1` 87 min; Study 2 Stage 1 about 40 min; Stage 2 110 min. A re-run of Stage 2 from the frozen models rebuilt every file in `results_v2/` byte for byte. Full reproduction of both studies takes about 4 hours, plus several hours for the optional `src/run_b1.py`.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Training, alignment, barrier, analysis and figure code |
| `tests/` | Unit tests (run first by `run.sh` / `run.ps1`) |
| `results/` | Main run (2% inclusion): barriers, lines, inclusion, H1b, learning-rate check, report tables |
| `results_inc1/`, `results_inc5/` | Inclusion sensitivity (1% and 5%) |
| `results_b1/`, `results_b1_inc1/`, `results_b1_inc5/` | Complete-group search (B1) at 2%, 1% and 5% inclusion |
| `results_wd_all/` | Weight-decay side-check at width 64 (B4; exploratory, inclusion ignored) |
| `figures/` | All figures written by the scripts |
| `preregistration/` | v1 rules (from the dated notebook) and v2 rules (committed before the v2 runs), deviations log |
| `results_v2/` | v2: polygons, Stage 1 counts and freeze, per-task Stage 2 results (`T*/`, `T*_inc1/`, `T*_inc5/`), verdicts |
| `paper/` | LaTeX source, bibliography, PDF, and a copy of the four figures the paper uses |

Script names: **B1** = complete-group search (`b1.py`, `run_b1.py`), **B2** = spare-neuron figure (`b2_figure.py`),
**B4** = weight norms and the weight-decay side-check (`b4.py`). `my_core.py` is a hand-written re-implementation
of three core functions, kept as a check on `core.py`.

## Data and credit

The coffee-roasting task comes from the *Machine Learning Specialization* (Andrew Ng, DeepLearning.AI and Stanford
Online). The data generator in `src/core.py` is a re-implementation of the task's labeling rule written for this project; **no course
files are included in this repository.**

## Use of AI assistance

I used Claude (Anthropic) as an AI research and coding assistant. The paper's *Authorship and use of AI assistance*
section states exactly what it did and what I did.

## Citation

```bibtex
@misc{lakhanpal2026signflips,
  author = {Mrigesh Lakhanpal},
  title  = {A Missing Sign Symmetry Accounts for the Linear-Mode-Connectivity Barrier in Minimal-Width Sigmoid Networks},
  year   = {2026},
  note   = {Preprint},
  doi    = {10.5281/zenodo.23314245},
  url    = {https://github.com/MrigeshLakhanpal/sign-flips-lmc}
}
```

## License

Code: MIT (see `LICENSE`). Paper, figures and result files: CC BY 4.0 (see `LICENSE-CC-BY-4.0.md`).

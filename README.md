# Sign Flips, Not Matching Failure, Account for the Linear-Mode-Connectivity Barrier in Minimal-Width Sigmoid Networks

[![DOI](https://zenodo.org/badge/1415245694.svg)](https://doi.org/10.5281/zenodo.23314245)

**Mrigesh Lakhanpal** · ORCID [0009-0007-9539-9091](https://orcid.org/0009-0007-9539-9091) · Preprint, October 2026

📄 **Paper:** [`paper/Lakhanpal_2026_sign_flips_LMC.pdf`](paper/Lakhanpal_2026_sign_flips_LMC.pdf) · DOI: [10.5281/zenodo.23314245](https://doi.org/10.5281/zenodo.23314245)

## Summary

Why permutation alignment fails to linearly connect narrow networks remains, to my knowledge, an open question
(Ainsworth et al., 2023, Section 5.3; Benzing et al., 2022, Section 3.1). One-hidden-layer sigmoid networks make an
exhaustive answer possible: for irreducible networks, the complete group of exact symmetries is finite and known
(permutations × sign flips, 48 elements at width 3).

- At width 3, permutation matching leaves a median training-error barrier of **17.95 points**; adding sign flips removes it (**0.00**).
- An exhaustive search over all 48 exact symmetries leaves at most **0.11 points** in every pair.
- One spare neuron joins a network to all of its 2^r·r! symmetry copies by a path of exactly constant function (proved).

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
  title  = {Sign Flips, Not Matching Failure, Account for the Linear-Mode-Connectivity Barrier in Minimal-Width Sigmoid Networks},
  year   = {2026},
  note   = {Preprint},
  doi    = {10.5281/zenodo.23314245},
  url    = {https://github.com/MrigeshLakhanpal/sign-flips-lmc}
}
```

## License

Code: MIT (see `LICENSE`). Paper, figures and result files: CC BY 4.0 (see `LICENSE-CC-BY-4.0.md`).

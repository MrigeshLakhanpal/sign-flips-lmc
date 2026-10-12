# Permutations Are Not Enough: Sign Flips Explain the Linear-Mode-Connectivity Barrier in Minimal-Width Sigmoid Networks

**Mrigesh Lakhanpal** · ORCID [0009-0007-9539-9091](https://orcid.org/0009-0007-9539-9091) · Preprint, October 2026

📄 **Paper:** [`paper/Lakhanpal_2026_sign_flips_LMC.pdf`](paper/Lakhanpal_2026_sign_flips_LMC.pdf) · DOI: *[added after the first Zenodo release]*

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
(about 87 minutes on a laptop CPU). A run from a clean copy reproduced `results/barriers.csv` and
`results/h1b.csv` byte for byte. All randomness is seeded.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Training, alignment, barrier, analysis and figure code |
| `tests/` | Unit tests (run first by `run.sh` / `run.ps1`) |
| `results/` | Result tables (CSV) and report tables |
| `figures/` | Figures used in the paper |
| `paper/` | LaTeX source, bibliography and PDF |

## Data and credit

The coffee-roasting task comes from the *Machine Learning Specialization* (Andrew Ng, DeepLearning.AI and Stanford
Online). The data generator in `src/core.py` is my own re-implementation of the task's labeling rule; **no course
files are included in this repository.**

## Use of AI assistance

I used Claude (Anthropic) as an AI research and coding assistant. The paper's *Authorship and use of AI assistance*
section states exactly what it did and what I did.

## Citation

```bibtex
@misc{lakhanpal2026permutations,
  author = {Mrigesh Lakhanpal},
  title  = {Permutations Are Not Enough: Sign Flips Explain the Linear-Mode-Connectivity Barrier in Minimal-Width Sigmoid Networks},
  year   = {2026},
  note   = {Preprint. DOI to be added}
}
```

## License

Code: MIT (see `LICENSE`). Paper and figures: CC BY 4.0 (see `LICENSE-CC-BY-4.0.md`).

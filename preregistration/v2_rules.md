# Pre-registration for v2: Sign flips and LMC across minimal widths

**Author:** Mrigesh Lakhanpal (ORCID 0009-0007-9539-9091). Drafted with Claude (Anthropic); every rule was decided by the author.
**Timestamp:** the Git commit that first adds this file. No v2 model is trained before that commit.
**Status:** DRAFT 2, 19 Oct 2026 (tanh dropped; H7 and the threshold justifications added). Becomes binding at commit.
**Deviations:** any later change goes in `preregistration/DEVIATIONS.md`, with date, reason, and whether it was made before or after seeing any barrier.

---

## 0. Two stages, one firewall

- **Stage 1 (training).** Train every network and record its training and test error. Decisions in Stage 1 may use **training error only**.
- **Stage 2 (alignment and barriers).** Starts only after the Stage 1 configuration is frozen and logged. No barrier, alignment or line comparison is computed before that.

The only Stage 1 decision rule is in Section 3.4.

## 1. What stays exactly as in v1 (v1.0.7, rules dated 7–9 Oct)

| Item | Setting |
|---|---|
| Model | 2 inputs → h sigmoid hidden units → 1 sigmoid output, with biases |
| Initialization | Glorot-uniform weights, zero biases |
| Training | Full-batch Adam, lr 0.01, 3,000 steps, binary cross-entropy, no weight decay |
| Data size | 2,000 train / 2,000 test, standardized with the training mean and SD |
| Inclusion | training error ≤ 2% (sensitivity at 1% and 5%) |
| Pairing | disjoint consecutive pairs of included networks, in seed order (primary) |
| Barrier | Entezari Eq. 1 on 25 evenly spaced α; training-error barrier in points decides every verdict; ε = 0.5 points |
| "Removed" | upper quartile of the per-pair residual ≤ ε |
| Nearly reducible flag | smallest \|output weight\| < 5% of largest; or two units on the same line (H1 tolerance); or a unit's activation SD < 0.01 on the training data |
| Line comparison (H1) | v1 `match_lines`, v1 tolerance |

## 2. Tasks

**T0: coffee (v1 task).** The existing 40 width-3 networks are reused unchanged. **80 new seeds (40–119) are added at width 3** with the same code, to enlarge the pair count (audit F4).

**Tk: convex polygons, k = 3, 4, 5, 6.**
- Inputs: x ~ Uniform([−1, 1]²). Train seed 100k + 0, test seed 100k + 1.
- Label: 1 if x lies strictly inside polygon P_k, else 0. Labels are an exact rule, so the data are separable and the Bayes error is 0.
- Polygon construction, deterministic. RNG seed 7000 + k.
  - Angles θ_j = 2πj/k + φ_j, with φ_j ~ U(−0.2·2π/k, 0.2·2π/k).
  - Radii r_j ~ U(0.85, 1.15).
  - Centre (0.10, −0.05). Then scale all radii so that the polygon area is **0.84**, i.e. 21% of the square, matching coffee's 20.95% positives.
  - Accept only if the polygon is convex, lies inside [−0.95, 0.95]², and every interior angle is ≤ 165° and every edge ≥ 0.25. Otherwise redraw with seed + 100, repeating until accepted.
- The accepted vertices are written to `results_v2/polygons.json` and plotted **before** any training.

*Why irregular and off-centre:* a regular, centred polygon has rotational and reflection symmetries in the data, which could make extra solutions equally good and confuse the analysis.

## 3. Widths, seeds, and minimal width

**3.1 Widths.** For task Tk: h ∈ {k−2, k−1, k, k+1, k+2}, with h ≥ 2.
**3.2 Seeds.** 100 per (task, width). Init seed = 100000·k + 1000·h + s.
**3.3 Operational minimal width h\*(k).** The smallest h in the grid at which **at least 10 networks** (so ≥ 5 disjoint pairs) meet inclusion. The included fraction at every width is reported.
- h\*(k) is **measured, not assumed**. If h\*(k) ≠ k, that is reported as a finding.
- H1 is then evaluated against the h\* lines the networks learn, not against the polygon's edges.

**3.4 The only Stage 1 rule.** If no width in a task's grid reaches 10 included networks, retrain **that task only** with 10,000 steps (all widths, same seeds). This happens once at most. Decided on training error only; logged in DEVIATIONS.md. If it still fails, the task is reported as "no minimal width found in range" and gets no barrier analysis.

## 4. Alignment modes (Stage 2)

At h\*(k) for each task:

| Mode | Definition |
|---|---|
| `none` | no alignment |
| `wm` | v1 weight matching (Hungarian on unit vectors (w, b, s)) |
| `wm_sign` | v1 sign-aware matching (Hungarian on \|S\|; Boufalis et al. Prop. 1) |
| `exact_perm` | all h! permutations, choosing the one with the lowest 25-point training-error barrier (v1 B1 rule) |
| `exact_sign` | all 2^h·h! permutations and flips, chosen the same way |

- Exhaustive search is run for h\* ≤ 6 (at most 46,080 elements).
- If h\* = 7 or more, only `none`, `wm` and `wm_sign` are run, and the claim for that task is limited to the matcher.
- Widths above h\*: `none`, `wm` and `wm_sign` are reported descriptively. **No claim** is made there (audit F7).
- **Irreducibility certificate.** Every network in an analyzed pair is checked with the Section 1 flags. Pairs with a flagged network are reported separately, and H4 is evaluated on unflagged pairs.

## 5. Measures (for every pair and every mode)

1. **Training-error barrier.** Primary; decides every verdict.
2. Test-error barrier.
3. Raw loss barrier (BCE, training set).
4. **Calibrated loss barrier.** At each α, the best of **81 log-spaced output scales from 0.01 to 16** (v1 used 41 scales from 0.25 to 8, which was too narrow). The number of grid-edge hits is reported.
5. **Dense grid.** The alignment chosen in Section 4 is re-evaluated on nested 97-point and 385-point α grids (`dense_grid.py`). This is reported as a robustness check; the 25-point barrier stays primary.
6. **Distance after alignment** (audit F14). Relative parameter distance ‖θ_A − Tθ_B‖ / ‖θ_A‖ after `exact_sign`, reported against the remaining barrier.
7. **Flip count.** The number of units flipped by `exact_sign`.

## 6. Hypotheses (confirmatory)

For each task Tk with a measured h\*(k) ≤ 6, on unflagged pairs:

- **H1-k: same lines.** All pairs learn the same h\* lines within the v1 tolerance.
- **H4a: exact symmetries remove the barrier.** `exact_sign` residual: upper quartile ≤ ε.
- **H4b: flips are needed.** Among pairs where `exact_sign` flips ≥ 1 unit, the median `exact_perm` barrier is > 2 points.
- **H4c: the simple matcher suffices.** `wm_sign` is within ε of `exact_sign` in ≥ 80% of pairs.
- **H5: the ordering is metric-robust.** Among pairs with ≥ 1 flip, barrier(`exact_perm`) > barrier(`exact_sign`) in ≥ 90% of pairs under each of measures 1–4.

For T0 (coffee, 120 seeds at width 3), H4a–H5 are re-tested on the enlarged pair set. The original 11 pairs stay as reported in v1.

**H7: the mechanism predicts the size of the barrier.** For each pair with ≥ 1 flip under `exact_sign`:
- Let F be the set of flipped units.
- Let M = Σ_{i∈F} |s_i^A| / 2, the midpoint-erasure bound on the logit shift.
- Let B_F be the training-error barrier when the `exact_sign` permutation is applied **without** its flips.

Pooled over all tasks, Spearman ρ(M, B_F) > 0, with a 95% bootstrap interval (10,000 resamples of pairs) that excludes 0. If the interval includes 0, it is reported as "weak".

*Why this matters:* H4 shows that flips remove the barrier. H7 tests whether our midpoint-erasure explanation predicts **how large** the barrier is from the weights alone, before any interpolation.

**Dropped from v2: tanh.** Mrigesh decided on 19 Oct to leave it out until he has studied it. The limitations section says so; sigmoid and tanh are related by an affine change of parameters.

### 6a. Why each threshold (fixed before any polygon data exist)

| Threshold | Value | Justification |
|---|---|---|
| ε | 0.5 points | v1, dated 8 Oct: 10 of 2,000 examples |
| "Removed" | upper quartile ≤ ε | same rule as v1 Q2 and B1, so v1 and v2 compare directly |
| H4b | > 2 points (4ε) | v1's best permutation-only barrier had a median of 17.88; 2 points is deliberately low, so passing means flips matter by a clear margin, not barely |
| H4c | ≥ 80% of pairs | v1, sign-aware matching against brute force within ε: 11/11 pairs at width 3, 19/20 at width 4, 11/20 at width 6. At minimal width agreement should be high; 80% allows a few misses by an approximate matcher |
| H5 | ≥ 90% of pairs | v1 width 3: 10 of 10 flip pairs under all four measures. 90% allows one miss in ten |
| h\* | ≥ 10 included networks | gives ≥ 5 disjoint pairs, the fewest for which an upper quartile means anything |
| Polygon area | 21% | matches coffee's positive rate (20.95%) |
| Interior angle ≤ 165°, edge ≥ 0.25 | — | a nearly flat corner or a tiny edge makes that edge almost redundant, so a network could drop it and "minimal width" would become ambiguous |

## 7. Excluded runs (descriptive, not a hypothesis)

At h\*(k) for every task, including T0, each network failing inclusion is classified in this order. The first matching class wins.

- **(c) Degenerate:** any nearly reducible flag.
- **(a) Not converged:** training error at step 3,000 differs from training error at step 2,700. Training logs error at steps 2,700 and 3,000.
- **(b) Wrong local minimum:** at least one ground-truth polygon edge (the coffee triangle has 3) has no unit line within the H1 tolerance.
- **Other:** none of the above.

Two further reports, both labelled exploratory:
- The headline numbers at h\* recomputed on all runs, ignoring inclusion.
- Barriers between excluded and included runs (reported, not interpreted).

Paper wording, fixed in advance: the claim covers "networks that reach the global solution"; the classes of the rest are reported.

## 8. Title rule (binding)

| Outcome | Title |
|---|---|
| H4a holds for every task with an analyzable h\* (T0 and at least three of T3–T6) | *A Missing Sign Symmetry Accounts for the Linear-Mode-Connectivity Barrier in Minimal-Width Sigmoid Networks* |
| H4a holds for some tasks only | "… in Small Sigmoid Networks", with the abstract naming the tasks and widths |
| H4a holds only for T0 | "… in Width-3 Sigmoid Networks" |

The task scope (2-D, convex polygons, separable labels) goes in the first sentence of the abstract.

## 9. What would count against the account

1. H4a failing at some k on unflagged pairs: some barrier remains that no exact symmetry removes.
2. H5 failing: the effect depends on the metric.
3. h\*(k) < k with H1 failing: the clean "one unit per edge" picture breaks.
4. Most excluded runs falling in class (b): there are several basins, so "single basin up to symmetry" must be limited to the global solution.

All four are reported whichever way they come out.

## 10. Analysis order

1. Commit this file.
2. Write `polygons.json` and its figure.
3. Stage 1 training (all tasks and the T0 extension). Log the training errors. Apply Section 3.4 if needed. Freeze.
4. Stage 2: the modes in Section 4, the measures in Section 5, the classification in Section 7.
5. Evaluate H1-k, H4a–c, H5 and H7, and apply the Section 8 title rule.
6. Anything else is labelled exploratory, with a date.

## 11. Software

The v1 code stays unchanged for v1 results. The new modules are tested on made-up data before any real run:
- `polygon.py`: the task generator;
- `exact_search.py`: the general-h exhaustive search, vectorized;
- `measures.py`: the widened calibration and the distance measure;
- `excluded.py`: the classification in Section 7;
- `mechanism.py`: M and B_F for H7.

One command rebuilds everything except the exhaustive search.

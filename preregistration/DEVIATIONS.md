# Deviations and clarifications of v2_rules.md (commit 349aa3a)

Each entry gives its date and whether it was made before or after seeing any v2 result.

**D1 (19 Oct 2026, before any v2 training). Clarification: data seeds.** Section 2 writes "Train seed 100k + 0, test seed 100k + 1". This means 100·k, so k = 3 uses seeds 300 and 301. (The init seed, written explicitly as 100000·k + 1000·h + s, is unchanged.)

**D2 (19 Oct 2026, before any v2 training). T0 seeds 0–39 are retrained.** Section 7 needs each network's training error at step 2,700, which v1 did not record. Training is full-batch and seeded, so retraining reproduces v1 exactly. `train_v2.py` compares the retrained parameters with v1's `results/models.pkl` and prints whether they are identical. If they differ, this file records it, and v1's networks are used for the first 40 seeds.

**D3 (19 Oct 2026, before any v2 training). Checkpoint after a 10,000-step retrain.** If Section 3.4 is triggered, class (a) compares the training error at step 9,700 with step 10,000, i.e. 300 steps before the end, the same gap as 2,700 vs 3,000.

**D4 (19 Oct 2026, before any v2 training). Exploratory modes in Section 7.** Barriers between excluded and included runs use `none` and `wm_sign`; each excluded run is paired with the first included run in seed order. The all-runs headline uses `none`, `wm`, `wm_sign` and `exact_sign`. Both are labelled exploratory.

**D5 (19 Oct 2026, before any v2 training). Dense grid.** Section 5 item 5 is applied to every mode at h\* (it is cheap), not only to the exact modes.

**D6 (19 Oct 2026, before any v2 training). Faster exact search, same rule.** `exact_search.py` computes each candidate's logit as a sum of per-unit terms. It is tested to give the same curves as the direct computation, and to choose the same alignment as v1's `b1.brute_force_exact` at widths 3 and 4.

**D7 (19 Oct 2026, after the Stage 1 freeze, before any barrier was computed). How the inclusion-sensitivity runs pick h\*.** Section 1 pre-registers sensitivity at 1% and 5% inclusion but does not say how h\* is set in those runs. Rule: h\* is re-derived with the same Section 3.3 rule (smallest width with ≥ 10 included networks) at that threshold, from training error only. Results go to `results_v2/<task>_inc1/` and `<task>_inc5/`, and the main analysis stays at 2%.
*Disclosure:* this was written after seeing the Stage 1 inclusion counts (`stage1.csv`, training error only, which the Section 0 firewall allows). Those counts show h\*(T5) = h\*(T6) = 4 at 2%, but h\* = k for every polygon at 1%. The main verdicts remain the 2% ones.

**D8 (19 Oct 2026, after seeing all Stage 2 results). EXPLORATORY H7 on measures without the error ceiling.** The pre-registered H7 (error barrier) failed: ρ = 0.007, CI [−0.18, 0.19]. The flip-free error barrier sits near each task's positive rate, i.e. it saturates at the all-negative classifier. `src/h7_explore.py` repeats the H7 correlation with the raw and calibrated *loss* barriers. This analysis was decided after seeing the results. It is reported as exploratory, and the pre-registered verdict stays "not supported".

**D9 (19 Oct 2026, after seeing all Stage 2 results). EXPLORATORY: same lines versus remaining barrier.** Split by H1 pass/fail, the exact_sign residual is ≤ 0.21 points in all 157 same-line pairs and 2.4–9.6 points in the 8 different-line pairs (main analysis). The same split appears in the 1% and 5% runs. Reported as exploratory.

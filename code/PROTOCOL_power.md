# Additional benchmarks for CEO vs the leading swarm algorithms (locked 2026-10-20, before any run)

Why: on CEC2022 D10 and CEC2020-RW, CEO ranks 1st and 2nd-3rd among 20 swarm algorithms, but the
differences to MPA, CS and ABC are not significant (12 or 7 problems are too few). This study adds
benchmark problems to test whether a real difference exists. Outcome reported whatever it is; no
algorithm, parameter or seed is changed after seeing results.

## Algorithms (frozen, no change)
CEO = CEO-v4 with tuned_v4.json; MPA, CS, ABC = swarm_orig.py as audited (SWARM_AUDIT.md),
population 30, original parameters. The three opponents were fixed by the CEC2022 D10 ranking
(positions 2-4) before this protocol.

## Benchmarks
| Suite | Dims | Budget | Functions | Runs | Seeds |
|---|---|---|---|---|---|
| CEC2017, official C code (cec2017_official/, unmodified cec17_test_func.cpp) | 10, 30 | 10,000 x D | F1, F3-F30 (29) | 30 | 1,000,000 + 10,000 D + 100 fn + run |
| CEC2022, official code (existing harness) | 20 | 1,000,000 | F1-F12 | 30 | official rule; CEO D20 runs already exist and are reused |

Box [-100, 100]^D, error = f - F*, errors below 1e-8 set to 0 (CEC rule).
Evaluator audit: f(o) = F* on 28/29 functions; F9 (Levy) has its minimum away from o by construction
in the official code (identical in an independent port). The tilleyd `cec2017` package was rejected
(F1 differs by x10; F10/F12/F16/F17/F20 miss F* at o).

## Analysis (fixed)
Per suite-dimension: Friedman over the 4 algorithms, Holm post-hoc with CEO as control; rank-sum per
function with Holm over the functions (W/T/L).
Primary test across everything: Wilcoxon signed-rank over all function-dimension cases with mean
error (CEC2022 D10 12 + CEC2022 D20 12 + CEC2017 D10 29 + CEC2017 D30 29 = 82 cases), CEO vs each of
MPA, CS, ABC, Holm over the three comparisons, alpha = 0.05. Also reported: sign-test counts.

## Deviations
- 2026-10-21, added after the locked analysis above was complete (results in results/analysis/power_summary.txt,
  unchanged): CEC2017 at D=50, the next standard CEC2017 dimension, requested by the author to test whether the
  advantage seen at D=30 persists. Same four algorithms, frozen configurations, official code, budget 10,000 x D =
  500,000 FEs, 30 runs, seeds 1,000,000 + 10,000 D + 100 fn + run. Analysis = the same per-suite procedure
  (Friedman + Holm post-hoc with CEO as control, rank-sum W/T/L with Holm over 29 functions) plus a signed-rank test
  over the 29 D50 functions, Holm over the three opponents. Reported as a separate supplementary test; the 82-case
  primary result is not re-pooled with D50. Run with run_power.py --grid d50 (default grid unchanged).
- 2026-10-21, after the D50 analysis was complete (results/analysis/power50_summary.txt, unchanged): the D50 runs
  stored only the final error, so a convergence figure could not be drawn. The same D50 jobs are rerun with the same
  frozen configurations and seeds, now also storing the error at the 16 CEC recording points
  (run_power.py --grid d50trace, output power50trace-test_s*.csv). Recording the trace does not change the search.
  Use: convergence figures only. The final errors of the rerun are checked against the original D50 runs; all
  statistics stay on the original runs.
- 2026-10-22, reviewer round 1 (SWEVO_cas_dc/review/round1_2026-10-22), written before any of these runs. Grid `rev1`
  of run_power.py, output rev1-test_s*.csv, same harness, official CEC2017 code, budget 10,000 x D, 30 runs, seeds
  1,000,000 + 10,000 D + 100 fn + run (so every new run is paired with the existing CEO run of the same seed):
  (a) L-SHADE and jSO, the implementations already used on CEC2022 (baselines.py, unchanged; their published
      parameters), on CEC2017 at D = 10, 30 and 50.
  (b) Emergence ablation at D = 30 with the frozen parameters: v4-noEmergence, v4-noGate, v4-noMacro, v4-sharedMem,
      and replacement controls v4-K1 (one brood, emergence kept), v4-equalPeriods (all broods share period P0),
      v4-gaussMoult (moults drawn from N(0,1) clipped to [-3,3] instead of Levy steps), v4-restart (one brood,
      theta = 0, so every failed gate restarts all but the best: a generic stagnation restart on the same core).
      The two new switches of ceo_v4.py default to the frozen behaviour; bitwise identity of the default and of the
      existing variants with the pre-change code was checked on 15 jobs before launch.
  (c) CEO-N0fixed: CEO with the initial population fixed at N0 = 110, its D = 10 size, at D = 30 and 50.
  Analysis fixed now: (a) Wilcoxon signed-rank on mean errors over cases plus sign test, Holm over the two opponents,
  for the 58 CEC2017 D10+D30 cases (the held-out design) and separately for each dimension; per-function rank-sum
  W/T/L with Holm over functions; Vargha-Delaney A12. (b) each variant against CEO: per-function rank-sum W/T/L with
  Holm over the 29 functions and a signed-rank test over the 29 mean errors, Holm over the eight variants.
  (c) CEO-N0fixed against MPA, CS and ABC with the D = 50 procedure, and against CEO. These analyses are reported
  whatever they show; no parameter of CEO is changed.
- 2026-10-25, advisor review (Phayung_Review/Comments_to_Authors_Overleaf.pdf), analysis only, no run changed:
  (M3) the primary test recomputed without the CEC2022 D10 block that selected the opponents (70 cases) and at the
  level of the 41 function families (one vote each, mean sign over dimensions); D50 relabelled as exploratory
  extension everywhere. (M9) RC15-RC21 table of the 16 published methods now uses the audited swarm-study
  implementations of ABC, ACOR, GWO, HHO, PSO and WOA (same seeds 0-29) instead of the earlier R1 versions, so that
  each algorithm name refers to one implementation; CEO stays 5th of 17. analyze_rev3.py, results/analysis/rev3_*.

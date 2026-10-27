# Reviewer round, study 2: equal-effort tuning at D = 30, fresh test on BBOB (locked 2026-10-22, written before any run)

## Why
The rev1 runs (PROTOCOL_power.md, Deviations 2026-10-22) showed CEO tied with L-SHADE and jSO on CEC2017 at D = 10 and
significantly worse at D = 30 and 50. The no-emergence variant was worse still, which points at the core settings, tuned
at D = 10 on classical functions. This study asks whether CEO, with settings chosen at D = 30, beats its success-history
parents when they receive the same tuning effort, on a suite that has never been used for any version of CEO.

## Suites
- Tuning (development): CEC2017, 29 functions, official code, D = 30, 300,000 FEs, 3 seeds per function,
  seed = 3,000,000 + 100 fn + s, s = 1..3. CEC2017 is already a development suite for this project.
- Test (unseen by every version of CEO): BBOB, the 24 noiseless functions of the COCO platform (cocoex 2.8.2), D = 20 and
  D = 40, box [-5, 5]^D, budget 10,000 D FEs, 30 runs per function, run r uses instance r (r = 1..30) and
  seed = 4,000,000 + 10,000 D + 100 fn + r. Error = f - f_opt, with f_opt = f(x_opt) from the suite; errors below 1e-8 are 0.
  The error at the CEC recording points floor(D^(k/5-3) B), k = 0..15, is stored for anytime plots.

## Algorithms and tuning (same rule, same budget for all three)
24 configurations per algorithm: configuration 0 is the current one, 23 are drawn with a fixed seed from the space below.
Selection: lowest mean rank, over the 29 functions, of the median error over the 3 seeds; ties go to the lower index.
The selection is computed by select_rev2.py and frozen in tuned_rev2.json before any test run.
- CEO (ceo_v4.py, unchanged): N0f [8, 25] integer, K [2, 8] integer, E [10, 200] log, theta [1, 5] integer,
  c_micro [0.2, 3] log, p [0.05, 0.25]. Configuration 0 = tuned_v4.json.
- L-SHADE (baselines.py): r_Ninit [5, 25] (N_init = round(r_Ninit D)), H [3, 10] integer, p [0.05, 0.25],
  arc_rate [1, 3]. Configuration 0 = published 18, 6, 0.11, 2.6.
- jSO (baselines.py): c_Ninit [10, 40] (N_init = round(c_Ninit sqrt(D) ln D)), H [3, 10] integer, p_max [0.1, 0.4]
  (p_min = p_max / 2), arc_rate [0.5, 2.5]. Configuration 0 = published 25, 5, 0.25, 1.0.
The parameter arguments added to baselines.py default to the published values; bitwise identity with the previous code
is checked before launch.

Test algorithms: CEO (frozen tuned_v4), CEO-T, L-SHADE, L-SHADE-T, jSO, jSO-T (T = configuration chosen here).

## Analysis (fixed now)
Primary: CEO-T against L-SHADE-T and against jSO-T over the 48 BBOB cases (24 functions x 2 dimensions): two-sided
Wilcoxon signed-rank on mean errors, Holm over the two opponents, alpha = 0.05; sign test and median Vargha-Delaney A12
reported with it. Also per dimension.
Secondary, descriptive: per-function rank-sum W/T/L with Holm over functions for every pair above; CEO-T against CEO
(effect of tuning at D = 30); L-SHADE-T and jSO-T against their published versions; CEO-T against the published
L-SHADE and jSO; anytime ECDF over the stored traces.
Everything is reported whatever it shows. No configuration is changed after the test runs start.

## Deviations
Dated entries only, appended below.
- 2026-10-22, before any test run: The batch environment runs Python 3.13, for which cocoex 2.8.2 has no wheel, so the test stage uses
  coco-experiment 2.8.3. On 192 BBOB problems (24 functions, D = 20 and 40, instances 1, 7, 15, 30) the two versions gave
  identical function values at five random points and identical f(x_opt).
- 2026-10-23, after the test runs, analysis only: the pre-registered median A12 over all 24 functions is 0.50 for most
  pairs because 10 functions at D = 20 and 5 at D = 40 are solved to the 1e-8 floor by all six methods in every run.
  The median over the remaining functions is reported next to it as a descriptive figure. Nothing else changed.
- 2026-10-25, advisor review, analysis only: per-function comparisons repeated as paired signed-rank tests over the
  30 instance-matched runs (run r = instance r for every method); ECDF recomputed with 51 fixed targets 1e2..1e-8 as
  a check on the outcome-derived targets. Reported next to the pre-registered analysis, which is unchanged.

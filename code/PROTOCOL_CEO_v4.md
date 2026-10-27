# CEO-v4: locked protocol (2026-10-16, written before any CEO-v4 run)

## Why a v4
Rounds 1 and R1 showed that the frozen CEO, here called CEO-v1, is not competitive. Against 16 opponents it was significantly better in 29.1% of
824 comparisons and worse in 44.7%. It lost every comparison with L-SHADE, LSHADE-cnEpSin and IMODE, and 58 of 71 with BIPOP-CMA-ES.

A code review on 2026-10-16 traced this to six causes in the v1 operators:
1. The nymph and emergence moves have no greedy selection, and the nymph step overshoots with eta up to 2.2.
2. The stagnation test is relative to |f|, so it depends on the bias of the objective.
3. The PDO handles stagnation by throwing away the brood's best point.
4. The emergence step scales with the box width, not with the brood's spread.
5. There are no difference vectors and no archive.
6. Clipping makes members stick to the bounds.

v4 is a redesign, not a tuned v1. Every result below is reported whatever it shows, and v4 is not changed after test results are seen.

## Algorithm: CEO-v4, implemented in `ceo_v4.py`
v4 keeps the brood and clock structure of CEO. That structure now decides *when* and *how* each brood re-diversifies. The search engine is a
greedy-selected DE core:

- **Broods and clocks.** The population is split round-robin into n_b broods. Brood k has an integer period P_k and a random phase phi_k, both counted in generations.
  The periods are spread around P0 by the factors {0.55, 0.75, 1.0, 1.35, 1.7, 2.1, 2.6, 3.2}, then made distinct. They are not prime, because the v1 ablation showed primality has no effect.
  P0 = max(5, round(G_est / E)), where G_est = MaxFE / ((N0 + Nmin) / 2) is the expected number of generations.
- **Nymph phase, every generation, for broods not emerging.** Mutation is current-to-pbest/1 with an archive:
  v_i = x_i + F_i (x_pbest − x_i) + F_i (x_r1 − x~_r2).
  x_pbest is drawn from the top p fraction of the *whole* population, x_r1 from the population, and x~_r2 from the population plus the archive.
  This is followed by midpoint bound repair, binomial crossover with CR_i, and greedy selection. Parents that lose go to the archive.
- **Brood-specific success-history memories.** F_i and CR_i are drawn from the H-slot memory *of the brood of i*, as in SHADE:
  F is Cauchy(M_F, 0.1), redrawn while ≤ 0 and capped at 1; CR is N(M_CR, 0.1), clipped to [0, 1].
  Each brood updates only its own memory, with the weighted Lehmer mean of its successful F and CR.
- **Progress-gated emergence.** When brood k's clock gate opens, (g − phi_k) mod P_k = 0, the brood's best is compared with its best at the previous gate.
  - If it improved, the brood is still progressing. It does *not* emerge, and its counter s_k is reset to 0.
  - Otherwise s_k increases by 1 and the brood emerges in one of two regimes:
    - **Micro emergence, while s_k ≤ theta.** Every member worse than the brood median is resampled around the brood best:
      x = x_bbest + c_micro · sigma_k ⊙ clip(Levy(1.5), −3, 3).
      Here sigma_k is the per-dimension standard deviation of the brood, with the population's spread used if the brood has fewer than 2 members.
      Bounds are repaired by the midpoint rule toward x_bbest. The resampled points replace those members unconditionally, and the old points go to the archive.
    - **Macro emergence, when s_k > theta.** All members are re-initialized uniformly in the box, except the global best if this brood holds it.
      The brood's memory is reset to 0.5 and its phase is redrawn, and s_k = 0.
  - An emerging brood does not run the nymph phase in that generation.
- **LPSR, by evaluations.** N(FE) = round(N0 − (N0 − Nmin) · FE / MaxFE), with N0 = round(N0f · D) and Nmin = max(4, n_b).
  The worst individuals are removed, but every brood keeps at least one member. The archive is capped at round(2.6 · N).
- **Fixed values.** H = 6, arc_rate = 2.6, Levy beta = 1.5.

Defaults, which are config 0 in tuning: N0f = 18, n_b = 4, E = 50, theta = 2, c_micro = 1.0, p = 0.11.

## Switches for the ablation
Each variant shares the tuned parameters:
- `v4-noGate` emerges on every clock gate, with no progress test.
- `v4-noMacro` uses micro emergence only.
- `v4-noEmergence` has no emergence at all, so it is a brood-memory L-SHADE.
- `v4-sharedMem` uses one memory for all broods.

## Harness: identical to the earlier baseline study, so its baseline runs are reused unchanged
- The test suite is the official CEC2022 Python reference code, through the adapter `cec2022_fast.py`. It was verified bitwise against the reference at D10 and D20 on 2026-10-16.
- Box [−100, 100]^D. D10 gets 200,000 FEs and D20 gets 1,000,000 FEs. Each function has 30 runs, with seeds from the official `Rand_Seeds.txt` rule (`cec_seed`).
- All methods share one budget wrapper, `budget.Objective`. Errors below 1e-8 are reported as 0.
- The baselines are reused from `runs/baselines_cec2022/`. They were run with this same harness, have 360 runs each per dimension, and are not rerun:
  DE, PSO and IPOP-CMA-ES, tuned under the same rule; L-SHADE and jSO; GWO and HHO. Two further optimizers of that
  study are not part of this paper, and their runs are not included.
- CEO-v1 is the frozen `ceo_final` from 22 September 2026, run through an adapter. It keeps its own population of 30 and its iteration horizon from `fe_budget.max_iter_for`, and it is not tuned.

## Tuning: the same rule and budget as the tuned baselines of the earlier study
- The tuning suite is `tuning_suite.py`: 12 shifted and rotated classical functions, disjoint from CEC2022. Runs are at D10 with 100,000 FEs.
  Seeds 910001–910003 are paired with instances 1–3.
- There are 24 configs. Config 0 is the default; the other 23 come from a seeded random draw over this space:
  N0f [8, 25] integer, n_b [2, 8] integer, E [10, 200] log, theta [1, 5] integer, c_micro [0.2, 3] log, p [0.05, 0.25].
- Selection: the lowest mean rank, over the 12 functions, of the median error over the 3 seeds, with ties broken toward the lower index.
  The result is frozen in `tuned_v4.json` before any test run. The ablation variants use the frozen v4 parameters and are not tuned.

## Test runs
- D10: v4, v4-noGate, v4-noMacro, v4-noEmergence, v4-sharedMem and CEO-v1, on 12 functions with 30 runs each.
- D20: v4 and CEO-v1, on 12 functions with 30 runs each. The ablations are not run at D20 because of cost, and this is disclosed.
- CEC2020-RW RC15–RC21: v4 only, through the R1 harness. That means `r1_lib.rw_problem`, the static penalty 1e6·Σmax(0, g), seeds 0–29, and budgets of 100,000 FEs, or 200,000 for RC16.
  The result is compared with the 17 R1 algorithms. The baseline-study and R1 harnesses are not mixed within one table.

## Analysis, fixed in advance
- To stay comparable with the analysis of the earlier baseline study, each pair is compared with the Wilcoxon rank-sum test per function and Holm correction over the 12 functions, at alpha 0.05.
  This is done for v4 against each baseline and against CEO-v1.
- A Friedman test on the mean errors of all algorithms, followed by a Holm post-hoc test with v4 as the control.
- Ablation questions, answered as v4 against each switch: does the progress gate, the macro regime, emergence at all, or brood-specific memory matter?
- For CEC2020-RW: the mean of the best penalized value, the feasibility count, the Holm W/T/L against the 17 R1 algorithms, and the average rank.
- No success threshold is set. v4 is reported as better, equal or worse than L-SHADE and jSO, whatever the outcome.
  If v4-noEmergence matches v4, the paper must state that the cicada-derived part adds nothing beyond a brood-memory L-SHADE.

## Compute
Test runs are split into seed shards; each shard writes its own CSV, see `runs/v4/`.

## Deviations
Deviations are dated and appended here only; nothing above is edited after the first run.

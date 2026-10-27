# PROTOCOL R1: post-review experiments for CEO_SWEVO (locked 2026-10-14, before any R1 run)

Origin: internal pre-submission review of 2026-10-14.
Name tag everywhere: **R1**. Code is in `experiments/r1/`. This repository holds the per-run results of
experiment E4, RC15 to RC21, in `runs/r1/out/`; E1 to E3 concern an earlier design and are not used in the paper.

## Fixed for every experiment
- CEO = `ceo_ablation.ceo_final`, frozen 2026-09-22. **No CEO parameter changes as a result of R1.**
- Population 30 for CEO and the eight core baselines. The SOTA methods keep their own population rules (see `sota_baselines.py`).
- Seeds 0–29 are shared, so runs are paired. The hard FE cap comes from `fe_budget.run_fe_matched`.
- Statistics are those of `analyze_fe.py`: errors rounded to 1e-8, Friedman on mean error, two-sided Wilcoxon, and Holm within each function across comparisons.
  For the ablation and sensitivity analyses we report uncorrected and Holm-over-functions counts side by side.
- Whatever comes out is reported. Configurations are not re-run or added after the results are seen.

## E1: ablation on CEC2022 D=10 (RM-8), reported as a post-hoc analysis
CEC2022 has already been used as a test suite, so E1 explains results; it is not a new held-out test.
- Configs: CEO full, re-run in the batch runs as a reproducibility anchor and compared with `results/fe/cec2022_d10/C*__CEO.csv`; the 7 `ABLATIONS` variants; `sync clocks, w/o PDO`. That makes 9 configs.
- 12 functions, 200,000 FEs, 30 runs.

## E2: parameter sensitivity on the classical design suite (RM-9), one at a time around the frozen config
- eta0 ∈ {1.1, 1.65, 2.75, 3.3}, L0 ∈ {0.1, 0.5, 1.0}, n_b ∈ {2, 3, 6, 8}, theta ∈ {2, 5, 8},
  E_target ∈ {10, 40}, gamma ∈ {2, 6}. That is 19 configs plus the frozen CEO re-run as the anchor.
- 23 classical functions, 15,000 FEs, 30 runs. Reported as the rank of each config and its better/tie/worse record against the frozen config.

## E3: restart / scheduled-reinitialization baselines (RM-10)
- **BIPOP-CMA-ES**: pycma 4.4.4 `cma.fmin2(..., restarts=9, bipop=True)`, x0 drawn uniformly per seed, sigma0 = 0.3·(ub−lb),
  box bounds through pycma's BoundTransform, defaults otherwise.
- **Saw-tooth GA**: our GA operators, with tournament selection, arithmetic crossover 0.8, mutation 0.1 and elitism 1.
  The population follows a saw-tooth with mean n̄=30, amplitude 27 and period 20 generations; at the start of each period it is
  refilled with uniform random individuals [Koumousis & Katsaras 2006]. These settings are ours and untuned. Unless the paper's values are verified first,
  the manuscript will not claim that they are the recommended ones.
- Suites: classical 23 functions at 15k FEs; CEC2017 D=30 at 300k; CEC2022 D=10 at 200k. 30 runs each.
  **CEC2022 D=20 is excluded** because of cost (~100 core-hours), and this is disclosed.
- Each restart baseline is added as an extra row/column next to the frozen results. The existing nine- and 15-algorithm rankings are not
  re-ranked silently; a separate ranking that includes them is reported.

## E4: real-world constrained problems (RM-11)
- CEC2020 real-world suite [Kumar et al. 2020, SWEVO 56:100693], using the enoppy 0.1.1 implementations. Mechanical subset:
  RC15 speed reducer D7, RC16 industrial refrigeration D14, RC17 tension/compression spring D3, RC18 pressure vessel D4,
  RC19 welded beam D4, RC20 three-bar truss D2, RC21 multiple disk clutch brake D5.
  Excluded: RC22 planetary gear, which is integer-valued; RC23 step-cone pulley, whose enoppy get_cons raises a shape error; and RC24 robot gripper,
  which runs an inner optimizer per evaluation. These exclusions are disclosed.
- Every algorithm uses the same static penalty: F = f + 1e6·Σ max(0, g_i). The problems have no equality constraints. enoppy's `amend_position` is applied
  where it is defined, as in its own `evaluate`.
- Budget is 100,000 FEs for D ≤ 10 and 200,000 for RC16. These are protocol choices; check them against Kumar 2020 before writing.
- Algorithms: CEO, the 8 core baselines, the 6 SOTA methods and the 2 restart baselines, 17 in total. 30 runs each.
- Reported per problem: the mean of the best penalized value, the feasibility rate of the best solution, and the mean objective over feasible runs.

## Not done in R1
CEC2022 D=20 for the E1 and E3 configs, and any CEO redesign. A new CEO variant would need its own protocol, as in PROTOCOL_CEO_v3.md.

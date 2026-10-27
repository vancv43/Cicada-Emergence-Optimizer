# CEO parameter sensitivity (locked 2026-10-20, before any run)

Purpose: describe how CEO's performance changes when one parameter moves away from its frozen value.
This is a descriptive study. The frozen parameters (tuned_v4.json) are NOT re-selected from it.

## Design
One-at-a-time variation around the frozen setting, five levels per parameter, frozen level included:

| Parameter | Levels (frozen in bold) |
|---|---|
| N0f, initial population per dimension | 8, **11**, 15, 20, 25 |
| n_b, number of broods | 2, 3, **5**, 6, 8 |
| E, expected emergences per brood over the run | 10, 40, **78.43**, 120, 200 |
| theta, failed emergences before a brood restart | 1, 2, 3, **4**, 5 |
| c_micro, micro-emergence step scale | 0.2, **0.273**, 0.5, 1.0, 3.0 |
| p, p-best fraction | 0.05, 0.10, 0.15, **0.198**, 0.25 |

Levels span the tuning search space of tune_v4.py. 25 configurations (24 variants + frozen).

Suite: official CEC2017 D = 10 (cec2017_official), F1 and F3-F30, budget 100,000, 30 runs,
seeds identical to PROTOCOL_power.md (so the frozen configuration reproduces the power-study CEO runs,
which serves as a cross-machine reproducibility check). CEC2022 and CEC2020-RW are not used, to keep the
sensitivity study off the main test sets.

## Analysis
Per parameter: Friedman test over its five levels on mean error across the 29 functions, mean ranks,
and the share of functions on which each level differs from the frozen level (rank-sum, Holm over
functions). Reported as a figure plus a table in the Supplementary.

## Deviations
(none yet)

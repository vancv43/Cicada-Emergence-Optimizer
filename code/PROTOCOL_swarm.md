# Swarm-intelligence comparison: protocol (locked 2026-10-17, before any test run)

Purpose (advisor request): rank the final CEO algorithm among classic swarm-intelligence algorithms on
two benchmark sets. CEO = the frozen CEO-v4 of PROTOCOL_CEO_v4.md; its results are not re-run or changed.
Version labels are experiment history only; the paper uses the name CEO.

## Algorithm set (19 swarm-intelligence algorithms + CEO = 20)

The advisor's list of 20, with Salp SSA and ChOA removed and AEO, ASO, AO proposed as replacements
(advisor, 2026-10-17). AEO (ecosystem energy flow) and ASO (atomic dynamics, physics) are not swarm
intelligence and are excluded on the advisor's second instruction. AO (Aquila hunting) is kept.

Code: `swarm_orig.py`. Rule: published equations first; authors' public code structure for details the
paper leaves open. Library versions were audited first and rejected where they deviate (SWARM_AUDIT.md).

| # | Algorithm | Original publication | Implementation |
|---|-----------|----------------------|----------------|
| 1 | PSO | Kennedy & Eberhart 1995; inertia Shi & Eberhart 1998 | shared-harness gbest PSO, tuned (earlier baseline study), CEC2022 results reused (same code, seeds) |
| 2 | ACO (ACOR) | Socha & Dorigo 2008, EJOR | own, Eqs. 7-9 |
| 3 | ABC | Karaboga & Basturk 2007, JOGO | own, Karaboga's code structure |
| 4 | GWO | Mirjalili et al. 2014, AES | EvoloPy GWO.py verbatim (authors' group port) |
| 5 | WOA | Mirjalili & Lewis 2016, AES | EvoloPy WOA.py verbatim |
| 6 | HHO | Heidari et al. 2019, FGCS | own, Eqs. 1-13; cross-checked against EvoloPy HHO |
| 7 | FA | Yang 2009/2010 | own |
| 8 | BA | Yang 2010, NICSO | own, Eqs. 2-6 |
| 9 | CS | Yang & Deb 2009, NaBIC | niapy 2.5.2 CuckooSearch (matches Yang's code) |
| 10 | BFO | Passino 2002, IEEE CSM | own |
| 11 | DA | Mirjalili 2016, NCA | own, Eqs. 3.1-3.10 |
| 12 | MFO | Mirjalili 2015, KBS | own, Eqs. 3.12-3.14 |
| 13 | SMO | Bansal et al. 2014, Memetic Comput. | own |
| 14 | ALO | Mirjalili 2015, AES | own, Eqs. 2.1-2.13 |
| 15 | SquirrelSA | Jain et al. 2019, SWEVO | own, Eqs. 4-17 |
| 16 | GOA | Saremi et al. 2017, AES | own, Eq. 2.7 |
| 17 | MPA | Faramarzi et al. 2020, ESWA | own, Eqs. 10-16 |
| 18 | KH | Gandomi & Alavi 2012, CNSNS | own, KH II (crossover) |
| 19 | AO | Abualigah et al. 2021, CAIE | own, Eqs. 3-17 |

Parameters: population 30 where the algorithm has a free population size (ABC: 15 food sources);
all other parameters are the values of the original paper (listed per algorithm in swarm_orig.py).
ACOR keeps the paper's archive k = 50 and m = 2 ants. Only CEO and PSO are tuned; disclosed.
Iteration schedules use the horizon implied by the budget; algorithms with a variable number of
evaluations per iteration (HHO, SMO) use the spent budget fraction as t/T. BFO repeats its
elimination-dispersal loop until the budget is spent.

## Benchmarks (two tables)

1. CEC2022, official code, D = 10, 200,000 FEs, official seed rule, 30 runs (CEO-v4 harness).
2. CEC2020 real-world RC15-RC21 through the R1 wrapper (static penalty), R1 budgets, seeds 0-29.
   All 20 algorithms are run through this same harness; the R1 swarm results are not mixed in.

D = 20 is not part of this comparison (advisor: two tables).

## Analysis (fixed)

Per benchmark: Friedman test over the 20 algorithms on mean error (CEC2022) or mean penalised objective
(RW), average ranks and the resulting position of CEO (1 = best); Holm post-hoc with CEO as control;
Wilcoxon rank-sum per function with Holm over the functions, CEO vs each algorithm (W/T/L).
Every algorithm that beats CEO is reported.

## Superseded

The first draft (mealpy-based, 2026-10-17 early morning) is not published. No test run used it
except 255 RW rows (PSO, ACO) that are discarded.

## Deviations
1. 2026-10-17, before any test run: second audit against the authors' MATLAB files and the papers
   (SWARM_AUDIT.md section 0) changed GWO (own GWO.m port), DA, GOA, MPA, HHO notes, FA, BA, AO and KH.
2. 2026-10-17: runs proceed in two phases. Phase A = 15 algorithms verified against a primary source
   (PSO reused + 14 run). Phase B = ACO, BFO, SMO, SquirrelSA after their papers are checked.
3. 2026-10-17: phase B released after checking ACOR, BFO, SMO and SquirrelSA against the original papers
   Phase B ran as three seed shards, `runs/swarm/out/ceo-swarmb-d10-s0..2`; RW after phase A.

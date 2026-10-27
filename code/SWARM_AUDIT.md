# Swarm baselines: fidelity audit (2026-10-17, before any test run)

## 0. Primary-source verification (second pass, advisor request for exact original formulas)

Original files kept in third_party/original_matlab/ (downloaded from the authors' sites).

| Alg. | Primary source checked | Result |
|---|---|---|
| GWO | authors' GWO.m (seyedalimirjalili.com) | own literal port; EvoloPy's alpha->beta->delta cascade is NOT in GWO.m, so EvoloPy GWO dropped |
| WOA | authors' WOA.m | EvoloPy WOA.py identical line by line, kept verbatim |
| DA | authors' DA.m, Levy.m, distance.m | match; food/enemy initialised to zero vectors as in DA.m |
| ALO | authors' ALO.m, Random_walk_around_antlion.m, RouletteWheelSelection.m | match after the 1-based walk-row fix |
| GOA | authors' GOA.m, S_func.m and paper Eq. 2.7 | paper Eq. 2.7 used (per-dimension s, full-distance unit vector); GOA.m applies it to dimension pairs, disclosed |
| MFO | authors' MFO.m and paper | paper used: flames from current moths + previous flames; MFO.m merges the previous iteration's moths (one-iteration lag), disclosed |
| MPA | authors' MPA.m, levy.m | match, including the Iter = 0 memory reset in both evaluation blocks |
| HHO | authors' HHO.m (aliasgharheidari.com) and paper | paper used where they differ: Levy x 0.01 (Eq. 9; HHO.m omits 0.01), comparison with stored F(X_i(t)) (HHO.m re-evaluates, costing evaluations), Y and Z clipped to the box; disclosed |
| ABC | official Python_ABC (artificialbeecolony, Karaboga's group) | match: one dimension, phi in [-1,1], fitness 1/(1+f), 0.9 fit/max + 0.1, one scout. limit = SN x D (Karaboga and Akay 2009); the repository default is 1500 = NP x D for NP = 50, D = 30 |
| AO | copy of AO.m (third-party repository) and paper | paper used where they differ: Eq. 3 X_M - X_best rand, vector X_M, U = 0.00565; disclosed |
| FA | Yang 2009 (arXiv:1003.1466), Yang 2010 (arXiv:1003.1409) | Eq. 9 with alpha = 0.2, beta0 = 1, randomisation x S_k, gamma = 1/Gamma^2 from the length-scale rule, brightest moves randomly, no alpha reduction (not in either paper) |
| BA | Yang 2010 (arXiv:1004.4170) | fmin = 0, fmax = 100 as implemented in the paper, A_i0 ~ U(1,2), r_i0 ~ U(0,1), alpha = gamma = 0.9 |
| CS | Yang and Deb 2009 (arXiv:1003.1594) | pseudocode is loose; Yang's cuckoo_search.m rules (niapy) used, pa = 0.25, n = 30 within the paper's 15-50 |
| KH | Gandomi and Alavi 2016 (JCEM 22:302, open access, same authors) | KH with crossover, N_max = 0.01, V_f = 0.02, D_max in [0.002, 0.010] (0.005 used), C_t = 0.5, epsilon = 1e-6 |
| ACOR | Socha and Dorigo 2008, EJOR (PDF supplied by the user) | Eqs. 7-9, Table 2 match; **Appendix A coordinate rotation added** (Eq. A.1 d^4 direction choice, Gram-Schmidt); solves a rotated ellipsoid to 1.8e-10 at 50k FEs |
| BFO | Passino 2002, IEEE CSM, boxed algorithm p. 61 | match after two fixes: J_cc on the sequentially updated population (step g), J_health over Nc+1 values |
| SMO | Bansal et al. 2014, Memetic Comput., Alg. 1-5 | match after fixes: prob_i with the group's max fitness, Algorithm 2 loop as written, GLL = 50 (Sect. 5.1) |
| SquirrelSA | Jain et al. 2019, SWEVO, Eqs. 2-17, Alg. 1 | match after fix: Eq. 14 relocation only for squirrels that did not move toward the hickory tree; per-iteration evaluation/sorting (Algorithm 1 omits it, Sect. 4.3 requires it) |

Paper-by-paper specifications and line-by-line analyses: ../../SW_References/*.md


Reference rule: published equations first, authors' public code for details the paper leaves open.

## 1. Library versions inspected and rejected

| Library class | Deviation from the original found by reading the code |
|---|---|
| mealpy 3.0.3 GWO, WOA, DA, GOA, AO, MFO, MPA | greedy one-to-one selection added (not in the originals) |
| mealpy WOA | random whale replaced by a fresh random point; one random number shared by A and C |
| mealpy HHO | population mean computed as a scalar (np.mean over all entries); S drawn from U(lb, ub) |
| mealpy ABC | roulette on raw objective values, so worse sources are preferred under minimisation; all dimensions changed |
| mealpy ALO | roulette on raw objective values |
| mealpy AO | loop variable `idx` overwritten inside the loop, the greedy update hits the wrong agent |
| mealpy CS | Levy step x 0.001 and 1/sqrt(epoch) damping, both absent from Yang and Deb 2009 |
| mealpy SMO, MFO, MPA | labelled "developed version" by mealpy |
| mealpy SquirrelSA | crashes (len() of an int); S_min formula differs from Eq. 13 |
| mealpy SSA | is Sparrow Search, not Salp Swarm |
| mealpy, all | harness passed epoch = 100000, freezing iteration schedules (also in the earlier baseline study GWO/HHO runs) |
| niapy 2.5.2 BA | local search and acceptance conditions inverted (rand < r, rand > A) |
| niapy KH | target term without K_hat X_hat, mutation formula differs, crossover uses the own position |
| niapy BFO | J_last not updated while swimming |
| niapy FA | moves i also when f_j == f_i, including j == i |
| EvoloPy MFO | `previous_population = Moth_pos` aliases the array; 0/1 index shift of the flame number |
| EvoloPy BAT | frequency Qmin + (Qmin - Qmax) rand (negative), bounds applied to the old position |
| EvoloPy HHO | acceptance compares with the fitness of the last evaluated hawk, S ~ N(0,1) |

Accepted as faithful: EvoloPy GWO.py and WOA.py (line-by-line ports of GWO.m and WOA.m by the authors'
group, vendored verbatim, Apache-2.0), niapy CuckooSearch (matches cuckoo_search.m).

## 2. Own implementations (swarm_orig.py): equations checked

| Alg. | Checked against | Notes |
|---|---|---|
| ACOR | Eqs. 7-9 | one kernel per ant for all dimensions; k=50, m=2, q=1e-4, xi=0.85 |
| ABC | Karaboga's code | one dimension per move, fitness 1/(1+f), p = 0.9 fit/max + 0.1, one scout, limit = SN D |
| HHO | Eqs. 1-13 | vector population mean; Y, Z compared with F(X_i(t)); cross-checked with EvoloPy HHO on shifted and plain sphere (same order of error) |
| FA | Yang 2009/2010 | strict brighter test, beta0 exp(-gamma r^2), alpha (rand-0.5) L, alpha x 0.97 per generation |
| BA | Eqs. 2-6 | per-bat loudness and pulse rate, local walk x* + eps mean(A) |
| BFO | Passino 2002 | tumble, swim with J_last update, J_cc swarming, reproduction, elimination-dispersal |
| DA | Eqs. 3.1-3.10, DA.m | separation sign, Levy when alone, walls, step cap |
| MFO | Eqs. 3.12-3.14 | flames from current moths and previous flames, 0-based flame index |
| SMO | Bansal et al. 2014 | LLP, GLP, LL/GL learning, LLD, GLD fission-fusion |
| ALO | Eqs. 2.1-2.13, ALO.m | walk row index fixed to the 1-based RWs(current_iter,:) during audit |
| SquirrelSA | Eqs. 4-17 | gliding distance from lift and drag, Eq. 13 seasonal constant |
| GOA | Eq. 2.7, GOA.m | vectorised, identical arithmetic |
| MPA | Eqs. 10-16, MPA.m | phase boundaries as in the code, two evaluations per iteration, marine memory |
| KH | Eqs. 2-21 | KH II; food centre with 1/K weights (shifted to positive if needed) |
| AO | Eqs. 3-17 | vector population mean of Eq. 4, paper spiral constants |

## 3. Checks run

- Budget exactness: every algorithm stops at exactly B evaluations (30,000 test).
- Determinism: identical best value for a repeated seed.
- Sanity on a shifted 10-D sphere at 30,000 evaluations: PSO, ACO, ABC, MFO, SMO, ALO, SquirrelSA, GOA,
  CS, MPA reach below 1e-5. GWO, WOA, HHO and AO are known to be biased toward the origin and do worse on
  shifted optima; HHO's result matches the authors' EvoloPy HHO on the same function (both converge to
  1e-11 or below when the optimum is at the origin). FA with the scale rule gamma = 1/sqrt(L) and BFO with
  the paper's step C = 0.1 barely move on a 200-wide box; this is the published parameterisation and is
  reported as found.

# Audit of the 11 unaudited opponents of tab:rw (RC15-RC33)

Started 2026-11-09, after the RC15-RC33 results were known. The purpose is fidelity to the
original publications, not the outcome of CEO. Every method found faulty is replaced or fixed
whatever its current rank, and the old results stay reported as superseded.

Code read: mealpy 3.0.3 (site-packages), runs/r1/code/algorithms.py, r1_lib.py,
CEO_v4/baselines.py.

## 1. Source of each opponent in the current tab:rw

| Method | Code used now | Audited before |
|---|---|---|
| ABC, ACOR, GWO, HHO, PSO, WOA | swarm_orig.py | yes, SWARM_AUDIT.md |
| L-SHADE | mealpy 3.0.3 evolutionary_based.SHADE.L_SHADE | no |
| LSHADE-cnEpSin | mealpy 3.0.3 sota_based.LSHADEcnEpSin | no |
| IMODE | mealpy 3.0.3 sota_based.IMODE | no |
| INFO | mealpy 3.0.3 math_based.INFO | no |
| HBA | mealpy 3.0.3 swarm_based.HBA | no |
| SCSO | mealpy 3.0.3 swarm_based.SCSO | no |
| GA | r1 algorithms.ga, generic real-coded GA | no |
| DE | r1 algorithms.de, DE/rand/1/bin F=0.5 CR=0.9 | no |
| Sawtooth GA | r1_lib.sawtooth_ga | no |
| BIPOP-CMA-ES | pycma, cma.fmin2 bipop=True | reference code of the method's author |

## 2. Findings

### mealpy L_SHADE, reference Tanabe and Fukunaga, CEC 2014
1. No linear population size reduction in effect: dyn_pop_size is computed, but self.pop is never
   truncated and every loop runs over the initial pop_size.
2. The archive receives the new trial, not the replaced parent.
3. Memory size H equals pop_size instead of 6, and the index k cycles over dyn_pop_size.
4. p drawn from U(0.15, 0.2) instead of 0.11; N_min = pop_size/5 instead of 4; archive size =
   pop_size instead of 2.6 N.
5. M_CR updated with a weighted arithmetic mean instead of the weighted Lehmer mean; no terminal
   value for CR.
6. Weights misaligned once dyn_pop_size < pop_size: fitness differences beyond dyn_pop_size stay at 1.
7. Bound handling by clipping instead of the midpoint toward the parent; replacement only on strict
   improvement instead of <=.
Verdict: not L-SHADE. The audited own implementation baselines.run_lshade, used in tab:sota and
tab:shade17, exists and is checked below.

### mealpy OriginalLSHADEcnEpSin, reference Awad et al., CEC 2017
Checked against the authors' code, third_party/original_matlab/LSHADE_cnEpSin (LSHADE_cnEpSin.m,
boundConstraint.m, updateArchive.m, gnR1R2.m), from the top-methods archive of
github.com/P-N-Suganthan/CEC2017-BoundContrained.
1. The frequency memory M_freq is never updated, so the adaptive sinusoidal configuration does not adapt.
2. The covariance-matrix crossover draws a new CR, while S_CR records the CR of the binomial branch.
3. Archive initialised with the whole initial population and capped at NP; original starts empty, rate 1.4.
4. N_min = 10 instead of 4; F clipped to [0.1, 1] instead of redrawn while <= 0.
5. Learning period 10 instead of 20; the sinusoidal configuration is chosen per individual instead of once
   per generation, and by a random draw after the learning period instead of the larger success rate.
6. Population reduction by epochs, not by evaluations.
Verdict: departs from the original. Ported in rw_evo_audited.run_lshade_cnepsin.
G_Max: the authors hard-code 2163, 2745, 3022 and 3401 for D = 10, 30, 50, 100. These equal the
generation count of the method's own reduction schedule with 18 D to 4 individuals and 10,000 D
evaluations; lpsr_generations reproduces all four, so the same rule sets G_Max for each RW budget.

### mealpy OriginalIMODE, reference Sallam et al., CEC 2020
Checked against the authors' code, third_party/original_matlab/IMODE (IMODE_main.m, IMODE.m, han_boun.m,
LS2.m, updateArchive.m), from github.com/P-N-Suganthan/2020-Bound-Constrained-Opt-Benchmark.
1. Operator credit is computed on a second, fresh random operator assignment, so it is unrelated
   to the operator that produced each trial.
2. CORRECTION 2026-11-09: operator 3, F*x_r1 + F*(x_phi - x_r3), is exactly the form of IMODE.m, so this
   is not a mealpy defect. The first reading of this audit was wrong. Paper check pending.
3. No linear population reduction; population max(30, 4D) instead of 6 D^2; memory 5 slots at 0.5
   instead of 20 D slots at 0.2; archive rate 2.6 not used.
4. No SQP local search in the last 15 % of the budget.
5. Archive stores the improved trials, not the replaced parents, and de-duplicates with set() on objects.
6. Midpoint repair uses the upper bound for a lower violation and vice versa.
Verdict: not IMODE. Ported from the authors' code in rw_evo_audited.run_imode.

### mealpy OriginalHBA, reference Hashim et al. 2022
1. alpha = C exp(-T/T) is a constant 2/e; the density factor must decrease as C exp(-t/T).
2. The choice between digging and honey phases is made per dimension, not per badger.
Verdict: departs; fix 1 alone changes the search schedule.

### mealpy OriginalINFO, reference Ahmadianfar et al. 2022
Checked against the authors' INFO.m, third_party/original_matlab/INFO, from aliasgharheidari.com.
1. No greedy selection; INFO.m keeps a new vector only if it is better, and updates X, Cost and Best_X
   immediately inside the loop over individuals.
2. Coefficients drawn from U(0,1) where INFO.m uses randn: z1, z2 in Eq. 8, Randn and the inner factor of
   Eq. 11.
3. Eq. 10: mealpy takes the trial component with probability 0.05 and mu from U(0, 0.05); INFO.m uses 0.5 and
   mu = 0.05 randn per component.
4. Two places where mealpy and INFO.m differ and the paper must decide: mealpy takes |MM|/omg inside exp,
   INFO.m takes MM/omg; INFO.m writes (Best - Better) twice in MM of Eq. 4.7-4.9, mealpy uses Best - Worst
   for the second term. The port follows INFO.m until the paper is checked.
Verdict: departs from the authors' code. Ported in rw_evo_audited.run_info, nP = 30.

### mealpy OriginalSCSO, reference Seyyedabbasi and Kiani 2023
Matches the published equations as far as checked: r_G, R, roulette angle over 1..360, random
candidate in exploration. Whether r and R are drawn per agent or per dimension must be checked
against the authors' SCSO.m.

### GA, Sawtooth GA, DE of the earlier study
- GA is a generic real-coded GA: tournament, arithmetic crossover 0.8, Gaussian mutation 0.1 with
  sigma 0.1 of the range, one elite. It is not Holland's binary GA, so the manuscript must call it
  a real-coded GA and not cite it as Holland 1975.
- Sawtooth GA follows Koumousis and Katsaras 2006: mean size 30, amplitude 27, period 20, worst
  removed while shrinking, random refill at each period. Within the recommended ranges of the paper.
- DE is DE/rand/1/bin with F=0.5, CR=0.9, clip repair and strict replacement. tab:sota uses a
  different DE, baselines.run_de, tuned with the same rule as CEO. One DE should be used throughout.

### Own L-SHADE and jSO, baselines.py, checked line by line
L-SHADE: r_Ninit=18, H=6, p=0.11, archive 2.6 N, M=0.5, terminal CR, weighted Lehmer means,
midpoint repair, <= replacement, reduction by evaluations to 4. Matches Tanabe and Fukunaga 2014.
jSO: N_init = 25 sqrt(D) ln(D), H=5, last slot 0.9, MF=0.3, MCR=0.8, p from 0.25 to 0.125, CR
floors 0.7 and 0.6, F cap 0.7 before 60 %, F_w 0.7, 0.8, 1.2, averaged memory update, archive 1.0.
Matches Brest et al. 2017.

## 3. Related finding outside RC15-RC33
tab:sota uses mealpy OriginalGWO and OriginalHHO through baselines.run_gwo and run_hho. SWARM_AUDIT.md
rejected exactly these versions. They rank last in tab:sota, but the manuscript criticises library
versions while using two of them.

## 4. Ports, tests and remaining checks (2026-11-09)

rw_evo_audited.py ports LSHADE-cnEpSin, IMODE and INFO. Deviations from the authors' code, all marked
DEVIATION in the source:
- LSHADE-cnEpSin: the trial after the eigen-space crossover is repaired with the method's own midpoint
  rule; the MATLAB code leaves it unrepaired because CEC2017 evaluates outside the box.
- IMODE: fmincon/sqp replaced by scipy SLSQP within the same 2 % evaluation cap; the upper-bound branch of
  repair type 2 uses 2 x_U - x, where han_boun.m writes 2 x_L - x; the run uses the whole budget instead of
  stopping 4 N evaluations early.
- INFO: a non-finite trial is evaluated at the parent and rejected, as a NaN trial is in INFO.m.
Tests: the G_Max rule reproduces the four constants of LSHADE_cnEpSin.m; exact budget 100,000 on a shifted
sphere at D = 10, final error <= 1e-24, identical results for the same seed; 5,000-FE dry runs on RC20,
RC25, RC26 and RC33 stay inside the box and use exactly the budget.

Open before launch:
- HBA and SCSO: no author code is publicly downloadable; the papers are needed. Held out until checked.
- Paper checks for IMODE operator 3 and INFO item 4.
- tab:sota: audited GWO and HHO already exist on CEC2022 D = 10 with the same seed rule and budget
  (runs/swarm, 360 runs each) and are reused; D = 20 needs 2 x 12 x 30 new runs.

## 5. HBA and SCSO against the authors' code (2026-11-09)

Sources kept in third_party/original_matlab/HBA/zip (MathWorks File Exchange 98204, honey_badger-6.0.0, same
HBA.m as the copy the user supplied) and third_party/original_matlab/SCSO (sand_cat_swarm-1.0.0, identical to
the user's copy in "Swarm Intelligen code/SCSO").

HBA.m vs run_hba: alpha = C exp(-t/tmax), intensity with eps and the cyclic neighbour, one r and F per badger,
r3, r4, r5, r7 per dimension, clipping, greedy replacement with < all match. One difference: HBA.m updates
X_prey once per iteration, after the loop over badgers; run_hba updates it at once inside the loop. The
paper's Algorithm 1 decides; as recalled it updates x_prey inside the loop with <=, which would also change
< to <= in both tests. Not yet confirmed, because the paper PDF is not on disk. HBA stays held out.

SCSO.m vs run_scso: clipping then evaluation at the start of each iteration, r_G = S - S t/T with t from 0,
r and R per cat, roulette over 1..360 weighted by the angle value, a random candidate drawn per dimension,
in-place updates, no greedy step all match. Open points for the paper:
1. SCSO.m takes cos of the integer angle in radians; run_scso uses degrees. The port follows the paper's
   wording of an angle between 0 and 360 degrees; to be confirmed.
2. Exploration: SCSO.m uses a random cat; the paper names the term the best-candidate position Pbc. If the
   paper defines Pbc as a random cat the port stands, otherwise it changes.
SCSO stays held out until these two points are read in the paper.

## 6. Resolution, 2026-11-09 (protocol D2)
INFO: the authors' brief version of the paper gives Eqs. 1-6 and Appendix A. Paper vs code conflicts found:
r range, extra rand on sigma x MeanRule, abs in the weight exponent (GitHub INFO.m only), +1 vs +eps in the
weighted-mean denominator, best update inside vs after the generation loop. The port now follows the paper on
all five. This replaces the earlier note that the port follows INFO.m on item 4.
HBA and SCSO: in the absence of the papers the ports follow the authors' code exactly; the prey update now
happens after each iteration and the SCSO angle is used in radians, as in the code.
MODE toolbox (Reynoso Meza): reviewed, not used, see D2 item 6.
Independent line-by-line review, 2026-11-09, against INFO.m (2022 and GitHub), the paper pages, HBA.m and
SCSO.m: all five INFO paper-over-code choices confirmed on the paper pages; HBA and SCSO match the code item by
item. Two edge cases fixed after the review: INFO with omega = 0 now gives +-inf or nan as MATLAB instead of
NaN weights, with np.cos so a non-finite cost cannot raise; SCSO roulette tie side as find(r <= C, 1).
Tests rerun after the fixes: all pass.

## 7. HBA paper read, 2026-11-09 (D2a)
Algorithm 1 of Hashim et al. 2022: intensity per badger inside the loop, <= acceptance, immediate prey update
with <=. HBA.m computes I once per iteration, uses < and updates the prey after the loop. Port now follows the
paper; open details (per-dimension r3, r4, r5, r7, eps in distances, cyclic x_{N+1}) follow HBA.m.

## 8. SCSO and INFO papers read, 2026-11-09 (D2b)
SCSO: angle in degrees per Algorithm 1; random candidate kept (paper: search step based on a random position).
INFO: published Eqs. 4.2-4.9 use |df/omega|; the brief version dropped the bars. Abs restored; the four other
paper-over-code items confirmed.

## 9. HBA zero distance, 2026-11-10 (D2c)
RWB__HBA__RC20__s00-30 raised ZeroDivisionError: d_i^2 = sum((x_i - x_prey + eps)^2) is exactly 0 when every
coordinate of x_i lies 2 ulp below a prey at the upper bound 1. Python's float division raises where MATLAB
returns Inf or NaN. run_hba now uses numpy division with errstate, so I_i = Inf (NaN when S_i = 0), clipping maps
+-Inf to the box and NaN candidates are rejected by <=, as in HBA.m. Bit-identical to the old code whenever
d_i^2 > 0 (checked on five problems, three seeds). Only the failed unit is rerun.

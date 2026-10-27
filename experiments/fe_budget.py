# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""
Function-evaluation (FE) matched execution harness.

Why: the original sweep (run_experiments.py) matched *iterations*
(pop=30, T=500) across algorithms. Instrumenting the objective showed that
this gives unequal budgets: ABC spends ~2N evaluations per iteration
(employed + onlooker phases) and HHO ~1.5-1.7N (extra Levy-dive
evaluations in its progressive rapid-dive branches), versus ~N for the
others. Iteration-matched results therefore gave ABC ~2x and HHO ~1.6x the
budget of every other algorithm.

This harness enforces an identical hard FE budget on every algorithm:
  * the objective is wrapped by a counter that records the best value ever
    evaluated (standard CEC practice: best error found within the budget);
  * once the budget is spent, the wrapper raises BudgetExhausted, which the
    runner catches -- no algorithm can use more than `budget` evaluations;
  * each algorithm's own iteration horizon `max_iter` (which drives its
    time-dependent schedule, e.g. GWO's a: 2->0, HHO's escaping energy,
    CEO's period scaling) is set from its nominal per-iteration cost so the
    schedule completes close to the budget rather than being truncated
    half-way.
"""
import math
import numpy as np


class BudgetExhausted(Exception):
    pass


# Nominal evaluations per iteration, in units of the population size N,
# measured by instrumenting each implementation on F1/F9/F21 (pop=30,
# T=500, seed 0): ABC 30030-30109 FEs -> 2.0N; HHO 22905-25984 -> ~1.65N;
# CEO 15096-15220 -> ~1.01N; all others exactly N (+ initial population).
NOMINAL_COST = {
    "ABC": 2.0,
    "HHO": 1.65,
}


class CountingObjective:
    def __init__(self, func, budget, n_checkpoints=100):
        self.func = func
        self.budget = int(budget)
        self.count = 0
        self.best = np.inf
        self.checkpoints = np.linspace(self.budget / n_checkpoints, self.budget,
                                       n_checkpoints).astype(int)
        self.curve = np.full(n_checkpoints, np.nan)
        self._next_cp = 0

    def _record(self):
        while (self._next_cp < len(self.checkpoints)
               and self.count >= self.checkpoints[self._next_cp]):
            self.curve[self._next_cp] = self.best
            self._next_cp += 1

    def __call__(self, X):
        X = np.atleast_2d(X)
        remaining = self.budget - self.count
        if remaining <= 0:
            raise BudgetExhausted()
        if X.shape[0] > remaining:
            vals = self.func(X[:remaining])
            self.count += remaining
            self.best = min(self.best, float(np.min(vals)))
            self._record()
            raise BudgetExhausted()
        vals = self.func(X)
        self.count += X.shape[0]
        self.best = min(self.best, float(np.min(vals)))
        self._record()
        return vals

    def finalize(self):
        # fill any checkpoints not yet reached (algorithm finished early)
        for k in range(self._next_cp, len(self.curve)):
            self.curve[k] = self.best
        return self.best, self.curve, self.count


def max_iter_for(alg_name, budget, pop_size):
    cost = NOMINAL_COST.get(alg_name, 1.0)
    return max(int(math.floor((budget - pop_size) / (cost * pop_size))), 10)


def run_fe_matched(alg_name, alg_func, func, lb, ub, dim, pop_size, budget,
                   seed, _epoch_scale=1.0, **kwargs):
    """_epoch_scale lengthens the iteration schedule for methods that spend
    fewer evaluations per iteration as a run proceeds (linear population
    reduction); the hard FE cap still stops the run at the budget."""
    obj = CountingObjective(func, budget)
    max_iter = max(int(max_iter_for(alg_name, budget, pop_size) * _epoch_scale), 10)
    try:
        alg_func(obj, lb, ub, dim, pop_size=pop_size, max_iter=max_iter,
                 seed=seed, **kwargs)
    except BudgetExhausted:
        pass
    best, curve, used = obj.finalize()
    return dict(best_fitness=best, curve=curve, fe_used=used, max_iter=max_iter)

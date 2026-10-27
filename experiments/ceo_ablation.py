# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""
Component-switchable Cicada Emergence Optimizer for the ablation study.

With all switches at their defaults, `ceo_switchable` reproduces
algorithms.ceo_v2(..., target_emergences=20) exactly (same RNG draw order);
this is checked by `_self_test()` below. The final CEO evaluated in the
manuscript is that configuration, frozen before the CEC2017 runs.

Switches (each isolates one component):
  use_emergence  False -> broods never emerge (nymph-phase search only);
                  PDO can then never fire either.
  use_mating     False -> no intra-brood arithmetic crossover after a jump.
  use_pdo        False -> stagnation never changes a brood's period/phase
                  and never reseeds the queen.
  reseed_queen   False -> PDO changes period/phase only (queen kept).
  prime_periods  False -> initial periods are plain integers (not rounded to
                  primes) and PDO moves to the next unused integer with no
                  coprimality test. Tests whether primality matters at all.
  n_broods       1 -> a single brood (no independently clocked broods).
  trace          optional callback(t, X, fit, brood_id, emerged, pdo_fired)
                  called after each iteration; observes only.
  sync_clocks    True -> all broods share one period and one phase, so they
                  emerge together (control for staggering; used with
                  use_pdo=False in run_sync_control.py, not in ABLATIONS).
"""
import numpy as np
import sympy

from algorithms import _bounds, _clip, _init_pop, _levy, _next_prime_coprime

_SPREAD = [0.55, 0.75, 1.0, 1.35, 1.7, 2.1, 2.6, 3.2]


def _initial_periods(max_iter, n_broods, target_emergences, prime_periods):
    base = max(int(max_iter / target_emergences), 5)
    periods, used = [], set()
    for s in _SPREAD[:n_broods]:
        if prime_periods:
            cand = sympy.nextprime(max(int(base * s), 5) - 1)
            while cand in used:
                cand = sympy.nextprime(cand)
        else:
            cand = max(int(base * s), 5)
            while cand in used:
                cand += 1
        used.add(cand)
        periods.append(cand)
    return np.array(periods, dtype=int)


def _next_plain(p_current, used):
    cand = p_current + 1
    while cand in used:
        cand += 1
    return cand


def ceo_switchable(func, lb, ub, dim, pop_size, max_iter, seed=0,
                   n_broods=4, stagnation_threshold=3, eta0=2.2, L0=0.25,
                   gamma=4.0, target_emergences=20, rel_eps=1e-4,
                   use_emergence=True, use_mating=True, use_pdo=True,
                   reseed_queen=True, prime_periods=True, sync_clocks=False,
                   trace=None):
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    X = _init_pop(pop_size, dim, lb, ub, rng)
    fit = func(X)

    brood_id = np.array([i % n_broods for i in range(pop_size)])
    periods = _initial_periods(max_iter, n_broods, target_emergences, prime_periods)
    phases = rng.integers(0, periods, size=n_broods)
    if sync_clocks:
        # control for staggering: every brood shares the median brood's
        # period and brood 0's phase; drawn after the default RNG calls so
        # sync_clocks=False keeps the frozen RNG order bit for bit
        common = periods[min(2, n_broods - 1)]
        periods = np.full(n_broods, common, dtype=int)
        phases = np.full(n_broods, phases[0] % common, dtype=int)
    brood_best_fit = np.array([fit[brood_id == b].min() for b in range(n_broods)])
    stagnation = np.zeros(n_broods, dtype=int)
    last_improve_iter = np.zeros(n_broods, dtype=int)
    used_periods = set(periods.tolist())

    best_idx = np.argmin(fit)
    gbest, gbest_fit = X[best_idx].copy(), fit[best_idx]
    curve = np.zeros(max_iter)
    pdo_triggers = 0

    for t in range(max_iter):
        emerged, pdo_fired = [], []
        for b in range(n_broods):
            members = np.where(brood_id == b)[0]
            emerging = use_emergence and ((t - phases[b]) % periods[b] == 0)
            if emerging:
                emerged.append(b)
            if not emerging:
                depth_frac = ((t - phases[b]) % periods[b]) / periods[b]
                eta = eta0 * (0.25 + 0.75 * (1.0 - depth_frac)) * (1.0 - 0.7 * t / max_iter)
                b_best = X[members[np.argmin(fit[members])]]
                r1 = rng.uniform(0, 1, size=(len(members), dim))
                r2 = rng.uniform(0, 1, size=(len(members), dim))
                X[members] = X[members] + eta * r1 * (b_best - X[members]) \
                                          + eta * r2 * (gbest - X[members])
                X[members] = _clip(X[members], lb, ub)
                fit[members] = func(X[members])
            else:
                queen = members[np.argmin(fit[members])]
                emerging_members = members[members != queen]
                amp = L0 * np.exp(-gamma * t / max_iter)
                for i in emerging_members:
                    levy = np.clip(_levy(dim, rng), -3, 3)
                    X[i] = X[queen] + amp * (ub - lb) * levy
                X[emerging_members] = _clip(X[emerging_members], lb, ub)
                fit[emerging_members] = func(X[emerging_members])
                if use_mating:
                    shuffled = rng.permutation(emerging_members)
                    for k in range(0, len(shuffled) - 1, 2):
                        i, j = shuffled[k], shuffled[k + 1]
                        lam = rng.uniform()
                        child = _clip(lam * X[i] + (1 - lam) * X[j], lb, ub)
                        child_fit = func(child.reshape(1, -1))[0]
                        worse = i if fit[i] > fit[j] else j
                        if child_fit < fit[worse]:
                            X[worse] = child
                            fit[worse] = child_fit
                new_best = fit[members].min()
                rel_improve = (brood_best_fit[b] - new_best) / (abs(brood_best_fit[b]) + 1e-12)
                if rel_improve > rel_eps:
                    brood_best_fit[b] = new_best
                    stagnation[b] = 0
                    last_improve_iter[b] = t
                else:
                    stagnation[b] += 1
                if use_pdo and stagnation[b] >= stagnation_threshold:
                    pdo_triggers += 1
                    pdo_fired.append(b)
                    tau = max(t - last_improve_iter[b], periods[b])
                    used_periods.discard(periods[b])
                    if prime_periods:
                        new_p = _next_prime_coprime(periods[b], tau, used_periods)
                    else:
                        new_p = _next_plain(periods[b], used_periods)
                    used_periods.add(new_p)
                    periods[b] = new_p
                    phases[b] = rng.integers(0, periods[b])
                    stagnation[b] = 0
                    if reseed_queen:
                        queen_new = rng.uniform(lb, ub)
                        X[queen] = queen_new
                        fit[queen] = func(queen_new.reshape(1, -1))[0]
                        brood_best_fit[b] = fit[members].min()
        best_idx = np.argmin(fit)
        if fit[best_idx] < gbest_fit:
            gbest_fit = fit[best_idx]
            gbest = X[best_idx].copy()
        curve[t] = gbest_fit
        if trace is not None:
            # read-only observer for behavior figures; draws no random numbers
            trace(t, X, fit, brood_id, emerged, pdo_fired)
    return dict(best_fitness=gbest_fit, best_position=gbest, curve=curve,
                pdo_triggers=pdo_triggers)


def ceo_final(func, lb, ub, dim, pop_size, max_iter, seed=0):
    """The frozen CEO configuration reported in the manuscript."""
    return ceo_switchable(func, lb, ub, dim, pop_size, max_iter, seed=seed)


ABLATIONS = {
    "CEO (full)": {},
    "w/o emergence": dict(use_emergence=False),
    "w/o mating": dict(use_mating=False),
    "w/o PDO": dict(use_pdo=False),
    "w/o queen reseed": dict(reseed_queen=False),
    "non-prime periods": dict(prime_periods=False),
    "single brood": dict(n_broods=1),
    "sparse schedule (E=7)": dict(target_emergences=7),
}


def _self_test():
    from algorithms import ceo_v2
    from benchmarks import BENCHMARKS
    for fn in ["F1", "F9", "F21"]:
        func, lb, ub, dim, _ = BENCHMARKS[fn]
        for seed in range(3):
            a = ceo_v2(func, lb, ub, dim, pop_size=30, max_iter=500, seed=seed,
                       target_emergences=20)["best_fitness"]
            b = ceo_final(func, lb, ub, dim, pop_size=30, max_iter=500, seed=seed)["best_fitness"]
            assert a == b, (fn, seed, a, b)
    print("self-test OK: ceo_final == ceo_v2(target_emergences=20)")


if __name__ == "__main__":
    _self_test()

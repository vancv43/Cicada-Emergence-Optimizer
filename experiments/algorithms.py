# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""
Reference implementations of 8 baseline metaheuristics (PSO, GA, DE, ACOR,
ABC, GWO, WOA, HHO) plus the proposed Cicada Emergence Optimizer (CEO).

All optimizers share the signature:
    optimize(func, lb, ub, dim, pop_size, max_iter, seed) ->
        dict(best_fitness, best_position, curve)
where `curve` is an array of length max_iter with the best-so-far fitness
at every iteration (for convergence plots). `lb`/`ub` may be scalars or
length-`dim` arrays.
"""
import math
import numpy as np


def _bounds(lb, ub, dim):
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    if lb.ndim == 0:
        lb = np.full(dim, lb)
    if ub.ndim == 0:
        ub = np.full(dim, ub)
    return lb, ub


def _clip(X, lb, ub):
    return np.clip(X, lb, ub)


def _init_pop(pop_size, dim, lb, ub, rng):
    return rng.uniform(lb, ub, size=(pop_size, dim))


# ----------------------------------------------------------------- PSO -----
def pso(func, lb, ub, dim, pop_size, max_iter, seed=0,
        w_max=0.9, w_min=0.4, c1=2.0, c2=2.0):
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    X = _init_pop(pop_size, dim, lb, ub, rng)
    V = np.zeros_like(X)
    fit = func(X)
    pbest, pbest_fit = X.copy(), fit.copy()
    g_idx = np.argmin(pbest_fit)
    gbest, gbest_fit = pbest[g_idx].copy(), pbest_fit[g_idx]
    curve = np.zeros(max_iter)
    vmax = 0.2 * (ub - lb)
    for t in range(max_iter):
        w = w_max - (w_max - w_min) * t / max_iter
        r1 = rng.uniform(0, 1, size=X.shape)
        r2 = rng.uniform(0, 1, size=X.shape)
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        V = np.clip(V, -vmax, vmax)
        X = _clip(X + V, lb, ub)
        fit = func(X)
        improved = fit < pbest_fit
        pbest[improved] = X[improved]
        pbest_fit[improved] = fit[improved]
        g_idx = np.argmin(pbest_fit)
        if pbest_fit[g_idx] < gbest_fit:
            gbest_fit = pbest_fit[g_idx]
            gbest = pbest[g_idx].copy()
        curve[t] = gbest_fit
    return dict(best_fitness=gbest_fit, best_position=gbest, curve=curve)


# ------------------------------------------------------------------ GA -----
def ga(func, lb, ub, dim, pop_size, max_iter, seed=0,
       crossover_rate=0.8, mutation_rate=0.1):
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    X = _init_pop(pop_size, dim, lb, ub, rng)
    fit = func(X)
    best_idx = np.argmin(fit)
    gbest, gbest_fit = X[best_idx].copy(), fit[best_idx]
    curve = np.zeros(max_iter)
    for t in range(max_iter):
        # tournament selection
        idx_a = rng.integers(0, pop_size, size=pop_size)
        idx_b = rng.integers(0, pop_size, size=pop_size)
        winners = np.where(fit[idx_a] < fit[idx_b], idx_a, idx_b)
        parents = X[winners]
        # crossover (arithmetic, pairwise)
        children = parents.copy()
        for i in range(0, pop_size - 1, 2):
            if rng.uniform() < crossover_rate:
                alpha = rng.uniform(0, 1, size=dim)
                p1, p2 = parents[i].copy(), parents[i + 1].copy()
                children[i] = alpha * p1 + (1 - alpha) * p2
                children[i + 1] = alpha * p2 + (1 - alpha) * p1
        # mutation (gaussian)
        mut_mask = rng.uniform(0, 1, size=children.shape) < mutation_rate
        sigma = 0.1 * (ub - lb)
        children = children + mut_mask * rng.normal(0, 1, size=children.shape) * sigma
        children = _clip(children, lb, ub)
        child_fit = func(children)
        # elitism: keep global best
        worst_idx = np.argmax(child_fit)
        children[worst_idx] = gbest
        child_fit[worst_idx] = gbest_fit
        X, fit = children, child_fit
        best_idx = np.argmin(fit)
        if fit[best_idx] < gbest_fit:
            gbest_fit = fit[best_idx]
            gbest = X[best_idx].copy()
        curve[t] = gbest_fit
    return dict(best_fitness=gbest_fit, best_position=gbest, curve=curve)


# ------------------------------------------------------------------ DE -----
def de(func, lb, ub, dim, pop_size, max_iter, seed=0, F=0.5, CR=0.9):
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    X = _init_pop(pop_size, dim, lb, ub, rng)
    fit = func(X)
    best_idx = np.argmin(fit)
    gbest, gbest_fit = X[best_idx].copy(), fit[best_idx]
    curve = np.zeros(max_iter)
    for t in range(max_iter):
        V = np.zeros_like(X)
        for i in range(pop_size):
            idxs = [j for j in range(pop_size) if j != i]
            a, b, c = X[rng.choice(idxs, 3, replace=False)]
            V[i] = a + F * (b - c)
        V = _clip(V, lb, ub)
        mask = rng.uniform(0, 1, size=X.shape) < CR
        j_rand = rng.integers(0, dim, size=pop_size)
        for i in range(pop_size):
            mask[i, j_rand[i]] = True
        U = np.where(mask, V, X)
        u_fit = func(U)
        improved = u_fit < fit
        X[improved] = U[improved]
        fit[improved] = u_fit[improved]
        best_idx = np.argmin(fit)
        if fit[best_idx] < gbest_fit:
            gbest_fit = fit[best_idx]
            gbest = X[best_idx].copy()
        curve[t] = gbest_fit
    return dict(best_fitness=gbest_fit, best_position=gbest, curve=curve)


# --------------------------------------------------------------- ACOR ------
def acor(func, lb, ub, dim, pop_size, max_iter, seed=0, archive_size=None, q=0.5, xi=0.85):
    """Ant Colony Optimization for continuous domains (Socha & Dorigo, 2008)."""
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    archive_size = archive_size or pop_size
    archive = _init_pop(archive_size, dim, lb, ub, rng)
    fit = func(archive)
    order = np.argsort(fit)
    archive, fit = archive[order], fit[order]
    curve = np.zeros(max_iter)
    ranks = np.arange(1, archive_size + 1)
    w = (1.0 / (q * archive_size * np.sqrt(2 * np.pi))) * \
        np.exp(-(ranks - 1) ** 2 / (2 * q ** 2 * archive_size ** 2))
    probs = w / np.sum(w)
    for t in range(max_iter):
        new_sols = np.zeros((pop_size, dim))
        for k in range(pop_size):
            l = rng.choice(archive_size, p=probs)
            sigma = xi * np.sum(np.abs(archive - archive[l]), axis=0) / (archive_size - 1 + 1e-12)
            new_sols[k] = archive[l] + rng.normal(0, 1, size=dim) * sigma
        new_sols = _clip(new_sols, lb, ub)
        new_fit = func(new_sols)
        combined = np.vstack([archive, new_sols])
        combined_fit = np.concatenate([fit, new_fit])
        order = np.argsort(combined_fit)[:archive_size]
        archive, fit = combined[order], combined_fit[order]
        curve[t] = fit[0]
    return dict(best_fitness=fit[0], best_position=archive[0].copy(), curve=curve)


# ------------------------------------------------------------------ ABC ----
def abc(func, lb, ub, dim, pop_size, max_iter, seed=0, limit=None):
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    limit = limit or (pop_size * dim // 2)
    n_employed = pop_size
    X = _init_pop(n_employed, dim, lb, ub, rng)
    fit = func(X)
    trial = np.zeros(n_employed, dtype=int)
    best_idx = np.argmin(fit)
    gbest, gbest_fit = X[best_idx].copy(), fit[best_idx]
    curve = np.zeros(max_iter)

    def greedy_search(i):
        k = rng.choice([j for j in range(n_employed) if j != i])
        phi = rng.uniform(-1, 1, size=dim)
        cand = X[i] + phi * (X[i] - X[k])
        cand = _clip(cand, lb, ub)
        cand_fit = func(cand.reshape(1, -1))[0]
        if cand_fit < fit[i]:
            X[i] = cand
            fit[i] = cand_fit
            trial[i] = 0
        else:
            trial[i] += 1

    for t in range(max_iter):
        for i in range(n_employed):
            greedy_search(i)
        fit_pos = np.where(fit >= 0, 1.0 / (1.0 + fit), 1.0 + np.abs(fit))
        probs = fit_pos / np.sum(fit_pos)
        i = 0
        count = 0
        while count < n_employed:
            if rng.uniform() < probs[i]:
                greedy_search(i)
                count += 1
            i = (i + 1) % n_employed
        scout_idx = np.argmax(trial)
        if trial[scout_idx] > limit:
            X[scout_idx] = rng.uniform(lb, ub, size=dim)
            fit[scout_idx] = func(X[scout_idx].reshape(1, -1))[0]
            trial[scout_idx] = 0
        best_idx = np.argmin(fit)
        if fit[best_idx] < gbest_fit:
            gbest_fit = fit[best_idx]
            gbest = X[best_idx].copy()
        curve[t] = gbest_fit
    return dict(best_fitness=gbest_fit, best_position=gbest, curve=curve)


# ------------------------------------------------------------------ GWO ----
def gwo(func, lb, ub, dim, pop_size, max_iter, seed=0):
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    X = _init_pop(pop_size, dim, lb, ub, rng)
    fit = func(X)
    curve = np.zeros(max_iter)

    def top3(X, fit):
        order = np.argsort(fit)[:3]
        return X[order[0]].copy(), fit[order[0]], X[order[1]].copy(), fit[order[1]], X[order[2]].copy(), fit[order[2]]

    alpha, alpha_f, beta, beta_f, delta, delta_f = top3(X, fit)
    for t in range(max_iter):
        a = 2 - 2 * t / max_iter
        for leader, in [(alpha,)]:
            pass
        A1 = 2 * a * rng.uniform(0, 1, size=X.shape) - a
        C1 = 2 * rng.uniform(0, 1, size=X.shape)
        D_alpha = np.abs(C1 * alpha - X)
        X1 = alpha - A1 * D_alpha

        A2 = 2 * a * rng.uniform(0, 1, size=X.shape) - a
        C2 = 2 * rng.uniform(0, 1, size=X.shape)
        D_beta = np.abs(C2 * beta - X)
        X2 = beta - A2 * D_beta

        A3 = 2 * a * rng.uniform(0, 1, size=X.shape) - a
        C3 = 2 * rng.uniform(0, 1, size=X.shape)
        D_delta = np.abs(C3 * delta - X)
        X3 = delta - A3 * D_delta

        X = _clip((X1 + X2 + X3) / 3.0, lb, ub)
        fit = func(X)
        combined_X = np.vstack([X, alpha[None, :], beta[None, :], delta[None, :]])
        combined_f = np.concatenate([fit, [alpha_f, beta_f, delta_f]])
        alpha, alpha_f, beta, beta_f, delta, delta_f = top3(combined_X, combined_f)
        curve[t] = alpha_f
    return dict(best_fitness=alpha_f, best_position=alpha, curve=curve)


# ------------------------------------------------------------------ WOA ----
def woa(func, lb, ub, dim, pop_size, max_iter, seed=0, b=1.0):
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    X = _init_pop(pop_size, dim, lb, ub, rng)
    fit = func(X)
    best_idx = np.argmin(fit)
    gbest, gbest_fit = X[best_idx].copy(), fit[best_idx]
    curve = np.zeros(max_iter)
    for t in range(max_iter):
        a = 2 - 2 * t / max_iter
        for i in range(pop_size):
            r1, r2 = rng.uniform(), rng.uniform()
            A = 2 * a * r1 - a
            C = 2 * r2
            p = rng.uniform()
            if p < 0.5:
                if abs(A) < 1:
                    D = np.abs(C * gbest - X[i])
                    X[i] = gbest - A * D
                else:
                    rand_idx = rng.integers(0, pop_size)
                    D = np.abs(C * X[rand_idx] - X[i])
                    X[i] = X[rand_idx] - A * D
            else:
                l = rng.uniform(-1, 1)
                Dp = np.abs(gbest - X[i])
                X[i] = Dp * np.exp(b * l) * np.cos(2 * np.pi * l) + gbest
        X = _clip(X, lb, ub)
        fit = func(X)
        best_idx = np.argmin(fit)
        if fit[best_idx] < gbest_fit:
            gbest_fit = fit[best_idx]
            gbest = X[best_idx].copy()
        curve[t] = gbest_fit
    return dict(best_fitness=gbest_fit, best_position=gbest, curve=curve)


# ------------------------------------------------------------------ HHO ----
def _levy(dim, rng, beta=1.5):
    sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
              (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, sigma, size=dim)
    v = rng.normal(0, 1, size=dim)
    return u / (np.abs(v) ** (1 / beta))


def hho(func, lb, ub, dim, pop_size, max_iter, seed=0):
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    X = _init_pop(pop_size, dim, lb, ub, rng)
    fit = func(X)
    best_idx = np.argmin(fit)
    rabbit, rabbit_fit = X[best_idx].copy(), fit[best_idx]
    curve = np.zeros(max_iter)
    for t in range(max_iter):
        E1 = 2 * (1 - t / max_iter)
        mean_X = np.mean(X, axis=0)
        for i in range(pop_size):
            E0 = 2 * rng.uniform() - 1
            E = E1 * E0
            q = rng.uniform()
            if abs(E) >= 1:  # exploration
                if q >= 0.5:
                    rand_idx = rng.integers(0, pop_size)
                    X[i] = X[rand_idx] - rng.uniform() * np.abs(X[rand_idx] - 2 * rng.uniform() * X[i])
                else:
                    X[i] = (rabbit - mean_X) - rng.uniform() * (lb + rng.uniform() * (ub - lb))
            else:  # exploitation
                r = rng.uniform()
                if r >= 0.5 and abs(E) >= 0.5:
                    J = 2 * (1 - rng.uniform())
                    X[i] = (rabbit - X[i]) - E * np.abs(J * rabbit - X[i])
                elif r >= 0.5 and abs(E) < 0.5:
                    X[i] = rabbit - E * np.abs(rabbit - X[i])
                elif r < 0.5 and abs(E) >= 0.5:
                    J = 2 * (1 - rng.uniform())
                    Y = rabbit - E * np.abs(J * rabbit - X[i])
                    Y = _clip(Y, lb, ub)
                    if func(Y.reshape(1, -1))[0] < fit[i]:
                        X[i] = Y
                    else:
                        Z = Y + rng.normal(0, 1, size=dim) * _levy(dim, rng)
                        Z = _clip(Z, lb, ub)
                        if func(Z.reshape(1, -1))[0] < fit[i]:
                            X[i] = Z
                else:
                    J = 2 * (1 - rng.uniform())
                    Y = rabbit - E * np.abs(J * rabbit - mean_X)
                    Y = _clip(Y, lb, ub)
                    if func(Y.reshape(1, -1))[0] < fit[i]:
                        X[i] = Y
                    else:
                        Z = Y + rng.normal(0, 1, size=dim) * _levy(dim, rng)
                        Z = _clip(Z, lb, ub)
                        if func(Z.reshape(1, -1))[0] < fit[i]:
                            X[i] = Z
        X = _clip(X, lb, ub)
        fit = func(X)
        best_idx = np.argmin(fit)
        if fit[best_idx] < rabbit_fit:
            rabbit_fit = fit[best_idx]
            rabbit = X[best_idx].copy()
        curve[t] = rabbit_fit
    return dict(best_fitness=rabbit_fit, best_position=rabbit, curve=curve)


# ======================================================================
#  Proposed: Cicada Emergence Optimizer (CEO)
# ======================================================================
_PRIMES = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]


def _next_prime_coprime(p_current, tau, used):
    """Smallest prime strictly greater than p_current that is coprime with
    the detected stagnation recurrence interval tau and not already in use
    by another brood (mirrors real cicada brood divergence to avoid
    interbreeding/period collision)."""
    import sympy  # lightweight; fallback below if unavailable
    cand = p_current
    for _ in range(200):
        cand = sympy.nextprime(cand)
        if tau <= 1 or cand % max(tau, 1) != 0:
            if cand not in used:
                return cand
    return p_current  # fallback: keep old period


def _scaled_initial_periods(max_iter, n_broods, target_emergences=7):
    """Choose brood periods proportional to the iteration budget so that the
    most active brood emerges roughly `target_emergences` times over the
    whole run (real cicada periods are long relative to a generation; here
    they must be long relative to the run, not a handful of iterations)."""
    import sympy
    base = max(int(max_iter / target_emergences), 5)
    spread = [0.55, 0.75, 1.0, 1.35, 1.7, 2.1, 2.6, 3.2][:n_broods]
    periods, used = [], set()
    for s in spread:
        cand = sympy.nextprime(max(int(base * s), 5) - 1)
        while cand in used:
            cand = sympy.nextprime(cand)
        used.add(cand)
        periods.append(cand)
    return np.array(periods, dtype=int)


def ceo(func, lb, ub, dim, pop_size, max_iter, seed=0,
        n_broods=4, stagnation_threshold=3, eta0=2.2, L0=0.25, gamma=4.0,
        target_emergences=7):
    """
    Cicada Emergence Optimizer.

    Population is partitioned into `n_broods` broods, each with its own
    integer emergence period P_b (scaled so the run contains roughly
    `target_emergences` emergences for the most active brood, then rounded
    to a real prime) and phase offset phi_b. Between emergences, agents
    perform small-step "nymph" local exploitation around the brood/global
    best with a depth-dependent shrinking step. At iteration t == phi_b
    (mod P_b), non-elite brood members undergo a synchronized Levy-flight
    mass-emergence jump of bounded amplitude (exploration) followed by
    intra-brood arithmetic-crossover "mating"; the current brood-best
    individual is preserved unchanged (a "queen" that does not emerge) so
    the brood cannot regress below its best-found niche. A brood whose best
    fitness fails to improve for `stagnation_threshold` consecutive
    emergences has its period adaptively increased to the next prime
    coprime with the empirically observed stagnation recurrence interval
    (Predator-Desynchronization Operator) and its phase re-randomized.
    """
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    X = _init_pop(pop_size, dim, lb, ub, rng)
    fit = func(X)

    # partition population into broods
    brood_id = np.array([i % n_broods for i in range(pop_size)])
    periods = _scaled_initial_periods(max_iter, n_broods, target_emergences)
    phases = rng.integers(0, periods, size=n_broods)
    brood_best_fit = np.array([fit[brood_id == b].min() for b in range(n_broods)])
    stagnation = np.zeros(n_broods, dtype=int)
    last_improve_iter = np.zeros(n_broods, dtype=int)
    last_emergence_iter = np.full(n_broods, -1)
    used_periods = set(periods.tolist())

    best_idx = np.argmin(fit)
    gbest, gbest_fit = X[best_idx].copy(), fit[best_idx]
    curve = np.zeros(max_iter)

    try:
        import sympy  # noqa: F401
        have_sympy = True
    except ImportError:
        have_sympy = False

    for t in range(max_iter):
        for b in range(n_broods):
            members = np.where(brood_id == b)[0]
            emerging = ((t - phases[b]) % periods[b] == 0)
            if not emerging:
                # --- Nymph phase: depth-dependent local exploitation ---
                depth_frac = ((t - phases[b]) % periods[b]) / periods[b]
                eta = eta0 * (0.25 + 0.75 * (1.0 - depth_frac)) * (1.0 - 0.7 * t / max_iter)
                b_best_idx = members[np.argmin(fit[members])]
                b_best = X[b_best_idx]
                r1 = rng.uniform(0, 1, size=(len(members), dim))
                r2 = rng.uniform(0, 1, size=(len(members), dim))
                X[members] = X[members] + eta * r1 * (b_best - X[members]) \
                                          + eta * r2 * (gbest - X[members])
                X[members] = _clip(X[members], lb, ub)
                fit[members] = func(X[members])
            else:
                # --- Emergence phase: synchronized mass diversification ---
                # the brood's own best ("queen") is preserved unchanged so
                # the brood cannot regress below its best-found niche
                queen = members[np.argmin(fit[members])]
                emerging_members = members[members != queen]
                amp = L0 * np.exp(-gamma * t / max_iter)
                for i in emerging_members:
                    levy = np.clip(_levy(dim, rng), -3, 3)
                    X[i] = X[queen] + amp * (ub - lb) * levy
                X[emerging_members] = _clip(X[emerging_members], lb, ub)
                fit[emerging_members] = func(X[emerging_members])
                # intra-brood mating (arithmetic crossover, replace weaker parent)
                shuffled = rng.permutation(emerging_members)
                for k in range(0, len(shuffled) - 1, 2):
                    i, j = shuffled[k], shuffled[k + 1]
                    lam = rng.uniform()
                    child = lam * X[i] + (1 - lam) * X[j]
                    child = _clip(child, lb, ub)
                    child_fit = func(child.reshape(1, -1))[0]
                    worse = i if fit[i] > fit[j] else j
                    if child_fit < fit[worse]:
                        X[worse] = child
                        fit[worse] = child_fit
                # stagnation bookkeeping + Predator-Desynchronization Operator
                new_best = fit[members].min()
                if new_best < brood_best_fit[b] - 1e-12:
                    brood_best_fit[b] = new_best
                    stagnation[b] = 0
                    tau = t - last_improve_iter[b]
                    last_improve_iter[b] = t
                else:
                    stagnation[b] += 1
                last_emergence_iter[b] = t
                if stagnation[b] >= stagnation_threshold:
                    tau = max(t - last_improve_iter[b], periods[b])
                    if have_sympy:
                        used_periods.discard(periods[b])
                        new_p = _next_prime_coprime(periods[b], tau, used_periods)
                        used_periods.add(new_p)
                        periods[b] = new_p
                    else:
                        periods[b] = periods[b] + 2  # crude fallback, kept odd-ish
                    phases[b] = rng.integers(0, periods[b])
                    stagnation[b] = 0

        best_idx = np.argmin(fit)
        if fit[best_idx] < gbest_fit:
            gbest_fit = fit[best_idx]
            gbest = X[best_idx].copy()
        curve[t] = gbest_fit
    return dict(best_fitness=gbest_fit, best_position=gbest, curve=curve)


def ceo_v2(func, lb, ub, dim, pop_size, max_iter, seed=0,
           n_broods=4, stagnation_threshold=3, eta0=2.2, L0=0.25, gamma=4.0,
           target_emergences=7, rel_eps=1e-4):
    """
    CEO-v2: fixes a root-cause bug found by post-hoc diagnostic analysis
    of v1 (see manuscript Section "Root-cause analysis and improved
    variant"). Two changes from `ceo()`, both isolated so their effect
    can be attributed:

    (F1) Stagnation is now detected via a *relative* improvement
    threshold (`rel_eps`, default 1e-4) instead of the original absolute
    1e-12 threshold. The original threshold was satisfied by essentially
    any continuous-valued Levy jump or mating outcome, so the Predator-
    Desynchronization Operator (PDO) almost never fired in v1 -- verified
    empirically: 0 triggers in a 500-iteration run on F9. This is a
    correctness fix, not a tuning change: the original intent ("detect
    when a brood stops making meaningful progress") was never actually
    implemented.

    (F2) When PDO fires, the stagnant queen itself is now reseeded
    (uniform-random reinitialization within the domain) in addition to
    the period/phase change. In v1, PDO only changed *when* the brood
    would next emerge, not *where* -- since emergence jumps center on
    the queen, a stuck queen kept re-trapping the brood regardless of
    period. Global elitism (`gbest`) is unaffected by this reseed, so no
    already-found progress is lost at the reporting level; only this
    brood's own exploratory anchor is forced to relocate.
    """
    rng = np.random.default_rng(seed)
    lb, ub = _bounds(lb, ub, dim)
    X = _init_pop(pop_size, dim, lb, ub, rng)
    fit = func(X)

    brood_id = np.array([i % n_broods for i in range(pop_size)])
    periods = _scaled_initial_periods(max_iter, n_broods, target_emergences)
    phases = rng.integers(0, periods, size=n_broods)
    brood_best_fit = np.array([fit[brood_id == b].min() for b in range(n_broods)])
    stagnation = np.zeros(n_broods, dtype=int)
    last_improve_iter = np.zeros(n_broods, dtype=int)
    used_periods = set(periods.tolist())

    best_idx = np.argmin(fit)
    gbest, gbest_fit = X[best_idx].copy(), fit[best_idx]
    curve = np.zeros(max_iter)
    pdo_triggers = 0

    for t in range(max_iter):
        for b in range(n_broods):
            members = np.where(brood_id == b)[0]
            emerging = ((t - phases[b]) % periods[b] == 0)
            if not emerging:
                depth_frac = ((t - phases[b]) % periods[b]) / periods[b]
                eta = eta0 * (0.25 + 0.75 * (1.0 - depth_frac)) * (1.0 - 0.7 * t / max_iter)
                b_best_idx = members[np.argmin(fit[members])]
                b_best = X[b_best_idx]
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
                shuffled = rng.permutation(emerging_members)
                for k in range(0, len(shuffled) - 1, 2):
                    i, j = shuffled[k], shuffled[k + 1]
                    lam = rng.uniform()
                    child = lam * X[i] + (1 - lam) * X[j]
                    child = _clip(child, lb, ub)
                    child_fit = func(child.reshape(1, -1))[0]
                    worse = i if fit[i] > fit[j] else j
                    if child_fit < fit[worse]:
                        X[worse] = child
                        fit[worse] = child_fit
                new_best = fit[members].min()
                # (F1) relative-improvement stagnation check
                rel_improve = (brood_best_fit[b] - new_best) / (abs(brood_best_fit[b]) + 1e-12)
                if rel_improve > rel_eps:
                    brood_best_fit[b] = new_best
                    stagnation[b] = 0
                    last_improve_iter[b] = t
                else:
                    stagnation[b] += 1
                if stagnation[b] >= stagnation_threshold:
                    pdo_triggers += 1
                    tau = max(t - last_improve_iter[b], periods[b])
                    used_periods.discard(periods[b])
                    new_p = _next_prime_coprime(periods[b], tau, used_periods)
                    used_periods.add(new_p)
                    periods[b] = new_p
                    phases[b] = rng.integers(0, periods[b])
                    stagnation[b] = 0
                    # (F2) reseed the stuck queen itself, not just the schedule
                    queen_new = rng.uniform(lb, ub)
                    X[queen] = queen_new
                    fit[queen] = func(queen_new.reshape(1, -1))[0]
                    brood_best_fit[b] = fit[members].min()
        best_idx = np.argmin(fit)
        if fit[best_idx] < gbest_fit:
            gbest_fit = fit[best_idx]
            gbest = X[best_idx].copy()
        curve[t] = gbest_fit
    return dict(best_fitness=gbest_fit, best_position=gbest, curve=curve,
                pdo_triggers=pdo_triggers)


ALGORITHMS = {
    "CEO": ceo,
    "PSO": pso,
    "GA": ga,
    "DE": de,
    "ACOR": acor,
    "ABC": abc,
    "GWO": gwo,
    "WOA": woa,
    "HHO": hho,
}

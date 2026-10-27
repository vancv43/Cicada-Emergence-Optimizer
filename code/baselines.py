# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""Baselines. Every runner takes (obj, lb, ub, dim, seed, params) and returns obj.best; the shared
budget.Objective raises BudgetExhausted at evaluation B+1, which is caught here."""
import os, sys, io, math, contextlib
import numpy as np
from budget import BudgetExhausted

HERE = os.path.dirname(os.path.abspath(__file__))


def _guard(fn):
    def wrapped(obj, *a, **k):
        try:
            fn(obj, *a, **k)
        except BudgetExhausted:
            pass
        return obj.best
    wrapped.__name__ = fn.__name__
    return wrapped


def _reflect(x, lb, ub):
    span = ub - lb; y = (x - lb) % (2 * span); y = np.where(y <= span, y, 2 * span - y); return lb + y


# ---------------- DE/rand/1/bin ----------------
@_guard
def run_de(obj, lb, ub, dim, seed, params=None):
    p = {'F': 0.5, 'CR': 0.9, 'N': 50, **(params or {})}
    rng = np.random.default_rng(seed); n = int(p['N']); lo = np.full(dim, lb, float); hi = np.full(dim, ub, float)
    X = rng.uniform(lo, hi, (n, dim)); fit = np.array([obj(x) for x in X])
    while True:
        for i in range(n):
            a, b, c = rng.choice(n - 1, 3, replace=False); a, b, c = [r + (r >= i) for r in (a, b, c)]
            v = _reflect(X[a] + p['F'] * (X[b] - X[c]), lo, hi)
            m = rng.random(dim) < p['CR']; m[rng.integers(dim)] = True
            u = np.where(m, v, X[i]); fu = obj(u)
            if fu <= fit[i]:
                X[i] = u; fit[i] = fu


# ---------------- PSO (global best, inertia weight) ----------------
@_guard
def run_pso(obj, lb, ub, dim, seed, params=None):
    p = {'w': 0.7298, 'c1': 1.49618, 'c2': 1.49618, 'N': 30, **(params or {})}
    rng = np.random.default_rng(seed); n = int(p['N']); lo = np.full(dim, lb, float); hi = np.full(dim, ub, float)
    vmax = 0.2 * (hi - lo)
    X = rng.uniform(lo, hi, (n, dim)); V = rng.uniform(-vmax, vmax, (n, dim))
    fit = np.array([obj(x) for x in X]); P = X.copy(); Pf = fit.copy(); g = int(np.argmin(Pf))
    while True:
        for i in range(n):
            r1 = rng.random(dim); r2 = rng.random(dim)
            V[i] = np.clip(p['w'] * V[i] + p['c1'] * r1 * (P[i] - X[i]) + p['c2'] * r2 * (P[g] - X[i]), -vmax, vmax)
            X[i] = _reflect(X[i] + V[i], lo, hi); fx = obj(X[i])
            if fx < Pf[i]:
                P[i] = X[i].copy(); Pf[i] = fx
                if fx < Pf[g]:
                    g = i


# ---------------- IPOP-CMA-ES (pycma reference implementation) ----------------
@_guard
def run_cmaes(obj, lb, ub, dim, seed, params=None):
    import cma
    p = {'sigma0': 0.3, 'popmult': 1.0, **(params or {})}
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(lb, ub, dim)
    lam = int(round(p['popmult'] * (4 + int(3 * math.log(dim)))))
    opts = {'bounds': [lb, ub], 'seed': int(rng.integers(1, 2**31 - 1)), 'verbose': -9, 'popsize': max(4, lam),
            'maxfevals': obj.max_fe, 'tolflatfitness': 10, 'CMA_stds': None}
    with contextlib.redirect_stdout(io.StringIO()):
        cma.fmin2(obj, x0, p['sigma0'] * (ub - lb), opts, restarts=50, incpopsize=2,
                  restart_from_best=False, bipop=False)
    # pycma stops by itself at maxfevals; an unused tail of the budget is not spent (counted as used).


# ---------------- L-SHADE (Tanabe & Fukunaga, CEC 2014) ----------------
def _lshade_core(obj, lb, ub, dim, seed, jso, params=None):
    """params (all optional, default = published): L-SHADE r_Ninit, H, p, arc_rate; jSO c_Ninit, H, p_max, arc_rate"""
    q = params or {}
    rng = np.random.default_rng(seed); lo = np.full(dim, lb, float); hi = np.full(dim, ub, float)
    maxfe = obj.max_fe
    if jso:
        n_init = int(round(q.get('c_Ninit', 25) * math.sqrt(dim) * math.log(dim))); H = int(q.get('H', 5)); arc_rate = q.get('arc_rate', 1.0)
        p_max = q.get('p_max', 0.25)
        MF = np.full(H, 0.3); MCR = np.full(H, 0.8)
    else:
        n_init = int(round(q.get('r_Ninit', 18) * dim)); H = int(q.get('H', 6)); arc_rate = q.get('arc_rate', 2.6)
        p_ls = q.get('p', 0.11)
        MF = np.full(H, 0.5); MCR = np.full(H, 0.5)
    n_min = 4; n = n_init
    X = rng.uniform(lo, hi, (n, dim)); fit = np.array([obj(x) for x in X])
    A = np.empty((0, dim)); k = 0
    while True:
        nfe = obj.n
        if jso:
            p_best = p_max - (p_max - p_max / 2) * nfe / maxfe
        else:
            p_best = p_ls
        r = rng.integers(0, H, n)
        if jso:
            MF[H - 1] = 0.9; MCR[H - 1] = 0.9
        # CR
        mcr = MCR[r]
        CR = np.where(np.isnan(mcr), 0.0, np.clip(rng.normal(np.nan_to_num(mcr), 0.1), 0, 1))
        if jso:
            if nfe < 0.25 * maxfe: CR = np.maximum(CR, 0.7)
            elif nfe < 0.5 * maxfe: CR = np.maximum(CR, 0.6)
        # F (Cauchy, regenerate while <= 0, truncate at 1)
        F = np.empty(n)
        for i in range(n):
            f = -1.0
            while f <= 0:
                f = MF[r[i]] + 0.1 * math.tan(math.pi * (rng.random() - 0.5))
            F[i] = min(f, 1.0)
            if jso and nfe < 0.6 * maxfe and F[i] > 0.7:
                F[i] = 0.7
        order = np.argsort(fit); npb = max(2, int(round(p_best * n)))
        PA = np.vstack([X, A]) if len(A) else X
        U = np.empty_like(X)
        for i in range(n):
            pb = order[rng.integers(npb)]
            r1 = rng.integers(n - 1); r1 += r1 >= i
            while True:
                r2 = rng.integers(len(PA))
                if r2 != i and r2 != r1:
                    break
            if jso:
                Fw = F[i] * (0.7 if nfe < 0.2 * maxfe else (0.8 if nfe < 0.4 * maxfe else 1.2))
            else:
                Fw = F[i]
            v = X[i] + Fw * (X[pb] - X[i]) + F[i] * (X[r1] - PA[r2])
            v = np.where(v < lo, (lo + X[i]) / 2, v); v = np.where(v > hi, (hi + X[i]) / 2, v)
            m = rng.random(dim) < CR[i]; m[rng.integers(dim)] = True
            U[i] = np.where(m, v, X[i])
        fu = np.empty(n)
        for i in range(n):
            fu[i] = obj(U[i])
        better = fu < fit; ok = fu <= fit
        if better.any():
            A = np.vstack([A, X[better]])
            df = np.abs(fit[better] - fu[better]); w = df / df.sum()
            SF = F[better]; SCR = CR[better]
            mf = np.sum(w * SF ** 2) / np.sum(w * SF)
            if np.isnan(MCR[k]) or SCR.max() == 0:
                mcr_new = np.nan
            else:
                mcr_new = np.sum(w * SCR ** 2) / np.sum(w * SCR)
            if jso:
                MF[k] = (mf + MF[k]) / 2
                MCR[k] = np.nan if np.isnan(mcr_new) or np.isnan(MCR[k]) else (mcr_new + MCR[k]) / 2
            else:
                MF[k] = mf; MCR[k] = mcr_new
            k = (k + 1) % (H - 1 if jso else H)
        X[ok] = U[ok]; fit[ok] = fu[ok]
        # linear population size reduction
        n_new = int(round((n_min - n_init) / maxfe * obj.n + n_init)); n_new = max(n_min, n_new)
        if n_new < n:
            keep = np.argsort(fit)[:n_new]; X = X[keep]; fit = fit[keep]; n = n_new
        amax = int(round(arc_rate * n))
        if len(A) > amax:
            A = A[rng.choice(len(A), amax, replace=False)]


@_guard
def run_lshade(obj, lb, ub, dim, seed, params=None):
    _lshade_core(obj, lb, ub, dim, seed, jso=False, params=params)


@_guard
def run_jso(obj, lb, ub, dim, seed, params=None):
    _lshade_core(obj, lb, ub, dim, seed, jso=True, params=params)


# ---------------- GWO / HHO (mealpy 3.0.3 originals) ----------------
def _mealpy(model_cls, obj, lb, ub, dim, seed, pop=30):
    from mealpy import FloatVar, Termination
    prob = {"obj_func": obj, "bounds": FloatVar(lb=[lb] * dim, ub=[ub] * dim), "minmax": "min", "log_to": None}
    model = model_cls(epoch=100000, pop_size=pop)  # validator max; 100000 epochs x 30 > any budget here
    model.solve(prob, termination=Termination(max_fe=obj.max_fe + 10**6), seed=int(seed) % (2**31 - 1))


@_guard
def run_gwo(obj, lb, ub, dim, seed, params=None):
    from mealpy.swarm_based import GWO
    _mealpy(GWO.OriginalGWO, obj, lb, ub, dim, seed)


@_guard
def run_hho(obj, lb, ub, dim, seed, params=None):
    from mealpy.swarm_based import HHO
    _mealpy(HHO.OriginalHHO, obj, lb, ub, dim, seed)


RUNNERS = {
    'DE': run_de, 'PSO': run_pso, 'IPOP-CMA-ES': run_cmaes, 'L-SHADE': run_lshade, 'jSO': run_jso,
    'GWO': run_gwo, 'HHO': run_hho,
}

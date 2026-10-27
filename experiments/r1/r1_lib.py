# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""
R1 post-review experiments (see PROTOCOL_R1.md). One module so the same code
runs on a workstation and inside a batch-run session.

A *chunk* is (exp, fn, cfg, s0, s1): one function, one configuration, seeds
s0..s1-1. It writes R1__<exp>__<fn>__<cfg>__s<s0>-<s1>.csv, so a cut-short
session keeps every finished chunk and a re-run skips chunks already present.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.dirname(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fe_budget import run_fe_matched, CountingObjective, BudgetExhausted, max_iter_for  # noqa: E402

POP = 30

# ----------------------------------------------------------- configurations
SENS = {  # E2 one-at-a-time grid around the frozen CEO
    **{f"eta0={v}": dict(eta0=v) for v in (1.1, 1.65, 2.75, 3.3)},
    **{f"L0={v}": dict(L0=v) for v in (0.1, 0.5, 1.0)},
    **{f"nb={v}": dict(n_broods=v) for v in (2, 3, 6, 8)},
    **{f"theta={v}": dict(stagnation_threshold=v) for v in (2, 5, 8)},
    **{f"E={v}": dict(target_emergences=v) for v in (10, 40)},
    **{f"gamma={v}": dict(gamma=v) for v in (2.0, 6.0)},
}
SYNC = {"sync clocks, w/o PDO": dict(sync_clocks=True, use_pdo=False)}
CORE = ["PSO", "GA", "DE", "ACOR", "ABC", "GWO", "WOA", "HHO"]
SOTA = ["L-SHADE", "LSHADE-cnEpSin", "IMODE", "INFO", "HBA", "SCSO"]
RESTART = ["BIPOP-CMA-ES", "Sawtooth-GA"]


def safe(name):
    return (name.replace(":", "_").replace("/", "").replace(" ", "_").replace(",", "")
            .replace("(", "").replace(")", "").replace("=", "-"))


def get_alg(cfg):
    """cfg -> (cost_name for the FE harness, callable)."""
    from ceo_ablation import ceo_switchable, ABLATIONS
    if cfg == "CEO":
        return "CEO", lambda *a, **k: ceo_switchable(*a, **k)
    for prefix, table in (("ABL:", None), ("SENS:", SENS)):
        if cfg.startswith(prefix):
            name = cfg[len(prefix):]
            kw = (ABLATIONS.get(name) or SYNC.get(name)) if prefix == "ABL:" else table[name]
            if kw is None:
                raise KeyError(cfg)
            return "CEO", (lambda kw: lambda f, lb, ub, d, pop_size, max_iter, seed=0:
                           ceo_switchable(f, lb, ub, d, pop_size, max_iter, seed=seed, **kw))(kw)
    if cfg in CORE:
        from algorithms import ALGORITHMS
        return cfg, ALGORITHMS[cfg]
    if cfg in SOTA:
        from sota_baselines import SOTA_ALGORITHMS
        return cfg, SOTA_ALGORITHMS[cfg]
    if cfg == "BIPOP-CMA-ES":
        return cfg, bipop_cmaes
    if cfg == "Sawtooth-GA":
        return cfg, sawtooth_ga
    raise KeyError(cfg)


# ---------------------------------------------------------- restart baselines
def bipop_cmaes(func, lb, ub, dim, pop_size, max_iter, seed=0):
    """BIPOP-CMA-ES from pycma; the counting objective stops it at the budget."""
    import cma
    lb = np.asarray(lb, float) * np.ones(dim)
    ub = np.asarray(ub, float) * np.ones(dim)
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(lb, ub)
    sigma0 = 0.3 * float(np.mean(ub - lb))
    opts = {"bounds": [lb.tolist(), ub.tolist()], "seed": seed + 1, "verbose": -9,
            "maxfevals": func.budget if hasattr(func, "budget") else 1e9,
            "tolfun": 1e-12, "tolx": 1e-12}
    f1 = lambda x: float(func(np.asarray(x).reshape(1, -1))[0])  # noqa: E731
    cma.fmin2(f1, x0, sigma0, options=opts, restarts=9, bipop=True)
    return dict(best_fitness=np.inf)


def sawtooth_ga(func, lb, ub, dim, pop_size, max_iter, seed=0, amplitude=27, period=20,
                crossover_rate=0.8, mutation_rate=0.1):
    """Saw-tooth GA after Koumousis and Katsaras (2006): the population shrinks
    linearly from n+A to n-A over each period, then is refilled with uniform
    random individuals. Operators are those of algorithms.ga. Runs until the
    counting objective raises BudgetExhausted."""
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, float) * np.ones(dim)
    ub = np.asarray(ub, float) * np.ones(dim)
    n_max, n_min = pop_size + amplitude, max(pop_size - amplitude, 2)
    X = rng.uniform(lb, ub, size=(n_max, dim))
    fit = func(X)
    t = 0
    while True:
        phase = t % period
        if phase == 0 and t > 0:  # reinitialization back to n_max
            k = n_max - len(X)
            if k > 0:
                Xn = rng.uniform(lb, ub, size=(k, dim))
                X, fit = np.vstack([X, Xn]), np.concatenate([fit, func(Xn)])
        n_t = int(round(n_max - (n_max - n_min) * phase / max(period - 1, 1)))
        n_t = max(min(n_t, len(X)), 2)
        order = np.argsort(fit, kind="stable")
        X, fit = X[order[:n_t]], fit[order[:n_t]]  # drop the worst to shrink
        best, best_fit = X[0].copy(), fit[0]
        a, b = rng.integers(0, n_t, n_t), rng.integers(0, n_t, n_t)
        parents = X[np.where(fit[a] < fit[b], a, b)]
        children = parents.copy()
        for i in range(0, n_t - 1, 2):
            if rng.uniform() < crossover_rate:
                al = rng.uniform(0, 1, dim)
                children[i] = al * parents[i] + (1 - al) * parents[i + 1]
                children[i + 1] = al * parents[i + 1] + (1 - al) * parents[i]
        mask = rng.uniform(0, 1, children.shape) < mutation_rate
        children = np.clip(children + mask * rng.normal(0, 1, children.shape) * 0.1 * (ub - lb), lb, ub)
        cfit = func(children)
        w = np.argmax(cfit)
        children[w], cfit[w] = best, best_fit  # elitism 1
        X, fit = children, cfit
        t += 1


# ------------------------------------------------------------------ problems
RW = {  # CEC2020 real-world, mechanical subset (Kumar et al. 2020) via enoppy
    "RC15": "WeightMinimizationSpeedReducerProblem",
    "RC16": "OptimalDesignIndustrialRefrigerationSystemProblem",
    "RC17": "TensionCompressionSpringDesignProblem",
    "RC18": "PressureVesselDesignProblem",
    "RC19": "WeldedBeamDesignProblem",
    "RC20": "ThreeBarTrussDesignProblem",
    "RC21": "MultipleDiskClutchBrakeDesignProblem",
}
PENALTY = 1e6


def rw_problem(code):
    from enoppy.paper_based import rwco_2020 as R
    p = getattr(R, RW[code])()
    amend = "amend_position" in type(p).__dict__

    def parts(x):
        x = np.array(x, dtype=float)
        if amend:
            x = p.amend_position(x, p.lb, p.ub)
        f = float(np.asarray(p.get_objs(x)).ravel()[0])
        v = float(np.sum(np.maximum(0.0, np.asarray(p.get_cons(x), float).ravel())))
        return f, v

    def func(X):
        X = np.atleast_2d(X)
        out = np.empty(len(X))
        for i, x in enumerate(X):
            f, v = parts(x)
            out[i] = f + PENALTY * v
        return out
    func.parts = parts
    return func, np.array(p.lb, float), np.array(p.ub, float), p.n_dims, 0.0


def problem(exp, fn):
    """-> func, lb, ub, dim, f_global, budget, suite label"""
    if exp in ("E2", "E3cl"):
        from benchmarks import BENCHMARKS
        func, lb, ub, dim, g = BENCHMARKS[fn]
        return func, lb, ub, dim, g, 15000
    if exp in ("E1", "E3c22"):
        import opfunu
        p = getattr(opfunu.cec_based.cec2022, f"F{int(fn[1:])}2022")(ndim=10)
        return (lambda X: np.array([p.evaluate(x) for x in np.atleast_2d(X)]),
                p.lb.copy(), p.ub.copy(), 10, float(p.f_global), 200000)
    if exp == "E3c17":
        import opfunu
        k = int(fn[1:])
        opf = 1 if k == 1 else k - 1  # opfunu drops the withdrawn F2
        p = getattr(opfunu.cec_based.cec2017, f"F{opf}2017")(ndim=30)
        return (lambda X: np.array([p.evaluate(x) for x in np.atleast_2d(X)]),
                p.lb.copy(), p.ub.copy(), 30, float(p.f_global), 300000)
    if exp == "E4":
        func, lb, ub, dim, g = rw_problem(fn)
        return func, lb, ub, dim, g, (200000 if dim > 10 else 100000)
    raise KeyError(exp)


class TrackingObjective(CountingObjective):
    """CountingObjective that also keeps the best point, for feasibility checks."""
    def __init__(self, func, budget):
        super().__init__(func, budget)
        self.best_x = None

    def __call__(self, X):
        X = np.atleast_2d(X)
        remaining = self.budget - self.count
        if remaining <= 0:
            raise BudgetExhausted()
        cut = X.shape[0] > remaining
        Xe = X[:remaining] if cut else X
        vals = self.func(Xe)
        self.count += Xe.shape[0]
        i = int(np.argmin(vals))
        if vals[i] < self.best:
            self.best, self.best_x = float(vals[i]), Xe[i].copy()
        self._record()
        if cut:
            raise BudgetExhausted()
        return vals


def run_chunk(exp, fn, cfg, s0, s1, out_dir, cal=None):
    name = f"R1__{exp}__{fn}__{safe(cfg)}__s{s0:02d}-{s1:02d}.csv"
    path = os.path.join(out_dir, name)
    if os.path.exists(path):
        return path, 0.0
    func, lb, ub, dim, gmin, budget = problem(exp, fn)
    cost_name, afunc = get_alg(cfg)
    scale = (cal or {}).get(f"{exp}:{fn}:{cfg}", (cal or {}).get(f"{exp}:{cfg}", 1.0))
    rows, t0 = [], time.time()
    for seed in range(s0, s1):
        row = dict(exp=exp, function=fn, algorithm=cfg, seed=seed)
        if exp == "E4":
            obj = TrackingObjective(func, budget)
            mi = max(int(max_iter_for(cost_name, budget, POP) * scale), 10)
            try:
                afunc(obj, lb, ub, dim, pop_size=POP, max_iter=mi, seed=seed)
            except BudgetExhausted:
                pass
            best, _, used = obj.finalize()
            f, v = func.parts(obj.best_x)
            row.update(best_fitness=best, error=best, objective=f, violation=v,
                       feasible=int(v <= 0), fe_used=used, max_iter=mi)
        else:
            r = run_fe_matched(cost_name, afunc, func, lb, ub, dim, POP, budget, seed,
                               _epoch_scale=scale)
            row.update(best_fitness=r["best_fitness"], error=r["best_fitness"] - gmin,
                       fe_used=r["fe_used"], max_iter=r["max_iter"], global_min=gmin)
        rows.append(row)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    return path, time.time() - t0


# ------------------------------------------------------------------ worklist
def all_chunks():
    from ceo_ablation import ABLATIONS
    cl = [f"F{i}" for i in range(1, 24)]
    c22 = [f"C{i:02d}" for i in range(1, 13)]
    c17 = ["C01"] + [f"C{i:02d}" for i in range(3, 31)]
    abl = ["CEO"] + ["ABL:" + k for k in ABLATIONS if k != "CEO (full)"] + ["ABL:" + k for k in SYNC]
    ch = []
    ch += [("E1", f, c) for f in c22 for c in abl]
    ch += [("E2", f, c) for f in cl for c in ["CEO"] + ["SENS:" + k for k in SENS]]
    ch += [("E3cl", f, c) for f in cl for c in RESTART]
    ch += [("E3c17", f, c) for f in c17 for c in RESTART]
    ch += [("E3c22", f, c) for f in c22 for c in RESTART]
    ch += [("E4", f, c) for f in RW for c in ["CEO"] + CORE + SOTA + RESTART]
    return ch

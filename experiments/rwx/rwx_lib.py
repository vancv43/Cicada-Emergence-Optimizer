# /*
#  * Cicada Emergence Optimizer: A Hybrid Success-History Differential Evolution with Staggered Brood Restarts
#  * Authors: Van Van Chau (1,2), Minh Tuan Nguyen (1,*), and Phayung Meesad (2,*)
#  * (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#  *     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam
#  *     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
#  * (2) Department of Information Technology Management, Faculty of Information Technology and Digital Innovation,
#  *     King Mongkut's University of Technology North Bangkok (KMUTNB),
#  *     1518 Pracharat 1 Rd, Wongsawang, Bangsue, Bangkok 10800, Thailand
#  *     phayung.m@itd.kmutnb.ac.th
#  * (*) Corresponding authors
#  * Repository: https://github.com/vancv43/Cicada-Emergence-Optimizer (MIT License)
#  */
"""CEC2020-RW extension RC22-RC33 (PROTOCOL_rwx.md). One unit = (algorithm, problem, seeds s0..s1-1)
and writes RWX__<alg>__<code>__sAA-BB.csv in the column layout of results/v4-rw.csv.

Three harnesses, each identical to the one that produced the RC15-RC21 results:
  CEO            -> CEO_v4/run_rw_v4.py   (ceo_v4.RUNNERS['CEO-v4'], tuned_v4.json, budget.Objective)
  19 swarm       -> CEO_v4/run_rw_swarm.py (swarm_orig.RUNNERS, budget.Objective)
  R1 set (10)    -> experiments/r1/r1_lib.run_chunk, exp E4 (TrackingObjective, max_iter_for, calibration)
Only the problem source changes: rw_official (official CEC2020 C++ code) instead of enoppy.
"""
import json, os, sys, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
# repository layout: code/ holds the algorithms and harness, experiments/ and experiments/r1/ the shared modules
REPO = os.path.dirname(os.path.dirname(HERE))
CODE = os.path.join(REPO, "code")
for _p in (CODE, os.path.join(REPO, "experiments"), os.path.join(REPO, "experiments", "r1"),
           os.path.join(REPO, "experiments", "rwx")):
    if _p not in sys.path:
        sys.path.append(_p)
from rw_official import rw_official  # noqa: E402

CODES = [f"RC{k}" for k in range(22, 34)]
SWARM = ['PSO', 'ABC', 'GWO', 'WOA', 'HHO', 'FA', 'BA', 'CS', 'DA', 'MFO', 'ALO', 'GOA', 'MPA', 'KH', 'AO',
         'ACO', 'BFO', 'SMO', 'SquirrelSA']                       # swarm_orig.VERIFIED + PHASE_B
R1SET = ['GA', 'DE', 'L-SHADE', 'LSHADE-cnEpSin', 'IMODE', 'INFO', 'HBA', 'SCSO', 'BIPOP-CMA-ES', 'Sawtooth-GA']
ALGS = ['CEO'] + SWARM + R1SET
SOTA = ['L-SHADE', 'LSHADE-cnEpSin', 'IMODE', 'INFO', 'HBA', 'SCSO']  # the only calibrated ones (as in R1 E4)


def problem(code):
    func, lb, ub, dim, g = rw_official(code)
    budget = 200000 if dim > 10 else 100000
    if os.environ.get("RWX_DRY_BUDGET"):          # dry runs only; results of a dry run are never kept
        budget = int(os.environ["RWX_DRY_BUDGET"])
    return func, lb, ub, dim, budget


def _seed_globals(seed):
    """mealpy's L-SHADE and LSHADE-cnEpSin draw F from scipy.stats.cauchy.rvs, which uses the global numpy
    state that mealpy's own seed does not set; seeding it here makes every run reproducible per seed."""
    import random
    random.seed(int(seed)); np.random.seed(int(seed) % (2 ** 32))


def name(alg, code, s0, s1):
    return f"RWX__{alg.replace(' ', '_')}__{code}__s{s0:02d}-{s1:02d}.csv"


def calibration(code, cfg, cache_dir):
    """Same throw-away rule as experiments/r1/calibrate_r1.py: one seed-0 run, epoch scale = budget/used,
    capped at 20. Deterministic, so every shard gets the same value. Not part of any reported result."""
    if cfg not in SOTA:
        return 1.0
    path = os.path.join(cache_dir, f"cal__{code}__{cfg}.json")
    if os.path.exists(path):
        return json.load(open(path))["scale"]
    import r1_lib as L
    from fe_budget import run_fe_matched
    func, lb, ub, dim, budget = problem(code)
    cost, alg = L.get_alg(cfg)
    _seed_globals(0)
    r = run_fe_matched(cost, alg, func, lb, ub, dim, L.POP, budget, seed=0)
    scale = round(min(budget / max(r["fe_used"], 1), 20.0), 4)
    os.makedirs(cache_dir, exist_ok=True)
    json.dump(dict(code=code, cfg=cfg, fe_used=int(r["fe_used"]), scale=scale), open(path + ".tmp", "w"))
    os.replace(path + ".tmp", path)
    return scale


def one_run(alg, code, seed, cache_dir):
    func, lb, ub, dim, budget = problem(code)
    row = dict(exp="RWX", function=code, algorithm=alg, seed=seed)
    if alg == 'CEO' or alg in SWARM:
        from budget import Objective
        obj = Objective(lambda x: func(x)[0], dim, budget, 0.0)
        if alg == 'CEO':
            from ceo_v4 import RUNNERS
            RUNNERS["CEO-v4"](obj, lb, ub, dim, seed, json.load(open(os.path.join(CODE, "tuned_v4.json")))["CEO-v4"])
        else:
            from swarm_orig import RUNNERS
            RUNNERS[alg](obj, lb, ub, dim, seed)
        best, used, bx = obj.best, obj.n, obj.best_x
        mi = np.nan
    else:
        import r1_lib as L
        from fe_budget import BudgetExhausted, max_iter_for
        cost_name, afunc = L.get_alg(alg)
        scale = calibration(code, alg, cache_dir)
        obj = L.TrackingObjective(func, budget)
        mi = max(int(max_iter_for(cost_name, budget, L.POP) * scale), 10)
        _seed_globals(seed)
        try:
            afunc(obj, lb, ub, dim, pop_size=L.POP, max_iter=mi, seed=seed)
        except BudgetExhausted:
            pass
        best, _, used = obj.finalize()
        bx = obj.best_x
    f, v = func.parts(bx)
    row.update(best_fitness=best, error=best, fe_used=used, objective=f, violation=v, feasible=int(v <= 0),
               max_iter=mi, best_x=json.dumps([float(t) for t in np.asarray(bx).ravel()]))
    return row


def run_unit(alg, code, s0, s1, out_dir):
    path = os.path.join(out_dir, name(alg, code, s0, s1))
    if os.path.exists(path):
        return path, 0.0
    t0 = time.time()
    rows = [one_run(alg, code, s, os.path.join(out_dir, "calibration")) for s in range(s0, s1)]
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    return path, time.time() - t0

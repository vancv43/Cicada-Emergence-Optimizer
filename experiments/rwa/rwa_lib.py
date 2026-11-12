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
"""Re-run of the audited evolutionary opponents on RC15-RC33 (CEO_v4/RW_EVO_AUDIT.md, PROTOCOL_rwx.md
Deviation D1). One unit = (algorithm, problem, seeds s0..s1-1) -> RWA__<alg>__<code>__sAA-BB.csv, same
columns as the RWX files. Harness = the one of CEO and the swarm methods: budget.Objective, exact budget.
Problem sources unchanged: enoppy 0.1.1 for RC15-RC21 (r1_lib.rw_problem), official C++ for RC22-RC33."""
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

CODES = [f"RC{k}" for k in range(15, 34)]
PREFIX = os.environ.get("RWA_PREFIX", "RWA")   # "RWA" = D1 run, "RWB" = D2 extra round (corrected INFO, HBA, SCSO)
ALGS = ['L-SHADE', 'jSO', 'DE', 'LSHADE-cnEpSin', 'IMODE', 'INFO', 'HBA', 'SCSO']


def problem(code):
    if int(code[2:]) <= 21:
        from r1_lib import rw_problem
        func, lb, ub, dim, _ = rw_problem(code)
    else:
        from rw_official import rw_official
        func, lb, ub, dim, _ = rw_official(code)
    budget = 200000 if dim > 10 else 100000
    if os.environ.get("RWA_DRY_BUDGET"):          # dry runs only; never kept
        budget = int(os.environ["RWA_DRY_BUDGET"])
    return func, lb, ub, dim, budget


def runner(alg):
    if alg in ('L-SHADE', 'jSO', 'DE'):
        import baselines as B
        params = json.load(open(os.path.join(CODE, "tuned_configs.json")))["DE"] if alg == 'DE' else None
        f = B.RUNNERS[alg]
        return lambda obj, lb, ub, dim, seed: f(obj, lb, ub, dim, seed, params)
    import rw_evo_audited as R
    return R.RUNNERS[alg]


def name(alg, code, s0, s1):
    return f"{PREFIX}__{alg}__{code}__s{s0:02d}-{s1:02d}.csv"


def one_run(alg, code, seed):
    from budget import Objective
    func, lb, ub, dim, budget = problem(code)
    obj = Objective(lambda x: func(x)[0], dim, budget, 0.0)
    runner(alg)(obj, lb, ub, dim, seed)
    bx = obj.best_x
    f, v = func.parts(bx)
    return dict(exp=PREFIX, function=code, algorithm=alg, seed=seed, best_fitness=obj.best, error=obj.best,
                fe_used=obj.n, objective=f, violation=v, feasible=int(v <= 0), max_iter=np.nan,
                best_x=json.dumps([float(t) for t in np.asarray(bx).ravel()]))


def run_unit(alg, code, s0, s1, out_dir):
    path = os.path.join(out_dir, name(alg, code, s0, s1))
    if os.path.exists(path):
        return path, 0.0
    t0 = time.time()
    rows = [one_run(alg, code, s) for s in range(s0, s1)]
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)
    return path, time.time() - t0

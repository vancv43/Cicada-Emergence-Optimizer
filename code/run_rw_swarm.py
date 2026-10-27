# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""Swarm baselines (swarm_orig.py) on CEC2020-RW RC15-RC21 (PROTOCOL_swarm.md) through the same wrapper as run_rw_v4.py:
r1_lib.rw_problem, static penalty, seeds 0-29, R1 budgets. Resumable. Writes results/swarm-rw.csv."""
import os, sys
import numpy as np, pandas as pd
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
R1 = os.path.join(os.path.dirname(HERE), "experiments", "r1")
CODES = ["RC15", "RC16", "RC17", "RC18", "RC19", "RC20", "RC21"]
OUT = os.path.join(HERE, "results", "swarm-rw.csv")


def job(args):
    alg, code, seed = args
    sys.path.insert(0, R1); sys.path.insert(0, os.path.dirname(R1))
    import warnings; warnings.filterwarnings("ignore")
    import r1_lib as L
    from budget import Objective
    from swarm_orig import RUNNERS
    func, lb, ub, dim, g, budget = L.problem("E4", code)[:6]
    obj = Objective(lambda x: func(x)[0], dim, budget, 0.0)
    RUNNERS[alg](obj, lb, ub, dim, seed)
    f, v = func.parts(obj.best_x)
    return dict(exp="E4", function=code, algorithm=alg, seed=seed, best_fitness=obj.best, error=obj.best,
                fe_used=obj.n, objective=f, violation=v, feasible=int(v <= 0))


if __name__ == "__main__":
    import swarm_orig
    ORDER = swarm_orig.VERIFIED if os.environ.get('SWARM_PHASE', 'A') == 'A' else swarm_orig.PHASE_B
    done = set()
    if os.path.exists(OUT):
        d = pd.read_csv(OUT); done = set(zip(d.algorithm, d.function, d.seed))
    jobs = [(a, c, s) for a in ORDER for c in CODES for s in range(30) if (a, c, s) not in done]
    print(len(jobs), "jobs", flush=True)
    with Pool(int(os.environ.get("PROCS", 3))) as pool:
        for k, r in enumerate(pool.imap_unordered(job, jobs), 1):
            pd.DataFrame([r]).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
            if k % 50 == 0:
                print(k, r["algorithm"], r["function"], flush=True)
    print("RW_SWARM_DONE", flush=True)

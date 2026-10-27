# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""CEO-v4 on CEC2020-RW RC15-RC21 through the R1 harness (PROTOCOL_CEO_v4.md): r1_lib.rw_problem,
static penalty, seeds 0-29, R1 budgets. Writes results/v4-rw.csv in the R1 E4 column layout."""
import os, sys, json
import numpy as np, pandas as pd
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
R1 = os.path.join(os.path.dirname(HERE), "experiments", "r1")
CODES = ["RC15", "RC16", "RC17", "RC18", "RC19", "RC20", "RC21"]


def job(args):
    code, seed = args
    sys.path.insert(0, R1); sys.path.insert(0, os.path.dirname(R1))
    import r1_lib as L
    from budget import Objective
    from ceo_v4 import RUNNERS
    func, lb, ub, dim, g, budget = L.problem("E4", code)[:6]
    obj = Objective(lambda x: func(x)[0], dim, budget, 0.0)
    RUNNERS["CEO-v4"](obj, lb, ub, dim, seed, json.load(open(os.path.join(HERE, "tuned_v4.json")))["CEO-v4"])
    f, v = func.parts(obj.best_x)
    return dict(exp="E4", function=code, algorithm="CEO-v4", seed=seed, best_fitness=obj.best, error=obj.best,
                fe_used=obj.n, objective=f, violation=v, feasible=int(v <= 0))


if __name__ == "__main__":
    with Pool(4) as pool:
        rows = pool.map(job, [(c, s) for c in CODES for s in range(30)])
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "results", "v4-rw.csv"), index=False)
    print("RW_DONE", len(rows))

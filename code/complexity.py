# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""Algorithm complexity following the CEC2022 technical report: T0 = time of the reference loop,
T1 = time of 200,000 evaluations of F1, T2 = mean time of 5 complete runs of the algorithm on F1 with
200,000 evaluations; reported (T2 - T1) / T0 for D = 10 and 20. One process, run sequentially on an
otherwise idle machine. Writes results/complexity.csv."""
import os, sys, time, json, math
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import cec2022_fast as C
from budget import Objective
from ceo_v4 import RUNNERS as CR
from swarm_orig import RUNNERS as SR, ORDER

B = 200_000
P = json.load(open(os.path.join(HERE, 'tuned_v4.json')))['CEO-v4']


def t0():
    t = time.perf_counter(); x = 0.0
    for i in range(1, 1_000_001):
        x = 0.55 + i; x = x + x; x = x / 2; x = x * x; x = math.sqrt(x); x = math.log(x); x = math.exp(x); x = x / (x + 2)
    return time.perf_counter() - t


def t1(dim):
    f = C.CEC2022Function(1, dim); X = np.random.default_rng(0).uniform(-100, 100, (B, dim))
    t = time.perf_counter()
    for x in X:
        f(x)
    return time.perf_counter() - t


def t2(name, dim):
    ts = []
    for s in range(5):
        f = C.CEC2022Function(1, dim); obj = Objective(f, dim, B, f.optimum); t = time.perf_counter()
        if name == 'CEO':
            CR['CEO-v4'](obj, -100., 100., dim, s, P)
        else:
            SR[name](obj, -100., 100., dim, s)
        ts.append(time.perf_counter() - t); assert obj.n == B
    return float(np.mean(ts))


if __name__ == '__main__':
    T0 = t0(); rows = []
    print(f'T0 = {T0:.3f} s', flush=True)
    for dim in (10, 20):
        T1 = t1(dim)
        for name in ['CEO'] + ORDER:
            T2 = t2(name, dim)
            rows.append(dict(algorithm=name, D=dim, T0=T0, T1=T1, T2=T2, ratio=(T2 - T1) / T0))
            print(f'D{dim} {name:11s} T1={T1:.2f} T2={T2:.2f} (T2-T1)/T0={(T2 - T1) / T0:.1f}', flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(HERE, 'results', 'complexity.csv'), index=False)
    print('COMPLEXITY_DONE', flush=True)

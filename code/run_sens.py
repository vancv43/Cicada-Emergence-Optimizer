# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""CEO one-at-a-time parameter sensitivity, PROTOCOL_sensitivity.md. Official CEC2017 D10, 30 runs.
One invocation = one shard. Resumable; hard stop. Output: <outdir>/sens-test_s<shard>.csv"""
import os, sys, json, time, argparse, warnings, queue
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np, pandas as pd
from multiprocessing import Pool
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'cec2017_official'))
from run_power import seed_of, FUNCS, RUNS

DIM, BUDGET = 10, 100_000
FROZEN = json.load(open(os.path.join(HERE, 'tuned_v4.json')))['CEO-v4']
LEVELS = {'N0f': [8, 11, 15, 20, 25], 'n_b': [2, 3, 5, 6, 8], 'E': [10, 40, FROZEN['E'], 120, 200],
          'theta': [1, 2, 3, 4, 5], 'c_micro': [0.2, FROZEN['c_micro'], 0.5, 1.0, 3.0],
          'p': [0.05, 0.10, 0.15, FROZEN['p'], 0.25]}


def configs():
    out = [('frozen', 'frozen', dict(FROZEN))]
    for k, vals in LEVELS.items():
        for v in vals:
            if v == FROZEN[k]:
                continue
            out.append((k, f'{k}={v:g}', {**FROZEN, k: v}))
    return out


def all_jobs():
    return [(c[1], f, r) for c in configs() for f in FUNCS['cec2017'] for r in range(1, RUNS + 1)]


def shard_jobs(shard, nshards):
    jobs = sorted(all_jobs(), key=lambda j: (j[1], j[0], j[2]))
    return jobs[shard::nshards]


def job(j):
    name, fn, run = j
    from budget import Objective
    from cec2017_official import CEC2017Function
    from ceo_v4 import RUNNERS
    cfg = {c[1]: c for c in configs()}[name]
    f = CEC2017Function(fn, DIM); obj = Objective(f, DIM, BUDGET, f.optimum)
    seed = seed_of('cec2017', DIM, fn, run); t = time.perf_counter()
    RUNNERS['CEO-v4'](obj, -100., 100., DIM, seed, cfg[2])
    return dict(param=cfg[0], config=name, function=f'F{fn}', run=run, seed=seed, error=obj.error(), fe=obj.n,
                sec=round(time.perf_counter() - t, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--procs', type=int, default=4); ap.add_argument('--outdir', default='results')
    ap.add_argument('--hard-stop-h', type=float, default=8.6)
    a = ap.parse_args()
    outdir = os.path.abspath(a.outdir); os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f'sens-test_s{a.shard}.csv')
    done = set()
    if os.path.exists(out):
        d = pd.read_csv(out); done = set(zip(d.config, d.function, d.run))
    jobs = [j for j in shard_jobs(a.shard, a.nshards) if (j[0], f'F{j[1]}', j[2]) not in done]
    print(f'shard {a.shard}/{a.nshards}: {len(jobs)} jobs', flush=True)
    t0 = time.time(); stop = t0 + a.hard_stop_h * 3600; n = 0
    pool = Pool(a.procs); q = queue.Queue(); it = iter(jobs); inflight = 0

    def submit():
        nonlocal inflight
        if time.time() < stop:
            j = next(it, None)
            if j is not None:
                pool.apply_async(job, (j,), callback=q.put, error_callback=q.put); inflight += 1
    for _ in range(a.procs):
        submit()
    while inflight:
        r = q.get(); inflight -= 1
        if isinstance(r, BaseException):
            raise r
        n += 1
        pd.DataFrame([r]).to_csv(out, mode='a', header=not os.path.exists(out), index=False)
        if n % 100 == 0:
            print(f'[{n}/{len(jobs)}] {(time.time()-t0)/3600:.2f} h', flush=True)
        submit()
    pool.close(); pool.join()
    left = len(jobs) - n
    print('SHARD_DONE' if left == 0 else f'SHARD_STOPPED_EARLY {left} jobs left', flush=True)


if __name__ == '__main__':
    main()

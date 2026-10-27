# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""Additional benchmarks, PROTOCOL_power.md: CEO, MPA, CS, ABC on official CEC2017 D10/D30 and
MPA, CS, ABC on official CEC2022 D20 (CEO D20 already exists). One invocation = one shard of the
(suite, dim, alg, fn, run) grid. Resumable; stops launching jobs after --hard-stop-h.
Output: <outdir>/power-test_s<shard>.csv"""
import os, sys, json, time, argparse, warnings
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np, pandas as pd
from functools import partial
from multiprocessing import Pool
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'cec2017_official'))

RUNS = 30
OPP = ['MPA', 'CS', 'ABC']
ABL30 = ['v4-noEmergence', 'v4-noGate', 'v4-noMacro', 'v4-sharedMem', 'v4-K1', 'v4-equalPeriods', 'v4-gaussMoult', 'v4-restart']
GRID = [('cec2017', 10, ['CEO'] + OPP), ('cec2017', 30, ['CEO'] + OPP), ('cec2022', 20, OPP)]
FUNCS = {'cec2017': [1] + list(range(3, 31)), 'cec2022': list(range(1, 13))}
BUDGET = {('cec2017', 10): 100_000, ('cec2017', 30): 300_000, ('cec2022', 20): 1_000_000, ('cec2017', 50): 500_000}
# supplementary grid added 2026-10-21 (PROTOCOL_power.md, Deviations); the default grid above is unchanged
GRIDS = {'main': (GRID, 'power-test'), 'd50': ([('cec2017', 50, ['CEO'] + OPP)], 'power50-test'),
         # same D50 jobs and seeds, also saving the error at the 16 CEC recording points for convergence figures
         'd50trace': ([('cec2017', 50, ['CEO'] + OPP)], 'power50trace-test'),
         # reviewer round 1 (Deviations 2026-10-22): the success-history parents on the held-out suite,
         # the emergence ablation and replacement controls at D=30, and CEO with a fixed initial population
         'rev1': ([('cec2017', d, ['L-SHADE', 'jSO']) for d in (10, 30, 50)]
                  + [('cec2017', 30, ABL30), ('cec2017', 30, ['CEO-N0fixed']), ('cec2017', 50, ['CEO-N0fixed'])], 'rev1-test')}
# relative cost per evaluation (us) for shard balancing only
ACOST = {'CEO': 20, 'MPA': 20, 'CS': 25, 'ABC': 20, 'L-SHADE': 30, 'jSO': 30, **{a: 20 for a in ABL30 + ['CEO-N0fixed']}}


def seed_of(suite, dim, fn, run):
    if suite == 'cec2022':  # identical to run_test_v4.cec_seed (official rule)
        seeds = np.loadtxt(os.path.join(HERE, 'cec2022_data', 'input_data', 'Rand_Seeds.txt'))
        return int(seeds[((dim // 10 * fn * RUNS + run) - RUNS) % 1000])
    return 1_000_000 + 10_000 * dim + 100 * fn + run


def all_jobs(grid=GRID):
    return [(s, d, a, f, r) for s, d, algs in grid for a in algs for f in FUNCS[s] for r in range(1, RUNS + 1)]


def cost(j):
    s, d, a, f, r = j
    return BUDGET[(s, d)] * (ACOST[a] / 20 * (10 if s == 'cec2017' else 150))   # measured us/FE


def shard_jobs(shard, nshards, grid=GRID):
    jobs = sorted(all_jobs(grid), key=lambda j: (-cost(j), j))
    loads = [0.0] * nshards; buckets = [[] for _ in range(nshards)]
    for j in jobs:
        k = int(np.argmin(loads)); buckets[k].append(j); loads[k] += cost(j)
    return buckets[shard]


def job(j, trace=False):
    suite, dim, alg, fn, run = j
    from budget import Objective
    if suite == 'cec2017':
        from cec2017_official import CEC2017Function
        f = CEC2017Function(fn, dim)
    else:
        import cec2022_fast as C
        f = C.CEC2022Function(fn, dim)
    obj = Objective(f, dim, BUDGET[(suite, dim)], f.optimum)
    seed = seed_of(suite, dim, fn, run)
    t = time.perf_counter()
    if alg == 'CEO' or alg.startswith('v4-') or alg == 'CEO-N0fixed':
        from ceo_v4 import RUNNERS
        RUNNERS['CEO-v4' if alg == 'CEO' else alg](obj, -100., 100., dim, seed, json.load(open(os.path.join(HERE, 'tuned_v4.json')))['CEO-v4'])
    elif alg in ('L-SHADE', 'jSO'):
        from baselines import RUNNERS
        RUNNERS[alg](obj, -100., 100., dim, seed)
    else:
        from swarm_orig import RUNNERS
        RUNNERS[alg](obj, -100., 100., dim, seed)
    r = dict(suite=suite, dim=dim, algorithm=alg, function=f'F{fn}', run=run, seed=seed, error=obj.error(),
             fe=obj.n, sec=round(time.perf_counter() - t, 2))
    if trace:
        r['trace'] = json.dumps([float(e) for e in obj.error_trace()])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--procs', type=int, default=4); ap.add_argument('--outdir', default='results')
    ap.add_argument('--hard-stop-h', type=float, default=8.6)
    ap.add_argument('--grid', choices=sorted(GRIDS), default='main')
    a = ap.parse_args()
    grid, stem = GRIDS[a.grid]
    outdir = os.path.abspath(a.outdir); os.makedirs(outdir, exist_ok=True)   # workers chdir for CEC2017
    out = os.path.join(outdir, f'{stem}_s{a.shard}.csv')
    done = set()
    if os.path.exists(out):
        d = pd.read_csv(out); done = set(zip(d.suite, d.dim, d.algorithm, d.function, d.run))
    jobs = [j for j in shard_jobs(a.shard, a.nshards, grid) if (j[0], j[1], j[2], f'F{j[3]}', j[4]) not in done]
    print(f'shard {a.shard}/{a.nshards}: {len(jobs)} jobs', flush=True)
    t0 = time.time(); stop = t0 + a.hard_stop_h * 3600; n = 0
    import queue
    run_job = partial(job, trace=a.grid.endswith('trace'))
    pool = Pool(a.procs); q = queue.Queue(); it = iter(jobs); inflight = 0; launched = 0

    def submit():
        nonlocal inflight, launched
        if time.time() < stop:
            j = next(it, None)
            if j is not None:
                pool.apply_async(run_job, (j,), callback=q.put, error_callback=q.put); inflight += 1; launched += 1
    for _ in range(a.procs):
        submit()
    while inflight:
        r = q.get(); inflight -= 1
        if isinstance(r, BaseException):
            raise r
        n += 1
        pd.DataFrame([r]).to_csv(out, mode='a', header=not os.path.exists(out), index=False)
        print(f'[{n}/{len(jobs)}] {r["suite"]} D{r["dim"]} {r["algorithm"]} {r["function"]} run{r["run"]} '
              f'err={r["error"]:.4g} {r["sec"]}s | {(time.time()-t0)/3600:.2f} h', flush=True)
        submit()
    pool.close(); pool.join()
    left = len(jobs) - n
    print('SHARD_DONE' if left == 0 else f'SHARD_STOPPED_EARLY {left} jobs left', flush=True)

if __name__ == '__main__':
    main()

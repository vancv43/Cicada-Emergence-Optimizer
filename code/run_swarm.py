# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""Swarm-intelligence baselines (swarm_orig.py) on official CEC2022 (PROTOCOL_swarm20.md), harness of run_test_v4.py. Same harness as the earlier baseline study:
official seed rule, one budget wrapper, 30 runs. One invocation = one shard of the (alg, fn, run) grid.
Resumable; stops launching jobs after --hard-stop-h. Output: <outdir>/swarm-test_D<dim>_s<shard>.csv"""
import os, sys, json, time, argparse, warnings
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np, pandas as pd
from multiprocessing import Pool
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

# PSO is the shared-harness tuned PSO, already run with the same seeds; the other 19 are run here
_PH = os.environ.get('SWARM_PHASE', 'A')
ALGS = {10: [a for a in (__import__('swarm_orig').VERIFIED if _PH == 'A' else __import__('swarm_orig').PHASE_B) if a != 'PSO']}
ALGS[20] = ALGS[10]
# relative cost per FE: CEC2022 evaluation (us, D10) + library overhead (us), for shard balancing only
FCOST = {1: 22, 2: 24, 3: 30, 4: 39, 5: 30, 6: 40, 7: 114, 8: 141, 9: 87, 10: 92, 11: 139, 12: 143}
ACOST = {'ACO': 250, 'ALO': 1500, 'KH': 75, 'GWO': 60, 'AO': 60, 'DA': 55, 'CS': 55, 'WOA': 50}
RUNS = 30
BUDGET = {10: 200_000, 20: 1_000_000}


def cec_seed(dim, fn, run):  # identical to the earlier baseline study
    seeds = np.loadtxt(os.path.join(HERE, 'cec2022_data', 'input_data', 'Rand_Seeds.txt'))
    idx = (dim // 10 * fn * RUNS + run) - RUNS
    return int(seeds[idx % 1000])


def shard_jobs(dim, shard, nshards):
    jobs = sorted([(a, fn, r) for fn in range(1, 13) for a in ALGS[dim] for r in range(1, RUNS + 1)],
                  key=lambda j: (-(FCOST[j[1]] + ACOST.get(j[0], 40)), j))
    loads = [0.0] * nshards; buckets = [[] for _ in range(nshards)]
    for j in jobs:
        k = int(np.argmin(loads)); buckets[k].append(j); loads[k] += FCOST[j[1]] + ACOST.get(j[0], 40)
    return buckets[shard]


def job(args):
    alg, fn, run, dim, params, budget = args
    import cec2022_fast as C
    from budget import Objective
    from swarm_orig import RUNNERS
    f = C.CEC2022Function(fn, dim)
    obj = Objective(f, dim, budget, f.optimum)
    seed = cec_seed(dim, fn, run)
    t = time.perf_counter(); RUNNERS[alg](obj, -100., 100., dim, seed)
    return dict(algorithm=alg, function=f'F{fn}', dim=dim, run=run, seed=seed, error=obj.error(), fe=obj.n,
                sec=round(time.perf_counter() - t, 2), trace=json.dumps([float(f'{v:.6g}') for v in obj.error_trace()]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, required=True); ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1); ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--outdir', default='results'); ap.add_argument('--prior', default='')
    ap.add_argument('--hard-stop-h', type=float, default=8.8)
    a = ap.parse_args()
    tuned = None
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, f'swarm-test_D{a.dim}_s{a.shard}.csv')
    done = set()
    for d_ in [a.outdir] + [p for p in a.prior.split(',') if p]:
        for root, _, files in os.walk(d_):
            for fn_ in files:
                if fn_.startswith(f'swarm-test_D{a.dim}_') and fn_.endswith('.csv'):
                    d = pd.read_csv(os.path.join(root, fn_)); done |= set(zip(d.algorithm, d.function, d.run))
    jobs = [j for j in shard_jobs(a.dim, a.shard, a.nshards) if (j[0], f'F{j[1]}', j[2]) not in done]
    print(f'D{a.dim} shard {a.shard}/{a.nshards}: {len(jobs)} jobs', flush=True)
    t0 = time.time(); stop = t0 + a.hard_stop_h * 3600
    import queue
    pool = Pool(a.procs); q = queue.Queue(); it = iter(jobs); n = 0; inflight = 0

    def submit():
        nonlocal inflight
        if time.time() < stop:
            j = next(it, None)
            if j is not None:
                pool.apply_async(job, ((*j, a.dim, tuned, BUDGET[a.dim]),), callback=q.put,
                                 error_callback=lambda e: q.put(e)); inflight += 1
    for _ in range(a.procs):
        submit()
    while inflight:
        r = q.get(); inflight -= 1
        if isinstance(r, BaseException):
            raise r
        n += 1
        pd.DataFrame([r]).to_csv(out, mode='a', header=not os.path.exists(out), index=False)
        print(f'[{n}/{len(jobs)}] {r["algorithm"]} {r["function"]} run{r["run"]} err={r["error"]:.4g} {r["sec"]}s '
              f'| {(time.time()-t0)/3600:.2f} h', flush=True)
        submit()
    pool.close(); pool.join()
    left = len(jobs) - n
    print('SHARD_DONE' if left == 0 else f'SHARD_STOPPED_EARLY {left} jobs left', flush=True)


if __name__ == '__main__':
    main()

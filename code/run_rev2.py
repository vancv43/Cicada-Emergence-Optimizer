# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""Study 2 of the reviewer round, PROTOCOL_rev2.md: equal-effort tuning of CEO, L-SHADE and jSO on CEC2017 at D = 30,
then a test on BBOB (cocoex) at D = 20 and 40. One invocation = one shard of one stage. Resumable; stops launching jobs
after --hard-stop-h. Output: <outdir>/rev2-tune_s<shard>.csv or <outdir>/rev2-test_s<shard>.csv"""
import os, sys, json, time, argparse, warnings, queue
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np, pandas as pd
from multiprocessing import Pool
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'cec2017_official'))

ALGS = ['CEO', 'L-SHADE', 'jSO']
NCONF, TUNE_DIM, TUNE_SEEDS = 24, 30, (1, 2, 3)
FUNCS17 = [1] + list(range(3, 31))
TEST_DIMS, BBOB_FNS, RUNS = (20, 40), list(range(1, 25)), 30
# integer flag, low, high, log scale
SPACE = {'CEO': [('N0f', 8, 25, 1, 0), ('n_b', 2, 8, 1, 0), ('E', 10, 200, 0, 1), ('theta', 1, 5, 1, 0),
                 ('c_micro', 0.2, 3, 0, 1), ('p', 0.05, 0.25, 0, 0)],
         'L-SHADE': [('r_Ninit', 5, 25, 0, 0), ('H', 3, 10, 1, 0), ('p', 0.05, 0.25, 0, 0), ('arc_rate', 1, 3, 0, 0)],
         'jSO': [('c_Ninit', 10, 40, 0, 0), ('H', 3, 10, 1, 0), ('p_max', 0.1, 0.4, 0, 0), ('arc_rate', 0.5, 2.5, 0, 0)]}
DEFAULT = {'L-SHADE': dict(r_Ninit=18, H=6, p=0.11, arc_rate=2.6), 'jSO': dict(c_Ninit=25, H=5, p_max=0.25, arc_rate=1.0)}


def ceo_frozen():
    return json.load(open(os.path.join(HERE, 'tuned_v4.json')))['CEO-v4']


def configs(alg):
    """configuration 0 = current setting, 1..23 = seeded random draws (PROTOCOL_rev2.md)"""
    rng = np.random.default_rng({'CEO': 31, 'L-SHADE': 32, 'jSO': 33}[alg] + 20261022)
    out = [ceo_frozen() if alg == 'CEO' else dict(DEFAULT[alg])]
    for _ in range(NCONF - 1):
        c = {}
        for name, lo, hi, is_int, is_log in SPACE[alg]:
            if is_int:
                c[name] = int(rng.integers(lo, hi + 1))
            elif is_log:
                c[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            else:
                c[name] = float(rng.uniform(lo, hi))
        out.append(c)
    return out


def run_alg(alg, obj, lb, ub, dim, seed, params):
    if alg == 'CEO':
        from ceo_v4 import RUNNERS
        RUNNERS['CEO-v4'](obj, lb, ub, dim, seed, params)
    else:
        from baselines import RUNNERS
        RUNNERS[alg](obj, lb, ub, dim, seed, params)


def tune_jobs():
    return [('tune', a, c, fn, s) for a in ALGS for c in range(NCONF) for fn in FUNCS17 for s in TUNE_SEEDS]


def test_algs():
    t = json.load(open(os.path.join(HERE, 'tuned_rev2.json')))
    return {'CEO': ceo_frozen(), 'CEO-T': t['CEO'], 'L-SHADE': {}, 'L-SHADE-T': t['L-SHADE'], 'jSO': {}, 'jSO-T': t['jSO']}


def test_jobs():
    return [('test', a, d, fn, r) for a in ['CEO', 'CEO-T', 'L-SHADE', 'L-SHADE-T', 'jSO', 'jSO-T']
            for d in TEST_DIMS for fn in BBOB_FNS for r in range(1, RUNS + 1)]


def bbob_problem(fn, dim, inst):
    import cocoex
    suite = cocoex.Suite('bbob', f'instances: {inst}', f'dimensions: {dim} function_indices: {fn}')
    p = suite.get_problem_by_function_dimension_instance(fn, dim, inst)
    return suite, p


def bbob_fopt(fn, dim, inst, tmpdir):
    """f_opt = f(x_opt), x_opt written by cocoex into the working directory (single process only)"""
    cwd = os.getcwd(); os.chdir(tmpdir)
    try:
        suite, p = bbob_problem(fn, dim, inst)
        p._best_parameter('print')
        f = [x for x in os.listdir('.') if 'best_parameter' in x][0]
        xo = np.loadtxt(f); os.remove(f)
        return float(p(xo))
    finally:
        os.chdir(cwd)


def cost(j):
    if j[0] == 'tune':
        return 300_000 * (25 if j[1] != 'CEO' else 20)
    return 10_000 * j[2] * (25 if 'CEO' not in j[1] else 20)


def shard_jobs(jobs, shard, nshards):
    jobs = sorted(jobs, key=lambda j: (-cost(j), j))
    loads = [0.0] * nshards; buckets = [[] for _ in range(nshards)]
    for j in jobs:
        k = int(np.argmin(loads)); buckets[k].append(j); loads[k] += cost(j)
    return buckets[shard]


def job(j, fopt=None):
    from budget import Objective
    t = time.perf_counter()
    if j[0] == 'tune':
        _, alg, c, fn, s = j
        from cec2017_official import CEC2017Function
        f = CEC2017Function(fn, TUNE_DIM)
        obj = Objective(f, TUNE_DIM, 10_000 * TUNE_DIM, f.optimum)
        seed = 3_000_000 + 100 * fn + s
        run_alg(alg, obj, -100., 100., TUNE_DIM, seed, configs(alg)[c])
        return dict(stage='tune', algorithm=alg, config=c, function=f'F{fn}', seed_idx=s, seed=seed, error=obj.error(),
                    fe=obj.n, sec=round(time.perf_counter() - t, 2))
    _, alg, dim, fn, r = j
    suite, p = bbob_problem(fn, dim, r)
    obj = Objective(p, dim, 10_000 * dim, fopt)
    seed = 4_000_000 + 10_000 * dim + 100 * fn + r
    run_alg(alg.replace('-T', ''), obj, -5., 5., dim, seed, test_algs()[alg])
    return dict(stage='test', algorithm=alg, dim=dim, function=f'f{fn}', run=r, instance=r, seed=seed, fopt=fopt,
                error=obj.error(), fe=obj.n, sec=round(time.perf_counter() - t, 2),
                trace=json.dumps([float(e) for e in obj.error_trace()]))


def run_job(args):
    return job(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['tune', 'test'], required=True)
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--procs', type=int, default=4); ap.add_argument('--outdir', default='results')
    ap.add_argument('--hard-stop-h', type=float, default=8.6)
    a = ap.parse_args()
    outdir = os.path.abspath(a.outdir); os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f'rev2-{a.stage}_s{a.shard}.csv')
    jobs = shard_jobs(tune_jobs() if a.stage == 'tune' else test_jobs(), a.shard, a.nshards)
    done = set()
    if os.path.exists(out):
        d = pd.read_csv(out)
        done = set(zip(d.algorithm, d.config, d.function, d.seed_idx)) if a.stage == 'tune' else set(zip(d.algorithm, d.dim, d.function, d.run))
    key = (lambda j: (j[1], j[2], f'F{j[3]}', j[4])) if a.stage == 'tune' else (lambda j: (j[1], j[2], f'f{j[3]}', j[4]))
    jobs = [j for j in jobs if key(j) not in done]
    fo = {}
    if a.stage == 'test':
        tmp = os.path.join(outdir, '_fopt'); os.makedirs(tmp, exist_ok=True)
        for (_, _, d, fn, r) in jobs:
            if (d, fn, r) not in fo:
                fo[(d, fn, r)] = bbob_fopt(fn, d, r, tmp)
    args = [(j, fo.get((j[2], j[3], j[4]))) for j in jobs]
    print(f'{a.stage} shard {a.shard}/{a.nshards}: {len(args)} jobs', flush=True)
    t0 = time.time(); stop = t0 + a.hard_stop_h * 3600; n = 0
    pool = Pool(a.procs); q = queue.Queue(); it = iter(args); inflight = 0

    def submit():
        nonlocal inflight
        if time.time() < stop:
            x = next(it, None)
            if x is not None:
                pool.apply_async(run_job, (x,), callback=q.put, error_callback=q.put); inflight += 1
    for _ in range(a.procs):
        submit()
    while inflight:
        r = q.get(); inflight -= 1
        if isinstance(r, BaseException):
            raise r
        n += 1
        pd.DataFrame([r]).to_csv(out, mode='a', header=not os.path.exists(out), index=False)
        print(f'[{n}/{len(args)}] {r["algorithm"]} {r["function"]} err={r["error"]:.4g} {r["sec"]}s | {(time.time()-t0)/3600:.2f} h', flush=True)
        submit()
    pool.close(); pool.join()
    left = len(args) - n
    print('SHARD_DONE' if left == 0 else f'SHARD_STOPPED_EARLY {left} jobs left', flush=True)


if __name__ == '__main__':
    main()

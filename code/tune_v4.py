# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""CEO-v4 tuning (PROTOCOL_CEO_v4.md): same rule and budget as the tuned baselines of the earlier study.
24 configs (config 0 = default) x 12 tuning-suite functions x 3 seeds, D=10, 100,000 FEs.
Writes results/v4-tuning_raw.csv, then selects and freezes tuned_v4.json."""
import os, sys, json, math, time, glob
import numpy as np, pandas as pd
from multiprocessing import Pool
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
from budget import Objective

DIM = 10; BUDGET = 100_000; N_CFG = 24; SEEDS = [910001, 910002, 910003]; FUNCS = list(range(12))
OUT = 'results/v4-tuning_raw.csv'
SPACE = [('N0f', 8, 25, 'lin', 1), ('n_b', 2, 8, 'lin', 1), ('E', 10, 200, 'log', 0),
         ('theta', 1, 5, 'lin', 1), ('c_micro', .2, 3., 'log', 0), ('p', .05, .25, 'lin', 0)]


def configs():
    rng = np.random.default_rng(20261016)
    out = [{}]
    for _ in range(N_CFG - 1):
        c = {}
        for name, lo, hi, sc, isint in SPACE:
            u = rng.random()
            v = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))) if sc == 'log' else lo + u * (hi - lo)
            c[name] = int(round(v)) if isint else float(v)
        out.append(c)
    return out


def job(args):
    ci, cfg, fn, seed = args
    from tuning_suite import TuningFunction
    from ceo_v4 import RUNNERS
    f = TuningFunction(fn, DIM, SEEDS.index(seed) + 1)
    obj = Objective(f, DIM, BUDGET, f.optimum)
    t = time.perf_counter(); RUNNERS['CEO-v4'](obj, -100., 100., DIM, seed, cfg)
    return dict(algorithm='CEO-v4', cfg=ci, function=f'TS_{fn:02d}', seed=seed, error=obj.error(), fe=obj.n,
                sec=round(time.perf_counter() - t, 2), params=json.dumps(cfg))


def select():
    d = pd.read_csv(OUT).drop_duplicates(['cfg', 'function', 'seed'])
    assert len(d) == N_CFG * 12 * 3, f'tuning grid incomplete: {len(d)}'
    assert (d.fe == BUDGET).all()
    med = d.groupby(['function', 'cfg']).error.median().unstack()
    mr = med.rank(axis=1, method='average').mean(0)
    best = int(sorted(mr.index, key=lambda c: (mr[c], c))[0])
    params = {**__import__('ceo_v4').DEFAULTS, **json.loads(d[d.cfg == best].params.iloc[0])}
    json.dump({'CEO-v4': params, 'selected_cfg': best, 'mean_rank': round(float(mr[best]), 3),
               'default_cfg0_mean_rank': round(float(mr[0]), 3)}, open('tuned_v4.json', 'w'), indent=1)
    t = pd.DataFrame(dict(cfg=mr.index, mean_rank=mr.values)).sort_values('mean_rank')
    t['params'] = [d[d.cfg == c].params.iloc[0] for c in t.cfg]
    t.to_csv('results/v4-tuning_selection.csv', index=False)
    print(t.head(8).to_string(index=False)); print('FROZEN:', params)


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == 'select':
        select(); sys.exit()
    done = set()
    if os.path.exists(OUT):
        d = pd.read_csv(OUT); done = set(zip(d.cfg, d.function, d.seed))
    jobs = [(ci, c, fn, s) for ci, c in enumerate(configs()) for fn in FUNCS for s in SEEDS
            if (ci, f'TS_{fn:02d}', s) not in done]
    print(f'{len(jobs)} jobs', flush=True); t0 = time.time()
    with Pool(int(os.environ.get('PROCS', 4))) as pool:
        for k, r in enumerate(pool.imap_unordered(job, jobs, chunksize=4), 1):
            pd.DataFrame([r]).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
            if k % 100 == 0:
                print(f'[{k}/{len(jobs)}] {(time.time()-t0)/60:.1f} min', flush=True)
    print('TUNING_DONE', flush=True)

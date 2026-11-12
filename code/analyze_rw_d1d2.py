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
"""tab:rw after Deviations D1/D2/D2c of PROTOCOL_rwx.md: CEO + 17 opponents on RC15-RC33.
Replaced by the audited runs: L-SHADE, DE, LSHADE-cnEpSin, IMODE (RWA, runs/rwa), INFO, HBA, SCSO (RWB,
runs/rwb incl. the D2c rerun of HBA RC20); added jSO (RWA). The INFO files of RWA are superseded (D2 item 4).
Kept unchanged: CEO, ABC, ACOR, GWO, HHO, PSO, WOA (audited swarm versions), BIPOP-CMA-ES, GA, Sawtooth-GA.
Analysis code, tests and outputs as fixed in the protocol (analyze_rw19.evo_table / deb_rank, same rules).
Also reports the superseded mealpy results next to the audited ones, per method (D1 item 5).
Output: results/analysis/rw19_audited/."""
import glob, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import analyze_rw19 as W

OUT = os.path.join(HERE, 'results', 'analysis', 'rw19_audited'); os.makedirs(OUT, exist_ok=True)
W.OUT = OUT; W.LOG.clear()
say = W.say
REPLACED = ['L-SHADE', 'DE', 'LSHADE-cnEpSin', 'IMODE', 'INFO', 'HBA', 'SCSO']
FROM_RWA = ['L-SHADE', 'DE', 'LSHADE-cnEpSin', 'IMODE', 'jSO']
FROM_RWB = ['INFO', 'HBA', 'SCSO']


def read(pkg, prefix, algs):
    fs = sorted(glob.glob(os.path.join(ROOT, pkg, 'out', '*', 'results', prefix + '__*.csv')))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d = d[d.algorithm.isin(algs)]
    assert not d.duplicated(['algorithm', 'function', 'seed']).any()
    return d


if __name__ == '__main__':
    _, ev_old = W.load()
    rwa = read('runs/rwa', 'RWA', FROM_RWA)
    rwb = read('runs/rwb', 'RWB', FROM_RWB)
    new = pd.concat([rwa, rwb], ignore_index=True)
    assert set(new.fe_used.unique()) <= {100000, 200000}, new.fe_used.unique()   # exact budget, D <= 10 / D > 10
    keep = ev_old[~ev_old.algorithm.isin(REPLACED)]
    ev = pd.concat([keep[W.C], new[W.C]], ignore_index=True)
    ev['value'] = ev.best_fitness.astype(float); ev['err'] = np.round(ev.value, 8)
    cnt = ev.groupby(['algorithm', 'function']).seed.apply(lambda s: sorted(s) == list(range(30)))
    assert cnt.all() and len(cnt) == 18 * 19, cnt[~cnt]
    assert np.isfinite(ev.value).all()
    say(f'evo set {ev.algorithm.nunique()} methods (CEO + 17), 19 problems x 30 seeds, complete; '
        f'audited runs {len(new)} rows, max FE used {new.fe_used.max()}')
    for tag, fns in (('RC15-21', W.OLD), ('RC22-33', W.NEW), ('RC15-33', W.OLD + W.NEW)):
        W.evo_table(ev, fns, tag)
        W.deb_rank(ev, fns, f'evo {tag}')
    fe = ev.groupby(['algorithm', 'function']).feasible.sum().unstack()
    fe.to_csv(os.path.join(OUT, 'feasible_counts.csv'))
    say('\nfeasible runs of 30 per audited method:')
    for a in FROM_RWA + FROM_RWB:
        say(f'  {a:15s} ' + ' '.join(f'{f[2:]}:{int(fe.loc[a, f])}' for f in W.OLD + W.NEW))
    say('penalised vs feasibility notes: ' + '; '.join(W.disagree(ev, W.OLD + W.NEW)))

    # superseded mealpy vs audited, per method: mean rank of the method among the 18 on each problem
    say('\n=== superseded (mealpy / R1) vs audited, mean penalised objective per problem')
    rows = []
    for a in REPLACED:
        o = ev_old[ev_old.algorithm == a].groupby('function').value.mean()
        n = ev[ev.algorithm == a].groupby('function').value.mean()
        better = int((n < o - 1e-8 * np.maximum(1, np.abs(o))).sum()); worse = int((n > o + 1e-8 * np.maximum(1, np.abs(o))).sum())
        say(f'  {a:15s} audited better on {better:2d}, worse on {worse:2d}, equal on {19 - better - worse:2d} of 19 problems')
        for f in W.OLD + W.NEW:
            rows.append(dict(algorithm=a, function=f, superseded_mean=o[f], audited_mean=n[f]))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'superseded_vs_audited.csv'), index=False)
    open(os.path.join(OUT, 'summary.txt'), 'w').write('\n'.join(W.LOG) + '\n')

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
"""CEC2020-RW mechanical problems RC15-RC33 (PROTOCOL_rwx.md, analysis fixed 2026-11-06).
RC15-RC21: existing files (R1 E4, results/swarm-rw.csv, results/v4-rw.csv). RC22-RC33: runs/rwx/rwx_all.csv.
Tables: (1) tab:rw = CEO + 16 opponents, paired signed-rank per problem, Holm over opponents (analyze_rw_audited rule);
(2) tab:swarmrw = CEO + 19 swarm, rank-sum per problem, Holm over problems (analyze_swarm rule);
(3) feasibility; (4) secondary CEC2020 feasibility-rule ranking. Each on RC15-21, RC22-33 and all 19."""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'r1')); sys.path.insert(0, os.path.join(ROOT, 'experiments')); sys.path.insert(0, HERE)
import analyze_r1 as R
import analyze_fe as A
import analyze_swarm as S
OUT = os.path.join(HERE, 'results', 'analysis', 'rw19'); os.makedirs(OUT, exist_ok=True)
SW6 = {'ABC': 'ABC', 'ACO': 'ACOR', 'GWO': 'GWO', 'HHO': 'HHO', 'PSO': 'PSO', 'WOA': 'WOA'}
C = ['function', 'algorithm', 'seed', 'best_fitness', 'objective', 'violation', 'feasible']
OLD = [f'RC{k}' for k in range(15, 22)]; NEW = [f'RC{k}' for k in range(22, 34)]
LOG = []


def say(s=''):
    print(s); LOG.append(str(s))


def load():
    sw = pd.read_csv(os.path.join(HERE, 'results', 'swarm-rw.csv'))
    v4 = pd.read_csv(os.path.join(HERE, 'results', 'v4-rw.csv')).replace({'algorithm': {'CEO-v4': 'CEO'}})
    r1 = R.load_r1(); r1 = r1[r1.exp == 'E4']
    x = pd.read_csv(os.path.join(ROOT, 'runs/rwx', 'rwx_all.csv'))
    swarm = pd.concat([sw[C], v4[C], x[x.algorithm.isin(['CEO'] + list(S.ORDER))][C]], ignore_index=True)
    keep = r1[~r1.algorithm.isin(list(SW6.values()) + ['CEO'])]
    evo = pd.concat([keep[C], sw[sw.algorithm.isin(SW6)][C], v4[C],
                     x[~x.algorithm.isin([a for a in S.ORDER if a not in SW6])][C]], ignore_index=True).replace({'algorithm': SW6})
    for d in (swarm, evo):
        d['value'] = d.best_fitness.astype(float); d['err'] = np.round(d.value, 8)
        cnt = d.groupby(['algorithm', 'function']).seed.apply(lambda s: sorted(s) == list(range(30)))
        assert cnt.all() and len(cnt) == d.algorithm.nunique() * 19, cnt[~cnt]
    return swarm, evo


def evo_table(e, fns, tag):
    others = sorted(a for a in e.algorithm.unique() if a != 'CEO')
    sub = e[e.function.isin(fns)]
    c = A.compare(sub, 'CEO', others, fns)
    M = sub.groupby(['function', 'algorithm']).err.mean().unstack()
    rk = M.rank(axis=1).mean().sort_values()
    chi2, p = stats.friedmanchisquare(*M.values.T)
    g = c.groupby('other').outcome.value_counts().unstack(fill_value=0)
    say(f'\n=== tab:rw {tag}: {M.shape[1]} methods x {len(fns)} problems, Friedman chi2={chi2:.1f} p={p:.2g}')
    rows = []
    for i, (a, r) in enumerate(rk.items(), 1):
        w = g.loc[a] if a in g.index else None
        wtl = '--' if w is None else f"{w.get('win', 0)}/{w.get('tie', 0)}/{w.get('loss', 0)}"
        say(f'  {i:2d}. {a:16s} {r:5.2f}  CEO W/T/L {wtl}')
        rows.append(dict(position=i, algorithm=a, avg_rank=r, wtl=wtl))
    tot = c.outcome.value_counts()
    say(f"  total CEO W/T/L: {tot.get('win', 0)}/{tot.get('tie', 0)}/{tot.get('loss', 0)}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f'rw_{tag}.csv'), index=False)
    c.to_csv(os.path.join(OUT, f'rw_{tag}_pvalues.csv'), index=False)


def deb_rank(d, fns, tag):
    """CEC2020 feasibility rule on each final point: feasible < infeasible; feasible by f, infeasible by v."""
    rows = []
    for f in fns:
        s = d[d.function == f].copy()
        key = np.where(s.violation <= 0, 0, 1)
        s = s.assign(k=key, o=np.where(key == 0, s.objective, s.violation))
        s['r'] = s.sort_values(['k', 'o']).reset_index().index.to_series().values[np.argsort(s.sort_values(['k', 'o']).index.argsort())] if False else 0
        order = s.sort_values(['k', 'o'], kind='mergesort').index
        s.loc[order, 'r'] = stats.rankdata(list(zip(s.loc[order, 'k'], s.loc[order, 'o'])), method='average') if False else np.arange(len(s))
        # average ranks for exact ties
        s['r'] = stats.rankdata(s.k.values * 1e300 + 0) if False else s['r']
        t = s.groupby(['k', 'o']).r.transform('mean'); s['r'] = t
        rows.append(s.groupby('algorithm').r.mean().rename(f))
    M = pd.concat(rows, axis=1).T
    rk = M.rank(axis=1).mean().sort_values()
    say(f'  secondary feasibility-rule ranking {tag}: ' + ', '.join(f'{a} {r:.2f}' for a, r in list(rk.items())[:8]) +
        f' ... CEO position {list(rk.index).index("CEO") + 1}/{len(rk)}')
    return M


def disagree(d, fns):
    out = []
    for f in fns:
        s = d[d.function == f]
        pen = s.groupby('algorithm').value.mean().rank()
        feas = s.groupby('algorithm').feasible.mean()
        if feas.max() == 0:
            out.append(f'{f}: no method feasible')
        elif (feas < 1).any() and pen.idxmin() != s[s.feasible == 1].groupby('algorithm').objective.mean().idxmin():
            out.append(f'{f}: best by penalty {pen.idxmin()} differs from best feasible mean')
    return out


if __name__ == '__main__':
    sw, ev = load()
    say(f'swarm set {sw.algorithm.nunique()} methods, evo set {ev.algorithm.nunique()} methods, 19 problems x 30 seeds, complete')
    for tag, fns in (('RC15-21', OLD), ('RC22-33', NEW), ('RC15-33', OLD + NEW)):
        S.L.clear(); S.say = say
        S.OUT = OUT
        S.analyse(sw[sw.function.isin(fns)], fns, f'swarm_{tag}')
        deb_rank(sw, fns, f'swarm {tag}')
        evo_table(ev, fns, tag)
        deb_rank(ev, fns, f'evo {tag}')
    fe = pd.concat([sw, ev]).drop_duplicates(['algorithm', 'function', 'seed']).groupby(['algorithm', 'function']).feasible.sum().unstack()
    fe.to_csv(os.path.join(OUT, 'feasible_counts.csv'))
    say('\nfeasible runs of 30, CEO: ' + ' '.join(f'{f}:{int(fe.loc["CEO", f])}' for f in OLD + NEW))
    say('problems where no method is ever feasible: ' + str([f for f in OLD + NEW if fe[f].max() == 0]))
    say('problems where CEO is not always feasible: ' + str([f for f in OLD + NEW if fe.loc['CEO', f] < 30]))
    say('penalised vs feasibility notes: ' + '; '.join(disagree(pd.concat([sw, ev]).drop_duplicates(['algorithm', 'function', 'seed']), NEW)))
    open(os.path.join(OUT, 'summary.txt'), 'w').write('\n'.join(LOG) + '\n')

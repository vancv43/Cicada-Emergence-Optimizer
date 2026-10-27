# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""CEO and the 16 published methods on CEC2020-RW RC15-RC21, with the audited swarm-study versions of ABC, ACOR,
GWO, HHO, PSO and WOA (swarm_orig.py, results/swarm-rw.csv) in place of the R1 library versions.
Paired signed-rank tests per problem over the 30 runs, Holm over the 16 opponents (R1 rule), and the average
Friedman rank of the mean penalized objective. Output: results/analysis/rev3_rw_table_audited.csv and
results/analysis/rev3_rw_pvalues_audited.csv"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'experiments', 'r1')); sys.path.insert(0, os.path.join(ROOT, 'experiments'))
import analyze_r1 as R
import analyze_fe as A
OUT = os.path.join(HERE, 'results', 'analysis')
SW = {'ABC': 'ABC', 'ACO': 'ACOR', 'GWO': 'GWO', 'HHO': 'HHO', 'PSO': 'PSO', 'WOA': 'WOA'}
COLS = ['function', 'algorithm', 'seed', 'best_fitness']


if __name__ == '__main__':
    r1 = R.load_r1(); r1 = r1[r1.exp == 'E4']
    sw = pd.read_csv(os.path.join(HERE, 'results', 'swarm-rw.csv')); sw = sw[sw.algorithm.isin(SW)].replace({'algorithm': SW})
    v = pd.read_csv(os.path.join(HERE, 'results', 'v4-rw.csv')).replace({'algorithm': {'CEO-v4': 'CEO'}})
    keep = r1[~r1.algorithm.isin(list(SW.values()) + ['CEO'])]          # 'CEO' in R1 is the superseded first design
    e = pd.concat([keep[COLS], sw[COLS], v[COLS]]).copy()
    e['err'] = np.round(e.best_fitness.astype(float), 8)
    assert (e.groupby(['function', 'algorithm']).seed.apply(lambda s: sorted(s) == list(range(30)))).all()
    others = sorted(a for a in e.algorithm.unique() if a != 'CEO'); fns = sorted(e.function.unique())
    c = A.compare(e, 'CEO', others, fns)
    M = e.groupby(['function', 'algorithm']).err.mean().unstack()
    rk = M.rank(axis=1).mean().sort_values()
    chi2, p = stats.friedmanchisquare(*M.values.T)
    g = c.groupby('other').outcome.value_counts().unstack(fill_value=0)
    print(f'Friedman over {M.shape[1]} methods: chi2={chi2:.1f} p={p:.2g}')
    for a, r in rk.items():
        w = g.loc[a] if a in g.index else None
        print('  %-16s %.2f  %s' % (a, r, '--' if w is None else f"{w.get('win', 0)}/{w.get('tie', 0)}/{w.get('loss', 0)}"))
    pd.DataFrame({'avg_rank': rk}).to_csv(os.path.join(OUT, 'rev3_rw_table_audited.csv'))
    c.to_csv(os.path.join(OUT, 'rev3_rw_pvalues_audited.csv'), index=False)

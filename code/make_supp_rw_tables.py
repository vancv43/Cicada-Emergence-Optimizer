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
"""Rows of Supplementary Tables S1-S6 (feasibility, mean unpenalized objective on RC15-RC33) and S10 (superseded
library versions vs audited ports), from the audited data set of analyze_rw_d1d2.py (PROTOCOL_rwx.md D1/D2/D2c).
Writes results/analysis/rw19_audited/supp_*.tex"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import analyze_rw19 as W
import analyze_rw_d1d2 as A
OUT = A.OUT
sw, ev_old = W.load()
new = pd.concat([A.read('runs/rwa', 'RWA', A.FROM_RWA), A.read('runs/rwb', 'RWB', A.FROM_RWB)], ignore_index=True)
ev = pd.concat([ev_old[~ev_old.algorithm.isin(A.REPLACED)][W.C], new[W.C]], ignore_index=True)
allr = pd.concat([sw[W.C], ev[W.C]]).drop_duplicates(['algorithm', 'function', 'seed'])
allr = allr.replace({'algorithm': {'Sawtooth-GA': 'Sawtooth GA'}})
algs = sorted([a for a in allr.algorithm.unique() if a != 'CEO'], key=str.lower) + ['CEO']
assert len(algs) == 32, len(algs)
blocks = [[f'RC{k}' for k in range(15, 22)], [f'RC{k}' for k in range(22, 28)], [f'RC{k}' for k in range(28, 34)]]
v = allr.groupby(['algorithm', 'function'])
feas, vmax, obj = v.feasible.sum(), v.violation.max(), v.objective.mean()
for i, fns in enumerate(blocks, 1):
    f1 = [f'{a} & ' + ' & '.join(str(int(feas[a, f])) if feas[a, f] == 30 else f'{int(feas[a, f])}, {vmax[a, f]:.2E}' for f in fns) + r' \\' for a in algs]
    f2 = [f'{a} & ' + ' & '.join(f'{obj[a, f]:.6g}' for f in fns) + r' \\' for a in algs]
    open(os.path.join(OUT, f'supp_feas{i}.tex'), 'w').write('\n'.join(f1) + '\n')
    open(os.path.join(OUT, f'supp_obj{i}.tex'), 'w').write('\n'.join(f2) + '\n')
old_rank = pd.read_csv(os.path.join(HERE, 'results', 'analysis', 'rw19', 'rw_RC15-33.csv')).set_index('algorithm')
new_rank = pd.read_csv(os.path.join(OUT, 'rw_RC15-33.csv')).set_index('algorithm')
cmp_ = pd.read_csv(os.path.join(OUT, 'superseded_vs_audited.csv'))
src = {'L-SHADE': 'mealpy 3.0.3', 'DE': 'mealpy 3.0.3, untuned', 'LSHADE-cnEpSin': 'mealpy 3.0.3', 'IMODE': 'mealpy 3.0.3',
       'INFO': 'mealpy 3.0.3', 'HBA': 'mealpy 3.0.3', 'SCSO': 'mealpy 3.0.3'}
rows = []
for a in A.REPLACED:
    c = cmp_[cmp_.algorithm == a]; o, n = c.superseded_mean.values, c.audited_mean.values
    tol = 1e-8 * np.maximum(1, np.abs(o)); b, w = int((n < o - tol).sum()), int((n > o + tol).sum())
    rows.append(f"{a} & {old_rank.loc[a, 'avg_rank']:.2f} & {new_rank.loc[a, 'avg_rank']:.2f} & {b}/{19 - b - w}/{w} \\\\")
rows.append(f"CEO & {old_rank.loc['CEO', 'avg_rank']:.2f} & {new_rank.loc['CEO', 'avg_rank']:.2f} & -- \\\\")
open(os.path.join(OUT, 'supp_superseded.tex'), 'w').write('\n'.join(rows) + '\n')
print('\n'.join(rows)); print('algs', len(algs))

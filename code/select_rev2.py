# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""Select the D = 30 configuration of CEO, L-SHADE and jSO by the rule fixed in PROTOCOL_rev2.md and freeze it in
tuned_rev2.json: lowest mean rank, over the 29 CEC2017 functions, of the median error over the 3 seeds; ties go to the
lower configuration index. Refuses to write unless every tuning run is present exactly once."""
import os, sys, glob, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
import run_rev2 as R

d = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(ROOT, 'runs/rev2t', 'out', '*', 'results', 'rev2-tune_s*.csv')))
exp = {(a, c, f'F{fn}', s) for (_, a, c, fn, s) in R.tune_jobs()}
got = list(zip(d.algorithm, d.config, d.function, d.seed_idx))
assert len(got) == len(set(got)) == len(exp) and set(got) == exp, (len(got), len(set(got)), len(exp))
assert (d.fe == 300_000).all() and d.error.notna().all()
d.loc[d.error < 1e-8, 'error'] = 0.0
out, lines = {}, []
for a in R.ALGS:
    med = d[d.algorithm == a].groupby(['function', 'config']).error.median().unstack()
    rank = med.rank(axis=1).mean().sort_index()
    best = int(rank.idxmin())                         # idxmin returns the first, i.e. lowest index, on ties
    out[a] = R.configs(a)[best]
    lines.append(f'{a}: chosen configuration {best} (mean rank {rank[best]:.2f}; configuration 0 mean rank {rank[0]:.2f}) -> {out[a]}')
json.dump(out, open(os.path.join(HERE, 'tuned_rev2.json'), 'w'), indent=1)
open(os.path.join(HERE, 'results', 'analysis', 'rev2_selection.txt'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))

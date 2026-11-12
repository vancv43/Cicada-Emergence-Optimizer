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
"""Secondary ranking of PROTOCOL_rwx.md item 4 (CEC2020 feasibility rule) for every method of the audited tab:rw,
per block. Output: results/analysis/rw19_audited/feas_rank.csv"""
import os, sys, io, contextlib
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_rw19 as W, analyze_rw_d1d2 as A
sw, ev_old = W.load()
new = pd.concat([A.read('runs/rwa', 'RWA', A.FROM_RWA), A.read('runs/rwb', 'RWB', A.FROM_RWB)], ignore_index=True)
ev = pd.concat([ev_old[~ev_old.algorithm.isin(A.REPLACED)][W.C], new[W.C]], ignore_index=True)
out = {}
for tag, fns in (('RC15-21', W.OLD), ('RC22-33', W.NEW), ('RC15-33', W.OLD + W.NEW)):
    with contextlib.redirect_stdout(io.StringIO()):
        M = W.deb_rank(ev, fns, tag)
    out[tag] = M.rank(axis=1).mean()
d = pd.DataFrame(out); d['position'] = d['RC15-33'].rank(method='min').astype(int)
d.sort_values('RC15-33').to_csv(os.path.join(A.OUT, 'feas_rank.csv')); print(d.sort_values('RC15-33').round(2))

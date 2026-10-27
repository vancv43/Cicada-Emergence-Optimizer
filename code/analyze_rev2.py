# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""Analysis of study 2 (PROTOCOL_rev2.md), exactly as fixed there, on the BBOB test runs.
Primary: CEO-T against L-SHADE-T and jSO-T over the 48 cases (24 functions x D = 20, 40): signed-rank on mean errors,
Holm over the two opponents, with sign test and median A12; also per dimension.
Secondary: per-function rank-sum W/T/L (Holm over functions) for every reported pair, Friedman ranks of the six methods.
Output: results/analysis/rev2_summary.txt, rev2_pairs.csv, rev2_mean_error.csv"""
import os, sys, glob
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
from analyze_rev1_c import holm, a12, wtl, paired

OUT = os.path.join(HERE, 'results', 'analysis'); L = []
ALGS = ['CEO', 'CEO-T', 'L-SHADE', 'L-SHADE-T', 'jSO', 'jSO-T']
FNS = [f'f{k}' for k in range(1, 25)]


def say(s=''):
    print(s); L.append(str(s))


def load():
    d = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(ROOT, 'runs/rev2b', 'out', '*', 'results', 'rev2-test_s*.csv')))
    key = ['algorithm', 'dim', 'function', 'run']
    assert len(d) == 6 * 2 * 24 * 30 and not d.duplicated(key).any(), (len(d), d.duplicated(key).sum())
    assert (d.fe == 10_000 * d.dim).all() and d.error.notna().all() and (d.error >= 0).all()
    d.loc[d.error < 1e-8, 'error'] = 0.0
    return d


def median_a12(d, a, b, dims):
    return float(np.median([a12(d[(d.dim == D) & (d.function == f) & (d.algorithm == a)].error,
                                d[(d.dim == D) & (d.function == f) & (d.algorithm == b)].error) for D in dims for f in FNS]))


def main():
    d = load()
    M = d.groupby(['dim', 'function', 'algorithm']).error.mean().unstack()
    M.to_csv(os.path.join(OUT, 'rev2_mean_error.csv'))
    say('=== PRIMARY: CEO-T vs L-SHADE-T and jSO-T, BBOB, 48 cases, signed-rank on mean error, Holm over 2')
    res = [paired(M, 'CEO-T', o) for o in ['L-SHADE-T', 'jSO-T']]
    for o, r, q in zip(['L-SHADE-T', 'jSO-T'], res, holm([r[3] for r in res])):
        say(f'  vs {o:9s}: better/worse/tie {r[0]}/{r[1]}/{r[2]}  signed-rank Holm p={q:.3g}  sign p={r[4]:.3g}  '
            f'median A12={median_a12(d, "CEO-T", o, (20, 40)):.2f}')
    for D in (20, 40):
        m = M.loc[D]; res = [paired(m, 'CEO-T', o) for o in ['L-SHADE-T', 'jSO-T']]
        say(f'  D{D}: ' + '; '.join(f'{o} B/W/T {r[0]}/{r[1]}/{r[2]} Holm p={q:.3g} sign p={r[4]:.3g} A12={median_a12(d, "CEO-T", o, (D,)):.2f}'
                                   for o, r, q in zip(['L-SHADE-T', 'jSO-T'], res, holm([r[3] for r in res]))))
    say('\n=== SECONDARY (descriptive): rank-sum W/T/L with Holm over the 24 functions, first method\'s side')
    rows = []
    pairs = [('CEO-T', 'L-SHADE-T'), ('CEO-T', 'jSO-T'), ('CEO-T', 'CEO'), ('L-SHADE-T', 'L-SHADE'), ('jSO-T', 'jSO'),
             ('CEO-T', 'L-SHADE'), ('CEO-T', 'jSO'), ('CEO', 'L-SHADE'), ('CEO', 'jSO')]
    for D in (20, 40):
        g = d[d.dim == D]
        for a, b in pairs:
            w, t, l, won, lost = wtl(g, a, b, FNS); bt, wo, ti, sr, sg = paired(M.loc[D], a, b)
            rows.append(dict(dim=D, a=a, b=b, win=w, tie=t, loss=l, better=bt, worse=wo, tie_mean=ti, signed_rank_p=sr, sign_p=sg,
                             a12=median_a12(d, a, b, (D,)), won_on=won, lost_on=lost))
            say(f'  D{D} {a:9s} vs {b:9s}: W/T/L {w}/{t}/{l}  B/W/T {bt}/{wo}/{ti}  signed-rank p={sr:.3g}  A12={rows[-1]["a12"]:.2f}')
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'rev2_pairs.csv'), index=False)
    say('\n=== Friedman over the six methods')
    for D in (20, 40):
        X = M.loc[D][ALGS].values; R = np.vstack([stats.rankdata(r) for r in X]); chi2, p = stats.friedmanchisquare(*X.T)
        say(f'  D{D}: chi2={chi2:.2f} p={p:.3g}; avg rank ' + ', '.join(f'{a} {v:.2f}' for a, v in sorted(zip(ALGS, R.mean(0)), key=lambda x: x[1])))
    open(os.path.join(OUT, 'rev2_summary.txt'), 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()

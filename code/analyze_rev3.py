# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""Reanalysis of finished runs for the advisor review of 2026-10-25 (Phayung_Review/Comments_to_Authors_Overleaf.pdf).
No optimisation run is repeated or changed; part (e) only re-times the reference loops.
(a) M3: primary test without the CEC2022 D10 block that selected the opponents, and at the level of function families.
(b) M6: realised generations and gates per brood of the frozen CEO on CEC2022 at D = 10.
(c) M8: instance-matched paired per-function tests on BBOB and an ECDF with fixed targets.
(d) M9: feasibility, unpenalised objective and residual violation on RC15-RC21 for every method.
(e) M10: repeated T0 and T1 to size the timing variation behind the complexity table.
Output: results/analysis/rev3_*.csv and rev3_summary.txt"""
import os, sys, glob, json, time
import numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
from analyze_rev1_c import holm, wtl
OUT = os.path.join(HERE, 'results', 'analysis'); L = []


def say(s=''):
    print(s, flush=True); L.append(str(s))


def paired_means(a, b):
    bt, wo, ti = int((a < b).sum()), int((a > b).sum()), int((a == b).sum())
    sr = stats.wilcoxon(a, b).pvalue if (a != b).any() else 1.0
    sg = stats.binomtest(bt, bt + wo).pvalue if bt + wo else 1.0
    return bt, wo, ti, sr, sg


def part_a():
    say('=== (a) M3: primary test with and without the CEC2022 D10 discovery block')
    m = pd.read_csv(os.path.join(OUT, 'power_mean_error.csv')).rename(columns={'Unnamed: 0': 'function'})
    prim = pd.read_csv(os.path.join(OUT, 'power_primary.csv')).set_index('opponent')
    opp = ['MPA', 'CS', 'ABC']
    for label, d in [('82 cases', m), ('70 cases, CEC2022 D10 removed', m[m.case != 'CEC2022_D10'])]:
        res = [paired_means(d.CEO.values, d[o].values) for o in opp]
        hs, hg = holm([r[3] for r in res]), holm([r[4] for r in res])
        for o, r, a, b in zip(opp, res, hs, hg):
            say(f'  {label:30s} vs {o}: B/W/T {r[0]}/{r[1]}/{r[2]}  signed-rank Holm p={a:.3g}  sign Holm p={b:.3g}')
            if label == '82 cases':
                assert (r[0], r[1], r[2]) == tuple(prim.loc[o, ['ceo_better', 'ceo_worse', 'tie']]), o
                assert np.isclose(r[3], prim.loc[o, 'wilcoxon_p'], rtol=1e-10), o
    # function families: one vote per suite-function, the mean sign over its dimensions
    m['suite'] = m.case.str.split('_').str[0]
    say('  function families, one vote each (mean sign of the case-level differences over the dimensions):')
    rows = []
    for o in opp:
        s = np.sign(m[o] - m.CEO)                     # +1 = CEO lower mean error
        v = s.groupby([m.suite, m.function]).mean()
        bt, wo, ti = int((v > 0).sum()), int((v < 0).sum()), int((v == 0).sum())
        rows.append((o, bt, wo, ti, stats.binomtest(bt, bt + wo).pvalue))
    for (o, bt, wo, ti, p), q in zip(rows, holm([r[4] for r in rows])):
        say(f'    vs {o}: {bt + wo + ti} families, better/worse/even {bt}/{wo}/{ti}  sign Holm p={q:.3g}')


def part_b():
    say('\n=== (b) M6: realised generations and gates per brood, frozen CEO, CEC2022 D10')
    import cec2022_fast as C
    from budget import Objective
    from ceo_v4 import ceo_v4
    from run_test_v4 import cec_seed
    p = json.load(open(os.path.join(HERE, 'tuned_v4.json')))['CEO-v4']; D, B = 10, 200_000
    N0 = max(int(round(p['N0f'] * D)), 2 * p['n_b'], 6); Nmin = max(4, p['n_b'])
    est_mean = B / ((N0 + Nmin) / 2); est_log = B / (N0 - Nmin) * np.log(N0 / Nmin)
    rows = []
    for fn in range(1, 13):
        f = C.CEC2022Function(fn, D); obj = Objective(f, D, B, f.optimum)
        st = ceo_v4(obj, -100., 100., D, cec_seed(D, fn, 1), p)
        G = st['generations']; P = np.array(st['P'])
        rows.append(dict(function=f'F{fn}', generations=G, **{f'gates_b{k + 1}': G / P[k] for k in range(len(P))},
                         micro=st['micro'], macro=st['macro'], gate_skips=st['gate_skips']))
    d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, 'rev3_gates.csv'), index=False)
    say(f'  N0={N0}, Nmin={Nmin}, periods {st["P"]}; generations estimated by Eq. 2: {est_mean:.0f}; '
        f'by the integral of the linear reduction: {est_log:.0f}; nominal E = {p["E"]:.1f}')
    say(f'  realised generations over F1-F12, run 1: median {d.generations.median():.0f}, range {d.generations.min()}-{d.generations.max()}')
    g = d[[c for c in d.columns if c.startswith('gates_b')]]
    say('  realised gates per brood, median over functions: ' + ', '.join(f'{c[6:]} {v:.0f}' for c, v in g.median().items()))


def part_c():
    say('\n=== (c) M8: BBOB, instance-matched paired tests per function (Wilcoxon signed-rank over the 30 runs, Holm over 24)')
    d = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(ROOT, 'runs/rev2b', 'out', '*', 'results', 'rev2-test_s*.csv')))
    assert len(d) == 8640
    d.loc[d.error < 1e-8, 'error'] = 0.0
    assert (d.groupby(['dim', 'function', 'run']).instance.nunique() == 1).all() and (d.instance == d.run).all()
    fns = [f'f{k}' for k in range(1, 25)]; rows = []
    for D in (20, 40):
        g = d[d.dim == D]
        for a, b in [('CEO-T', 'L-SHADE-T'), ('CEO-T', 'jSO-T'), ('CEO', 'L-SHADE'), ('CEO', 'jSO')]:
            ps, dirs = [], []
            for f in fns:
                x = g[(g.algorithm == a) & (g.function == f)].sort_values('run').error.values
                y = g[(g.algorithm == b) & (g.function == f)].sort_values('run').error.values
                ps.append(stats.wilcoxon(x, y).pvalue if (x != y).any() else 1.0)
                dirs.append(np.sign(np.median(y - x)) or np.sign(np.mean(y - x)))
            q = holm(ps); dirs = np.array(dirs)
            w, l = int(((q < .05) & (dirs > 0)).sum()), int(((q < .05) & (dirs < 0)).sum())
            w2, t2, l2, _, _ = wtl(g, a, b, fns)
            rows.append(dict(dim=D, a=a, b=b, paired_win=w, paired_tie=24 - w - l, paired_loss=l, ranksum_win=w2, ranksum_tie=t2, ranksum_loss=l2))
            say(f'  D{D} {a:6s} vs {b:9s}: paired W/T/L {w}/{24 - w - l}/{l}   rank-sum W/T/L {w2}/{t2}/{l2}')
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'rev3_bbob_paired.csv'), index=False)
    # ECDF with fixed targets 10^2 ... 10^-8, independent of the observed outcomes
    from budget import cec_checkpoints
    algs = ['CEO', 'CEO-T', 'L-SHADE', 'L-SHADE-T', 'jSO', 'jSO-T']; tg = np.logspace(2, -8, 51)
    say('  ECDF with 51 fixed targets from 1e2 to 1e-8 (equal weight per function and run):')
    for D in (20, 40):
        g = d[d.dim == D]; ck = np.array(cec_checkpoints(D, 10_000 * D)); E = {}
        for a in algs:
            T = np.maximum(np.array([json.loads(t) for t in g[g.algorithm == a].trace]), 1e-8)
            E[a] = (T[:, :, None] <= tg[None, None, :]).mean(axis=(0, 2))
        lead = [i for i in range(len(ck)) if E['CEO'][i] > max(E[o][i] for o in algs if o != 'CEO') + 1e-12]
        say(f'    D{D}: frozen CEO reaches the most targets at {len(lead)} of 16 recording points, '
            f'from {ck[lead[0]] / (10_000 * D):.1%} to {ck[lead[-1]] / (10_000 * D):.1%} of the budget' if lead else f'    D{D}: no lead')
        say('      final shares: ' + ', '.join(f'{a} {E[a][-1]:.3f}' for a in algs))


def part_d():
    say('\n=== (d) M9: RC15-RC21, feasibility, unpenalised objective and residual violation')
    sys.path.insert(0, os.path.join(ROOT, 'experiments', 'r1')); sys.path.insert(0, os.path.join(ROOT, 'experiments'))
    import analyze_r1 as R
    r1 = R.load_r1(); r1 = r1[(r1.exp == 'E4') & (r1.algorithm != 'CEO')]
    sw = pd.read_csv(os.path.join(HERE, 'results', 'swarm-rw.csv'))
    v4 = pd.read_csv(os.path.join(HERE, 'results', 'v4-rw.csv')).replace({'algorithm': {'CEO-v4': 'CEO'}})
    cols = ['function', 'algorithm', 'seed', 'best_fitness', 'objective', 'violation', 'feasible']
    sw = sw.replace({'algorithm': {'ACO': 'ACOR'}})                  # the audited swarm-study versions replace the R1 ones
    r1 = r1[~r1.algorithm.isin(sw.algorithm.unique())]
    d = pd.concat([v4[cols], sw[cols].assign(study='swarm'), r1[cols].assign(study='published')], ignore_index=True)
    d['study'] = d['study'].fillna('CEO')
    assert (d.groupby(['function', 'algorithm']).size() == 30).all()
    s = d.groupby(['algorithm', 'function']).agg(feasible=('feasible', 'sum'), objective_mean=('objective', 'mean'),
                                                  objective_best=('objective', 'min'), violation_max=('violation', 'max'),
                                                  penalized_mean=('best_fitness', 'mean')).reset_index()
    s.to_csv(os.path.join(OUT, 'rev3_rw_feasibility.csv'), index=False, float_format='%.17g')
    d.to_csv(os.path.join(OUT, 'rev3_rw_runs_full_precision.csv'), index=False, float_format='%.17g')
    say(f'  {d.algorithm.nunique()} methods x 7 problems x 30 runs; CEO feasible runs per problem: '
        + ', '.join(f'{r.function} {r.feasible}' for r in s[s.algorithm == 'CEO'].itertuples()))
    bad = s[s.feasible < 30]
    say('  cells with fewer than 30 feasible runs: ' + ('; '.join(f'{r.algorithm} {r.function} {r.feasible}' for r in bad.itertuples()) or 'none'))


def part_e():
    say('\n=== (e) M10: timing variation, five repetitions of T0 and T1 on the machine of the complexity study')
    import complexity as Cx
    c = pd.read_csv(os.path.join(HERE, 'results', 'complexity.csv'))
    rows = []
    for D in (10, 20):
        t0s = [Cx.t0() for _ in range(5)]; t1s = [Cx.t1(D) for _ in range(5)]
        rows.append(dict(D=D, T0_mean=np.mean(t0s), T0_sd=np.std(t0s, ddof=1), T1_mean=np.mean(t1s), T1_sd=np.std(t1s, ddof=1)))
        r = rows[-1]; neg = c[(c.D == D) & (c.ratio < 0.5)]
        say(f'  D{D}: T0 {r["T0_mean"]:.3f} s (sd {r["T0_sd"]:.3f}), T1 {r["T1_mean"]:.3f} s (sd {r["T1_sd"]:.3f}); '
            f'one sd of T1 is {r["T1_sd"] / r["T0_mean"]:.2f} in units of T0; smallest ratios: '
            + ', '.join(f'{x.algorithm} {x.ratio:.2f}' for x in neg.itertuples()))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'rev3_timing_jitter.csv'), index=False)


if __name__ == '__main__':
    parts = sys.argv[1:] or ['a', 'b', 'c', 'd', 'e']
    for p in parts:
        globals()[f'part_{p}']()
    open(os.path.join(OUT, 'rev3_summary.txt' if len(parts) == 5 else f'rev3_summary_{"".join(parts)}.txt'), 'w').write('\n'.join(L) + '\n')

# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""Swarm comparison analysis fixed in PROTOCOL_swarm.md. Works on whatever is on disk and reports completeness.
Per benchmark: Friedman over CEO + swarm algorithms on mean value, average ranks (position of CEO),
Holm post-hoc with CEO as control, Wilcoxon rank-sum per function with Holm over functions (W/T/L)."""
import os, sys, glob
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); CEO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results", "analysis"); os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
from swarm_orig import ORDER
L = []


def say(s=""):
    print(s); L.append(str(s))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def analyse(d, fns, tag, value="value"):
    algs = ["CEO"] + [a for a in ORDER if a in set(d.algorithm)]
    M = np.array([[d[(d.algorithm == a) & (d.function == f)][value].mean() for a in algs] for f in fns])
    R = np.vstack([stats.rankdata(r) for r in M]); avg = R.mean(0)
    chi2, p = stats.friedmanchisquare(*M.T); n, k = M.shape
    se = np.sqrt(k * (k + 1) / (6 * n)); pz = 2 * stats.norm.sf(np.abs((avg - avg[0]) / se))
    adj = dict(zip(algs[1:], holm(pz[1:])))
    rows = []
    for a in algs[1:]:
        ps, dirs = [], []
        for f in fns:
            x = d[(d.algorithm == "CEO") & (d.function == f)][value].values
            y = d[(d.algorithm == a) & (d.function == f)][value].values
            ps.append(1.0 if np.array_equal(np.sort(x), np.sort(y)) else stats.ranksums(x, y).pvalue)
            dirs.append(np.sign(np.median(y) - np.median(x)) or np.sign(y.mean() - x.mean()))
        q = holm(ps); dirs = np.array(dirs)
        w = int(((q < .05) & (dirs > 0)).sum()); l = int(((q < .05) & (dirs < 0)).sum())
        rows.append(dict(algorithm=a, avg_rank=avg[algs.index(a)], holm_p=adj[a], win=w, tie=len(fns) - w - l, loss=l,
                         losses=" ".join(f for f, qq, s in zip(fns, q, dirs) if qq < .05 and s < 0)))
    t = pd.DataFrame(rows + [dict(algorithm="CEO", avg_rank=avg[0])]).sort_values("avg_rank").reset_index(drop=True)
    t.insert(0, "position", np.arange(1, len(t) + 1))
    t.to_csv(os.path.join(OUT, f"swarm_{tag}.csv"), index=False)
    pos = int(t[t.algorithm == "CEO"].position.iloc[0])
    say(f"\n=== {tag}: {len(algs)} algorithms x {len(fns)} functions, Friedman chi2={chi2:.1f} p={p:.2g}; CEO position {pos}/{len(algs)}")
    for r in t.itertuples():
        extra = "" if r.algorithm == "CEO" else f"  Holm p={r.holm_p:.3g}  CEO W/T/L {int(r.win)}/{int(r.tie)}/{int(r.loss)} {('lost on ' + r.losses) if r.losses else ''}"
        say(f"  {r.position:2d}. {r.algorithm:11s} {r.avg_rank:5.2f}{extra}")
    wt = t.dropna(subset=["win"]); say(f"  total W/T/L for CEO: {int(wt.win.sum())}/{int(wt.tie.sum())}/{int(wt.loss.sum())}")
    return t


def rw():
    s = pd.read_csv(os.path.join(HERE, "results", "swarm-rw.csv"))
    c = pd.read_csv(os.path.join(HERE, "results", "v4-rw.csv")); c["algorithm"] = "CEO"
    d = pd.concat([s, c], ignore_index=True)
    d["value"] = d.best_fitness.astype(float)
    cnt = d.groupby(["algorithm", "function"]).size()
    assert (cnt == 30).all(), cnt[cnt != 30]
    assert d.duplicated(["algorithm", "function", "seed"]).sum() == 0
    fns = sorted(d.function.unique())
    say(f"CEC2020-RW: {d.algorithm.nunique()} algorithms, {len(fns)} problems, 30 seeds each, complete")
    t = analyse(d, fns, "RW")
    fe = d.groupby(["algorithm", "function"]).feasible.sum().unstack()
    fe.to_csv(os.path.join(OUT, "swarm_RW_feasible.csv"))
    say("  feasible runs (of 30) per problem: " + "; ".join(f"{a} {fe.loc[a].astype(int).tolist()}" for a in t.algorithm))


def cec():
    sys.path.insert(0, HERE)
    import analyze_v4 as A
    v = A.load(10); v = v[v.algorithm.isin(["CEO-v4", "PSO"])].replace({"algorithm": {"CEO-v4": "CEO"}})
    s = [pd.read_csv(f) for f in glob.glob(os.path.join(CEO, "runs/swarm", "out", "**", "swarm-test_D10_*.csv"), recursive=True)]
    s = pd.concat(s).drop_duplicates(["algorithm", "function", "run"]) if s else pd.DataFrame()
    d = pd.concat([v, s], ignore_index=True); d["value"] = d.error
    cnt = d.groupby("algorithm").size()
    say(f"\nCEC2022 D10 runs per algorithm (360 = complete): {cnt.to_dict()}")
    full = [a for a in cnt.index if cnt[a] == 360]
    if len(full) < 20:
        say(f"  incomplete: {20 - len(full)} algorithms not finished; table not produced"); return
    analyse(d, A.FN, "CEC2022_D10")


if __name__ == "__main__":
    rw(); cec()
    open(os.path.join(OUT, "swarm_summary.txt"), "w").write("\n".join(L) + "\n")

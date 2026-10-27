# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""Final analysis fixed in PROTOCOL_power.md (CEO vs MPA, CS, ABC). Read-only on finished runs.
Per suite-dimension: Friedman on mean error, Holm post-hoc with CEO as control, rank-sum per function
with Holm over functions (W/T/L). Primary: Wilcoxon signed-rank on mean error over all 82
function-dimension cases, CEO vs each opponent, Holm over the three; sign-test counts also reported.
Errors below 1e-8 are set to 0 (CEC rule). Writes results/analysis/power_*.csv and power_summary.txt."""
import glob, os, sys
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results", "analysis")
ALGS = ["CEO", "MPA", "CS", "ABC"]; L = []


def say(s=""):
    print(s); L.append(str(s))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def load():
    p = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(ROOT, "runs/power", "out", "*", "results", "power-test_s*.csv"))])
    p = p[["suite", "dim", "algorithm", "function", "run", "error"]]
    v4 = pd.concat([pd.read_csv(f, usecols=["algorithm", "function", "dim", "run", "error"])
                    for f in glob.glob(os.path.join(ROOT, "runs/v4", "out", "ceo-v4-d20-*", "results", "*.csv"))])
    v4 = v4[v4.algorithm == "CEO-v4"].assign(algorithm="CEO", suite="cec2022")
    sw = pd.concat([pd.read_csv(f, usecols=["algorithm", "function", "dim", "run", "error"])
                    for f in glob.glob(os.path.join(ROOT, "runs/swarm", "out", "*", "results", "swarm-test_D10_*.csv"))])
    sw = sw[sw.algorithm.isin(ALGS[1:])].drop_duplicates(["algorithm", "function", "run"])
    c10 = pd.concat([pd.read_csv(f, usecols=["algorithm", "function", "dim", "run", "error"])
                     for f in glob.glob(os.path.join(ROOT, "runs/v4", "out", "ceo-v4-d10-*", "results", "*.csv"))])
    c10 = c10[c10.algorithm == "CEO-v4"].assign(algorithm="CEO")
    d10 = pd.concat([sw, c10]).assign(suite="cec2022")
    d = pd.concat([p, v4, d10], ignore_index=True)
    d.loc[d.error < 1e-8, "error"] = 0.0
    return d


def block(g, tag):
    fns = sorted(g.function.unique(), key=lambda f: int(f[1:]))
    n = g.groupby(["algorithm", "function"]).size()
    assert n.min() == n.max() == 30 and len(n) == 4 * len(fns), (tag, n.min(), n.max(), len(n))
    M = np.array([[g[(g.algorithm == a) & (g.function == f)].error.mean() for a in ALGS] for f in fns])
    R = np.vstack([stats.rankdata(r) for r in M]); avg = R.mean(0)
    chi2, p = stats.friedmanchisquare(*M.T); nn, k = M.shape
    se = np.sqrt(k * (k + 1) / (6 * nn)); pz = 2 * stats.norm.sf(np.abs((avg - avg[0]) / se)); adj = holm(pz[1:])
    say(f"\n=== {tag}: {len(fns)} functions x 30 runs, Friedman chi2={chi2:.2f} p={p:.3g}")
    order = np.argsort(avg)
    say("  avg rank: " + ", ".join(f"{ALGS[i]} {avg[i]:.2f}" for i in order))
    rows = []
    for j, a in enumerate(ALGS[1:], 1):
        ps, dirs = [], []
        for f in fns:
            x = g[(g.algorithm == "CEO") & (g.function == f)].error.values
            y = g[(g.algorithm == a) & (g.function == f)].error.values
            ps.append(1.0 if np.array_equal(np.sort(x), np.sort(y)) else stats.ranksums(x, y).pvalue)
            dirs.append(np.sign(np.median(y) - np.median(x)) or np.sign(y.mean() - x.mean()))
        q = holm(ps); dirs = np.array(dirs)
        w = int(((q < .05) & (dirs > 0)).sum()); l = int(((q < .05) & (dirs < 0)).sum())
        lost = " ".join(f for f, qq, s in zip(fns, q, dirs) if qq < .05 and s < 0)
        say(f"  CEO vs {a:3s}: Friedman-Holm p={adj[j-1]:.3g}  W/T/L {w}/{len(fns)-w-l}/{l}" + (f"  lost on {lost}" if lost else ""))
        rows.append(dict(case=tag, opponent=a, avg_rank_ceo=avg[0], avg_rank_opp=avg[j], friedman_holm_p=adj[j-1], win=w, tie=len(fns)-w-l, loss=l, lost_on=lost))
    mean = pd.DataFrame(M, index=fns, columns=ALGS); mean["case"] = tag
    return rows, mean


def main():
    d = load(); rows, means = [], []
    for (s, dim) in [("cec2022", 10), ("cec2022", 20), ("cec2017", 10), ("cec2017", 30)]:
        r, m = block(d[(d.suite == s) & (d.dim == dim)], f"{s.upper()}_D{dim}"); rows += r; means.append(m)
    C = pd.concat(means)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "power_blocks.csv"), index=False)
    C.to_csv(os.path.join(OUT, "power_mean_error.csv"))
    say(f"\n=== PRIMARY: Wilcoxon signed-rank on mean error over {len(C)} cases, Holm over 3")
    ps, prim = [], []
    for a in ALGS[1:]:
        b = int((C.CEO < C[a]).sum()); w = int((C.CEO > C[a]).sum()); t = int((C.CEO == C[a]).sum())
        p = stats.wilcoxon(C.CEO, C[a]).pvalue; ps.append(p)
        sp = stats.binomtest(b, b + w).pvalue
        prim.append(dict(opponent=a, ceo_better=b, ceo_worse=w, tie=t, wilcoxon_p=p, sign_test_p=sp))
    for r, q in zip(prim, holm(ps)):
        r["wilcoxon_holm_p"] = q
        say(f"  CEO vs {r['opponent']:3s}: better {r['ceo_better']}, worse {r['ceo_worse']}, tie {r['tie']}; "
            f"Wilcoxon p={r['wilcoxon_p']:.3g}, Holm p={q:.3g}; sign test p={r['sign_test_p']:.3g}")
    pd.DataFrame(prim).to_csv(os.path.join(OUT, "power_primary.csv"), index=False)
    say("\nPer-suite signed-rank (secondary, descriptive):")
    for case, g in C.groupby("case"):
        say(f"  {case}: " + "; ".join(f"vs {a} better {int((g.CEO<g[a]).sum())}/worse {int((g.CEO>g[a]).sum())}" for a in ALGS[1:]))
    open(os.path.join(OUT, "power_summary.txt"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()

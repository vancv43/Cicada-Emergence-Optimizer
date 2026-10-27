# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""Re-analysis of finished runs for the reviewer round (no new optimisation).
Per-block view of the pre-registered test (signed-rank and sign test within each suite and dimension),
Vargha-Delaney A12 effect sizes, the CEC2022 D=20 block, and overall Friedman statistics for the tables.
The pooled 82-case result is recomputed and asserted equal to the locked analysis (power_primary.csv).
Output: results/analysis/rev1_b_summary.txt, rev1_b_blocks.csv"""
import os, sys, glob
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import analyze_power as P
import analyze_v4 as V
import manuscript_tables as M

OUT = P.OUT; L = []


def say(s=""):
    print(s); L.append(str(s))


def a12(x, y):
    """Vargha-Delaney A12: probability that a CEO run has a lower error than an opponent run, ties count half."""
    x, y = np.asarray(x)[:, None], np.asarray(y)[None, :]
    return float(((x < y).sum() + 0.5 * (x == y).sum()) / (x.size * y.size))


def load_all():
    d = P.load()
    p50 = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(P.ROOT, "runs/power50", "out", "*", "results", "power50-test_s*.csv")))
    p50 = p50[["suite", "dim", "algorithm", "function", "run", "error"]].copy()
    p50.loc[p50.error < 1e-8, "error"] = 0.0
    return pd.concat([d, p50], ignore_index=True)


def blocks(d):
    rows = []
    for (s, dim) in [("cec2022", 10), ("cec2022", 20), ("cec2017", 10), ("cec2017", 30), ("cec2017", 50)]:
        g = d[(d.suite == s) & (d.dim == dim)]
        m = g.groupby(["function", "algorithm"]).error.mean().unstack()
        for a in P.ALGS[1:]:
            b = int((m.CEO < m[a]).sum()); w = int((m.CEO > m[a]).sum()); t = int((m.CEO == m[a]).sum())
            diff = (m.CEO - m[a]).values
            wp = stats.wilcoxon(m.CEO, m[a]).pvalue if (diff != 0).any() else 1.0
            sp = stats.binomtest(b, b + w).pvalue if b + w else 1.0
            A = [a12(g[(g.function == f) & (g.algorithm == "CEO")].error, g[(g.function == f) & (g.algorithm == a)].error) for f in m.index]
            rows.append(dict(block=f"{s.upper()} D{dim}", opponent=a, n=len(m), better=b, worse=w, tie=t,
                             signed_rank_p=wp, sign_p=sp, a12_median=float(np.median(A)), a12_share_above_071=float(np.mean(np.array(A) >= 0.71))))
    return pd.DataFrame(rows)


def pooled_check(d):
    prim = pd.read_csv(os.path.join(OUT, "power_primary.csv")).set_index("opponent")
    pre = d[~((d.suite == "cec2017") & (d.dim == 50))]
    C = pre.groupby(["suite", "dim", "function", "algorithm"]).error.mean().unstack()
    assert len(C) == 82
    for a in P.ALGS[1:]:
        p = stats.wilcoxon(C.CEO, C[a]).pvalue
        assert np.isclose(p, prim.loc[a, "wilcoxon_p"], rtol=1e-12), (a, p)
    say("pooled 82-case signed-rank p values recomputed and equal to the locked analysis")
    say("sign test over 82 cases: " + ", ".join(f"{a} p={prim.loc[a, 'sign_test_p']:.3g}" for a in P.ALGS[1:]))


def friedman_rw():
    p = pd.read_csv(os.path.join(OUT, "rw_pvalues.csv")); p = p[p.other != "CEO-v1"]
    m = p.pivot_table(index="function", columns="other", values="mean_other")
    m["CEO"] = p.groupby("function").mean_target.first()
    chi2, pv = stats.friedmanchisquare(*m.values.T)
    say(f"Friedman RW, {m.shape[1]} methods x {m.shape[0]} problems: chi2={chi2:.2f} p={pv:.3g}")


def a12_sota():
    for dim in (10, 20):
        d = V.load(dim)
        for a in ["jSO", "L-SHADE"]:
            A = [a12(d[(d.function == f) & (d.algorithm == "CEO-v4")].error, d[(d.function == f) & (d.algorithm == a)].error) for f in V.FN]
            say(f"CEC2022 D{dim} CEO vs {a}: median A12 {np.median(A):.2f}, functions with A12>=0.71: {int((np.array(A) >= 0.71).sum())}, <=0.29: {int((np.array(A) <= 0.29).sum())} of 12")


if __name__ == "__main__":
    d = load_all()
    pooled_check(d)
    B = blocks(d); B.to_csv(os.path.join(OUT, "rev1_b_blocks.csv"), index=False)
    say("\nPer-block view (signed-rank and sign test within the block, uncorrected, descriptive):")
    for r in B.itertuples():
        say(f"  {r.block:12s} vs {r.opponent:3s} n={r.n:2d} better/worse/tie {r.better}/{r.worse}/{r.tie} "
            f"signed-rank p={r.signed_rank_p:.3g} sign p={r.sign_p:.3g} median A12={r.a12_median:.2f} large-effect share={r.a12_share_above_071:.2f}")
    say("")
    friedman_rw()
    for dim in (10, 20):
        _, chi2, pv = V.friedman(V.load(dim), ["CEO-v4"] + M.SOTA)
        say(f"Friedman SOTA D{dim}, 8 methods: chi2={chi2:.2f} p={pv:.3g}")
    a12_sota()
    open(os.path.join(OUT, "rev1_b_summary.txt"), "w").write("\n".join(L) + "\n")

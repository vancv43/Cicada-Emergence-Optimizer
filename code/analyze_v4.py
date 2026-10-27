# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""CEO-v4 analysis fixed in PROTOCOL_CEO_v4.md. Works on whatever is on disk and reports completeness.
CEC2022: v4 runs (runs/v4/out) + earlier baseline study baselines (same harness, reused unchanged).
  - Wilcoxon rank-sum per function, Holm over the 12 functions, v4 vs each algorithm (as the analysis of the earlier baseline study)
  - Friedman on mean errors + Holm post-hoc with CEO-v4 as control
  - ablation: v4 vs each switch; v4 vs CEO-v1
CEC2020-RW: results/v4-rw.csv + the 17 R1 algorithms, paired signed-rank with Holm over opponents (R1 rule).
Writes results/analysis/*.csv and v4_summary.txt."""
import glob, os, sys
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); CEO = os.path.dirname(HERE)
BASEDIR = os.path.join(CEO, "runs", "baselines_cec2022")
OUT = os.path.join(HERE, "results", "analysis"); os.makedirs(OUT, exist_ok=True)
BASE = ["DE", "PSO", "IPOP-CMA-ES", "L-SHADE", "jSO", "GWO", "HHO"]
FN = [f"F{i}" for i in range(1, 13)]
L = []


def say(s=""):
    print(s); L.append(str(s))


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def load(dim):
    v = [pd.read_csv(f) for f in glob.glob(os.path.join(CEO, "runs/v4", "out", "**", f"ceo-v4-test_D{dim}_*.csv"), recursive=True)]
    b = [pd.read_csv(f) for f in glob.glob(os.path.join(BASEDIR, "*", "results", f"baseline-test_D{dim}_*.csv"))]
    v = pd.concat(v).drop_duplicates(["algorithm", "function", "run"]) if v else pd.DataFrame(columns=["algorithm", "function", "run", "seed", "error", "fe"])
    b = pd.concat(b).drop_duplicates(["algorithm", "function", "run"]); b = b[b.algorithm.isin(BASE)]
    # the same official seed rule must have produced the same seeds
    chk = v.merge(b[b.algorithm == "L-SHADE"][["function", "run", "seed"]], on=["function", "run"], suffixes=("", "_b"))
    assert (chk.seed == chk.seed_b).all(), "seed mismatch with the baseline harness"
    return pd.concat([v, b], ignore_index=True)


def ranksum_table(d, target, others):
    rows = []
    for o in others:
        ps, dirs = [], []
        for fn in FN:
            a = d[(d.algorithm == target) & (d.function == fn)].error.values
            c = d[(d.algorithm == o) & (d.function == fn)].error.values
            if len(a) < 30 or len(c) < 30:
                ps.append(np.nan); dirs.append(0); continue
            p = 1.0 if np.array_equal(np.sort(a), np.sort(c)) else stats.ranksums(a, c).pvalue
            ps.append(p); dirs.append(np.sign(np.median(c) - np.median(a)) or np.sign(c.mean() - a.mean()))
        ok = ~np.isnan(ps); adj = np.full(len(ps), np.nan); adj[ok] = holm(np.array(ps)[ok])
        w = int(np.sum((adj < .05) & (np.array(dirs) > 0))); l = int(np.sum((adj < .05) & (np.array(dirs) < 0)))
        rows.append(dict(other=o, n=int(ok.sum()), win=w, tie=int(ok.sum()) - w - l, loss=l,
                         wins=" ".join(f for f, a_, s in zip(FN, adj, dirs) if a_ < .05 and s > 0),
                         losses=" ".join(f for f, a_, s in zip(FN, adj, dirs) if a_ < .05 and s < 0)))
    return pd.DataFrame(rows)


def friedman(d, algs):
    M = np.array([[d[(d.algorithm == a) & (d.function == f)].error.mean() for a in algs] for f in FN])
    R = np.vstack([stats.rankdata(r) for r in M]); avg = R.mean(0)
    chi2, p = stats.friedmanchisquare(*M.T); n, k = M.shape
    se = np.sqrt(k * (k + 1) / (6 * n)); c = algs.index("CEO-v4")
    z = (avg - avg[c]) / se; pz = 2 * stats.norm.sf(np.abs(z)); oth = [i for i in range(k) if i != c]
    adj = dict(zip([algs[i] for i in oth], holm(pz[oth])))
    t = pd.DataFrame(dict(algorithm=algs, avg_rank=avg)).sort_values("avg_rank")
    t["holm_p_vs_v4"] = t.algorithm.map(adj)
    return t, chi2, p


def cec(dim):
    d = load(dim)
    cnt = d.groupby("algorithm").size()
    say(f"\n=== CEC2022 D{dim} (official code): runs per algorithm {cnt.to_dict()}")
    v4algs = [a for a in d.algorithm.unique() if a.startswith(("CEO", "v4-"))]
    if (cnt.get("CEO-v4", 0) < 360):
        say("  CEO-v4 incomplete; partial results only")
    full = [a for a in cnt.index if cnt[a] == 360]
    if "CEO-v4" not in full:
        return
    others = [a for a in full if a != "CEO-v4"]
    t = ranksum_table(d, "CEO-v4", others); t.to_csv(os.path.join(OUT, f"ranksum_D{dim}.csv"), index=False)
    say("  CEO-v4 vs each, rank-sum + Holm over 12 functions (W/T/L):")
    for r in t.itertuples():
        say(f"    {r.other:16s} {r.win}/{r.tie}/{r.loss}  wins[{r.wins}] losses[{r.losses}]")
    comp = [a for a in full if not a.startswith("v4-")]
    fr, chi2, p = friedman(d, comp); fr.to_csv(os.path.join(OUT, f"friedman_D{dim}.csv"), index=False)
    say(f"  Friedman over {len(comp)} algorithms: chi2={chi2:.2f} p={p:.3g}")
    for r in fr.itertuples():
        say(f"    {r.algorithm:16s} {r.avg_rank:.2f}  Holm p vs v4 = {r.holm_p_vs_v4 if not np.isnan(r.holm_p_vs_v4) else '-'}")
    mean = d[d.algorithm.isin(full)].groupby(["function", "algorithm"]).error.mean().unstack()
    mean[sorted(mean.columns, key=lambda a: (a == "CEO-v4", a))].to_csv(os.path.join(OUT, f"mean_error_D{dim}.csv"))


def rw():
    f = os.path.join(HERE, "results", "v4-rw.csv")
    if not os.path.exists(f):
        say("\n=== CEC2020-RW: v4 runs not on disk yet"); return
    sys.path.insert(0, os.path.join(CEO, "experiments", "r1")); sys.path.insert(0, os.path.join(CEO, "experiments"))
    import analyze_r1 as R
    import analyze_fe as A
    r1 = R.load_r1(); e4 = r1[r1.exp == "E4"][["function", "algorithm", "seed", "best_fitness", "feasible"]]
    v = pd.read_csv(f)[["function", "algorithm", "seed", "best_fitness", "feasible"]]
    e4 = pd.concat([e4.replace({"algorithm": {"CEO": "CEO-v1"}}), v], ignore_index=True)
    e4["err"] = np.round(e4.best_fitness.astype(float), 8)
    say(f"\n=== CEC2020-RW RC15-RC21 (R1 harness): {e4.groupby('algorithm').size().to_dict()}")
    algs = sorted(e4.algorithm.unique()); others = [a for a in algs if a != "CEO-v4"]
    c = A.compare(e4, "CEO-v4", others, sorted(e4.function.unique()))
    c.to_csv(os.path.join(OUT, "rw_pvalues.csv"), index=False)
    w, t, l = (c.outcome == "win").sum(), (c.outcome == "tie").sum(), (c.outcome == "loss").sum()
    say(f"  CEO-v4 vs 17 others over 7 problems, signed-rank + Holm over opponents: W/T/L {w}/{t}/{l} of {len(c)}")
    g = c.groupby("other").outcome.value_counts().unstack(fill_value=0)
    for o, r in g.iterrows():
        say(f"    {o:16s} {r.get('win', 0)}/{r.get('tie', 0)}/{r.get('loss', 0)}")
    rk = e4.groupby(["function", "algorithm"]).err.mean().unstack().rank(axis=1).mean().sort_values()
    rk.to_csv(os.path.join(OUT, "rw_avg_rank.csv"))
    say("  average rank: " + ", ".join(f"{a} {x:.2f}" for a, x in rk.items()))
    fe = e4.groupby(["function", "algorithm"]).feasible.sum().unstack()
    say(f"  CEO-v4 feasible runs per problem: {fe['CEO-v4'].astype(int).to_dict()}")


if __name__ == "__main__":
    for dim in (10, 20):
        try:
            cec(dim)
        except AssertionError as e:
            say(f"D{dim}: {e}")
    rw()
    open(os.path.join(OUT, "v4_summary.txt"), "w").write("\n".join(L) + "\n")

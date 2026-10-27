# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""
Analysis of the FE-matched sweeps (results/fe/<suite>/*.csv).

Conventions (fixed before looking at results):
  * error = best_fitness - known optimum; errors are rounded to 1e-8 before
    ranking/testing, so values that agree to 1e-8 count as ties (CEC
    convention of treating errors < 1e-8 as zero).
  * Friedman ranks: per function, algorithms ranked by mean error (average
    ranks for ties), then averaged over functions; Friedman chi-square test
    and Iman-Davenport F statistic on the same rank matrix.
  * Pairwise: two-sided Wilcoxon signed-rank test, paired by seed, CEO vs
    each other algorithm on each function; Holm correction across all
    comparisons made within that function (8 for the nine-algorithm runs,
    14 for the 15-algorithm CEC2022 runs); alpha = 0.05. The direction of a
    significant result comes from the signed rank sums W+ - W- of the same
    test.
Usage: python analyze_fe.py [suite ...]   (default: every suite on disk)
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "fe")
OUT = os.path.join(RES, "analysis")
os.makedirs(OUT, exist_ok=True)

BASELINES = ["PSO", "GA", "DE", "ACOR", "ABC", "GWO", "WOA", "HHO"]
MAIN = ["CEO"] + BASELINES
SOTA = ["L-SHADE", "LSHADE-cnEpSin", "IMODE", "INFO", "HBA", "SCSO"]


def load(suite):
    files = glob.glob(os.path.join(RES, suite, "*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["err"] = np.round(df["error"].astype(float), 8)
    if not np.isfinite(df["err"]).all():
        raise ValueError(f"{suite}: non-finite errors on disk")
    return df


def fn_key(f):
    return (f[0], int(f[1:]))


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    if np.isnan(p).any():
        raise ValueError("NaN p-value: a run produced a non-finite error")
    for k, idx in enumerate(order):
        running = max(running, (m - k) * p[idx])
        adj[idx] = min(running, 1.0)
    return adj


def paired_test(a, b):
    """Two-sided Wilcoxon signed-rank p and a signed direction taken from the
    rank sums the test itself uses (negative = target better), so the reported
    winner always agrees with the test statistic."""
    d = a - b
    nz = d[d != 0]
    if nz.size == 0:
        return 1.0, 0.0
    try:
        p = stats.wilcoxon(a, b, zero_method="wilcox").pvalue
    except ValueError:
        p = 1.0
    r = stats.rankdata(np.abs(nz))
    return p, float(r[nz > 0].sum() - r[nz < 0].sum())


def compare(df, target, others, funcs):
    rows = []
    for f in funcs:
        sub = df[df.function == f]
        ta = sub[sub.algorithm == target].sort_values("seed")
        a = ta["err"].values
        ps, meds = [], []
        for o in others:
            tb = sub[sub.algorithm == o].sort_values("seed")
            if not np.array_equal(ta["seed"].values, tb["seed"].values):
                raise ValueError(f"{f}: {target} and {o} do not share the same seeds")
            b = tb["err"].values
            p, direction = paired_test(a, b)
            ps.append(p)
            meds.append(direction)
        padj = holm(ps)
        for o, p, pa, med in zip(others, ps, padj, meds):
            ma = sub[sub.algorithm == target]["err"].mean()
            mb = sub[sub.algorithm == o]["err"].mean()
            if pa >= 0.05:
                out = "tie"
            else:
                out = "win" if (med < 0 or (med == 0 and ma < mb)) else "loss"
            rows.append(dict(function=f, other=o, p=p, p_holm=pa, outcome=out,
                             mean_target=ma, mean_other=mb))
    return pd.DataFrame(rows)


def friedman(df, algs, funcs):
    M = np.array([[df[(df.function == f) & (df.algorithm == a)]["err"].mean()
                   for a in algs] for f in funcs])
    R = np.vstack([stats.rankdata(r) for r in M])
    avg = R.mean(axis=0)
    chi2, p = stats.friedmanchisquare(*[M[:, j] for j in range(len(algs))])
    n, k = M.shape
    ff = (n - 1) * chi2 / (n * (k - 1) - chi2) if n * (k - 1) != chi2 else np.inf
    p_id = stats.f.sf(ff, k - 1, (k - 1) * (n - 1))  # sf avoids the 1e-16 floor of 1 - cdf
    ranks = pd.DataFrame(dict(algorithm=algs, avg_rank=avg)).sort_values("avg_rank")
    return ranks, dict(chi2=chi2, p=p, iman_davenport=ff, p_id=p_id, n=n, k=k), M


def summarize(suite, algs, tag=""):
    df = load(suite)
    df = df[df.algorithm.isin(algs)]
    funcs = sorted(df.function.unique(), key=fn_key)
    stats_tab = (df.groupby(["function", "algorithm"])["err"]
                   .agg(["mean", "std", "median", "min"]).reset_index())
    fit = (df.groupby(["function", "algorithm"])["best_fitness"]
             .agg(fit_mean="mean", fit_std="std").reset_index())
    stats_tab = stats_tab.merge(fit, on=["function", "algorithm"])
    stats_tab.to_csv(os.path.join(OUT, f"{suite}{tag}_stats.csv"), index=False)
    ranks, ftest, _ = friedman(df, algs, funcs)
    ranks.to_csv(os.path.join(OUT, f"{suite}{tag}_friedman.csv"), index=False)
    cmp = compare(df, "CEO", [a for a in algs if a != "CEO"], funcs)
    cmp.to_csv(os.path.join(OUT, f"{suite}{tag}_wilcoxon.csv"), index=False)
    wtl = cmp.pivot_table(index="other", columns="outcome", values="p",
                          aggfunc="count", fill_value=0)
    for c in ["win", "tie", "loss"]:
        if c not in wtl:
            wtl[c] = 0
    wtl = wtl[["win", "tie", "loss"]].reindex([a for a in algs if a != "CEO"])
    wtl.to_csv(os.path.join(OUT, f"{suite}{tag}_wtl.csv"))
    fe = df.groupby("algorithm")["fe_used"].agg(["min", "mean", "max"])
    fe.to_csv(os.path.join(OUT, f"{suite}{tag}_fe_used.csv"))
    with open(os.path.join(OUT, f"{suite}{tag}_summary.txt"), "w") as fh:
        fh.write(f"suite={suite} functions={len(funcs)} runs/cell="
                 f"{df.groupby(['function','algorithm']).size().min()}\n")
        fh.write(ranks.to_string(index=False) + "\n")
        fh.write(f"Friedman chi2={ftest['chi2']:.3f} p={ftest['p']:.3g} "
                 f"Iman-Davenport F={ftest['iman_davenport']:.3f} p={ftest['p_id']:.3g} "
                 f"(n={ftest['n']}, k={ftest['k']})\n")
        fh.write(wtl.to_string() + "\n")
        fh.write(f"TOTAL W/T/L = {wtl['win'].sum()}/{wtl['tie'].sum()}/{wtl['loss'].sum()}\n")
        fh.write("FE used:\n" + fe.to_string() + "\n")
    print(open(os.path.join(OUT, f"{suite}{tag}_summary.txt")).read())
    return df, funcs


def ablation():
    df = load("classical")
    variants = ["CEO"] + sorted(a for a in df.algorithm.unique() if a.startswith("ABL:")) + ["CEO-v1"]
    df = df[df.algorithm.isin(variants)]
    funcs = sorted(df.function.unique(), key=fn_key)
    ranks, ftest, M = friedman(df, variants, funcs)
    rows = []
    for v in variants[1:]:
        c = compare(df, v, ["CEO"], funcs)
        rows.append(dict(variant=v, better=(c.outcome == "win").sum(),
                         tie=(c.outcome == "tie").sum(),
                         worse=(c.outcome == "loss").sum()))
    t = pd.DataFrame(rows).merge(ranks, left_on="variant", right_on="algorithm", how="left")
    t = t.drop(columns="algorithm")
    t.to_csv(os.path.join(OUT, "ablation.csv"), index=False)
    with open(os.path.join(OUT, "ablation_summary.txt"), "w") as fh:
        fh.write(ranks.to_string(index=False) + "\n")
        fh.write(f"Friedman chi2={ftest['chi2']:.3f} p={ftest['p']:.3g}\n")
        fh.write("variant vs full CEO (Wilcoxon per function, alpha .05): "
                 "better = variant significantly lower error\n")
        fh.write(t.to_string(index=False) + "\n")
    print(open(os.path.join(OUT, "ablation_summary.txt")).read())


if __name__ == "__main__":
    suites = sys.argv[1:] or [s for s in ["classical", "scal50", "scal100", "cec2017",
                                          "cec2022_d10", "cec2022_d20"]
                              if os.path.isdir(os.path.join(RES, s))]
    for s in suites:
        summarize(s, MAIN)
        if s.startswith("cec2022"):
            summarize(s, MAIN + SOTA, tag="_all15")
        if s == "classical":
            ablation()

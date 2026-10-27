# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""Sensitivity analysis fixed in PROTOCOL_sensitivity.md. Read-only on finished runs.
Per parameter: Friedman over its five levels on mean error across CEC2017 D10 F1, F3-F30, mean ranks,
and per level the number of functions better/worse than the frozen level (rank-sum, Holm over functions).
Also checks that the frozen configuration reproduces the power-study CEO runs.
Writes results/analysis/sens_*.csv and sens_summary.txt."""
import glob, os
import numpy as np, pandas as pd
from scipy import stats
import analyze_power as A

FROZEN = {"N0f": "11", "n_b": "5", "E": "78.43", "theta": "4", "c_micro": "0.273", "p": "0.198"}


def main():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(A.ROOT, "runs/sens", "out", "*", "results", "sens-test_s*.csv"))])
    assert d.duplicated(["config", "function", "run"]).sum() == 0 and d.error.notna().all() and (d.fe == 100_000).all()
    n = d.groupby(["config", "function"]).size(); assert n.min() == n.max() == 30, (n.min(), n.max())
    A.say(f"configs {d.config.nunique()}, functions {d.function.nunique()}, rows {len(d)}; every cell 30 runs")
    d.loc[d.error < 1e-8, "error"] = 0.0
    fns = sorted(d.function.unique(), key=lambda f: int(f[1:]))
    frozen = d[d.param == "frozen"] if "frozen" in set(d.param) else None
    p = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(A.ROOT, "runs/power", "out", "*", "results", "power-test_s*.csv"))])
    p = p[(p.suite == "cec2017") & (p.dim == 10) & (p.algorithm == "CEO")].copy(); p.loc[p.error < 1e-8, "error"] = 0.0
    rows, means = [], []
    for par in [k for k in FROZEN]:
        g = d[d.param == par]
        fz = g[g.config == f"{par}={FROZEN[par]}"]
        if fz.empty and frozen is not None:
            g = pd.concat([g, frozen.assign(param=par, config=f"{par}={FROZEN[par]}")]); fz = g[g.config == f"{par}={FROZEN[par]}"]
        levels = sorted(g.config.unique(), key=lambda c: float(c.split("=")[1]))
        M = np.array([[g[(g.config == c) & (g.function == f)].error.mean() for c in levels] for f in fns])
        R = np.vstack([stats.rankdata(r) for r in M]); avg = R.mean(0)
        chi2, pf = stats.friedmanchisquare(*M.T)
        A.say(f"\n=== {par}: Friedman chi2={chi2:.2f} p={pf:.3g}")
        for c, r in zip(levels, avg):
            x = fz.sort_values(["function", "run"]); y = g[g.config == c]
            if c == fz.config.iloc[0]:
                b = w = 0
            else:
                ps, dirs = [], []
                for f in fns:
                    xf = x[x.function == f].error.values; yf = y[y.function == f].error.values
                    ps.append(1.0 if np.array_equal(np.sort(xf), np.sort(yf)) else stats.ranksums(xf, yf).pvalue)
                    dirs.append(np.sign(np.median(yf) - np.median(xf)) or np.sign(yf.mean() - xf.mean()))
                q = A.holm(ps); dirs = np.array(dirs)
                b = int(((q < .05) & (dirs < 0)).sum()); w = int(((q < .05) & (dirs > 0)).sum())
            tag = " (frozen)" if c == fz.config.iloc[0] else ""
            A.say(f"  {c:14s} mean rank {r:.2f}  vs frozen: better on {b}, worse on {w}{tag}")
            rows.append(dict(param=par, config=c, frozen=bool(tag), mean_rank=r, friedman_p=pf, better_than_frozen=b, worse_than_frozen=w))
        m = pd.DataFrame(M, index=fns, columns=levels); m["param"] = par; means.append(m)
    pd.DataFrame(rows).to_csv(os.path.join(A.OUT, "sens_table.csv"), index=False)
    pd.concat(means).to_csv(os.path.join(A.OUT, "sens_mean_error.csv"))
    # reproducibility: frozen config vs power-study CEO, same seeds
    fz = d[d.config.isin([f"{k}={v}" for k, v in FROZEN.items()]) | (d.param == "frozen")].drop_duplicates(["function", "run"])
    m = fz.merge(p, on=["function", "run"], suffixes=("_s", "_p"))
    same = np.isclose(m.error_s, m.error_p, rtol=1e-9, atol=0)
    A.say(f"\nReproducibility: frozen vs power-study CEO CEC2017 D10, {len(m)} matched runs, identical {same.sum()}, max |diff| {np.abs(m.error_s - m.error_p).max():.3g}")
    open(os.path.join(A.OUT, "sens_summary.txt"), "w").write("\n".join(A.L) + "\n")


if __name__ == "__main__":
    main()

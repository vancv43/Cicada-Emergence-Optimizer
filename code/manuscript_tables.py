# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""Numbers for the manuscript tables, computed from the finished runs only (no new optimisation).
Uses the analysis functions fixed in PROTOCOL_CEO_v4.md (analyze_v4) and the swarm analysis CSVs.
Published comparison methods only: the superseded first CEO design is left out of the manuscript tables,
and the Friedman ranks are recomputed without it.
Writes results/analysis/manuscript_numbers.txt and results/analysis/tab_*.tex."""
import os
import numpy as np, pandas as pd
import analyze_v4 as A

OUT = A.OUT
SOTA = ["jSO", "L-SHADE", "DE", "IPOP-CMA-ES", "PSO", "GWO", "HHO"]
ABL = ["v4-noGate", "v4-noMacro", "v4-sharedMem", "v4-noEmergence"]
LOG = []


def say(s=""):
    print(s); LOG.append(str(s))


def fmt(x):
    if x == 0:
        return "0"
    return f"{x:.2e}".replace("e+00", "").replace("e-0", "e-").replace("e+0", "e+")


def sota(dim):
    d = A.load(dim)
    algs = ["CEO-v4"] + SOTA
    fr, chi2, p = A.friedman(d, algs)
    rs = A.ranksum_table(d, "CEO-v4", SOTA).set_index("other")
    say(f"\nSOTA D{dim}: Friedman over {len(algs)} chi2={chi2:.2f} p={p:.2g}")
    rows = []
    for pos, r in enumerate(fr.itertuples(), 1):
        a = r.algorithm
        wtl = "" if a == "CEO-v4" else f"{rs.loc[a, 'win']}/{rs.loc[a, 'tie']}/{rs.loc[a, 'loss']}"
        hp = "" if a == "CEO-v4" else f"{r.holm_p_vs_v4:.3g}"
        say(f"  {pos}. {a:12s} {r.avg_rank:.2f} holm={hp} W/T/L={wtl}")
        rows.append((pos, a, r.avg_rank, hp, wtl))
    st = d.groupby(["function", "algorithm"]).error.agg(["mean", "std"])
    return rows, st, d


def ablation():
    d = A.load(10)
    rs = A.ranksum_table(d, "CEO-v4", ABL).set_index("other")
    st = d.groupby(["function", "algorithm"]).error.agg(["mean", "std"])
    say("\nAblation D10 (CEO vs variant, rank-sum + Holm): " +
        ", ".join(f"{a} {rs.loc[a, 'win']}/{rs.loc[a, 'tie']}/{rs.loc[a, 'loss']}" for a in ABL))
    return rs, st


def rw_sota():
    p = pd.read_csv(os.path.join(OUT, "rw_pvalues.csv"))
    p = p[p.other != "CEO-v1"]
    g = p.groupby("other").outcome.value_counts().unstack(fill_value=0)
    # average rank on the mean penalised objective without the superseded design
    m = p.pivot_table(index="function", columns="other", values="mean_other")
    m["CEO"] = p.groupby("function").mean_target.first()
    rk = m.rank(axis=1).mean().sort_values()
    say(f"\nRW vs {len(m.columns) - 1} published methods: W/T/L {(p.outcome == 'win').sum()}/{(p.outcome == 'tie').sum()}/{(p.outcome == 'loss').sum()}")
    for pos, (a, x) in enumerate(rk.items(), 1):
        w = "" if a == "CEO" else f"{g.loc[a].get('win', 0)}/{g.loc[a].get('tie', 0)}/{g.loc[a].get('loss', 0)}"
        say(f"  {pos}. {a:15s} {x:.2f} {w}")
    return rk, g


if __name__ == "__main__":
    r10, s10, _ = sota(10)
    r20, s20, _ = sota(20)
    rs_ab, s_ab = ablation()
    rk, g = rw_sota()
    tops = ["jSO", "L-SHADE", "DE", "IPOP-CMA-ES", "CEO-v4"]
    say("\nMean (std) at D10 / D20 for " + ", ".join(tops))
    for fn in A.FN:
        say(f"  {fn}: " + " | ".join(f"{a} {fmt(s10.loc[(fn, a), 'mean'])} ({fmt(s10.loc[(fn, a), 'std'])}) / "
                                      f"{fmt(s20.loc[(fn, a), 'mean'])} ({fmt(s20.loc[(fn, a), 'std'])})" for a in tops))
    say("\nAblation mean (std) D10")
    for fn in A.FN:
        say(f"  {fn}: " + " | ".join(f"{a} {fmt(s_ab.loc[(fn, a), 'mean'])} ({fmt(s_ab.loc[(fn, a), 'std'])})" for a in ABL + ["CEO-v4"]))
    open(os.path.join(OUT, "manuscript_numbers.txt"), "w").write("\n".join(LOG) + "\n")

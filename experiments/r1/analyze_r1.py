# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""
R1 analysis (PROTOCOL_R1.md). Works on whatever R1 chunks are on disk and says
how complete each part is; rerun after the clean-up round for final numbers.

Reads runs/r1/out/**/R1__*.csv, de-duplicates (exp, fn, cfg, seed) keeping
the first file found, and compares against the frozen results in results/fe/.
Statistics follow analyze_fe.py (errors rounded to 1e-8, two-sided Wilcoxon,
Holm within a function). Writes results/r1/analysis/*.csv + r1_summary.txt.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)
import analyze_fe as A  # noqa: E402

PKG = os.path.join(os.path.dirname(EXP), "runs/r1", "out")
OUT = os.path.join(EXP, "results", "r1", "analysis")
os.makedirs(OUT, exist_ok=True)
LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


def load_r1():
    files = sorted(glob.glob(os.path.join(PKG, "**", "R1__*.csv"), recursive=True))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(["exp", "function", "algorithm", "seed"], keep="first")
    df["err"] = np.round(df["error"].astype(float), 8)
    return df


def frozen(folder, label_map=None):
    files = glob.glob(os.path.join(EXP, "results", "fe", folder, "*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["err"] = np.round(df["error"].astype(float), 8)
    return df[["function", "algorithm", "seed", "err"]]


def complete(df):
    """keep (function, algorithm) cells with all 30 seeds"""
    n = df.groupby(["function", "algorithm"]).seed.transform("nunique")
    return df[n == 30]


def btw(df, target, ref, funcs):
    """better/tie/worse of target vs ref per function (uncorrected), plus Holm over functions."""
    if not funcs:
        return dict(n=0)
    c = A.compare(df, target, [ref], funcs)
    c["p_holm_fn"] = A.holm(list(c.p))
    sig = c[c.p_holm_fn < 0.05]
    return dict(n=len(funcs), better=int((c.outcome == "win").sum()), tie=int((c.outcome == "tie").sum()),
                worse=int((c.outcome == "loss").sum()),
                better_holm=int((sig.outcome == "win").sum()), worse_holm=int((sig.outcome == "loss").sum()),
                fns_better=" ".join(c[c.outcome == "win"].function), fns_worse=" ".join(c[c.outcome == "loss"].function))


def main():
    r1 = load_r1()
    say(f"R1 rows {len(r1)}, chunks files {len(glob.glob(os.path.join(PKG, '**', 'R1__*.csv'), recursive=True))}")

    # ---- A0: reproducibility anchor (E1 CEO in the batch runs vs frozen local CEC2022 D10)
    e1 = complete(r1[r1.exp == "E1"])
    ref22 = frozen("cec2022_d10")
    a = e1[e1.algorithm == "CEO"].merge(ref22[ref22.algorithm == "CEO"], on=["function", "seed"], suffixes=("_k", "_l"))
    if len(a):
        d = (a.err_k - a.err_l).abs().groupby(a.function).max()
        say(f"A0 anchor: CEO re-run vs frozen on {d.size} CEC2022 functions, max |diff| per fn: "
            + ", ".join(f"{k}={v:.3g}" for k, v in d.items()))

    # ---- A1: E1 ablation on CEC2022 D10 (post-hoc)
    say("\nA1 E1 ablation vs full CEO, CEC2022 D10 (complete functions only)")
    rows = []
    for v in sorted(x for x in e1.algorithm.unique() if x != "CEO"):
        fs = sorted(set(e1[e1.algorithm == v].function) & set(e1[e1.algorithm == "CEO"].function))
        r = dict(variant=v, **btw(e1, v, "CEO", fs))
        rows.append(r)
        say(f"  {v:28s} n={r['n']:2d} better/tie/worse={r.get('better')}/{r.get('tie')}/{r.get('worse')} "
            f"holm-fn b/w={r.get('better_holm')}/{r.get('worse_holm')}  worse:[{r.get('fns_worse', '')}] better:[{r.get('fns_better', '')}]")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "E1_ablation_cec2022_d10.csv"), index=False)

    # ---- A2: E2 sensitivity on classical (vs batch-run CEO anchor, else frozen local CEO)
    say("\nA2 E2 sensitivity vs CEO anchor re-run in the same batch environment, classical suite (complete functions only)")
    e2 = complete(r1[r1.exp == "E2"])
    loc = frozen("classical")
    base = loc[loc.algorithm == "CEO"]
    # compare within one environment: the batch-run CEO anchor, not the frozen runs
    e2b = e2[["function", "algorithm", "seed", "err"]]
    rows = []
    for v in sorted(x for x in e2.algorithm.unique() if x != "CEO"):
        fs = sorted(set(e2[e2.algorithm == v].function) & set(e2[e2.algorithm == "CEO"].function), key=A.fn_key)
        r = dict(config=v, **btw(e2b, v, "CEO", fs))
        rows.append(r)
        say(f"  {v:16s} n={r['n']:2d} b/t/w={r.get('better')}/{r.get('tie')}/{r.get('worse')} holm-fn b/w={r.get('better_holm')}/{r.get('worse_holm')}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "E2_sensitivity_classical.csv"), index=False)
    ck = e2[e2.algorithm == "CEO"].merge(base, on=["function", "seed"], suffixes=("_k", "_l"))
    if len(ck):
        say(f"  anchor CEO classical re-run vs frozen: max |diff| {float((ck.err_k - ck.err_l).abs().max()):.3g} on {ck.function.nunique()} fns")

    # ---- A3: E3 restart baselines vs CEO, Holm over all others per function
    say("\nA3 E3 restart baselines (CEO W/T/L vs each, Holm over all algorithms per function; complete functions only)")
    suites = {"E3cl": ("classical", ["CEO", "PSO", "GA", "DE", "ACOR", "ABC", "GWO", "WOA", "HHO"]),
              "E3c17": ("cec2017", ["CEO", "PSO", "GA", "DE", "ACOR", "ABC", "GWO", "WOA", "HHO"]),
              "E3c22": ("cec2022_d10", None)}
    rows = []
    for exp, (folder, keep) in suites.items():
        new = complete(r1[r1.exp == exp])[["function", "algorithm", "seed", "err"]]
        old = frozen(folder)
        if keep:
            old = old[old.algorithm.isin(keep)]
        fs = sorted(set(new[new.algorithm == "BIPOP-CMA-ES"].function) & set(new[new.algorithm == "Sawtooth-GA"].function), key=A.fn_key)
        if not fs:
            say(f"  {exp}: no function complete for both restart baselines yet")
            continue
        df = pd.concat([old[old.function.isin(fs)], new[new.function.isin(fs)]], ignore_index=True)
        algs = sorted(df.algorithm.unique())
        others = [a for a in algs if a != "CEO"]
        c = A.compare(df, "CEO", others, fs)
        ranks, ft, _ = A.friedman(df, algs, fs)
        for b in ("BIPOP-CMA-ES", "Sawtooth-GA"):
            cb = c[c.other == b]
            w, t, l = (cb.outcome == "win").sum(), (cb.outcome == "tie").sum(), (cb.outcome == "loss").sum()
            rows.append(dict(suite=exp, baseline=b, n_fn=len(fs), ceo_win=w, tie=t, ceo_loss=l))
        rk = ranks.set_index("algorithm").avg_rank.sort_values()
        say(f"  {exp} on {len(fs)} fns ({' '.join(fs)}):")
        say("     avg rank: " + ", ".join(f"{k} {v:.2f}" for k, v in rk.items()))
        for r in rows[-2:]:
            say(f"     CEO vs {r['baseline']}: {r['ceo_win']}/{r['tie']}/{r['ceo_loss']}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "E3_restart_baselines.csv"), index=False)

    # ---- A4: E4 real-world constrained
    say("\nA4 E4 CEC2020-RW (complete cells only): mean best penalized value, feasible runs/30")
    e4 = complete(r1[r1.exp == "E4"])
    if len(e4):
        t = e4.groupby(["function", "algorithm"]).agg(mean=("best_fitness", "mean"), std=("best_fitness", "std"),
                                                      feasible=("feasible", "sum")).reset_index()
        t.to_csv(os.path.join(OUT, "E4_realworld.csv"), index=False)
        for fn, g in t.groupby("function"):
            g = g.sort_values("mean")
            say(f"  {fn}: " + "; ".join(f"{r.algorithm} {r['mean']:.6g} ({int(r.feasible)})" for _, r in g.iterrows()))
    open(os.path.join(OUT, "r1_summary.txt"), "w").write("\n".join(LINES) + "\n")


if __name__ == "__main__":
    main()

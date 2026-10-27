# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""Supplementary CEC2017 D50 analysis (PROTOCOL_power.md, Deviations 2026-10-21). Read-only on finished runs.
Same per-suite procedure as analyze_power.block, plus Wilcoxon signed-rank on mean error over the 29 D50
functions, Holm over the three opponents. Not pooled with the 82-case primary test.
Writes results/analysis/power50_*.csv and power50_summary.txt."""
import glob, os
import numpy as np, pandas as pd
from scipy import stats
import analyze_power as A


def main():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(A.ROOT, "runs/power50", "out", "*", "results", "power50-test_s*.csv"))])
    assert d.duplicated(["algorithm", "function", "run"]).sum() == 0 and d.error.notna().all()
    assert (d.fe == 500_000).all() and set(d.dim) == {50}
    d.loc[d.error < 1e-8, "error"] = 0.0
    rows, C = A.block(d, "CEC2017_D50")
    pd.DataFrame(rows).to_csv(os.path.join(A.OUT, "power50_blocks.csv"), index=False)
    C.to_csv(os.path.join(A.OUT, "power50_mean_error.csv"))
    A.say(f"\n=== SUPPLEMENTARY: Wilcoxon signed-rank on mean error over {len(C)} D50 functions, Holm over 3")
    ps, prim = [], []
    for a in A.ALGS[1:]:
        b = int((C.CEO < C[a]).sum()); w = int((C.CEO > C[a]).sum()); t = int((C.CEO == C[a]).sum())
        p = stats.wilcoxon(C.CEO, C[a]).pvalue; ps.append(p)
        prim.append(dict(opponent=a, ceo_better=b, ceo_worse=w, tie=t, wilcoxon_p=p, sign_test_p=stats.binomtest(b, b + w).pvalue))
    for r, q in zip(prim, A.holm(ps)):
        r["wilcoxon_holm_p"] = q
        A.say(f"  CEO vs {r['opponent']:3s}: better {r['ceo_better']}, worse {r['ceo_worse']}, tie {r['tie']}; "
              f"Wilcoxon p={r['wilcoxon_p']:.3g}, Holm p={q:.3g}; sign test p={r['sign_test_p']:.3g}")
    pd.DataFrame(prim).to_csv(os.path.join(A.OUT, "power50_primary.csv"), index=False)
    open(os.path.join(A.OUT, "power50_summary.txt"), "w").write("\n".join(A.L) + "\n")


if __name__ == "__main__":
    main()

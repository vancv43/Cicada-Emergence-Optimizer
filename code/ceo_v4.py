# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""
CEO-v4: brood-scheduled, progress-gated re-diversification around a greedy
current-to-pbest/1 core with archive, brood-specific success-history memories
and linear population size reduction. Definition: PROTOCOL_CEO_v4.md.

Every runner takes (obj, lb, ub, dim, seed, params) like the baselines of the earlier study;
`obj` is budget.Objective, which raises BudgetExhausted at evaluation B+1.

Switches for the ablation (all default True):
  gate          False -> a brood emerges at every clock gate, progressing or not
  macro         False -> micro emergence only, never a brood restart
  emergence     False -> no emergence at all (a brood-memory L-SHADE)
  brood_memory  False -> one success-history memory shared by all broods
"""
import math

import numpy as np

from budget import BudgetExhausted

SPREAD = [0.55, 0.75, 1.0, 1.35, 1.7, 2.1, 2.6, 3.2]
DEFAULTS = dict(N0f=18, n_b=4, E=50.0, theta=2, c_micro=1.0, p=0.11)
H, ARC_RATE, BETA = 6, 2.6, 1.5
_SIG_U = (math.gamma(1 + BETA) * math.sin(math.pi * BETA / 2)
          / (math.gamma((1 + BETA) / 2) * BETA * 2 ** ((BETA - 1) / 2))) ** (1 / BETA)


def levy(rng, shape):
    """Mantegna Levy steps, clipped to [-3, 3] as in CEO-v1."""
    u = rng.normal(0.0, _SIG_U, shape)
    v = rng.normal(0.0, 1.0, shape)
    return np.clip(u / np.abs(v) ** (1 / BETA), -3.0, 3.0)


def brood_periods(n_b, P0):
    """Distinct integer periods spread around P0 (no primality, see protocol)."""
    out, used = [], set()
    for s in SPREAD[:n_b]:
        c = max(2, int(round(s * P0)))
        while c in used:
            c += 1
        used.add(c)
        out.append(c)
    return np.array(out, dtype=int)


def cauchy_F(rng, mu):
    """SHADE rule: Cauchy(mu, 0.1), redrawn while <= 0, capped at 1."""
    F = mu + 0.1 * np.tan(np.pi * (rng.random(mu.shape) - 0.5))
    bad = F <= 0
    while bad.any():
        F[bad] = mu[bad] + 0.1 * np.tan(np.pi * (rng.random(int(bad.sum())) - 0.5))
        bad = F <= 0
    return np.minimum(F, 1.0)


def _repair(Y, centre, lo, hi):
    """Midpoint repair toward `centre`, as in L-SHADE."""
    Y = np.where(Y < lo, (lo + centre) / 2, Y)
    return np.where(Y > hi, (hi + centre) / 2, Y)


def _eval(obj, Y, fit, rows):
    for j, i in enumerate(rows):
        fit[i] = obj(Y[j])


def ceo_v4(obj, lb, ub, dim, seed, params=None, gate=True, macro=True,
           emergence=True, brood_memory=True, callback=None, equal_periods=False, moult="levy"):
    p = {**DEFAULTS, **(params or {})}
    rng = np.random.default_rng(seed)
    lo, hi = np.full(dim, lb, float), np.full(dim, ub, float)
    maxfe = obj.max_fe
    n_b, theta, c_micro, pfrac = int(p["n_b"]), int(p["theta"]), float(p["c_micro"]), float(p["p"])
    N0 = max(int(round(p["N0f"] * dim)), 2 * n_b, 6)
    Nmin = max(4, n_b)
    P0 = max(5, int(round(maxfe / ((N0 + Nmin) / 2) / float(p["E"]))))
    P = np.full(n_b, P0, dtype=int) if equal_periods else brood_periods(n_b, P0)
    phi = rng.integers(0, P)

    X = rng.uniform(lo, hi, (N0, dim))
    fit = np.full(N0, np.inf)
    st = dict(generations=0, micro=0, macro=0, gate_skips=0, P=P.tolist())
    try:
        _eval(obj, X, fit, range(N0))
        brood = np.arange(N0) % n_b
        n_mem = n_b if brood_memory else 1
        MF = np.full((n_mem, H), 0.5)
        MCR = np.full((n_mem, H), 0.5)
        kptr = np.zeros(n_mem, dtype=int)
        A = np.empty((0, dim))
        gate_best = np.array([fit[brood == k].min() for k in range(n_b)])
        s = np.zeros(n_b, dtype=int)
        g = 0
        while True:
            ev = []                                  # observe-only record for `callback`
            n = len(X)
            # ---- clock gates: which broods emerge this generation
            emerging = np.zeros(n_b, dtype=bool)
            if emergence and g > 0:
                for k in np.where((g - phi) % P == 0)[0]:
                    bb = fit[brood == k].min()
                    if bb < gate_best[k]:          # brood still progressing
                        s[k] = 0
                        emerging[k] = not gate
                        st["gate_skips"] += int(gate)
                    else:
                        s[k] += 1
                        emerging[k] = True
                    gate_best[k] = bb

            # ---- nymph phase: current-to-pbest/1/bin with archive, greedy selection
            idx = np.where(~emerging[brood])[0]
            m = idx.size
            if m:
                mi = brood[idx] if brood_memory else np.zeros(m, dtype=int)
                r = rng.integers(0, H, m)
                mcr = MCR[mi, r]
                CR = np.where(np.isnan(mcr), 0.0, np.clip(rng.normal(np.nan_to_num(mcr), 0.1), 0.0, 1.0))
                F = cauchy_F(rng, MF[mi, r])
                order = np.argsort(fit)
                npb = max(2, int(round(pfrac * n)))
                pb = order[rng.integers(0, npb, m)]
                r1 = rng.integers(0, n - 1, m)
                r1 += r1 >= idx
                PA = np.vstack([X, A]) if len(A) else X
                r2 = rng.integers(0, len(PA), m)
                clash = (r2 == idx) | (r2 == r1)
                while clash.any():
                    r2[clash] = rng.integers(0, len(PA), int(clash.sum()))
                    clash = (r2 == idx) | (r2 == r1)
                Xi = X[idx]
                V = Xi + F[:, None] * (X[pb] - Xi) + F[:, None] * (X[r1] - PA[r2])
                V = _repair(V, Xi, lo, hi)
                mask = rng.random((m, dim)) < CR[:, None]
                mask[np.arange(m), rng.integers(0, dim, m)] = True
                U = np.where(mask, V, Xi)
                fu = np.empty(m)
                for j in range(m):
                    fu[j] = obj(U[j])
                fpar = fit[idx]
                better, ok = fu < fpar, fu <= fpar
                if better.any():
                    A = np.vstack([A, Xi[better]])
                    df = np.abs(fpar[better] - fu[better])
                    mb, Fb, CRb = mi[better], F[better], CR[better]
                    for mg in np.unique(mb):
                        sel = mb == mg
                        w = df[sel] / df[sel].sum() if df[sel].sum() > 0 else np.full(sel.sum(), 1 / sel.sum())
                        MF[mg, kptr[mg]] = np.sum(w * Fb[sel] ** 2) / np.sum(w * Fb[sel])
                        if np.isnan(MCR[mg, kptr[mg]]) or CRb[sel].max() == 0:
                            MCR[mg, kptr[mg]] = np.nan
                        else:
                            MCR[mg, kptr[mg]] = np.sum(w * CRb[sel] ** 2) / np.sum(w * CRb[sel])
                        kptr[mg] = (kptr[mg] + 1) % H
                X[idx[ok]] = U[ok]
                fit[idx[ok]] = fu[ok]

            # ---- emergence: micro around the brood best, macro restart after theta failures
            for k in np.where(emerging)[0]:
                mem = np.where(brood == k)[0]
                ib = mem[np.argmin(fit[mem])]
                ig = int(np.argmin(fit))
                if macro and s[k] > theta:
                    re = mem[mem != ig]
                    if re.size:
                        A = np.vstack([A, X[re]])
                        X[re] = rng.uniform(lo, hi, (re.size, dim))
                        _eval(obj, X[re], fit, re)
                    if brood_memory:
                        MF[k], MCR[k], kptr[k] = 0.5, 0.5, 0
                    phi[k] = rng.integers(0, P[k])
                    s[k] = 0
                    st["macro"] += 1
                    ev.append((k, "macro"))
                else:
                    re = mem[fit[mem] > np.median(fit[mem])]
                    if re.size == 0 and mem.size > 1:
                        re = mem[mem != ib][:1]
                    if re.size:
                        sig = X[mem].std(axis=0) if mem.size >= 2 else X.std(axis=0)
                        L = levy(rng, (re.size, dim)) if moult == "levy" else np.clip(rng.normal(0.0, 1.0, (re.size, dim)), -3.0, 3.0)
                        Y = _repair(X[ib] + c_micro * sig * L, X[ib], lo, hi)
                        A = np.vstack([A, X[re]])
                        X[re] = Y
                        _eval(obj, Y, fit, re)
                    st["micro"] += 1
                    ev.append((k, "micro"))
                gate_best[k] = fit[mem].min()

            # ---- linear population size reduction by evaluations, every brood keeps one member
            n_new = max(Nmin, int(round(N0 - (N0 - Nmin) * obj.n / maxfe)))
            if n_new < len(X):
                counts = np.bincount(brood, minlength=n_b)
                ig = int(np.argmin(fit))
                drop = []
                for i in np.argsort(fit)[::-1]:
                    if len(drop) == len(X) - n_new:
                        break
                    if i != ig and counts[brood[i]] > 1:
                        drop.append(i)
                        counts[brood[i]] -= 1
                keep = np.setdiff1d(np.arange(len(X)), drop)
                X, fit, brood = X[keep], fit[keep], brood[keep]
            amax = int(round(ARC_RATE * len(X)))
            if len(A) > amax:
                A = A[rng.choice(len(A), amax, replace=False)]
            if callback is not None:                 # observation only, never alters the search
                callback(g, obj.n, X, fit, brood, emerging, ev)
            g += 1
            st["generations"] = g
    except BudgetExhausted:
        pass
    return st


def _guard(fn):
    def wrapped(obj, lb, ub, dim, seed, params=None):
        try:
            fn(obj, lb, ub, dim, seed, params)
        except BudgetExhausted:
            pass
        return obj.best
    return wrapped


def run_ceo_v1(obj, lb, ub, dim, seed, params=None):
    """Frozen CEO-v1 (ceo_ablation.ceo_final), N=30, horizon from fe_budget.max_iter_for."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
    from ceo_ablation import ceo_final
    from fe_budget import max_iter_for
    func = lambda Xb: np.array([obj(x) for x in np.atleast_2d(Xb)])
    ceo_final(func, lb, ub, dim, 30, max_iter_for("CEO", obj.max_fe, 30), seed=seed % (2 ** 32))


RUNNERS = {
    "CEO-v4": _guard(lambda o, lb, ub, d, s, p: ceo_v4(o, lb, ub, d, s, p)),
    "v4-noGate": _guard(lambda o, lb, ub, d, s, p: ceo_v4(o, lb, ub, d, s, p, gate=False)),
    "v4-noMacro": _guard(lambda o, lb, ub, d, s, p: ceo_v4(o, lb, ub, d, s, p, macro=False)),
    "v4-noEmergence": _guard(lambda o, lb, ub, d, s, p: ceo_v4(o, lb, ub, d, s, p, emergence=False)),
    "v4-sharedMem": _guard(lambda o, lb, ub, d, s, p: ceo_v4(o, lb, ub, d, s, p, brood_memory=False)),
    "CEO-v1": _guard(run_ceo_v1),
    # replacement controls for the reviewer round (PROTOCOL_power.md, Deviations 2026-10-22)
    "v4-K1": _guard(lambda o, lb, ub, d, s, p: ceo_v4(o, lb, ub, d, s, {**(p or {}), "n_b": 1}, equal_periods=True)),
    "v4-equalPeriods": _guard(lambda o, lb, ub, d, s, p: ceo_v4(o, lb, ub, d, s, p, equal_periods=True)),
    "v4-gaussMoult": _guard(lambda o, lb, ub, d, s, p: ceo_v4(o, lb, ub, d, s, p, moult="gauss")),
    "v4-restart": _guard(lambda o, lb, ub, d, s, p: ceo_v4(o, lb, ub, d, s, {**(p or {}), "n_b": 1, "theta": 0}, equal_periods=True)),
    "CEO-N0fixed": _guard(lambda o, lb, ub, d, s, p: ceo_v4(o, lb, ub, d, s, {**(p or {}), "N0f": 110.0 / d})),
}


def _self_test():
    """Budget respected, determinism per seed, every switch runs, f(o)=0 reachable on sphere."""
    from budget import Objective
    from tuning_suite import TuningFunction
    f = TuningFunction(0, 10, 1)
    res = {}
    for name, run in RUNNERS.items():
        o1 = Objective(f, 10, 20000, 0.0)
        o2 = Objective(f, 10, 20000, 0.0)
        b1, b2 = run(o1, -100., 100., 10, 7, None), run(o2, -100., 100., 10, 7, None)
        assert o1.n == 20000, (name, o1.n)
        assert b1 == b2, (name, b1, b2)
        res[name] = b1
    st = ceo_v4(Objective(f, 10, 20000, 0.0), -100., 100., 10, 7)
    assert st["micro"] + st["gate_skips"] > 0, st
    print("self-test OK:", {k: f"{v:.3g}" for k, v in res.items()}, st)


if __name__ == "__main__":
    _self_test()

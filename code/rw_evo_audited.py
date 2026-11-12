# /*
#  * Cicada Emergence Optimizer: A Hybrid Success-History Differential Evolution with Staggered Brood Restarts
#  * Authors: Van Van Chau (1,2), Minh Tuan Nguyen (1,*), and Phayung Meesad (2,*)
#  * (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#  *     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam
#  *     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
#  * (2) Department of Information Technology Management, Faculty of Information Technology and Digital Innovation,
#  *     King Mongkut's University of Technology North Bangkok (KMUTNB),
#  *     1518 Pracharat 1 Rd, Wongsawang, Bangsue, Bangkok 10800, Thailand
#  *     phayung.m@itd.kmutnb.ac.th
#  * (*) Corresponding authors
#  * Repository: https://github.com/vancv43/Cicada-Emergence-Optimizer (MIT License)
#  */
"""Audited evolutionary opponents for the CEC2020-RW comparison (RW_EVO_AUDIT.md).

Each runner takes (obj, lb, ub, dim, seed, params=None), with obj a budget.Objective, and stops when the
objective raises BudgetExhausted at evaluation B+1. Schedules use the budget B = obj.max_fe.

Ports of the authors' MATLAB code, kept in third_party/original_matlab/:
  LSHADE-cnEpSin  Awad, Ali, Suganthan, CEC 2017      LSHADE_cnEpSin.m, boundConstraint.m, updateArchive.m
  IMODE           Sallam, Elsayed, Chakrabortty, Ryan, CEC 2020   IMODE_main.m, IMODE.m, han_boun.m, LS2.m
  INFO            Ahmadianfar, Heidari et al., ESWA 2022           INFO.m
Indices are 0-based; MATLAB round is round-half-away-from-zero (_mround). Every departure from the
authors' code is marked DEVIATION and listed in RW_EVO_AUDIT.md.
"""
import math
import numpy as np
from budget import BudgetExhausted


def _guard(fn):
    def wrapped(obj, *a, **k):
        try:
            fn(obj, *a, **k)
        except BudgetExhausted:
            pass
        return obj.best
    wrapped.__name__ = fn.__name__
    return wrapped


def _mround(x):
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def _box(lb, ub, dim):
    return np.full(dim, lb, float) * 1.0, np.full(dim, ub, float) * 1.0


def _cauchy_pos(rng, mu, scale=0.1):
    """mu + scale*tan(pi*(rand-0.5)), redrawn while <= 0 (the authors' rule)."""
    v = mu + scale * np.tan(np.pi * (rng.random(len(mu)) - 0.5))
    bad = v <= 0
    while bad.any():
        v[bad] = mu[bad] + scale * np.tan(np.pi * (rng.random(int(bad.sum())) - 0.5))
        bad = v <= 0
    return v


def _gn_r1r2(rng, n1, n2, n0):
    """gnR1R2.m: r1 in [0,n1) != i, r2 in [0,n2) != i, r1."""
    r0 = np.arange(n0)
    r1 = rng.integers(0, n1, n0)
    pos = r1 == r0
    while pos.any():
        r1[pos] = rng.integers(0, n1, int(pos.sum())); pos = r1 == r0
    r2 = rng.integers(0, n2, n0)
    pos = (r2 == r1) | (r2 == r0)
    while pos.any():
        r2[pos] = rng.integers(0, n2, int(pos.sum())); pos = (r2 == r1) | (r2 == r0)
    return r0, r1, r2


def _midpoint(v, parent, lo, hi):
    """boundConstraint.m: a violating component moves halfway from the parent to the bound."""
    v = v.copy()
    m = v < lo
    v[m] = ((parent + lo) / 2)[m] if parent.ndim == 1 else ((parent + lo[None, :]) / 2)[m]
    m = v > hi
    v[m] = ((parent + hi) / 2)[m] if parent.ndim == 1 else ((parent + hi[None, :]) / 2)[m]
    return v


def _update_archive(rng, A, new, cap):
    """updateArchive.m: append, drop duplicate rows, keep a random subset of size floor(cap) if over."""
    if cap <= 0 or len(new) == 0:
        return A
    allp = np.vstack([A, new]) if len(A) else new.copy()
    allp = np.unique(allp, axis=0)
    k = int(math.floor(cap))
    if len(allp) > k:
        allp = allp[rng.permutation(len(allp))[:k]]
    return allp


def _eval_rows(obj, U):
    return np.array([obj(u) for u in U])


def lpsr_generations(n_max, n_min, budget):
    """Number of generations of an L-SHADE run under linear reduction by evaluations. With
    n_max = 18 D and budget 10,000 D it gives the G_Max constants of LSHADE_cnEpSin.m: 2163 for D = 10."""
    nfes, n, g = n_max, n_max, 0
    while nfes < budget:
        g += 1
        nfes += n
        plan = _mround((n_min - n_max) / budget * nfes + n_max)
        if n > plan:
            n = max(plan, n_min)
    return g


# ------------------------------------------------------------------ LSHADE-cnEpSin
@_guard
def run_lshade_cnepsin(obj, lb, ub, dim, seed, params=None):
    rng = np.random.default_rng(seed)
    lo, hi = _box(lb, ub, dim); B = obj.max_fe; D = dim
    p_best_rate, arc_rate, H = 0.11, 1.4, 5
    NP = 18 * D; n_max, n_min = NP, 4
    pb, ps, freq_init, LP = 0.4, 0.5, 0.5, 20
    # LSHADE_cnEpSin.m hard-codes G_Max for D = 10, 30, 50, 100 at 10,000 D evaluations; these are the
    # generation counts of its own reduction schedule, so the same rule is applied to the RW budget.
    G_Max = lpsr_generations(n_max, n_min, B)
    SEL = _mround(ps * NP)
    pop = lo + rng.random((NP, D)) * (hi - lo)
    fit = _eval_rows(obj, pop); nfes = NP
    M_sf = np.full(H, 0.5); M_cr = np.full(H, 0.5); M_fr = np.full(H, freq_init); mpos = 0
    arch_np = arc_rate * NP; A = np.empty((0, D))
    gF1, gF2, bF1, bF2 = [], [], [], []
    gg = 0
    while nfes < B:
        gg += 1
        sidx = np.argsort(fit, kind="stable")
        mri = rng.integers(0, H, NP)
        mu_sf, mu_cr, mu_fr = M_sf[mri], M_cr[mri], M_fr[mri]
        cr = rng.normal(mu_cr, 0.1); cr[mu_cr == -1] = 0; cr = np.clip(cr, 0, 1)
        sf = _cauchy_pos(rng, mu_sf)
        fr = _cauchy_pos(rng, mu_fr)
        sf = np.minimum(sf, 1); fr = np.minimum(fr, 1)
        flag1 = flag2 = False
        if nfes <= B / 2:
            if gg <= LP:
                use1 = rng.random() < 0.5
            else:
                ns1 = sum(gF1[gg - LP - 1:gg - 1]); nf1 = sum(bF1[gg - LP - 1:gg - 1])
                ns2 = sum(gF2[gg - LP - 1:gg - 1]); nf2 = sum(bF2[gg - LP - 1:gg - 1])
                S1 = ns1 / (ns1 + nf1) + 0.01; S2 = ns2 / (ns2 + nf2) + 0.01
                use1 = S1 / (S1 + S2) > S2 / (S1 + S2)
            if use1:
                sf = np.full(NP, 0.5 * (math.sin(2 * math.pi * freq_init * gg + math.pi) * ((G_Max - gg) / G_Max) + 1))
                flag1 = True
            else:
                sf = 0.5 * (np.sin(2 * np.pi * fr * gg) * (gg / G_Max) + 1)
                flag2 = True
        popAll = np.vstack([pop, A]) if len(A) else pop
        r0, r1, r2 = _gn_r1r2(rng, NP, len(popAll), NP)
        pNP = max(_mround(p_best_rate * NP), 2)
        ri = np.maximum(1, np.ceil(rng.random(NP) * pNP)).astype(int) - 1
        pbest = pop[sidx[ri]]
        vi = pop + sf[:, None] * (pbest - pop + pop[r1] - popAll[r2])
        vi = _midpoint(vi, pop, lo, hi)
        J = rng.integers(0, D, NP)
        crs = rng.random((NP, D)) < cr[:, None]
        rows = np.arange(NP)
        if rng.random() < pb:
            best = pop[sidx[0]]
            dis = np.linalg.norm(pop - best, axis=1)
            Xsel = pop[np.argsort(dis, kind="stable")[:SEL]]
            xm = Xsel.mean(axis=0)
            C = (Xsel - xm).T @ (Xsel - xm) / (SEL - 1)
            C = np.triu(C) + np.triu(C, 1).T
            ev, R = np.linalg.eigh(C)
            if ev.max() > 1e20 * ev.min():
                C = C + (ev.max() / 1e20 - ev.min()) * np.eye(D)
                ev, R = np.linalg.eigh(C)
            Xr = pop @ R; vr = vi @ R
            Ur = Xr.copy(); Ur[rows, J] = vr[rows, J]; Ur[crs] = vr[crs]
            ui = Ur @ R.T
            # DEVIATION: the authors' code does not repair ui after the rotation (CEC2017 evaluates outside
            # the box); here the box is part of the problem, so ui is repaired with its own midpoint rule.
            ui = _midpoint(ui, pop, lo, hi)
        else:
            ui = pop.copy(); ui[rows, J] = vi[rows, J]; ui[crs] = vi[crs]
        cfit = _eval_rows(obj, ui); nfes += NP
        dif = np.abs(fit - cfit); I = fit > cfit
        goodCR, goodF, goodFr, dv = cr[I], sf[I], fr[I], dif[I]
        nbad = int((~I).sum())
        if flag1:
            gF1.append(len(goodF)); bF1.append(nbad); gF2.append(1); bF2.append(1)
        if flag2:
            gF2.append(len(goodF)); bF2.append(nbad); gF1.append(1); bF1.append(1)
        A = _update_archive(rng, A, pop[I], arch_np)
        repl = cfit < fit
        fit = np.where(repl, cfit, fit); pop = pop.copy(); pop[repl] = ui[repl]
        if len(goodCR) > 0:
            w = dv / dv.sum()
            M_sf[mpos] = (w @ goodF ** 2) / (w @ goodF)
            if goodCR.max() == 0 or M_cr[mpos] == -1:
                M_cr[mpos] = -1
            else:
                M_cr[mpos] = (w @ goodCR ** 2) / (w @ goodCR)
            if goodFr.max() == 0 or M_fr[mpos] == -1:
                M_fr[mpos] = -1
            else:
                M_fr[mpos] = (w @ goodFr ** 2) / (w @ goodFr)
            mpos = (mpos + 1) % H
        plan = _mround((n_min - n_max) / B * nfes + n_max)
        if NP > plan:
            red = NP - plan
            if NP - red < n_min:
                red = NP - n_min
            for _ in range(red):
                worst = np.argsort(fit, kind="stable")[-1]
                pop = np.delete(pop, worst, axis=0); fit = np.delete(fit, worst)
            NP -= red
            SEL = _mround(ps * NP)
            arch_np = _mround(arc_rate * NP)
            if len(A) > arch_np:
                A = A[rng.permutation(len(A))[:arch_np]]


# ------------------------------------------------------------------ IMODE
class _LSStop(Exception):
    pass


def _ls_sqp(obj, x0, f0, lo, hi, n_eval):
    """LS2.m with fmincon/sqp replaced by scipy SLSQP (DEVIATION: MATLAB fmincon is not available).
    At most n_eval evaluations through the shared budget; returns the best point seen."""
    from scipy.optimize import minimize
    state = {"n": 0, "x": x0.copy(), "f": f0}

    def fun(x):
        if state["n"] >= n_eval:
            raise _LSStop
        xc = np.clip(x, lo, hi)
        y = obj(xc); state["n"] += 1
        if y < state["f"]:
            state["f"] = y; state["x"] = xc.copy()
        return y
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            minimize(fun, x0, method="SLSQP", bounds=list(zip(lo, hi)), options={"maxiter": 10 ** 6})
    except _LSStop:
        pass
    return state["x"], state["f"]


@_guard
def run_imode(obj, lb, ub, dim, seed, params=None):
    rng = np.random.default_rng(seed)
    lo, hi = _box(lb, ub, dim); B = obj.max_fe; n = dim
    PS = 6 * n * n; n_min = 4; prob_ls = 0.1
    x = lo + rng.random((PS, n)) * (hi - lo)
    fitx = _eval_rows(obj, x); cur = PS
    b = int(np.argmin(fitx)); bestold, bestx = fitx[b], x[b].copy()
    prob = np.full(3, 1 / 3)
    arch_rate = 2.6; arch_np = arch_rate * PS; A = np.empty((0, n))
    mem = 20 * n; M_f = np.full(mem, 0.2); M_cr = np.full(mem, 0.2); hist = 0
    init_pop = PS; PS1 = PS
    while cur < B:
        upd = _mround((n_min - init_pop) / B * cur + init_pop)
        if PS1 > upd:
            red = PS1 - upd
            if PS1 - red < n_min:
                red = PS1 - n_min
            x = x[:PS1 - red]; fitx = fitx[:PS1 - red]; PS1 -= red   # rows are sorted after each generation
            arch_np = _mround(arch_rate * PS1)
            if len(A) > arch_np:
                A = A[rng.permutation(len(A))[:arch_np]]
        # ---- IMODE.m
        mri = rng.integers(0, mem, PS1)
        mu_sf, mu_cr = M_f[mri], M_cr[mri]
        cr = rng.normal(mu_cr, 0.1); cr[mu_cr == -1] = 0; cr = np.clip(cr, 0, 1)
        F = np.minimum(_cauchy_pos(rng, mu_sf), 1)
        o = np.argsort(fitx, kind="stable"); fitx = fitx[o]; x = x[o]
        cr = np.sort(cr)
        popAll = np.vstack([x, A]) if len(A) else x
        r0, r1, r2 = _gn_r1r2(rng, PS1, len(popAll), PS1)
        r3 = rng.integers(0, PS1, PS1)
        pos = (r3 == r0) | (r3 == r1) | (r3 == r2)
        while pos.any():
            r3[pos] = rng.integers(0, PS1, int(pos.sum())); pos = (r3 == r0) | (r3 == r1) | (r3 == r2)
        bb = rng.random(PS1)
        l2 = prob[0] + prob[1]
        op1 = bb <= prob[0]; op2 = (bb > prob[0]) & (bb <= l2); op3 = bb > l2
        vi = np.zeros((PS1, n))
        pNP = max(_mround(0.25 * PS1), 1)
        phix = x[np.maximum(1, np.ceil(rng.random(PS1) * pNP)).astype(int) - 1]
        Fm = F[:, None]
        vi[op1] = (x + Fm * (phix - x + x[r1] - popAll[r2]))[op1]
        vi[op2] = (x + Fm * (phix - x + x[r1] - x[r3]))[op2]
        pNP = max(_mround(0.5 * PS1), 2)
        phix = x[np.maximum(1, np.ceil(rng.random(PS1) * pNP)).astype(int) - 1]
        vi[op3] = (Fm * x[r1] + Fm * (phix - x[r3]))[op3]     # as in IMODE.m
        # han_boun.m, one of three repairs per generation
        hb = int(rng.integers(1, 4))
        L = np.broadcast_to(lo, vi.shape); U = np.broadcast_to(hi, vi.shape)
        if hb == 1:
            m = vi < L; vi[m] = ((x + L) / 2)[m]
            m = vi > U; vi[m] = ((x + U) / 2)[m]
        elif hb == 2:
            m = vi < L; vi[m] = np.minimum(U, np.maximum(L, 2 * L - x))[m]
            # DEVIATION: han_boun.m writes 2*x_L - x2 for the upper bound as well, which sends every upper
            # violation to the lower bound; the reflection 2*x_U - x2 is used, as for the lower bound.
            m = vi > U; vi[m] = np.maximum(L, np.minimum(U, 2 * U - x))[m]
        else:
            m = vi < L; vi[m] = (L + rng.random() * (U - L))[m]
            m = vi > U; vi[m] = (L + rng.random() * (U - L))[m]
        if rng.random() < 0.4:
            mask = rng.random((PS1, n)) > cr[:, None]
            mask[np.arange(PS1), rng.integers(0, n, PS1)] = False
            ui = vi.copy(); ui[mask] = x[mask]
        else:
            ui = x.copy()
            start = rng.integers(0, n, PS1)
            for i in range(PS1):
                l = start[i]
                while rng.random() < cr[i] and l < n - 1:
                    l += 1
                ui[i, start[i]:l + 1] = vi[i, start[i]:l + 1]
        new = _eval_rows(obj, ui); cur += PS1
        diff = np.abs(fitx - new); I = new < fitx
        goodCR, goodF = cr[I], F[I]
        A = _update_archive(rng, A, x[I], math.ceil(arch_np))
        with np.errstate(divide="ignore", invalid="ignore"):
            d2 = np.maximum(0, fitx - new) / np.abs(fitx)
        cs = np.zeros(3)
        for k, m in enumerate((op1, op2, op3)):
            if m.any():
                v = float(np.mean(d2[m]))
                cs[k] = 0.0 if np.isnan(v) else max(0.0, v)    # MATLAB max(0, NaN) = 0
        if np.all(cs != 0) and np.isfinite(cs.sum()):
            prob = np.clip(cs / cs.sum(), 0.1, 0.9)
        else:
            prob = np.full(3, 1 / 3)
        fitx = np.where(I, new, fitx); x = x.copy(); x[I] = ui[I]
        if len(goodCR) > 0:
            w = diff[I] / diff[I].sum()
            M_f[hist] = (w @ goodF ** 2) / (w @ goodF)
            if goodCR.max() == 0 or M_cr[hist] == -1:
                M_cr[hist] = -1
            else:
                M_cr[hist] = (w @ goodCR ** 2) / (w @ goodCR)
            hist = (hist + 1) % mem
        else:
            M_cr[hist] = 0.5; M_f[hist] = 0.5
        o = np.argsort(fitx, kind="stable"); fitx = fitx[o]; x = x[o]
        if fitx[0] < bestold:
            bestold, bestx = fitx[0], x[0].copy()
        # ---- local search of IMODE_main.m in the last 15 % of the budget
        if 0.85 * B < cur < B and rng.random() < prob_ls:
            n_ls = min(math.ceil(0.02 * B), B - cur)
            n0 = obj.n
            xs, fs = _ls_sqp(obj, bestx, bestold, lo, hi, n_ls)
            cur += obj.n - n0
            if bestold - fs > 0:
                bestold, bestx = fs, xs.copy()
                x[PS1 - 1] = bestx; fitx[PS1 - 1] = bestold
                o = np.argsort(fitx, kind="stable"); fitx = fitx[o]; x = x[o]
                prob_ls = 0.1
            else:
                prob_ls = 0.01


# ------------------------------------------------------------------ INFO
EPS_INFO = 1e-25   # the small constant epsilon of INFO, value of INFO.m


def _exp_ov(v):
    """exp that returns inf instead of raising OverflowError; a non-finite trial is handled below."""
    try:
        return math.exp(v)
    except OverflowError:
        return float("inf")       # math.exp(nan) = nan and math.exp(-inf) = 0 pass through unchanged


@_guard
def run_info(obj, lb, ub, dim, seed, params=None):
    rng = np.random.default_rng(seed)
    lo, hi = _box(lb, ub, dim); B = obj.max_fe; nP = 30
    MaxIt = max(1, (B - nP) // nP)        # iterations that the budget allows after initialisation
    X = lo + rng.random((nP, dim)) * (hi - lo)
    Cost = _eval_rows(obj, X); M = Cost.copy()
    ind = np.argsort(Cost, kind="stable")
    Best_X, Best_C = X[ind[0]].copy(), Cost[ind[0]]
    Worst_X, Worst_C = X[ind[-1]].copy(), Cost[ind[-1]]
    k = int(rng.integers(2, 6)) - 1
    Better_X, Better_C = X[ind[k]].copy(), Cost[ind[k]]
    it = 0
    while True:
        it += 1
        alpha = 2 * math.exp(-4 * (it / MaxIt))
        MB, MBt, MW = Best_C, Better_C, Worst_C
        for i in range(nP):
            de = 2 * rng.random() * alpha - alpha
            sg = 2 * rng.random() * alpha - alpha
            A1 = rng.permutation(nP); A1 = A1[A1 != i]
            a, b, c = A1[0], A1[1], A1[2]
            epsi = EPS_INFO * rng.random()
            with np.errstate(all="ignore"):
                # Published paper, ESWA 195:116516, Eqs. 4.1-4.10: w_k = cos(dF + pi) exp(-|dF / omega|) as in the GitHub
                # INFO.m (the authors' brief version and the 2022 INFO.m omit the absolute value); denominator
                # w1 + w2 + w3 + eps as in the paper (INFO.m adds 1). PROTOCOL_rwx.md D2 and D2b.
                omg = max(M[a], M[b], M[c])
                MM = (M[a] - M[b], M[a] - M[c], M[b] - M[c])
                W = [float(np.cos(m + math.pi)) * _exp_ov(-abs(float(np.float64(m) / np.float64(omg)))) for m in MM]   # Eqs. 4.2-4.4, 4.7-4.9: exp(-|df/omega|)
                WM1 = de * (W[0] * (X[a] - X[b]) + W[1] * (X[a] - X[c]) + W[2] * (X[b] - X[c])) / (sum(W) + EPS_INFO) + epsi
                omg = MW                                     # Eq. 1.10, omega = f(x_ws)
                MM = (MB - MBt, MB - MW, MBt - MW)          # paper Eqs. 1.7-1.9; INFO.m repeats MB - MBt
                W = [float(np.cos(m + math.pi)) * _exp_ov(-abs(float(np.float64(m) / np.float64(omg)))) for m in MM]   # Eqs. 4.2-4.4, 4.7-4.9: exp(-|df/omega|)
                WM2 = de * (W[0] * (Best_X - Better_X) + W[1] * (Best_X - Worst_X) + W[2] * (Better_X - Worst_X)) / (sum(W) + EPS_INFO) + epsi
                r = rng.uniform(0.0, 0.5)                    # Eq. 3, r in [0, 0.5]; INFO.m draws from [0.1, 0.5]
                MR = r * WM1 + (1 - r) * WM2
                if rng.random() < 0.5:                       # Eq. 1, sigma x MeanRule; INFO.m adds a factor rand
                    z1 = X[i] + sg * MR + rng.standard_normal() * (Best_X - X[a]) / (MB - M[a] + 1)
                    z2 = Best_X + sg * MR + rng.standard_normal() * (X[a] - X[b]) / (M[a] - M[b] + 1)
                else:
                    z1 = X[a] + sg * MR + rng.standard_normal() * (X[b] - X[c]) / (M[b] - M[c] + 1)
                    z2 = Better_X + sg * MR + rng.standard_normal() * (X[a] - X[b]) / (M[a] - M[b] + 1)
            u = np.empty(dim)
            for j in range(dim):
                mu = 0.05 * rng.standard_normal()
                if rng.random() < 0.5:
                    if rng.random() < 0.5:
                        u[j] = z1[j] + mu * abs(z1[j] - z2[j])
                    else:
                        u[j] = z2[j] + mu * abs(z1[j] - z2[j])
                else:
                    u[j] = X[i, j]
            if rng.random() < 0.5:
                L = rng.random() < 0.5
                v1 = (1 - L) * 2 * rng.random() + L
                v2 = rng.random() * L + (1 - L)
                Xavg = (X[a] + X[b] + X[c]) / 3
                phi = rng.random()
                Xrnd = phi * Xavg + (1 - phi) * (phi * Better_X + (1 - phi) * Best_X)
                Rn = rng.standard_normal(dim) if L else np.full(dim, rng.standard_normal())
                if rng.random() < 0.5:
                    u = Best_X + Rn * (MR + rng.standard_normal() * (Best_X - X[a]))
                else:
                    u = Xrnd + Rn * (MR + rng.standard_normal() * (v1 * Best_X - v2 * Xrnd))
            u = np.where(u > hi, hi, np.where(u < lo, lo, u))     # BC: clip to the box
            if not np.all(np.isfinite(u)):
                u = X[i].copy()   # DEVIATION: a non-finite trial is replaced by the parent; it still costs one evaluation
                #            and is rejected, as a NaN trial is in INFO.m
            nc = obj(u)
            if nc < Cost[i]:
                X[i] = u; Cost[i] = nc; M[i] = nc
        # Algorithm 1: the best vector is updated once per generation, after the loop over vectors
        # (INFO.m updates it inside the loop)
        ind = np.argsort(Cost, kind="stable")
        if Cost[ind[0]] < Best_C:
            Best_X, Best_C = X[ind[0]].copy(), Cost[ind[0]]
        Worst_X, Worst_C = X[ind[-1]].copy(), Cost[ind[-1]]
        k = int(rng.integers(2, 6)) - 1
        Better_X, Better_C = X[ind[k]].copy(), Cost[ind[k]]



# ------------------------------------------------------------------ HBA
@_guard
def run_hba(obj, lb, ub, dim, seed, params=None):
    """Honey badger algorithm, Hashim et al. 2022, Math. Comput. Simul. 192:84-110, Eqs. 1-6 and Algorithm 1.
    beta = 6, C = 2 (paper defaults); N = 30, the study-wide swarm population. alpha = C exp(-t/t_max) decreases
    over the run (mealpy keeps it at C exp(-1)). Paper over the authors' HBA.m (PROTOCOL_rwx.md D2a): the
    intensity I_i is computed inside the loop over badgers from the current x_i, x_{i+1} and x_prey; a new
    position replaces x_i if f_new <= f_i; x_prey is replaced at once if f_new <= f_prey. Details the paper leaves
    open follow HBA.m: one r and one F per badger, r3, r4, r5, r7 per dimension, eps inside the squared
    distances, x_{N+1} = x_1, clipping to the box."""
    rng = np.random.default_rng(seed)
    lo, hi = _box(lb, ub, dim); B = obj.max_fe; N = 30; beta, C = 6.0, 2.0
    tmax = max(1, (B - N) // N)
    X = lo + rng.random((N, dim)) * (hi - lo)                             # Eq. 1
    fit = _eval_rows(obj, X)
    b = int(np.argmin(fit)); prey, fprey = X[b].copy(), fit[b]
    eps = np.finfo(float).eps
    t = 0
    while True:
        t += 1
        alpha = C * math.exp(-t / tmax)                                   # Eq. 3
        for i in range(N):
            Si = float(np.sum((X[i] - X[(i + 1) % N] + eps) ** 2))        # Eq. 2, S = (x_i - x_{i+1})^2
            di2 = float(np.sum((X[i] - prey + eps) ** 2))                 # d_i = x_prey - x_i
            with np.errstate(divide="ignore", invalid="ignore"):          # Eq. 2, r2; IEEE division as in HBA.m:
                Ii = float(np.float64(rng.random() * Si) / np.float64(4 * math.pi * di2))  # d_i = 0 -> Inf or NaN
            # DEVIATION D2c (RW_EVO_AUDIT.md section 9): Python raised ZeroDivisionError when d_i^2 = 0 exactly
            # (HBA on RC20). MATLAB returns Inf/NaN; np.clip then maps +-Inf to the bounds and keeps NaN, and a
            # NaN candidate is never accepted, as in HBA.m. Runs where d_i^2 > 0 are bit-identical.
            r = rng.random(); F = 1.0 if rng.random() <= 0.5 else -1.0    # Algorithm 1 r; Eq. 5, r6
            d = prey - X[i]
            if r < 0.5:                                                   # digging phase, Eq. 4
                r3, r4, r5 = rng.random(dim), rng.random(dim), rng.random(dim)
                xn = prey + F * beta * Ii * prey + F * r3 * alpha * d * np.abs(np.cos(2 * np.pi * r4) * (1 - np.cos(2 * np.pi * r5)))
            else:                                                         # honey phase, Eq. 6
                xn = prey + F * rng.random(dim) * alpha * d
            xn = np.clip(xn, lo, hi)
            fn = obj(xn)
            if fn <= fit[i]:                                              # Algorithm 1: f_new <= f_i
                X[i] = xn; fit[i] = fn
            if fn <= fprey:                                               # Algorithm 1: f_new <= f_prey, at once
                prey, fprey = xn.copy(), fn


# ------------------------------------------------------------------ SCSO
def _roulette(rng, p):
    c = np.cumsum(p / p.sum())
    return int(np.searchsorted(c, rng.random(), side="left"))   # first C >= r, as find(r <= C, 1)


@_guard
def run_scso(obj, lb, ub, dim, seed, params=None):
    """Sand cat swarm optimization, Seyyedabbasi and Kiani 2023, Eng. Comput. 39:2627-2651, Eqs. 1-6 and
    Algorithm 1, with the authors' SCSO.m (sand_cat_swarm 1.0.0) for details the paper leaves open. The roulette
    angle is in degrees as in Algorithm 1 (SCSO.m takes radians, D2b). N = 30, S_M = 2,
    r_G = S_M - S_M t/T, R = 2 r_G rand - r_G, r = r_G rand; exploitation |R| <= 1 with a roulette angle
    over 1..360, otherwise exploration toward a random candidate. No greedy step: every cat moves, the best
    position found so far is kept. Positions are updated in place, cat by cat and dimension by dimension."""
    rng = np.random.default_rng(seed)
    lo, hi = _box(lb, ub, dim); B = obj.max_fe; N = 30; SM = 2.0
    T = max(1, B // N)
    P = lo + rng.random((N, dim)) * (hi - lo)
    pang = np.arange(1, 361, dtype=float)
    best, fbest = P[0].copy(), np.inf
    t = 0
    while True:
        for i in range(N):
            P[i] = np.clip(P[i], lo, hi)
            f = obj(P[i])
            if f < fbest:
                fbest, best = f, P[i].copy()
        rG = SM - SM * t / T                                              # Eq. 1
        for i in range(N):
            r = rng.random() * rG                                         # Eq. 2
            R = 2 * rG * rng.random() - rG                                # Eq. 3
            for j in range(dim):
                theta = _roulette(rng, pang) + 1
                if -1 <= R <= 1:                                          # attack, Eq. 5
                    rp = abs(rng.random() * best[j] - P[i, j])
                    P[i, j] = best[j] - r * rp * math.cos(math.radians(theta))   # Algorithm 1: 0 deg <= theta <= 360 deg (SCSO.m takes radians, D2b)
                else:                                                     # search, Eq. 4
                    cp = int(rng.random() * N)
                    P[i, j] = r * (P[cp, j] - rng.random() * P[i, j])
        t += 1


RUNNERS = {"LSHADE-cnEpSin": run_lshade_cnepsin, "IMODE": run_imode, "INFO": run_info,
           "HBA": run_hba, "SCSO": run_scso}

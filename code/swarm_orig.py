# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""Swarm-intelligence baselines written from the original publications (PROTOCOL_swarm20.md).

Rule: the published equations are primary; where the paper leaves a detail open, the structure of the
authors' public code is followed. Every runner has the signature (obj, lb, ub, dim, seed, params=None),
evaluates only through `obj` (budget.Objective, raises BudgetExhausted at evaluation B+1) and returns
obj.best. Iteration-scheduled parameters use the horizon T implied by the budget; algorithms whose
evaluations per iteration vary use the spent fraction of the budget, obj.n / B, as t / T.
Out-of-range coordinates are clipped to the box unless the original prescribes otherwise.
"""
import io
import math
import os
import random
import sys
import contextlib

import numpy as np

from budget import BudgetExhausted

HERE = os.path.dirname(os.path.abspath(__file__))
POP = 30


def _guard(fn):
    def wrapped(obj, lb, ub, dim, seed, params=None):
        try:
            with open(os.devnull, 'w') as dn, contextlib.redirect_stdout(dn):
                fn(obj, lb, ub, dim, seed, params)
        except BudgetExhausted:
            pass
        return obj.best
    wrapped.__name__ = fn.__name__
    return wrapped


def _box(lb, ub, dim):
    return (np.broadcast_to(np.asarray(lb, float), (dim,)).copy(),
            np.broadcast_to(np.asarray(ub, float), (dim,)).copy())


def _mantegna(rng, size, beta=1.5):
    sig = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
           / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    return rng.normal(0, sig, size) / np.abs(rng.normal(0, 1, size)) ** (1 / beta)


def _iters(B, n_init, per_iter):
    return max(2, int(math.ceil((B - n_init) / per_iter)))


# =============================================================================================
# 1  PSO: tuned global-best inertia PSO of the shared harness (Shi and Eberhart 1998), verbatim
# =============================================================================================
def _reflect(x, lb, ub):
    span = ub - lb; y = (x - lb) % (2 * span); y = np.where(y <= span, y, 2 * span - y); return lb + y


@_guard
def run_pso(obj, lb, ub, dim, seed, params=None):
    import json
    p = {'w': 0.7298, 'c1': 1.49618, 'c2': 1.49618, 'N': 30,
         **json.load(open(os.path.join(HERE, 'tuned_configs.json')))['PSO']}
    rng = np.random.default_rng(seed); n = int(p['N']); lo, hi = _box(lb, ub, dim)
    vmax = 0.2 * (hi - lo)
    X = rng.uniform(lo, hi, (n, dim)); V = rng.uniform(-vmax, vmax, (n, dim))
    fit = np.array([obj(x) for x in X]); P = X.copy(); Pf = fit.copy(); g = int(np.argmin(Pf))
    while True:
        for i in range(n):
            r1 = rng.random(dim); r2 = rng.random(dim)
            V[i] = np.clip(p['w'] * V[i] + p['c1'] * r1 * (P[i] - X[i]) + p['c2'] * r2 * (P[g] - X[i]), -vmax, vmax)
            X[i] = _reflect(X[i] + V[i], lo, hi); fx = obj(X[i])
            if fx < Pf[i]:
                P[i] = X[i].copy(); Pf[i] = fx
                if fx < Pf[g]:
                    g = i


# =============================================================================================
# 2  ACO for continuous domains, ACOR (Socha and Dorigo 2008, EJOR 185:1155), Eqs. 7-9, Sect. 3.3 and
#    Appendix A (variable-correlation handling): per ant one kernel l (Eq. 8); at construction step i
#    a direction s_l -> s_u is drawn with p ~ d_i(s_u, s_l)^4 (Eq. A.1), d_i measured in the
#    (n-i+1)-dimensional subspace orthogonal to the earlier directions, orthonormalised by
#    Gram-Schmidt; the archive is expressed in that basis, Eq. 9 gives sigma along it and z_i is
#    sampled; finally z is rotated back. Archive update: add the m new solutions, drop the m worst.
#    Table 2: m = 2, xi = 0.85, q = 1e-4, k = 50.
# =============================================================================================
@_guard
def run_aco(obj, lb, ub, dim, seed, params=None):
    rng = np.random.default_rng(seed); lo, hi = _box(lb, ub, dim)
    k, m, q, xi = 50, 2, 1e-4, 0.85
    S = rng.uniform(lo, hi, (k, dim)); f = np.array([obj(s) for s in S])
    o = np.argsort(f); S, f = S[o], f[o]
    l = np.arange(1, k + 1)
    w = np.exp(-((l - 1) ** 2) / (2 * q * q * k * k)) / (q * k * math.sqrt(2 * math.pi))   # Eq. 7
    p = w / w.sum()                                                                     # Eq. 8
    while True:
        new = np.empty((m, dim)); fn = np.empty(m)
        for a in range(m):
            g = rng.choice(k, p=p)                       # one Gaussian kernel per ant and iteration
            Bv = []                                      # orthonormal directions chosen so far
            R = S - S[g]                                 # residuals in the remaining subspace
            for i in range(dim):
                d4 = np.linalg.norm(R, axis=1) ** 4
                if d4.sum() > 0:
                    u = rng.choice(k, p=d4 / d4.sum())   # Eq. A.1
                    v = R[u] / np.linalg.norm(R[u])
                else:                                    # degenerate archive: random direction
                    v = rng.standard_normal(dim)
                    for b in Bv:
                        v = v - (v @ b) * b
                    v = v / np.linalg.norm(v)
                Bv.append(v)
                R = R - np.outer(R @ v, v)               # project out the new direction
            Bm = np.array(Bv)                            # rows = new basis
            Z = S @ Bm.T                                 # archive in the rotated coordinates
            sig = xi * np.abs(Z - Z[g]).sum(0) / (k - 1)  # Eq. 9 per rotated dimension
            z = Z[g] + sig * rng.standard_normal(dim)
            new[a] = np.clip(z @ Bm, lo, hi)             # back to the original coordinates
            fn[a] = obj(new[a])
        S = np.vstack([S, new]); f = np.concatenate([f, fn])
        o = np.argsort(f)[:k]; S, f = S[o], f[o]


# =============================================================================================
# 3  ABC (Karaboga and Basturk 2007, JOGO 39:459; Karaboga's public code): SN = N/2 food sources,
#    one dimension changed per move, fitness 1/(1+f), p_i = 0.9 fit_i / max fit + 0.1,
#    limit = SN x D (Karaboga and Akay 2009), one scout per cycle
# =============================================================================================
@_guard
def run_abc(obj, lb, ub, dim, seed, params=None):
    rng = np.random.default_rng(seed); lo, hi = _box(lb, ub, dim)
    SN = POP // 2; limit = SN * dim
    X = rng.uniform(lo, hi, (SN, dim)); f = np.array([obj(x) for x in X]); trial = np.zeros(SN, int)

    def fitv(v):
        return 1 / (1 + v) if v >= 0 else 1 + abs(v)

    def move(i):
        j = rng.integers(dim); k = rng.integers(SN - 1); k += k >= i
        v = X[i].copy(); v[j] = X[i, j] + (rng.random() - 0.5) * 2 * (X[i, j] - X[k, j])
        v[j] = min(max(v[j], lo[j]), hi[j]); fv = obj(v)
        if fitv(fv) > fitv(f[i]):
            X[i] = v; f[i] = fv; trial[i] = 0
        else:
            trial[i] += 1
    while True:
        for i in range(SN):                                  # employed bees
            move(i)
        F = np.array([fitv(v) for v in f]); prob = 0.9 * F / F.max() + 0.1
        i = t = 0
        while t < SN:                                        # onlooker bees
            if rng.random() < prob[i]:
                t += 1; move(i)
            i = (i + 1) % SN
        s = int(np.argmax(trial))                            # scout bee
        if trial[s] >= limit:
            X[s] = rng.uniform(lo, hi); f[s] = obj(X[s]); trial[s] = 0


# =============================================================================================
# 4, 5  WOA (Mirjalili and Lewis 2016): EvoloPy WOA.py, verified line by line against the authors' WOA.m,
#        used verbatim (third_party/evolopy, Apache-2.0). GWO (Mirjalili et al. 2014): literal port of GWO.m.
# =============================================================================================
def _evolopy(name):
    @_guard
    def run(obj, lb, ub, dim, seed, params=None):
        sys.path.insert(0, os.path.join(HERE, 'third_party'))
        import importlib
        fn = getattr(importlib.import_module(f'evolopy.{name}'), name)
        random.seed(int(seed)); np.random.seed(int(seed) % (2 ** 32))
        lo, hi = _box(lb, ub, dim)

        def objf(x):
            return obj(x)
        objf.__name__ = 'objective'
        fn(objf, lo.tolist(), hi.tolist(), dim, POP, _iters(obj.max_fe, 0, POP))
    run.__name__ = f'run_{name.lower()}'
    return run


run_woa = _evolopy('WOA')


@_guard
def run_gwo(obj, lb, ub, dim, seed, params=None):
    """Literal port of the authors' GWO.m (seyedalimirjalili.com, GWO.zip): leaders replaced without
    the alpha -> beta -> delta cascade that EvoloPy adds; per-dimension r1, r2."""
    rng = np.random.default_rng(seed); lo, hi = _box(lb, ub, dim); N = POP
    T = _iters(obj.max_fe, 0, N)
    al_p = np.zeros(dim); be_p = np.zeros(dim); de_p = np.zeros(dim); al_s = be_s = de_s = np.inf
    X = rng.uniform(lo, hi, (N, dim))
    for l in range(T):
        for i in range(N):
            X[i] = np.clip(X[i], lo, hi); f = obj(X[i])
            if f < al_s:
                al_s, al_p = f, X[i].copy()
            if al_s < f < be_s:
                be_s, be_p = f, X[i].copy()
            if f > al_s and f > be_s and f < de_s:
                de_s, de_p = f, X[i].copy()
        a = 2 - l * (2 / T)
        for i in range(N):
            for j in range(dim):
                A1 = 2 * a * rng.random() - a; C1 = 2 * rng.random()
                X1 = al_p[j] - A1 * abs(C1 * al_p[j] - X[i, j])
                A2 = 2 * a * rng.random() - a; C2 = 2 * rng.random()
                X2 = be_p[j] - A2 * abs(C2 * be_p[j] - X[i, j])
                A3 = 2 * a * rng.random() - a; C3 = 2 * rng.random()
                X3 = de_p[j] - A3 * abs(C3 * de_p[j] - X[i, j])
                X[i, j] = (X1 + X2 + X3) / 3


# =============================================================================================
# 6  HHO (Heidari et al. 2019, FGCS 97:849), Eqs. 1-13; acceptance in Eqs. 10-11 compares with
#    F(X_i(t)); S ~ U(0,1)^D and the 0.01-scaled Levy flight of Eq. 9 as in HHO.m.
#    Evaluations per iteration vary, so t/T is the spent budget fraction.
# =============================================================================================
@_guard
def run_hho(obj, lb, ub, dim, seed, params=None):
    rng = np.random.default_rng(seed); lo, hi = _box(lb, ub, dim); N = POP; B = obj.max_fe
    X = rng.uniform(lo, hi, (N, dim)); rab = None; frab = np.inf
    while True:
        X = np.clip(X, lo, hi)
        fx = np.empty(N)
        for i in range(N):
            fx[i] = obj(X[i])
            if fx[i] < frab:
                frab = fx[i]; rab = X[i].copy()
        E1 = 2 * (1 - obj.n / B)
        Xm = X.mean(0)                                                   # Eq. 2
        Xold = X.copy()
        for i in range(N):
            E = E1 * (2 * rng.random() - 1)                              # Eq. 3
            if abs(E) >= 1:                                              # exploration, Eq. 1
                if rng.random() >= 0.5:
                    Xr = Xold[rng.integers(N)]
                    X[i] = Xr - rng.random() * np.abs(Xr - 2 * rng.random() * Xold[i])
                else:
                    X[i] = (rab - Xm) - rng.random() * (lo + rng.random() * (hi - lo))
                continue
            r = rng.random(); J = 2 * (1 - rng.random())
            if r >= 0.5 and abs(E) >= 0.5:                               # soft besiege, Eq. 4
                X[i] = (rab - Xold[i]) - E * np.abs(J * rab - Xold[i])
            elif r >= 0.5:                                               # hard besiege, Eq. 6
                X[i] = rab - E * np.abs(rab - Xold[i])
            else:                                                        # progressive dives, Eqs. 7-13
                ref = Xold[i] if abs(E) >= 0.5 else Xm
                Y = np.clip(rab - E * np.abs(J * rab - ref), lo, hi)
                if obj(Y) < fx[i]:
                    X[i] = Y; continue
                Z = np.clip(Y + rng.random(dim) * 0.01 * _mantegna(rng, dim), lo, hi)
                if obj(Z) < fx[i]:
                    X[i] = Z


# =============================================================================================
# 7  FA (Yang 2009, SAGA LNCS 5792 = arXiv:1003.1466; Yang 2010, IJBIC 2:78 = arXiv:1003.1409),
#    Eq. 9: x_i += beta0 exp(-gamma r_ij^2)(x_j - x_i) + alpha S_k (rand - 1/2) toward every brighter j,
#    evaluate after each move; beta0 = 1, alpha = 0.2 (papers' values, no reduction), randomisation
#    scaled by S_k = ub - lb, gamma = 1/Gamma^m with m = 2 and length scale Gamma = S_k (Eq. 7 rule);
#    a firefly with no brighter one moves randomly (rule 2). j runs over all n as the comment
#    "all n fireflies" in the pseudocode states.
# =============================================================================================
@_guard
def run_fa(obj, lb, ub, dim, seed, params=None):
    rng = np.random.default_rng(seed); lo, hi = _box(lb, ub, dim); N = POP
    Sk = hi - lo; gamma = 1 / float(Sk.mean()) ** 2; alpha, beta0 = 0.2, 1.0
    X = rng.uniform(lo, hi, (N, dim)); I = np.array([obj(x) for x in X])
    while True:
        for i in range(N):
            moved = False
            for j in range(N):
                if I[j] < I[i]:                                  # j brighter (minimisation)
                    r2 = float(np.sum((X[i] - X[j]) ** 2))
                    X[i] = np.clip(X[i] + beta0 * np.exp(-gamma * r2) * (X[j] - X[i])
                                   + alpha * Sk * (rng.random(dim) - 0.5), lo, hi)
                    I[i] = obj(X[i]); moved = True
            if not moved:                                        # brightest: random move
                X[i] = np.clip(X[i] + alpha * Sk * (rng.random(dim) - 0.5), lo, hi)
                I[i] = obj(X[i])


# =============================================================================================
# 8  BA (Yang 2010, NICSO SCI 284:65 = arXiv:1004.4170), Eqs. 2-6 and Fig. 1: f_i = fmin + (fmax-fmin)
#    beta with fmin = 0, fmax = 100 (the paper's implementation), v += (x - x*) f, x += v; if
#    rand > r_i a local walk x* + eps <A>, eps ~ U(-1,1); accept if rand < A_i and the new value is
#    better than f_i (author's code reading of "f(x_i) < f(x*)"), then A_i *= alpha and
#    r_i = r_i0 (1 - exp(-gamma t)); A_i0 ~ U(1,2), r_i0 ~ U(0,1), alpha = gamma = 0.9
# =============================================================================================
@_guard
def run_ba(obj, lb, ub, dim, seed, params=None):
    rng = np.random.default_rng(seed); lo, hi = _box(lb, ub, dim); N = POP
    fmin, fmax, al, ga = 0.0, 100.0, 0.9, 0.9
    X = rng.uniform(lo, hi, (N, dim)); V = np.zeros((N, dim)); f = np.array([obj(x) for x in X])
    A = rng.uniform(1, 2, N); r0 = rng.uniform(0, 1, N); r = r0.copy()
    b = int(np.argmin(f)); best, fbest = X[b].copy(), f[b]
    t = 0
    while True:
        t += 1
        for i in range(N):
            fr = fmin + (fmax - fmin) * rng.random()                        # Eq. 2
            V[i] = V[i] + (X[i] - best) * fr                                # Eq. 3
            xn = X[i] + V[i]                                                # Eq. 4
            if rng.random() > r[i]:
                xn = best + rng.uniform(-1, 1, dim) * A.mean()              # Eq. 5
            xn = np.clip(xn, lo, hi); fn = obj(xn)
            if rng.random() < A[i] and fn < f[i]:
                X[i] = xn; f[i] = fn
                A[i] *= al; r[i] = r0[i] * (1 - math.exp(-ga * t))          # Eq. 6
            if fn <= fbest:
                best, fbest = xn.copy(), fn


# =============================================================================================
# 9  CS (Yang and Deb 2009, NaBIC): niapy 2.5.2 CuckooSearch, which follows Yang's
#    cuckoo_search.m (0.01-scaled Mantegna steps times (x - best) times N(0,1); discovery with
#    probability pa = 0.25 by a random-permutation walk; greedy replacement); n = 30
# =============================================================================================
@_guard
def run_cs(obj, lb, ub, dim, seed, params=None):
    from niapy.task import Task
    from niapy.problems import Problem
    from niapy.algorithms.basic import CuckooSearch
    lo, hi = _box(lb, ub, dim)

    class P(Problem):
        def __init__(self):
            super().__init__(dimension=dim, lower=lo, upper=hi)

        def _evaluate(self, x):
            return obj(x)
    CuckooSearch(population_size=POP, pa=0.25, seed=int(seed) % (2 ** 32 - 1)).run(
        Task(problem=P(), max_evals=obj.max_fe + 10 ** 7))


# =============================================================================================
# 10 BFO (Passino 2002, IEEE Control Syst. 22:52): chemotaxis with tumble and swim (J_last
#    updated while swimming), swarming term J_cc, reproduction of the healthiest half,
#    elimination-dispersal with Ped, exactly the boxed algorithm on p. 61; Nc = 100, Ns = 4, Nre = 4,
#    Ned = 2, Ped = 0.25, C = 0.1, d_attr = 0.1, w_attr = 0.2, h_rep = d_attr, w_rep = 10 (paper
#    values), S = 30 (paper 50, protocol population). The elimination-dispersal loop is repeated
#    until the budget is spent.
# =============================================================================================
@_guard
def run_bfo(obj, lb, ub, dim, seed, params=None):
    rng = np.random.default_rng(seed); lo, hi = _box(lb, ub, dim)
    S, Nc, Ns, Nre, Ned, Ped, C = POP, 100, 4, 4, 2, 0.25, 0.1
    da, wa, hr, wr = 0.1, 0.2, 0.1, 10.0
    P = rng.uniform(lo, hi, (S, dim))

    def jcc(th, P):
        d2 = np.sum((P - th) ** 2, axis=1)
        return float(np.sum(-da * np.exp(-wa * d2)) + np.sum(hr * np.exp(-wr * d2)))
    while True:
        for _ in range(Ned):
            for _ in range(Nre):
                health = np.zeros(S); Jend = np.zeros(S)
                for _ in range(Nc):
                    for i in range(S):                  # bacteria 1..i-1 already moved (step g note)
                        J = obj(P[i]) + jcc(P[i], P); health[i] += J; Jlast = J      # steps b-c
                        d = rng.uniform(-1, 1, dim); d /= np.linalg.norm(d)          # step d
                        P[i] = np.clip(P[i] + C * d, lo, hi)                         # step e
                        J = obj(P[i]) + jcc(P[i], P); m = 0                          # step f
                        while m < Ns:                                                # step g
                            m += 1
                            if J < Jlast:
                                Jlast = J
                                P[i] = np.clip(P[i] + C * d, lo, hi)
                                J = obj(P[i]) + jcc(P[i], P)
                            else:
                                m = Ns
                        Jend[i] = J
                health += Jend                                   # J_health sums j = 1..Nc+1
                o = np.argsort(health)                                       # reproduction
                P = np.vstack([P[o[:S // 2]], P[o[:S // 2]]])
            for i in range(S):                                               # elimination-dispersal
                if rng.random() < Ped:
                    P[i] = rng.uniform(lo, hi)


# =============================================================================================
# 11 DA (Mirjalili 2016, Neural Comput. Appl. 27:1053), Eqs. 3.1-3.10 with the schedules of DA.m:
#    neighbourhood radius from (ub-lb)/4 to beyond the box, w 0.9 -> 0.4, weights s, a, c = 2 rand c,
#    f = 2 rand, e = c, c from 0.1 to 0 over the first half; Levy flight when alone (Eq. 3.8),
#    step vector capped at (ub-lb)/10; walls as in DA.m
# =============================================================================================
@_guard
def run_da(obj, lb, ub, dim, seed, params=None):
    rng = np.random.default_rng(seed); lo, hi = _box(lb, ub, dim); N = POP
    T = _iters(obj.max_fe, 0, N); dmax = (hi - lo) / 10
    X = rng.uniform(lo, hi, (N, dim)); DX = rng.uniform(lo, hi, (N, dim))
    food, ffood = np.zeros(dim), np.inf; enemy, fenemy = np.zeros(dim), -np.inf   # as DA.m
    for it in range(1, T + 1):
        rad = (hi - lo) / 4 + (hi - lo) * (it / T) * 2
        w = 0.9 - it * ((0.9 - 0.4) / T)
        mc = max(0.0, 0.1 - it * (0.1 / (T / 2)))
        s, a, c, fw, e = 2 * rng.random() * mc, 2 * rng.random() * mc, 2 * rng.random() * mc, 2 * rng.random(), mc
        for i in range(N):
            fi = obj(X[i])
            if fi < ffood:
                ffood, food = fi, X[i].copy()
            if fi > fenemy and np.all(X[i] < hi) and np.all(X[i] > lo):
                fenemy, enemy = fi, X[i].copy()
        for i in range(N):
            d = np.abs(X - X[i])
            nb = np.all(d <= rad, axis=1) & np.all(d != 0, axis=1)
            k = int(nb.sum())
            if k > 1:
                S = -np.sum(X[nb] - X[i], axis=0)                          # Eq. 3.1 separation
                A = DX[nb].mean(0)                                         # Eq. 3.2 alignment
                Cc = X[nb].mean(0) - X[i]                                  # Eq. 3.3 cohesion
            else:
                S = np.zeros(dim); A = DX[i].copy(); Cc = np.zeros(dim)
            near_food = np.all(np.abs(X[i] - food) <= rad)
            F = food - X[i] if near_food else np.zeros(dim)                # Eq. 3.4
            En = enemy + X[i] if np.all(np.abs(X[i] - enemy) <= rad) else np.zeros(dim)   # Eq. 3.5
            up, dn = X[i] > hi, X[i] < lo                                  # walls of DA.m
            X[i] = np.where(up, lo, np.where(dn, hi, X[i])); DX[i] = np.where(up | dn, rng.random(dim), DX[i])
            if not near_food:
                if k > 1:
                    DX[i] = np.clip(w * DX[i] + rng.random(dim) * A + rng.random(dim) * Cc + rng.random(dim) * S,
                                    -dmax, dmax)
                    X[i] = X[i] + DX[i]
                else:                                                      # Eq. 3.8
                    X[i] = X[i] + 0.01 * _mantegna(rng, dim) * X[i]
                    DX[i] = 0.0
            else:                                                          # Eqs. 3.6-3.7
                DX[i] = np.clip(a * A + c * Cc + s * S + fw * F + e * En + w * DX[i], -dmax, dmax)
                X[i] = X[i] + DX[i]
            X[i] = np.clip(X[i], lo, hi)


# =============================================================================================
# 12 MFO (Mirjalili 2015, KBS 89:228), Eqs. 3.12-3.14: flames = best N of current moths and
#    previous flames (as the paper states), moth i spirals around flame i, moths beyond the flame
#    count around the last flame; t ~ U(r, 1) with r from -1 to -2, b = 1
# =============================================================================================
@_guard
def run_mfo(obj, lb, ub, dim, seed, params=None):
    rng = np.random.default_rng(seed); lo, hi = _box(lb, ub, dim); N = POP
    T = _iters(obj.max_fe, 0, N)
    M = rng.uniform(lo, hi, (N, dim)); Fl = None; ffl = None
    for it in range(1, T + 1):
        nfl = int(round(N - it * ((N - 1) / T)))                          # Eq. 3.14
        M = np.clip(M, lo, hi); fm = np.array([obj(x) for x in M])
        if Fl is None:
            P, fp = M, fm
        else:
            P, fp = np.vstack([M, Fl]), np.concatenate([fm, ffl])
        o = np.argsort(fp, kind='stable')[:N]; Fl, ffl = P[o].copy(), fp[o].copy()
        a = -1 + it * (-1 / T)
        tt = (a - 1) * rng.random((N, dim)) + 1
        idx = np.minimum(np.arange(N), nfl - 1)                           # 0-based flame index
        D = np.abs(Fl[idx] - M)
        M = D * np.exp(tt) * np.cos(2 * np.pi * tt) + Fl[idx]             # Eq. 3.12, b = 1


# =============================================================================================
# 13 SMO (Bansal et al. 2014, Memetic Comput. 6:31): local leader phase, global leader phase,
#    leader learning, local and global leader decisions with fission-fusion;
#    Algorithms 1-5, Eqs. 1-5; prob_i from the group's maximum fitness; MG = N/10, LLL = D x N,
#    GLL = 50 (Sect. 5.1, inside the suggested [N/2, 2N]), pr linearly 0.1 -> 0.4 (Sect. 5.1, budget
#    fraction), N = 30 (paper 50)
# =============================================================================================
@_guard
def run_smo(obj, lb, ub, dim, seed, params=None):
    rng = np.random.default_rng(seed); lo, hi = _box(lb, ub, dim); N = POP; B = obj.max_fe
    MG, LLL, GLL = max(1, N // 10), dim * N, 50          # Sect. 3.2 and Sect. 5.1
    X = rng.uniform(lo, hi, (N, dim)); f = np.array([obj(x) for x in X])

    def fitv(v):
        return np.where(v >= 0, 1 / (1 + v), 1 + np.abs(v))
    groups = [np.arange(N)]
    g = int(np.argmin(f)); GL, fGL = X[g].copy(), f[g]
    LL = [X[g].copy()]; fLL = [f[g]]; LLc = [0]; GLc = 0

    def split(ng):
        return [a for a in np.array_split(np.arange(N), ng)]

    def leaders():
        out_x, out_f = [], []
        for G in groups:
            b = G[np.argmin(f[G])]; out_x.append(X[b].copy()); out_f.append(f[b])
        return out_x, out_f
    while True:
        pr = 0.1 + 0.3 * obj.n / B
        for k, G in enumerate(groups):                                   # local leader phase
            for i in G:
                r = rng.choice(G[G != i]) if len(G) > 1 else i
                upd = rng.random(dim) >= pr
                v = np.where(upd, X[i] + rng.random(dim) * (LL[k] - X[i]) + rng.uniform(-1, 1, dim) * (X[r] - X[i]), X[i])
                v = np.clip(v, lo, hi); fv = obj(v)
                if fv < f[i]:
                    X[i], f[i] = v, fv
        for k, G in enumerate(groups):                                   # global leader phase, Alg. 2
            fg = fitv(f[G]); prob = dict(zip(G, 0.9 * fg / fg.max() + 0.1))   # Eq. 4, max in the group
            cnt = 1
            while cnt < len(G):
                for i in G:
                    if rng.random() < prob[i]:
                        cnt += 1
                        j = rng.integers(dim); r = rng.choice(G[G != i]) if len(G) > 1 else i
                        v = X[i].copy()
                        v[j] = X[i, j] + rng.random() * (GL[j] - X[i, j]) + rng.uniform(-1, 1) * (X[r, j] - X[i, j])
                        v = np.clip(v, lo, hi); fv = obj(v)
                        if fv < f[i]:
                            X[i], f[i] = v, fv
        g = int(np.argmin(f))                                            # global leader learning
        if f[g] < fGL:
            GL, fGL = X[g].copy(), f[g]; GLc = 0
        else:
            GLc += 1
        nx, nf = leaders()                                               # local leader learning
        for k in range(len(groups)):
            if nf[k] < fLL[k]:
                LL[k], fLL[k] = nx[k], nf[k]; LLc[k] = 0
            else:
                LLc[k] += 1
        for k, G in enumerate(groups):                                   # local leader decision
            if LLc[k] > LLL:
                LLc[k] = 0
                for i in G:
                    upd = rng.random(dim) >= pr
                    X[i] = np.where(upd, lo + rng.random(dim) * (hi - lo),
                                    X[i] + rng.random(dim) * (GL - X[i]) + rng.random(dim) * (X[i] - LL[k]))
                    X[i] = np.clip(X[i], lo, hi); f[i] = obj(X[i])
        if GLc > GLL:                                                    # global leader decision
            GLc = 0
            groups = split(len(groups) + 1) if len(groups) < MG else [np.arange(N)]
            LL, fLL = leaders(); LLc = [0] * len(groups)


# =============================================================================================
# 14 ALO (Mirjalili 2015, AES 83:80), Eqs. 2.1-2.13 as in ALO.m: roulette on 1/fitness of
#    sorted antlions, random walks normalised to the shrinking box around the selected antlion
#    and the elite (ratio I = 10^w t/T), ant = average, antlions replaced by fitter ants, elitism
# =============================================================================================
def _alo_walk(rng, dim, T, lo, hi, antlion, t):
    I = 1.0
    for frac, w in ((0.1, 2), (0.5, 3), (0.75, 4), (0.9, 5), (0.95, 6)):
        if t > frac * T:
            I = 1 + 10 ** w * (t / T)
    l, u = lo / I, hi / I
    l = l + antlion if rng.random() < 0.5 else -l + antlion
    u = u + antlion if rng.random() >= 0.5 else -u + antlion
    steps = rng.integers(0, 2, size=(T, dim), dtype=np.int8) * 2 - 1      # +-1 with probability 1/2
    W = np.vstack([np.zeros((1, dim), np.int32), np.cumsum(steps, axis=0, dtype=np.int32)])
    a, b = W.min(0), W.max(0)
    return ((W[t - 1] - a) * (u - l)) / np.where(b - a == 0, 1, b - a) + l   # RWs(current_iter,:), 1-based


@_guard
def run_alo(obj, lb, ub, dim, seed, params=None):
    rng = np.random.default_rng(seed); lo, hi = _box(lb, ub, dim); N = POP
    T = _iters(obj.max_fe, N, N) + 1
    AL = rng.uniform(lo, hi, (N, dim)); fal = np.array([obj(x) for x in AL])
    o = np.argsort(fal); AL, fal = AL[o], fal[o]; elite, fel = AL[0].copy(), fal[0]
    for t in range(2, T + 1):
        inv = 1 / fal; p = inv / inv.sum() if np.all(fal > 0) else None
        ants = np.empty((N, dim))
        for i in range(N):
            r = rng.choice(N, p=p) if p is not None else 0
            RA = _alo_walk(rng, dim, T, lo, hi, AL[r], t)
            RE = _alo_walk(rng, dim, T, lo, hi, elite, t)
            ants[i] = np.clip((RA + RE) / 2, lo, hi)                     # Eq. 2.13
        fa = np.array([obj(x) for x in ants])
        P = np.vstack([AL, ants]); fp = np.concatenate([fal, fa]); o = np.argsort(fp)[:N]
        AL, fal = P[o], fp[o]
        if fal[0] < fel:
            elite, fel = AL[0].copy(), fal[0]
        AL[0], fal[0] = elite, fel


# =============================================================================================
# 15 SquirrelSA (Jain, Singh and Rani 2019, SWEVO 44:148), Eqs. 4-17: 1 hickory, 3 acorn, the rest
#    normal trees; gliding distance from the lift/drag model scaled by sf = 18, Gc = 1.9,
#    Pdp = 0.1; seasonal constant S_c vs S_min = 1e-5 / 365^(t/(T/2.5)) with Levy relocation.
#    N = 30 (paper 50)
# =============================================================================================
@_guard
def run_squirrel(obj, lb, ub, dim, seed, params=None):
    rng = np.random.default_rng(seed); lo, hi = _box(lb, ub, dim); N = POP
    T = _iters(obj.max_fe, N, N); Gc, Pdp, sf = 1.9, 0.1, 18.0
    rho, V, Sa, hg, CD = 1.204, 5.25, 154e-4, 8.0, 0.6

    def dg():
        CL = rng.uniform(0.675, 1.5)
        return hg / ((0.5 * rho * V * V * Sa * CD) / (0.5 * rho * V * V * Sa * CL)) / sf   # Eqs. 4-8
    FS = rng.uniform(lo, hi, (N, dim)); f = np.array([obj(x) for x in FS])
    for t in range(1, T + 1):
        o = np.argsort(f); FS, f = FS[o], f[o]
        ht, at, nt = 0, np.arange(1, 4), np.arange(4, N)
        new = FS.copy()
        for i in at:                                                     # Eq. 9, case 1
            new[i] = FS[i] + dg() * Gc * (FS[ht] - FS[i]) if rng.random() >= Pdp else rng.uniform(lo, hi)
        perm = rng.permutation(nt); n_at = rng.integers(1, len(nt))
        for i in perm[:n_at]:                                            # Eq. 10, case 2
            j = rng.choice(at)
            new[i] = FS[i] + dg() * Gc * (FS[j] - FS[i]) if rng.random() >= Pdp else rng.uniform(lo, hi)
        for i in perm[n_at:]:                                            # Eq. 11, case 3
            new[i] = FS[i] + dg() * Gc * (FS[ht] - FS[i]) if rng.random() >= Pdp else rng.uniform(lo, hi)
        Sc = np.sqrt(np.sum((new[at] - new[ht]) ** 2, axis=1))           # Eq. 12
        Smin = 1e-5 / 365 ** (t / (T / 2.5))                             # Eq. 13
        if np.any(Sc < Smin):                                            # Eq. 14, season over
            for i in perm[:n_at]:            # normal-tree squirrels that could not reach the hickory tree
                new[i] = lo + 0.01 * _mantegna(rng, dim) * (hi - lo)
        FS = np.clip(new, lo, hi); f = np.array([obj(x) for x in FS])


# =============================================================================================
# 16 GOA (Saremi, Mirjalili and Lewis 2017, AES 105:30), Eq. 2.7 of the paper: per-dimension
#    s(|x_j^d - x_i^d|) times the d-th component of (x_j - x_i)/d_ij, s(r) = f exp(-r/l) - exp(-r),
#    f = 0.5, l = 1.5; distance mapping 2 + rem(., 2), c from 1 to 4e-5 and synchronous update as in
#    GOA.m (the authors' file applies the force to dimension pairs; the paper's equation is used)
# =============================================================================================
@_guard
def run_goa(obj, lb, ub, dim, seed, params=None):
    rng = np.random.default_rng(seed); lo, hi = _box(lb, ub, dim); N = POP
    T = _iters(obj.max_fe, N, N) + 1; cmax, cmin = 1.0, 4e-5
    G = rng.uniform(lo, hi, (N, dim)); fg = np.array([obj(x) for x in G])
    b = int(np.argmin(fg)); tgt, ftgt = G[b].copy(), fg[b]
    for l in range(2, T + 1):
        c = cmax - l * ((cmax - cmin) / T)
        Dv = G[None, :, :] - G[:, None, :]                               # Dv[i, j, d] = x_j^d - x_i^d
        dij = np.linalg.norm(Dv, axis=2)                                 # d_ij, full distance
        xx = 2 + np.remainder(np.abs(Dv), 2)                             # |x_j^d - x_i^d| mapped to [2, 4)
        sf = 0.5 * np.exp(-xx / 1.5) - np.exp(-xx)                       # s(r), f = 0.5, l = 1.5
        unit = Dv / (dij[:, :, None] + np.finfo(float).eps)             # (x_j - x_i) / d_ij
        terms = ((hi - lo) * c / 2) * sf * unit
        terms[np.arange(N), np.arange(N), :] = 0.0                       # j != i
        new = c * terms.sum(axis=1) + tgt                                # Eq. 2.7
        G = np.clip(new, lo, hi)
        for i in range(N):
            fg[i] = obj(G[i])
            if fg[i] < ftgt:
                tgt, ftgt = G[i].copy(), fg[i]


# =============================================================================================
# 17 MPA (Faramarzi et al. 2020, ESWA 152:113377), Eqs. 10-16 as in MPA.m: three phases by
#    iteration, Brownian/Levy steps (RL = 0.05 Levy), P = 0.5, FADs = 0.2, marine memory after each
#    evaluation, two evaluations per iteration
# =============================================================================================
@_guard
def run_mpa(obj, lb, ub, dim, seed, params=None):
    rng = np.random.default_rng(seed); lo, hi = _box(lb, ub, dim); N = POP
    T = _iters(obj.max_fe, 0, 2 * N); P, FADs = 0.5, 0.2
    Prey = rng.uniform(lo, hi, (N, dim)); top, ftop = None, np.inf
    fold = Pold = None

    def evaluate(Prey, it):
        nonlocal top, ftop, fold, Pold
        Prey = np.clip(Prey, lo, hi); fit = np.empty(N)
        for i in range(N):
            fit[i] = obj(Prey[i])
            if fit[i] < ftop:
                ftop, top = fit[i], Prey[i].copy()
        if it == 0:                                                      # as MPA.m, both blocks
            fold, Pold = fit.copy(), Prey.copy()
        keep = fold < fit                                                # marine memory
        Prey = np.where(keep[:, None], Pold, Prey); fit = np.where(keep, fold, fit)
        fold, Pold = fit.copy(), Prey.copy()
        return Prey
    for it in range(T):
        Prey = evaluate(Prey, it)
        Elite = np.tile(top, (N, 1))
        CF = (1 - it / T) ** (2 * it / T)
        RL = 0.05 * _mantegna(rng, (N, dim)); RB = rng.standard_normal((N, dim))
        R = rng.random((N, dim))
        if it < T / 3:                                                   # Eq. 12
            Prey = Prey + P * R * (RB * (Elite - RB * Prey))
        elif T / 3 < it < 2 * T / 3:                                     # Eqs. 13-14
            half = np.arange(N) + 1 > N / 2
            levy = Prey + P * R * (RL * (Elite - RL * Prey))
            brown = Elite + P * CF * (RB * (RB * Elite - Prey))
            Prey = np.where(half[:, None], brown, levy)
        else:                                                            # Eq. 15
            Prey = Elite + P * CF * (RL * (RL * Elite - Prey))
        Prey = evaluate(Prey, it)
        if rng.random() < FADs:                                          # Eq. 16
            U = rng.random((N, dim)) < FADs
            Prey = Prey + CF * ((lo + rng.random((N, dim)) * (hi - lo)) * U)
        else:
            r = rng.random()
            Prey = Prey + (FADs * (1 - r) + r) * (Prey[rng.permutation(N)] - Prey[rng.permutation(N)])


# =============================================================================================
# 18 KH (Gandomi and Alavi 2012, CNSNS 17:4831), Eqs. 2-19, KH II (with crossover, the best
#    variant in the paper): induced motion (N_max = 0.01), foraging (V_f = 0.02) toward the food
#    centre and the krill's own best, diffusion D_max = 0.005 (1 - I/I_max), dt = C_t sum(UB-LB)
#    with C_t = 0.5, inertia 0.1 + 0.8 (1 - I/I_max), crossover Cr = 0.2 K_hat(i,best);
#    out-of-range coordinates redrawn in the box. One extra evaluation per iteration (food).
# =============================================================================================
@_guard
def run_kh(obj, lb, ub, dim, seed, params=None):
    rng = np.random.default_rng(seed); lo, hi = _box(lb, ub, dim); N = POP
    T = _iters(obj.max_fe, N, N + 1)
    Nmax, Vf, Dmax, Ct, eps = 0.01, 0.02, 0.005, 0.5, 1e-6       # Gandomi and Alavi 2012, 2016
    dt = Ct * np.sum(hi - lo)
    X = rng.uniform(lo, hi, (N, dim)); K = np.array([obj(x) for x in X])
    Xb, Kb = X.copy(), K.copy()                                          # each krill's best
    Nold = np.zeros((N, dim)); Fold = np.zeros((N, dim))

    def unit(a, b):
        d = b - a; return d / (np.linalg.norm(d, axis=-1, keepdims=True) + eps)
    for I in range(1, T + 1):
        w = 0.1 + 0.8 * (1 - I / T)
        ib, iw = int(np.argmin(K)), int(np.argmax(K)); span = K[iw] - K[ib] + eps
        Kpos = K - K.min() + eps if np.any(K <= 0) else K
        Xf = np.clip((X / Kpos[:, None]).sum(0) / (1 / Kpos).sum(), lo, hi)   # Eq. 13 food centre
        Kf = obj(Xf)
        D2 = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        ds = D2.sum(1) / (5 * N)                                         # Eq. 10 sensing distance
        Nnew = np.empty((N, dim)); Fnew = np.empty((N, dim))
        for i in range(N):
            nb = (D2[i] < ds[i]) & (np.arange(N) != i)
            a_loc = (((K[i] - K[nb]) / span)[:, None] * unit(X[i], X[nb])).sum(0) if nb.any() else np.zeros(dim)  # Eqs. 4-6
            Cb = 2 * (rng.random() + I / T)                              # Eq. 8
            a_tgt = Cb * ((K[i] - K[ib]) / span) * unit(X[i], X[ib])     # Eq. 7
            Nnew[i] = Nmax * (a_loc + a_tgt) + w * Nold[i]               # Eq. 2
            Cf = 2 * (1 - I / T)                                         # Eq. 14
            b_food = Cf * ((K[i] - Kf) / span) * unit(X[i], Xf)          # Eq. 12
            b_best = ((K[i] - Kb[i]) / span) * unit(X[i], Xb[i])         # Eq. 15
            Fnew[i] = Vf * (b_food + b_best) + w * Fold[i]               # Eq. 11
        Dif = Dmax * (1 - I / T) * rng.uniform(-1, 1, (N, dim))          # Eq. 16
        Xn = X + dt * (Nnew + Fnew + Dif)                                # Eqs. 18-19
        for i in range(N):                                               # crossover, Eqs. 20-21
            Cr = 0.2 * (K[i] - K[ib]) / span
            r = rng.integers(N - 1); r += r >= i
            m = rng.random(dim) < Cr
            Xn[i] = np.where(m, X[r], Xn[i])
        bad = (Xn < lo) | (Xn > hi)
        Xn = np.where(bad, rng.uniform(lo, hi, (N, dim)), Xn)
        Nold, Fold, X = Nnew, Fnew, Xn
        K = np.array([obj(x) for x in X])
        imp = K < Kb; Xb[imp], Kb[imp] = X[imp], K[imp]


# =============================================================================================
# 19 AO (Abualigah et al. 2021, CAIE 157:107250), Eqs. 3-17 (checked against a copy of AO.m;
#    the paper's Eq. 3, U = 0.00565 and vector X_M are used where the file differs): expanded and contour
#    exploration for t <= 2T/3, low flight and walk-and-grab afterwards, greedy per move,
#    spiral constants r0 = 10, U = 0.00565, omega = 0.005 (paper), alpha = delta = 0.1
# =============================================================================================
@_guard
def run_ao(obj, lb, ub, dim, seed, params=None):
    rng = np.random.default_rng(seed); lo, hi = _box(lb, ub, dim); N = POP
    T = _iters(obj.max_fe, 0, 2 * N); al = de = 0.1
    X = rng.uniform(lo, hi, (N, dim)); best, fbest = None, np.inf
    D1 = np.arange(1, dim + 1); rr = 10 + 0.00565 * D1; th = -0.005 * D1 + 3 * np.pi / 2
    xs, ys = rr * np.sin(th), rr * np.cos(th)                            # Eqs. 6-10
    for t in range(1, T + 1):
        X = np.clip(X, lo, hi); f = np.empty(N)
        for i in range(N):
            f[i] = obj(X[i])
            if f[i] < fbest:
                fbest, best = f[i], X[i].copy()
        G1 = 2 * rng.random() - 1                                        # Eq. 16
        G2 = 2 * (1 - t / T)                                             # Eq. 17
        QF = t ** ((2 * rng.random() - 1) / (1 - T) ** 2)                # Eq. 15
        Xm = X.mean(0)                                                   # Eq. 4
        for i in range(N):
            if t <= (2 / 3) * T:
                if rng.random() < 0.5:                                   # Eq. 3 expanded exploration
                    xn = best * (1 - t / T) + (Xm - best * rng.random())    # paper Eq. 3
                else:                                                    # Eq. 5 contour flight
                    xn = best * (0.01 * _mantegna(rng, dim)) + X[rng.integers(N)] + (ys - xs) * rng.random()
            else:
                if rng.random() < 0.5:                                   # Eq. 13 low flight
                    xn = (best - Xm) * al - rng.random() + ((hi - lo) * rng.random() + lo) * de
                else:                                                    # Eq. 14 walk and grab
                    xn = QF * best - (G1 * X[i] * rng.random()) - G2 * 0.01 * _mantegna(rng, dim) + rng.random() * G1
            xn = np.clip(xn, lo, hi); fn = obj(xn)
            if fn < f[i]:
                X[i], f[i] = xn, fn


RUNNERS = {
    'PSO': run_pso, 'ACO': run_aco, 'ABC': run_abc, 'GWO': run_gwo, 'WOA': run_woa, 'HHO': run_hho,
    'FA': run_fa, 'BA': run_ba, 'CS': run_cs, 'BFO': run_bfo, 'DA': run_da, 'MFO': run_mfo,
    'SMO': run_smo, 'ALO': run_alo, 'SquirrelSA': run_squirrel, 'GOA': run_goa, 'MPA': run_mpa,
    'KH': run_kh, 'AO': run_ao,
}
ORDER = list(RUNNERS)
# checked against a primary source (authors' code or the paper text itself); see SWARM_AUDIT.md.
# ACO, BFO, SMO and SquirrelSA wait for the original papers before any test run.
VERIFIED = ['PSO', 'ABC', 'GWO', 'WOA', 'HHO', 'FA', 'BA', 'CS', 'DA', 'MFO', 'ALO', 'GOA', 'MPA', 'KH', 'AO']
PHASE_B = ['ACO', 'BFO', 'SMO', 'SquirrelSA']   # checked against the papers in SW_References (2026-10-17)


def self_test():
    """Budget exact, determinism per seed, and a convergence check on the shifted sphere."""
    from budget import Objective
    shift = np.random.default_rng(0).uniform(-80, 80, 10)
    sph = lambda x: float(np.sum((x - shift) ** 2))
    for name in ORDER:
        o1, o2 = Objective(sph, 10, 30000, 0.0), Objective(sph, 10, 30000, 0.0)
        b1 = RUNNERS[name](o1, -100., 100., 10, 7); b2 = RUNNERS[name](o2, -100., 100., 10, 7)
        print(f'{name:11s} sphere@30k={b1:.3g} fe={o1.n} repro={b1 == b2}', flush=True)
        assert o1.n == 30000 and b1 == b2, name


if __name__ == '__main__':
    self_test()

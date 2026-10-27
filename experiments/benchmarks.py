# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""
Classical benchmark suite (Yao, Liu & Lin 1999; as used in Mirjalili et al.
GWO 2014 / WOA 2016 / Heidari et al. HHO 2019 appendices).

F1-F7:  unimodal, scalable dimension
F8-F13: multimodal, scalable dimension
F14-F23: multimodal, fixed low dimension

Each entry: name -> (function, lower_bound, upper_bound, default_dim, known_global_min)
All functions take a 2D array X of shape (pop_size, dim) and return a 1D array
of shape (pop_size,) — vectorized for speed.
"""
import numpy as np


def f1_sphere(X):
    return np.sum(X ** 2, axis=1)


def f2_schwefel222(X):
    return np.sum(np.abs(X), axis=1) + np.prod(np.abs(X), axis=1)


def f3_schwefel12(X):
    # sum_{i=1}^{D} (sum_{j=1}^{i} x_j)^2
    cs = np.cumsum(X, axis=1)
    return np.sum(cs ** 2, axis=1)


def f4_schwefel221(X):
    return np.max(np.abs(X), axis=1)


def f5_rosenbrock(X):
    x = X[:, :-1]
    xp1 = X[:, 1:]
    return np.sum(100.0 * (xp1 - x ** 2) ** 2 + (x - 1) ** 2, axis=1)


def f6_step(X):
    return np.sum(np.floor(X + 0.5) ** 2, axis=1)


def f7_quartic_noise(X):
    D = X.shape[1]
    idx = np.arange(1, D + 1)
    return np.sum(idx * (X ** 4), axis=1) + np.random.uniform(0, 1, size=X.shape[0])


def f8_schwefel226(X):
    D = X.shape[1]
    return 418.9829 * D - np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1)


def f9_rastrigin(X):
    D = X.shape[1]
    return np.sum(X ** 2 - 10 * np.cos(2 * np.pi * X) + 10, axis=1)


def f10_ackley(X):
    D = X.shape[1]
    s1 = np.sum(X ** 2, axis=1)
    s2 = np.sum(np.cos(2 * np.pi * X), axis=1)
    return (-20 * np.exp(-0.2 * np.sqrt(s1 / D))
            - np.exp(s2 / D) + 20 + np.e)


def f11_griewank(X):
    D = X.shape[1]
    idx = np.arange(1, D + 1)
    s = np.sum(X ** 2, axis=1) / 4000.0
    p = np.prod(np.cos(X / np.sqrt(idx)), axis=1)
    return s - p + 1


def _u(x, a, k, m):
    r = np.zeros_like(x)
    r = np.where(x > a, k * (x - a) ** m, r)
    r = np.where(x < -a, k * (-x - a) ** m, r)
    return r


def f12_penalized1(X):
    D = X.shape[1]
    y = 1 + (X + 1) / 4.0
    yp1 = np.concatenate([y[:, 1:], np.zeros((X.shape[0], 1))], axis=1)
    term = 10 * (np.sin(np.pi * y[:, 0])) ** 2
    mid = np.sum((y[:, :-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * yp1[:, :-1]) ** 2), axis=1)
    last = (y[:, -1] - 1) ** 2
    penalty = np.sum(_u(X, 10, 100, 4), axis=1)
    return (np.pi / D) * (term + mid + last) + penalty


def f13_penalized2(X):
    D = X.shape[1]
    x = X
    xp1 = np.concatenate([X[:, 1:], np.zeros((X.shape[0], 1))], axis=1)
    term1 = np.sin(3 * np.pi * x[:, 0]) ** 2
    mid = np.sum((x[:, :-1] - 1) ** 2 * (1 + np.sin(3 * np.pi * xp1[:, :-1]) ** 2), axis=1)
    last = (x[:, -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[:, -1]) ** 2)
    penalty = np.sum(_u(X, 5, 100, 4), axis=1)
    return 0.1 * (term1 + mid + last) + penalty


# ---- Fixed-dimension multimodal (F14-F23) ----

_F14_A = np.array([[-32, -16, 0, 16, 32] * 5,
                    sum(([v] * 5 for v in [-32, -16, 0, 16, 32]), [])])


def f14_shekel_foxholes(X):
    a = _F14_A  # 2x25
    out = np.zeros(X.shape[0])
    for n in range(X.shape[0]):
        x = X[n]
        s = 0.0
        for j in range(25):
            s += 1.0 / (j + 1 + (x[0] - a[0, j]) ** 6 + (x[1] - a[1, j]) ** 6)
        out[n] = 1.0 / (1.0 / 500.0 + s) if (1.0 / 500.0 + s) != 0 else 1e10
    return out


_F15_A = np.array([0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627,
                    0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
_F15_B = 1.0 / np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])


def f15_kowalik(X):
    out = np.zeros(X.shape[0])
    for n in range(X.shape[0]):
        x = X[n]
        b = _F15_B
        num = x[0] * (b ** 2 + b * x[1])
        den = b ** 2 + b * x[2] + x[3]
        out[n] = np.sum((_F15_A - num / den) ** 2)
    return out


def f16_six_hump_camel(X):
    x1, x2 = X[:, 0], X[:, 1]
    return (4 * x1 ** 2 - 2.1 * x1 ** 4 + (x1 ** 6) / 3.0
            + x1 * x2 - 4 * x2 ** 2 + 4 * x2 ** 4)


def f17_branin(X):
    x1, x2 = X[:, 0], X[:, 1]
    a, b, c, r, s, t = 1, 5.1 / (4 * np.pi ** 2), 5 / np.pi, 6, 10, 1 / (8 * np.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


def f18_goldstein_price(X):
    x1, x2 = X[:, 0], X[:, 1]
    p1 = (1 + (x1 + x2 + 1) ** 2 * (19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2))
    p2 = (30 + (2 * x1 - 3 * x2) ** 2 * (18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2))
    return p1 * p2


_HART3_A = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
_HART3_C = np.array([1, 1.2, 3, 3.2])
_HART3_P = np.array([[0.3689, 0.1170, 0.2673],
                      [0.4699, 0.4387, 0.7470],
                      [0.1091, 0.8732, 0.5547],
                      [0.03815, 0.5743, 0.8828]])


def f19_hartman3(X):
    out = np.zeros(X.shape[0])
    for n in range(X.shape[0]):
        x = X[n]
        s = 0.0
        for i in range(4):
            inner = np.sum(_HART3_A[i] * (x - _HART3_P[i]) ** 2)
            s += _HART3_C[i] * np.exp(-inner)
        out[n] = -s
    return out


_HART6_A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                      [0.05, 10, 17, 0.1, 8, 14],
                      [3, 3.5, 1.7, 10, 17, 8],
                      [17, 8, 0.05, 10, 0.1, 14]])
_HART6_C = np.array([1, 1.2, 3, 3.2])
_HART6_P = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                      [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                      [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                      [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])


def f20_hartman6(X):
    out = np.zeros(X.shape[0])
    for n in range(X.shape[0]):
        x = X[n]
        s = 0.0
        for i in range(4):
            inner = np.sum(_HART6_A[i] * (x - _HART6_P[i]) ** 2)
            s += _HART6_C[i] * np.exp(-inner)
        out[n] = -s
    return out


_SHEKEL_A = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6],
                       [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1],
                       [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
_SHEKEL_C = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])


def _shekel(X, m):
    out = np.zeros(X.shape[0])
    for n in range(X.shape[0]):
        x = X[n]
        s = 0.0
        for i in range(m):
            denom = np.sum((x - _SHEKEL_A[i]) ** 2) + _SHEKEL_C[i]
            s += 1.0 / denom
        out[n] = -s
    return out


def f21_shekel5(X):
    return _shekel(X, 5)


def f22_shekel7(X):
    return _shekel(X, 7)


def f23_shekel10(X):
    return _shekel(X, 10)


# name -> (func, lb, ub, default_dim, known_global_min)
BENCHMARKS = {
    "F1": (f1_sphere, -100, 100, 30, 0.0),
    "F2": (f2_schwefel222, -10, 10, 30, 0.0),
    "F3": (f3_schwefel12, -100, 100, 30, 0.0),
    "F4": (f4_schwefel221, -100, 100, 30, 0.0),
    "F5": (f5_rosenbrock, -30, 30, 30, 0.0),
    "F6": (f6_step, -100, 100, 30, 0.0),
    "F7": (f7_quartic_noise, -1.28, 1.28, 30, 0.0),
    "F8": (f8_schwefel226, -500, 500, 30, 0.0),  # min at x=420.9687 per dim; f=0 there
    "F9": (f9_rastrigin, -5.12, 5.12, 30, 0.0),
    "F10": (f10_ackley, -32, 32, 30, 0.0),
    "F11": (f11_griewank, -600, 600, 30, 0.0),
    "F12": (f12_penalized1, -50, 50, 30, 0.0),
    "F13": (f13_penalized2, -50, 50, 30, 0.0),
    "F14": (f14_shekel_foxholes, -65.536, 65.536, 2, 1.0),
    "F15": (f15_kowalik, -5, 5, 4, 0.0003075),
    "F16": (f16_six_hump_camel, -5, 5, 2, -1.0316),
    "F17": (f17_branin, [-5, 0], [10, 15], 2, 0.398),
    "F18": (f18_goldstein_price, -2, 2, 2, 3.0),
    "F19": (f19_hartman3, 0, 1, 3, -3.86),
    "F20": (f20_hartman6, 0, 1, 6, -3.32),
    "F21": (f21_shekel5, 0, 10, 4, -10.1532),
    "F22": (f22_shekel7, 0, 10, 4, -10.4028),
    "F23": (f23_shekel10, 0, 10, 4, -10.5363),
}

UNIMODAL = ["F1", "F2", "F3", "F4", "F5", "F6", "F7"]
MULTIMODAL_SCALABLE = ["F8", "F9", "F10", "F11", "F12", "F13"]
MULTIMODAL_FIXED = ["F14", "F15", "F16", "F17", "F18", "F19", "F20", "F21", "F22", "F23"]

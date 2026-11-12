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
"""CEC2020 real-world constrained problems from the official C++ code
(Kumar et al. 2020; C++ port by V. Stanovov, GPL-3, linked at run time, not vendored).

rw_official(code) returns the same interface as r1_lib.rw_problem:
func(X) -> penalised objective, func.parts(x) -> (f, violation), lb, ub, dim, 0.0.
Violation = sum max(0, g) + sum max(0, |h| - 1e-4), the CEC2020 tolerance for equalities.
A non-finite penalised value is replaced by 1e30. Build, see experiments/rwx/README.md: g++ -O2 -std=c++14 -shared -fPIC -I<eigen-3.4.0> cec2020rwc.cpp -o cec2020rwc.so.
"""
import os
from ctypes import CDLL, c_double, c_int, POINTER
import numpy as np

D = [9,11,7,6,9,38,48,2,3,3,7,7,5,10,7,14,3,4,4,2,5,9,5,7,4,22,10,10,4,3,4,5,30]
GN = [0,0,14,1,2,0,0,2,1,3,4,9,3,10,11,15,4,4,5,3,8,10,8,7,7,86,3,9,1,8,0,6,30]
HN = [8,9,0,4,4,32,38,0,1,0,4,0,0,0,0,0,0,0,0,0,0,1,3,0,0,0,0,0,0,0,0,0,0]
PENALTY = 1e6
EQ_TOL = 1e-4
NONFINITE = 1e30   # penalised value for nan/inf, so no algorithm sees a non-finite fitness
_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        path = os.environ.get("CEC2020RWC_SO",
                              os.path.join(os.path.dirname(os.path.abspath(__file__)), "cec2020rwc.so"))
        L = CDLL(path)
        L.get_bounds.argtypes = [c_int, POINTER(c_double), POINTER(c_double)]
        L.get_bounds.restype = None
        L.cec20_func.argtypes = [POINTER(c_double)] * 1 + [c_int] + [POINTER(c_double)] * 3
        L.cec20_func.restype = None
        _LIB = L
    return _LIB


def rw_official(code):
    k = int(code[2:])
    n, ng, nh = D[k - 1], GN[k - 1], HN[k - 1]
    L = _lib()
    lo, up = (c_double * n)(), (c_double * n)()
    L.get_bounds(k, lo, up)
    lb, ub = np.array(lo[:], float), np.array(up[:], float)
    xb, fb = (c_double * n)(), (c_double * 1)()
    gb, hb = (c_double * max(ng, 1))(), (c_double * max(nh, 1))()

    def parts(x):
        x = np.asarray(x, float).ravel()
        for i in range(n):
            xb[i] = x[i]
        L.cec20_func(xb, k, fb, gb, hb)
        v = 0.0
        if ng:
            v += float(np.sum(np.maximum(0.0, np.array(gb[:ng]))))
        if nh:
            v += float(np.sum(np.maximum(0.0, np.abs(np.array(hb[:nh])) - EQ_TOL)))
        return float(fb[0]), v

    def func(X):
        X = np.atleast_2d(X)
        out = np.empty(len(X))
        for i, x in enumerate(X):
            f, v = parts(x)
            y = f + PENALTY * v
            out[i] = y if np.isfinite(y) else NONFINITE  # e.g. RC25 with R = Ro gives log(1) = 0 in a divisor
        return out
    func.parts = parts
    return func, lb, ub, n, 0.0

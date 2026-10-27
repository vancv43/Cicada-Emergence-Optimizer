# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""Self-built tuning suite (PROTOCOL.md deviation 2, 2026-10-10). Pure numpy, no external package.

12 classical base functions, each shifted by o and rotated by an orthogonal M (both drawn from a fixed seed
per function and instance), domain [-100, 100]^D, f(o) = 0 exactly. Disjoint from the CEC2022 test suite in
its shift vectors, rotation matrices and function composition (CEC2022 hybrid/composition functions are not
used); several base functions (e.g. Rosenbrock, Rastrigin, Levy) also occur inside CEC2022, as in every
standard suite.
"""
import numpy as np

E = np.e


def _sphere(z): return float(z @ z)
def _ellipsoid(z): d = len(z); return float(np.sum(10 ** (6 * np.arange(d) / max(1, d - 1)) * z * z))
def _bent_cigar(z): return float(z[0] ** 2 + 1e6 * np.sum(z[1:] ** 2))
def _discus(z): return float(1e6 * z[0] ** 2 + np.sum(z[1:] ** 2))
def _rosenbrock(z): z = z + 1.0; return float(np.sum(100 * (z[1:] - z[:-1] ** 2) ** 2 + (z[:-1] - 1) ** 2))
def _rastrigin(z): return float(np.sum(z * z - 10 * np.cos(2 * np.pi * z) + 10))
def _ackley(z):
    d = len(z); return float(-20 * np.exp(-0.2 * np.sqrt(z @ z / d)) - np.exp(np.sum(np.cos(2 * np.pi * z)) / d) + 20 + E)
def _griewank(z): return float(z @ z / 4000 - np.prod(np.cos(z / np.sqrt(np.arange(1, len(z) + 1)))) + 1)
def _weierstrass(z, a=0.5, b=3.0, kmax=20):
    k = np.arange(kmax + 1); ak = a ** k; bk = b ** k
    s = np.sum(ak[None, :] * np.cos(2 * np.pi * bk[None, :] * (z[:, None] + 0.5)))
    return float(s - len(z) * np.sum(ak * np.cos(np.pi * bk)))
def _levy(z):
    w = 1 + z / 4
    return float(np.sin(np.pi * w[0]) ** 2 + np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
                 + (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2))
def _happycat_core(z, alpha=0.125):
    # HappyCat with its minimum moved to z = 0: evaluate at x = z - 1, whose minimum is x = -1
    x = z - 1.0; d = len(x); r2 = x @ x; s = np.sum(x)
    return abs(r2 - d) ** (2 * alpha) + (0.5 * r2 + s) / d + 0.5
def _schwefel12(z): return float(np.sum(np.cumsum(z) ** 2))

# (name, base function, input scale so that [-100, 100] maps onto the function's usual domain)
FUNCS = [
    ('sphere', _sphere, 1.0), ('ellipsoid', _ellipsoid, 1.0), ('bent_cigar', _bent_cigar, 1.0),
    ('discus', _discus, 1.0), ('schwefel12', _schwefel12, 1.0), ('rosenbrock', _rosenbrock, 0.02048),
    ('rastrigin', _rastrigin, 0.0512), ('ackley', _ackley, 0.32768), ('griewank', _griewank, 6.0),
    ('weierstrass', _weierstrass, 0.005), ('levy', _levy, 0.1), ('happycat', _happycat_core, 0.05),
]


def _rotation(rng, d):
    q, r = np.linalg.qr(rng.normal(size=(d, d))); return q * np.sign(np.diag(r))


class TuningFunction:
    def __init__(self, idx, dim, instance):
        self.name, self._f, self._s = FUNCS[idx]
        rng = np.random.default_rng(20261010 + 1000 * idx + instance)
        self.o = rng.uniform(-80, 80, dim); self.M = _rotation(rng, dim); self.dim = dim
        self.optimum = 0.0

    def __call__(self, x):
        z = self.M @ ((np.asarray(x, float) - self.o) * self._s)
        return float(self._f(z))


def check():
    """f(o) == 0 and f(random) > 0 for every function, D in (10, 20), instances 1..3."""
    rng = np.random.default_rng(0)
    for D in (10, 20):
        for i in range(len(FUNCS)):
            for inst in (1, 2, 3):
                f = TuningFunction(i, D, inst)
                assert abs(f(f.o)) < 1e-9, (f.name, D, inst, f(f.o))
                vals = [f(rng.uniform(-100, 100, D)) for _ in range(20)]
                assert min(vals) > 0, (f.name, D, min(vals))
    return True


if __name__ == '__main__':
    print('tuning suite check:', check())

# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""One budget wrapper used by every algorithm, so FE accounting is identical across methods."""
import numpy as np


class BudgetExhausted(Exception):
    pass


def cec_checkpoints(dim, max_fe):
    """CEC2022 recording points floor(D^(k/5-3) * MaxFES), k = 0..15."""
    return sorted({max(1, min(max_fe, int(dim ** (k / 5.0 - 3.0) * max_fe))) for k in range(16)})


class Objective:
    def __init__(self, f, dim, max_fe, f_opt=0.0):
        self.f = f; self.dim = dim; self.max_fe = int(max_fe); self.f_opt = f_opt
        self.n = 0; self.best = np.inf; self.best_x = None
        self.ck = cec_checkpoints(dim, self.max_fe); self._ci = 0; self.trace = []

    def __call__(self, x):
        if self.n >= self.max_fe:
            raise BudgetExhausted
        x = np.asarray(x, dtype=float).ravel()
        y = float(self.f(x)); self.n += 1
        if y < self.best:
            self.best = y; self.best_x = x.copy()
        while self._ci < len(self.ck) and self.n >= self.ck[self._ci]:
            self.trace.append(self.best); self._ci += 1
        return y

    def error(self, v=None):
        e = (self.best if v is None else v) - self.f_opt
        return 0.0 if e < 1e-8 else float(e)

    def error_trace(self):
        t = list(self.trace) + [self.best] * (len(self.ck) - len(self.trace))
        return [self.error(v) for v in t]

# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""Fast adapter for the official CEC2022 Python reference (P-N-Suganthan/2022-SO-BO).
The reference re-reads its input_data/*.txt files with np.loadtxt on EVERY evaluation.
Here the module gets a private numpy namespace whose loadtxt is memoised by absolute path,
so numerics are unchanged (same arrays) but disk I/O happens once per file.
Equivalence with the unpatched harness is checked by verify_fast_harness.py."""
import os, sys, types, functools
import numpy as np
# self-contained copy of the official reference (CEC2022.py + input_data) so the package is self-contained
_CEC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cec2022_data')
sys.path.insert(0, _CEC_DIR)
import CEC2022 as _ref

@functools.lru_cache(maxsize=None)
def _load(path):
    a = np.loadtxt(path); a.setflags(write=False); return a

def _cached_loadtxt(fname, *a, **k):
    p = fname if os.path.isabs(fname) else os.path.join(_CEC_DIR, fname)
    return _load(os.path.abspath(p)).copy()

import numba as _nb

@_nb.njit(cache=True)
def _rotate_nb(x, nx, Mr):
    # identical loop order and left-to-right accumulation as the reference rotatefunc
    xrot = np.zeros(nx)
    for i in range(nx):
        for j in range(nx):
            xrot[i] = xrot[i] + x[j] * Mr[i, j]
    return xrot

def _rotatefunc(x, nx, Mr):
    return _rotate_nb(np.ascontiguousarray(x, dtype=np.float64), int(nx), np.ascontiguousarray(Mr, dtype=np.float64))

_ref.rotatefunc = _rotatefunc

_ns = types.ModuleType('np_cached'); _ns.__dict__.update(np.__dict__); _ns.loadtxt = _cached_loadtxt
_ref.np = _ns

OPTIMA = {1:300.,2:400.,3:600.,4:800.,5:900.,6:1800.,7:2000.,8:2200.,9:2300.,10:2400.,11:2600.,12:2700.}

class CEC2022Function:
    def __init__(self, fn, dim):
        assert dim in (2, 10, 20)
        self._f = _ref.cec2022_func(func_num=fn); self.dim = dim; self.fn = fn; self.optimum = OPTIMA[fn]
    def __call__(self, x):
        x = np.asarray(x, float).reshape(self.dim, 1)
        self._f.values(x); return float(self._f.ObjFunc[0])

def cec2022_benchmarks(dim):
    return {f'CEC22_F{fn}': (CEC2022Function(fn, dim), -100.0, 100.0, OPTIMA[fn]) for fn in range(1, 13)}

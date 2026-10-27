# Van Van Chau(1,2), Minh Tuan Nguyen(1), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam,
#     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     1518 Pracharat 1 Road, Wongsawang, Bang Sue, Bangkok 10800, Thailand, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding author
#
"""CEC2017 bound-constrained suite through the unmodified official C code (cec17_test_func.cpp from
github.com/P-N-Suganthan/CEC2017-BoundContrained, CEC17_fast_pow.zip). F2 is excluded as in the
official suite. Optimum value F*_i = 100 i. The C code reads input_data/ relative to the working
directory, so every evaluation runs with this directory as cwd (set once per process)."""
import ctypes
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, 'libcec17.so')
FUNCS = [1] + list(range(3, 31))


def _build():
    cxx = 'clang++' if subprocess.run(['which', 'clang++'], capture_output=True).returncode == 0 else 'g++'
    subprocess.run([cxx, '-O2', '-shared', '-fPIC', '-I' + os.path.join(HERE, 'shim'), '-o', LIB,
                    os.path.join(HERE, 'wrap.cpp')], check=True)


_lib = None


def _load():
    global _lib
    if _lib is None:
        if not os.path.exists(LIB):
            _build()
        _lib = ctypes.CDLL(LIB)
        _lib.cec17.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                               ctypes.c_int, ctypes.c_int, ctypes.c_int]
        os.chdir(HERE)
    return _lib


class CEC2017Function:
    def __init__(self, fn, dim):
        assert fn in FUNCS and dim in (10, 30, 50, 100), (fn, dim)
        self.fn, self.dim, self.optimum = fn, dim, 100.0 * fn
        self.lib = _load()
        self._f = np.zeros(1)

    def __call__(self, x):
        x = np.ascontiguousarray(x, dtype=np.float64)
        self.lib.cec17(x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       self._f.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), self.dim, 1, self.fn)
        return float(self._f[0])

    def shift(self):
        return np.loadtxt(os.path.join(HERE, 'input_data', f'shift_data_{self.fn}.txt')).ravel()[:self.dim]

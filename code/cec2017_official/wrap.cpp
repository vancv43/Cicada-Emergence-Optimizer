// Thin C wrapper around the unmodified official CEC2017 code (P-N-Suganthan/CEC2017-BoundContrained).
#include <stdlib.h>
double *OShift, *M, *y, *z, *x_bound;
int ini_flag = 0, n_flag, func_flag, *SS;
#include "cec17_test_func.cpp"
extern "C" void cec17(double *x, double *f, int nx, int mx, int func_num) { cec17_test_func(x, f, nx, mx, func_num); }

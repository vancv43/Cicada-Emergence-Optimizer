/*
 * Cicada Emergence Optimizer: A Hybrid Success-History Differential Evolution with Staggered Brood Restarts
 * Authors: Van Van Chau (1,2), Minh Tuan Nguyen (1,*), and Phayung Meesad (2,*)
 * (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
 *     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam
 *     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
 * (2) Department of Information Technology Management, Faculty of Information Technology and Digital Innovation,
 *     King Mongkut's University of Technology North Bangkok (KMUTNB),
 *     1518 Pracharat 1 Rd, Wongsawang, Bangsue, Bangkok 10800, Thailand
 *     phayung.m@itd.kmutnb.ac.th
 * (*) Corresponding authors
 * Repository: https://github.com/vancv43/Cicada-Emergence-Optimizer (MIT License)
 */
// Thin C wrapper around the unmodified official CEC2017 code (P-N-Suganthan/CEC2017-BoundContrained).
#include <stdlib.h>
double *OShift, *M, *y, *z, *x_bound;
int ini_flag = 0, n_flag, func_flag, *SS;
#include "cec17_test_func.cpp"
extern "C" void cec17(double *x, double *f, int nx, int mx, int func_num) { cec17_test_func(x, f, nx, mx, func_num); }

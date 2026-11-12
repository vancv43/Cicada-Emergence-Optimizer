#!/bin/bash
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
# Recomputes every table of the paper from the saved runs in ../runs and results/; no optimization run.
# analyze_rev3.py part (b) re-runs CEO on CEC2022 at D = 10 to count gates and part (e) re-times the reference
# loops, so its timing numbers vary from machine to machine.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
echo '[analysis] analyze_v4.py'
$PY analyze_v4.py
echo '[analysis] manuscript_tables.py'
$PY manuscript_tables.py
echo '[analysis] analyze_swarm.py'
$PY analyze_swarm.py
echo '[analysis] analyze_rw_audited.py'
$PY analyze_rw_audited.py
echo '[analysis] analyze_power.py'
$PY analyze_power.py
echo '[analysis] analyze_power50.py'
$PY analyze_power50.py
echo '[analysis] analyze_sens.py'
$PY analyze_sens.py
echo '[analysis] analyze_rev1_b.py'
$PY analyze_rev1_b.py
echo '[analysis] analyze_rev1_c.py'
$PY analyze_rev1_c.py
echo '[analysis] select_rev2.py'
$PY select_rev2.py
echo '[analysis] analyze_rev2.py'
$PY analyze_rev2.py
echo '[analysis] analyze_table_cells.py'
$PY analyze_table_cells.py
echo '[analysis] analyze_rev3.py'
$PY analyze_rev3.py
echo '[analysis] analyze_rw19.py'
$PY analyze_rw19.py
echo '[analysis] analyze_rw_d1d2.py'
$PY analyze_rw_d1d2.py
echo '[analysis] feas_rank_audited.py'
$PY feas_rank_audited.py
echo '[analysis] make_supp_rw_tables.py'
$PY make_supp_rw_tables.py
echo '[analysis] done, outputs in results/analysis/'

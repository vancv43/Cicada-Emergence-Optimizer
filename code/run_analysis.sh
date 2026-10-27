#!/bin/bash
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
echo '[analysis] done, outputs in results/analysis/'

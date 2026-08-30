#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_all.sh -- run the full multipole-ladder climb l = 2..7 to completion,
#               then build the framed figures.  Uses the FAST hybrid engine
#               (short ExpMPO warm-up + two-site TDVP) with the validated,
#               converged default settings (dt = 0.1, t = 12, chi = 24).
#
# Resumable: if interrupted, just re-run -- each register resumes from its
# last checkpoint in out_mps/.
#
# Rough single-core guide (8 GB laptop; multi-threaded BLAS is faster):
#   l2 ~1 min, l3 ~1 min, l4 ~20 min, l5 ~40 min, l6 ~1.5 h, l7 ~2.5-4 h.
#   Peak memory < 1 GB throughout.  Total l2..l7 fits in one overnight run.
#   l4 and l5 alone already give a clean 4-point trend (M = 5,12,21,32).
# ---------------------------------------------------------------------------
set -e
TFINAL=${TFINAL:-12}    # evolution time (units R_S/c)
DT=${DT:-0.1}           # integrator step (validated to ~1e-3)
CHI=${CHI:-24}          # bond-dimension cap (converged; trunc err ~1e-6)
DSNAP=${DSNAP:-1}       # snapshot spacing

for L in 2 3 4 5 6 7; do
  echo "==================  l_max = $L  =================="
  python3 run_mps_climb.py --lmax $L --tfinal $TFINAL --dt $DT \
          --chimax $CHI --dsnap $DSNAP --walltime 1e9
done

echo "==================  figures  =================="
python3 make_mps_figures.py
echo "Done.  Results in out_mps/  (climb_l*_d4.json/.npz + figM1..figM4)."
echo "Results in out_mps/ are ready for analysis and paper integration."

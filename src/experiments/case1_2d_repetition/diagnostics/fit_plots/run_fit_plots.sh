#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
SCRIPT="src/experiments/case1_2d_smoke.py"
FIT_ROOT="src/experiments/case1_2d_repetition/diagnostics/fit_plots"

cd "$PROJECT_ROOT"

python "$SCRIPT" \
  --seed 148 \
  --n-subject 2000 \
  --coef-type base4 \
  --sigma2-function sin2 \
  --rho 0.9 \
  --R 4 \
  --S 25 \
  --p0 4 \
  --beta 2.0,1.0,-1.0,0.5 \
  --sigma2 1.0 \
  --covariance-mode exchangeable_varying_sigma \
  --signal-bandwidth 0.18 \
  --variance-bandwidth 0.18 \
  --plot-functions \
  --plot-a-indices all \
  --plot-max-a-panels 16 \
  --output-root "$FIT_ROOT/A__n2000_base4_sin2_rho0p9_seed148_rep1"

python "$SCRIPT" \
  --seed 127 \
  --n-subject 2000 \
  --coef-type base6 \
  --sigma2-function sin \
  --rho 0.9 \
  --R 4 \
  --S 25 \
  --p0 4 \
  --beta 2.0,1.0,-1.0,0.5 \
  --sigma2 1.0 \
  --covariance-mode exchangeable_varying_sigma \
  --signal-bandwidth 0.18 \
  --variance-bandwidth 0.18 \
  --plot-functions \
  --plot-a-indices all \
  --plot-max-a-panels 16 \
  --output-root "$FIT_ROOT/sigma_miae__n2000_base6_sin_rho0p9_seed127_rep0"

python "$SCRIPT" \
  --seed 145 \
  --n-subject 2000 \
  --coef-type base2 \
  --sigma2-function sin2 \
  --rho 0.9 \
  --R 4 \
  --S 25 \
  --p0 4 \
  --beta 2.0,1.0,-1.0,0.5 \
  --sigma2 1.0 \
  --covariance-mode exchangeable_varying_sigma \
  --signal-bandwidth 0.18 \
  --variance-bandwidth 0.18 \
  --plot-functions \
  --plot-a-indices all \
  --plot-max-a-panels 16 \
  --output-root "$FIT_ROOT/sigma_rmise__n2000_base2_sin2_rho0p9_seed145_rep2"

#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
SCRIPT="src/experiments/case2_3d_smoke.py"
FIT_ROOT="src/experiments/case2_3d_repetition/diagnostics/fit_plots"
EXPORT_SCRIPT="$FIT_ROOT/export_all_A_plots.py"

cd "$PROJECT_ROOT"

run_one() {
  local label="$1"
  local seed="$2"
  local coef_type="$3"
  local signal_bandwidth="$4"
  local output_root="$FIT_ROOT/$label"

  python "$SCRIPT" \
    --output-root "$output_root" \
    --seed "$seed" \
    --n-subject 5000 \
    --coef-type "$coef_type" \
    --sigma2-function sin2 \
    --rho 0.9 \
    --R 6 \
    --S 27 \
    --p0 4 \
    --beta 2.0,1.0,-1.0,0.5 \
    --sigma2 1.0 \
    --covariance-mode exchangeable_varying_sigma \
    --signal-bandwidth "$signal_bandwidth" \
    --variance-bandwidth 0.18 \
    --ridge 1e-4 \
    --a-eval-mode anchor_grid \
    --a-eval-num-points 500 \
    --a-eval-grid quantile \
    --a-interp linear \
    --plot-functions \
    --plot-a-indices all \
    --plot-max-a-panels 16

  python "$EXPORT_SCRIPT" "$output_root"
}

run_one "A_base1__n5000_base1_sin2_rho0p9_h0p18_seed147_rep24" 147 base1 0.18
run_one "A_base2__n5000_base2_sin2_rho0p9_h0p18_seed124_rep1" 124 base2 0.18
run_one "A_base3__n5000_base3_sin2_rho0p9_h0p08_seed139_rep16" 139 base3 0.08
run_one "A_base4__n5000_base4_sin2_rho0p9_h0p18_seed136_rep13" 136 base4 0.18
run_one "A_base5__n5000_base5_sin2_rho0p9_h0p08_seed124_rep1" 124 base5 0.08
run_one "A_base6__n5000_base6_sin2_rho0p9_h0p08_seed124_rep1" 124 base6 0.08

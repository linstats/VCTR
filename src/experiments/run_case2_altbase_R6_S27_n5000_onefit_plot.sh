#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Keep matplotlib/font caches out of the home directory on local machines and HPC.
CACHE_ROOT="${TMPDIR:-/tmp}/vctr-matplotlib-cache"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${CACHE_ROOT}/mplconfig}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg-cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

N_SUBJECT="${N_SUBJECT:-5000}"
SEED="${SEED:-123}"
COEF_TYPE="${COEF_TYPE:-base1}"
RHO="${RHO:-0.3}"
SIGMA2="${SIGMA2:-1.0}"
BETA="${BETA:-2.0,1.0,-1.0,0.5}"
COVARIANCE_MODE="${COVARIANCE_MODE:-exchangeable_varying_sigma}"
SIGNAL_BANDWIDTH="${SIGNAL_BANDWIDTH:-0.20}"
VARIANCE_BANDWIDTH="${VARIANCE_BANDWIDTH:-0.20}"
RIDGE="${RIDGE:-1e-4}"

# Indices are Python/NumPy 0-based. This default plots paper A[1,1] and A[4,1].
PLOT_A_INDICES="${PLOT_A_INDICES:-0:0,3:0}"
PLOT_MAX_A_PANELS="${PLOT_MAX_A_PANELS:-2}"

RUN_NAME="${RUN_NAME:-case2_R6_S27_n5000_${COEF_TYPE}_seed${SEED}_plot_$(date +%Y%m%d_%H%M%S)}"

python src/experiments/paired_case2_altbase_repetition.py \
  --n-subject-values "${N_SUBJECT}" \
  --coef-types "${COEF_TYPE}" \
  --n-rep 1 \
  --seed-base "${SEED}" \
  --R 6 \
  --S 27 \
  --p0 4 \
  --beta "${BETA}" \
  --sigma2 "${SIGMA2}" \
  --rho "${RHO}" \
  --rho-values "${RHO}" \
  --covariance-mode "${COVARIANCE_MODE}" \
  --signal-bandwidth "${SIGNAL_BANDWIDTH}" \
  --variance-bandwidth "${VARIANCE_BANDWIDTH}" \
  --ridge "${RIDGE}" \
  --n-jobs 1 \
  --run-name "${RUN_NAME}" \
  --save-data \
  --save-estimates \
  --plot-functions \
  --plot-a-indices "${PLOT_A_INDICES}" \
  --plot-max-a-panels "${PLOT_MAX_A_PANELS}"

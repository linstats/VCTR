#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/2026-tensor}"
CONDA_MODULE="${CONDA_MODULE:-miniconda/4.12}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-$HOME/conda-envs/vctr-py310}"
N_JOBS="${N_JOBS:-12}"
DRY_RUN="${DRY_RUN:-0}"

PBS_SCRIPT="${PBS_SCRIPT:-hpc/paired_case1_altbase_varying_var_parallel.pbs}"
SIGMA2_FUNCTIONS="${SIGMA2_FUNCTIONS:-sin:sin2:mixed}"
RUN_ROOT="${RUN_ROOT:-hpc_runs/a1a4_varying_sigma/hpc_raw_parts}"

PART_REPS=(4 4 4 4 4 4 3 3)
PART_SEEDS=(123 127 131 135 139 143 147 150)

cd "$PROJECT_ROOT"

echo "[submit] project_root=$PROJECT_ROOT"
echo "[submit] pbs_script=$PBS_SCRIPT"
echo "[submit] sigma2_functions=$SIGMA2_FUNCTIONS"
echo "[submit] run_root=$RUN_ROOT"
echo "[submit] n_jobs=$N_JOBS"
echo "[submit] dry_run=$DRY_RUN"

for idx in "${!PART_REPS[@]}"; do
  part_id=$((idx + 1))
  n_rep="${PART_REPS[$idx]}"
  seed_base="${PART_SEEDS[$idx]}"

  echo "[submit] part=${part_id} n_rep=${n_rep} seed_base=${seed_base} sigma2_functions=${SIGMA2_FUNCTIONS}"
  qsub_args=(
    qsub
    -v "PROJECT_ROOT=$PROJECT_ROOT,CONDA_MODULE=$CONDA_MODULE,CONDA_ENV_PATH=$CONDA_ENV_PATH,N_JOBS=$N_JOBS,SIGMA2_FUNCTIONS=$SIGMA2_FUNCTIONS,RUN_ROOT=$RUN_ROOT,PART_ID=$part_id,N_REP=$n_rep,SEED_BASE=$seed_base"
    "$PBS_SCRIPT"
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run]'
    printf ' %q' "${qsub_args[@]}"
    printf '\n'
  else
    "${qsub_args[@]}"
  fi
done

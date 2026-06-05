#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/2026-tensor}"
CONDA_MODULE="${CONDA_MODULE:-miniconda/4.12}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-$HOME/conda-envs/vctr-py310}"
N_JOBS="${N_JOBS:-12}"
DRY_RUN="${DRY_RUN:-0}"

PBS_SCRIPT="${PBS_SCRIPT:-hpc/paired_case2_altbase_remaining_parallel.pbs}"
RUN_ROOT="${RUN_ROOT:-hpc_case2_remaining_anchor500_h018_R6}"
MANIFEST_ROOT="${MANIFEST_ROOT:-src/experiments/paired_case2_altbase_repetition/remaining_manifests_anchor500_h018_R6}"

cd "$PROJECT_ROOT"

echo "[submit] project_root=$PROJECT_ROOT"
echo "[submit] pbs_script=$PBS_SCRIPT"
echo "[submit] run_root=$RUN_ROOT"
echo "[submit] manifest_root=$MANIFEST_ROOT"
echo "[submit] n_jobs=$N_JOBS"
echo "[submit] dry_run=$DRY_RUN"

module purge
module load "$CONDA_MODULE"
source activate "$CONDA_ENV_PATH"

echo "[submit] python=$(command -v python)"
python --version

python src/experiments/paired_case2_altbase_repetition/build_remaining_manifests.py \
  --output-dir "$MANIFEST_ROOT" \
  --n-parts 8

for part_id in 1 2 3 4 5 6 7 8; do
  manifest_path="${MANIFEST_ROOT}/part${part_id}.csv"
  echo "[submit] part=${part_id} manifest_path=${manifest_path}"
  qsub_args=(
    qsub
    -v "PROJECT_ROOT=$PROJECT_ROOT,CONDA_MODULE=$CONDA_MODULE,CONDA_ENV_PATH=$CONDA_ENV_PATH,N_JOBS=$N_JOBS,RUN_ROOT=$RUN_ROOT,PART_ID=$part_id,MANIFEST_PATH=$manifest_path"
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

#!/bin/bash
set -euo pipefail

PROJECT_ROOT=$HOME/2026-tensor
MANIFEST_DIR=$HOME/2026-tensor/hpc/manifests/case2_backfill_20260529

qsub -v PROJECT_ROOT=$PROJECT_ROOT,CONDA_MODULE=miniconda/4.12,CONDA_ENV_PATH=$HOME/conda-envs/vctr-py310,N_JOBS=12,MANIFEST_PATH=$MANIFEST_DIR/part1_missing.csv,RUN_NAME=run_case2_altbase_backfill_part1_20260529 hpc/paired_case2_altbase_backfill_parallel.pbs
qsub -v PROJECT_ROOT=$PROJECT_ROOT,CONDA_MODULE=miniconda/4.12,CONDA_ENV_PATH=$HOME/conda-envs/vctr-py310,N_JOBS=12,MANIFEST_PATH=$MANIFEST_DIR/part2_missing.csv,RUN_NAME=run_case2_altbase_backfill_part2_20260529 hpc/paired_case2_altbase_backfill_parallel.pbs
qsub -v PROJECT_ROOT=$PROJECT_ROOT,CONDA_MODULE=miniconda/4.12,CONDA_ENV_PATH=$HOME/conda-envs/vctr-py310,N_JOBS=12,MANIFEST_PATH=$MANIFEST_DIR/part3_missing.csv,RUN_NAME=run_case2_altbase_backfill_part3_20260529 hpc/paired_case2_altbase_backfill_parallel.pbs
qsub -v PROJECT_ROOT=$PROJECT_ROOT,CONDA_MODULE=miniconda/4.12,CONDA_ENV_PATH=$HOME/conda-envs/vctr-py310,N_JOBS=12,MANIFEST_PATH=$MANIFEST_DIR/part4_missing.csv,RUN_NAME=run_case2_altbase_backfill_part4_20260529 hpc/paired_case2_altbase_backfill_parallel.pbs
qsub -v PROJECT_ROOT=$PROJECT_ROOT,CONDA_MODULE=miniconda/4.12,CONDA_ENV_PATH=$HOME/conda-envs/vctr-py310,N_JOBS=12,MANIFEST_PATH=$MANIFEST_DIR/part5_missing.csv,RUN_NAME=run_case2_altbase_backfill_part5_20260529 hpc/paired_case2_altbase_backfill_parallel.pbs
qsub -v PROJECT_ROOT=$PROJECT_ROOT,CONDA_MODULE=miniconda/4.12,CONDA_ENV_PATH=$HOME/conda-envs/vctr-py310,N_JOBS=12,MANIFEST_PATH=$MANIFEST_DIR/part6_missing.csv,RUN_NAME=run_case2_altbase_backfill_part6_20260529 hpc/paired_case2_altbase_backfill_parallel.pbs
qsub -v PROJECT_ROOT=$PROJECT_ROOT,CONDA_MODULE=miniconda/4.12,CONDA_ENV_PATH=$HOME/conda-envs/vctr-py310,N_JOBS=12,MANIFEST_PATH=$MANIFEST_DIR/part7_missing.csv,RUN_NAME=run_case2_altbase_backfill_part7_20260529 hpc/paired_case2_altbase_backfill_parallel.pbs
qsub -v PROJECT_ROOT=$PROJECT_ROOT,CONDA_MODULE=miniconda/4.12,CONDA_ENV_PATH=$HOME/conda-envs/vctr-py310,N_JOBS=12,MANIFEST_PATH=$MANIFEST_DIR/part8_missing.csv,RUN_NAME=run_case2_altbase_backfill_part8_20260529 hpc/paired_case2_altbase_backfill_parallel.pbs

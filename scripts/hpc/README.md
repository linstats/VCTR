# NUS HPC batch templates

This directory contains PBS templates for paired Case 2 on the NUS `parallel` queue.

## Active template

Use:

- `paired_case2_reduced24_5parts.pbs`

for the current recommended production run:

- `n_subject-values 1000 2000`
- `coef-types sqrt quadratic bump sin`
- `rho-values 0.0 0.3 0.6 0.9`
- total `n_rep=10`
- `5` PBS jobs
- `2` repetitions per job
- `24` CPUs per job
- `128gb` memory per job

The older `paired_case2_full_parallel.pbs` remains in the repo as a reference for the earlier 12-core, 3-sample-size plan, but it is not the preferred submission path now.

## Why this reduced plan

The original full Case 2 grid with `n_subject-values 1000 1500 2000` and `n_rep=30` is too expensive for a reliable 48-hour single-job workflow.

The current production compromise keeps the model structure fixed:

- `R=5`
- `S=64`
- `coef_types` unchanged
- `rho_values` unchanged

and reduces cost by:

- dropping `n_subject=1500`
- using total `n_rep=10`
- splitting into `5` smaller jobs so each job fits the queue walltime better

Each job now runs:

- `2 * 4 * 4 * 2 = 64` fits

which is substantially safer under `48:00:00` walltime than the earlier larger parts.

## Queue fit

On this cluster:

- `serial` allows only `1` CPU
- `parallel` requires at least `12` CPUs
- the user-level observed practical ceiling is `96` CPUs total across concurrent jobs

This is why the production plan uses `24` CPUs per job and lets PBS queue excess jobs automatically.

## Recommended upload layout

Copy the repo to your HPC home once:

```bash
scp -r /Users/lin/Desktop/Research/2026-tensor e0829076@atlas9.nus.edu.sg:~
```

If you already copied it before, update it with `rsync` from your own machine instead.

## Python environment

This template assumes your own conda environment and runs:

```bash
module purge
module load miniconda/4.12
source activate $HOME/conda-envs/vctr-py310
```

inside the PBS job before launching Python.

If you move the environment or need a different module name, override them via `qsub -v`.

## Submission pattern

Keep the current canary running if it is already healthy, then submit these five production jobs:

```bash
cd ~/2026-tensor

qsub -v PROJECT_ROOT=$HOME/2026-tensor,CONDA_MODULE=miniconda/4.12,CONDA_ENV_PATH=$HOME/conda-envs/vctr-py310,N_JOBS=24,N_REP=2,SEED_BASE=123,RUN_NAME=run_case2_r24_ns1000_2000_part1 scripts/hpc/paired_case2_reduced24_5parts.pbs
qsub -v PROJECT_ROOT=$HOME/2026-tensor,CONDA_MODULE=miniconda/4.12,CONDA_ENV_PATH=$HOME/conda-envs/vctr-py310,N_JOBS=24,N_REP=2,SEED_BASE=125,RUN_NAME=run_case2_r24_ns1000_2000_part2 scripts/hpc/paired_case2_reduced24_5parts.pbs
qsub -v PROJECT_ROOT=$HOME/2026-tensor,CONDA_MODULE=miniconda/4.12,CONDA_ENV_PATH=$HOME/conda-envs/vctr-py310,N_JOBS=24,N_REP=2,SEED_BASE=127,RUN_NAME=run_case2_r24_ns1000_2000_part3 scripts/hpc/paired_case2_reduced24_5parts.pbs
qsub -v PROJECT_ROOT=$HOME/2026-tensor,CONDA_MODULE=miniconda/4.12,CONDA_ENV_PATH=$HOME/conda-envs/vctr-py310,N_JOBS=24,N_REP=2,SEED_BASE=129,RUN_NAME=run_case2_r24_ns1000_2000_part4 scripts/hpc/paired_case2_reduced24_5parts.pbs
qsub -v PROJECT_ROOT=$HOME/2026-tensor,CONDA_MODULE=miniconda/4.12,CONDA_ENV_PATH=$HOME/conda-envs/vctr-py310,N_JOBS=24,N_REP=2,SEED_BASE=131,RUN_NAME=run_case2_r24_ns1000_2000_part5 scripts/hpc/paired_case2_reduced24_5parts.pbs
```

Seed coverage:

- part1: `123, 124`
- part2: `125, 126`
- part3: `127, 128`
- part4: `129, 130`
- part5: `131, 132`

Together these provide the desired total `10` non-overlapping repetitions.

## Monitoring

Check queue status:

```bash
qstat -u e0829076
```

Inspect one job:

```bash
qstat -f <job_id>
```

Watch the incremental progress file written by one job:

```bash
cat src/experiments/paired_case2_repetition/run_case2_r24_ns1000_2000_part1/progress.json
```

and inspect CSV outputs:

```bash
ls src/experiments/paired_case2_repetition/run_case2_r24_ns1000_2000_part1/results
```

## Notes

- The repository default for paired runs is conceptually `ridge=0`; using `1e-4` is a numerical stabilization choice and should be described that way in notes or manuscript text.
- The current template already loads your conda environment inside the job, so interactive shell activation is not enough by itself; the PBS script must do it too.
- Do not reuse old run names like `run_case2_full_part1` if those directories already exist on HPC; the experiment script creates run directories with `exist_ok=False`.

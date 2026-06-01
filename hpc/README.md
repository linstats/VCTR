# `hpc/` 目录说明

这个目录存放 NUS HPC 相关说明文件，以及历史 PBS 脚本归档。

## 文件列表

### `archive/`

- 历史 PBS 脚本归档目录
- 当前包含：
  - `archive/paired_case2_reduced24_5parts.pbs`
  - `archive/paired_case2_full_parallel.pbs`
- 这两个脚本由于对应旧的模型/DGP 阶段，现统一作为历史参考保留

### `paired_case1_altbase_varsigma_parallel.pbs`

- 当前正式的 `paired_case1_altbase` varying-sigma HPC 模板
- 直接放在 `hpc/` 根目录
- 默认资源：
  - `parallel`
  - `12` CPUs
  - `24gb` memory
  - `36h` walltime
- 默认实验：
  - `n_subject = 1000 2000`
  - `coef_types = base1 base2 base3 base4`
  - `rho_values = 0.0 0.3 0.6 0.9`
  - `R = 4`, `S = 25`, `p0 = 4`
  - `beta = 2.0,1.0,-1.0,0.5`
  - `covariance_mode = exchangeable_varying_sigma`
  - `signal_bandwidth = 0.18`
  - `variance_bandwidth = 0.18`
  - `ridge = 1e-4`

### `paired_case2_altbase_varsigma_parallel.pbs`

- 当前正式的 `paired_case2_altbase` varying-sigma HPC 模板
- 直接放在 `hpc/` 根目录
- 默认资源：
  - `parallel`
  - `12` CPUs
  - `16gb` memory
  - `36h` walltime
- 默认实验：
  - `n_subject = 2000 5000`
  - `coef_types = base1 base2 base3 base4`
  - `rho_values = 0.0 0.3 0.6 0.9`
  - `R = 6`, `S = 27`, `p0 = 4`
  - `beta = 2.0,1.0,-1.0,0.5`
  - `covariance_mode = exchangeable_varying_sigma`
  - `signal_bandwidth = 0.20`
  - `variance_bandwidth = 0.20`
  - `ridge = 1e-4`
- 默认提交组织方式：
  - `10` parts
  - 每个 part `3` reps
  - `12` CPUs + `16gb` + `36h`

### `paired_case2_altbase_backfill_parallel.pbs`

- `paired_case2_altbase` 精确补跑的 PBS 模板
- 配合：
  - `src/experiments/paired_case2_altbase_repetition/audit_case2_hpc_parts.py`
  - `src/experiments/paired_case2_altbase_repetition/paired_case2_altbase_backfill.py`
- 默认资源：
  - `parallel`
  - `12` CPUs
  - `16gb` memory
  - `72h` walltime
- 通过环境变量传入：
  - `MANIFEST_PATH`
  - `RUN_NAME`
  - `N_JOBS`
- manifest 每行一个缺失任务，字段固定为：
  - `part,n_subject,coef_type,rho_true,rep,seed`

### `README.md`

- 本目录说明文件
- 只负责介绍 `hpc/` 目录当前结构与定位

## 当前定位

- `hpc/` 现在是仓库顶层的 HPC 入口目录
- 如果后续有新的正式 PBS 模板，建议直接放在 `hpc/` 下
- 旧模板统一放在 `hpc/archive/`，避免与现行脚本混淆

## 使用建议

- 如果你只是想了解 HPC 目录结构，先看本文件
- 如果你需要回看旧的批处理写法或较早期的 Case 2 方案，查看 `hpc/archive/`

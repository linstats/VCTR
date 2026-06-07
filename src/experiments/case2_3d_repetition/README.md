# Case 2 3D HPC 汇总

目前 `case2_3d_repetition` 根目录合并了两轮 HPC 结果：

- 第一轮：[`hpc_runs/a1a4_constant/`](hpc_runs/a1a4_constant/)
  - `coef_type = base1, base2, base3, base4`
  - `sigma2_function = constant`
- 第二轮：[`hpc_runs/a1a4_varsigma_a5a6_allsigma/`](hpc_runs/a1a4_varsigma_a5a6_allsigma/)
  - `coef_type = base1, base2, base3, base4` 且 `sigma2_function = sin, sin2, mixed`
  - `coef_type = base5, base6` 且 `sigma2_function = constant, sin, sin2, mixed`

合并后，根目录一共覆盖：

- `coef_type = base1, base2, base3, base4, base5, base6`
- `sigma2_function = constant, sin, sin2, mixed`
- `n_subject = 2000, 5000`
- `rho = 0.0, 0.3, 0.6, 0.9`
- 每个组合 `30 reps`

共同设置：

- `R = 6`, `S = 27`, `p0 = 4`
- `beta = (2.0, 1.0, -1.0, 0.5)`
- `sigma2 = 1.0`
- `covariance_mode = exchangeable_varying_sigma`

两轮 HPC 的差异设置：

- 第一轮 `a1a4_constant/`
  - `a_eval_mode = full`
  - `signal_bandwidth = 0.20`
  - `variance_bandwidth = 0.20`
  - `ridge = 1e-4`
- 第二轮 `a1a4_varsigma_a5a6_allsigma/`
  - `a_eval_mode = anchor_grid`
  - `a_eval_num_points = 500`
  - `signal_bandwidth = 0.18`
  - `variance_bandwidth = 0.18`
  - `ridge = 1e-4`

## 结果文件

[`raw_results.csv`](raw_results.csv) 是两轮 HPC 每个 fit 的逐次结果拼接表，一共合并了 `960 + 4800 = 5760` 条记录。

[`summary_results.csv`](summary_results.csv) 是固定 `n_subject` / `sigma2_function` / `coef_type` / `rho` / `a_eval_mode` 后、聚合 30 reps 的汇总表，共 `192` 个 config。

---

**注意点**：

- 第一轮 `a1a4_constant/` 的原始结果没有记录 `sigma2_function`、`a_eval_mode`、`a_eval_selected_points`、`rho_error`。
- 在根目录合并表中，这 32 个 constant configs 被统一补记为：
  - `sigma2_function = constant`
  - `a_eval_mode = full`
  - `a_eval_selected_points` 留空
- 由于缺少 `rho_error`，这 32 个 config 在 [`summary_results.csv`](summary_results.csv) 里的 `rho_error_mean/std` 与 `rho_mae/rho_rmse` 留空；但 `rho_abs_error` 仍然保留。

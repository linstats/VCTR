# Case 2 - 3D 张量模拟设置

目前 Case 2 的结果来自三轮 HPC 实验。

## 实验批次

第一轮：[`hpc_runs/a1a4_constant/`](hpc_runs/a1a4_constant/)

- `coef_type = base1, base2, base3, base4`
- `sigma2_function = constant`
- `signal_bandwidth = 0.20`

第二轮：[`hpc_runs/a1a4_varsigma_a5a6_allsigma/`](hpc_runs/a1a4_varsigma_a5a6_allsigma/)

- `coef_type = base1, base2, base3, base4`，且 `sigma2_function = sin, sin2, mixed`
- `coef_type = base5, base6`，且 `sigma2_function = constant, sin, sin2, mixed`
- `signal_bandwidth = 0.18`

第三轮：[`hpc_runs/a3a5a6_small_h_sensitivity/`](hpc_runs/a3a5a6_small_h_sensitivity/)

- 这是 A3/A5/A6 的小带宽敏感性实验。
- `coef_type = base3, base5, base6`
- `sigma2_function = constant, sin, sin2, mixed`
- 当 `n_subject = 2000` 时，`signal_bandwidth = 0.12, 0.14`
- 当 `n_subject = 5000` 时，`signal_bandwidth = 0.08, 0.10`

## 实验覆盖

完整实验覆盖：

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
- `ridge = 1e-4`

各轮实验的主要差异：

- 第一轮使用 `a_eval_mode = full`、`signal_bandwidth = 0.20`、`variance_bandwidth = 0.20`。
- 第二轮使用 `a_eval_mode = anchor_grid`、`a_eval_num_points = 500`、`signal_bandwidth = 0.18`、`variance_bandwidth = 0.18`。
- 第三轮继续使用 `anchor_grid` 和 `a_eval_num_points = 500`，固定 `variance_bandwidth = 0.18`，仅缩小 A3/A5/A6 的 `signal_bandwidth`。

## 结果文件

[`raw_results.csv`](raw_results.csv) 是每个 fit 的逐次结果拼接表，包含：

- 原主实验的 `5760` 条记录；
- A3/A5/A6 小带宽敏感性实验的 `5760` 条记录。

合计 `11520` 条记录。根表保留了原有的 32 个结果字段，其中实际带宽记录在 `signal_bandwidth_input` 和 `best_signal_bandwidth` 中。

[`summary_results.csv`](summary_results.csv) 包含 `192` 个 config，每个 config 汇总 `30 reps`：

- A1/A2/A4 保留原主实验结果；
- A3/A5/A6 在每个 `(coef_type, n_subject, rho, sigma2_function)` setting 内，分别从两种新候选带宽中选择 `miae_final_mean` 更小的一行。

## 注意点

- 第一轮 `a1a4_constant/` 的原始结果没有记录 `sigma2_function`、`a_eval_mode`、`a_eval_selected_points` 和 `rho_error`。
- 在根目录合并表中，这部分结果统一补记为：
  - `sigma2_function = constant`
  - `a_eval_mode = full`
  - `a_eval_selected_points` 留空
- 由于缺少 `rho_error`，对应 config 在 [`summary_results.csv`](summary_results.csv) 中的 `rho_error_mean/std` 和 `rho_mae/rho_rmse` 留空，但 `rho_abs_error` 仍然保留。
- A3/A5/A6 的带宽是按每个 simulation setting 的 `miae_final_mean` 单独选择，因此论文中应明确说明该结果采用 oracle bandwidth selection。

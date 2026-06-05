# HPC一共跑了三次

## `a1a4_constant/`

- `coef_type = base1, base2, base3, base4`
- `sigma2_function = constant`
- `n_subject = 1000, 2000`
- `rho = 0.0, 0.3, 0.6, 0.9`
- ``a1a4_constant/results/raw_results.csv`：
  - 是这 10 个 part 的逐次结果拼接表，一共合并了 `960` 条记录。
- ``a1a4_constant/results/summary_results.csv`：
  - 是对 `sigma2_function = constant` 下的合并结果再按
    `n_subject + coef_type + rho_true + covariance_mode + signal_bandwidth_method`
    聚合后的汇总表。
  - 每一行对应一个 `(n_subject, coef_type, rho)` 组合在 `30 reps` 上的均值和标准差。

## `a1a4_varying_sigma/`

- `coef_type = base1, base2, base3, base4`
- `sigma2_function = sin, sin2, mixed`
- `n_subject = 1000, 2000`
- `rho = 0.0, 0.3, 0.6, 0.9`
- `a1a4_varying_sigma/results/raw_results.csv`：
  - 是 `sin`、`sin2`、`mixed` 三类 `sigma2_function` 下全部 24 个 part 的逐次结果拼接表，一共合并了 `2880` 条记录。
- `a1a4_varying_sigma/results/summary_results.csv`：
  - 是对合并结果再按
    `n_subject + coef_type + rho_true + sigma2_function + covariance_mode + signal_bandwidth_method`
    聚合后的汇总表。
  - 每一行对应一个 `(n_subject, coef_type, rho, sigma2_function)` 组合在 `30 reps` 上的均值和标准差。

## `a5a6_allsigma/`

- `coef_type = base5, base6`
- `sigma2_function = constant, sin, sin2, mixed`
- `n_subject = 1000, 2000`
- `rho = 0.0, 0.3, 0.6, 0.9`
- `a5a6_allsigma/results/raw_results.csv`：
  - 是 `base5/base6`、`constant/sin/sin2/mixed` 全部 8 个 part 的逐次结果拼接表，一共合并了 `1920` 条记录。
- `a5a6_allsigma/results/summary_results.csv`：
  - 是对合并结果再按
    `n_subject + coef_type + rho_true + sigma2_function + covariance_mode + signal_bandwidth_method`
    聚合后的汇总表。
  - 每一行对应一个 `(n_subject, coef_type, rho, sigma2_function)` 组合在 `30 reps` 上的均值和标准差。

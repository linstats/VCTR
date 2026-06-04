# Anchor Check 摘要

这个目录保存了一个针对 Case 2 paired-eye VCTR 加速方案的本地验证实验，配置为：

- `n = 5000`
- `coef_type = base5`
- `rho = 0.6`
- `sigma2(t) = mixed`
- `R = 6`, `S = 27`
- `h = 0.18`

目标是比较默认 `full` 算法与 `anchor-grid` 近似算法。

## 1. Anchor vs Full

为了做公平比较，下表只使用共同的 `seed=123` 和 `124`。

所有误差指标均按 `原数值 × 100` 展示。

| 模式 | 平均时间 (s) | 相对 `full` 加速倍数 | `MIAE_final × 100` | `RMISE_final × 100` | `beta_RMSE_final × 100` | `sigma2_RMISE × 100` | `rho_MAE × 100` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `full` | 712.89 | 1.00x | 5.8516 | 8.4721 | 1.4134 | 94.8427 | 29.9458 |
| `anchor = 250` | 59.80 | 11.92x | 5.8526 | 8.4734 | 1.4140 | 94.8876 | 29.9518 |
| `anchor = 500` | 119.46 | 5.97x | 5.8519 | 8.4724 | 1.4132 | 94.8533 | 29.9469 |
| `anchor = 1000` | 232.36 | 3.07x | 5.8517 | 8.4722 | 1.4134 | 94.8448 | 29.9461 |

结论：

- 所有 `anchor` 设置都比 `full` 快得多。
- 在这组实验里，误差差异可以忽略不计。
- 对这两个共同 seed 来说，`anchor = 500` 和 `anchor = 1000` 与 `full` 几乎不可区分。

## 2. 不同 Anchor 设置之间的比较

下表使用全部五个 `anchor` seed，即 `123:127`，并汇报 `mean ± std`。

所有误差指标均按 `原数值 × 100` 展示。

| Anchor | 时间 mean ± std (s) | `MIAE_final × 100` | `RMISE_final × 100` | `beta_RMSE_final × 100` | `sigma2_RMISE × 100` | `rho_MAE × 100` |
|---|---:|---:|---:|---:|---:|---:|
| `250` | 59.54 ± 0.31 | 5.8563 ± 0.0954 | 8.5256 ± 0.1823 | 1.3578 ± 0.5268 | 97.0381 ± 2.8641 | 30.8906 ± 1.4598 |
| `500` | 119.67 ± 0.76 | 5.8556 ± 0.0954 | 8.5247 ± 0.1824 | 1.3576 ± 0.5271 | 97.0053 ± 2.8662 | 30.8866 ± 1.4598 |
| `1000` | 209.01 ± 32.07 | 5.8554 ± 0.0954 | 8.5244 ± 0.1824 | 1.3576 ± 0.5270 | 96.9967 ± 2.8668 | 30.8855 ± 1.4590 |

结论：

- 运行时间随 `anchor` 数单调上升，即 `250 < 500 < 1000`。
- 精度变化只出现在第 4 到第 5 位小数。
- 在这组实验里，`anchor = 500` 看起来是最均衡的速度-精度折中点。

## 文件

- 原始 fit 级结果：[results/raw_results.csv](/Users/lin/Desktop/Research/2026-tensor/src/experiments/paired_case2_altbase_repetition/anchor_check/results/raw_results.csv)
- 汇总结果：[results/summary_results.csv](/Users/lin/Desktop/Research/2026-tensor/src/experiments/paired_case2_altbase_repetition/anchor_check/results/summary_results.csv)

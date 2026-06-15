# Case 2 A3/A5/A6 小带宽敏感性实验

本目录完整记录 Case 2 中 A3/A5/A6 从发现异常、调整信号带宽，到完成
HPC 补充实验和更新最终汇总表的过程。这里是该带宽敏感性实验的主要审计记录。

## 原始问题

原主实验对 A3/A5/A6 主要使用固定信号带宽 `h_A = 0.18`。结果出现两个值得
检查的现象：

1. 相比 A1/A2/A4，A3/A5/A6 的配对估计 `A*` 需要较大的 `rho` 才能明显优于
   iid 初始估计 `A^dagger`，其中 A5 的改善最弱。
2. A3/A5/A6 的 `sigma^2(t)` MIAE 在 `n = 2000` 增加到 `n = 5000` 后反而
   增大，与通常的样本量趋势不符。

我们的解释假设是：A3/A5/A6 的函数形状较复杂，统一使用 `h_A = 0.18`
可能产生较强的过度平滑。Stage 2 使用 Stage 1 残差估计 `sigma^2(t)` 和
`rho`，因此 `A^dagger(t)` 的平滑偏差可能进入协方差估计，并进一步影响
Stage 3 的配对估计。该机制是根据算法结构和后续敏感性结果提出的解释，
不是单独完成的因果证明。

## 本地诊断

### 单次 pilot

首先固定以下 setting：

- `coef_type = base5`
- `sigma2_function = sin`
- `rho = 0.6`
- `variance_bandwidth = 0.18`
- `seed = 123`
- `n = 2000, h_A = 0.14`
- `n = 5000, h_A = 0.08`

结果、估计文件和函数图保存在：

```text
../../test/test_base5_small_h_pilot/
```

该 pilot 显示，减小 `h_A` 后，`A(t)`、`sigma^2(t)` 和 `rho` 的误差均明显
改善，且 `A*` 开始优于 `A^dagger`。

### 单次带宽网格

随后仍在 `base5 + sin + rho=0.6 + seed=123` 下测试：

- `n = 2000`: `h_A = 0.10, 0.12, 0.14, 0.16, 0.18`
- `n = 5000`: `h_A = 0.06, 0.08, 0.10, 0.12, 0.18`

结果保存在：

```text
../../test/test_base5_signal_bandwidth_grid/
```

该网格确认 `h_A = 0.18` 对 A5 存在明显过度平滑，并据此将 HPC 候选范围
缩小为：

- `n = 2000`: `h_A = 0.12, 0.14`
- `n = 5000`: `h_A = 0.08, 0.10`

## HPC 实验设计

补充实验覆盖：

- `coef_type = base3, base5, base6`
- `sigma2_function = constant, sin, sin2, mixed`
- `rho = 0.0, 0.3, 0.6, 0.9`
- `n = 2000`: `h_A = 0.12, 0.14`
- `n = 5000`: `h_A = 0.08, 0.10`
- `30` repetitions，seeds `123--152`
- `R = 6`, `S = 27`, `p0 = 4`
- `beta = (2.0, 1.0, -1.0, 0.5)`
- `variance_bandwidth = 0.18`
- `a_eval_mode = anchor_grid`，500 个 quantile anchors
- `ridge = 1e-4`

总规模为：

```text
3 coefficient functions x 4 variance functions x 4 rho values
x 4 (n, h_A) combinations x 30 repetitions = 5760 fits
```

任务被拆成 8 个 part，每个 part 包含 45 个完整 bundle、720 fits。同一个
`(coef_type, sigma2_function, seed)` bundle 中的 16 个
`(n, h_A, rho)` 任务保存在同一 part，以保持带宽比较使用相同 seed。

## 完整性检查

`merge_meta.json` 记录的合并结果为：

- 预期任务：5760
- 完成任务：5760
- 成功：5760
- 失败：0
- 缺失、重复、额外任务：均为 0
- 带宽不一致：0
- 敏感性汇总：192 行，每行 30 repetitions

## 主要结果

与原带宽结果相比，两种小带宽候选的 192 个结果全部降低了 final MIAE：

- 平均改善：31.5%
- 中位数改善：28.4%
- 改善范围：5.2%--65.4%

采用后述逐 setting 选择后，A3/A5/A6 的 `sigma^2(t)` MIAE 从
`n = 2000` 到 `n = 5000` 均下降，没有 setting 继续恶化：

- A3：平均下降 43.7%
- A5：平均下降 31.0%
- A6：平均下降 41.5%

更新后的 A3/A5/A6 中，`A*` 相对 `A^dagger` 的平均 MIAE 改善随 `rho`
增加：

- `rho = 0.0`: -0.3%
- `rho = 0.3`: 1.7%
- `rho = 0.6`: 9.1%
- `rho = 0.9`: 27.5%

因此，原先“复杂函数必须在很大的 `rho` 下才有改善”和“样本量增大后
方差估计反而变差”的异常现象，在小带宽实验中基本消失。这支持过度平滑
及其残差污染解释，但不单独构成该机制的严格证明。

## 最终汇总规则

Case 2 根目录 `summary_results.csv` 的处理规则为：

- A1/A2/A4 保留原主实验结果；
- A3/A5/A6 在每个
  `(coef_type, n_subject, rho, sigma2_function)` setting 内，从两个新候选
  带宽中选择 `miae_final_mean` 更小的一行；
- 根汇总最终仍包含 192 个 setting，每个 setting 为 30 repetitions。

选定带宽的次数为：

| n | coefficient | smaller candidate | larger candidate |
|---:|---|---:|---:|
| 2000 | A3 | 0.12: 10 | 0.14: 6 |
| 2000 | A5 | 0.12: 16 | 0.14: 0 |
| 2000 | A6 | 0.12: 5 | 0.14: 11 |
| 5000 | A3 | 0.08: 5 | 0.10: 11 |
| 5000 | A5 | 0.08: 16 | 0.10: 0 |
| 5000 | A6 | 0.08: 4 | 0.10: 12 |

## 解释限制

上述最终选择使用了模拟真值定义的 `miae_final_mean`，属于逐 setting 的
oracle bandwidth selection。同一批 repetitions 同时参与候选带宽比较和最终
结果报告，因此结果可能存在选择性乐观偏差，也不能直接对应真实数据中可用的
带宽选择程序。

在论文或报告中应明确称其为 oracle bandwidth sensitivity result。若要将其
作为完全数据驱动的主模拟结果，需要另行使用不依赖真实 `A(t)` 的选择规则，
例如交叉验证或独立 tuning/evaluation repetitions。

## 文件说明

```text
manifests/part1.csv ... part8.csv       HPC 精确任务清单
manifests/manifest_summary.json         任务覆盖和分片统计
hpc_raw_parts/part1/ ... part8/         各 HPC part 的逐任务输出
results/raw_results.csv                 5760 条敏感性实验结果
results/summary_results.csv             192 个带宽候选配置汇总
results/bandwidth_comparison.csv         原带宽与两个小带宽的比较表
provenance.csv                           各 part 结果来源
merge_meta.json                          完整性审计结果
run_config.json                          合并时的运行配置
```

敏感性实验的 5760 条 raw 结果已经追加到 Case 2 根目录 `raw_results.csv`。
原批次结果和本目录结果仍分别保留，便于回溯。

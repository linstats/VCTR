# Case1 - 2D张量模拟设置

目前 Case 1 一共跑了：

- `coef_type = base1, base2, base3, base4, base5, base6`
- `sigma2_function = constant, sin, sin2, mixed`
- `n_subject = 1000, 2000`
- `rho = 0.0, 0.3, 0.6, 0.9`
- 每个组合 `30 reps`

共同设置：

- `R = 4`, `S = 25`, `p0 = 4`
- `beta = (2.0, 1.0, -1.0, 0.5)`
- `sigma2 = 1.0`
- `covariance_mode = exchangeable_varying_sigma`
- `signal_bandwidth = 0.18`
- `variance_bandwidth = 0.18`
- `ridge = 1e-4`

## 结果文件

`raw_results.csv`：是每个 fit 的逐次结果拼接表，一共合并了 `192×30=5760` 条记录。

`summary_results.csv`：是固定 `n_subject` / `sigma2_function` /  `coef_type` / `rho` 、聚合 30 reps 后，共 `192` 个 config 的汇总表。

---

**小注意点**：以下 config 的 `rho_error` 没有被记录，所以在总表中留空了

`coef_type = base1, base2, base3, base4` &`sigma2_function = constant`

但记录了abs error，可以查看原始表格。

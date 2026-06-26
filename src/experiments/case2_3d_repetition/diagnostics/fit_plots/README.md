# fit_plots

这个目录用于保存 Case 2 单次 good fit 的复跑结果，用来单独查看 `A(t)` 和 `sigma^2(t)` 的估计曲线。

来源：

- 这些 config 是从 `case2_3d_repetition/raw_results.csv` 中挑出的 representative good fit。
- 挑选时先对齐当前 `summary_results.csv` 的带宽口径，再对每个 `coef_type` 按 `miae_final` 最小选择一个 run。
- 当前 6 个 run 统一使用 `n_subject = 5000`、`rho = 0.9`、`sigma2_function = sin2`。
- A1/A2/A4 使用原主实验带宽 `h_A = 0.18`；A3/A5/A6 使用当前 summary 的小带宽 oracle 口径。

目录约定：

- `A_base1__n5000_base1_sin2_rho0p9_h0p18_seed147_rep24/`
- `A_base2__n5000_base2_sin2_rho0p9_h0p18_seed124_rep1/`
- `A_base3__n5000_base3_sin2_rho0p9_h0p08_seed139_rep16/`
- `A_base4__n5000_base4_sin2_rho0p9_h0p18_seed136_rep13/`
- `A_base5__n5000_base5_sin2_rho0p9_h0p08_seed124_rep1/`
- `A_base6__n5000_base6_sin2_rho0p9_h0p08_seed124_rep1/`

每个子目录在运行后都会包含：

- `data/`
- `estimates/`
- `results/`
- `plots/`
- `plots_all_A/`

复跑方法：

```bash
bash src/experiments/case2_3d_repetition/diagnostics/fit_plots/run_fit_plots.sh
```

这些诊断结果只用于人工检查和画图，不作为正式汇总表的一部分。

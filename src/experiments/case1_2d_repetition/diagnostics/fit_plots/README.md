# fit_plots

这个目录保存 3 个单次 good fit 的复跑结果，用来单独查看 `A(t)` 和 `sigma^2(t)` 的估计曲线。

来源：

- 这些 config 是从 [raw_results.csv](/Users/lin/Desktop/Research/2026-tensor/src/experiments/case1_2d_repetition/raw_results.csv) 中挑出的 good fit。
- `rep` 只是原始来源标识；真正控制复跑数据复现的是 `seed` 和完整 config。

目录约定：

- `A__n2000_base4_sin2_rho0p9_seed148_rep1/`
- `sigma_miae__n2000_base6_sin_rho0p9_seed127_rep0/`
- `sigma_rmise__n2000_base2_sin2_rho0p9_seed145_rep2/`

每个子目录在运行后都会包含：

- `data/`
- `estimates/`
- `results/`
- `plots/`

复跑方法：

```bash
bash src/experiments/case1_2d_repetition/diagnostics/fit_plots/run_fit_plots.sh
```

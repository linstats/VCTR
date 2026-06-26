# Diagnostics

这个目录保存 `case2_3d_repetition` 的诊断型产物。

- 这里的内容用于单独复跑少量 config、画 `A(t)` 和 `sigma^2(t)`，方便人工检查。
- 这里的结果不属于正式主表，不会写回：
  - `raw_results.csv`
  - `summary_results.csv`
  - `hpc_runs/`

当前计划中的诊断任务：

- `fit_plots/`: 从 Case 2 已完成结果中挑选少量 representative good fit，复跑并输出函数图。

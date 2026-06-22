# Hyperparameter CV outputs

本目录保存 GRAPE `X-only VCTR` full three-stage hyperparameter CV 的精简导出结果。

来源 run:

```text
src/experiments/grape/runs/hyperpar_cv/x_only_grid_v1/
```

核心结论：

| image_type | S | R | h | hbar | rmse_std | rmse_iop |
| :-- | :-- | --: | --: | --: | --: | --: |
| CFP | `3x4x1` | 1 | 1.80 | 0.25 | 0.988956 | 4.132781 |
| ROI | `6x2x1` | 1 | 0.85 | 0.30 | 0.962024 | 4.020231 |

导出文件：

- `x_only_grid_v1_summary_best_by_image.csv`: CFP/ROI 的最佳配置。
- `x_only_grid_v1_summary_all.csv`: 53,760 个候选的完整 candidate-level 排序结果。
- `x_only_grid_v1_top20_by_image.csv`: 每个 image type 的 top 20 candidate。
- `x_only_grid_v1_best_per_sr_top15.csv`: 每个 image type 中按 `(S,R)` 去重后的 top 15 structural configs。
- `x_only_grid_v1_checks.json`: 行数、重复候选、finite metrics、fold count 等完整性检查。

未复制到本目录：

- `fold_metrics.csv`: fold-level 明细较大，保留在 run directory。
- `shard_*/`: HPC checkpoint/audit 文件，保留在 run directory。

排序规则：

```text
image_type, rmse_std, rmse_iop, mape_std_pct, mape_iop_pct
```

主选择指标是 `subject_id` grouped 5-fold CV 下的 standardized-scale `rmse_std`。

# GRAPE bandwidth CV summaries

本目录保存可进入 repo 的精选 bandwidth CV 汇总结果。完整运行产物位于 `src/experiments/grape/runs/cv_bandwidth/`，默认不进 git。

## Files

- `v1_summary_all.csv`：v1 全量 40 个 `(image_type, S, R)` 搜索结果。
- `v1_summary_best_by_image.csv`：v1 中 CFP/ROI 各自最佳组合。
- `v1_failures.csv`：v1 中非 success 任务；实际均为 `no_eligible_signal_bandwidth`，没有程序失败。
- `v2_summary_all.csv`：v2 top candidates 扩大 `h` grid 结果。
- `v2_summary_best_by_image.csv`：v2 中 CFP/ROI 各自最佳组合。
- `v3_summary_all.csv`：v3 near-global `h` check 结果。
- `v3_summary_best_by_image.csv`：v3 最终 CFP/ROI 候选。

## Current Main Candidates

| image_type | S | R | h | hbar | signal_cv_score |
| :-- | :-- | --: | --: | --: | --: |
| CFP | `2x2x1` | 1 | 1.00 | 1.20 | 1.216699 |
| ROI | `2x2x1` | 1 | 0.80 | 0.40 | 1.207834 |

These are the current candidates for downstream final fitting, residual diagnostics, and ablation/baseline comparisons.

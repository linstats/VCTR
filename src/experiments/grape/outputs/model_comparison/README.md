# GRAPE model comparison outputs

本目录保存可进入 repo 的模型对比精简结果。完整运行产物位于 `src/experiments/grape/runs/model_comparison/`，默认不进 git。

## Files

- `v1_best_models_summary_metrics.csv`: model-level held-out metrics。
- `v1_best_models_fold_metrics.csv`: fold-level held-out metrics。
- `v2_patient_grouped_summary_metrics.csv`: patient-level grouped model-level held-out metrics。
- `v2_patient_grouped_fold_metrics.csv`: patient-level grouped fold-level held-out metrics。

## Current Main Result: v2_patient_grouped

固定 bandwidth CV v3 选出的低复杂度配置：

| image_type | S | R | h | hbar |
| :-- | :-- | --: | --: | --: |
| CFP | `2x2x1` | 1 | 1.00 | 1.20 |
| ROI | `2x2x1` | 1 | 0.80 | 0.40 |

Patient-level grouped 5-fold held-out CV 的主要结果：

| image_type | best_model | best_rmse_iop | best_mae_iop | next_best_rmse_iop | best_vs_z_only_rmse |
| :-- | :-- | --: | --: | --: | --: |
| CFP | `x_only_vctr` | 4.200719 | 3.366127 | 4.900183 | -0.699464 |
| ROI | `x_only_vctr` | 4.123525 | 3.279062 | 4.894819 | -0.776658 |

Model-level RMSE ordering:

| image_type | model | rmse_iop | mae_iop |
| :-- | :-- | --: | --: |
| CFP | `x_only_vctr` | 4.200719 | 3.366127 |
| CFP | `z_only_linear` | 4.900183 | 3.909173 |
| CFP | `xz_iid_vctr` | 4.925098 | 3.950620 |
| CFP | `xz_linear` | 4.927607 | 3.918716 |
| CFP | `xz_paired_vctr` | 4.932472 | 3.930317 |
| ROI | `x_only_vctr` | 4.123525 | 3.279062 |
| ROI | `xz_linear` | 4.894819 | 3.907188 |
| ROI | `z_only_linear` | 4.900183 | 3.909173 |
| ROI | `xz_paired_vctr` | 4.912271 | 3.918568 |
| ROI | `xz_iid_vctr` | 4.945556 | 3.966955 |

Interpretation:

- `X-only VCTR` is the strongest prediction model in both CFP and ROI runs.
- **只要加入 `Z`，任何模型都变差了。我们猜测高维 covariate `Z` 给建模带来了麻烦。**
- Adding `Z` to VCTR worsens held-out RMSE by about 0.73 IOP units for CFP and 0.79 for ROI, roughly 17 to 19 percent relative to `X-only VCTR`.
- `X+Z paired VCTR` is slightly worse than `X+Z iid VCTR` for CFP and slightly better for ROI; the differences are small relative to the gap from `X-only VCTR`.
- The result is a prediction ablation, not a final inferential claim.
- The prediction result should be paired with residual diagnostics, covariance diagnostics, and coefficient interpretation before being used as the empirical conclusion.

## Historical v1_best_models

`v1_best_models` used pair_id-level 5-fold held-out CV. Because one patient can contribute multiple visits, this split can place different visits from the same `subject_id` in both train and holdout. Keep it only as a historical ablation run; use `v2_patient_grouped` for the current prediction table.

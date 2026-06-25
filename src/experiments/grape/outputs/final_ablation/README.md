# Final ablation outputs

本目录保存 GRAPE 最终本地消融实验的精简输出。完整运行结果位于：

```text
src/experiments/grape/runs/final_ablation/v1_full_cv_selected/
```

本实验固定使用 full three-stage hyperparameter CV 选出的最终 X-only 配置：

| image_type | S | R | h | hbar |
| :-- | :-- | --: | --: | --: |
| CFP | `3x4x1` | 1 | 1.80 | 0.25 |
| ROI | `6x2x1` | 1 | 0.85 | 0.30 |

split 使用 `subject_id` grouped 5-fold CV。主解释指标是 `rmse_iop`，同时保留 standardized-scale `rmse_std`。

## 主结果表

| image_type | model | rmse_std | rmse_iop | mae_iop | mape_iop_pct |
| :-- | :-- | --: | --: | --: | --: |
| CFP | `x_only_paired_vctr` | 0.988956 | 4.132781 | 3.309435 | 21.419909 |
| CFP | `x_only_linear` | 1.013342 | 4.234688 | 3.348505 | 21.763200 |
| CFP | `x_only_iid_vctr` | 1.022268 | 4.271987 | 3.364259 | 21.614368 |
| CFP | `z_only_linear` | 1.172592 | 4.900183 | 3.909173 | 25.284430 |
| CFP | `xz_paired_vctr` | 1.179956 | 4.930955 | 3.967902 | 25.609186 |
| CFP | `xz_iid_vctr` | 1.181844 | 4.938844 | 3.974475 | 25.423321 |
| CFP | `xz_linear` | 1.185646 | 4.954732 | 3.914034 | 25.309745 |
| ROI | `x_only_paired_vctr` | 0.962024 | 4.020231 | 3.187813 | 20.521406 |
| ROI | `x_only_iid_vctr` | 0.963300 | 4.025567 | 3.174278 | 20.346723 |
| ROI | `x_only_linear` | 0.963461 | 4.026238 | 3.134336 | 20.129718 |
| ROI | `xz_linear` | 1.134346 | 4.740354 | 3.672429 | 23.596788 |
| ROI | `xz_iid_vctr` | 1.161392 | 4.853376 | 3.777580 | 24.104269 |
| ROI | `xz_paired_vctr` | 1.168287 | 4.882191 | 3.863819 | 24.785606 |
| ROI | `z_only_linear` | 1.172592 | 4.900183 | 3.909173 | 25.284430 |

主要结论：CFP 和 ROI 的最佳 held-out RMSE 都来自 `x_only_paired_vctr`。加入 `Z` 后，linear 和 VCTR 模型均明显变差。

## 结果解读

最终消融结果支持一条清晰主线：当前 GRAPE prediction 中，主要有效信息来自 fundus image 的 CP reduced features，而不是 subject-level/vector covariates `Z`。

- 图像特征有明确预测价值：`x_only_paired_vctr` 相比 `z_only_linear`，CFP 的 RMSE IOP 降低 0.7674，ROI 降低 0.8800。
- 加入 `Z` 是稳定的负贡献：无论 linear 还是 VCTR，`X+Z` 都比 `X-only` 更差；这说明当前高维 `Z` 在 patient-level held-out prediction 下没有带来泛化收益。
- varying coefficient / paired refit 的预测收益应谨慎表述：CFP 上最终 paired VCTR 比 X-only linear 好 0.1019 RMSE IOP，paired refit 相比 iid VCTR 好 0.1392；ROI 上 X-only linear、iid VCTR、paired VCTR 基本接近。
- ROI 最终 RMSE 略优于 CFP：ROI best 为 4.0202，CFP best 为 4.1328，但差距不大，应表述为 held-out RMSE slightly favors ROI。

因此，当前 empirical prediction evidence 更适合概括为：**X-only paired VCTR 是最终最佳预测模型；图像特征贡献明确，`Z` 当前不帮忙；paired covariance-aware refit 在 CFP 上有小幅收益，但不是所有设定下都带来显著预测提升。**

## 论文式消融表

`delta_rmse_iop < 0` 表示该组件相对 reference 改善 prediction；`delta_rmse_iop > 0` 表示变差。

| image_type | ablation step | model | reference | delta_rmse_iop | pct_delta_rmse_iop |
| :-- | :-- | :-- | :-- | --: | --: |
| CFP | linear baseline | `x_only_linear` | `z_only_linear` | -0.665495 | -13.581028 |
| CFP | + varying coefficient | `x_only_iid_vctr` | `x_only_linear` | 0.037299 | 0.880798 |
| CFP | + paired covariance-aware refit | `x_only_paired_vctr` | `x_only_iid_vctr` | -0.139206 | -3.258569 |
| CFP | + adding Z | `xz_paired_vctr` | `x_only_paired_vctr` | 0.798174 | 19.313238 |
| ROI | linear baseline | `x_only_linear` | `z_only_linear` | -0.873945 | -17.834940 |
| ROI | + varying coefficient | `x_only_iid_vctr` | `x_only_linear` | -0.000672 | -0.016685 |
| ROI | + paired covariance-aware refit | `x_only_paired_vctr` | `x_only_iid_vctr` | -0.005336 | -0.132549 |
| ROI | + adding Z | `xz_paired_vctr` | `x_only_paired_vctr` | 0.861960 | 21.440560 |

解释边界：

- `linear baseline` 表示从 only-Z linear baseline 转到 X-only linear baseline，检验图像 CP reduced features 的预测价值。
- `+ varying coefficient` 表示从 X-only linear baseline 转到 X-only iid VCTR。
- `+ paired covariance-aware refit` 表示从 X-only iid VCTR 转到 X-only paired VCTR。
- `+ adding Z` 表示在 paired VCTR 中加入 `Z` 后的变化。
- `mape_std_pct` 没有列入主表，因为 standardized response 接近 0 时 denominator 很小，容易产生较大的辅助指标。

## 文件

- `v1_full_cv_selected_summary_metrics.csv`: model-level held-out metrics。
- `v1_full_cv_selected_ablation_table.csv`: 程序生成的 pairwise contrast 表。
- `v1_full_cv_selected_residual_summary.csv`: residual summary，用于后续 diagnosis。

`predictions.csv` 和 fold-level 明细保留在完整 run 目录中，不复制到本目录。

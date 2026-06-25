# GRAPE evaluation

本目录保存当前 GRAPE paired-eye 实证分析的调参和模型评估入口。

## Hyperparameter CV

当前最终超参数选择入口是：

```bash
python src/experiments/grape/evaluation/hyperpar_cv.py \
  --config src/experiments/grape/configs/hyperpar_cv/x_only_grid_v1.json \
  --task-index 1 \
  --num-tasks 5 \
  --max-workers 12
```

该脚本按 `subject_id` grouped CV 对每个 `(image_type, S, R, h, hbar)` 做完整三阶段 held-out prediction：

```text
A dagger -> Sigma(hbar) -> A star -> holdout prediction
```

当前固定设定：

- `z_mode = none`
- `a_eval_mode = full`
- `ridge = 1e-6`

这里的 ridge 只作为数值稳定化。主排序指标是 standardized scale 的 `rmse_std`，同时输出 `rmse_iop`, `mape_std_pct`, `mape_iop_pct`。

全部 shard 完成后聚合：

```bash
python src/experiments/grape/evaluation/hyperpar_cv.py \
  --config src/experiments/grape/configs/hyperpar_cv/x_only_grid_v1.json \
  --aggregate
```

输出默认保存到：

```text
src/experiments/grape/runs/hyperpar_cv/x_only_grid_v1/
```

## Final Ablation

固定 full-CV 选出的最终超参数后，当前最终消融入口是：

```bash
python src/experiments/grape/evaluation/final_ablation.py \
  --config src/experiments/grape/configs/final_ablation/v1_full_cv_selected.json
```

`final_ablation.py` 在同一 `subject_id` grouped held-out split 下比较：

- `z_only_linear`: only vector covariates `Z`
- `x_only_linear`: reduced image features `X_star` only
- `xz_linear`: reduced image features `X_star` plus `Z`
- `x_only_iid_vctr`: `X_star` VCTR using iid stage-1 prediction
- `x_only_paired_vctr`: full paired-eye VCTR using `X_star` only
- `xz_iid_vctr`: `X_star + Z` VCTR using iid stage-1 prediction
- `xz_paired_vctr`: full paired-eye VCTR using `X_star + Z`

当前固定配置来自 `hyperpar_cv.py` 的 full three-stage held-out prediction CV：

| image_type | S | R | h | hbar |
| :-- | :-- | --: | --: | --: |
| CFP | `3x4x1` | 1 | 1.80 | 0.25 |
| ROI | `6x2x1` | 1 | 0.85 | 0.30 |

输出默认保存到：

```text
src/experiments/grape/runs/final_ablation/v1_full_cv_selected/
```

精简汇总结果同步保存到：

```text
src/experiments/grape/outputs/final_ablation/
```

当前结果显示 CFP 和 ROI 的最佳 held-out RMSE 都来自 `x_only_paired_vctr`。加入 `Z` 会明显恶化 prediction；paired covariance-aware refit 对 X-only CFP 有小幅收益，对 X-only ROI 的收益很小。

## Historical Model Comparison

`compare_models.py` 和 `configs/model_comparison/` 保留为早期开发对比入口。已有 `v2_patient_grouped` 结果基于旧 bandwidth CV 固定候选，不再作为最终 empirical ablation table。

## Archive Boundary

旧的 `cv_bandwidth.py` stagewise CV 已归档，不再作为最终超参数选择入口：

```text
src/experiments/grape/archive/stagewise_bandwidth_cv/
```

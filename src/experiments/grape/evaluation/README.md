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

## Model Comparison

固定最终超参数后，模型比较入口是：

```bash
python src/experiments/grape/evaluation/compare_models.py \
  --config src/experiments/grape/configs/model_comparison/v2_patient_grouped.json
```

`compare_models.py` 在同一 held-out split 下比较：

- `z_only_linear`: only vector covariates `Z`
- `xz_linear`: reduced image features `X_star` plus `Z`
- `x_only_vctr`: paired VCTR using `X_star` only
- `xz_iid_vctr`: `X_star + Z` VCTR using iid stage-1 prediction
- `xz_paired_vctr`: full paired-eye VCTR using `X_star + Z`

当前已有的 `v2_patient_grouped` 结果显示 CFP 和 ROI 都由 `x_only_vctr` 最优，加入 `Z` 没有改善 held-out prediction。该结果基于旧 bandwidth CV 固定候选；正式模型比较应在 `hyperpar_cv.py` 完成后，用新的最佳 `(S, R, h, hbar)` 重跑。

输出默认保存到：

```text
src/experiments/grape/runs/model_comparison/
```

精简汇总结果同步保存到：

```text
src/experiments/grape/outputs/model_comparison/
```

## Archive Boundary

旧的 `cv_bandwidth.py` stagewise CV 已归档，不再作为最终超参数选择入口：

```text
src/experiments/grape/archive/stagewise_bandwidth_cv/
```

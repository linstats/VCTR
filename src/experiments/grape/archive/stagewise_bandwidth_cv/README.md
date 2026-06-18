# Stagewise bandwidth CV archive

本目录保存 GRAPE 早期 stagewise bandwidth CV 实验线。它只作为历史记录和结果追溯来源，不再作为当前最终超参数选择方法。

## Why archived

这条线包含 `modeling/cv_bandwidth.py`、旧 bandwidth grid configs、旧 PBS 任务表和 `outputs/cv_bandwidth/` 汇总。它被归档的原因是：

- 早期实验曾在超参数选择阶段使用 `Z`，而当前 empirical 结论要求先做 `X-only VCTR` 调参。
- 多数任务使用 `anchor_grid` 加速；当前最终调参要求 `a_eval_mode=full`。
- 旧选择目标是 stagewise `signal_cv_score` / variance CV，不是最终 full three-stage held-out prediction RMSE。

## Current replacement

当前主线是：

```text
src/experiments/grape/evaluation/hyperpar_cv.py
src/experiments/grape/configs/hyperpar_cv/x_only_grid_v1.json
src/experiments/grape/hpc/hyperpar_cv_x_only_grid_v1.pbs
```

该主线使用 `subject_id` grouped 5-fold CV，对每个 `(S, R, h, hbar)` 完整执行：

```text
A dagger -> Sigma(hbar) -> A star -> holdout prediction
```

最终排序主指标是 standardized scale 的 `rmse_std`。

## Contents

```text
configs/bandwidth_grids/  # v1-v6 stagewise bandwidth configs
hpc/                      # old cv_bandwidth PBS and task/batch tables
modeling/                 # old cv_bandwidth runners and aggregator
outputs/cv_bandwidth/     # selected tracked summaries from old runs
runs/cv_bandwidth/        # local/HPC run outputs if present
```

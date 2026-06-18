# GRAPE HPC

本目录只保存当前 GRAPE 实证分析仍在使用的 PBS 文件。旧的 stagewise bandwidth CV PBS 和任务表已归档到：

```text
src/experiments/grape/archive/stagewise_bandwidth_cv/hpc/
```

## X-only full three-stage hyperparameter CV

当前最终超参数选择使用：

```text
src/experiments/grape/hpc/hyperpar_cv_x_only_grid_v1.pbs
```

提交：

```bash
qsub src/experiments/grape/hpc/hyperpar_cv_x_only_grid_v1.pbs
```

该任务使用 5 个 PBS array shards，每个 shard 12 cores。每个候选完整执行：

```text
A dagger -> Sigma(hbar) -> A star -> holdout prediction
```

固定设定：

- `z_mode = none`
- `split_group = subject_id`
- `a_eval_mode = full`
- `ridge = 1e-6`

这里的 ridge 是数值稳定化，不是主要方法设定。主排序指标是 standardized scale 的 `rmse_std`。

全部 shard 完成后聚合：

```bash
$HOME/conda-envs/vctr-py310/bin/python \
  src/experiments/grape/evaluation/hyperpar_cv.py \
  --config src/experiments/grape/configs/hyperpar_cv/x_only_grid_v1.json \
  --aggregate
```

输出目录：

```text
src/experiments/grape/runs/hyperpar_cv/x_only_grid_v1/
```

## Logs

PBS stdout/stderr 默认写入：

```text
src/experiments/grape/hpc/logs/
```

`logs/` 是运行产物，不进 git。

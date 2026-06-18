# GRAPE modeling

本目录保存 GRAPE paired-eye 实证建模入口。

## Bandwidth CV

单个 `(image_type, S, R)` 任务：

```bash
python src/experiments/grape/modeling/cv_bandwidth.py \
  --image-type cfp \
  --S 3x3x1 \
  --R 2 \
  --run-name local_smoke \
  --task-id 1 \
  --bandwidth-config src/experiments/grape/configs/bandwidth_grids/v1_adaptive_support.json
```

HPC array 任务：

```bash
python src/experiments/grape/modeling/run_cv_bandwidth_batch.py \
  --batch-csv src/experiments/grape/hpc/cv_bandwidth_batches_v1.csv \
  --batch-index 1 \
  --task-csv src/experiments/grape/hpc/cv_bandwidth_tasks_v1.csv \
  --max-workers 6 \
  --a-eval-mode anchor_grid \
  --a-eval-num-points 80
```

batch runner 会并行调用单任务 runner，并在终端打印每个原子任务的 `elapsed_seconds`。batch-level timing 保存在 `runs/cv_bandwidth/{run_name}/batch_XX/batch_result.json`。

聚合一个 run：

```bash
python src/experiments/grape/modeling/aggregate_cv_bandwidth.py \
  --run-name v1_adaptive_support
```

输出默认保存在：

```text
src/experiments/grape/runs/cv_bandwidth/{run_name}/
```

# GRAPE HPC

本目录保存 GRAPE 实证分析的 HPC 任务文件。

第一轮 bandwidth CV 推荐使用 batch PBS：5 个 PBS array jobs，每个 job 12 cores，内部并行跑多个 `(image_type, S, R)` 原子任务。

```bash
qsub src/experiments/grape/hpc/cv_bandwidth_batch_array.pbs
```

任务表分两层：

```text
src/experiments/grape/hpc/cv_bandwidth_tasks_v1.csv
src/experiments/grape/hpc/cv_bandwidth_batches_v1.csv
```

其中 `cv_bandwidth_tasks_v1.csv` 是 40 个原子任务，`cv_bandwidth_batches_v1.csv` 把这些原子任务按估计成本分成 5 个 batch。

每个原子任务输出到：

```text
src/experiments/grape/runs/cv_bandwidth/v1_adaptive_support/
```

每个 batch 额外输出：

```text
src/experiments/grape/runs/cv_bandwidth/v1_adaptive_support/batch_XX/batch_result.json
```

`batch_result.json` 记录 `batch_elapsed_seconds`，每个原子任务的 `result.json` 和 run-level `manifest.csv` 记录 `elapsed_seconds`。PBS 日志中也会打印每个任务完成时的 elapsed time。

HPC 运行结束后聚合：

```bash
$HOME/conda-envs/vctr-py310/bin/python \
  src/experiments/grape/modeling/aggregate_cv_bandwidth.py \
  --run-name v1_adaptive_support
```

旧的单任务 PBS `cv_bandwidth_array.pbs` 保留为 fallback；它会提交 40 个单核 array tasks。

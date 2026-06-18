# GRAPE 实证工作区

本目录是当前 paired-eye GRAPE 实证分析的主工作区。继承来的 MATLAB/R 代码位于 `code_and_data/real_data`，它是 iid baseline 的参考实现；新的 paired-eye 数据准备、调参和模型运行都应放在这里。

```text
src/experiments/grape/
├── data/          # raw/audit/tensors/features 数据层级
├── preprocess/    # 表格构建、图像预处理
├── features/      # CP reduced feature package 构建
├── evaluation/    # 当前调参、模型比较和后续评估入口
├── configs/       # 当前实验配置
├── hpc/           # 当前 PBS 模板
├── diagnostics/   # 残差、协方差和系数解释诊断
├── outputs/       # 论文使用的小型最终表格和图像
├── runs/          # 本地/HPC 运行结果，默认不进 git
└── archive/       # 历史实验线
```

## 当前数据入口

当前审计表入口是：

```bash
python src/experiments/grape/preprocess/build_tables.py
```

它读取 `data/raw/` 下的 GRAPE 原始 Excel 和图像文件夹，并重新生成：

- `data/audit/interim_visits.csv`
- `data/audit/processed_paired.csv`
- `data/audit/build_summary.json`

Level-2 图像张量入口是：

```bash
python src/experiments/grape/preprocess/build_tensors.py
```

它读取 `data/audit/processed_paired.csv`，按主分析口径生成 `data/tensors/cfp_192_iop_le35/` 和 `data/tensors/roi_192_iop_le35/`。`tensors/` 只保存 resize/flip 后的图像张量和图像 manifest，不保存 `y/Z/t`。

Level-3 CP 特征入口是：

```bash
python src/experiments/grape/features/build_features.py --S 3x3x1 --R 2
```

它读取 `data/tensors/` 和 `data/audit/processed_paired.csv`，生成 `data/features/*/S3x3x1_R2/`。`features/` 是第一层完整模型输入包，包含 `X_star + y/Z/t`。

批量生成第一轮本地特征 grid：

```bash
python src/experiments/grape/features/build_feature_grid.py
```

当前默认 grid 为：

- 图像类型：`cfp`, `roi`
- 分区：`S = 2x2x1, 3x3x1, 4x4x1, 6x6x1, 8x8x1`
- CP rank：`R = 1, 2, 3, 4`

该命令会跳过已经完整存在的 feature package，并在 `data/features/build_feature_grid_summary.json` 记录构建结果。

## 当前主线

当前最终超参数选择入口是 full three-stage held-out prediction CV：

```bash
python src/experiments/grape/evaluation/hyperpar_cv.py \
  --config src/experiments/grape/configs/hyperpar_cv/x_only_grid_v1.json \
  --task-index 1 \
  --num-tasks 5 \
  --max-workers 12
```

它在 `subject_id` grouped 5-fold CV 下，对所有当前 CFP/ROI feature packages 的 `(S, R, h, hbar)` 做完整 held-out prediction：

```text
A dagger -> Sigma(hbar) -> A star -> holdout prediction
```

当前主排序指标是 standardized scale 的 `rmse_std`。固定设定为 `z_mode=none`, `a_eval_mode=full`, `ridge=1e-6`；这里的 ridge 只是数值稳定化。

全部 shard 完成后聚合：

```bash
python src/experiments/grape/evaluation/hyperpar_cv.py \
  --config src/experiments/grape/configs/hyperpar_cv/x_only_grid_v1.json \
  --aggregate
```

输出目录：

```text
src/experiments/grape/runs/hyperpar_cv/x_only_grid_v1/
```

## 模型对比入口

固定最终超参数后，模型对比入口是：

```bash
python src/experiments/grape/evaluation/compare_models.py \
  --config src/experiments/grape/configs/model_comparison/v2_patient_grouped.json
```

完整运行产物默认写入：

```text
src/experiments/grape/runs/model_comparison/
```

`runs/` 默认不进 git；可进入 repo 的精简汇总位于：

```text
src/experiments/grape/outputs/model_comparison/
```

当前更可靠的既有模型对比结果是 `v2_patient_grouped`：在 true patient-level grouped 5-fold held-out CV 下，CFP 和 ROI 的最佳 RMSE 都来自 `X-only VCTR`。加入 `Z` 后的 `X+Z paired VCTR` 没有改善 held-out prediction，paired GLS 相比 iid stage-1 prediction 的 RMSE 差异也很小。该结论仍基于旧 bandwidth CV 固定的候选；正式版本应等待 `hyperpar_cv.py` 的 full three-stage 结果后重跑。

## 已归档实验线

旧的 stagewise bandwidth CV 已归档到：

```text
src/experiments/grape/archive/stagewise_bandwidth_cv/
```

该线保留为历史记录，不再作为最终超参数选择依据。归档原因：

- 早期 run 曾把 `Z` 纳入超参数选择。
- 多数 run 使用 `anchor_grid` 加速，而当前最终选择要求 `a_eval_mode=full`。
- 旧目标函数主要是 stagewise `signal_cv_score` / variance CV，不是最终 three-stage held-out prediction RMSE。

## 计划中的实证流程

1. 从 GRAPE 原始文件构建 visit-level 表和 paired-eye 表，输出到 `data/audit/`。
2. 在 notebooks 或 scratch scripts 中探索 raw/interim 数据；这类代码不视为正式实验。
3. 生成 Level-2 标准图像张量：raw image -> OS 水平翻转 -> resize 到 `192 x 192 x 3`，输出到 `data/tensors/`。
4. 针对候选 `(S, R)` 从 `tensors/` 生成 Level-3 CP reduced features，输出到 `data/features/`；这里才保存 `X_star + y/Z/t`。
5. 用 `evaluation/hyperpar_cv.py` 选择 `X-only VCTR` 的 `(S, R, h, hbar)`。
6. 在 HPC 上运行大规模调参和模型拟合，运行结果放在 `runs/`。
7. 只把论文最终使用的小型表格和图像整理到 `outputs/`。
8. 对 full-CV 选出的最终配置运行 true patient-level grouped 模型对比、残差诊断、paired-eye 协方差诊断和系数解释，确保 empirical pipeline 能回应 paired-eye dependence 的审稿意见。

## 数据说明

- `data/README.md` 说明 Plan A 的 `raw/audit/tensors/features` 数据层级。
- `data/raw/README.md` 说明下载后的 GRAPE 原始文件、图像命名规则和 raw data counts。
- `data/audit/README.md` 说明 paired-eye 工作流使用的审计表和 QC 标记。

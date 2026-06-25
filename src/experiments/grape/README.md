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

## 数据预处理

当前流程第一步是从 raw data 构建 paired-eye analysis inputs。

审计表入口：

```bash
python src/experiments/grape/preprocess/build_tables.py
```

它读取 `data/raw/` 下的 GRAPE 原始 Excel 和图像文件夹，并重新生成：

- `data/audit/interim_visits.csv`
- `data/audit/processed_paired.csv`
- `data/audit/build_summary.json`

Level-2 图像张量入口：

```bash
python src/experiments/grape/preprocess/build_tensors.py
```

它读取 `data/audit/processed_paired.csv`，按主分析口径生成 `data/tensors/cfp_192_iop_le35/` 和 `data/tensors/roi_192_iop_le35/`。`tensors/` 只保存 resize/flip 后的图像张量和图像 manifest，不保存 `y/Z/t`。

Level-3 CP 特征入口：

```bash
python src/experiments/grape/features/build_features.py --S 3x3x1 --R 2
```

它读取 `data/tensors/` 和 `data/audit/processed_paired.csv`，生成 `data/features/*/S3x3x1_R2/`。`features/` 是第一层完整模型输入包，包含 `X_star + y/Z/t`。

批量生成当前本地 feature grid：

```bash
python src/experiments/grape/features/build_feature_grid.py
```

当前已生成的 feature packages 为 CFP/ROI 各 56 个：

- 图像类型：`cfp`, `roi`
- 分区：`S = 2x2x1, 2x3x1, 2x4x1, 2x6x1, 3x2x1, 3x3x1, 3x4x1, 3x6x1, 4x2x1, 4x3x1, 4x4x1, 4x6x1, 6x2x1, 6x3x1, 6x4x1, 6x6x1, 8x8x1`
- CP rank：`R = 1, 2, 3, 4`

该命令会跳过已经完整存在的 feature package，并在 `data/features/build_feature_grid_summary.json` 记录构建结果。当前 full-CV 选出的最佳配置为：

| image_type | S | R | h | hbar |
| :-- | :-- | --: | --: | --: |
| CFP | `3x4x1` | 1 | 1.80 | 0.25 |
| ROI | `6x2x1` | 1 | 0.85 | 0.30 |

## 超参数选择

当前流程第二步是 full three-stage held-out prediction CV，用于选择 `X-only VCTR` 的 `(S, R, h, hbar)`：

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

完整 run 输出目录：

```text
src/experiments/grape/runs/hyperpar_cv/x_only_grid_v1/
```

## 消融实验

当前流程第三步是固定 full-CV 选出的最终超参数，运行本地最终消融实验：

```bash
python src/experiments/grape/evaluation/final_ablation.py \
  --config src/experiments/grape/configs/final_ablation/v1_full_cv_selected.json
```

完整 run 输出目录：

```text
src/experiments/grape/runs/final_ablation/v1_full_cv_selected/
```

`runs/` 默认不进 git；可进入 repo 的精简汇总位于：

```text
src/experiments/grape/outputs/final_ablation/
```

当前最终消融结果是 `v1_full_cv_selected`：在 true patient-level grouped 5-fold held-out CV 下，CFP 和 ROI 的最佳 RMSE 都来自 `x_only_paired_vctr`。加入 `Z` 后的 linear / VCTR 模型均明显变差；paired covariance-aware refit 对 X-only CFP 有小幅收益，对 X-only ROI 的收益很小。

固定配置来自 full three-stage hyperparameter CV：

| image_type | S | R | h | hbar |
| :-- | :-- | --: | --: | --: |
| CFP | `3x4x1` | 1 | 1.80 | 0.25 |
| ROI | `6x2x1` | 1 | 0.85 | 0.30 |

## 已归档实验线

旧的 stagewise bandwidth CV 已归档到：

```text
src/experiments/grape/archive/stagewise_bandwidth_cv/
```

该线保留为历史记录，不再作为最终超参数选择依据。归档原因：

- 早期 run 曾把 `Z` 纳入超参数选择。
- 多数 run 使用 `anchor_grid` 加速，而当前最终选择要求 `a_eval_mode=full`。
- 旧目标函数主要是 stagewise `signal_cv_score` / variance CV，不是最终 three-stage held-out prediction RMSE。

## 当前流程

目前 GRAPE empirical pipeline 分为三步：

1. **数据预处理**：raw files -> audit tables -> resized/flipped tensors -> CP reduced feature packages。
2. **超参数选择**：用 `evaluation/hyperpar_cv.py` 做 `subject_id` grouped full three-stage CV，选择 `X-only VCTR` 的 `(S, R, h, hbar)`。
3. **消融实验**：用 `evaluation/final_ablation.py` 固定最终配置，比较 linear / VCTR、X-only / X+Z、iid / paired refit。

完整运行结果放在 `runs/`，默认不进 git；论文使用的小型表格整理到 `outputs/`。后续还需继续做 residual diagnostics、paired-eye covariance diagnostics 和 coefficient interpretation，以回应 paired-eye dependence 的审稿意见。

## 数据说明

- `data/README.md` 说明 Plan A 的 `raw/audit/tensors/features` 数据层级。
- `data/raw/README.md` 说明下载后的 GRAPE 原始文件、图像命名规则和 raw data counts。
- `data/audit/README.md` 说明 paired-eye 工作流使用的审计表和 QC 标记。

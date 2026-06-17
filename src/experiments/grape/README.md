# GRAPE 实证工作区

本目录是当前 paired-eye GRAPE 实证分析的主工作区。继承来的 MATLAB/R 代码位于 `code_and_data/real_data`，它是 iid baseline 的参考实现；新的 paired-eye 数据准备、调参和模型运行都应放在这里。

```text
src/experiments/grape/
├── data/          # raw/audit/tensors/features 数据层级
├── preprocess/    # 表格构建、图像预处理、数据划分
├── hpc/           # PBS 模板与 HPC 提交辅助脚本
├── runs/          # 本地/HPC 运行结果，默认不进 git
└── outputs/       # 论文使用的小型最终表格和图像
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

## 计划中的实证流程

1. 从 GRAPE 原始文件构建 visit-level 表和 paired-eye 表，输出到 `data/audit/`。
2. 在 notebooks 或 scratch scripts 中探索 raw/interim 数据；这类代码不视为正式实验。
3. 生成 Level-2 标准图像张量：raw image -> OS 水平翻转 -> resize 到 `192 x 192 x 3`，输出到 `data/tensors/`。
4. 针对候选 `(S, R)` 从 `tensors/` 生成 Level-3 CP reduced features，输出到 `data/features/`；这里才保存 `X_star + y/Z/t`。
5. 调整实证超参数，例如分区 `S`、CP rank `R` 和 bandwidth `h`；优先使用 subject-level cross-validation。
6. 在 HPC 上运行大规模调参和模型拟合，运行结果放在 `runs/`。
7. 只把论文最终使用的小型表格和图像整理到 `outputs/`。

## 数据说明

- `data/README.md` 说明 Plan A 的 `raw/audit/tensors/features` 数据层级。
- `data/raw/README.md` 说明下载后的 GRAPE 原始文件、图像命名规则和 raw data counts。
- `data/audit/README.md` 说明 paired-eye 工作流使用的审计表和 QC 标记。

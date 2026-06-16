# GRAPE 实证工作区

本目录是当前 paired-eye GRAPE 实证分析的主工作区。继承来的 MATLAB/R 代码位于 `code_and_data/real_data`，它是 iid baseline 的参考实现；新的 paired-eye 数据准备、调参和模型运行都应放在这里。

```text
src/experiments/grape/
├── data/          # GRAPE 原始文件与派生分析表
├── preprocess/    # 表格构建、图像预处理、数据划分
├── hpc/           # PBS 模板与 HPC 提交辅助脚本
├── runs/          # 本地/HPC 运行结果，默认不进 git
└── outputs/       # 论文使用的小型最终表格和图像
```

## 当前数据入口

当前预处理入口是：

```bash
python src/experiments/grape/preprocess/build_tables.py
```

它读取 `data/raw/` 下的 GRAPE 原始 Excel 和图像文件夹，并重新生成：

- `data/interim_visits.csv`
- `data/processed_paired.csv`
- `data/build_summary.json`

默认使用 `data/processed_paired.csv` 作为 paired-eye 建模入口。它一行对应同一受试者、同一时间点的一对 OD/OS 观测，包含 paired response (`iop_od`, `iop_os`)、双眼图像路径、非盲点 VF 协变量的左右眼均值，以及 QC 标记。

## 计划中的实证流程

1. 从 GRAPE 原始文件构建 visit-level 表和 paired-eye 表。
2. 在 notebooks 或 scratch scripts 中探索 raw/interim 数据；这类代码不视为正式实验。
3. 针对每组候选设置进行图像预处理，包括 resize、OS 水平翻转、空间分区和 blockwise CP 特征提取。
4. 调整实证超参数，例如图像尺寸、分区 `S`、CP rank `R` 和 bandwidth `h`；优先使用 subject-level cross-validation。
5. 在 HPC 上运行大规模调参和模型拟合，运行结果放在 `runs/`。
6. 只把论文最终使用的小型表格和图像整理到 `outputs/`。

## 数据说明

- `data/README.md` 说明 paired-eye 工作流使用的派生数据表。
- `data/raw/README.md` 说明下载后的 GRAPE 原始文件、图像命名规则和 raw data counts。

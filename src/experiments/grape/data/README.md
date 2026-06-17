# GRAPE 数据目录

本目录按照 Plan A 将 GRAPE 实证数据分为四层：原始数据、审计表、标准图像张量和模型特征。

```text
src/experiments/grape/data/
├── raw/
├── audit/
├── tensors/
└── features/
```

## `raw/`

原始下载数据。该目录保持源数据不变，包括：

- `CFPs/`
- `ROIs/`
- `VF_and_clinical_information.xlsx`
- `README.md`

## `audit/`

从 raw Excel 和图像索引构建出的可审计表格。这里定义样本、配对关系和 QC 标记，但不存放模型数组：

- `interim_visits.csv`
- `processed_paired.csv`
- `build_summary.json`
- `README.md`

## `tensors/`

Level 2 标准图像张量。该层只对 raw CFP/ROI 图像做：

```text
raw image -> OS 水平翻转 -> resize 到 192 x 192 x 3
```

该层不做图像切块，不做 CP 分解，也不保存 `y/Z/t`。

## `features/`

Level 3 模型特征包。这里的输出由 `tensors/` 和指定的 `(S, R)` 生成，并包含：

- `X_star.npy`
- `y.npy`
- `Z.npy`
- `t.npy`
- 特征 manifest / 元数据

这是第一层可以被视为完整模型输入包的数据。

当前已经生成第一轮本地特征 grid：

```text
features/
├── cfp_192_iop_le35/S{2x2x1,3x3x1,4x4x1,6x6x1,8x8x1}_R{1,2,3,4}/
└── roi_192_iop_le35/S{2x2x1,3x3x1,4x4x1,6x6x1,8x8x1}_R{1,2,3,4}/
```

每个目录包含：

- `X_star.npy`：`273 x 2 x R x S`，即 `n_pair x eye x R x S`
- `y.npy`：`273 x 2`，在 OD/OS 展平后做 z-score
- `Z.npy`：`273 x 60`，包含未变换的 `is_female` 和逐列 z-score 后的 59 个非盲点 VF 均值
- `t.npy`：`273`，由 `age_at_visit` min-max 归一化到 `[0, 1]`
- `cp_components.npz`：每个 block 的 CP 分解组件
- `manifest.csv` 和 `meta.json`：样本索引、生成参数和变换记录

批量构建记录保存在：

- `features/build_feature_grid_summary.json`

# GRAPE 审计表

`audit/` 保存从 GRAPE 原始 Excel 和图像索引构建出的可审计表格。这些文件定义 paired-eye 样本、配对关系和 QC 标记；它们不是模型数组。

这些表格由下面的脚本重新生成：

```bash
python src/experiments/grape/preprocess/build_tables.py
```

## `interim_visits.csv`

`interim_visits.csv` 是 visit-level 事实表：一行对应一只眼的一次 visit。它把 raw Excel 记录与 CFP/ROI 图像路径和存在性标记合并起来。该表不做左右眼配对，也不做模型预处理。

当前规模：

| visit 记录数 | 受试者 | 生理眼 | 带 CFP/ROI 图像的 visit |
| :------------ | -------: | ------------: | -------------------------: |
| 1115 | 144 | 263 | 631 |

不是每条 Excel visit 都有图像，也不是每位受试者在每个时间点都有双眼数据。

## `processed_paired.csv`

`processed_paired.csv` 由 `interim_visits.csv` 构建而来。若同一受试者在同一个 `interval_years` 下同时有 OD 和 OS 记录，且双眼都有对应 CFP/ROI 图像，则两条 eye-level visits 会合并成一行 paired row。

当前规模：

| 带图像 visits | 进入配对的 visits | OD/OS 配对数 | 主分析 `IOP <= 35` 后配对数 |
| :----------- | ------------: | -----------: | -------------------------------------: |
| 631 | 552 | 276 | 273 |

主要字段包括：

- response：`iop_od`, `iop_os`
- 图像路径：`cfp_path_od`, `cfp_path_os`, `roi_path_od`, `roi_path_os`
- paired row 共享的标量协变量：年龄和性别
- 眼别特异的 raw VF 列，后缀为 `_od` 和 `_os`
- 非盲点 VF 的左右眼均值协变量：`z_vf_*_mean`

当前 paired-eye 模型暂不使用眼别特异的 `Z_ij`，因此 VF 协变量采用 OD/OS 左右眼均值。盲点位置 `VF 21` 和 `VF 32` 不进入 `z_vf_*_mean` 协变量。

## `build_summary.json`

`build_summary.json` 是机器可读的 QC 摘要。它记录 raw 图像数量、visit 数量、paired/unpaired 数量、盲点 VF 位置，以及不同 IOP inclusion rule 下的计数。

`processed_paired.csv` 保留 inclusion flags，而不是物理删除行：

- `include_primary_iop35`：主分析口径；排除任一眼 `IOP > 35` 的 paired visits。276 对中保留 273 对。
- `include_sensitivity_iop30_low7`：敏感性分析口径；排除任一眼 `IOP = 7` 或 `IOP > 30` 的 paired visits。276 对中保留 270 对。
- `include_old_iop_rule`：旧 iid 论文口径；若某只眼出现 row-level `IOP` 超出 mean +/- 2 SD，则该眼相关 paired row 被排除。276 对中保留 244 对。

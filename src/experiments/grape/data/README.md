# GRAPE 派生数据

下载后的原始数据放在 `raw/`，其原始文件结构和 raw data counts 见 `raw/README.md`，

经 `src/experiments/grape/preprocess/build_tables.py` 预处理后得到当前目录：

```text
src/experiments/grape/data/
├── raw/
├── interim_visits.csv
├── processed_paired.csv
└── build_summary.json
```

---

#### 问诊审计表 `interim_visits.csv`

该表是把 GRAPE 原始 Excel 与 CFP/ROI 图像索引整理成**「一行、一只眼、一次 visit」的问诊审计表**。它不配对、不做任何预处理。数据规模：

| visit 记录次数 | 受试者 | 生理眼 | 带 CFP/ROI 图像的 visit |
| :------------- | ------ | :----- | :---------------------- |
| 1115           | 144    | 263    | 631                     |

- **同一位受试者的每只眼可能有多次 visit，且 OD/OS 的 visit 次数不一定相同。**
- **并非每次 visit 都有对应图像；只有带 CFP/ROI 图像的 visit 才能进入 image-based 分析。**

---

#### 配对数据表 `processed_paired.csv`

在 `interim_visits.csv` 的基础上，**若同一患者、同一随访时间点的 OD/OS 两只眼均有 visit 记录，且两只眼均有对应 CFP/ROI 图像**，则合并为 `processed_paired.csv` 中的一行配对样本。当前规模：

| 带 CFP/ROI 图像的 visit | 进入配对的 visits | OD/OS 配对数 |
| :---------------------- | ----------------- | :----------- |
| 631                     | 552               | 276          |

- 剩余79 条 visits 有图像但未能配对。

 `processed_paired.csv` 主要字段包括：

- response：`iop_od`, `iop_os`；
- 图像路径：`cfp_path_od`, `cfp_path_os`, `roi_path_od`, `roi_path_os`；
- 配对样本共享的标量协变量，例如年龄和性别；
- 原始眼别特异 VF 列，后缀为 `_od` 和 `_os`；
- 非盲点 VF 协变量的左右眼均值，列名为 `z_vf_*_mean`。

当前 paired-eye 模型暂不使用眼别特异的 `Z_{ij}`，因此 `processed_paired.csv` 中的 VF 协变量采用左右眼均值。盲点位置 `VF 21` 和 `VF 32` 不进入 `z_vf_*_mean` 协变量。

---

####  `build_summary.json`

机器可读的构建摘要，用于快速 QC。它记录 raw 图像数量、visit 数量、paired/unpaired 数量、被排除的盲点 VF 位置，以及旧 iid 论文 IOP 排除规则下的计数。

`processed_paired.csv` 保留原论文的 outlier 规则标记 (response > avg ± 2std)，而不是直接删除行：

- `pair_has_iop_outlier`
- `include_old_iop_rule`

如果套用旧规则，276 对中有 244 对会被保留、32 对会被排除。

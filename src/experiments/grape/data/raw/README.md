# GRAPE 原始数据

本目录保存下载后的 GRAPE 原始文件，即项目内部预处理之前的数据。

```text
src/experiments/grape/data/raw/
├── CFPs/
│   └── *.jpg
├── ROIs/
│   └── *.jpg
└── VF_and_clinical_information.xlsx
```

## 文件内容

- CFP (`Color Fundus Photograph`)：完整彩色眼底照片，包含视盘、黄斑、血管和周边视网膜。
- ROI (`Region of Interest`)：以视盘/视神经乳头为中心，从对应 CFP 裁剪出的局部区域。
- `VF_and_clinical_information.xlsx`：临床信息、visit 时间、response 和 visual-field 数值。

CFP 和 ROI 的文件名格式为：

```text
subject_laterality_visit.jpg
```

例如，`12_OS_3.jpg` 表示 12 号受试者、左眼 (`OS`)、第 3 次 visit。`OD` 表示右眼。每张 ROI 应该有一张同名 CFP；每条 image visit 应该对应 Excel 中一条 `Corresponding CFP` 非空的记录。

## Excel 结构

Excel 包含 baseline 和 follow-up sheets。`Follow-up` sheet 已经包含 baseline visit (`Visit Number = 1`, `Interval Years = 0`)，因此不要把 `Baseline` 纵向追加到 `Follow-up`。

当前预处理以 `Follow-up` 作为 visit-level 主表，并按受试者和眼别补充 baseline 字段。其中，visit 时年龄计算为：

```text
age_at_visit = baseline_age + interval_years
```

## 原始规模

当前下载得到的原始数据包括：

- 144 位受试者；
- 263 只生理眼；
- 1115 条 follow-up visit 记录；
- 631 张 CFP 图像；
- 631 张 ROI 图像。

不是每位受试者都有双眼数据，也不是 Excel 中每条 visit 都有图像。没有对应图像的 Excel 记录不能直接进入 image-based 建模。

## 与原 iid 实证的关系

原 iid VCTR 实证使用的是单眼 image visits。它将图像 resize 到 `192 x 192 x 3`，对 OS 图像做水平翻转，然后分别在 CFP 和 ROI 分析中通过 10-fold cross-validation 选择 `S = 3 x 3 x 1` 和 `R = 2`。

原论文报告的 iid 样本量 591 是经过额外匹配、缺失处理和 IOP 异常值筛选后的结果。它不等于 631 条 raw image visits，也不等于当前 paired-eye 表的样本量。

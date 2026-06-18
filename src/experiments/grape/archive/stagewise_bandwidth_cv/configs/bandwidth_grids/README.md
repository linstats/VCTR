# GRAPE bandwidth grid 配置

本目录保存 GRAPE 实证 bandwidth 设计。每次重新设计 `h` / `hbar` 时，新增一个 JSON 文件，不覆盖旧版本。

当前第一版：

- `v1_adaptive_support.json`

该版本先给出较宽的候选 `signal h`，再按 `(S,R)` 的局部线性参数量和 5-fold 训练 support 自动过滤过小的 `h`。

## 当前 X-only 重新调参配置

- `v4_x_only_patient_grouped.json`

该版本用于重新调参当前表现最好的 `X-only VCTR`。它和早期 v1-v3 的主要区别是：

- `z_mode = none`：在 support eligibility、signal bandwidth CV、variance bandwidth CV 和最终拟合前移除 `Z`。
- `split_group = subject_id`：同一个病人的所有 visits 保持在同一个 fold 中。
- 搜索完整 CFP/ROI feature grid：`S = 2x2x1, 3x3x1, 4x4x1, 6x6x1, 8x8x1`，`R = 1, 2, 3, 4`。
- `signal_h_candidates` 合并 v1 的小 bandwidth 和 v3 的 near-global bandwidth。

对应任务表：

```text
src/experiments/grape/hpc/cv_bandwidth_tasks_v4_x_only_patient_grouped.csv
src/experiments/grape/hpc/cv_bandwidth_batches_v4_x_only_patient_grouped.csv
```

## 矩形切分第一阶段粗搜

- `v5_x_only_rectangles_stage1.json`

该版本用于补充新生成的矩形切分 feature package。它只搜索新增矩形 `S`，不重复 v4 已覆盖的 square `S`：

- `S = 2x3x1, 2x4x1, 2x6x1, 3x2x1, 3x4x1, 3x6x1, 4x2x1, 4x3x1, 4x6x1, 6x2x1, 6x3x1, 6x4x1`
- `R = 1, 2, 3`
- 图像类型：`cfp`, `roi`
- 总任务数：72
- `z_mode = none`
- `split_group = subject_id`

该阶段使用 coarse `h` grid，并扩展到 `h=4.0`，用于和 v4 的 square results 合并后选出每个 image type 的 top candidates。第二阶段只对 top 3-5 做局部 `h` refinement。

对应任务表：

```text
src/experiments/grape/hpc/cv_bandwidth_tasks_v5_x_only_rectangles_stage1.csv
src/experiments/grape/hpc/cv_bandwidth_batches_v5_x_only_rectangles_stage1.csv
```

## Top candidates 第二阶段局部细化

- `v6_refine_low_h.json`
- `v6_refine_mid_h.json`
- `v6_refine_high_h.json`
- `v6_refine_boundary_h.json`

该阶段对 v4+v5 合并后的每个 image type top 5 做局部 `h` / `hbar` refinement，不再搜索全部 feature grid。任务表中每个 task 指向适合自身上一阶段最优 `h` 的 refinement config。

候选：

- CFP: `3x2x1_R1`, `3x3x1_R1`, `2x2x1_R1`, `2x3x1_R1`, `3x2x1_R2`
- ROI: `6x2x1_R1`, `4x2x1_R1`, `2x2x1_R2`, `3x2x1_R1`, `2x2x1_R1`

对应任务表：

```text
src/experiments/grape/hpc/cv_bandwidth_tasks_v6_x_only_top_refinement.csv
src/experiments/grape/hpc/cv_bandwidth_batches_v6_x_only_top_refinement.csv
```

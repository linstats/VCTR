# GRAPE Level 3 特征包

`features/` 是 GRAPE 实证的 Level 3 数据层：由 `tensors/` 中的标准图像张量和指定的 `(S, R)` 生成 blockwise CP reduced features。这里是第一层完整模型输入，包含 `X_star + y/Z/t`。

生成入口：

```bash
python src/experiments/grape/features/build_features.py --S 3x3x1 --R 2
```

批量生成 grid：

```bash
python src/experiments/grape/features/build_feature_grid.py \
  --image-types cfp roi \
  --S-grid 2x3x1 3x2x1 2x4x1 4x2x1 3x4x1 4x3x1 2x6x1 6x2x1 3x6x1 6x3x1 4x6x1 6x4x1 \
  --R-grid 1 2 3
```

## 当前已生成组合

当前包含 CFP 和 ROI 两类图像：

```text
cfp_192_iop_le35/
roi_192_iop_le35/
```

每类图像下已生成：

```text
S = 2x2x1, 2x3x1, 2x4x1, 2x6x1,
    3x2x1, 3x3x1, 3x4x1, 3x6x1,
    4x2x1, 4x3x1, 4x4x1, 4x6x1,
    6x2x1, 6x3x1, 6x4x1, 6x6x1,
    8x8x1
R = 1, 2, 3 for all S above
R = 4 for the original square grid only:
    2x2x1, 3x3x1, 4x4x1, 6x6x1, 8x8x1
```

共 `112` 个 feature packages：

- 每类图像 `56` 个 packages。
- `17` 个分区均包含 `R = 1, 2, 3`。
- 原始 square grid 额外包含 `R = 4`。

最近一次矩形 grid 构建记录在 `build_feature_grid_summary.json`：

- 构建时间：`2026-06-17T11:04:17Z`。
- 新增矩形分区：`2x3x1`, `3x2x1`, `2x4x1`, `4x2x1`, `3x4x1`, `4x3x1`, `2x6x1`, `6x2x1`, `3x6x1`, `6x3x1`, `4x6x1`, `6x4x1`。
- 新增 rank：`R = 1, 2, 3`。
- 新增结果：`72` 个 packages，`0` 个失败。

## 每个 package 内容

每个 `(image_type, S, R)` 目录包含：

- `X_star.npy`：CP 后的图像特征，shape 为 `n_pair x 2 x R x S`
- `y.npy`：response，shape 为 `n_pair x 2`
- `Z.npy`：vector covariates，包含 `is_female` 和 59 个非盲点 VF 均值
- `t.npy`：归一化后的 `age_at_visit`
- `cp_components.npz`：每个 block 的 CP 分解组件
- `manifest.csv`：样本索引
- `meta.json`：生成参数、变量变换和 CP 诊断

当前主分析样本数为 `273` 对 paired visits。

## CP 收敛诊断

当前 CP-ALS 使用 `max_iter = 50`。达到 `max_iter` 不等于 CP 失败，只表示该 block 在 50 次迭代内没有达到当前 `tol`。

总体情况：

- 所有 `R = 1` 组合均提前收敛。
- 当前所有 `R = 2, 3, 4` 组合均至少有一个 block 达到 `max_iter = 50`。
- 多数达到上限的组合 median relative change 仍在约 `0.002-0.010`，可先用于第一轮 HPC 建模调参。

高 residual-change 组合需要后续重点检查：

| image | combination | max relative change |
| :---- | :---------- | ------------------: |
| CFP | `S8x8x1_R4` | 0.0843 |
| CFP | `S6x6x1_R4` | 0.0346 |
| ROI | `S6x4x1_R3` | 0.0311 |
| ROI | `S6x6x1_R3` | 0.0273 |
| CFP | `S4x2x1_R3` | 0.0259 |
| CFP | `S6x2x1_R2` | 0.0233 |
| ROI | `S8x8x1_R3` | 0.0223 |
| CFP | `S6x3x1_R3` | 0.0222 |
| CFP | `S6x6x1_R3` | 0.0219 |
| ROI | `S4x4x1_R4` | 0.0203 |
| ROI | `S2x4x1_R2` | 0.0200 |

若最终最优模型落在这些组合上，建议提高 `--max-iter` 后重建对应 feature package，并做敏感性检查。

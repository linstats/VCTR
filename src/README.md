# `src/` 代码说明

`src/` 是当前 paired-eye VCTR 的 Python 主开发线，不再承担 `iid VCTR` 的主线复现任务。

## 当前定位

- 面向 paired-eye VCTR
- 默认参考 paired-eye 论文 `VaryingCoefPLM.pdf`
- 当前默认估计器是 `PairedEyeVCTRModel`
- 当前默认协方差模式是 `exchangeable_varying_sigma`

## 目录结构

```text
src/
  dgps/
  data/
  features/
  metrics/
  models/
  experiments/
  utils/
```

各目录职责如下。

### `dgps/`

- paired 仿真数据生成机制
- 当前包含 active 的 paired Case 1 / Case 2 altbase reduced-feature DGP
- 两个 DGP 均支持 `base1` 到 `base6`，以及 `constant`、`sin`、`sin2`、`mixed` 四种 `sigma^2(t)` 函数

### `data/`

- paired-eye 数据容器
- 负责 subject-level / eye-level 数据组织与基础 shape 约定

### `features/`

- paired 主线的张量分块与投影特征接口
- 当前仍以接口骨架为主

### `metrics/`

- paired 主线使用的误差指标与评估函数

### `models/`

- paired-eye VCTR 的核心模型模块
- 包含结果对象、协方差估计 helper 与主 estimator
- 当前主 estimator 已支持可选的 anchor-grid acceleration，用于大样本下减少 `A(t)` 的 evaluation 点数；默认仍为 `full`，保证旧实验兼容
- 具体模型说明见 [src/models/README.md](/Users/lin/Desktop/Research/2026-tensor/src/models/README.md)

### `experiments/`

- paired 仿真实验入口
- 当前主要包括：
  - `case1_2d_smoke.py`
  - `case1_2d_repetition.py`
  - `case2_3d_smoke.py`
  - `case2_3d_repetition.py`
- 支持可选绘图输出，用于检查估计的 `A[r,s](t)` 与 `sigma^2(t)`
- 历史 constant variance 脚本及结果已归档到：
  - `archive/archive_method_is_const_var/`
- 具体实验参数和输出说明见 [src/experiments/README.md](/Users/lin/Desktop/Research/2026-tensor/src/experiments/README.md)

### `utils/`

- 从旧主线整理出的通用数值工具
- 当前主要保留 kernel、spline、penalty 与绘图相关工具
- 具体工具职责见 [src/utils/README.md](/Users/lin/Desktop/Research/2026-tensor/src/utils/README.md)

## 阅读建议

- 如果你想先看代码结构，从本 README 开始即可
- 如果你想看估计器的输入输出、三阶段实现和参数设计，请直接看 [src/models/README.md](/Users/lin/Desktop/Research/2026-tensor/src/models/README.md)

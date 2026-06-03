# DGPs

本目录保存 paired-eye VCTR 的 Python 模拟数据生成过程。

当前 active DGP 是 [paired_case1_altbase.py](/Users/lin/Desktop/Research/2026-tensor/src/dgps/paired_case1_altbase.py)。另外还提供了一个很薄的 3D-equivalent wrapper [paired_case2_altbase.py](/Users/lin/Desktop/Research/2026-tensor/src/dgps/paired_case2_altbase.py)；旧的 [paired_case1.py](/Users/lin/Desktop/Research/2026-tensor/src/dgps/archive/paired_case1.py) 和 [paired_case2.py](/Users/lin/Desktop/Research/2026-tensor/src/dgps/archive/paired_case2.py) 已归档到 [archive](/Users/lin/Desktop/Research/2026-tensor/src/dgps/archive)。

## 当前实现范围

当前 `src/dgps` 中的实现是 **paired-eye reduced-feature DGP**，不是 raw-tensor DGP。

也就是说，这里的代码直接生成

`y_{ij} = <X_{ij}^*, A(t_i)> + z_i^T beta + epsilon_{ij}`

其中：

- `X_{ij}^* in R^{R x S}`
- `A(t_i) in R^{R x S}`
- 每个 subject 有两只眼，`j in {0, 1}`

当前实现**不会**显式生成原始图像张量，也**没有**显式编码下面这条流程：

`raw tensor -> block partition -> CP decomposition -> stacked reduced features`

因此在解释模拟设定时，要区分：

- 论文层面的 raw-tensor design
- Python 当前代码层面的 reduced-feature implementation

## `paired_case1_altbase.py` 生成什么

[paired_case1_altbase.py](/Users/lin/Desktop/Research/2026-tensor/src/dgps/paired_case1_altbase.py) 中的 `PairedCase1AltbaseDGP` 是当前默认使用的 Case 1 新模拟设定。

它会生成：

- `t_i ~ Uniform(0, 1)`，并排序
- `z_i ~ N(0, I_{p0})`
- `X_{ij}^* ~ N(0, I)`，每个 subject 的形状为 `(2, R, S)`
- `A(t_i)`，形状为 `(R, S)`，并采用可分离结构
  - `A(t_i)[r, s] = base(t_i) * sqrt(r / R) * sqrt(s / S)`
- paired-eye 噪声，服从 exchangeable covariance
  - `Sigma_i = sigma2(t_i) * [[1, rho], [rho, 1]]`
  - `sigma2(t)` 由 `sigma2_function` 控制，可为常数或随 `t` 变化

默认参数为：

- `n_subject = 1000`
- `R = 4`
- `S = 25`
- `p0 = 4`
- `coef_type = "base1"`
- `beta_true = (2.0, 1.0, -1.0, 0.5)`（未显式传入时使用）
- `sigma2 = 1.0`
- `sigma2_function = "constant"`
- `rho = 0.3`

## Altbase 的 6 个 base

当前支持 6 个系数函数基底：

- `base1(t) = 5.0 * (t - 0.2)^2`
- `base2(t) = exp(-((3t - 1)^2)) - 0.75`
- `base3(t) = sin(2pi(t - 0.5))`
- `base4(t) = 0.45 * base1(t) + 0.35 * base2(t) + 0.20 * base3(t)`
- `base5(t) = 1.10 * exp(-0.5 * ((t - 0.30) / 0.08)^2) - 0.95 * exp(-0.5 * ((t - 0.72) / 0.11)^2)`
- `base6(t) = 18.0 * (t - 0.2) * (t - 0.55) * (t - 0.85)`，并在当前样本的 `t` 网格上中心化

因此 `coef_type` 必须是：

- `base1`
- `base2`
- `base3`
- `base4`
- `base5`
- `base6`

## DGP variance functions

当前 DGP 方差函数由 `sigma2_function` 控制，支持：

- `constant`: `sigma2(t) = sigma2`
- `sin`: `sigma2(t) = sigma2 * (1 + 0.3 * sin(2pi t))`
- `sin2`: `sigma2(t) = sigma2 * (0.5 + 0.5 * sin(pi t)^2)`
- `mixed`: `sigma2(t) = sigma2 * (1 + 0.25 * cos(2pi t) + 0.1 * sin(4pi t))`

这些函数只改变 DGP 的真实误差方差曲线；估计器仍通过 `exchangeable_varying_sigma` 从 stage-1 残差估计 `\hat{\sigma}^2(t)` 和共享 `\hat{\rho}`。

## 与 raw-tensor 设定的对应

`paired_case1_altbase.py` 仍然是 reduced-feature DGP，但它的 metadata 里保留了一组 raw-equivalent 尺度，便于和论文设定对照：

- `raw_equivalent_p1 = 60`
- `raw_equivalent_p2 = 60`
- `raw_equivalent_p1_prime = 12`
- `raw_equivalent_p2_prime = 12`

这意味着它可被理解为一个 `5 x 5` block 方案的 reduced-feature 对应：

- `S^(1) = 5`
- `S^(2) = 5`
- `S = 25`

这里的 raw 尺度只是解释层面的映射，不是代码里显式生成的张量维度。

## 输出对象

`sample(seed=...)` 返回 [PairedEyeDataset](/Users/lin/Desktop/Research/2026-tensor/src/data) 风格的数据对象，核心字段包括：

- `subject_ids`
- `eye_ids`
- `t`
- `X`
- `Z`
- `y`
- `A_true`
- `beta_true`
- `Sigma_true`
- `meta`

其中 `meta["dgp"] = "paired_case1_altbase"`，`meta["case"] = "case1_altbase"`。

## `paired_case2_altbase.py` 生成什么

[paired_case2_altbase.py](/Users/lin/Desktop/Research/2026-tensor/src/dgps/paired_case2_altbase.py) 表示当前的 3D-equivalent altbase 设计。

它是一个 **thin wrapper**：

- 不会重新实现新的 paired-eye 生成机制
- 直接复用 `PairedCase1AltbaseDGP` 的全部采样逻辑
- 只修改默认 reduced-feature 规模和 metadata

因此，它和 `paired_case1_altbase.py` 的共同点是：

- 同样生成 `t_i ~ Uniform(0,1)`、`z_i ~ N(0, I_{p0})`
- 同样直接生成 `X_{ij}^* ~ N(0, I)`，而不是显式 raw tensor
- 同样使用 `A(t_i)[r,s] = base(t_i) * sqrt(r/R) * sqrt(s/S)`
- 同样使用 `base1` 到 `base6`
- 同样使用 paired-eye exchangeable covariance
- 同样默认 `p0 = 4` 和 `beta_true = (2.0, 1.0, -1.0, 0.5)`

它与 `paired_case1_altbase.py` 的主要区别只有：

- 默认 `R = 3`
- 默认 `S = 27`
- metadata 标签改为
  - `meta["dgp"] = "paired_case2_altbase"`
  - `meta["case"] = "case2_altbase"`
- raw-equivalent 解释改为 3D 版本

默认参数可理解为下面这组 3D-equivalent 设计：

- `p1 = p2 = p3 = 48`
- `p1' = p2' = p3' = 16`
- `S^(1) = S^(2) = S^(3) = 3`
- `S = 27`
- `R = 3`

但要注意，这里的 raw 尺度仍然只是解释层面的映射。当前代码真正实现的，依然是直接模拟：

- `X_{ij}^* in R^{3 x 27}`
- `A(t_i) in R^{3 x 27}`

也就是说，`paired_case2_altbase.py` 应理解为：

- manuscript 层面的 3D case design
- code 层面的 reduced-feature paired DGP

而**不是**显式 raw-tensor 3D DGP。

## 归档 DGP 简述

[archive/paired_case1.py](/Users/lin/Desktop/Research/2026-tensor/src/dgps/archive/paired_case1.py) 和 [archive/paired_case2.py](/Users/lin/Desktop/Research/2026-tensor/src/dgps/archive/paired_case2.py) 保留的是较早的 paired reduced-feature Case 1 / Case 2 版本。

它们与当前 `altbase` 的共同点是：

- 都是 paired-eye reduced-feature DGP
- 都直接生成 `X_{ij}^*`、`A(t_i)` 和 exchangeable paired noise
- 都没有显式实现 raw tensor 到 reduced feature 的分块与 CP 分解流程

它们与当前 `altbase` 系列的主要区别是：

- 默认规模不同
- 旧版使用 `sqrt / quadratic / bump / sin` 这组 base
- 当前 active 版本改成了 `base1` 到 `base6` 的新 altbase 设计
- 当前 active 版本同时支持
  - `paired_case1_altbase.py`：2D-equivalent `R = 4, S = 25`
  - `paired_case2_altbase.py`：3D-equivalent `R = 3, S = 27`

如果只是运行当前 active altbase 模拟：

- 2D-equivalent 设计优先使用 [paired_case1_altbase.py](/Users/lin/Desktop/Research/2026-tensor/src/dgps/paired_case1_altbase.py)
- 3D-equivalent 设计优先使用 [paired_case2_altbase.py](/Users/lin/Desktop/Research/2026-tensor/src/dgps/paired_case2_altbase.py)

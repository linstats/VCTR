# 变系数张量回归项目

这个仓库服务于 Varying Coefficient Tensor Regression（VCTR）研究，目前需要明确区分三条代码线：

- `code_and_data/`
  - MATLAB `iid VCTR` 基线代码
- `archive/python_iid_vctr/`
  - 上述 `iid` 基线的 Python 归档移植
- `src/`
  - 当前 active 的 paired-eye VCTR Python 主开发线

当前 Python 主线已经不再承担 `iid` 复现任务，而是明确转向双眼 paired VCTR。

## 方法主线

仓库当前对应两份核心参考材料：

- [VaryingCoefLM.pdf](/Users/lin/Desktop/Research/2026-tensor/VaryingCoefLM.pdf)
  - `iid VCTR` 基线参考文章
- [VaryingCoefPLM.pdf](/Users/lin/Desktop/Research/2026-tensor/VaryingCoefPLM.pdf)
  - paired-eye VCTR 主目标参考文章

当前 paired-eye VCTR 的目标模型可以写作

$$
y_{ij} = \langle \mathcal{X}_{ij}, \mathcal{A}(t_i)\rangle + \mathbf{z}_i^\top \boldsymbol{\beta} + \epsilon_{ij},
$$

其中：

- `\mathcal{A}(t)` 是随 `t` 变化的张量系数函数；
- 同一受试者双眼误差需要按受试者内相关结构建模；
- 当前 Python 主线默认采用 `exchangeable_varying_sigma`，即 `\sigma^2(t)` 随 `t` 变化，而 `\rho` 为共享常数；
- `\boldsymbol{\beta}` 在当前实现中仍是全局常向量，不随 `t` 变化。

当前 `src/` 中的实现遵循三阶段思路：

1. working `iid` 拟合 `A^\dagger(t)` 与 `\beta^\dagger`
2. 基于第一阶段残差估计 `\sigma^2(t)` 与 `\rho`
3. 用协方差加权重估 `A^*(t)`，再基于 `y^*` 做 GLS 得到最终 `\beta^*`

## 仓库结构

### `code_and_data/`

- He Jiaxin 留下来的 MATLAB 基线代码
- 当前统一视为 `iid VCTR` 代码，不应默认解释为 paired-eye 已实现

### `archive/python_iid_vctr/`

- 旧的 Python `iid` 研究线归档
- 主要用于保留历史移植结果和与 MATLAB 基线对照

### `src/`

- 当前 paired-eye VCTR 的主开发线
- 包含 paired 仿真、数据容器、模型、实验入口与通用工具
- 当前实验主入口已切换到 altbase 版本，并以 `exchangeable_varying_sigma` 为默认正式方案

### `hpc/`

- NUS HPC 相关目录
- 当前用于集中放置 HPC 说明、现行 PBS 模板与历史 PBS 模板
- 历史 Case 2 PBS 脚本现已整理到 `hpc/archive/`

## 阅读顺序

如果你是第一次看这个仓库，建议按下面顺序进入：

1. 先读本 README，确认项目主线和目录结构
2. 再看 [src/README.md](/Users/lin/Desktop/Research/2026-tensor/src/README.md)，了解当前 Python 主线的代码组织
3. 如果需要理解估计器实现，再看 [src/models/README.md](/Users/lin/Desktop/Research/2026-tensor/src/models/README.md)

## 当前状态

- `code_and_data/` 仍是 `iid VCTR` 的 MATLAB 基线
- `archive/python_iid_vctr/` 保留旧的 Python `iid` 研究线
- `src/` 默认面向 paired-eye VCTR
- 当前 paired Python 主线已经实现 covariance-aware 的三阶段估计框架，并以 `exchangeable_varying_sigma` 为默认模式
- 当前 paired 实验主入口为 `src/experiments/paired_case1_altbase_repetition.py`

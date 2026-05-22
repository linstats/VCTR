# 变系数张量回归

这个仓库服务于 Varying Coefficient Tensor Regression（VCTR）项目，但现在需要明确区分三条线：

- MATLAB `iid VCTR` 基线：`code_and_data/`
- 归档的 Python `iid VCTR` 移植：`archive/python_iid_vctr/`
- 当前主开发线：`src/` 中的 paired-eye VCTR

当前 Python 主线已经不再承担 `iid` 复现任务，而是明确转向双眼 paired VCTR。

## 论文关系

这个仓库当前对应两份核心参考材料：

- [VaryingCoefLM.pdf](/Users/lin/Desktop/Research/2026-tensor/VaryingCoefLM.pdf)
  - 对应 `iid VCTR` 基线。
  - `code_and_data/` 是 He Jiaxin 留下来的 MATLAB 基线代码。
  - `archive/python_iid_vctr/` 保存了这套 `iid` 代码的 Python 移植历史版本。

- [VaryingCoefPLM.pdf](/Users/lin/Desktop/Research/2026-tensor/VaryingCoefPLM.pdf)
  - 对应 paired-eye VCTR。
  - `src/` 现在默认服务于这条主线。

## 当前模型定位

### 1. 基线模型：iid VCTR

现有 MATLAB 基线及其归档 Python 移植，处理的是 `iid VCTR`，即张量效应随索引变量变化，但不显式建模同一受试者双眼之间的相关性。

可写作

$$
y_i = \langle \mathcal{X}_i, \mathcal{A}(t_i)\rangle + \mathbf{z}_i^\top \boldsymbol{\beta} + \epsilon_i.
$$

其中：

- $`\mathcal{X}_i`$ 是张量图像协变量；
- $`t_i`$ 是索引变量；
- $`\mathcal{A}(t_i)`$ 是随 $`t_i`$ 变化的张量系数；
- $`\mathbf{z}_i`$ 是标量协变量；
- 误差按 `iid` 路线处理。

### 2. 当前主线模型：paired-eye VCTR

当前 Python 主线的目标是 paired-eye VCTR，对应设定为

$$
y_{ij} = \langle \mathcal{X}_{ij}, \mathcal{A}(t_i)\rangle + \mathbf{z}_i^\top \boldsymbol{\beta} + \epsilon_{ij},
$$

其中第 $`i`$ 个受试者有第 $`j`$ 只眼睛的数据。与 `iid` 基线相比，paired 版本的关键区别是：

- 张量协变量是按眼睛索引的 $`\mathcal{X}_{ij}`$；
- 响应是按眼睛索引的 $`y_{ij}`$；
- 同一受试者两只眼的误差项 $`(\epsilon_{i1}, \epsilon_{i2})^\top`$ 需要按受试者内相关结构建模；
- Python 主线的接口、目录结构和实验入口都以 paired 为默认目标，而不是继续兼容 `iid`。

当前 `src/` 中的 paired 主线，采用的三阶段实现约定是：

1. 第一阶段：先按 working `iid` 模型估计

   ```math
   \hat{\mathcal{A}}^{\dagger}(t_i)
   \;\to\;
   y_{ij}^{\dagger}
   =
   y_{ij}
   -
   \langle \mathbf{X}_{ij}^{*}, \hat{\mathbf{A}}^{\dagger}(t_i)\rangle
   \;\to\;
   \hat{\boldsymbol{\beta}}^{\dagger}.
   ```

2. 第二阶段：用第一阶段残差估计受试者内协方差

   ```math
   \hat{\Sigma}
   =
   \hat{\sigma}^{2}
   \begin{pmatrix}
   1 & \hat{\rho} \\
   \hat{\rho} & 1
   \end{pmatrix}.
   ```

3. 第三阶段：带入 $`\hat{\Sigma}`$ 做加权重估

   ```math
   \hat{\mathcal{A}}^{*}(t_i)
   \;\to\;
   y_{ij}^{*}
   =
   y_{ij}
   -
   \langle \mathbf{X}_{ij}^{*}, \hat{\mathbf{A}}^{*}(t_i)\rangle
   \;\to\;
   \hat{\boldsymbol{\beta}}^{*}.
   ```

这里第 3 阶段明确采用 $`A^{*} \to y^{*} \to \beta^{*}`$ 的闭环实现，而不是继续使用第一阶段的 $`y^{\dagger}`$。
当前默认 `ridge = 0`，以保持与论文第 2.3 节的无正则化公式一致；若后续手动开启 ridge，应理解为数值稳定策略，而不是论文原式的一部分。

## 仓库结构

### 根目录文件

- `VaryingCoefLM.pdf`
  - `iid VCTR` 基线参考文章。

- `VaryingCoefPLM.pdf`
  - paired-eye VCTR 主目标参考文章。

- `AGENTS.md`
  - 本地协作与实现约束说明。

### MATLAB 基线代码：`code_and_data/`

`code_and_data/` 是 He Jiaxin 留下来的 MATLAB 基线代码，应统一视为 `iid VCTR` 代码。

- `code_and_data/simulation`
  - `iid VCTR` 模拟实验主代码。
  - 包含 `est_vctr_case1.m` 到 `est_vctr_case4_refine.m` 等主脚本。

- `code_and_data/real_data`
  - 真实数据分析代码，包含 MATLAB 和 R。
  - 即便脚本名中带 `eye`，当前也不应默认解释为“paired-eye VCTR 已实现完成”。

- `code_and_data/toolbox`
  - 依赖的 MATLAB 工具箱。

### 归档的 Python `iid` 代码：`archive/python_iid_vctr/`

这里保存的是先前针对 `code_and_data/` 的 Python 移植结果，主要用于：

- 保留移植痕迹；
- 和 MATLAB 基线对照；
- 避免与当前 paired 主线混淆。

该目录不是当前开发主线。

### 当前 Python 主线：`src/`

`src/` 现在已经转为 paired-eye VCTR 的主开发线，不再保留 `iid Case 1-4` 的主线入口。

当前结构为：

- `src/data`
  - paired-eye 数据容器。
  - 默认按 subject-level 与 eye-level 两层组织数据。

- `src/features`
  - paired 主线的张量分块与投影特征接口。

- `src/metrics`
  - paired 主线可复用的误差和结构识别指标。

- `src/models`
  - paired-eye VCTR 的核心模型接口与实现位置。

- `src/experiments`
  - paired 仿真实验入口。
  - 当前已落地的是 `paired_case1_smoke.py` 与 `paired_case1_repetition.py`。

- `src/utils`
  - 从旧 `iid` 主线提炼出来的通用 kernel / spline / penalty 工具。

## 代码与研究路线的对应关系

目前仓库里的三条线对应关系是：

1. `code_and_data/`
   He Jiaxin 留下来的 MATLAB `iid VCTR` 基线。

2. `archive/python_iid_vctr/`
   上述 MATLAB `iid` 基线的 Python 移植归档。

3. `src/`
   当前 active 的 paired-eye VCTR 主开发线。

因此，当前研发路线应理解为：

`保留 iid MATLAB 基线 -> 保存 iid Python 归档 -> 在 src/ 上直接开发 paired-eye VCTR。`

## 当前状态

现阶段最重要的事实有三点：

- `code_and_data/` 仍是 `iid VCTR` 的权威 MATLAB 基线；
- `archive/python_iid_vctr/` 保存了旧的 Python `iid` 研究线；
- `src/` 现在默认面向 paired-eye VCTR，而不是继续服务 `iid` 复现。

## 使用上的注意事项

- 许多 MATLAB 脚本仍依赖硬编码绝对路径，运行前通常要先改数据路径或图像路径。
- GRAPE 数据集需要单独下载，本仓库不直接包含完整原始数据。
- 当前若讨论 Python 主线代码，默认应按 paired-eye VCTR 理解；若讨论 `iid` Python 移植，应明确指向 `archive/python_iid_vctr/`。

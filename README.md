# 变系数张量回归

这个仓库用于整理 Varying Coefficient Tensor Regression（VCTR）项目的论文材料与代码实现，主要包含三部分内容：

- 论文与修改相关材料；
- 论文原始 MATLAB 实现；
- 面向后续重构与复现的 Python 框架。

## 模型概述

这个项目研究的是：当协变量是张量图像时，张量效应如何随着一个索引变量平滑变化。在论文的双眼配对设定中，第 $`i`$ 个受试者第 $`j`$ 只眼睛的响应写作

$$
y_{ij} = \langle \mathcal{X}_{ij}, \mathcal{A}(t_i)\rangle + \mathbf{z}_i^\top \boldsymbol{\beta} + \epsilon_{ij},
$$

其中：

- $`\mathcal{X}_{ij}`$ 是张量图像协变量；
- $`t_i`$ 是索引变量，论文中主要是年龄；
- $`\mathcal{A}(t_i)`$ 是随 $`t_i`$ 平滑变化的张量系数；
- $`\mathbf{z}_i`$ 是标量协变量向量；
- 同一受试者左右眼之间的相关性通过误差结构建模。

这篇论文的方法可以分成两层理解：

1. 张量降维。
   先把每个张量划分成若干空间子块，再在每个子块内做低秩 CP 分解，从而得到降维后的表示 $`\mathbf{X}_{ij}^{*}`$。

2. 变系数估计。
   对降维后的系数函数 $`\mathbf{A}(t)`$，可以采用两类方法：
   - 非稀疏版本：使用局部线性核平滑估计；
   - 稀疏/结构识别版本：使用 B-spline 展开加分组惩罚，再进行 refine。

在实现层面，常用的约化模型写法是

$$
y_{ij} \approx \langle \mathbf{X}_{ij}^{*}, \mathbf{A}(t_i)\rangle + \mathbf{z}_i^\top \boldsymbol{\beta} + \epsilon_{ij}.
$$

## 仓库结构

### 根目录文件

- `VaryingCoefPLM.pdf`：当前 VCTR 主论文快照。
- `VaryingCoefLM.pdf`：相关论文材料。
- `CRediT example.pdf`：作者贡献说明参考材料。
- `AGENTS.md`：仓库级工作说明与约束。

### 原始 MATLAB 代码：`code_and_data `

需要注意，这个目录名末尾带一个空格：`code_and_data `。后续在命令行中操作这个目录时需要显式加引号。

这个目录是之前学生留下的基线实现，也是目前最接近论文原始流程的代码来源。

- `code_and_data /simulation`
  - 论文模拟实验的 MATLAB 主代码。
  - `est_vctr_case1.m`、`est_vctr_case2.m`：对应主要 2D / 3D 非稀疏估计设定。
  - `est_vctr_case3_refine.m`、`est_vctr_case4_refine.m`：对应带结构识别与 refine 的设定。
  - `pred_*`、`plot_*`：用于生成预测结果汇总和论文图形。
  - `dp.m`、`bspline_basis.m`、`bspline_basismatrix.m` 等是共享辅助函数。

- `code_and_data /real_data`
  - GRAPE 眼底图像真实数据分析代码，包含 MATLAB 和 R。
  - `data_process.R`：真实数据预处理入口。
  - `eye_select_RS.m`：对分块数 $`S`$ 和秩 $`R`$ 做交叉验证选择。
  - `eye_penalty_ref.m`：惩罚估计与 refine 主流程。
  - `eye_bootstrap*.m`：bootstrap 与区间估计相关脚本。
  - `pred_eye_*.m`：真实数据上的模型比较脚本。

- `code_and_data /toolbox`
  - 原始 MATLAB 实现依赖的第三方工具箱。
  - 包括 `tensor_toolbox`、`TensorReg` 和 `SparseReg`。

### Python 重构代码：`src/`

`src/` 不是对 MATLAB 的逐句翻译，而是想把原先“按 case 复制脚本”的实验代码，整理成更清晰的模块化框架，便于扩展新的 DGP、估计器和评估流程。

- `src/data`
  - 数据容器与模拟数据统一表示。
  - 当前核心文件是 `dataset.py`。

- `src/dgps`
  - 模拟数据生成模块。
  - `case1_baseline.py` 到 `case4_baseline.py` 对应论文中的四类主要模拟设定。
  - `base.py` 提供共享的 DGP 抽象。

- `src/metrics`
  - 模拟评估指标模块。
  - 当前已有 `estimation.py` 和 `selection.py`。

- `src/features`
  - 预留给张量特征工程的模块。
  - 后续适合放分块、投影、真实图像基构造等功能。

- `src/vctr`
  - 预期用于放置主要估计器与 refine 逻辑。
  - 目前仍然是框架骨架，尚未完整承接 MATLAB 工作流。

- `src/experiments`
  - 面向复现实验的入口脚本。
  - `reproduce_case1_matlab.py` 到 `reproduce_case4_matlab.py` 主要用于按 MATLAB 口径重现论文中的模拟设定。

## 方法与代码的对应关系

如果把论文方法看成一个流水线，大致可以拆成：

1. 生成或预处理张量数据；
2. 将张量划分为空间子块；
3. 在各子块上构造低维表示；
4. 估计变系数张量效应和标量协变量效应；
5. 需要时再做 spline + penalty 的结构识别；
6. 在筛除零项、压缩常数项后做 refine；
7. 评估估计误差、变量选择效果和预测性能。

在这个仓库里，这条流程目前分布为：

- 与原论文实现最贴近的代码：`code_and_data /simulation` 和 `code_and_data /real_data`；
- 面向后续工程化重构的代码：`src/data`、`src/dgps`、`src/features`、`src/vctr`、`src/metrics`、`src/experiments`。

## 当前状态

就现阶段而言，MATLAB 代码仍然是论文实验与真实数据分析的权威参考实现；Python 代码的价值主要在于工程重构，尤其包括：

- 将 DGP 和估计器解耦；
- 把 Case I-IV 从“脚本名”改造成“配置驱动实验”；
- 为后续补充 paired-eye 结构、更多 DGP、统一实验入口做准备。

## 使用上的注意事项

- 许多 MATLAB 脚本仍然依赖硬编码的绝对路径，运行前通常需要先改数据路径或图像路径。
- GRAPE 数据集需要单独下载，本仓库并不直接包含原始真实数据。
- 由于 `code_and_data ` 目录名末尾带空格，命令行中引用该路径时应始终加引号。

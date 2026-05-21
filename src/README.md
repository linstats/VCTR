# `src` 目录设计说明

## 目标

这里用于承载 VCTR 的 Python 重构版本。重构目标不是逐句翻译 MATLAB，而是把当前 simulation 脚本改成一个可扩展的实验框架，方便支持新的 DGP：

- 空间相关图像：Gaussian random field / Gaussian process
- matrix normal / tensor normal
- 基于真实图像 PCA 的原图空间生成

核心原则：

- 按模块组织，不按 Case 复制脚本
- DGP、估计方法、refine、评估彼此解耦
- 两类估计方法和 refine 都应成为可调用模块

## 建议目录

```text
src/
  README.md
  data/
  dgps/
  metrics/
  features/
  experiments/
    results/
  vctr/
    estimators/
    refine/
    utils/
```

## 各模块职责

### `data/`

负责统一数据容器。

当前已实现：

- `SimulationDataset`

### `dgps/`

负责生成模拟数据，统一输出 `t, X, Z, y, A_true, beta_true`。

当前已实现：

- `case1_baseline.py`
- `case2_baseline.py`
- `case3_baseline.py`
- `case4_baseline.py`

后续计划支持：

- `gaussian_field.py`
- `matrix_normal.py`
- `pca_image.py`

### `features/`

负责把原始图像 `X` 转成方法实际使用的特征表示。

当前还是预留模块，后续主要用于原图空间生成、分块和投影表示。

建议包括：

- 分块 `partition.py`
- CP/张量投影 `cp_projection.py`
- PCA basis `real_image_basis.py`

### `vctr/estimators/`

负责两类主要估计方法。

建议拆成：

- `local_linear.py`
  - 对应 Section 2 的非稀疏局部线性核平滑
  - 默认评估模式使用 `matlab_middle_random`，即随机抽取中间区间评估点，以贴近原 MATLAB simulation 设计
- `penalized_spline.py`
  - 对应 Section 3 的 penalized spline + structure identification

### `vctr/refine/`

负责 refine / re-estimation。

建议单独放：

- `kernel_refit.py`

不要把 refine 隐式写进 penalized estimator 里，因为后续需要单独比较：

- unpenalized
- penalized only
- penalized + refine
- oracle refine

### `metrics/`

负责 simulation 指标：

- estimation error
- selection accuracy
- prediction error

### `experiments/`

放当前阶段的可直接运行实验入口，包括：

- smoke run
- Case 1/2 repetition
- 后续单个 case 的实验入口

当前已实现：

- `run_case1_smoke.py`
- `run_case1_repetition.py`
- `run_case2_smoke.py`
- `run_case2_repetition.py`
- `run_case3_smoke.py`
- `run_case3_repetition.py`
- `run_case4_smoke.py`
- `run_case4_repetition.py`
- `reproduce_case1_matlab.py`
- `reproduce_case2_matlab.py`
- `reproduce_case3_matlab.py`
- `reproduce_case4_matlab.py`

实验结果默认写到：

- `experiments/results/`

其中：

- `run_case*.py` 是当前通用实验入口
- `reproduce_case*_matlab.py` 是按原 MATLAB simulation 口径整理的复现实验脚本

核心方法实现仍然放在 `vctr/` 下，不放在这里。

## 推荐的最小接口

当前实现的最小主流程是：

```python
dataset = dgp.sample(seed=...)
result = estimator.fit(dataset.X, dataset.Z, dataset.y, dataset.t)

if refiner is not None:
    result = refiner.refit(dataset, result)

metrics = evaluator.evaluate(result, dataset)
```

后续如果引入原图空间特征工程，再补入 `feature_map` 这一层。

## 为什么要这样拆

当前 MATLAB 脚本的问题是：

- DGP 和 estimator 写死在一起
- kernel smoothing 代码重复
- refine 和 structure identification 耦合
- 新 DGP 只能继续复制新脚本

迁移到 Python 后，应把“论文 case”变成配置，而不是文件名。

## 建议迁移顺序

1. 已完成 `local_linear.py`，并跑通 Case 1/2
2. 已完成 `penalized_spline.py`，并跑通 Case 3/4 的结构识别主链
3. 已完成 `kernel_refit.py`，并接通 Case 3/4 的 refine
4. 已补充 `reproduce_case1-4_matlab.py`，用于按 MATLAB simulation 口径复现
5. 后续重点是：
   - paired-eye / `y_{ij}` 结构升级
   - 新 DGP 模块
   - prediction / plotting 相关迁移

## 当前结论

这次重构的关键不是把 MATLAB 改成 Python，而是把 simulation 代码改成：

- `DGP`
- `Estimator`
- `Refine`
- `Metrics`
- `Experiment`

五部分解耦的框架。

其中最优先独立出来的模块是：

- `estimators/local_linear.py`
- `estimators/penalized_spline.py`
- `refine/kernel_refit.py`

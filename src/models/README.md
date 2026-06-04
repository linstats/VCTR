# `src/models/` 模型模块说明

这个目录承载 paired-eye VCTR 的核心模型实现。当前它不是只包含主 estimator，而是由三部分共同组成：

- `base.py`
  - 定义模型层公共结果对象与抽象接口
- `covariance.py`
  - 定义协方差估计与 residual 重组 helper
- `paired_vctr.py`
  - 定义当前主 estimator `PairedEyeVCTRModel`

## 1. 模块概览

当前实现围绕一个三阶段 paired-eye VCTR 工作流展开：

1. stage 1：working `iid` 拟合
2. stage 2：基于 stage-1 残差估计受试者内协方差
3. stage 3：用协方差加权重估 `A(t)` 和最终 `\beta`

当前默认协方差模式是：

- `exchangeable_varying_sigma`

可选旧模式是：

- `exchangeable_constant`

## 2. 输入数据约定

主 estimator 的输入是 `PairedEyeDataset`，定义在 `src/data/paired_dataset.py`。

当前模型默认假设输入已经在 reduced-feature 空间，即：

- `X` 的 shape 为 `(n_subject, 2, *feature_shape)`
- 对当前 paired Case 1 / Case 2 仿真，通常已经是 `(n_subject, 2, R, S)` 或可展平为 `(n_subject, 2, p)`

其余核心输入约定：

- `t`
  - shape `(n_subject,)`
  - subject-level index variable
- `Z`
  - shape `(n_subject, p0)`
  - 标量协变量
- `y`
  - shape `(n_subject, 2)`
  - 每个 subject 两只眼的响应

## 3. 三阶段估计流程

### Stage 1：working `iid`

`paired_vctr.py` 先把 paired 数据摊平成 eye-level `iid` 视图，然后做局部线性回归，得到：

- `\hat A^\dagger(t_i)`
- `y_{ij}^\dagger = y_{ij} - \langle X_{ij}^*, \hat A^\dagger(t_i)\rangle`
- `\hat\beta^\dagger`

这一步由 `PairedEyeVCTRModel.initial_fit_iid()` 驱动，内部主要使用：

- `_fit_initial_iid_with_bandwidth()`
- `_estimate_stage1_A()`
- `_solve_beta_ols()`

### Stage 2：协方差估计

这一步使用 stage-1 残差来估计受试者内协方差。

当前支持两种模式。

#### `exchangeable_constant`

假设所有 subject 共用同一个 marginal variance：

$$
\Sigma_0 =
\sigma^2
\begin{pmatrix}
1 & \rho \\
\rho & 1
\end{pmatrix}.
$$

这一路径主要由 `covariance.py` 中的：

- `estimate_exchangeable_covariance()`

完成。

#### `exchangeable_varying_sigma`

这是当前默认模式。它允许：

- `\sigma^2(t)` 随 `t` 变化
- `\rho` 仍是共享常数

对应的 subject-level covariance block 为：

$$
\Sigma_i
=
\sigma^2(t_i)
\begin{pmatrix}
1 & \rho \\
\rho & 1
\end{pmatrix}.
$$

这一路径主要由 `covariance.py` 中的：

- `estimate_exchangeable_varying_sigma_covariance()`

完成。

它会先：

- 把 flat residual regroup 成 `(n_subject, 2)` 的 `residual_pairs`

再：

- 用 kernel smoothing 在 `t_i` 上估计 `sigma2_hat_t`
- 估计共享 `rho_hat`
- 构造每个 subject 的 `Sigma_hat_blocks`

### Stage 3：协方差加权重估

最终重估由 `PairedEyeVCTRModel.refit_with_covariance()` 完成。

这一阶段：

1. 用每个 subject 自己的 `2 x 2` 协方差块 `\Sigma_i^{-1}` 做局部加权，重估 `A^*(t)`
2. 再构造

$$
y_{ij}^* = y_{ij} - \langle X_{ij}^*, \hat A^*(t_i)\rangle
$$

3. 最后用 `y^*` 做全局 GLS，得到最终 `\hat\beta^*`

这里要特别注意：

- 当前代码实现明确采用闭环 `A^* -> y^* -> beta^*`
- 最终 `beta_hat` 是单个全局向量，不是 `beta(t)`

## 4. 文件与类说明

### `base.py`

这个文件定义模型层公共接口和结果对象。

#### `BasePairedVCTRModel`

- 抽象基类
- 当前只要求实现 `fit(dataset) -> PairedVCTRResult`

#### `InitialIidResult`

表示 stage 1 的输出，主要包含：

- `A_hat`
  - stage-1 的 `A^\dagger(t)` 估计
- `beta_hat`
  - stage-1 的 `\beta^\dagger`
- `fitted_values`
  - stage-1 拟合值
- `residuals`
  - stage-1 残差
- `subject_ids` / `eye_ids`
  - 用于后续 regroup residuals
- `meta`
  - 记录 stage-1 相关带宽和中间量

#### `CovarianceEstimate`

表示 stage 2 的输出，主要包含：

- `covariance_mode`
  - 当前使用的协方差模式
- `rho_hat`
  - 共享相关系数估计
- `sigma2_hat_t`
  - subject-level 的 `\sigma^2(t_i)` 估计
- `Sigma_hat_blocks`
  - 每个 subject 的 `2 x 2` covariance block
- `Sigma_hat`
  - 汇总型 `2 x 2` 表示，主要为兼容与摘要用途
- `sigma2_hat`
  - 标量摘要型 marginal variance
- `residual_pairs`
  - regroup 之后的 stage-1 paired residuals
- `meta`
  - stage-2 方法、带宽与数值保护信息

#### `PairedVCTRResult`

表示完整三阶段后的最终输出，主要包含：

- `initial`
  - 对应 stage 1 的 `InitialIidResult`
- `covariance`
  - 对应 stage 2 的 `CovarianceEstimate`
- `A_hat`
  - 最终 `A^*(t)`
- `beta_hat`
  - 最终全局 `\beta^*`
- `fitted_values`
  - 最终拟合值
- `meta`
  - 记录 stage-3 相关带宽、`y_star`、`Sigma_inv_blocks` 等中间结果

### `covariance.py`

这个文件负责协方差层的辅助计算。

#### `regroup_residuals_by_subject()`

- 把 flat residuals 按 `(subject, eye)` 重组为 `(n_subject, 2)`
- 这是 stage 2 从 eye-level 残差回到 paired 结构的关键一步

#### `build_exchangeable_blocks()`

- 根据 `sigma2_hat_t` 和 `rho_hat` 构造 subject-specific `2 x 2` covariance blocks

#### `invert_blocks()`

- 对 `Sigma_hat_blocks` 逐块求逆
- stage 3 依赖这些 `\Sigma_i^{-1}` 做 weighted local fit 和 GLS

#### `estimate_exchangeable_covariance()`

- constant 模式的 helper
- 对所有 subject 使用共享 marginal variance

#### `estimate_exchangeable_varying_sigma_covariance()`

- varying-sigma 模式的 helper
- 用 bandwidth 对 stage-1 squared residuals 做 kernel smoothing
- 输出 `sigma2_hat_t`、`rho_hat` 和 `Sigma_hat_blocks`

### `paired_vctr.py`

这个文件定义当前主 estimator `PairedEyeVCTRModel`。

它负责：

- stage 1 的 working `iid` 拟合
- stage 2 的带宽解析和协方差模式切换
- stage 3 的 covariance-aware refit

主要 public 方法是：

- `fit(dataset)`
- `fit_paired(dataset)`

其中 `fit_paired()` 当前只是 `fit()` 的兼容别名。

## 5. 关键参数

`PairedEyeVCTRModel` 当前最重要的参数如下。

### `covariance_mode`

可选值：

- `exchangeable_varying_sigma`
  - 当前默认模式
- `exchangeable_constant`
  - 旧的共享常方差模式

### `signal_bandwidth*`

这组参数控制 stage 1 和 stage 3 使用的信号平滑带宽 `h`：

- `signal_bandwidth`
- `signal_bandwidth_method`
- `signal_bandwidth_grid`
- `signal_bandwidth_cv_folds`
- `signal_bandwidth_cv_seed`

行为约定：

- 传单个 `signal_bandwidth`
  - 固定使用该值
- 不传单值但传 `signal_bandwidth_grid`
  - 在 estimator 内自动选择
- 两者都不传
  - 使用默认固定值

### `variance_bandwidth*`

这组参数控制 stage 2 中 `\sigma^2(t)` 平滑使用的 `\bar h`：

- `variance_bandwidth`
- `variance_bandwidth_method`
- `variance_bandwidth_grid`
- `variance_bandwidth_cv_folds`
- `variance_bandwidth_cv_seed`

行为约定与 signal bandwidth 类似：

- 传单值则固定
- 传 grid 则自动选择
- 都不传则使用 estimator 默认值

如果 `covariance_mode = exchangeable_constant`，则这一组参数不参与计算。

### `ridge`

- 默认 `ridge = 0`
- 与论文第 2.3 节的无正则化公式保持一致
- 若手动设置为非零，应理解为数值稳定策略

### `a_eval_*`

这组参数控制 stage 1 和 stage 3 中 `A(t)` 的 evaluation grid。

- `a_eval_mode`
- `a_eval_num_points`
- `a_eval_grid`
- `a_interp`

行为约定：

- `a_eval_mode = "full"`
  - 默认模式
  - 在全部 subject 的 `t_i` 上直接估计 `A(t_i)`
- `a_eval_mode = "anchor_grid"`
  - 只在一组 anchor `t0` 上估计 `A(t0)`
  - 再沿 `t` 轴插值回全部 `t_i`
  - 当前首版只支持确定性的 `quantile` / `uniform` grid 和 `linear` interpolation

其中：

- `a_eval_num_points`
  - 请求使用多少个 anchor evaluation points
  - 若 `a_eval_num_points >= n_subject`，则实际行为退化为 full-eval
- `a_eval_grid`
  - 当前支持 `quantile` 和 `uniform`
- `a_interp`
  - 当前支持 `linear`

这组参数只改变 stage 1 / stage 3 外层 `t0` 的估计点数，不改变：

- stage 2 的 `\sigma^2(t)` kernel smoothing 路径
- 最终输出 `A_hat` 的 shape；插值回填后仍是 `(n_subject, R, S)`
- 默认结果口径；不显式启用时仍与旧实验兼容

## 6. 输出对象与结果解释

调用：

```python
result = model.fit(dataset)
```

后，返回的是 `PairedVCTRResult`。

使用时可以按阶段理解：

- `result.initial`
  - stage 1 的输出
- `result.covariance`
  - stage 2 的输出
- `result.A_hat`
  - stage 3 最终 `A^*(t)`
- `result.beta_hat`
  - stage 3 最终全局 `\beta^*`
- `result.fitted_values`
  - stage 3 最终拟合值

如果要进一步看过程信息，可以读 `meta`。

常见有用内容包括：

- stage 1
  - `signal_bandwidth_selected`
  - `signal_bandwidth_method`
  - `a_eval_mode`
  - `a_eval_selected_points`
  - `y_dagger`
- stage 2
  - `variance_bandwidth_selected`
  - `variance_bandwidth_method`
  - `rho_clipped`
- stage 3
  - `a_eval_used_acceleration`
  - `y_star`
  - `Sigma_inv_blocks`

## 7. 当前实现边界

- 当前 estimator 工作在 reduced-feature 空间
- `features/` 下真正的张量分块与 CP 投影仍是后续工作
- 因此当前 `models/` 的输入默认不是原始图像张量，而是已经整理好的 paired reduced features

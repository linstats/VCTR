# DGPs

本目录保存的是 Python 主实现当前正在使用的 paired-eye 模拟数据生成过程。

## 当前范围

当前 Python 中的 DGP 是 **reduced-feature DGP**，不是 raw-tensor DGP。

这意味着，这里的代码**不会**显式生成带有原始张量维度的图像张量，例如：

- Case 1 的 raw tensor 设定：`p1 = p2 = 80`，块大小 `p1' = p2' = 20`
- Case 2 的 raw tensor 设定：`p1 = p2 = p3 = 40`

相反，当前代码是直接生成降维后的表示：

`y_{ij} = <X_{ij}^*, A(t_i)> + z_i^T beta + epsilon_{ij}`

其中：

- `X_{ij}^* in R^{R x S}`
- `A(t_i) in R^{R x S}`
- 每个 subject 有两只眼，`j in {0, 1}`

因此在当前实现里：

- `R` 表示每个 block 的 reduced rank-feature 维度
- `S` 表示降维后 block 或 feature 的个数
- `S = 16` 被用作 `4 x 4` 分块数量的 reduced-form 对应
- `S = 64` 被用作更大分块数量的 reduced-form 对应

当前代码**没有**显式实现下面这条中间流程：

raw tensor -> 按 block 分块 -> 每个 block 做 CP decomposition -> 堆叠成 reduced feature matrix。

## `paired_case1.py` 生成了什么

[paired_case1.py](/Users/lin/Desktop/Research/2026-tensor/src/dgps/paired_case1.py) 中的 `PairedCase1DGP` 会生成：

- `t_i ~ Uniform(0, 1)`，并排序
- `z_i ~ N(0, I_{p0})`
- `X_{ij}^* ~ N(0, I)`，每个 subject 的形状为 `(2, R, S)`
- `A(t_i)`，形状为 `(R, S)`，并采用可分离结构
  - `A(t_i)[r, s] = base(t_i) * sqrt(r / R) * sqrt(s / S)`
- paired-eye 噪声，服从 exchangeable covariance
  - `Sigma = sigma2 * [[1, rho], [rho, 1]]`

目前支持的系数函数基底为：

- `sqrt`
- `quadratic`
- `bump`
- `sin`

## `paired_case2.py` 生成了什么

[paired_case2.py](/Users/lin/Desktop/Research/2026-tensor/src/dgps/paired_case2.py) 中的 `PairedCase2DGP` 与 Case 1 使用**同一套 reduced-feature DGP 结构**。

当前两者在实现上的实际区别只体现在默认参数：

- Case 1 默认值：
  - `n_subject = 2000`
  - `R = 10`
  - `S = 16`
- Case 2 默认值：
  - `n_subject = 1000`
  - `R = 5`
  - `S = 64`

因此就当前代码而言：

- Case 1 和 Case 2 并不是通过不同的 raw-tensor 构造来区分
- 它们是通过不同的 reduced-feature 默认规模和 metadata 标签来区分

## 与 MATLAB baseline 的关系

`code_and_data/simulation` 下继承来的 MATLAB 模拟脚本，同样也是直接生成 reduced-feature 数组：

- Case 1 使用 `X = randn(n, R, S)`，其中 `R = 10`, `S = 16`
- Case 2 使用 `X = randn(n, R, S)`，其中 `R = 5`, `S = 64`

因此，当前 Python DGP 与 MATLAB baseline 在下面这个意义上是一致的：

- 两者都是直接在 reduced-feature 空间中生成数据

但 Python 代码并不是对 MATLAB iid 模拟的逐字复制，因为 Python 额外加入了：

- paired-eye 的 subject 结构
- 通过 `rho` 建模的 subject 内相关性
- paired estimator 中的 covariance-aware refit
- 当前模型拟合流程中的 `sigma(t)` covariance 升级

## 参数量与样本量经验公式

下面这套估算需要分成两种 covariance mode 来看：

- `exchangeable_constant`
- `exchangeable_varying_sigma`

二者在 **stage 1 / stage 3 的信号回归** 上是相同的；
区别在于 `exchangeable_varying_sigma` 还多了一步 **stage 2 的 `sigma^2(t)` kernel smoothing**。

### `exchangeable_constant`

对于当前 [paired_vctr.py](/Users/lin/Desktop/Research/2026-tensor/src/models/paired_vctr.py) 中的 paired estimator，关键的 reduced-feature 数量是：

`q = R * S`

在每一个目标时间点 `t0` 上，局部线性 paired 拟合要求解的参数个数为：

`P_signal = p0 + 2q = p0 + 2RS`

其中包括：

- `p0` 个普通标量协变量系数
- `RS` 个 `A(t0)` 的局部系数项
- `RS` 个局部线性展开带来的斜率项

这意味着：

- 参数量会随着 `R * S` 线性增长
- `R` 或 `S` 翻倍时，局部参数维度大致也会翻倍

#### 有效局部样本量

设：

- `n` 个 subject
- 每个 subject 有 `2` 只眼
- 信号带宽为 `h_signal`
- 指标变量 `t` 大致分布在 `[0, 1]`

则在一个内部点附近的 kernel 窗口中，大约包含：

- `2 h_signal n` 个 subject
- `4 h_signal n` 个 eye-level 标量观测

#### 最低可识别条件

一个最基本的局部可解条件近似为：

`4 h_signal n > p0 + 2RS`

等价地：

`n > (p0 + 2RS) / (4 h_signal)`

这只是局部可识别的下界，不代表数值上已经稳定。

#### 实用参考规则

如果希望估计更稳定，一个比较实用的经验公式是：

`n_subject ~= (p0 + 2RS) / h_signal`

它大致对应“局部有效观测数约为参数数的 4 倍”这一经验比例。

这条公式最适合用来估算 `exchangeable_constant` 下的样本量主尺度。

### `exchangeable_varying_sigma`

在 `exchangeable_varying_sigma` 模式下，stage 1 和 stage 3 的信号回归部分仍然满足上面的同一套公式；
因此 `P_signal = p0 + 2RS` 这一局部回归维度**本身不会改变**。

但完整三阶段流程里还多了一步：

- 用 stage 1 残差估计 `sigma^2(t)`
- 再把这个 `sigma^2(t)` 估计结果送回 stage 3 的 GLS

当前实现中，这一步不是再解一个 `p0 + 2RS` 维的大线性系统，而是对 stage 1 残差平方做一维 kernel smoothing：

- 每个 subject 提供一对残差平方
- 在每个 `t0` 上平滑得到一个标量 `sigma^2(t0)`

因此，单看“局部参数个数”，`varying_sigma` 并不会把 `P_signal` 改写成更大的高维线性系统。

但是，单看参数个数会低估真实样本量需求，因为：

- stage 2 用的是 **估计残差**，不是 oracle residual
- stage 2 的 smoothing 误差会继续传到 stage 3
- 这会让完整流程比 `exchangeable_constant` 更敏感

也就是说：

- **参数维度层面**：不是简单把一个新的高维参数块直接加到 `P_signal` 上
- **样本量层面**：应当把 `sigma(t)` smoothing 视作一项额外负担，不能只取 `max(...)`

#### stage 2 的有效样本量

设方差平滑带宽为 `h_var`。

则在一个内部点附近，stage 2 大约可用的 subject 数为：

`N_var_eff ~= 2 h_var n`

更实用地，可以把它理解为：

- 至少希望每个局部窗口内有 `m_var` 个 subject
- 常用经验范围可取 `m_var = 30 ~ 50`

于是有一个 stage 2 的纯 smoothing 参考：

`n_subject,var_only ~= m_var / (2 h_var)`

#### 完整三阶段的保守参考

在当前 paired three-stage workflow 里，更保守、也更贴近实践的参考应写成：

`n_subject,varying_sigma ~= (p0 + 2RS) / h_signal + delta_sigma`

其中：

- 第一项来自 stage 1 / stage 3 的高维信号估计
- `delta_sigma` 表示 stage 2 的 `sigma^2(t)` smoothing 及其误差传播带来的额外样本量预算

一个实用写法是把 `delta_sigma` 直接表示成第一项的比例惩罚：

`n_subject,varying_sigma ~= k_sigma * (p0 + 2RS) / h_signal`

其中经验上可取：

- `k_sigma = 1.15 ~ 1.35`

这比只写 `max((p0 + 2RS) / h_signal, m_var / (2 h_var))` 更保守，也更符合当前实现的实际行为。

换句话说：

- `exchangeable_constant`:
  - `n_subject,const ~= (p0 + 2RS) / h_signal`
- `exchangeable_varying_sigma`:
  - `n_subject,var ~= 1.15 ~ 1.35 * (p0 + 2RS) / h_signal`

如果想进一步把 `h_var` 显式写进来，可以把 `delta_sigma` 记成一个与 `1 / h_var` 同阶的补充项；
但在当前仓库常见配置下，主导尺度仍然主要由 `(p0 + 2RS) / h_signal` 决定，而 `sigma(t)` smoothing 体现为一个额外惩罚项。

### 示例

假设 `p0 = 2`。

- Case 1 风格默认值：
  - `R = 10`, `S = 16`
  - `RS = 160`
  - `P_signal = 322`
  - 若 `h_signal = 0.13`，则 `exchangeable_constant` 下的实用参考值为
    - `n ~= 322 / 0.13 ~= 2477`
- Case 2 风格默认值：
  - `R = 5`, `S = 64`
  - `RS = 320`
  - `P_signal = 642`
  - 若 `h_signal = 0.25`，则 `exchangeable_constant` 下的实用参考值为
    - `n ~= 642 / 0.25 ~= 2568`

这些数值只是规划时的参考，不是硬性门槛。

## 解释时必须注意的事项

在本仓库讨论 simulation 时，需要明确你描述的是哪一层：

- 论文中的 **raw-tensor design**
- 还是 Python 当前实现中的 **reduced-feature implementation**

对于 `src/dgps` 里的现行代码，正确的描述应当是：

- paired-eye
- reduced-feature
- 直接生成 `X^*` 和 `A(t)`
- 并未显式生成带有 `p1`, `p2`, `p3`, `p1'`, `p2'` 的 raw tensor 维度

## `paired_case1_altbase.py` 生成了什么

[paired_case1_altbase.py](/Users/lin/Desktop/Research/2026-tensor/src/dgps/paired_case1_altbase.py) 中的 `PairedCase1AltbaseDGP` 保持和当前 Case 1 相同的 paired-eye reduced-feature DGP 结构，但替换为一组新的 base 设计：

- 默认 `n_subject = 1000`
- 默认 `R = 4`
- 默认 `S = 25`
- 默认 `p0 = 4`
- 默认 `beta = (2.0, 1.0, -1.0, 0.5)`

4 个 base 定义为：

- `base1(t) = 5.0 * (t - 0.2)^2`
- `base2(t) = exp(-((3t - 1)^2)) - 0.75`
- `base3(t) = sin(2pi(t - 0.5))`
- `base4(t) = 0.45 * base1(t) + 0.35 * base2(t) + 0.20 * base3(t)`

这套设计可以看作下述 raw-tensor 分块方案的 reduced-feature 对应：

- `p1 = p2 = 60`
- `p1' = p2' = 12`
- `S^(1) = S^(2) = 5`
- `S = 25`

在当前实现中，这些 raw-tensor 尺度仍然只是解释层面的对应，不是代码中显式生成的张量维度。

### 样本量与参数量参考

对于这套 altbase 默认值：

- `RS = 4 * 25 = 100`
- `P_signal = p0 + 2RS = 4 + 200 = 204`

若信号带宽取默认 `h_signal = 0.18`，则 `exchangeable_constant` 下经验公式给出：

- 实用参考样本量 `n_subject ~= (p0 + 2RS) / h_signal ~= 204 / 0.18 ~= 1133`

若再考虑 `exchangeable_varying_sigma`，并且方差带宽也取 `h_var = 0.18`，则：

- 纯 stage 2 smoothing 的 subject-level 下界仍然不高
- 但完整 three-stage 稳定性更适合按比例惩罚处理

因此在这套设计下，更推荐写成：

- `exchangeable_constant` 主参考值约为 `1133`
- `exchangeable_varying_sigma` 主参考值约为：
  - `1.15 * 1133 ~= 1303`
  - 到
  - `1.35 * 1133 ~= 1530`

也就是说，实践上更建议：

- `exchangeable_constant`: `1000 ~ 1200`
- `exchangeable_varying_sigma`: `1300 ~ 1500`

当前默认 `n_subject = 1000` 是一个刻意更轻量的默认值：

- 在 `exchangeable_constant` 下接近可用下界
- 在 `exchangeable_varying_sigma` 下明显偏紧
- 便于 smoke 和中小规模 repetition 先跑通
- 若追求更稳的正式结果，建议：
  - `exchangeable_constant` 至少提高到 `1100 ~ 1200`
  - `exchangeable_varying_sigma` 提高到 `1300 ~ 1500`

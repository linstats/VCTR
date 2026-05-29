# Utils

`src/utils/` 保存 paired-eye VCTR 主线可复用的轻量工具。这里不放实验入口，也不放模型估计流程；这些分别属于 `src/experiments/` 和 `src/models/`。

## 文件职责

- `kernels.py`: kernel smoothing 使用的 Epanechnikov kernel 与权重函数。
- `splines.py`: B-spline basis 和 open-uniform knots 构造工具，用于后续 sparse / structure-identifying 版本。
- `penalties.py`: Lasso / SCAD / MCP 等 penalty derivative 与 LQA 权重工具。
- `plotting.py`: 实验诊断图工具，用于画估计和真值的 `A[r,s](t)` 与 `sigma^2(t)`。

## 绘图工具

`plotting.py` 当前提供两个公开 helper：

- `parse_a_indices(spec, shape)`: 解析 `A` 分量索引字符串。
- `save_function_plots(...)`: 保存 `A` 函数图和 `sigma^2(t)` 函数图。

索引约定：

- `A` 分量索引使用 Python/NumPy 的 0-based index。
- 例如论文记号 `A[1,1]`、`A[4,1]`，在 CLI 中写作 `0:0,3:0`。
- `all` 表示按 row-major 顺序选择全部 `R x S` 分量，再由 `max_a_panels` 截断。

绘图行为：

- `A` 图会画 final estimate；如果存在，也会叠加 stage-1 estimate 和模拟真值。
- `sigma^2(t)` 图会画 `result.covariance.sigma2_hat_t`；如果能从 `dataset.meta` 或 `dataset.Sigma_true` 推断真值，也会叠加真值。
- `exchangeable_constant` 下的 `sigma2_hat_t` 是常数向量，因此自然画成水平线。
- `exchangeable_varying_sigma` 下的 `sigma2_hat_t` 随 subject-level `t` 变化。

## 使用位置

通常不要在模型层调用绘图工具。当前设计是：

- `src/models/`: 只负责拟合并返回 `result`。
- `src/dgps/`: 只负责生成 `dataset` 和模拟真值。
- `src/experiments/`: 在用户显式传入 `--plot-functions` 时调用 `src.utils.plotting` 并保存图。

这保证了 estimator API 不因诊断图功能而改变，也保证默认实验脚本仍可按原命令运行。

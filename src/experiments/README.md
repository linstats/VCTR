# Experiments

本目录保存当前正在使用的 paired-eye 实验入口脚本。

## 当前脚本

- `case1_2d_smoke.py`
  - 用于 alternative-base paired Case 1 设计的单次 smoke test。
  - 使用较新的 `base1` 到 `base6` 系数函数设定，并采用了估计量 `\hat{\sigma}(t)`。
- `case1_2d_repetition.py`
  - 用于 alternative-base paired Case 1 设计的重复模拟主脚本。
  - 这是当前 2D-equivalent altbase varying-sigma 实验的主要入口脚本。
- `case2_3d_smoke.py`
  - 用于 3D-equivalent alternative-base paired Case 2 设计的单次 smoke test。
  - 默认对应 reduced-feature `R = 3, S = 27`。
- `case2_3d_repetition.py`
  - 用于 3D-equivalent alternative-base paired Case 2 设计的重复模拟主脚本。
  - 核心单次拟合逻辑复用了 `case2_3d_smoke.py`。

这 4 个脚本都采用当前 paired-eye altbase DGP，并默认使用 `varying_sigma` 工作流。

## 主要参数

常用数据生成参数：

- `--n-subject` 或 `--n-subject-values`: subject 数量。
- `--coef-type` 或 `--coef-types`: 系数函数类型，当前支持 `base1` 到 `base6`。Case 1 repetition 默认运行 `base1` 到 `base4`；Case 2 repetition 默认覆盖 `base1` 到 `base6`。
- `--R`, `--S`: reduced-feature 维度，其中 `A(t)` 与 `X^*` 的形状为 `R x S`。
- `--p0`, `--beta`: 协变量维度与真实 `beta`。
- `--sigma2`, `--rho` / `--rho-values`: paired-eye 误差方差与眼间相关。
- `--sigma2-function` / `--sigma2-functions`: DGP 方差函数，支持 `constant`, `sin`, `sin2`, `mixed`。

常用估计参数：

- `--covariance-mode`: 当前主线通常使用 `exchangeable_varying_sigma`；也可用 `exchangeable_constant`。
- `--signal-bandwidth`: `A(t)` local-linear smoothing 的 bandwidth；省略并提供 grid 时可触发 CV。
- `--variance-bandwidth`: `sigma^2(t)` smoothing 的 bandwidth；constant covariance 模式下不使用。
- `--ridge`: 数值稳定项。论文默认公式对应 `ridge = 0`；非零值应说明为稳定化选择。
- `--a-eval-mode`: `A(t)` evaluation 模式，当前支持 `full` 和 `anchor_grid`；默认 `full`。
- `--a-eval-num-points`: anchor-grid 模式下请求使用的 `t0` 点数；默认 `500`。
- `--a-eval-grid`: anchor-grid 的取点方式；当前支持 `quantile` 和 `uniform`。
- `--a-interp`: anchor-grid 回填到全部 `t_i` 时使用的插值方式；当前支持 `linear`。
- `--prompt-accelerate-large-n`: 交互式 CLI 下，当 `n_subject > threshold` 时是否询问启用加速。
- `--large-n-threshold`: 触发交互询问的大样本阈值；当前默认 `2000`。
- `--n-jobs`: repetition 脚本的并行 worker 数。

关于 anchor-grid acceleration：

- 默认仍是 `full`，保证旧命令和历史实验结果口径不变。
- `anchor_grid` 只减少 stage 1 / stage 3 的外层 `t0` evaluation 点数。
- 实现方式是：先在 anchor 点估计 `A(t0)`，再插值回全部 `t_i`。
- 当 `a_eval_num_points >= n_subject` 时，实际行为会退化为 full-eval。
- 对 repetition 脚本，如果当前是交互式 CLI，且本次 run 的 `max(n_subject_values) > large_n_threshold`，并且用户没有显式传 `--a-eval-mode`，脚本会询问是否启用 `anchor_grid`。

## 当前结果目录

当前主线的批量实验结果已经分别整理到两个 repetition 目录下：

- `case1_2d_repetition/`
  - `raw_results.csv` / `summary_results.csv`
    - Case 1 根目录汇总结果
  - `hpc_runs/`
    - 已完成的 HPC 汇总结果
    - 当前主要包括 `a1a4_constant/`、`a1a4_varying_sigma/`、`a5a6_allsigma/`
  - `diagnostics/`
    - 诊断绘图与函数拟合可视化
  - `test/`
    - 本地 smoke、局部验证和临时试跑目录

- `case2_3d_repetition/`
  - `raw_results.csv` / `summary_results.csv`
    - `raw_results.csv` 含 11,520 个 fit：原主实验 5,760 条，加 A3/A5/A6 小带宽敏感性实验 5,760 条
    - `summary_results.csv` 含 192 个 setting：A1/A2/A4 保留原结果；A3/A5/A6 逐 setting 选择候选带宽中 `miae_final_mean` 更小者
  - `hpc_runs/`
    - 已完成的 HPC 汇总结果
    - 当前包括 `a1a4_constant/`、`a1a4_varsigma_a5a6_allsigma/` 与 `a3a5a6_small_h_sensitivity/`
    - 小带宽实验的诊断过程、HPC 配置、完整性审计和 oracle 选择限制记录在 `a3a5a6_small_h_sensitivity/README.md`
  - `test/`
    - 本地 anchor-grid 对比、A5 小带宽 pilot/grid、局部 smoke 和临时试跑目录

如果需要查看每个 repetition 目录内部的结果覆盖范围与汇总说明，优先阅读：

- `case1_2d_repetition/README.md`
- `case2_3d_repetition/README.md`

## 可选函数绘图

当前 4 个主线脚本都支持可选绘图，默认关闭，保证旧命令不受影响。

- `--plot-functions`: 打开绘图。
- `--plot-a-indices`: 指定要画的 `A[r,s](t)` 分量，使用 Python/NumPy 的 0-based index；例如 `0:0,3:0`。
- `--plot-max-a-panels`: 限制单张 `A` 图中的 panel 数；设为 `0` 时只画 `sigma^2(t)`。

绘图内容：

- `A_functions.png`: 画最终估计 `A_hat_final`，如果存在也叠加 stage-1 `A_hat_iid` 和模拟真值 `A_true`。
- `sigma2_function.png`: 画估计的 `sigma2_hat_t`，如果能从 DGP metadata 或 `Sigma_true` 推断，也叠加真值。
- 对 `exchangeable_constant`，`sigma2_hat_t` 是常数向量，因此图中是水平线；对 `exchangeable_varying_sigma`，图中是随 `t` 变化的曲线。

示例：画数学记号中的 `A[1,1](t)`、`A[4,1](t)` 和 `sigma^2(t)`，脚本中写成 0-based `0:0,3:0`。

```bash
python src/experiments/case1_2d_repetition.py \
  --n-subject-values 1000 \
  --coef-types base1 \
  --n-rep 1 \
  --R 4 \
  --S 25 \
  --covariance-mode exchangeable_varying_sigma \
  --signal-bandwidth 0.18 \
  --variance-bandwidth 0.18 \
  --plot-functions \
  --plot-a-indices 0:0,3:0 \
  --plot-max-a-panels 2
```

也可以直接运行已配置好的 Case 2 单次 fit 脚本：

```bash
bash src/experiments/run_case2_altbase_R6_S27_n5000_onefit_plot.sh
```

该脚本可通过环境变量覆盖默认值，例如：

```bash
COEF_TYPE=base4 SEED=456 PLOT_A_INDICES=0:0,3:0,5:26 PLOT_MAX_A_PANELS=3 \
  bash src/experiments/run_case2_altbase_R6_S27_n5000_onefit_plot.sh
```

如果想显式启用 anchor-grid acceleration，可以在 repetition 脚本中传入例如：

```bash
python src/experiments/case2_3d_repetition.py \
  --n-subject-values 5000 \
  --coef-types base5 \
  --rho-values 0.6 \
  --sigma2-functions mixed \
  --n-rep 1 \
  --R 6 \
  --S 27 \
  --covariance-mode exchangeable_varying_sigma \
  --signal-bandwidth 0.18 \
  --variance-bandwidth 0.18 \
  --a-eval-mode anchor_grid \
  --a-eval-num-points 500 \
  --a-eval-grid quantile \
  --a-interp linear
```

## 归档脚本

- `archive/archive_method_is_const_var/`

以上文件夹不再作为当前主线实验入口，原因是：

- 采用了 He Jiaxin 原文同一套 reduced-feature DGP 设定
- 没有引入估计量 σ̂(t)

对应的旧结果目录也一并保存在 `archive/archive_method_is_const_var/` 下，供回溯和历史对照使用。

## 输出文件夹

每个 `*_smoke.py` 或 `*_repetition.py` 脚本，都会把结果写到同目录下一个与脚本同名的输出文件夹中。例如：

- `case1_2d_smoke.py` -> `case1_2d_smoke/`
- `case1_2d_repetition.py` -> `case1_2d_repetition/`
- `case2_3d_smoke.py` -> `case2_3d_smoke/`
- `case2_3d_repetition.py` -> `case2_3d_repetition/`

对于归档脚本，如果再次运行，则输出也会写到 `archive/archive_method_is_const_var/` 目录体系下。

常见输出包括：

- `run_config.json`
- `progress.json`
- `results/raw_results.csv`
- `results/summary_results.csv`
- `plots/*_A_functions.png`，仅在传入 `--plot-functions` 时生成
- `plots/*_sigma2_function.png`，仅在传入 `--plot-functions` 时生成

当前 `raw_results.csv` / `summary_results.csv` 也会记录与加速模式相关的字段，例如：

- `a_eval_mode`
- `a_eval_selected_points`

Case 2 当前根 summary 还记录实际选定的 `best_signal_bandwidth_mean`。其中 A3/A5/A6 的值来自逐 `(coef_type,n_subject,rho,sigma2_function)` setting 的选择。

Repetition 脚本默认只保存汇总 CSV；需要保存每次重复的 dataset 或 estimate 时，分别传入 `--save-data` 或 `--save-estimates`。

## 本地验证目录

当前一个重要的本地算法验证目录是：

- `case2_3d_repetition/test/anchor_check/`

它用于对 `full` 与 `anchor_grid` 做大样本本地验证，当前包含：

- `run_config.json`
- `progress.json`
- `results/raw_results.csv`
- `results/summary_results.csv`
- `plots/`
- `README.md`

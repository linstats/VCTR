# Experiments

本目录保存当前正在使用的 paired-eye 实验入口脚本。

## 当前脚本

- `paired_case1_altbase_smoke.py`
  - 用于 alternative-base paired Case 1 设计的单次 smoke test。
  - 使用较新的 `base1` 到 `base6` 系数函数设定，并采用了估计量 `\hat{\sigma}(t)`。
- `paired_case1_altbase_repetition.py`
  - 用于 alternative-base paired Case 1 设计的重复模拟主脚本。
  - 这是当前 2D-equivalent altbase varying-sigma 实验的主要入口脚本。
- `paired_case2_altbase_smoke.py`
  - 用于 3D-equivalent alternative-base paired Case 2 设计的单次 smoke test。
  - 默认对应 reduced-feature `R = 3, S = 27`。
- `paired_case2_altbase_repetition.py`
  - 用于 3D-equivalent alternative-base paired Case 2 设计的重复模拟主脚本。
  - 核心单次拟合逻辑复用了 `paired_case2_altbase_smoke.py`。
- `paired_case2_altbase_repetition/audit_case2_hpc_parts.py`
  - 用于审计 HPC 上中断/超时的 Case 2 part 结果快照。
  - 根据 `run_config.json` 与 `results/raw_results.csv` 恢复理论任务集，并输出精确缺失清单与 backfill 提交脚本。
- `paired_case2_altbase_repetition/paired_case2_altbase_backfill.py`
  - 用于按缺失任务 manifest 精确补跑 Case 2 重复模拟。
  - 不再依赖 `n_rep` 自动展开任务，而是逐条消费 `part,n_subject,coef_type,rho_true,rep,seed` 清单。
- `paired_case2_altbase_repetition/merge_case2_hpc_parts.py`
  - 用于把本地旧的 `part1-8` 中断快照、HPC backfill 的 `part1-8` 完整结果，以及 HPC 上完整的 `part9-10` 合并成一套最终结果。
  - 合并键固定为 `(n_subject, coef_type, rho_true, rep, seed)`，并优先保留旧快照中已成功的 `part1-8` 记录。
- `run_case2_altbase_R6_S27_n5000_onefit_plot.sh`
  - Case 2 风格的单次 fit 便捷脚本，固定 `n = 5000, R = 6, S = 27` 并打开函数绘图。
  - 内部调用 `paired_case2_altbase_repetition.py --n-rep 1`，因此会生成独立 `run_name` 输出目录。

这 4 个脚本都采用当前 paired-eye altbase DGP，并默认使用 `varying_sigma` 工作流。

## 主要参数

常用数据生成参数：

- `--n-subject` 或 `--n-subject-values`: subject 数量。
- `--coef-type` 或 `--coef-types`: 系数函数类型，当前支持 `base1` 到 `base6`；脚本默认仍跑 `base1` 到 `base4`，A5/A6 需要显式传入 `base5 base6`。
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

## Case 1 当前进度

Case 1 altbase 现在分成两个结果方向：

- A1-A4: `constant/sin/sin2/mixed` 四种 DGP variance 已跑完；其中 `sin/sin2/mixed` 的结果位于 `paired_case1_altbase_repetition/hpc_varying_var_retry1/`，已用于 LaTeX 表格补充。
- A5-A6: `constant/sin/sin2/mixed` 四种 DGP variance 的 8-part HPC 补充实验已完成 merge，结果目录为 `paired_case1_altbase_repetition/hpc_base56_allsigma/`，并已用于 LaTeX 表格补充。

LaTeX 表格生成和排版材料位于 `docs/0607-prorgress/`；HPC 提交模板和 seed 分段细节见 `hpc/README.md`。

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
python src/experiments/paired_case1_altbase_repetition.py \
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
python src/experiments/paired_case2_altbase_repetition.py \
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

- `archive_const_var/paired_case1_smoke.py`
- `archive_const_var/paired_case1_repetition.py`
- `archive_const_var/paired_case2_smoke.py`
- `archive_const_var/paired_case2_repetition.py`

以上 4 个脚本已经归档，不再作为当前主线实验入口，原因是：

- 采用了 He Jiaxin 原文同一套 reduced-feature DGP 设定
- 没有引入估计量 σ̂(t)

对应的旧结果目录也一并保存在 `archive_const_var/` 下，供回溯和历史对照使用。

## 输出文件夹

每个 `*_smoke.py` 或 `*_repetition.py` 脚本，都会把结果写到同目录下一个与脚本同名的输出文件夹中。例如：

- `paired_case1_altbase_smoke.py` -> `paired_case1_altbase_smoke/`
- `paired_case1_altbase_repetition.py` -> `paired_case1_altbase_repetition/`
- `paired_case2_altbase_smoke.py` -> `paired_case2_altbase_smoke/`
- `paired_case2_altbase_repetition.py` -> `paired_case2_altbase_repetition/`

对于归档脚本，如果再次运行，则输出也会写到 `archive_const_var/` 目录体系下。

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

HPC 审计与补跑附加输出：

- `paired_case2_altbase_repetition/hpc_snapshot_*/audit/part*_missing.csv`
- `paired_case2_altbase_repetition/hpc_snapshot_*/audit/all_missing.csv`
- `paired_case2_altbase_repetition/hpc_snapshot_*/audit/submit_backfill.sh`
- `paired_case2_altbase_repetition/backfill_runs/<run_name>/results/raw_results.csv`

Case 1 HPC 结果目录目前常见为：

- `paired_case1_altbase_repetition/hpc_const_var/`: Case 1(a), A1-A4, constant DGP variance。
- `paired_case1_altbase_repetition/hpc_varying_var_retry1/`: Case 1(b)-1(d), A1-A4, `sin/sin2/mixed` DGP variance；这批结果已用于 LaTeX 表格补充。
- `paired_case1_altbase_repetition/hpc_base56_allsigma/`: A5-A6 的 `constant/sin/sin2/mixed` all-sigma HPC 补充实验结果；当前目录下已包含 merge 后的 `run_config.json`、`results/raw_results.csv`、`results/summary_results.csv` 和 `merge_meta.json`。
- `paired_case1_altbase_repetition/summarize_hpc_varying_var.py`: 汇总多 part HPC 结果并生成 `results/summary_results.csv` 的脚本。
- `paired_case1_altbase_repetition/summarize_hpc_base56_allsigma.py`: 汇总 `hpc_base56_allsigma/part1-8` 并生成总 `run_config.json`、`results/raw_results.csv`、`results/summary_results.csv` 和 `merge_meta.json` 的脚本。

Smoke 脚本还会保存：

- `data/seed_XXXX_dataset.npz`
- `estimates/seed_XXXX_estimate.npz`
- `results/summary.json`
- `results/metrics.json`

Repetition 脚本默认只保存汇总 CSV；需要保存每次重复的 dataset 或 estimate 时，分别传入 `--save-data` 或 `--save-estimates`。

## 本地验证目录

当前一个重要的本地算法验证目录是：

- `paired_case2_altbase_repetition/anchor_check/`

它用于对 `full` 与 `anchor_grid` 做大样本本地验证，当前包含：

- `run_config.json`
- `progress.json`
- `results/raw_results.csv`
- `results/summary_results.csv`
- `plots/`
- `README.md`

## 说明

- `src/experiments` 是当前 active paired-eye 实验目录。
- 当前主线实验入口是 altbase 版本：
  - `paired_case1_altbase_smoke.py`
  - `paired_case1_altbase_repetition.py`
  - `paired_case2_altbase_smoke.py`
  - `paired_case2_altbase_repetition.py`
- 其中可按解释层区分为：
  - Case 1 altbase: 2D-equivalent 设计，默认 `R = 4, S = 25`
  - Case 2 altbase: 3D-equivalent 设计，默认 `R = 3, S = 27`
- 这两条线在代码实现上都属于 reduced-feature paired DGP，不显式生成 raw tensor。
- `archive_const_var/` 保存旧的 constant / non-`\hat{\sigma}(t)` 实验脚本与结果。
- 旧的 iid reproduction 脚本仍然归档保存在 `archive/python_iid_vctr/src/experiments/` 下。

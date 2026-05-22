# `src/` 目录说明

`src/` 现在不再承担 `iid VCTR` 的 Python 复现任务，而是明确转向 paired-eye VCTR 的主开发线。

## 当前定位

- 主线目标：实现双眼 paired 的 VCTR。
- 默认参考：`VaryingCoefPLM.pdf`。
- 不再兼容：旧的 `iid VCTR` Python 主线。

旧的 `iid VCTR` Python 移植代码已归档到：

- `archive/python_iid_vctr/`

该归档用于：

- 保留历史移植痕迹；
- 方便后续与 MATLAB `iid` 基线核对；
- 避免和 paired 主线混在一起。

## 当前目录结构

```text
src/
  data/
  features/
  metrics/
  models/
  experiments/
  utils/
```

各模块职责：

- `data/`
  - paired-eye 数据容器。
  - 默认按 subject-level 与 eye-level 两层组织数据。
- `features/`
  - paired 主线的张量分块与投影特征构造接口。
  - 当前只保留骨架，等待 paired 仿真和真实数据流程定型。
- `metrics/`
  - paired 主线仍可复用的误差指标与结构识别指标。
- `models/`
  - paired-eye VCTR 的核心模型接口与实现位置。
  - 当前主模型是 `PairedEyeVCTRModel`。
  - 现阶段采用三阶段 paired 工作流：
    1. `A_dagger -> y_dagger -> beta_dagger`
    2. `(A_dagger, beta_dagger) -> Sigma_hat`
    3. `A_star -> y_star -> beta_star`
  - 第 3 阶段明确使用新的 `y_star = y - <X^*, A_star>` 再做 GLS，而不是继续沿用第 1 阶段的 `y_dagger`。
  - bandwidth 现在支持两种模式：
    - fixed：直接给一个 `bandwidth`
    - auto：显式给 `bandwidth_grid`，并在模型内部做 stage-1 subject-level `K`-fold CV
  - 当前默认行为不是自动选 `h`：
    - 若未显式提供 `bandwidth`，也未显式提供 `bandwidth_grid`，则默认固定 `h = 0.13`
    - 此时 `bandwidth_method` 记为 `default_fixed`
  - 当前自动选择只针对第 1 阶段 working iid 拟合；选出的 `best_bandwidth` 由三阶段共享。
  - 当前自动选择的默认方法是 `stage1_kfold_cv`，默认 `5` 折，按 subject 分折而不是按单只眼分折。
  - 当前默认候选 grid 为 `(0.08, 0.10, 0.13, 0.16, 0.20)`，但只有在显式提供 `bandwidth_grid` 时才会启用。
  - `InitialIidResult.meta` 和最终 `PairedVCTRResult.meta` 中都会记录：
    - `bandwidth_selected`
    - `bandwidth_method`
    - `bandwidth_grid`
    - `bandwidth_cv_scores`
    - `bandwidth_cv_metric`
    - 自动 CV 时还会记录 `bandwidth_cv_folds` 和 `bandwidth_cv_seed`
  - 默认 `ridge = 0`，以对应论文第 2.3 节的无正则化公式；若手动设置为非零，只应视为数值稳定选项。
- `experiments/`
  - 当前已有的 paired 实验入口主要是 `paired_case1_smoke.py` 和 `paired_case1_repetition.py`。
  - `paired_case1_smoke.py` 默认把输出写到同名目录 `src/experiments/paired_case1_smoke/` 下的 `data/`、`estimates/`、`results/`。
  - `paired_case1_repetition.py` 现在可用于批量重复模拟：
    - 支持 `n_subject_values`、`coef_types`、`rho_values`、`n_rep`
    - 支持固定 bandwidth
    - 也支持自动 bandwidth 选择，但只有显式提供 `bandwidth_grid` 时才会启用
    - 自动 bandwidth 选择当前采用 stage-1 subject-level `5`-fold CV
    - 支持多进程并行 `--n-jobs`
    - 支持 `--run-name`，便于手动命名一次完整实验
    - 每次运行会单独创建一个 `run_xxxxxxxx_xxxxxx/` 目录；也可手动指定 `--run-name`
    - run 根目录下会保存：
      - `run_config.json`：本次运行的配置
      - `progress.json`：当前进度与已完成任务数
    - `results/` 下会保存：
      - `raw_results.csv`
      - `summary_results.csv`
    - `raw_results.csv` 逐条保存每个 repetition 的：
      - `n_subject`
      - `coef_type`
      - `rho_true`
      - `bandwidth_method`
      - `best_bandwidth`
      - `miae_iid` / `rmise_iid`
      - `beta_mae_iid` / `beta_rmse_iid`
      - `miae_final` / `rmise_final`
      - `beta_mae_final` / `beta_rmse_final`
      - `sigma2_abs_error`
      - `rho_abs_error`
      - `Sigma_fro_error`
      - `elapsed_seconds`
      - `success` / `error_message`
    - `summary_results.csv` 按
      - `n_subject`
      - `coef_type`
      - `rho_true`
      - `bandwidth_method`
      分组汇总，并输出各指标的 `mean/std`
    - 若启用 `--save-data` 或 `--save-estimates`，对应文件会按
      - `n_subject`
      - `coef_type`
      - `rho`
      - `rep`
      - `seed`
      命名，避免不同配置互相覆盖
    - 结果按任务完成后增量写盘，不需要等整批结束后一次性落盘
    - summary 中包含 `best_bandwidth_mean` 和 `best_bandwidth_std`
- `utils/`
  - 从旧 `iid` 主线中提炼出来的通用数值工具。
  - 当前保留 kernel、spline、penalty 相关函数。

## 当前结论

`src/` 现在的任务不是“继续完善 iid 复现”，而是为 paired-eye VCTR 建立清晰、单一、可扩展的主开发线。

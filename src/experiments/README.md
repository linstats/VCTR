# Experiments

本目录保存当前正在使用的 paired-eye 实验入口脚本。

## 当前脚本

- `paired_case1_altbase_smoke.py`
  - 用于 alternative-base paired Case 1 设计的单次 smoke test。
  - 使用较新的 `base1` 到 `base4` 系数函数设定，并采用了估计量 `\hat{\sigma}(t)`。
- `paired_case1_altbase_repetition.py`
  - 用于 alternative-base paired Case 1 设计的重复模拟主脚本。
  - 这是当前 2D-equivalent altbase varying-sigma 实验的主要入口脚本。
- `paired_case2_altbase_smoke.py`
  - 用于 3D-equivalent alternative-base paired Case 2 设计的单次 smoke test。
  - 默认对应 reduced-feature `R = 3, S = 27`。
- `paired_case2_altbase_repetition.py`
  - 用于 3D-equivalent alternative-base paired Case 2 设计的重复模拟主脚本。
  - 核心单次拟合逻辑复用了 `paired_case2_altbase_smoke.py`。

这 4 个脚本都采用当前 paired-eye altbase DGP，并默认使用 `varying_sigma` 工作流。

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

# GRAPE model comparison configs

本目录保存最终模型对比实验配置。

当前文件中的 `v1_best_models.json` 和 `v2_patient_grouped.json` 都基于旧 stagewise bandwidth CV 固定候选。它们保留用于复现实验过程和解释 `Z` ablation；正式最终模型比较应等待 `evaluation/hyperpar_cv.py` 完成后，用 full three-stage held-out prediction CV 选出的 `(S, R, h, hbar)` 新建配置。

`v1_best_models.json` 使用 v3 bandwidth check 后固定的低复杂度 CFP/ROI 配置，对比：

- `Z-only` linear regression
- `X+Z` linear regression
- `X-only` VCTR
- `X+Z` iid VCTR
- `X+Z` paired VCTR

## v1_best_models

- CV 口径：当前 v1 run 是 pair_id-level 5-fold held-out CV，不是 true patient-level grouped CV。
- 固定超参数：`ridge = 0.0`，`a_eval_mode = anchor_grid`，`a_eval_num_points = 80`。
- CFP 配置：`S = 2x2x1`, `R = 1`, `signal_h = 1.0`, `variance_hbar = 1.2`。
- ROI 配置：`S = 2x2x1`, `R = 1`, `signal_h = 0.8`, `variance_hbar = 0.4`。

这些配置来自 `outputs/cv_bandwidth/v3_summary_best_by_image.csv`。该 JSON 只固定模型对比需要的最终候选，不重新搜索 `(S, R, h, hbar)`。

## v2_patient_grouped

`v2_patient_grouped.json` 使用同一组 CFP/ROI 固定超参数，但设置 `split_group = subject_id`。它会把同一个 `subject_id` 的所有 visits 放在同一 fold 中，避免 longitudinal rows 跨 train/holdout。这应作为下一版正式 prediction table 的运行入口。

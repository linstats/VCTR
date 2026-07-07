# GRAPE diagnostics

本目录预留给固定最终超参数后的诊断脚本和小型诊断输出整理。

计划中的诊断包括：

- held-out residual diagnostics
- paired-eye covariance diagnostics
- coefficient-function interpretation

运行产物仍应优先写入 `src/experiments/grape/runs/`；只有论文需要引用的小型表格或图像再整理到 `src/experiments/grape/outputs/`。

## Coefficient-function bootstrap

当前第一个实现是固定 full-CV 超参数的 ROI X-only patient-cluster bootstrap pilot：

```bash
# 可恢复的 B=100 拟合；先加 --replicates 5 可做 smoke test
python src/experiments/grape/diagnostics/bootstrap_coefficients.py \
  --config src/experiments/grape/configs/coefficient_bootstrap/roi_x_only_at_pilot_b100.json

python src/experiments/grape/diagnostics/aggregate_coefficient_bootstrap.py \
  --config src/experiments/grape/configs/coefficient_bootstrap/roi_x_only_at_pilot_b100.json

python src/experiments/grape/figures/plot_at_bootstrap.py \
  --config src/experiments/grape/configs/coefficient_bootstrap/roi_x_only_at_pilot_b100.json
```

抽样单位是 `manifest.csv` 中的真实 `subject_id`。每次抽中患者时保留其全部 visits 和每次 visit 的 OD/OS，并为重复抽中的 patient copy 重新生成唯一 `pair_id`。CP features、数据变换和 `(S,R,h,hbar)` 均固定。

完整 checkpoint、bootstrap draws、长表和 pilot 图保存在：

```text
src/experiments/grape/runs/coefficient_bootstrap/roi_x_only_at_pilot_b100/
```

`B=100` 只用于流程和数值稳定性检查，不应复制到 `outputs/` 或作为最终论文置信区间。最终 pointwise CI 至少应使用 `B=1000`；若需要 simultaneous confidence bands，应单独实现并使用更大的 `B`。

最终 pointwise percentile CI 配置为 `roi_x_only_at_final_b2000.json` 和 `cfp_x_only_at_final_b2000.json`，两者均使用 `B=2000`。当前研究流程不要求 simultaneous confidence bands。

## X+Z coefficient bootstrap

`cfp_xz_inherit_xonly_tuning_b2000.json` 和 `roi_xz_inherit_xonly_tuning_b2000.json` 使用各 image type 的 X-only-selected `(S,R,h,hbar)` 拟合 X+Z paired VCTR。每个 checkpoint 同时保存公共年龄网格上的 `A_hat` 和 60 维最终 `beta_hat`；聚合后生成 `coefficient_summary.csv` 与完整审计表 `beta_summary_all.csv`。

两套 image run 完成后，`compare_xz_beta_bootstrap.py` 合并 CFP/ROI beta summary，并生成只保留“任一 image 的 nominal 95% percentile CI 不含 0”的展示表。该筛选没有进行 multiple-testing adjustment，因此必须描述为 nominally significant。

正式配置默认使用 4 个 process workers，并将每个 worker 的 BLAS 线程限制为 1。建议 CFP 与 ROI 顺序运行，避免两个多进程任务同时争用 CPU。所有 run 均支持 checkpoint/resume。

## X + VF-PCA + gender bootstrap pilot

ROI PCA diagnostics 使用：

```bash
python src/experiments/grape/diagnostics/bootstrap_coefficients.py \
  --config src/experiments/grape/configs/coefficient_bootstrap/roi_x_vf_pca_gender_patient_pilot_b500.json

python src/experiments/grape/diagnostics/aggregate_coefficient_bootstrap.py \
  --config src/experiments/grape/configs/coefficient_bootstrap/roi_x_vf_pca_gender_patient_pilot_b500.json

python src/experiments/grape/figures/plot_at_bootstrap.py \
  --config src/experiments/grape/configs/coefficient_bootstrap/roi_x_vf_pca_gender_patient_pilot_b500.json
```

该 pilot 使用 `K=1`、真实患者 cluster bootstrap 和固定 full-sample PCA
basis。固定 basis 保证所有 replicate 的 `vf_pc_01` beta 含义一致，但区间不
包含 PCA basis 估计或 K 选择的不确定性。

## A(t) stability sensitivity

`compare_at_stability_candidates.py` 汇总三个预先定义的 ROI 候选：

- `S=6x2x1, h=0.85, hbar=0.30` reference
- `S=6x2x1, h=1.20, hbar=0.40` smoother bandwidth
- `S=3x2x1, h=0.60, hbar=0.25` reduced partition

每个候选使用固定 `K=1`、相同 patient-grouped prediction folds 和相同
`B=200` patient-bootstrap seed。汇总指标包括 prediction RMSE、中央年龄区间
CI width、curve roughness、bootstrap sign agreement 以及 ridge-stabilized
stage-3 local-system condition number。不同 `S` 的 roughness 位于不同 CP basis，
不能直接作为唯一选择标准。

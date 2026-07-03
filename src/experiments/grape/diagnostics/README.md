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

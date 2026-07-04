# Coefficient-bootstrap configs

本目录保存固定最终模型超参数后的 GRAPE coefficient-function bootstrap 配置。

当前配置：

```text
roi_x_only_at_pilot_b100.json
cfp_x_only_at_pilot_b100.json
roi_x_only_at_final_b2000.json
cfp_x_only_at_final_b2000.json
roi_x_only_at_h055_sensitivity_b2000.json
cfp_x_only_at_h055_sensitivity_b2000.json
cfp_xz_inherit_xonly_tuning_b2000.json
roi_xz_inherit_xonly_tuning_b2000.json
```

它使用 full three-stage CV 选出的 ROI X-only 配置 `S=6x2x1, R=1, h=0.85, hbar=0.30`，按真实患者 `subject_id` 做 `B=100` cluster bootstrap。CP reduced features、响应/年龄变换和超参数在 bootstrap 中固定。

该配置仅用于流程、耗时和数值稳定性 pilot，不是最终论文置信区间配置。完整运行结果保存在 `runs/coefficient_bootstrap/`，不导出到 `outputs/`。

两个 `*_final_b2000.json` 配置分别固定 full-CV 选出的 ROI/CFP X-only 超参数，生成 `B=2000` 的 95% pointwise percentile confidence intervals。当前流程不生成 simultaneous confidence bands。

两个 `*_h055_sensitivity_b2000.json` 配置使用 `h=0.55, hbar=0.30`。该 `h` 是对应固定 `(S,R)` 中 prediction RMSE 距最优值不超过 1% 的最小候选，用于提高年龄方向的分辨率；它们是 sensitivity analysis，不能替代 prediction-selected 主分析。使用 `bootstrap_coefficients.py --original-only` 可先检查全样本曲线而不启动 bootstrap。

两个 `*_xz_inherit_xonly_tuning_b2000.json` 配置拟合 X+Z paired VCTR，同时保存 `A(t)` 和 60 维 `beta` bootstrap draws。其 `(S,R,h,hbar)` 继承自各 image type 的 X-only prediction model，因此结果应描述为 inherited-X-only-tuning sensitivity，而不是 X+Z-optimal model。

# Coefficient-bootstrap configs

本目录保存固定最终模型超参数后的 GRAPE coefficient-function bootstrap 配置。

配置按研究用途分为三类。完整状态同时记录在 `../../experiment_registry.csv`。

## Main

```text
roi_x_only_at_final_b2000.json
cfp_x_only_at_final_b2000.json
```

这两个配置固定 full-CV 选出的 ROI/CFP X-only 超参数，生成 `B=2000` 的95% pointwise percentile confidence intervals。当前流程不生成 simultaneous confidence bands。

## Sensitivity

```text
roi_x_only_at_h055_sensitivity_b2000.json
cfp_x_only_at_h055_sensitivity_b2000.json
cfp_xz_inherit_xonly_tuning_b2000.json
roi_xz_inherit_xonly_tuning_b2000.json
```

`*_h055_sensitivity_b2000.json` 使用较小窗宽，不能替代prediction-selected主分析。`*_xz_inherit_xonly_tuning_b2000.json` 拟合60维X+Z模型，但继承X-only tuning，因此也是sensitivity analysis。

## Pilot

```text
roi_x_only_at_pilot_b100.json
cfp_x_only_at_pilot_b100.json
cfp_xz6_pilot_b100.json
roi_xz6_pilot_b100.json
cfp_xz6_h013_pilot_b500.json
roi_xz6_h013_pilot_b500.json
cfp_xz6_inherit_xonly_tuning_patient_b500.json
roi_xz6_inherit_xonly_tuning_patient_b500.json
roi_xz6_inherit_xonly_tuning_patient_b500_repeat2.json
roi_xz3_postselect60_inherit_xonly_tuning_row_pilot_b500.json
```

- `*_x_only_at_pilot_b100.json`：沿用各 image type 的 X-only 最优配置，按 `subject_id` 做 `B=100` cluster bootstrap，只验证流程、耗时和稳定性。
- `*_xz6_pilot_b100.json`：保留性别与 VF 位置 1、22、25、28、31 共 6 个 `Z`，按 `pair_id` 对 paired-visit rows 做 `B=100` bootstrap。
- `*_xz6_h013_pilot_b500.json`：沿用 6Z/paired-row 设计，改为 `h=0.13, hbar=0.25, B=500`，并保存 `A(t)`、beta 和 `sigma(t)` bootstrap 结果；`min_local_support_pairs=30` 用来标记高龄端低支持区域。
- `*_xz6_inherit_xonly_tuning_patient_b500.json`：继承各 image type 的 X-only 最优 `(S,R,h,hbar)`，保留同一组 6Z，按真实患者做 `B=500` cluster bootstrap，用于周报稳定性检查。
- `roi_xz6_inherit_xonly_tuning_patient_b500_repeat2.json`：ROI 的独立随机种子重复，用于检查 `B=500` percentile CI 的 Monte Carlo 重复性；模型和重采样设定不变。
- `roi_xz3_postselect60_inherit_xonly_tuning_row_pilot_b500.json`：从既有 ROI 60Z 分析中按名义显著性事后选取 VF00、VF04、VF60，继承 X-only tuning，按 paired row 做 `B=500` bootstrap；仅作 post-selection exploratory pilot。

所有 pilot 仅用于探索，不作为正式论文推断。完整运行结果保存在 `runs/coefficient_bootstrap/`。

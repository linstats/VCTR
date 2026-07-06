# GRAPE experiment index

Use `experiment_registry.csv` as the machine-readable source of truth. Raw computational artifacts remain under `runs/`; manuscript-facing or compact exports remain under `outputs/`.

## Main results

- `hyperpar_cv/x_only_grid_v1`: patient-grouped selection of the X-only tuning parameters.
- `final_ablation/v1_full_cv_selected`: current patient-grouped prediction comparison.
- `cfp_x_only_at_final_b2000` and `roi_x_only_at_final_b2000`: main X-only pointwise coefficient-function intervals.

## Sensitivity analyses

- `*_x_only_at_h055_sensitivity_b2000`: smaller-bandwidth X-only sensitivity.
- `*_xz_inherit_xonly_tuning_b2000`: 60-covariate X+Z sensitivity using X-only-selected tuning.
- `xz_inherit_xonly_tuning_b2000`: joint CFP/ROI beta summary for the preceding runs.

## Pilots

- `*_x_only_at_pilot_b100`: workflow validation only.
- `*_xz6_pilot_b100`: six-covariate paired-row bootstrap pilot.
- `*_xz6_h013_pilot_b500`: small-bandwidth six-covariate paired-row pilot. It is exploratory and unstable in older ages.
- `xz6_h013_pilot_b500`: joint beta summary for that pilot.

## Historical/supporting

- `model_comparison/v1_best_models`: historical pair-level split; do not use as the current prediction table.
- `model_comparison/v2_patient_grouped`: supporting patient-grouped comparison.
- `figures/paired_image_partitions_v1`: reproducible figure asset.

## Storage policy

- Completed run directories are immutable.
- `replicates/` checkpoints are retained for now; no checkpoint was deleted during this cleanup.
- A run called `pilot` must not be promoted to manuscript inference without a new config and run name.
- New experiments must be added to `experiment_registry.csv` when their config is created.

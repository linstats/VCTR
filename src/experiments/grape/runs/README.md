# GRAPE runs

This directory is the immutable machine-output store. Start from `../EXPERIMENT_INDEX.md` or `../experiment_registry.csv` rather than browsing run names without context.

| Workflow | Current interpretation |
| --- | --- |
| `hyperpar_cv/x_only_grid_v1` | main tuning result |
| `final_ablation/v1_full_cv_selected` | main prediction comparison |
| `coefficient_bootstrap/*_final_b2000` | main X-only intervals |
| `coefficient_bootstrap/*_sensitivity_b2000` | bandwidth sensitivity |
| `coefficient_bootstrap/*_xz_inherit_xonly_tuning_b2000` | 60-Z sensitivity |
| `coefficient_bootstrap/*pilot*` | exploratory only |
| `model_comparison/v1_best_models` | historical pair-level split |
| `model_comparison/v2_patient_grouped` | supporting patient-grouped result |

All 16 coefficient-bootstrap fit directories currently report complete runs with zero persisted failures. Per-replicate checkpoints are intentionally retained.

# GRAPE configs

Configuration files are grouped by workflow. Their research status is recorded in `../experiment_registry.csv`; filenames alone are not sufficient to determine whether a run is main, sensitivity, pilot, or historical.

- `hyperpar_cv/`: X-only tuning search.
- `final_ablation/`: patient-grouped final model comparisons.
- `coefficient_bootstrap/`: main, sensitivity, and pilot coefficient inference configs; see its README for the classification.
- `model_comparison/`: historical and supporting model-comparison configs.
- `figures/`: reproducible figure specifications.

Do not overwrite a config after its run has started. Create a new config and run name instead.

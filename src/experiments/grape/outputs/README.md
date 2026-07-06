# GRAPE outputs

This directory contains compact, human-facing exports. Full checkpoints and bootstrap draws belong in `../runs/`.

- `hyperpar_cv/`: compact tuning summaries.
- `final_ablation/`: current patient-grouped prediction tables.
- `model_comparison/`: historical/supporting comparison tables; consult the experiment registry before use.
- `coefficient_bootstrap/xz_inherit_xonly_tuning_b2000/`: 60-Z sensitivity export.
- `coefficient_bootstrap/xz6_h013_pilot_b500/`: exploratory small-bandwidth pilot export.
- `figures/`: reusable figure assets.

The canonical experiment classification is in `../experiment_registry.csv`.

# GRAPE outputs

This directory contains compact, human-facing exports. Full checkpoints and bootstrap draws belong in `../runs/`.

- `hyperpar_cv/`: compact tuning summaries.
- `final_ablation/`: current patient-grouped prediction tables.
- `model_comparison/`: historical/supporting comparison tables; consult the experiment registry before use.
- `coefficient_bootstrap/xz_inherit_xonly_tuning_b2000/`: 60-Z sensitivity export.
- `coefficient_bootstrap/xz6_h013_pilot_b500/`: exploratory small-bandwidth pilot export.
- `coefficient_bootstrap/xz6_inherit_xonly_tuning_patient_b500/`: weekly-report X+Z6 patient-bootstrap pilot export using X-only-selected tuning.
- `coefficient_bootstrap/roi_xz6_inherit_xonly_tuning_patient_b500_repeat2/`: independent-seed ROI repeatability check for that pilot.
- `coefficient_bootstrap/roi_xz3_postselect60_inherit_xonly_tuning_row_pilot_b500/`: ROI X+Z3 post-selection paired-row bootstrap pilot.
- `figures/`: reusable figure assets.

The canonical experiment classification is in `../experiment_registry.csv`.

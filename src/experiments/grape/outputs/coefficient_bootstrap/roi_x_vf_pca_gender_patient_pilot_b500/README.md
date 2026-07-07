# ROI X + VF-PCA + gender patient-bootstrap pilot

Exploratory `B=500` patient-cluster bootstrap diagnostics for the ROI
`X + VF-PCA + gender` paired VCTR model. All 500 replicates completed
successfully.

The model uses `K=1`, the modal nested-CV choice in
`v2_roi_five_model`. The full-sample patient-equal PCA basis is fixed across
bootstrap replicates, so the intervals are conditional on the selected PCA
representation and do not include PCA-basis selection uncertainty.

PC1 explains 62.43% of the full-sample bilateral-mean VF variance.

## Beta summary

| variable | beta_hat_iop | bootstrap_se_iop | 95% percentile CI | excludes zero |
| :-- | --: | --: | :-- | :-- |
| gender (`is_female`) | 0.0532 | 0.6664 | [-1.0514, 1.5099] | no |
| `vf_pc_01` | 0.1466 | 0.0586 | [0.0382, 0.2652] | yes |

The `vf_pc_01` coefficient is per one-unit PC score under the saved PCA
normalization. Its sign must be interpreted together with `pca_loadings.csv`.

## Files

- `roi_at_pointwise_ci.png` / `.pdf`: 12-block ROI coefficient functions with 95% pointwise intervals
- `beta_summary_all.csv`: gender and VF-PC1 coefficient summary
- `coefficient_summary.csv`: complete fixed-grid A(t) summary
- `pca_loadings.csv`: VF-PC1 loading definition
- `aggregation_metadata.json` and `run_metadata.json`: provenance

This is a diagnostic pilot, not final manuscript inference.

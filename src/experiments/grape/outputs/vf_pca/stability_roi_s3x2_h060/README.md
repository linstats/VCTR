# stability_roi_s3x2_h060

Nested patient-grouped CV evaluation of bilateral-mean VF PCA covariates.

PCA standardization, covariance estimation, and transformation are fit only on training patients.
Repeated visits receive inverse visit-count weights so every training patient has equal total PCA weight.
Sex is excluded from PCA and evaluated as a separate scalar covariate.

This experiment tests the incremental prediction value of bilateral OD/OS-mean VF covariates; it does not test eye-specific VF effects.

## Outputs

- `summary_metrics.csv` and `ablation_table.csv`: outer-CV prediction results
- `prediction_contrast_uncertainty.csv`: patient-cluster bootstrap CIs for fixed OOF prediction contrasts
- `fold_metrics.csv` and `predictions.csv`: outer-fold details
- `inner_cv_metrics.csv` and `selected_k_by_fold.csv`: nested K selection audit
- `explained_variance_by_fold.csv`: fold-local PCA variance summary
- `pca_loadings_by_fold.csv` and `loading_stability.csv`: fold-local loadings audit
- `run_metadata.json`: provenance and design limitations

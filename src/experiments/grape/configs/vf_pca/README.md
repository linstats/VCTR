# VF PCA configurations

`v1_fixed_x_tuning.json` evaluates whether a low-dimensional representation of
the 59 bilateral-mean VF covariates adds held-out prediction value beyond the
current X-only model.

The experiment uses patient-grouped nested CV. The outer folds estimate
prediction performance; inner folds select only the number of PCs. VF
standardization and PCA are fit on training patients only. Existing CFP/ROI
`S`, `R`, `h`, and `hbar` values remain fixed.

Sex is not included in PCA. Models with and without sex are retained so that
VF and sex contributions remain distinguishable.

Run a small workflow check with:

```bash
python src/experiments/grape/evaluation/vf_pca_ablation.py --smoke
```

Run the complete experiment with:

```bash
python src/experiments/grape/evaluation/vf_pca_ablation.py \
  --config src/experiments/grape/configs/vf_pca/v1_fixed_x_tuning.json
```

The current `Z` representation contains a bilateral OD/OS mean for each VF
location. Results therefore do not establish whether eye-specific VF values
have incremental prediction value.

## ROI five-model comparison

`v2_roi_five_model.json` runs only ROI and compares, on identical outer folds:

- outer-training response mean (`y_bar`)
- X-only paired VCTR
- X + VF-PCA paired VCTR
- X + VF-PCA + gender paired VCTR
- X + 60Z paired VCTR

The 60Z model contains gender plus all 59 bilateral-mean VF variables. Its VF
columns are standardized within each training fold using the same
patient-equal weighting convention as PCA.

```bash
python src/experiments/grape/evaluation/vf_pca_ablation.py \
  --config src/experiments/grape/configs/vf_pca/v2_roi_five_model.json
```

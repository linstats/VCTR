# Repository Guidelines

## Project Structure & Module Organization
- `VaryingCoefPLM.pdf` is the manuscript for Varying Coefficients Tensor Regression (VCTR).
- `code_and_data ` (note the trailing space in the directory name) contains all code and data utilities.
- `code_and_data /simulation` holds MATLAB scripts for the simulation study.
- `code_and_data /real_data` holds MATLAB and R scripts for the GRAPE fundus-image analysis.
- `code_and_data /toolbox` includes third‑party MATLAB dependencies (`tensor_toolbox`, `TensorReg`, `SparseReg`).

## Paper Snapshot: Methods (VCTR)
- Model (two eyes per subject): `y_{ij} = \langle \mathcal{X}_{ij}, \mathcal{A}(t_i)\rangle + \mathbf{z}_i^\top \boldsymbol{\beta} + \epsilon_{ij}`, with `t` as index (age), `\mathcal{X}_{ij}` tensor image, `\mathbf{z}_i` scalar covariates, and eye‑level correlation via a 2x2 covariance for `(\epsilon_{i1}, \epsilon_{i2})`.
- Dimension reduction:
  1. Partition each tensor into `S` sub‑tensors (spatial blocks).
  2. Apply CP decomposition within each block to obtain projection features and stack them as `\mathbf{X}_{ij}^{*} \in \mathbb{R}^{R\times S}` (rank `R` per block).
- Estimation (non‑sparse version):
  - Define reduced coefficient matrix `\mathbf{A}(t_i) := \{\langle \mathcal{U}_r^{(s)}, \mathcal{A}^{(s)}(t_i)\rangle\}_{r,s} \in \mathbb{R}^{R\times S}`, and use `y_{ij} \approx \langle \mathbf{X}_{ij}^{*}, \mathbf{A}(t_i)\rangle + \mathbf{z}_i^\top \boldsymbol{\beta} + \epsilon_{ij}`.
  - Local linear kernel smoothing for `\mathcal{A}(t)` / equivalently `\mathbf{A}(t)` with bandwidth `h`, then estimate `\boldsymbol{\beta}`.
  - Weighted estimator accounts for within‑subject (two‑eye) correlation.
- Sparse/structure‑identifying version:
  - B‑spline basis for `a_rs(t)`, decompose into constant vs varying parts.
  - Group penalties (SCAD/Lasso/MCP) on constant and varying components.
  - BIC selects tuning parameters; refined kernel smoothing after structure identification.

## Simulation Study (Paper Sections 4.1–4.2)
- Case I (2D): `p1=p2=80`, `S=16`, `R=10`. Case II (3D): `p1=p2=p3=40`, `S=64`, `R=5`.
  - Assess estimation consistency of `\mathcal{A}(t)` and `\boldsymbol{\beta}`.
  - Compare VCTR vs constant‑coefficient tensor models (prediction error via CV).
- Case III: spatial correlation across partitions (AR(1) structure).  
  - Evaluate variable selection accuracy.
- Case IV: high‑dimensional tensor (`p1=p2=p3=80`, `S=64`, `R=20`).
  - Compare Lasso/SCAD/MCP for structure identification under overfitting.

## Real Data Study (GRAPE Fundus Images)
- Data: `n=591` samples, images resized to `192x192x3`.
- OS images are horizontally flipped, then combined with OD images.
- Index variable `t`: age (normalized). Covariates `z`: gender + 59 visual‑field (VF) values.
- 10‑fold CV selects `S=3x3x1` and `R=2` for CFP and ROI images.
- Findings: gender effects are insignificant; VF locations near the optic disc show negative association; ROI models show more varying coefficients near optic disc regions.

## Code Map & How It Relates to the Paper
- Simulation scripts:
  - Estimation: `code_and_data /simulation/est_vctr_case1.m`, `est_vctr_case2.m`.
  - Sparse/structure ID + refinement: `est_vctr_case3_refine.m`, `est_vctr_case4_refine.m`.
  - Prediction/plots: `pred_summary_case1.m`, `pred_summary_case2.m`, `pred_vctr_case3.m`, `pred_vctr_case4.m`, `plot_case1&2.m`, `plot_case3.m`, `plot_case4.m`.
- Real data:
  - Preprocess: `code_and_data /real_data/data_process.R`.
  - Select `(S,R)`: `code_and_data /real_data/eye_select_RS.m` (10‑fold CV).
  - Penalized estimation and refinement: `eye_penalty_ref.m`, `eye_bootstrap*.m`.
  - Model comparison: `pred_eye_*.m`.
- Many scripts use absolute paths for image directories; update `image_path` and data file locations before running.

## Build, Test, and Development Commands
- No unified build system. Run scripts in MATLAB (R is used only for preprocessing).
- Examples:
  - Simulation Case I: run `est_vctr_case1` from `code_and_data /simulation`.
  - Sparse Case III refinement: run `est_vctr_case3_refine`.
  - Real‑data preprocessing: run `data_process.R` in `code_and_data /real_data`.

## Dependencies & Environment
- MATLAB R2023a (per `code_and_data /README.txt`).
- MATLAB toolboxes: `tensor_toolbox`, `TensorReg`, `SparseReg`.
- R required for `data_process.R`.
- GRAPE dataset must be downloaded separately:
```text
https://doi.org/10.6084/m9.figshare.c.6406319.v1
https://doi.org/10.1038/s41597-023-02424-4
```

## AoAS Review Summary & Resubmission Checklist
- Decision: Reject with resubmission; major concern is weak integration between methods (Sections 2–4) and the empirical study.
- Gaps cited by editor:
  - Simulation study not matched to real data.
  - Unclear if paired eyes from one patient are included; if so, how cluster dependence is handled.
  - Methodology and data analysis feel disconnected.
- Required resubmission items:
  - Cover letter must say this is a “Resubmission”.
  - Cite original submission number `AOAS2512-059` and the handling Area Editor.
  - Provide a detailed response to all reviewer comments in a supplemental file.
- Action items for revision:
  - Explicitly document paired‑eye handling and correlation modeling in both methods and data analysis.
  - Align simulation design with the GRAPE data structure (paired eyes, partition/R choices, covariates).
  - Integrate method steps with the empirical pipeline (data preprocessing → partition/decomposition → estimation → interpretation).

## Notes Before Further Work
- The current `code_and_data ` folder is the baseline implementation from the previous student.
- Suggested next steps:
  - Standardize configuration paths (avoid hard‑coded absolute paths).
  - Add a reproducible run script for each table/figure.
  - Add a brief “method vs. data” mapping section in the manuscript to address AoAS feedback.

## Notation Convention (Unified)
- Use tensor notation `\mathcal{}` consistently:
  - `\mathcal{X}_{ij}`, `\mathcal{A}(t_i)`, `\widetilde{\mathcal{X}}^{(s)}`, `\mathcal{U}_r^{(s)}`.
  - `\mathcal{H}_{ij}(t_i)`, `\mathcal{G}` for 3-way spline-expanded coefficient/design tensors in Sec 3.
- Use vector notation `\mathbf{}` and `\boldsymbol{}` consistently:
  - `\mathbf{x}`, `\mathbf{z}`, `\boldsymbol{\beta}`.
- Use matrix notation `\mathbf{}` consistently:
  - `\mathbf{X}_{ij}^{*} \in \mathbb{R}^{R\times S}`: stacked projection features after blockwise CP.
  - `\mathbf{A}(t_i) := \{\langle \mathcal{U}_r^{(s)}, \mathcal{A}^{(s)}(t_i)\rangle\}_{r,s} \in \mathbb{R}^{R\times S}`: stacked coefficient functions in the same basis.
- Reduced-model shorthand (preferred in implementation discussion):
  - `y_{ij} \approx \langle \mathbf{X}_{ij}^{*}, \mathbf{A}(t_i)\rangle + \mathbf{z}_i^\top \boldsymbol{\beta} + \epsilon_{ij}`.
- Penalized-model shorthand (Sec 3):
  - `y_{ij} \approx \langle \mathcal{H}_{ij}(t_i), \mathcal{G}\rangle + \mathbf{z}_i^\top \boldsymbol{\beta} + \epsilon_{ij}`.
- For manuscript writing, responses, and LaTeX/code generation in this repo, follow these symbols by default unless the user explicitly asks for a different notation in a local section.

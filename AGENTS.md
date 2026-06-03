# Repository Guidelines

## Project Structure & Module Organization
- `VaryingCoefLM.pdf` is the reference article for the current iid VCTR baseline.
- `VaryingCoefPLM.pdf` is the reference manuscript for the paired-eye VCTR target model.
- `code_and_data` contains the MATLAB baseline inherited from He Jiaxin. This codebase should currently be treated as iid VCTR code, not a verified paired-eye implementation.
- `archive/python_iid_vctr` preserves the previous Python port of the iid baseline. It is archived for reference and should not be treated as the active development line.
- `code_and_data/simulation` holds MATLAB scripts for the iid simulation study.
- `code_and_data/real_data` holds MATLAB and R scripts for the GRAPE fundus-image analysis inherited from the iid baseline workflow.
- `code_and_data/toolbox` includes third‑party MATLAB dependencies (`tensor_toolbox`, `TensorReg`, `SparseReg`).
- `src` is now the active paired-eye VCTR development line. It no longer serves as the main home for iid reproduction.

## Paper Snapshot: Two Levels
- Current code baseline (iid VCTR; reference: `VaryingCoefLM.pdf`):
  - Use an iid varying-coefficient tensor regression view, conceptually `y_i = \langle \mathcal{X}_i, \mathcal{A}(t_i)\rangle + \mathbf{z}_i^\top \boldsymbol{\beta} + \epsilon_i`.
  - No paired-eye dependence should be assumed in current `code_and_data` or `src` unless explicitly verified in a specific module.
- Project target (paired-eye VCTR; reference: `VaryingCoefPLM.pdf`):
  - Target model is `y_{ij} = \langle \mathcal{X}_{ij}, \mathcal{A}(t_i)\rangle + \mathbf{z}_i^\top \boldsymbol{\beta} + \epsilon_{ij}`.
  - The key methodological upgrade is explicit modeling of within-subject dependence between the two eyes.
- Dimension reduction:
  1. Partition each tensor into `S` sub‑tensors (spatial blocks).
  2. Apply CP decomposition within each block to obtain projection features and stack them as `\mathbf{X}_{ij}^{*} \in \mathbb{R}^{R\times S}` (rank `R` per block).
- Estimation ideas shared across the workflow:
  - Define reduced coefficient matrix `\mathbf{A}(t_i) := \{\langle \mathcal{U}_r^{(s)}, \mathcal{A}^{(s)}(t_i)\rangle\}_{r,s} \in \mathbb{R}^{R\times S}`, and use `y_{ij} \approx \langle \mathbf{X}_{ij}^{*}, \mathbf{A}(t_i)\rangle + \mathbf{z}_i^\top \boldsymbol{\beta} + \epsilon_{ij}`.
  - Local linear kernel smoothing for `\mathcal{A}(t)` / equivalently `\mathbf{A}(t)` with bandwidth `h`, then estimate `\boldsymbol{\beta}`.
  - Current paired Python implementation uses an explicit three-stage workflow:
    1. `\mathbf{A}^{\dagger}(t_i) -> y_{ij}^{\dagger} -> \boldsymbol{\beta}^{\dagger}`
    2. `(\mathbf{A}^{\dagger}, \boldsymbol{\beta}^{\dagger}) -> \hat{\Sigma}`
    3. `\mathbf{A}^{*}(t_i) -> y_{ij}^{*} -> \boldsymbol{\beta}^{*}`
  - In the active `src` code, stage 3 uses the closed loop `\mathbf{A}^{*} -> y^{*} -> \boldsymbol{\beta}^{*}` rather than reusing stage-1 `y^{\dagger}` for the final GLS step.
  - Default paired runs should use `ridge = 0` so that the implementation matches the unregularized formulas in Section 2.3; any nonzero ridge should be described as a numerical stabilization choice.
- Sparse/structure‑identifying version:
  - B‑spline basis for `a_rs(t)`, decompose into constant vs varying parts.
  - Group penalties (SCAD/Lasso/MCP) on constant and varying components.
  - BIC selects tuning parameters; refined kernel smoothing after structure identification.

## Current Status Summary
- `code_and_data` is the MATLAB iid baseline from He Jiaxin.
- `archive/python_iid_vctr` is the archived Python port of that iid baseline.
- `src` is now reserved for the paired-eye VCTR target model.
- The repository's real research goal is to upgrade iid VCTR into paired-eye VCTR, guided by `VaryingCoefPLM.pdf`.
- Do not describe `code_and_data` or `archive/python_iid_vctr` as already implementing paired-eye dependence unless the user explicitly narrows to a verified file and asks for that local detail.

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
  - MATLAB iid baseline: `code_and_data/simulation/est_vctr_case1.m`, `est_vctr_case2.m`.
  - Sparse/structure ID + refinement: `est_vctr_case3_refine.m`, `est_vctr_case4_refine.m`.
  - Prediction/plots: `pred_summary_case1.m`, `pred_summary_case2.m`, `pred_vctr_case3.m`, `pred_vctr_case4.m`, `plot_case1&2.m`, `plot_case3.m`, `plot_case4.m`.
- Real data:
  - Preprocess: `code_and_data/real_data/data_process.R`.
  - Select `(S,R)`: `code_and_data/real_data/eye_select_RS.m` (10‑fold CV).
  - Penalized estimation and refinement: `eye_penalty_ref.m`, `eye_bootstrap*.m`.
  - Model comparison: `pred_eye_*.m`.
- Python port:
  - Archived iid port: `archive/python_iid_vctr/src/dgps/case1_baseline.py` to `case4_baseline.py`.
  - Archived reproduction drivers: `archive/python_iid_vctr/src/experiments/reproduce_case1_matlab.py` to `reproduce_case4_matlab.py`.
- Active paired line: `src/data`, `src/features`, `src/models`, `src/experiments`, `src/utils`.
  - Current active paired experiment entry:
    - `src/dgps/paired_case1_altbase.py`
    - `src/experiments/paired_case1_altbase_smoke.py`
    - `src/experiments/paired_case1_altbase_repetition.py`
  - Current Case 1 altbase DGP supports:
    - coefficient functions `base1` through `base6`
    - DGP variance functions `constant`, `sin`, `sin2`, and `mixed`
  - Historical constant / non-`\hat{\sigma}(t)` paired scripts are archived under:
    - `src/experiments/archive_const_var/`
- Many scripts use absolute paths for image directories; update `image_path` and data file locations before running.

## Build, Test, and Development Commands
- No unified build system. Run scripts in MATLAB (R is used only for preprocessing).
- Examples:
  - Simulation Case I: run `est_vctr_case1` from `code_and_data/simulation`.
  - Sparse Case III refinement: run `est_vctr_case3_refine`.
  - Real‑data preprocessing: run `data_process.R` in `code_and_data/real_data`.

## Dependencies & Environment
- MATLAB R2023a (per `code_and_data/README.txt`).
- MATLAB toolboxes: `tensor_toolbox`, `TensorReg`, `SparseReg`.
- R required for `data_process.R`.
- GRAPE dataset must be downloaded separately:
```text
https://doi.org/10.6084/m9.figshare.c.6406319.v1
https://doi.org/10.1038/s41597-023-02424-4
```

## NUS HPC Notes
- Keep only durable HPC facts here. Detailed queues, PBS templates, submission commands, seed splits, and monitoring commands belong in `hpc/README.md`.
- Reliable direct login is `ssh e0829076@atlas9.nus.edu.sg`; `hopper.nus.edu.sg` timed out from the tested path.
- HPC repo copy is `/home/svu/e0829076/2026-tensor`.
- Working Python environment is `/home/svu/e0829076/conda-envs/vctr-py310`; batch scripts must explicitly load `miniconda/4.12` and activate this environment.
- Use `hpc/` as the source of truth for PBS templates. Current important templates include:
  - `hpc/paired_case1_altbase_varying_var_parallel.pbs` for Case 1 A1-A4 time-varying DGP variance.
  - `hpc/paired_case1_altbase_base56_allsigma_parallel.pbs` for Case 1 A5-A6 all-sigma backfill.
- Use `src/experiments/README.md` as the source of truth for experiment result directories and Case 1 A1-A4 / A5-A6 progress.

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
- The current `code_and_data` folder is the iid baseline implementation from the previous student He Jiaxin.
- The current `archive/python_iid_vctr` folder should be treated as the archived Python port of that iid baseline.
- The current `src` folder should be treated as the active paired-eye VCTR line.
- Suggested next steps:
  - Use `archive/python_iid_vctr` only for reference and backtracking.
  - Implement the paired-eye data flow and estimator logic inside `src`.
  - Separate clearly in writing and code comments what belongs to the iid baseline versus the paired-eye target model.
  - Standardize configuration paths (avoid hard‑coded absolute paths).
  - Add a reproducible run script for each table/figure.
  - Add a brief “method vs. data” mapping section in the manuscript to address AoAS feedback.

## Notation Convention (Unified)
- Use tensor notation `\mathcal{}` consistently:
  - For paired-eye target discussions: `\mathcal{X}_{ij}`, `\mathcal{A}(t_i)`, `\widetilde{\mathcal{X}}^{(s)}`, `\mathcal{U}_r^{(s)}`.
  - For iid baseline discussions, use the corresponding single-index form when that is more faithful to the current code path.
  - `\mathcal{H}_{ij}(t_i)`, `\mathcal{G}` for 3-way spline-expanded coefficient/design tensors in Sec 3.
- Use vector notation `\mathbf{}` and `\boldsymbol{}` consistently:
  - `\mathbf{x}`, `\mathbf{z}`, `\boldsymbol{\beta}`.
- Use matrix notation `\mathbf{}` consistently:
  - `\mathbf{X}_{ij}^{*} \in \mathbb{R}^{R\times S}`: stacked projection features after blockwise CP.
  - `\mathbf{A}(t_i) := \{\langle \mathcal{U}_r^{(s)}, \mathcal{A}^{(s)}(t_i)\rangle\}_{r,s} \in \mathbb{R}^{R\times S}`: stacked coefficient functions in the same basis.
- Reduced-model shorthand (preferred in implementation discussion):
  - For iid baseline discussion: `y_i \approx \langle \mathbf{X}_{i}^{*}, \mathbf{A}(t_i)\rangle + \mathbf{z}_i^\top \boldsymbol{\beta} + \epsilon_i`.
  - For paired-eye target discussion: `y_{ij} \approx \langle \mathbf{X}_{ij}^{*}, \mathbf{A}(t_i)\rangle + \mathbf{z}_i^\top \boldsymbol{\beta} + \epsilon_{ij}`.
- Penalized-model shorthand (Sec 3):
  - `y_{ij} \approx \langle \mathcal{H}_{ij}(t_i), \mathcal{G}\rangle + \mathbf{z}_i^\top \boldsymbol{\beta} + \epsilon_{ij}`.
- For manuscript writing, responses, and LaTeX/code generation in this repo, follow these symbols by default unless the user explicitly asks for a different notation in a local section.

# Figure Provenance

This directory stores figures used by `docs/0607-prorgress/0607-progress.tex`.

The table below records where each figure came from and the simulation setting behind it. Paths are relative to the repository root.

## Active Figures In `0607-progress.tex`

| Figure file | LaTeX use | Original source | Setting / meaning |
|---|---|---|---|
| `fig_case1_A_r01_s18.png` | Case 1 selected coefficient-function curve | `src/experiments/case1_2d_repetition/diagnostics/fit_plots/A__n2000_base4_sin2_rho0p9_seed148_rep1/plots_all_A/A_r01_s18.png` | Case 1; `coef_type=base4`; `A[1,18](t)` in 0-based indices; `n=2000`; `rho=0.9`; `sigma2_function=sin2`; `seed=148`; `rep=1`; `R=4`; `S=25`; `h_A=0.18`; `h_sigma=0.18`; `ridge=0`. |
| `fig_case1_A_r00_s13.png` | Case 1 selected coefficient-function curve | `src/experiments/case1_2d_repetition/diagnostics/fit_plots/sigma_miae__n2000_base6_sin_rho0p9_seed127_rep0/plots_all_A/A_r00_s13.png` | Case 1; `coef_type=base6`; `A[0,13](t)`; `n=2000`; `rho=0.9`; `sigma2_function=sin`; `seed=127`; `rep=0`; `R=4`; `S=25`; `h_A=0.18`; `h_sigma=0.18`; `ridge=0`. |
| `fig_case1_A_r02_s21.png` | Case 1 selected coefficient-function curve | `src/experiments/case1_2d_repetition/diagnostics/fit_plots/sigma_rmise__n2000_base2_sin2_rho0p9_seed145_rep2/plots_all_A/A_r02_s21.png` | Case 1; `coef_type=base2`; `A[2,21](t)`; `n=2000`; `rho=0.9`; `sigma2_function=sin2`; `seed=145`; `rep=2`; `R=4`; `S=25`; `h_A=0.18`; `h_sigma=0.18`; `ridge=0`. |
| `fig_case1_A_r00_s05.png` | Case 1 selected coefficient-function curve | `src/experiments/case1_2d_repetition/diagnostics/fit_plots/sigma_rmise__n2000_base2_sin2_rho0p9_seed145_rep2/plots_all_A/A_r00_s05.png` | Case 1; `coef_type=base2`; `A[0,5](t)`; `n=2000`; `rho=0.9`; `sigma2_function=sin2`; `seed=145`; `rep=2`; `R=4`; `S=25`; `h_A=0.18`; `h_sigma=0.18`; `ridge=0`. |
| `fig_case1_sigma2_seed0127.png` | Case 1 selected variance-function curve | `src/experiments/case1_2d_repetition/diagnostics/fit_plots/sigma_miae__n2000_base6_sin_rho0p9_seed127_rep0/plots/seed_0127_sigma2_function.png` | Case 1; `coef_type=base6`; `n=2000`; `rho=0.9`; `sigma2_function=sin`; true `sigma^2(t)=1+0.3 sin(2 pi t)`; `seed=127`; `rep=0`; `R=4`; `S=25`; `h_A=0.18`; `h_sigma=0.18`; `ridge=0`; `sigma2_miae=0.042591`; `sigma2_rmise=0.067075`. |
| `fig_case1_sigma2_seed0145.png` | Case 1 selected variance-function curve | `src/experiments/case1_2d_repetition/diagnostics/fit_plots/sigma_rmise__n2000_base2_sin2_rho0p9_seed145_rep2/plots/seed_0145_sigma2_function.png` | Case 1; `coef_type=base2`; `n=2000`; `rho=0.9`; `sigma2_function=sin2`; true `sigma^2(t)=0.5+0.5 sin^2(pi t)`; `seed=145`; `rep=2`; `R=4`; `S=25`; `h_A=0.18`; `h_sigma=0.18`; `ridge=0`; `sigma2_miae=0.044616`; `sigma2_rmise=0.050044`. |
| `fig_case2_A_r01_s14.png` | Case 2 selected coefficient-function curve | `src/experiments/case2_3d_repetition/diagnostics/fit_plots/A_base5__n5000_base5_sin2_rho0p9_h0p08_seed124_rep1/plots_all_A/A_r01_s14.png` | Case 2; `coef_type=base5`; `A[1,14](t)`; `n=5000`; `rho=0.9`; `sigma2_function=sin2`; `seed=124`; `rep=1`; `R=6`; `S=27`; `h_A=0.08`; `h_sigma=0.18`; `a_eval_mode=anchor_grid`; `a_eval_selected_points=500`; `ridge=1e-4`. |
| `fig_case2_A_r04_s07.png` | Case 2 selected coefficient-function curve | `src/experiments/case2_3d_repetition/diagnostics/fit_plots/A_base3__n5000_base3_sin2_rho0p9_h0p08_seed139_rep16/plots_all_A/A_r04_s07.png` | Case 2; `coef_type=base3`; `A[4,7](t)`; `n=5000`; `rho=0.9`; `sigma2_function=sin2`; `seed=139`; `rep=16`; `R=6`; `S=27`; `h_A=0.08`; `h_sigma=0.18`; `a_eval_mode=anchor_grid`; `a_eval_selected_points=500`; `ridge=1e-4`. |
| `fig_case2_sigma2_seed0136.png` | Case 2 selected variance-function curve | `src/experiments/case2_3d_repetition/diagnostics/fit_plots/sigma_mixed__n5000_base1_mixed_rho0p9_h0p18_seed136_rep13/plots/seed_0136_sigma2_function.png` | Case 2; `coef_type=base1`; `n=5000`; `rho=0.9`; `sigma2_function=mixed`; true `sigma^2(t)=1+0.25 cos(2 pi t)+0.1 sin(4 pi t)`; `seed=136`; `rep=13`; `R=6`; `S=27`; `h_A=0.18`; `h_sigma=0.18`; `a_eval_mode=anchor_grid`; `a_eval_selected_points=500`; `ridge=1e-4`; `sigma2_miae=0.061600`; `sigma2_rmise=0.087286`. |
| `fig_case2_sigma2_seed0136_1.png` | Case 2 selected variance-function curve | `src/experiments/case2_3d_repetition/diagnostics/fit_plots/A_base4__n5000_base4_sin2_rho0p9_h0p18_seed136_rep13/plots/seed_0136_sigma2_function.png` | Case 2; `coef_type=base4`; `n=5000`; `rho=0.9`; `sigma2_function=sin2`; true `sigma^2(t)=0.5+0.5 sin^2(pi t)`; `seed=136`; `rep=13`; `R=6`; `S=27`; `h_A=0.18`; `h_sigma=0.18`; `a_eval_mode=anchor_grid`; `a_eval_selected_points=500`; `ridge=1e-4`; `sigma2_miae=0.043587`; `sigma2_rmise=0.051152`. |

## Available But Currently Commented / Not Included

| Figure file | Status | Original source | Setting / meaning |
|---|---|---|---|
| `fig_case2_At_plot.png` | The corresponding `includegraphics` line is currently commented in `0607-progress.tex`. | Generated by `docs/0607-prorgress/0607-plots.ipynb`, cell 0. | Representative Case 2 coefficient functions `alpha_1` to `alpha_6` at `R=6`, `S=27`, `r=3`, `s=14`; saved by `plt.savefig("fig_case2_At_plot.png", dpi=600, bbox_inches="tight")`. |
| `fig1.tex` | Not included in the current `0607-progress.tex`. | Hand-written TikZ file in this directory. | Generic example figure with loss curves; not part of the active Case 1 / Case 2 simulation figures. |

## Notes

- `A[r,s](t)` indices in filenames are Python/NumPy 0-based indices.
- Case 2 A3/A5 figures use the small-bandwidth oracle sensitivity setting used by the current Case 2 summary.
- All diagnostic figures are copied snapshots. If a source diagnostic is regenerated, copy the updated PNG into this directory before recompiling the LaTeX document.

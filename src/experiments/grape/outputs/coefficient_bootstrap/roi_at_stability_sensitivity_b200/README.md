# ROI A(t) stability sensitivity

Exploratory comparison of three ROI `X + VF-PC1 + gender` paired VCTR
candidates. Prediction uses identical patient-grouped outer folds. Inference
uses `B=200` patient-cluster bootstrap draws with the same bootstrap seed.
Metrics below use ages 20–68.

| candidate | RMSE | delta RMSE | median CI width | q90 CI width | roughness | median sign agreement |
| :-- | --: | --: | --: | --: | --: | --: |
| `S6x2, h=0.85` reference | 3.9433 | 0.0000 | 1.2641 | 2.1534 | 17.1968 | 0.8150 |
| `S6x2, h=1.20` smoother | 3.9503 | 0.0070 (0.18%) | 1.2639 | 2.1425 | 1.0958 | 0.8325 |
| `S3x2, h=0.60` reduced | 4.0294 | 0.0861 (2.18%) | 0.6814 | 1.1921 | 45.8103 | 0.7350 |

## Interpretation

The `S6x2, h=1.20` candidate is the best smoother sensitivity: prediction and
CI width are effectively unchanged, while within the same coefficient basis
the numerical roughness falls by about 94%. Its condition number is larger
partly because the local-slope columns are scaled by `1/h`; that metric is not
directly comparable across bandwidth parameterizations.

The `S3x2` candidate halves CI width and greatly improves the local-system
condition number, but its prediction loss exceeds the prespecified 2% rule,
its sign agreement is lower, and its roughness is not directly comparable to
`S6x2` because the CP coefficient basis changes. It should not replace the
current model based on this pilot.

No candidate was selected by significance count or visual appearance. These
are B=200 sensitivity diagnostics, not final intervals.

## Files

- `comparison_summary.csv`: complete numerical comparison
- `comparison_metadata.json`: definitions and limitations
- `reference_s6x2_h085_main.png`: central-age reference figure
- `smoother_s6x2_h120_main.png`: central-age smoother figure
- `reduced_s3x2_h060_main.png`: central-age reduced-partition figure

Full checkpoints, tables, PDFs, and full-age figures remain in each run under
`src/experiments/grape/runs/coefficient_bootstrap/`.

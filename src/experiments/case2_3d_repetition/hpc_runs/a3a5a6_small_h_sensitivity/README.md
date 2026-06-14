# Case 2 A3/A5/A6 small-bandwidth sensitivity experiment

This directory records the completed small-`h_A` sensitivity run. Its raw
results have been appended to the Case 2 root raw table. For A3/A5/A6, the
root summary keeps the lower-`miae_final_mean` candidate separately within
each `(coef_type, n_subject, rho, sigma2_function)` setting.

Expected generated layout:

```text
manifests/part1.csv ... part8.csv
hpc_raw_parts/part1/ ... part8/
results/raw_results.csv
results/summary_results.csv
results/bandwidth_comparison.csv
provenance.csv
merge_meta.json
run_config.json
```

The expected final coverage is 5760 successful fits, 192 summary rows, and 30
repetitions per summary row.

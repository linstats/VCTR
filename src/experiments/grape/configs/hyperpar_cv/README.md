# Hyperparameter CV configs

本目录保存 full three-stage held-out prediction CV 的配置。

当前配置：

```text
x_only_grid_v1.json
```

用途：对所有当前 CFP/ROI feature packages 的 `(S, R, h, hbar)` 做 X-only VCTR 调参。主排序指标是 `subject_id` grouped 5-fold CV 下的 `rmse_std`，辅助输出 `rmse_iop`, `mape_std_pct`, `mape_iop_pct`。

固定设定：`z_mode=none`, `a_eval_mode=full`, `ridge=1e-6`。其中 ridge 只用于数值稳定化。

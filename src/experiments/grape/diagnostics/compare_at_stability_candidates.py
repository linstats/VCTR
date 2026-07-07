"""Compare prediction and bootstrap stability for predefined ROI A(t) candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.grape.diagnostics.bootstrap_coefficients import (  # noqa: E402
    GRAPE_ROOT,
    RUN_ROOT,
    load_config,
    resolve_path,
)


DEFAULT_CONFIGS = (
    GRAPE_ROOT / "configs" / "coefficient_bootstrap" / "roi_x_vf_pca_gender_s6x2_h085_stability_pilot_b200.json",
    GRAPE_ROOT / "configs" / "coefficient_bootstrap" / "roi_x_vf_pca_gender_s6x2_h120_stability_pilot_b200.json",
    GRAPE_ROOT / "configs" / "coefficient_bootstrap" / "roi_x_vf_pca_gender_s3x2_h060_stability_pilot_b200.json",
)
DEFAULT_OUTPUT = RUN_ROOT / "roi_at_stability_sensitivity_b200"
PREDICTION_RUN_ROOT = GRAPE_ROOT / "runs" / "vf_pca"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", type=Path, default=list(DEFAULT_CONFIGS))
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--prediction-run-root", type=Path, default=PREDICTION_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--age-min", type=float, default=20.0)
    parser.add_argument("--age-max", type=float, default=68.0)
    return parser.parse_args()


def curve_roughness(summary: pd.DataFrame) -> float:
    values: list[float] = []
    for (_, _), group in summary.groupby(["rank", "block"], sort=True):
        ordered = group.sort_values("t")
        t = ordered["t"].to_numpy(dtype=float)
        curve = ordered["A_hat"].to_numpy(dtype=float)
        first = np.gradient(curve, t)
        second = np.gradient(first, t)
        values.append(float(np.trapezoid(np.square(second), t)))
    return float(np.mean(values))


def candidate_row(
    config: dict[str, Any],
    *,
    run_root: Path,
    prediction_run_root: Path,
    age_min: float,
    age_max: float,
) -> dict[str, Any]:
    run_dir = run_root / str(config["name"])
    coefficient = pd.read_csv(run_dir / "coefficient_summary.csv")
    central = coefficient[coefficient["age"].between(age_min, age_max)].copy()
    if central.empty:
        raise ValueError(f"No coefficient rows in age range for {config['name']}.")
    widths = central["ci_upper_pointwise"] - central["ci_lower_pointwise"]
    excludes_zero = (central["ci_lower_pointwise"] > 0) | (central["ci_upper_pointwise"] < 0)

    prediction_path = prediction_run_root / str(config["prediction_run"]) / "summary_metrics.csv"
    prediction = pd.read_csv(prediction_path)
    prediction = prediction[prediction["model"] == "x_sex_vf_pca_paired_vctr"]
    if len(prediction) != 1:
        raise ValueError(f"Expected one prediction row in {prediction_path}; found {len(prediction)}.")
    prediction_row = prediction.iloc[0]

    with np.load(run_dir / "original_fit.npz") as original:
        age_grid = np.asarray(original["age_grid"], dtype=float)
        central_grid = (age_grid >= age_min) & (age_grid <= age_max)
        condition = np.asarray(original["local_design_condition_numbers"], dtype=float)[central_grid]
    with np.load(run_dir / "bootstrap_draws.npz") as draws:
        A_bootstrap = np.asarray(draws["A_bootstrap"], dtype=float)[:, central_grid]
        A_original = np.asarray(draws["A_original"], dtype=float)[central_grid]
    original_sign = np.sign(A_original)[None, ...]
    sign_agreement = np.mean(np.sign(A_bootstrap) == original_sign, axis=0).reshape(-1)

    return {
        "candidate": str(config["candidate_label"]),
        "bootstrap_run": str(config["name"]),
        "prediction_run": str(config["prediction_run"]),
        "S": str(config["S"]),
        "R": int(config["R"]),
        "signal_h": float(config["signal_h"]),
        "variance_hbar": float(config["variance_hbar"]),
        "n_coefficient_functions": int(np.prod([int(value) for value in str(config["S"]).split("x")])),
        "rmse_iop": float(prediction_row["rmse_iop"]),
        "mae_iop": float(prediction_row["mae_iop"]),
        "median_ci_width": float(widths.median()),
        "q90_ci_width": float(widths.quantile(0.9)),
        "max_ci_width": float(widths.max()),
        "curve_roughness_mean": curve_roughness(central),
        "median_sign_agreement": float(np.median(sign_agreement)),
        "q10_sign_agreement": float(np.quantile(sign_agreement, 0.1)),
        "fraction_sign_agreement_ge_0_90": float(np.mean(sign_agreement >= 0.9)),
        "fraction_pointwise_ci_excludes_zero": float(excludes_zero.mean()),
        "median_local_design_condition": float(np.median(condition)),
        "q90_local_design_condition": float(np.quantile(condition, 0.9)),
        "max_local_design_condition": float(np.max(condition)),
        "n_bootstrap": int(central["n_success"].iloc[0]),
        "age_min": float(age_min),
        "age_max": float(age_max),
    }


def main() -> None:
    args = parse_args()
    if args.age_min >= args.age_max:
        raise ValueError("age-min must be smaller than age-max.")
    run_root = resolve_path(args.run_root)
    prediction_run_root = resolve_path(args.prediction_run_root)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = [load_config(resolve_path(path)) for path in args.configs]
    summary = pd.DataFrame(
        [
            candidate_row(
                config,
                run_root=run_root,
                prediction_run_root=prediction_run_root,
                age_min=float(args.age_min),
                age_max=float(args.age_max),
            )
            for config in configs
        ]
    )
    reference_rmse = float(summary.loc[summary["candidate"].str.contains("reference"), "rmse_iop"].iloc[0])
    summary["delta_rmse_iop_vs_reference"] = summary["rmse_iop"] - reference_rmse
    summary["pct_delta_rmse_iop_vs_reference"] = 100.0 * summary["delta_rmse_iop_vs_reference"] / reference_rmse
    summary["within_2pct_reference_rmse"] = summary["pct_delta_rmse_iop_vs_reference"] <= 2.0
    summary.to_csv(output_dir / "comparison_summary.csv", index=False)
    metadata = {
        "configs": [str(resolve_path(path)) for path in args.configs],
        "age_range": [float(args.age_min), float(args.age_max)],
        "selection_warning": "Exploratory diagnostics only; do not select by significance count or visual appearance.",
        "condition_number": "ridge-stabilized stage-3 GLS normal equation",
        "sign_agreement": "fraction of bootstrap draws matching the full-sample coefficient sign at each grid point",
    }
    (output_dir / "comparison_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

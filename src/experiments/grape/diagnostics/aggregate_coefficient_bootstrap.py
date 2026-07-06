"""Aggregate GRAPE coefficient-bootstrap checkpoints into estimates and intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.grape.diagnostics.bootstrap_coefficients import (  # noqa: E402
    DEFAULT_CONFIG,
    GRAPE_ROOT,
    RUN_ROOT,
    load_config,
    rel_to_repo,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--min-success", type=int, default=2)
    return parser.parse_args()


def block_coordinates(block_index: int, s_value: str) -> tuple[int, int]:
    dims = tuple(int(value) for value in s_value.split("x"))
    if len(dims) != 3 or dims[2] != 1:
        raise ValueError("Current GRAPE image plotting expects S1xS2x1.")
    s1 = dims[0]
    return block_index % s1 + 1, block_index // s1 + 1


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    run_root = resolve_path(args.run_root)
    run_name = args.run_name or str(config["name"])
    run_dir = run_root / run_name
    original_path = run_dir / "original_fit.npz"
    if not original_path.exists():
        raise FileNotFoundError(f"Missing original fit: {original_path}")

    replicate_paths = sorted((run_dir / "replicates").glob("bootstrap_*.npz"))
    if len(replicate_paths) < int(args.min_success):
        raise ValueError(
            f"Only {len(replicate_paths)} successful replicates; require at least {int(args.min_success)}."
        )

    with np.load(original_path) as original:
        t_grid = np.asarray(original["t_grid"], dtype=float)
        age_grid = np.asarray(original["age_grid"], dtype=float)
        A_original = np.asarray(original["A_hat"], dtype=float)
        beta_original = (
            np.asarray(original["beta_hat"], dtype=float)
            if "beta_hat" in original.files
            else np.empty(0, dtype=float)
        )
        z_names = (
            np.asarray(original["Z_names"], dtype=str)
            if "Z_names" in original.files
            else np.empty(0, dtype=str)
        )
        y_sd = float(original["y_sd"]) if "y_sd" in original.files else float("nan")
        sigma2_original = (
            np.asarray(original["sigma2_hat_grid"], dtype=float)
            if "sigma2_hat_grid" in original.files
            else np.empty(0, dtype=float)
        )
        local_support_pairs = (
            np.asarray(original["local_support_pairs"], dtype=int)
            if "local_support_pairs" in original.files
            else np.empty(0, dtype=int)
        )
        variance_support_pairs = (
            np.asarray(original["variance_support_pairs"], dtype=int)
            if "variance_support_pairs" in original.files
            else local_support_pairs.copy()
        )

    draws: list[np.ndarray] = []
    beta_draws: list[np.ndarray] = []
    sigma2_draws: list[np.ndarray] = []
    replicate_ids: list[int] = []
    for path in replicate_paths:
        with np.load(path) as data:
            A_draw = np.asarray(data["A_hat"], dtype=float)
            if A_draw.shape != A_original.shape:
                raise ValueError(f"Shape mismatch in {path}: {A_draw.shape} != {A_original.shape}")
            if not np.all(np.isfinite(A_draw)):
                raise ValueError(f"Non-finite coefficient draw in {path}")
            draws.append(A_draw)
            if beta_original.size:
                if "beta_hat" not in data.files:
                    raise ValueError(f"Missing beta_hat in X+Z checkpoint {path}")
                beta_draw = np.asarray(data["beta_hat"], dtype=float)
                if beta_draw.shape != beta_original.shape:
                    raise ValueError(f"Beta shape mismatch in {path}: {beta_draw.shape} != {beta_original.shape}")
                if not np.all(np.isfinite(beta_draw)):
                    raise ValueError(f"Non-finite beta draw in {path}")
                beta_draws.append(beta_draw)
            if sigma2_original.size:
                if "sigma2_hat_grid" not in data.files:
                    raise ValueError(f"Missing sigma2_hat_grid in checkpoint {path}")
                sigma2_draw = np.asarray(data["sigma2_hat_grid"], dtype=float)
                if sigma2_draw.shape != sigma2_original.shape:
                    raise ValueError(
                        f"Sigma2 shape mismatch in {path}: {sigma2_draw.shape} != {sigma2_original.shape}"
                    )
                if not np.all(np.isfinite(sigma2_draw)) or np.any(sigma2_draw <= 0.0):
                    raise ValueError(f"Invalid sigma2 curve in {path}")
                sigma2_draws.append(sigma2_draw)
            replicate_ids.append(int(data["replicate"]))

    A_bootstrap = np.stack(draws, axis=0)
    confidence_level = float(config.get("confidence_level", 0.95))
    alpha = 1.0 - confidence_level
    lower = np.quantile(A_bootstrap, alpha / 2.0, axis=0)
    upper = np.quantile(A_bootstrap, 1.0 - alpha / 2.0, axis=0)
    bootstrap_mean = np.mean(A_bootstrap, axis=0)
    bootstrap_se = np.std(A_bootstrap, axis=0, ddof=1)
    bootstrap_bias = bootstrap_mean - A_original

    rows: list[dict[str, object]] = []
    n_rank = A_original.shape[1]
    n_blocks = A_original.shape[2]
    for grid_idx, (t_value, age_value) in enumerate(zip(t_grid, age_grid, strict=True)):
        for rank_idx in range(n_rank):
            for block_idx in range(n_blocks):
                block_row, block_col = block_coordinates(block_idx, str(config["S"]))
                rows.append(
                    {
                        "image_type": str(config["image_type"]),
                        "rank": rank_idx + 1,
                        "block": block_idx + 1,
                        "block_row": block_row,
                        "block_col": block_col,
                        "t": float(t_value),
                        "age": float(age_value),
                        "A_hat": float(A_original[grid_idx, rank_idx, block_idx]),
                        "bootstrap_mean": float(bootstrap_mean[grid_idx, rank_idx, block_idx]),
                        "bootstrap_bias": float(bootstrap_bias[grid_idx, rank_idx, block_idx]),
                        "bootstrap_se": float(bootstrap_se[grid_idx, rank_idx, block_idx]),
                        "ci_lower_pointwise": float(lower[grid_idx, rank_idx, block_idx]),
                        "ci_upper_pointwise": float(upper[grid_idx, rank_idx, block_idx]),
                        "confidence_level": confidence_level,
                        "ci_method": str(config.get("ci_method", "percentile")),
                        "n_success": int(A_bootstrap.shape[0]),
                    }
                )

    beta_bootstrap = (
        np.stack(beta_draws, axis=0)
        if beta_original.size
        else np.empty((A_bootstrap.shape[0], 0), dtype=float)
    )
    sigma2_bootstrap = (
        np.stack(sigma2_draws, axis=0)
        if sigma2_original.size
        else np.empty((A_bootstrap.shape[0], 0), dtype=float)
    )
    sigma_original = np.sqrt(sigma2_original)
    sigma_bootstrap = np.sqrt(sigma2_bootstrap)
    np.savez_compressed(
        run_dir / "bootstrap_draws.npz",
        A_bootstrap=A_bootstrap,
        A_original=A_original,
        beta_bootstrap=beta_bootstrap,
        beta_original=beta_original,
        sigma2_bootstrap=sigma2_bootstrap,
        sigma2_original=sigma2_original,
        sigma_bootstrap=sigma_bootstrap,
        sigma_original=sigma_original,
        local_support_pairs=local_support_pairs,
        variance_support_pairs=variance_support_pairs,
        Z_names=z_names,
        y_sd=np.array(y_sd),
        t_grid=t_grid,
        age_grid=age_grid,
        replicate_ids=np.asarray(replicate_ids, dtype=int),
    )
    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "coefficient_summary.csv", index=False)
    if beta_original.size:
        if z_names.shape[0] != beta_original.shape[0]:
            raise ValueError("Z_names and beta_hat must have the same length.")
        beta_mean = np.mean(beta_bootstrap, axis=0)
        beta_bias = beta_mean - beta_original
        beta_se = np.std(beta_bootstrap, axis=0, ddof=1)
        beta_lower = np.quantile(beta_bootstrap, alpha / 2.0, axis=0)
        beta_upper = np.quantile(beta_bootstrap, 1.0 - alpha / 2.0, axis=0)
        beta_rows: list[dict[str, object]] = []
        for idx, variable in enumerate(z_names):
            excludes_zero = bool(beta_lower[idx] > 0.0 or beta_upper[idx] < 0.0)
            beta_rows.append(
                {
                    "image_type": str(config["image_type"]),
                    "variable": str(variable),
                    "variable_type": "sex" if str(variable) == "is_female" else "vf",
                    "beta_hat_std": float(beta_original[idx]),
                    "bootstrap_mean_std": float(beta_mean[idx]),
                    "bootstrap_bias_std": float(beta_bias[idx]),
                    "bootstrap_se_std": float(beta_se[idx]),
                    "ci_lower_std": float(beta_lower[idx]),
                    "ci_upper_std": float(beta_upper[idx]),
                    "beta_hat_iop": float(beta_original[idx] * y_sd),
                    "bootstrap_mean_iop": float(beta_mean[idx] * y_sd),
                    "bootstrap_bias_iop": float(beta_bias[idx] * y_sd),
                    "bootstrap_se_iop": float(beta_se[idx] * y_sd),
                    "ci_lower_iop": float(beta_lower[idx] * y_sd),
                    "ci_upper_iop": float(beta_upper[idx] * y_sd),
                    "ci_excludes_zero": excludes_zero,
                    "significance_scope": "nominal_95pct_percentile_ci",
                    "confidence_level": confidence_level,
                    "ci_method": str(config.get("ci_method", "percentile")),
                    "n_success": int(beta_bootstrap.shape[0]),
                }
            )
        pd.DataFrame(beta_rows).to_csv(run_dir / "beta_summary_all.csv", index=False)
    if sigma2_original.size:
        if sigma2_original.shape != t_grid.shape:
            raise ValueError("sigma2_hat_grid and t_grid must have the same shape.")
        if local_support_pairs.shape != t_grid.shape:
            raise ValueError("local_support_pairs and t_grid must have the same shape.")
        if variance_support_pairs.shape != t_grid.shape:
            raise ValueError("variance_support_pairs and t_grid must have the same shape.")
        min_support = int(config.get("min_variance_support_pairs", config.get("min_local_support_pairs", 0)))
        sigma2_mean = np.mean(sigma2_bootstrap, axis=0)
        sigma2_se = np.std(sigma2_bootstrap, axis=0, ddof=1)
        sigma2_lower = np.quantile(sigma2_bootstrap, alpha / 2.0, axis=0)
        sigma2_upper = np.quantile(sigma2_bootstrap, 1.0 - alpha / 2.0, axis=0)
        sigma_mean = np.mean(sigma_bootstrap, axis=0)
        sigma_se = np.std(sigma_bootstrap, axis=0, ddof=1)
        sigma_lower = np.quantile(sigma_bootstrap, alpha / 2.0, axis=0)
        sigma_upper = np.quantile(sigma_bootstrap, 1.0 - alpha / 2.0, axis=0)
        variance_rows: list[dict[str, object]] = []
        for idx, (t_value, age_value) in enumerate(zip(t_grid, age_grid, strict=True)):
            variance_rows.append(
                {
                    "image_type": str(config["image_type"]),
                    "t": float(t_value),
                    "age": float(age_value),
                    "sigma2_hat_std": float(sigma2_original[idx]),
                    "sigma2_bootstrap_mean_std": float(sigma2_mean[idx]),
                    "sigma2_bootstrap_se_std": float(sigma2_se[idx]),
                    "sigma2_ci_lower_pointwise_std": float(sigma2_lower[idx]),
                    "sigma2_ci_upper_pointwise_std": float(sigma2_upper[idx]),
                    "sigma_hat_std": float(sigma_original[idx]),
                    "sigma_bootstrap_mean_std": float(sigma_mean[idx]),
                    "sigma_bootstrap_se_std": float(sigma_se[idx]),
                    "sigma_ci_lower_pointwise_std": float(sigma_lower[idx]),
                    "sigma_ci_upper_pointwise_std": float(sigma_upper[idx]),
                    "sigma_hat_iop": float(sigma_original[idx] * y_sd),
                    "sigma_bootstrap_mean_iop": float(sigma_mean[idx] * y_sd),
                    "sigma_bootstrap_se_iop": float(sigma_se[idx] * y_sd),
                    "sigma_ci_lower_pointwise_iop": float(sigma_lower[idx] * y_sd),
                    "sigma_ci_upper_pointwise_iop": float(sigma_upper[idx] * y_sd),
                    "local_support_pairs": int(local_support_pairs[idx]),
                    "variance_support_pairs": int(variance_support_pairs[idx]),
                    "min_variance_support_pairs": min_support,
                    "support_ok": bool(variance_support_pairs[idx] >= min_support),
                    "confidence_level": confidence_level,
                    "ci_method": str(config.get("ci_method", "percentile")),
                    "n_success": int(sigma2_bootstrap.shape[0]),
                }
            )
        pd.DataFrame(variance_rows).to_csv(run_dir / "variance_summary.csv", index=False)

        support_by_t = {
            float(t_value): (int(local_support_pairs[idx]), bool(local_support_pairs[idx] >= min_support))
            for idx, t_value in enumerate(t_grid)
        }
        summary["local_support_pairs"] = summary["t"].map(lambda value: support_by_t[float(value)][0])
        summary["support_ok"] = summary["t"].map(lambda value: support_by_t[float(value)][1])
        summary.to_csv(run_dir / "coefficient_summary.csv", index=False)
    aggregation = {
        "run_dir": rel_to_repo(run_dir),
        "successful_replicates": int(A_bootstrap.shape[0]),
        "coefficient_shape": list(A_original.shape),
        "bootstrap_draw_shape": list(A_bootstrap.shape),
        "beta_shape": list(beta_original.shape),
        "beta_bootstrap_draw_shape": list(beta_bootstrap.shape),
        "sigma2_shape": list(sigma2_original.shape),
        "sigma2_bootstrap_draw_shape": list(sigma2_bootstrap.shape),
        "confidence_level": confidence_level,
        "ci_method": str(config.get("ci_method", "percentile")),
        "run_status": (
            "pilot_not_for_manuscript"
            if "pilot" in str(config["name"]).lower()
            else "final_pointwise_percentile"
        ),
    }
    (run_dir / "aggregation_metadata.json").write_text(
        json.dumps(aggregation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

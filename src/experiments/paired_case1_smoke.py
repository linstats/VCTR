"""Smoke experiment for paired Case 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dgps import PairedCase1DGP
from src.metrics import (
    beta_rmse,
    miae,
    rho_abs_error,
    sigma2_abs_error,
    sigma_frobenius_error,
)
from src.models import PairedEyeVCTRModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-subject", type=int, default=80)
    parser.add_argument("--R", type=int, default=4)
    parser.add_argument("--S", type=int, default=4)
    parser.add_argument("--p0", type=int, default=2)
    parser.add_argument("--coef-type", type=str, default="quadratic")
    parser.add_argument(
        "--beta",
        type=str,
        default=None,
        help="Comma-separated beta vector. Defaults to the Case I paper setting (3,...,3).",
    )
    parser.add_argument("--sigma2", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--bandwidth", type=float, default=None)
    parser.add_argument("--bandwidth-method", type=str, default="stage1_kfold_cv")
    parser.add_argument(
        "--bandwidth-grid",
        type=str,
        default=None,
        help="Comma-separated bandwidth candidates. If provided while --bandwidth is omitted, auto CV is used.",
    )
    parser.add_argument("--ridge", type=float, default=0.0)
    return parser.parse_args()


def parse_beta(beta_arg: str | None, p0: int) -> tuple[float, ...] | None:
    """Parse a comma-separated beta vector argument."""

    if beta_arg is None:
        return None
    parts = [part.strip() for part in beta_arg.split(",") if part.strip()]
    beta = tuple(float(part) for part in parts)
    if len(beta) != p0:
        raise ValueError("--beta must contain exactly p0 comma-separated values.")
    return beta


def parse_bandwidth_grid(grid_arg: str | None) -> tuple[float, ...] | None:
    """Parse a comma-separated bandwidth grid."""

    if grid_arg is None:
        return None
    parts = [part.strip() for part in grid_arg.split(",") if part.strip()]
    if not parts:
        raise ValueError("--bandwidth-grid must not be empty.")
    return tuple(float(part) for part in parts)


def to_json_safe(obj):
    """Convert nested metrics/config objects into JSON-safe values."""

    if isinstance(obj, dict):
        return {key: to_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(value) for value in obj]
    if isinstance(obj, tuple):
        return [to_json_safe(value) for value in obj]
    if isinstance(obj, np.generic):
        return to_json_safe(obj.item())
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def save_outputs(
    output_root: Path,
    seed: int,
    dataset,
    result,
    metrics: dict[str, float | int | str],
) -> None:
    data_dir = output_root / "data"
    estimates_dir = output_root / "estimates"
    results_dir = output_root / "results"
    for path in (data_dir, estimates_dir, results_dir):
        path.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        data_dir / f"seed_{seed:04d}_dataset.npz",
        subject_ids=dataset.subject_ids,
        eye_ids=dataset.eye_ids,
        t=dataset.t,
        Z=dataset.Z,
        X=dataset.X,
        y=dataset.y,
        A_true=dataset.A_true,
        beta_true=dataset.beta_true,
        Sigma_true=dataset.Sigma_true,
    )
    np.savez_compressed(
        estimates_dir / f"seed_{seed:04d}_estimate.npz",
        A_hat_iid=result.initial.A_hat,
        beta_hat_iid=result.initial.beta_hat,
        residuals_iid=result.initial.residuals,
        best_bandwidth=result.initial.meta["bandwidth_selected"],
        bandwidth_method=result.initial.meta["bandwidth_method"],
        bandwidth_grid=np.asarray(result.initial.meta["bandwidth_grid"], dtype=float),
        bandwidth_cv_scores=np.asarray(result.initial.meta["bandwidth_cv_scores"], dtype=object),
        sigma2_hat=result.covariance.sigma2_hat,
        rho_hat=result.covariance.rho_hat,
        Sigma_hat=result.covariance.Sigma_hat,
        A_hat_final=result.A_hat,
        beta_hat_final=result.beta_hat,
        fitted_values=result.fitted_values,
    )
    with (results_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(to_json_safe(metrics), f, indent=2)
    with (results_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(to_json_safe(metrics), f, indent=2)


def main() -> None:
    args = parse_args()
    output_root = Path(__file__).with_suffix("")
    beta_true = parse_beta(args.beta, args.p0)
    bandwidth_grid = parse_bandwidth_grid(args.bandwidth_grid)

    dataset = PairedCase1DGP(
        n_subject=args.n_subject,
        R=args.R,
        S=args.S,
        p0=args.p0,
        coef_type=args.coef_type,
        beta_true=beta_true,
        sigma2=args.sigma2,
        rho=args.rho,
    ).sample(seed=args.seed)

    model = PairedEyeVCTRModel(
        bandwidth=args.bandwidth,
        bandwidth_method=args.bandwidth_method,
        bandwidth_grid=bandwidth_grid,
        ridge=args.ridge,
    )
    result = model.fit(dataset)
    best_bandwidth = float(result.initial.meta["bandwidth_selected"])

    metrics = {
        "seed": args.seed,
        "n_subject": args.n_subject,
        "R": args.R,
        "S": args.S,
        "p0": args.p0,
        "coef_type": args.coef_type,
        "beta_true": dataset.beta_true.tolist(),
        "bandwidth": args.bandwidth,
        "best_bandwidth": best_bandwidth,
        "bandwidth_method": result.initial.meta["bandwidth_method"],
        "bandwidth_grid": result.initial.meta["bandwidth_grid"],
        "bandwidth_cv_scores": result.initial.meta["bandwidth_cv_scores"],
        "ridge": args.ridge,
        "miae_iid": miae(dataset.A_true, result.initial.A_hat),
        "miae_final": miae(dataset.A_true, result.A_hat),
        "beta_rmse_iid": beta_rmse(dataset.beta_true, result.initial.beta_hat),
        "beta_rmse_final": beta_rmse(dataset.beta_true, result.beta_hat),
        "sigma2_abs_error": sigma2_abs_error(args.sigma2, result.covariance.sigma2_hat),
        "rho_abs_error": rho_abs_error(args.rho, result.covariance.rho_hat),
        "Sigma_fro_error": sigma_frobenius_error(dataset.Sigma_true, result.covariance.Sigma_hat),
    }

    save_outputs(output_root, args.seed, dataset, result, metrics)
    print(json.dumps(to_json_safe(metrics), indent=2))


if __name__ == "__main__":
    main()

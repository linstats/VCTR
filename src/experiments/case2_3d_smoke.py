"""Smoke experiment for Case 2 3D."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dgps import SUPPORTED_SIGMA2_FUNCTIONS, PairedCase2AltbaseDGP
from src.metrics import (
    beta_rmse,
    miae,
    rho_abs_error,
    rho_error,
    sigma2_miae,
    sigma2_rmise,
)
from src.models import PairedEyeVCTRModel
from src.utils.plotting import parse_a_indices, save_function_plots


@dataclass(slots=True)
class Case23DSmokeConfig:
    seed: int = 1
    n_subject: int = 1000
    R: int = 3
    S: int = 27
    p0: int = 4
    a_eval_mode: str = "full"
    a_eval_num_points: int = 500
    a_eval_grid: str = "quantile"
    a_interp: str = "linear"
    coef_type: str = "base1"
    beta_true: tuple[float, ...] | None = None
    sigma2: float = 1.0
    sigma2_function: str = "constant"
    rho: float = 0.3
    covariance_mode: str = "exchangeable_varying_sigma"
    signal_bandwidth: float | None = 0.20
    signal_bandwidth_method: str = "stage1_kfold_cv"
    signal_bandwidth_grid: tuple[float, ...] | None = None
    variance_bandwidth: float | None = 0.20
    variance_bandwidth_method: str = "stage2_kfold_cv"
    variance_bandwidth_grid: tuple[float, ...] | None = None
    ridge: float = 0.0
    plot_functions: bool = False
    plot_a_indices: str = "all"
    plot_max_a_panels: int = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to the script-matched folder beside this file.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-subject", type=int, default=1000)
    parser.add_argument("--R", type=int, default=3)
    parser.add_argument("--S", type=int, default=27)
    parser.add_argument("--p0", type=int, default=4)
    parser.add_argument("--a-eval-mode", type=str, default="full", choices=["full", "anchor_grid"])
    parser.add_argument("--a-eval-num-points", type=int, default=500)
    parser.add_argument("--a-eval-grid", type=str, default="quantile", choices=["quantile", "uniform"])
    parser.add_argument("--a-interp", type=str, default="linear", choices=["linear"])
    parser.add_argument("--coef-type", type=str, default="base1")
    parser.add_argument(
        "--beta",
        type=str,
        default="2.0,1.0,-1.0,0.5",
        help="Comma-separated beta vector. Default matches the 3D altbase design.",
    )
    parser.add_argument("--sigma2", type=float, default=1.0)
    parser.add_argument("--sigma2-function", type=str, default="constant", choices=SUPPORTED_SIGMA2_FUNCTIONS)
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--covariance-mode", type=str, default="exchangeable_varying_sigma")
    parser.add_argument("--signal-bandwidth", type=float, default=0.20)
    parser.add_argument("--signal-bandwidth-method", type=str, default="stage1_kfold_cv")
    parser.add_argument(
        "--signal-bandwidth-grid",
        type=str,
        default=None,
        help="Comma-separated signal-bandwidth candidates. If provided while --signal-bandwidth is omitted, auto CV is used.",
    )
    parser.add_argument("--variance-bandwidth", type=float, default=0.20)
    parser.add_argument("--variance-bandwidth-method", type=str, default="stage2_kfold_cv")
    parser.add_argument(
        "--variance-bandwidth-grid",
        type=str,
        default=None,
        help="Comma-separated variance-bandwidth candidates. If provided while --variance-bandwidth is omitted, auto CV is used.",
    )
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument(
        "--plot-functions",
        action="store_true",
        help="Save diagnostic plots for selected A[r,s](t) components and sigma^2(t).",
    )
    parser.add_argument(
        "--plot-a-indices",
        type=str,
        default="all",
        help="Zero-based A component indices for plotting, e.g. 0:0,1:4, or all.",
    )
    parser.add_argument(
        "--plot-max-a-panels",
        type=int,
        default=16,
        help="Maximum number of A component panels to draw when plotting. Use 0 to disable A panels.",
    )
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
        raise ValueError("bandwidth grid must not be empty.")
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


def build_config(args: argparse.Namespace) -> Case23DSmokeConfig:
    """Build the reusable smoke config from CLI args."""

    return Case23DSmokeConfig(
        seed=args.seed,
        n_subject=args.n_subject,
        R=args.R,
        S=args.S,
        p0=args.p0,
        a_eval_mode=args.a_eval_mode,
        a_eval_num_points=args.a_eval_num_points,
        a_eval_grid=args.a_eval_grid,
        a_interp=args.a_interp,
        coef_type=args.coef_type,
        beta_true=parse_beta(args.beta, args.p0),
        sigma2=args.sigma2,
        sigma2_function=args.sigma2_function,
        rho=args.rho,
        covariance_mode=args.covariance_mode,
        signal_bandwidth=args.signal_bandwidth,
        signal_bandwidth_method=args.signal_bandwidth_method,
        signal_bandwidth_grid=parse_bandwidth_grid(args.signal_bandwidth_grid),
        variance_bandwidth=args.variance_bandwidth,
        variance_bandwidth_method=args.variance_bandwidth_method,
        variance_bandwidth_grid=parse_bandwidth_grid(args.variance_bandwidth_grid),
        ridge=args.ridge,
        plot_functions=args.plot_functions,
        plot_a_indices=args.plot_a_indices,
        plot_max_a_panels=args.plot_max_a_panels,
    )


def run_case23d_once(config: Case23DSmokeConfig):
    """Run one reusable Case 2 3D fit and return dataset, fit, metrics."""

    dataset = PairedCase2AltbaseDGP(
        n_subject=config.n_subject,
        R=config.R,
        S=config.S,
        p0=config.p0,
        coef_type=config.coef_type,
        beta_true=config.beta_true,
        sigma2=config.sigma2,
        sigma2_function=config.sigma2_function,
        rho=config.rho,
    ).sample(seed=config.seed)

    model = PairedEyeVCTRModel(
        covariance_mode=config.covariance_mode,
        a_eval_mode=config.a_eval_mode,
        a_eval_num_points=config.a_eval_num_points,
        a_eval_grid=config.a_eval_grid,
        a_interp=config.a_interp,
        signal_bandwidth=config.signal_bandwidth,
        signal_bandwidth_method=config.signal_bandwidth_method,
        signal_bandwidth_grid=config.signal_bandwidth_grid,
        variance_bandwidth=config.variance_bandwidth,
        variance_bandwidth_method=config.variance_bandwidth_method,
        variance_bandwidth_grid=config.variance_bandwidth_grid,
        ridge=config.ridge,
    )
    result = model.fit(dataset)
    best_signal_bandwidth = float(result.initial.meta["signal_bandwidth_selected"])
    sigma2_true_t = np.asarray(dataset.meta["sigma2_true_t"], dtype=float)
    rho_signed_error = rho_error(config.rho, result.covariance.rho_hat)

    metrics = {
        "seed": config.seed,
        "n_subject": config.n_subject,
        "R": config.R,
        "S": config.S,
        "p0": config.p0,
        "a_eval_mode": result.initial.meta.get("a_eval_mode"),
        "a_eval_requested_num_points": result.initial.meta.get("a_eval_requested_num_points"),
        "a_eval_selected_points": result.initial.meta.get("a_eval_selected_points"),
        "a_eval_grid": result.initial.meta.get("a_eval_grid"),
        "a_interp": result.initial.meta.get("a_interp"),
        "a_eval_used_acceleration": result.initial.meta.get("a_eval_used_acceleration"),
        "coef_type": config.coef_type,
        "beta_true": dataset.beta_true.tolist(),
        "sigma2": config.sigma2,
        "sigma2_function": config.sigma2_function,
        "covariance_mode": config.covariance_mode,
        "signal_bandwidth": config.signal_bandwidth,
        "best_signal_bandwidth": best_signal_bandwidth,
        "signal_bandwidth_method": result.initial.meta["signal_bandwidth_method"],
        "signal_bandwidth_grid": result.initial.meta["signal_bandwidth_grid"],
        "signal_bandwidth_cv_scores": result.initial.meta["signal_bandwidth_cv_scores"],
        "variance_bandwidth": config.variance_bandwidth,
        "best_variance_bandwidth": result.covariance.meta.get("variance_bandwidth_selected"),
        "variance_bandwidth_method": result.covariance.meta.get("variance_bandwidth_method"),
        "variance_bandwidth_grid": result.covariance.meta.get("variance_bandwidth_grid"),
        "variance_bandwidth_cv_scores": result.covariance.meta.get("variance_bandwidth_cv_scores"),
        "ridge": config.ridge,
        "miae_iid": miae(dataset.A_true, result.initial.A_hat),
        "miae_final": miae(dataset.A_true, result.A_hat),
        "beta_rmse_iid": beta_rmse(dataset.beta_true, result.initial.beta_hat),
        "beta_rmse_final": beta_rmse(dataset.beta_true, result.beta_hat),
        "sigma2_miae": sigma2_miae(sigma2_true_t, result.covariance.sigma2_hat_t),
        "sigma2_rmise": sigma2_rmise(sigma2_true_t, result.covariance.sigma2_hat_t),
        "rho_error": rho_signed_error,
        "rho_abs_error": rho_abs_error(config.rho, result.covariance.rho_hat),
        "rho_mae": abs(rho_signed_error),
        "rho_rmse": abs(rho_signed_error),
        "Sigma_fro_error": None,
    }
    return dataset, result, metrics


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
        sigma2_true_t=np.asarray(dataset.meta.get("sigma2_true_t", []), dtype=float),
    )
    np.savez_compressed(
        estimates_dir / f"seed_{seed:04d}_estimate.npz",
        A_hat_iid=result.initial.A_hat,
        beta_hat_iid=result.initial.beta_hat,
        residuals_iid=result.initial.residuals,
        covariance_mode=result.covariance.covariance_mode,
        best_signal_bandwidth=result.initial.meta["signal_bandwidth_selected"],
        signal_bandwidth_method=result.initial.meta["signal_bandwidth_method"],
        signal_bandwidth_grid=np.asarray(result.initial.meta["signal_bandwidth_grid"], dtype=float),
        signal_bandwidth_cv_scores=np.asarray(result.initial.meta["signal_bandwidth_cv_scores"], dtype=object),
        best_variance_bandwidth=result.covariance.meta.get("variance_bandwidth_selected"),
        variance_bandwidth_method=result.covariance.meta.get("variance_bandwidth_method"),
        variance_bandwidth_grid=np.asarray(result.covariance.meta.get("variance_bandwidth_grid", []), dtype=float),
        variance_bandwidth_cv_scores=np.asarray(result.covariance.meta.get("variance_bandwidth_cv_scores", []), dtype=object),
        sigma2_hat_t=result.covariance.sigma2_hat_t,
        sigma2_hat_mean=result.covariance.sigma2_hat,
        rho_hat=result.covariance.rho_hat,
        Sigma_hat=result.covariance.Sigma_hat,
        Sigma_hat_blocks=result.covariance.Sigma_hat_blocks,
        A_hat_final=result.A_hat,
        beta_hat_final=result.beta_hat,
        fitted_values=result.fitted_values,
    )
    with (results_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(to_json_safe(metrics), f, indent=2)
    with (results_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(to_json_safe(metrics), f, indent=2)


def maybe_save_plots(output_root: Path, seed: int, dataset, result, config: Case23DSmokeConfig) -> list[Path]:
    """Save optional function diagnostics for this smoke run."""

    if not config.plot_functions:
        return []
    max_a_panels = int(config.plot_max_a_panels)
    if max_a_panels < 0:
        raise ValueError("--plot-max-a-panels must be nonnegative.")
    a_indices = [] if max_a_panels == 0 else parse_a_indices(config.plot_a_indices, result.A_hat.shape[-2:])
    return save_function_plots(
        output_dir=output_root / "plots",
        stem=f"seed_{seed:04d}",
        dataset=dataset,
        result=result,
        a_indices=a_indices,
        max_a_panels=max_a_panels,
    )


def main() -> None:
    args = parse_args()
    output_root = args.output_root if args.output_root is not None else Path(__file__).with_suffix("")
    config = build_config(args)
    dataset, result, metrics = run_case23d_once(config)
    plot_paths = maybe_save_plots(output_root, config.seed, dataset, result, config)
    if plot_paths:
        metrics["plot_paths"] = [str(path) for path in plot_paths]
    save_outputs(output_root, config.seed, dataset, result, metrics)
    print(json.dumps(to_json_safe(metrics), indent=2))


if __name__ == "__main__":
    main()

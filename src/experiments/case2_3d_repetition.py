"""Repeated Case 2 3D simulation driver."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dgps import SUPPORTED_SIGMA2_FUNCTIONS
from src.metrics import beta_mae, rmise
from src.models import PairedEyeVCTRModel
from src.utils.plotting import parse_a_indices, save_function_plots

from src.experiments.case2_3d_smoke import (
    Case23DSmokeConfig,
    parse_beta,
    parse_bandwidth_grid,
    run_case23d_once,
    to_json_safe,
)


DEFAULT_COEF_TYPES = ("base1", "base2", "base3", "base4", "base5", "base6")
DEFAULT_SIGMA2_FUNCTIONS = SUPPORTED_SIGMA2_FUNCTIONS


@dataclass(slots=True)
class Case23DRecord:
    n_subject: int
    coef_type: str
    rep: int
    seed: int
    success: int
    error_message: str
    elapsed_seconds: float
    covariance_mode: str
    signal_bandwidth_input: str
    signal_bandwidth_method: str
    best_signal_bandwidth: float | None
    variance_bandwidth_input: str
    variance_bandwidth_method: str
    best_variance_bandwidth: float | None
    sigma2_true: float
    sigma2_function: str
    a_eval_mode: str
    a_eval_selected_points: int
    rho_true: float
    rho_error: float | None
    miae_iid: float | None
    rmise_iid: float | None
    beta_mae_iid: float | None
    beta_rmse_iid: float | None
    miae_final: float | None
    rmise_final: float | None
    beta_mae_final: float | None
    beta_rmse_final: float | None
    sigma2_miae: float | None
    sigma2_rmise: float | None
    rho_abs_error: float | None
    Sigma_fro_error: float | None


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes > 0:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-subject-values", type=int, nargs="+", default=[1000, 2000])
    parser.add_argument("--coef-types", type=str, nargs="+", default=list(DEFAULT_COEF_TYPES))
    parser.add_argument("--n-rep", type=int, default=30)
    parser.add_argument("--seed-base", type=int, default=123)
    parser.add_argument("--R", type=int, default=3)
    parser.add_argument("--S", type=int, default=27)
    parser.add_argument("--p0", type=int, default=4)
    parser.add_argument("--a-eval-mode", type=str, default="full", choices=["full", "anchor_grid"])
    parser.add_argument("--a-eval-num-points", type=int, default=500)
    parser.add_argument("--a-eval-grid", type=str, default="quantile", choices=["quantile", "uniform"])
    parser.add_argument("--a-interp", type=str, default="linear", choices=["linear"])
    parser.add_argument(
        "--beta",
        type=str,
        default="2.0,1.0,-1.0,0.5",
        help="Comma-separated beta vector. Default matches the 3D altbase design.",
    )
    parser.add_argument("--sigma2", type=float, default=1.0)
    parser.add_argument(
        "--sigma2-function",
        type=str,
        default=None,
        choices=SUPPORTED_SIGMA2_FUNCTIONS,
        help="Optional single sigma2(t) function. If provided, overrides --sigma2-functions.",
    )
    parser.add_argument(
        "--sigma2-functions",
        type=str,
        nargs="+",
        default=list(DEFAULT_SIGMA2_FUNCTIONS),
        choices=SUPPORTED_SIGMA2_FUNCTIONS,
        help="Sigma2(t) functions to iterate over in repetition runs.",
    )
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument(
        "--rho-values",
        type=float,
        nargs="+",
        default=None,
        help="Optional list of rho values. If provided, overrides --rho for batch runs.",
    )
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
    parser.add_argument("--large-n-threshold", type=int, default=2000)
    parser.add_argument("--prompt-accelerate-large-n", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run directory name. Defaults to run_YYYYMMDD_HHMMSS.",
    )
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--save-estimates", action="store_true")
    parser.add_argument(
        "--plot-functions",
        action="store_true",
        help="Save diagnostic plots for selected A[r,s](t) components and sigma^2(t) for every successful repetition.",
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
        help="Maximum number of A component panels to draw per repetition. Use 0 to disable A panels.",
    )
    return parser.parse_args()


def default_run_name() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def resolved_rho_values(args: argparse.Namespace) -> list[float]:
    return [float(value) for value in (args.rho_values or [args.rho])]


def resolved_sigma2_functions(args: argparse.Namespace) -> list[str]:
    """Return the sigma2(t) functions to iterate over for this run."""

    if args.sigma2_function is not None:
        return [str(args.sigma2_function)]
    return [str(value) for value in args.sigma2_functions]


def a_eval_mode_explicitly_requested() -> bool:
    return any(arg == "--a-eval-mode" or arg.startswith("--a-eval-mode=") for arg in sys.argv[1:])


def maybe_prompt_for_large_n_acceleration(args: argparse.Namespace) -> None:
    if not args.prompt_accelerate_large_n:
        return
    if max(args.n_subject_values) <= args.large_n_threshold:
        return
    if a_eval_mode_explicitly_requested():
        return
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return

    prompt = (
        f"Detected n_subject > {args.large_n_threshold}. "
        f"Enable anchor-grid acceleration with {args.a_eval_num_points} evaluation points? [Y/n]: "
    )
    try:
        response = input(prompt).strip().lower()
    except EOFError:
        return
    if response in {"", "y", "yes"}:
        args.a_eval_mode = "anchor_grid"


def prepare_run_root(base_dir: Path, run_name: str | None) -> Path:
    resolved_name = run_name or default_run_name()
    run_root = base_dir / resolved_name
    run_root.mkdir(parents=True, exist_ok=False)
    return run_root


def output_paths(run_root: Path) -> dict[str, Path]:
    results_dir = run_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return {
        "results_dir": results_dir,
        "raw": results_dir / "raw_results.csv",
        "summary": results_dir / "summary_results.csv",
        "progress": run_root / "progress.json",
        "config": run_root / "run_config.json",
    }


def rho_label(rho: float) -> str:
    return f"{rho:.3f}".replace("-", "m").replace(".", "p")


def artifact_stem(
    n_subject: int,
    coef_type: str,
    sigma2_function: str,
    rho: float,
    rep: int,
    seed: int,
) -> str:
    return f"n{n_subject}_{coef_type}_{sigma2_function}_rho{rho_label(rho)}_rep{rep:03d}_seed{seed:04d}"


def maybe_save_dataset_with_stem(output_root: Path, stem: str, dataset) -> None:
    data_dir = output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        data_dir / f"{stem}_dataset.npz",
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


def maybe_save_estimate_with_stem(output_root: Path, stem: str, result) -> None:
    estimates_dir = output_root / "estimates"
    estimates_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        estimates_dir / f"{stem}_estimate.npz",
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


def maybe_save_plots_with_stem(output_root: Path, stem: str, dataset, result, args: argparse.Namespace) -> list[Path]:
    """Save optional function diagnostics under a unique repetition stem."""

    max_a_panels = int(args.plot_max_a_panels)
    if max_a_panels < 0:
        raise ValueError("--plot-max-a-panels must be nonnegative.")
    a_indices = [] if max_a_panels == 0 else parse_a_indices(args.plot_a_indices, result.A_hat.shape[-2:])
    return save_function_plots(
        output_dir=output_root / "plots",
        stem=stem,
        dataset=dataset,
        result=result,
        a_indices=a_indices,
        max_a_panels=max_a_panels,
    )


def run_one(
    *,
    n_subject: int,
    coef_type: str,
    sigma2_function: str,
    rho_true: float,
    rep: int,
    seed: int,
    beta_true: tuple[float, ...],
    args: argparse.Namespace,
    signal_bandwidth_grid: tuple[float, ...] | None,
    variance_bandwidth_grid: tuple[float, ...] | None,
    output_root: Path,
) -> Case23DRecord:
    start = time.perf_counter()
    if args.signal_bandwidth is not None:
        signal_bandwidth_input = f"{args.signal_bandwidth:.12g}"
        failure_signal_bandwidth_method = "fixed"
    elif signal_bandwidth_grid is None:
        signal_bandwidth_input = f"{PairedEyeVCTRModel.DEFAULT_SIGNAL_BANDWIDTH:.12g}"
        failure_signal_bandwidth_method = "default_fixed"
    else:
        signal_bandwidth_input = "auto"
        failure_signal_bandwidth_method = args.signal_bandwidth_method
    if args.covariance_mode == "exchangeable_constant":
        variance_bandwidth_input = "not_used"
        failure_variance_bandwidth_method = "not_used"
    elif args.variance_bandwidth is not None:
        variance_bandwidth_input = f"{args.variance_bandwidth:.12g}"
        failure_variance_bandwidth_method = "fixed"
    elif variance_bandwidth_grid is None:
        variance_bandwidth_input = f"{PairedEyeVCTRModel.DEFAULT_VARIANCE_BANDWIDTH:.12g}"
        failure_variance_bandwidth_method = "default_fixed"
    else:
        variance_bandwidth_input = "auto"
        failure_variance_bandwidth_method = args.variance_bandwidth_method

    try:
        config = Case23DSmokeConfig(
            seed=seed,
            n_subject=n_subject,
            R=args.R,
            S=args.S,
            p0=args.p0,
            a_eval_mode=args.a_eval_mode,
            a_eval_num_points=args.a_eval_num_points,
            a_eval_grid=args.a_eval_grid,
            a_interp=args.a_interp,
            coef_type=coef_type,
            beta_true=beta_true,
            sigma2=args.sigma2,
            sigma2_function=sigma2_function,
            rho=rho_true,
            covariance_mode=args.covariance_mode,
            signal_bandwidth=args.signal_bandwidth,
            signal_bandwidth_method=args.signal_bandwidth_method,
            signal_bandwidth_grid=signal_bandwidth_grid,
            variance_bandwidth=args.variance_bandwidth,
            variance_bandwidth_method=args.variance_bandwidth_method,
            variance_bandwidth_grid=variance_bandwidth_grid,
            ridge=args.ridge,
            plot_functions=False,
        )
        dataset, result, metrics = run_case23d_once(config)

        if args.save_data:
            maybe_save_dataset_with_stem(
                output_root=output_root,
                stem=artifact_stem(n_subject, coef_type, sigma2_function, rho_true, rep, seed),
                dataset=dataset,
            )
        if args.save_estimates:
            maybe_save_estimate_with_stem(
                output_root=output_root,
                stem=artifact_stem(n_subject, coef_type, sigma2_function, rho_true, rep, seed),
                result=result,
            )
        if args.plot_functions:
            maybe_save_plots_with_stem(
                output_root=output_root,
                stem=artifact_stem(n_subject, coef_type, sigma2_function, rho_true, rep, seed),
                dataset=dataset,
                result=result,
                args=args,
            )

        elapsed = time.perf_counter() - start
        return Case23DRecord(
            n_subject=n_subject,
            coef_type=coef_type,
            rep=rep,
            seed=seed,
            success=1,
            error_message="",
            elapsed_seconds=elapsed,
            covariance_mode=result.covariance.covariance_mode,
            signal_bandwidth_input=signal_bandwidth_input,
            signal_bandwidth_method=result.initial.meta["signal_bandwidth_method"],
            best_signal_bandwidth=float(result.initial.meta["signal_bandwidth_selected"]),
            variance_bandwidth_input=variance_bandwidth_input,
            variance_bandwidth_method=result.covariance.meta.get("variance_bandwidth_method"),
            best_variance_bandwidth=result.covariance.meta.get("variance_bandwidth_selected"),
            sigma2_true=args.sigma2,
            sigma2_function=sigma2_function,
            a_eval_mode=result.initial.meta.get("a_eval_mode", "full"),
            a_eval_selected_points=int(result.initial.meta.get("a_eval_selected_points", n_subject)),
            rho_true=rho_true,
            rho_error=metrics["rho_error"],
            miae_iid=metrics["miae_iid"],
            rmise_iid=rmise(dataset.A_true, result.initial.A_hat),
            beta_mae_iid=beta_mae(dataset.beta_true, result.initial.beta_hat),
            beta_rmse_iid=metrics["beta_rmse_iid"],
            miae_final=metrics["miae_final"],
            rmise_final=rmise(dataset.A_true, result.A_hat),
            beta_mae_final=beta_mae(dataset.beta_true, result.beta_hat),
            beta_rmse_final=metrics["beta_rmse_final"],
            sigma2_miae=metrics["sigma2_miae"],
            sigma2_rmise=metrics["sigma2_rmise"],
            rho_abs_error=metrics["rho_abs_error"],
            Sigma_fro_error=None,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return Case23DRecord(
            n_subject=n_subject,
            coef_type=coef_type,
            rep=rep,
            seed=seed,
            success=0,
            error_message=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=elapsed,
            covariance_mode=args.covariance_mode,
            signal_bandwidth_input=signal_bandwidth_input,
            signal_bandwidth_method=failure_signal_bandwidth_method,
            best_signal_bandwidth=None,
            variance_bandwidth_input=variance_bandwidth_input,
            variance_bandwidth_method=failure_variance_bandwidth_method,
            best_variance_bandwidth=None,
            sigma2_true=args.sigma2,
            sigma2_function=sigma2_function,
            a_eval_mode=args.a_eval_mode,
            a_eval_selected_points=min(n_subject, args.a_eval_num_points) if args.a_eval_mode == "anchor_grid" else n_subject,
            rho_true=rho_true,
            rho_error=None,
            miae_iid=None,
            rmise_iid=None,
            beta_mae_iid=None,
            beta_rmse_iid=None,
            miae_final=None,
            rmise_final=None,
            beta_mae_final=None,
            beta_rmse_final=None,
            sigma2_miae=None,
            sigma2_rmise=None,
            rho_abs_error=None,
            Sigma_fro_error=None,
        )


def write_run_config(run_root: Path, args: argparse.Namespace, total_jobs: int) -> None:
    run_config = {
        "script": "src/experiments/case2_3d_repetition.py",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_root.name,
        "total_jobs": total_jobs,
        "n_subject_values": args.n_subject_values,
        "coef_types": args.coef_types,
        "n_rep": args.n_rep,
        "seed_base": args.seed_base,
        "R": args.R,
        "S": args.S,
        "p0": args.p0,
        "a_eval_mode": args.a_eval_mode,
        "a_eval_num_points": args.a_eval_num_points,
        "a_eval_grid": args.a_eval_grid,
        "a_interp": args.a_interp,
        "beta": args.beta,
        "sigma2": args.sigma2,
        "sigma2_function": args.sigma2_function,
        "sigma2_functions": resolved_sigma2_functions(args),
        "rho": args.rho,
        "rho_values": args.rho_values,
        "covariance_mode": args.covariance_mode,
        "signal_bandwidth": args.signal_bandwidth,
        "signal_bandwidth_method": args.signal_bandwidth_method,
        "signal_bandwidth_grid": args.signal_bandwidth_grid,
        "variance_bandwidth": args.variance_bandwidth,
        "variance_bandwidth_method": args.variance_bandwidth_method,
        "variance_bandwidth_grid": args.variance_bandwidth_grid,
        "ridge": args.ridge,
        "large_n_threshold": args.large_n_threshold,
        "prompt_accelerate_large_n": args.prompt_accelerate_large_n,
        "n_jobs": args.n_jobs,
        "save_data": args.save_data,
        "save_estimates": args.save_estimates,
        "plot_functions": args.plot_functions,
        "plot_a_indices": args.plot_a_indices,
        "plot_max_a_panels": args.plot_max_a_panels,
    }
    with output_paths(run_root)["config"].open("w", encoding="utf-8") as f:
        json.dump(to_json_safe(run_config), f, indent=2)


def initialize_raw_csv(run_root: Path) -> None:
    raw_path = output_paths(run_root)["raw"]
    fieldnames = list(Case23DRecord.__dataclass_fields__.keys())
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_raw_record(run_root: Path, record: Case23DRecord) -> None:
    raw_path = output_paths(run_root)["raw"]
    fieldnames = list(Case23DRecord.__dataclass_fields__.keys())
    with raw_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(asdict(record))


def rewrite_summary_csv(run_root: Path, summary: list[dict[str, float | int | str]]) -> None:
    if not summary:
        return
    summary_path = output_paths(run_root)["summary"]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def write_progress_snapshot(
    run_root: Path,
    *,
    completed_jobs: int,
    total_jobs: int,
    records: list[Case23DRecord],
    global_start: float,
) -> None:
    elapsed_total = time.perf_counter() - global_start
    avg_elapsed = elapsed_total / completed_jobs if completed_jobs else 0.0
    eta_seconds = avg_elapsed * (total_jobs - completed_jobs) if completed_jobs else None
    progress = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "completed_jobs": completed_jobs,
        "total_jobs": total_jobs,
        "n_success": int(sum(record.success for record in records)),
        "n_fail": int(sum(1 - record.success for record in records)),
        "elapsed_seconds": elapsed_total,
        "eta_seconds": eta_seconds,
        "latest_record": None if not records else to_json_safe(asdict(records[-1])),
    }
    with output_paths(run_root)["progress"].open("w", encoding="utf-8") as f:
        json.dump(to_json_safe(progress), f, indent=2)


def build_tasks(
    args: argparse.Namespace,
    beta_true: tuple[float, ...],
    signal_bandwidth_grid: tuple[float, ...] | None,
    variance_bandwidth_grid: tuple[float, ...] | None,
    output_root: Path,
) -> list[dict]:
    tasks: list[dict] = []
    for n_subject in args.n_subject_values:
        for coef_type in args.coef_types:
            for sigma2_function in resolved_sigma2_functions(args):
                for rho_true in resolved_rho_values(args):
                    for rep in range(args.n_rep):
                        seed = args.seed_base + rep
                        tasks.append(
                            {
                                "n_subject": n_subject,
                                "coef_type": coef_type,
                                "sigma2_function": sigma2_function,
                                "rho_true": rho_true,
                                "rep": rep,
                                "seed": seed,
                                "beta_true": beta_true,
                                "args": args,
                                "signal_bandwidth_grid": signal_bandwidth_grid,
                                "variance_bandwidth_grid": variance_bandwidth_grid,
                                "output_root": output_root,
                            }
                        )
    return tasks


def print_progress_line(
    *,
    record: Case23DRecord,
    completed_jobs: int,
    total_jobs: int,
    global_start: float,
    n_rep: int,
) -> None:
    status = "done" if record.success else "fail"
    elapsed_total = time.perf_counter() - global_start
    avg_elapsed = elapsed_total / completed_jobs
    eta_seconds = avg_elapsed * (total_jobs - completed_jobs)
    print(
        f"[{completed_jobs}/{total_jobs}] {status} "
        f"n_subject={record.n_subject} coef={record.coef_type:10s} "
        f"sigma2={record.sigma2_function:8s} rho={record.rho_true:.3f} "
        f"rep={record.rep + 1}/{n_rep} seed={record.seed} "
        f"best_h={record.best_signal_bandwidth if record.best_signal_bandwidth is not None else 'NA'} "
        f"best_hbar={record.best_variance_bandwidth if record.best_variance_bandwidth is not None else 'NA'} "
        f"elapsed={format_duration(record.elapsed_seconds)} "
        f"eta={format_duration(eta_seconds)}"
    )


def summarize(records: Iterable[Case23DRecord]) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[int, str, float, str, str, int, str, str], list[Case23DRecord]] = {}
    for rec in records:
        grouped.setdefault(
            (
                rec.n_subject,
                rec.coef_type,
                rec.rho_true,
                rec.sigma2_function,
                rec.a_eval_mode,
                rec.a_eval_selected_points,
                rec.covariance_mode,
                rec.signal_bandwidth_method,
            ),
            [],
        ).append(rec)

    metric_fields = (
        "miae_iid",
        "rmise_iid",
        "beta_mae_iid",
        "beta_rmse_iid",
        "miae_final",
        "rmise_final",
        "beta_mae_final",
        "beta_rmse_final",
        "sigma2_miae",
        "sigma2_rmise",
        "rho_abs_error",
        "rho_error",
        "best_signal_bandwidth",
        "best_variance_bandwidth",
        "elapsed_seconds",
    )
    rows: list[dict[str, float | int | str]] = []
    for (
        n_subject,
        coef_type,
        rho_true,
        sigma2_function,
        a_eval_mode,
        a_eval_selected_points,
        covariance_mode,
        signal_bandwidth_method,
    ), vals in sorted(grouped.items()):
        row: dict[str, float | int | str] = {
            "n_subject": n_subject,
            "coef_type": coef_type,
            "rho_true": rho_true,
            "sigma2_function": sigma2_function,
            "a_eval_mode": a_eval_mode,
            "a_eval_selected_points": a_eval_selected_points,
            "covariance_mode": covariance_mode,
            "signal_bandwidth_method": signal_bandwidth_method,
            "n_rep": len(vals),
            "n_success": int(sum(v.success for v in vals)),
            "n_fail": int(sum(1 - v.success for v in vals)),
        }
        for field in metric_fields:
            arr = np.array([getattr(v, field) for v in vals if getattr(v, field) is not None], dtype=float)
            row[f"{field}_mean"] = float(np.mean(arr)) if arr.size else None
            row[f"{field}_std"] = float(np.std(arr, ddof=0)) if arr.size else None
        rho_errors = np.array([v.rho_error for v in vals if v.rho_error is not None], dtype=float)
        row["rho_mae"] = float(np.mean(np.abs(rho_errors))) if rho_errors.size else None
        row["rho_rmse"] = float(np.sqrt(np.mean(np.square(rho_errors)))) if rho_errors.size else None
        rows.append(row)
    return rows


def maybe_write_outputs(run_root: Path, summary: list[dict[str, float | int | str]]) -> None:
    rewrite_summary_csv(run_root, summary)
    paths = output_paths(run_root)
    print(f"Wrote raw results to {paths['raw']}")
    print(f"Wrote summary results to {paths['summary']}")
    print(f"Wrote run config to {paths['config']}")


def print_summary(summary: list[dict[str, float | int | str]]) -> None:
    def fmt(value) -> str:
        if value is None:
            return "NA"
        return f"{float(value):.4f}"

    print("\nPaired Case 2 altbase repetition summary")
    for row in summary:
        print(
            f"n_subject={row['n_subject']}, coef={row['coef_type']}, rho={float(row['rho_true']):.3f}, "
            f"sigma2={row['sigma2_function']}, mode={row['covariance_mode']}, "
            f"a_eval={row['a_eval_mode']}({row['a_eval_selected_points']}), "
            f"signal_method={row['signal_bandwidth_method']}: "
            f"MIAE_final={fmt(row['miae_final_mean'])} ({fmt(row['miae_final_std'])}), "
            f"RMISE_final={fmt(row['rmise_final_mean'])} ({fmt(row['rmise_final_std'])}), "
            f"beta_MAE_final={fmt(row['beta_mae_final_mean'])} ({fmt(row['beta_mae_final_std'])}), "
            f"beta_RMSE_final={fmt(row['beta_rmse_final_mean'])} ({fmt(row['beta_rmse_final_std'])}), "
            f"best_h={fmt(row['best_signal_bandwidth_mean'])} ({fmt(row['best_signal_bandwidth_std'])}), "
            f"best_hbar={fmt(row['best_variance_bandwidth_mean'])} ({fmt(row['best_variance_bandwidth_std'])}), "
            f"sigma2_MIAE={fmt(row['sigma2_miae_mean'])} ({fmt(row['sigma2_miae_std'])}), "
            f"sigma2_RMISE={fmt(row['sigma2_rmise_mean'])} ({fmt(row['sigma2_rmise_std'])}), "
            f"rho_MAE={fmt(row['rho_mae'])}, rho_RMSE={fmt(row['rho_rmse'])}, "
            f"elapsed={fmt(row['elapsed_seconds_mean'])} ({fmt(row['elapsed_seconds_std'])})"
        )


def run(args: argparse.Namespace) -> tuple[list[Case23DRecord], Path]:
    base_output_root = Path(__file__).with_suffix("")
    maybe_prompt_for_large_n_acceleration(args)
    beta_true = parse_beta(args.beta, args.p0)
    if beta_true is None:
        raise ValueError("parsed beta must not be None for repetition runs.")
    signal_bandwidth_grid = parse_bandwidth_grid(args.signal_bandwidth_grid)
    variance_bandwidth_grid = parse_bandwidth_grid(args.variance_bandwidth_grid)
    run_root = prepare_run_root(base_output_root, args.run_name)
    tasks = build_tasks(args, beta_true, signal_bandwidth_grid, variance_bandwidth_grid, run_root)
    records: list[Case23DRecord] = []
    total_jobs = len(tasks)
    completed_jobs = 0
    global_start = time.perf_counter()
    write_run_config(run_root, args, total_jobs)
    initialize_raw_csv(run_root)

    print(f"[run] total_jobs={total_jobs} n_jobs={args.n_jobs} run_dir={run_root}")

    if args.n_jobs == 1:
        for task in tasks:
            record = run_one(**task)
            records.append(record)
            completed_jobs += 1
            append_raw_record(run_root, record)
            rewrite_summary_csv(run_root, summarize(records))
            write_progress_snapshot(
                run_root,
                completed_jobs=completed_jobs,
                total_jobs=total_jobs,
                records=records,
                global_start=global_start,
            )
            print_progress_line(
                record=record,
                completed_jobs=completed_jobs,
                total_jobs=total_jobs,
                global_start=global_start,
                n_rep=args.n_rep,
            )
    else:
        max_workers = args.n_jobs if args.n_jobs > 0 else (os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_one, **task) for task in tasks]
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                completed_jobs += 1
                append_raw_record(run_root, record)
                rewrite_summary_csv(run_root, summarize(records))
                write_progress_snapshot(
                    run_root,
                    completed_jobs=completed_jobs,
                    total_jobs=total_jobs,
                    records=records,
                    global_start=global_start,
                )
                print_progress_line(
                    record=record,
                    completed_jobs=completed_jobs,
                    total_jobs=total_jobs,
                    global_start=global_start,
                    n_rep=args.n_rep,
                )

    for n_subject in args.n_subject_values:
        for coef_type in args.coef_types:
            for sigma2_function in resolved_sigma2_functions(args):
                for rho_true in resolved_rho_values(args):
                    group_records = [
                        record
                        for record in records
                        if record.n_subject == n_subject
                        and record.coef_type == coef_type
                        and record.sigma2_function == sigma2_function
                        and record.rho_true == rho_true
                    ]
                    group_summary = summarize(group_records)[0]
                    print(
                        f"[group done] n_subject={n_subject} coef={coef_type:10s} "
                        f"sigma2={sigma2_function:8s} rho={rho_true:.3f} "
                        f"success={group_summary['n_success']}/{group_summary['n_rep']} "
                        f"MIAE_final={group_summary['miae_final_mean'] if group_summary['miae_final_mean'] is not None else 'NA'} "
                        f"({group_summary['miae_final_std'] if group_summary['miae_final_std'] is not None else 'NA'}) "
                        f"best_h={group_summary['best_signal_bandwidth_mean'] if group_summary['best_signal_bandwidth_mean'] is not None else 'NA'} "
                        f"({group_summary['best_signal_bandwidth_std'] if group_summary['best_signal_bandwidth_std'] is not None else 'NA'}) "
                        f"best_hbar={group_summary['best_variance_bandwidth_mean'] if group_summary['best_variance_bandwidth_mean'] is not None else 'NA'} "
                        f"({group_summary['best_variance_bandwidth_std'] if group_summary['best_variance_bandwidth_std'] is not None else 'NA'})"
                    )
    return records, run_root


def main() -> None:
    args = parse_args()
    records, run_root = run(args)
    summary = summarize(records)
    maybe_write_outputs(run_root, summary)
    print_summary(summary)


if __name__ == "__main__":
    main()
